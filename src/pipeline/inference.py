"""Unified End-to-End Inference & Server-Side Evaluation Pipeline for PathSnap.

Implements the two strictly separated entry points required by `prompts/05_POST_MERGE_INTEGRATION.md`:
1. `predict_trip(trip_dir, artifacts_dir="artifacts", ...) -> dict`
   - Reads ONLY `<trip_id>.track`, `<trip_id>.nodes`, and `<trip_id>.arcs` (NEVER reads `.route`).
   - Runs Person 1's `extract_trip_features` -> Person 3's `artifacts/final_model.joblib` ->
     Person 4's `RoadGraph` + `ViterbiDecoder` (`artifacts/hmm_config.json`).
2. `evaluate_trip_server_side(prediction, route_file_path, trip_dir=None) -> dict`
   - Strictly separated server-side evaluation against `<trip_id>.route`.
   - Computes observation-level Top-1 accuracy (for both Classifier-Only and Classifier+HMM),
     Route Continuity, and Sequence-level Precision / Recall / F1 / Jaccard / LCS accuracy.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.data.spatial import ENUProjector
from src.features.candidates import align_ground_truth
from src.features.inference_features import extract_trip_features
from src.graph.road_graph import RoadGraph
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.viterbi import HMMConfig, ViterbiDecoder
from src.viz.folium_map import create_folium_map


def _extract_polyline_from_arcs(road_graph: RoadGraph, arc_ids: List[int]) -> List[List[float]]:
    """Convert a sequence of road arc IDs into a [[lat, lon], ...] polyline."""
    coords: List[List[float]] = []
    for arc_id in arc_ids:
        if 0 <= arc_id < road_graph.n_arcs:
            (u_lon, u_lat), (v_lon, v_lat) = road_graph.get_arc_coords(int(arc_id))
            pt_u = [float(u_lat), float(u_lon)]
            pt_v = [float(v_lat), float(v_lon)]
            if not coords or coords[-1] != pt_u:
                coords.append(pt_u)
            coords.append(pt_v)
    return coords


def predict_trip(
    trip_dir: str | Path,
    artifacts_dir: str | Path = "artifacts",
    generate_map: bool = False,
    output_map_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run zero-leakage end-to-end route reconstruction on a single trip directory.

    Strictly reads ONLY `.track`, `.nodes`, and `.arcs` by staging those three files in an
    isolated temporary directory, guaranteeing `.route` can never be accessed during inference.
    """
    t_total_start = time.perf_counter()
    src_dir = Path(trip_dir)
    trip_id = src_dir.name
    art_dir = Path(artifacts_dir)

    preproc_path = art_dir / "preprocessor.joblib"
    feat_cols_path = art_dir / "feature_columns.json"
    model_path = art_dir / "final_model.joblib"
    hmm_cfg_path = art_dir / "hmm_config.json"

    for req in (preproc_path, feat_cols_path, model_path, hmm_cfg_path):
        if not req.exists():
            raise FileNotFoundError(f"Required inference artifact not found: {req}")

    with tempfile.TemporaryDirectory(prefix=f"pathsnap_infer_{trip_id}_") as tmp_root:
        staged_dir = Path(tmp_root) / trip_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("track", "nodes", "arcs"):
            src_file = src_dir / f"{trip_id}.{ext}"
            if not src_file.exists():
                raise FileNotFoundError(f"Missing required input file: {src_file}")
            shutil.copy(src_file, staged_dir / f"{trip_id}.{ext}")

        t_feat_start = time.perf_counter()
        cand_df = extract_trip_features(
            str(staged_dir),
            str(preproc_path),
            str(feat_cols_path),
        )
        feat_time_sec = time.perf_counter() - t_feat_start

        trip_obj = load_trip(staged_dir, load_route=False)
        assert trip_obj.route is None, "Safety check failed: .route must be None during inference"
        road_graph = RoadGraph.from_trip_dir(str(staged_dir), trip_id=trip_id)

    feature_cols: List[str] = json.loads(feat_cols_path.read_text(encoding="utf-8"))

    t_cls_start = time.perf_counter()
    model = joblib.load(model_path)
    pred_probs = np.asarray(model.predict_proba(cand_df[feature_cols])[:, 1], dtype=np.float64)
    cls_time_sec = time.perf_counter() - t_cls_start

    cand_df = cand_df.copy()
    cand_df["pred_prob"] = pred_probs

    obs_groups = cand_df.groupby("obs_idx", sort=True)
    obs_indices = sorted(obs_groups.groups.keys())

    cls_only_arcs: List[int] = []
    observations: List[Dict[str, Any]] = []
    gps_track = trip_obj.track

    for obs_idx in obs_indices:
        group = obs_groups.get_group(obs_idx).sort_values(
            by=["pred_prob", "cand_rank"], ascending=[False, True], kind="mergesort"
        )
        best_row = group.iloc[0]
        cls_only_arcs.append(int(best_row["arc_id"]))
        cand_list = [(int(r["arc_id"]), float(r["pred_prob"])) for _, r in group.iterrows()]
        lon = float(gps_track[obs_idx, 0]) if obs_idx < len(gps_track) else float(road_graph.nodes[0, 0])
        lat = float(gps_track[obs_idx, 1]) if obs_idx < len(gps_track) else float(road_graph.nodes[0, 1])
        ts = float(gps_track[obs_idx, 2]) if obs_idx < len(gps_track) else float(obs_idx)
        observations.append(
            {
                "obs_idx": int(obs_idx),
                "timestamp": ts,
                "lon": lon,
                "lat": lat,
                "candidates": cand_list,
            }
        )

    t_hmm_start = time.perf_counter()
    hmm_cfg_dict = json.loads(hmm_cfg_path.read_text(encoding="utf-8"))
    valid_cfg = {k: v for k, v in hmm_cfg_dict.items() if k in HMMConfig.__dataclass_fields__}
    hmm_config = HMMConfig(**valid_cfg)
    decoder = ViterbiDecoder(hmm_config)
    viterbi_res = decoder.decode(observations, road_graph)
    hmm_time_sec = time.perf_counter() - t_hmm_start

    hmm_arcs_per_obs = [int(a) for a in viterbi_res["predicted_arc_ids"]]
    predicted_route_seq = [int(a) for a in viterbi_res["collapsed_route"]]
    cls_collapsed_seq = collapse_route_sequence(cls_only_arcs)

    cls_cont = float(compute_route_continuity(cls_only_arcs, road_graph, max_step_dist_m=100.0))
    hmm_cont = float(compute_route_continuity(hmm_arcs_per_obs, road_graph, max_step_dist_m=100.0))

    predicted_coords = _extract_polyline_from_arcs(road_graph, predicted_route_seq)
    cls_only_coords = _extract_polyline_from_arcs(road_graph, cls_collapsed_seq)
    gps_coords = [[float(row[1]), float(row[0]), float(row[2])] for row in gps_track]

    folium_html: Optional[str] = None
    if generate_map or output_map_path is not None:
        folium_html = create_folium_map(
            trip_id=trip_id,
            gps_points=gps_track,
            road_graph=road_graph,
            ground_truth_route=None,
            predicted_route=predicted_route_seq,
            classifier_only_route=cls_collapsed_seq,
            output_html_path=str(output_map_path) if output_map_path else None,
        )

    total_time_sec = time.perf_counter() - t_total_start

    return {
        "trip_id": trip_id,
        "num_observations": int(len(gps_track)),
        "num_candidates": int(len(cand_df)),
        "predicted_arc_ids_per_obs": hmm_arcs_per_obs,
        "classifier_only_arc_ids_per_obs": cls_only_arcs,
        "predicted_route_sequence": predicted_route_seq,
        "classifier_only_route_sequence": cls_collapsed_seq,
        "predicted_coordinates": predicted_coords,
        "classifier_only_coordinates": cls_only_coords,
        "gps_coordinates": gps_coords,
        "route_continuity": float(hmm_cont),
        "classifier_only_continuity": float(cls_cont),
        "continuity_gain": float(hmm_cont - cls_cont),
        "feature_extraction_time_sec": float(feat_time_sec),
        "classifier_inference_time_sec": float(cls_time_sec),
        "hmm_decode_time_sec": float(hmm_time_sec),
        "inference_time_sec": float(total_time_sec),
        "candidate_predictions": cand_df[["trip_id", "obs_idx", "arc_id", "cand_rank", "pred_prob"]],
        "folium_html": folium_html,
    }


def evaluate_trip_server_side(
    prediction: Dict[str, Any],
    route_file_path: str | Path,
    trip_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Server-side evaluation comparing `predict_trip()` output against ground-truth `<trip_id>.route`."""
    route_path = Path(route_file_path)
    if not route_path.exists():
        raise FileNotFoundError(f"Ground-truth route file not found: {route_path}")

    gt_route = np.atleast_1d(np.loadtxt(str(route_path), dtype=np.int64)).astype(int).tolist()
    t_dir = Path(trip_dir) if trip_dir is not None else route_path.parent
    trip_id = prediction["trip_id"]

    hmm_seq_metrics = compute_sequence_metrics(prediction["predicted_route_sequence"], gt_route, deduplicate=False)
    cls_seq_metrics = compute_sequence_metrics(prediction["classifier_only_arc_ids_per_obs"], gt_route)

    trip_obj = load_trip(t_dir, load_route=True)
    proj = ENUProjector.from_track(trip_obj.track)
    p_xy = proj.project(trip_obj.track[:, :2])
    node_xy = proj.project(trip_obj.nodes)
    dt_obs = np.r_[0.0, np.diff(trip_obj.track[:, 2])]
    alignment = align_ground_truth(p_xy, node_xy, trip_obj.arcs, trip_obj.route, dt_obs)

    gt_arcs_arr = alignment.arc_id.astype(np.int64)
    cls_arcs = np.asarray(prediction["classifier_only_arc_ids_per_obs"], dtype=np.int64)
    hmm_arcs = np.asarray(prediction["predicted_arc_ids_per_obs"], dtype=np.int64)

    cls_top1_all = float(np.mean(cls_arcs == gt_arcs_arr))
    hmm_top1_all = float(np.mean(hmm_arcs == gt_arcs_arr))

    return {
        "trip_id": trip_id,
        "num_observations": int(prediction["num_observations"]),
        "num_gt_route_arcs": int(len(gt_route)),
        "num_pred_route_arcs": int(len(prediction["predicted_route_sequence"])),
        "classifier_only_top1_accuracy": round(cls_top1_all, 6),
        "hmm_top1_accuracy": round(hmm_top1_all, 6),
        "top1_accuracy_gain": round(hmm_top1_all - cls_top1_all, 6),
        "classifier_only_continuity": round(float(prediction["classifier_only_continuity"]), 6),
        "hmm_continuity": round(float(prediction["route_continuity"]), 6),
        "continuity_gain": round(float(prediction["continuity_gain"]), 6),
        "classifier_only_seq_f1": round(float(cls_seq_metrics.get("sequence_f1", 0.0)), 6),
        "hmm_seq_f1": round(float(hmm_seq_metrics.get("sequence_f1", 0.0)), 6),
        "seq_f1_gain": round(
            float(hmm_seq_metrics.get("sequence_f1", 0.0) - cls_seq_metrics.get("sequence_f1", 0.0)), 6
        ),
        "inference_time_sec": round(float(prediction["inference_time_sec"]), 6),
    }
