"""Unified End-to-End Inference & Server-Side Evaluation Pipeline for PathSnap.

Implements the two strictly separated entry points:
1. `predict_trip(trip_dir, artifacts_dir="artifacts", ...) -> dict`
   - Reads ONLY `<trip_id>.track`, `<trip_id>.nodes`, and `<trip_id>.arcs` (NEVER reads `.route`).
   - Runs `extract_trip_features` -> `artifacts/final_model.joblib` ->
     `RoadGraph` + `ViterbiDecoder` (`artifacts/hmm_config.json`).
2. `evaluate_trip_server_side(prediction, route_file_path=None, trip_dir=None, include_map_html=True) -> dict`
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

_ARTIFACT_CACHE: Dict[str, Dict[str, Any]] = {}


def load_artifacts(artifacts_dir: str | Path = "artifacts") -> Dict[str, Any]:
    """Load and cache preprocessor, feature columns, final tuned model, and HMM config."""
    art_dir = Path(artifacts_dir).resolve()
    key = str(art_dir)
    if key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    preproc_path = art_dir / "preprocessor.joblib"
    feat_cols_path = art_dir / "feature_columns.json"
    model_path = art_dir / "final_model.joblib"
    meta_path = art_dir / "final_model_meta.json"
    hmm_cfg_path = art_dir / "hmm_config.json"

    for req in (preproc_path, feat_cols_path, model_path, hmm_cfg_path):
        if not req.exists():
            raise FileNotFoundError(f"Required inference artifact not found: {req}")

    feature_cols: List[str] = json.loads(feat_cols_path.read_text(encoding="utf-8"))
    model_meta: Dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    hmm_cfg_dict: Dict[str, Any] = json.loads(hmm_cfg_path.read_text(encoding="utf-8"))
    valid_cfg = {k: v for k, v in hmm_cfg_dict.items() if k in HMMConfig.__dataclass_fields__}
    hmm_config = HMMConfig(**valid_cfg)

    bundle = {
        "artifacts_dir": art_dir,
        "preprocessor_path": preproc_path,
        "feature_columns_path": feat_cols_path,
        "feature_columns": feature_cols,
        "model": joblib.load(model_path),
        "model_meta": model_meta,
        "hmm_config": hmm_config,
        "hmm_config_dict": hmm_cfg_dict,
    }
    _ARTIFACT_CACHE[key] = bundle
    return bundle


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
    include_map_html: bool = False,
    output_map_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run zero-leakage end-to-end route reconstruction on a single trip directory.

    Strictly reads ONLY `.track`, `.nodes`, and `.arcs` by staging those three files in an
    isolated temporary directory, guaranteeing `.route` can never be accessed during inference.
    """
    t_total_start = time.perf_counter()
    src_dir = Path(trip_dir)
    trip_id = src_dir.name
    bundle = load_artifacts(artifacts_dir)

    preproc_path = bundle["preprocessor_path"]
    feat_cols_path = bundle["feature_columns_path"]
    feature_cols = bundle["feature_columns"]
    model = bundle["model"]
    hmm_config = bundle["hmm_config"]

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

    t_cls_start = time.perf_counter()
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
    if generate_map or include_map_html or output_map_path is not None:
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
    n_obs = int(len(gps_track))
    n_cands = int(len(cand_df))

    return {
        "trip_id": trip_id,
        "num_observations": n_obs,
        "n_observations": n_obs,
        "num_candidates": n_cands,
        "n_candidate_rows": n_cands,
        "n_nodes": int(road_graph.n_nodes),
        "n_arcs": int(road_graph.n_arcs),
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
        "log_likelihood": float(viterbi_res["log_likelihood"]),
        "n_disconnects_handled": int(viterbi_res["n_disconnects_handled"]),
        "feature_extraction_time_sec": float(feat_time_sec),
        "classifier_inference_time_sec": float(cls_time_sec),
        "hmm_decode_time_sec": float(hmm_time_sec),
        "inference_time_sec": float(total_time_sec),
        "timing_breakdown_sec": {
            "feature_extraction_sec": round(float(feat_time_sec), 4),
            "classifier_inference_sec": round(float(cls_time_sec), 4),
            "hmm_viterbi_sec": round(float(hmm_time_sec), 4),
            "total_sec": round(float(total_time_sec), 4),
        },
        "candidate_predictions": cand_df[["trip_id", "obs_idx", "arc_id", "cand_rank", "pred_prob"]],
        "folium_html": folium_html,
    }


def evaluate_trip_server_side(
    prediction: Dict[str, Any],
    route_file_path: Optional[str | Path] = None,
    trip_dir: Optional[str | Path] = None,
    include_map_html: bool = True,
) -> Dict[str, Any]:
    """Server-side evaluation comparing `predict_trip()` output against ground-truth `<trip_id>.route`.

    Supports being called as:
      - `evaluate_trip_server_side(prediction, route_file_path, trip_dir=...)`
      - `evaluate_trip_server_side(prediction, trip_dir)` (where 2nd arg is a directory)
    """
    trip_id = prediction["trip_id"]
    resolved_dir: Optional[Path] = Path(trip_dir) if trip_dir is not None else None
    resolved_route: Optional[Path] = None

    if route_file_path is not None:
        p = Path(route_file_path)
        if p.is_dir():
            resolved_dir = p
            resolved_route = p / f"{trip_id}.route"
        else:
            resolved_route = p
            if resolved_dir is None:
                resolved_dir = p.parent
    elif resolved_dir is not None:
        resolved_route = resolved_dir / f"{trip_id}.route"

    if resolved_route is None or not resolved_route.exists():
        return {
            "trip_id": trip_id,
            "has_ground_truth": False,
            "num_observations": int(prediction["num_observations"]),
        }

    gt_route = np.atleast_1d(np.loadtxt(str(resolved_route), dtype=np.int64)).astype(int).tolist()
    t_dir = resolved_dir if resolved_dir is not None else resolved_route.parent

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

    hmm_metrics_full = {
        **hmm_seq_metrics,
        "observation_top1_accuracy": round(hmm_top1_all, 6),
        "route_continuity": round(float(prediction["route_continuity"]), 6),
    }
    cls_metrics_full = {
        **cls_seq_metrics,
        "observation_top1_accuracy": round(cls_top1_all, 6),
        "route_continuity": round(float(prediction["classifier_only_continuity"]), 6),
    }

    folium_html = prediction.get("folium_html")
    if include_map_html:
        road_graph = RoadGraph.from_trip_dir(str(t_dir), trip_id=trip_id)
        folium_html = create_folium_map(
            trip_id=trip_id,
            gps_points=trip_obj.track,
            road_graph=road_graph,
            ground_truth_route=gt_route,
            predicted_route=prediction["predicted_route_sequence"],
            classifier_only_route=prediction["classifier_only_route_sequence"],
        )

    return {
        "trip_id": trip_id,
        "has_ground_truth": True,
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
        "hmm_metrics": hmm_metrics_full,
        "classifier_only_metrics": cls_metrics_full,
        "improvements": {
            "continuity_delta": round(float(prediction["continuity_gain"]), 6),
            "sequence_f1_delta": round(
                float(hmm_seq_metrics.get("sequence_f1", 0.0) - cls_seq_metrics.get("sequence_f1", 0.0)), 6
            ),
            "lcs_ratio_delta": round(
                float(hmm_seq_metrics.get("lcs_ratio", 0.0) - cls_seq_metrics.get("lcs_ratio", 0.0)), 6
            ),
            "obs_top1_delta": round(hmm_top1_all - cls_top1_all, 6),
        },
        "inference_time_sec": round(float(prediction["inference_time_sec"]), 6),
        "folium_html": folium_html,
    }
