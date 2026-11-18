"""Unified end-to-end PathSnap inference and server-side evaluation pipeline.

Strictly adheres to the zero-leakage academic contract:
1. predict_trip(): consumes ONLY .track, .nodes, and .arcs (NEVER reads .route).
2. evaluate_trip_server_side(): strictly separated server-side evaluation against .route.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.features.build_features import compute_trip_features
from src.features.inference_features import extract_trip_features
from src.graph.road_graph import RoadGraph
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_observation_accuracy,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.viterbi import HMMConfig, ViterbiDecoder
from src.viz.folium_map import create_folium_map

_ARTIFACT_CACHE: Dict[str, Any] = {}


def load_artifacts(artifacts_dir: str | Path = "artifacts") -> Dict[str, Any]:
    """Load and cache trained classifier, preprocessor, feature columns, and HMM config."""
    art_path = Path(artifacts_dir).resolve()
    cache_key = str(art_path)
    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    preproc_path = art_path / "preprocessor.joblib"
    feat_cols_path = art_path / "feature_columns.json"
    model_path = art_path / "final_model.joblib"
    meta_path = art_path / "final_model_meta.json"
    hmm_cfg_path = art_path / "hmm_config.json"

    feature_cols: List[str] = json.loads(feat_cols_path.read_text(encoding="utf-8"))
    model_meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    hmm_raw: Dict[str, Any] = json.loads(hmm_cfg_path.read_text(encoding="utf-8")) if hmm_cfg_path.exists() else {}

    loaded_model = joblib.load(model_path)
    if isinstance(loaded_model, dict):
        model = (
            loaded_model.get("model")
            or loaded_model.get("estimator")
            or loaded_model.get("clf")
            or loaded_model
        )
    else:
        model = loaded_model

    hmm_config = HMMConfig(
        beta=float(hmm_raw.get("beta", 25.0)),
        emission_weight=float(hmm_raw.get("emission_weight", 2.0)),
        disconnect_penalty=float(hmm_raw.get("disconnect_penalty", 30.0)),
        max_route_dist_m=float(hmm_raw.get("max_route_dist_m", 1000.0)),
        gap_threshold_s=float(hmm_raw.get("gap_threshold_s", 3.0)),
        eps=float(hmm_raw.get("eps", 1e-7)),
    )

    bundle = {
        "artifacts_dir": str(art_path),
        "preprocessor_path": str(preproc_path),
        "feature_cols_path": str(feat_cols_path),
        "feature_columns": feature_cols,
        "model": model,
        "model_meta": model_meta,
        "hmm_config": hmm_config,
        "hmm_config_dict": hmm_raw,
    }
    _ARTIFACT_CACHE[cache_key] = bundle
    return bundle


def _predict_candidate_probabilities(
    df_features: pd.DataFrame, model: Any, feature_cols: List[str]
) -> np.ndarray:
    """Run classifier probability inference on candidate rows."""
    X = df_features[feature_cols]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        pred_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X).ravel()
        pred_prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        pred_prob = model.predict(X).ravel()
    return np.clip(np.asarray(pred_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)


def _build_observations_and_clf_top1(
    df_cand: pd.DataFrame, gps_track: np.ndarray, road_graph: RoadGraph
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Group candidate predictions by obs_idx for ViterbiDecoder and extract classifier-only top-1."""
    observations: List[Dict[str, Any]] = []
    clf_top1_arcs: List[int] = []

    obs_groups = df_cand.groupby("obs_idx", sort=True)
    for obs_idx in sorted(obs_groups.groups.keys()):
        group = obs_groups.get_group(obs_idx)
        sorted_group = group.sort_values(by="pred_prob", ascending=False)
        cand_list = [
            (int(row["arc_id"]), float(row["pred_prob"]))
            for _, row in sorted_group.iterrows()
        ]
        if cand_list:
            clf_top1_arcs.append(cand_list[0][0])
        else:
            clf_top1_arcs.append(-1)

        idx_int = int(obs_idx)
        if gps_track is not None and idx_int < len(gps_track):
            lon = float(gps_track[idx_int, 0])
            lat = float(gps_track[idx_int, 1])
            ts = float(gps_track[idx_int, 2])
        else:
            lon = float(road_graph.nodes[0, 0])
            lat = float(road_graph.nodes[0, 1])
            ts = float(idx_int)

        observations.append(
            {
                "obs_idx": idx_int,
                "timestamp": ts,
                "lon": lon,
                "lat": lat,
                "candidates": cand_list,
            }
        )
    return observations, clf_top1_arcs


def _arcs_to_latlon_polyline(arc_ids: List[int], road_graph: RoadGraph) -> List[List[float]]:
    """Convert a sequence of arc_ids into [[lat, lon], ...] coordinates."""
    coords: List[List[float]] = []
    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= road_graph.n_arcs:
            continue
        (lon1, lat1), (lon2, lat2) = road_graph.get_arc_coords(int(arc_id))
        pt1 = [float(lat1), float(lon1)]
        pt2 = [float(lat2), float(lon2)]
        if not coords or coords[-1] != pt1:
            coords.append(pt1)
        coords.append(pt2)
    return coords


def predict_trip(
    trip_dir: str | Path,
    artifacts_dir: str | Path = "artifacts",
    include_map_html: bool = True,
) -> Dict[str, Any]:
    """End-to-end prediction pipeline for a single trip.

    Reads ONLY .track, .nodes, and .arcs (NEVER reads .route).
    """
    t0 = time.perf_counter()
    trip_path = Path(trip_dir)
    bundle = load_artifacts(artifacts_dir)

    # 1. Load raw trip without ground-truth route
    trip = load_trip(trip_path, load_route=False)
    road_graph = RoadGraph(trip.nodes, trip.arcs, trip_id=trip.trip_id)

    # 2. Extract 27 candidate features (strictly without .route)
    t_feat_start = time.perf_counter()
    df_features = extract_trip_features(
        str(trip_path),
        bundle["preprocessor_path"],
        bundle["feature_cols_path"],
    )
    feat_time_sec = time.perf_counter() - t_feat_start

    # 3. Predict emission probabilities with tuned model
    t_clf_start = time.perf_counter()
    df_cand = df_features.copy()
    df_cand["pred_prob"] = _predict_candidate_probabilities(
        df_cand, bundle["model"], bundle["feature_columns"]
    )
    clf_time_sec = time.perf_counter() - t_clf_start

    # 4. Run HMM + Viterbi decoding
    t_hmm_start = time.perf_counter()
    observations, clf_top1_arcs = _build_observations_and_clf_top1(
        df_cand, trip.track, road_graph
    )
    decoder = ViterbiDecoder(config=bundle["hmm_config"])
    decode_res = decoder.decode(observations, road_graph)
    hmm_time_sec = time.perf_counter() - t_hmm_start

    predicted_arc_ids: List[int] = decode_res["predicted_arc_ids"]
    predicted_route_seq: List[int] = decode_res["collapsed_route"]
    clf_only_route_seq: List[int] = collapse_route_sequence(clf_top1_arcs)

    # 5. Compute unsupervised route continuity & diagnostics
    hmm_continuity = float(compute_route_continuity(predicted_arc_ids, road_graph))
    clf_continuity = float(compute_route_continuity(clf_top1_arcs, road_graph))
    total_time_sec = time.perf_counter() - t0

    folium_html = ""
    if include_map_html:
        folium_html = create_folium_map(
            trip_id=trip.trip_id,
            gps_points=trip.track,
            road_graph=road_graph,
            ground_truth_route=None,
            predicted_route=predicted_route_seq,
            classifier_only_route=clf_only_route_seq,
            output_html_path=None,
        )

    gps_coords = [[float(row[1]), float(row[0])] for row in trip.track[:: max(1, len(trip.track) // 500)]]

    return {
        "trip_id": trip.trip_id,
        "n_observations": int(len(observations)),
        "n_candidate_rows": int(len(df_cand)),
        "n_nodes": int(road_graph.n_nodes),
        "n_arcs": int(road_graph.n_arcs),
        "predicted_arc_ids_per_obs": predicted_arc_ids,
        "predicted_route_sequence": predicted_route_seq,
        "classifier_only_arc_ids_per_obs": clf_top1_arcs,
        "classifier_only_route_sequence": clf_only_route_seq,
        "predicted_coordinates": _arcs_to_latlon_polyline(predicted_route_seq, road_graph),
        "gps_coordinates": gps_coords,
        "route_continuity": hmm_continuity,
        "classifier_only_continuity": clf_continuity,
        "continuity_gain": float(hmm_continuity - clf_continuity),
        "log_likelihood": float(decode_res["log_likelihood"]),
        "n_disconnects_handled": int(decode_res["n_disconnects_handled"]),
        "inference_time_sec": float(total_time_sec),
        "timing_breakdown_sec": {
            "feature_extraction": float(feat_time_sec),
            "classifier_inference": float(clf_time_sec),
            "hmm_viterbi_decode": float(hmm_time_sec),
            "total": float(total_time_sec),
        },
        "folium_html": folium_html,
    }


def evaluate_trip_server_side(
    prediction: Dict[str, Any],
    trip_dir: str | Path,
) -> Dict[str, Any]:
    """Strictly server-side evaluation comparing prediction output against .route.

    Ground-truth .route never enters the model or feature extraction.
    """
    trip_path = Path(trip_dir)
    route_file = trip_path / f"{trip_path.name}.route"
    if not route_file.exists():
        candidates = list(trip_path.glob("*.route"))
        if candidates:
            route_file = candidates[0]
        else:
            return {"has_ground_truth": False}

    trip_with_gt = load_trip(trip_path, load_route=True)
    road_graph = RoadGraph(trip_with_gt.nodes, trip_with_gt.arcs, trip_id=trip_with_gt.trip_id)
    gt_route_arcs = [int(a) for a in trip_with_gt.route.tolist()] if trip_with_gt.route is not None else []

    hmm_obs_arcs = prediction["predicted_arc_ids_per_obs"]
    clf_obs_arcs = prediction["classifier_only_arc_ids_per_obs"]

    hmm_seq_metrics = compute_sequence_metrics(hmm_obs_arcs, gt_route_arcs, deduplicate=True)
    clf_seq_metrics = compute_sequence_metrics(clf_obs_arcs, gt_route_arcs, deduplicate=True)

    # Compute observation-level Top-1 accuracy using GT alignment on server side
    hmm_obs_acc: Optional[float] = None
    clf_obs_acc: Optional[float] = None
    try:
        df_labeled, _ = compute_trip_features(trip_with_gt, with_labels=True, feature_set="v2")
        pos_rows = df_labeled[df_labeled["is_match"] == 1].sort_values("obs_idx")
        gt_by_obs = dict(zip(pos_rows["obs_idx"].astype(int), pos_rows["arc_id"].astype(int)))
        aligned_gt = [gt_by_obs.get(i, -999) for i in range(len(hmm_obs_arcs))]
        valid_mask = [g != -999 for g in aligned_gt]
        if any(valid_mask):
            hmm_valid = [p for p, m in zip(hmm_obs_arcs, valid_mask) if m]
            clf_valid = [p for p, m in zip(clf_obs_arcs, valid_mask) if m]
            gt_valid = [g for g, m in zip(aligned_gt, valid_mask) if m]
            hmm_obs_acc = float(compute_observation_accuracy(hmm_valid, gt_valid))
            clf_obs_acc = float(compute_observation_accuracy(clf_valid, gt_valid))
    except Exception:
        pass

    # Re-render map with ground-truth route overlay for evaluator display
    folium_html_with_gt = create_folium_map(
        trip_id=trip_with_gt.trip_id,
        gps_points=trip_with_gt.track,
        road_graph=road_graph,
        ground_truth_route=gt_route_arcs,
        predicted_route=prediction["predicted_route_sequence"],
        classifier_only_route=prediction["classifier_only_route_sequence"],
        output_html_path=None,
    )

    return {
        "has_ground_truth": True,
        "gt_route_length": len(collapse_route_sequence(gt_route_arcs)),
        "hmm_metrics": {
            **hmm_seq_metrics,
            "route_continuity": prediction["route_continuity"],
            "observation_top1_accuracy": hmm_obs_acc,
        },
        "classifier_only_metrics": {
            **clf_seq_metrics,
            "route_continuity": prediction["classifier_only_continuity"],
            "observation_top1_accuracy": clf_obs_acc,
        },
        "improvements": {
            "sequence_f1_delta": float(hmm_seq_metrics["sequence_f1"] - clf_seq_metrics["sequence_f1"]),
            "lcs_ratio_delta": float(hmm_seq_metrics["lcs_ratio"] - clf_seq_metrics["lcs_ratio"]),
            "continuity_delta": float(prediction["route_continuity"] - prediction["classifier_only_continuity"]),
            "obs_top1_delta": (
                float(hmm_obs_acc - clf_obs_acc)
                if (hmm_obs_acc is not None and clf_obs_acc is not None)
                else None
            ),
        },
        "folium_html": folium_html_with_gt,
    }
