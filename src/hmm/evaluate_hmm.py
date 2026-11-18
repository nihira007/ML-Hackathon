"""PathSnap - Classifier-Only vs. HMM/Viterbi Benchmark Evaluation Runner.

Takes candidate predictions (either from parquet or simulated/extracted candidates),
evaluates both:
1. Classifier-Only Baseline (per-observation argmax)
2. Classifier + HMM/Viterbi Decoder

Computes:
- Observation-level Pointwise Top-1 Accuracy
- Route Continuity (%)
- Sequence-level Precision, Recall, F1, LCS ratio, Jaccard similarity (vs. ground-truth .route)
- Decoding latency / inference throughput
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.graph.road_graph import RoadGraph
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_observation_accuracy,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.viterbi import HMMConfig, ViterbiDecoder


def evaluate_trip_predictions(
    trip_id: str,
    df_trip_candidates: pd.DataFrame,
    road_graph: RoadGraph,
    ground_truth_route: Optional[List[int]] = None,
    hmm_config: Optional[HMMConfig] = None,
) -> Dict[str, Any]:
    """Evaluate both Classifier-Only and HMM/Viterbi reconstruction for a single trip.

    Args:
        trip_id: Trip ID string.
        df_trip_candidates: DataFrame containing at least:
            ['obs_idx', 'arc_id', 'pred_prob'] and optionally ['is_match', 'timestamp', 'lon', 'lat'].
        road_graph: RoadGraph for the trip.
        ground_truth_route: Optional list of chronological ground truth arc IDs.
        hmm_config: Optional HMM hyperparameters.

    Returns:
        Dict with comparative evaluation metrics.
    """
    config = hmm_config or HMMConfig()
    decoder = ViterbiDecoder(config)

    # Group by observation index
    obs_groups = df_trip_candidates.groupby("obs_idx", sort=True)
    obs_indices = sorted(obs_groups.groups.keys())

    # 1. Classifier-Only (Argmax per observation)
    t0_cls = time.perf_counter()
    cls_only_arcs: List[int] = []
    observations: List[Dict[str, Any]] = []

    for obs_idx in obs_indices:
        group = obs_groups.get_group(obs_idx)
        best_row = group.sort_values(by="pred_prob", ascending=False).iloc[0]
        cls_only_arcs.append(int(best_row["arc_id"]))

        # Build candidate list for HMM
        cand_list = [
            (int(r["arc_id"]), float(r["pred_prob"]))
            for _, r in group.sort_values(by="pred_prob", ascending=False).iterrows()
        ]

        # Extract coordinates if available
        lon = float(group["lon"].iloc[0]) if "lon" in group.columns else float(road_graph.nodes[0, 0])
        lat = float(group["lat"].iloc[0]) if "lat" in group.columns else float(road_graph.nodes[0, 1])
        ts = float(group["timestamp"].iloc[0]) if "timestamp" in group.columns else float(obs_idx)

        observations.append({
            "obs_idx": int(obs_idx),
            "timestamp": ts,
            "lon": lon,
            "lat": lat,
            "candidates": cand_list,
        })

    t1_cls = time.perf_counter()
    cls_runtime_sec = t1_cls - t0_cls

    # 2. HMM / Viterbi Decoding
    t0_hmm = time.perf_counter()
    hmm_result = decoder.decode(observations, road_graph)
    t1_hmm = time.perf_counter()
    hmm_runtime_sec = t1_hmm - t0_hmm

    hmm_arcs = hmm_result["predicted_arc_ids"]
    hmm_collapsed = hmm_result["collapsed_route"]

    # Compute Route Continuity
    cls_continuity = compute_route_continuity(cls_only_arcs, road_graph)
    hmm_continuity = compute_route_continuity(hmm_arcs, road_graph)

    # Pointwise accuracy against ground-truth aligned labels if is_match present
    has_labels = "is_match" in df_trip_candidates.columns
    if has_labels:
        true_aligned = []
        for obs_idx in obs_indices:
            group = obs_groups.get_group(obs_idx)
            match_rows = group[group["is_match"] == 1]
            if not match_rows.empty:
                true_aligned.append(int(match_rows.iloc[0]["arc_id"]))
            else:
                true_aligned.append(-1)
        cls_top1 = compute_observation_accuracy(cls_only_arcs, true_aligned)
        hmm_top1 = compute_observation_accuracy(hmm_arcs, true_aligned)
    else:
        cls_top1 = None
        hmm_top1 = None

    # Sequence-level metrics against ground-truth .route
    if ground_truth_route is not None:
        cls_seq_metrics = compute_sequence_metrics(cls_only_arcs, ground_truth_route)
        hmm_seq_metrics = compute_sequence_metrics(hmm_collapsed, ground_truth_route, deduplicate=False)
    else:
        cls_seq_metrics = {}
        hmm_seq_metrics = {}

    return {
        "trip_id": trip_id,
        "n_observations": len(obs_indices),
        "classifier_only": {
            "route_continuity": round(cls_continuity, 4),
            "top1_accuracy": round(cls_top1, 4) if cls_top1 is not None else None,
            "sequence_metrics": cls_seq_metrics,
            "runtime_sec": round(cls_runtime_sec, 4),
            "predicted_arcs": cls_only_arcs,
        },
        "hmm_viterbi": {
            "route_continuity": round(hmm_continuity, 4),
            "top1_accuracy": round(hmm_top1, 4) if hmm_top1 is not None else None,
            "sequence_metrics": hmm_seq_metrics,
            "runtime_sec": round(hmm_runtime_sec, 4),
            "predicted_arcs": hmm_arcs,
            "collapsed_route": hmm_collapsed,
            "log_likelihood": round(hmm_result["log_likelihood"], 2),
            "disconnects_handled": hmm_result["n_disconnects_handled"],
        },
        "improvement": {
            "continuity_gain": round(hmm_continuity - cls_continuity, 4),
            "top1_accuracy_gain": round((hmm_top1 - cls_top1), 4) if (hmm_top1 is not None and cls_top1 is not None) else None,
            "seq_f1_gain": round(
                hmm_seq_metrics.get("sequence_f1", 0.0) - cls_seq_metrics.get("sequence_f1", 0.0), 4
            ) if ground_truth_route else None,
        },
    }
