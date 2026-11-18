"""PathSnap - Classifier-Only vs. HMM/Viterbi Benchmark Evaluation Runner.

Takes candidate predictions from any model's prediction parquet (or DataFrame),
merges track geometry from the trip directory, and evaluates:
1. Classifier-Only Baseline (per-observation argmax)
2. Classifier + HMM/Viterbi Decoder

Outputs:
- Observation-level Pointwise Top-1 Accuracy
- Route Continuity (%)
- Sequence-level Precision, Recall, F1, LCS ratio, Jaccard similarity (vs. ground-truth .route)
- Decoding latency / throughput
- Per-trip CSV and aggregated summary JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.road_graph import RoadGraph
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_observation_accuracy,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.viterbi import HMMConfig, ViterbiDecoder


def load_trip_track(trip_dir: str | Path, trip_id: str) -> np.ndarray:
    """Load GPS track [lon, lat, timestamp] from <trip_id>.track."""
    track_path = Path(trip_dir) / f"{trip_id}.track"
    if not track_path.exists():
        candidates = list(Path(trip_dir).glob("*.track"))
        if candidates:
            track_path = candidates[0]
        else:
            raise FileNotFoundError(f"Missing .track in {trip_dir}")
    return np.loadtxt(str(track_path), dtype=np.float64)


def load_trip_route(trip_dir: str | Path, trip_id: str) -> Optional[List[int]]:
    """Load ground truth route arc sequence from <trip_id>.route if present."""
    route_path = Path(trip_dir) / f"{trip_id}.route"
    if not route_path.exists():
        candidates = list(Path(trip_dir).glob("*.route"))
        if candidates:
            route_path = candidates[0]
        else:
            return None
    return np.loadtxt(str(route_path), dtype=np.int32).tolist()


def evaluate_trip_predictions(
    trip_id: str,
    df_trip_candidates: pd.DataFrame,
    road_graph: RoadGraph,
    gps_track: Optional[np.ndarray] = None,
    ground_truth_route: Optional[List[int]] = None,
    hmm_config: Optional[HMMConfig] = None,
) -> Dict[str, Any]:
    """Evaluate both Classifier-Only and HMM/Viterbi reconstruction for a single trip.

    Args:
        trip_id: Trip ID string.
        df_trip_candidates: DataFrame containing at least:
            ['obs_idx', 'arc_id', 'pred_prob'] and optionally ['is_match'].
        road_graph: RoadGraph for the trip.
        gps_track: Optional (N, 3) array of [lon, lat, timestamp].
        ground_truth_route: Optional list of chronological ground truth arc IDs.
        hmm_config: Optional HMM hyperparameters.

    Returns:
        Dict with comparative evaluation metrics.
    """
    config = hmm_config or HMMConfig()
    decoder = ViterbiDecoder(config)

    # Group candidate rows strictly by observation index
    obs_groups = df_trip_candidates.groupby("obs_idx", sort=True)
    obs_indices = sorted(obs_groups.groups.keys())

    # Build observation dictionaries for Viterbi
    observations: List[Dict[str, Any]] = []
    cls_only_arcs: List[int] = []

    t0_cls = time.perf_counter()
    for obs_idx in obs_indices:
        group = obs_groups.get_group(obs_idx)
        # Sort candidates descending by predicted probability
        sorted_group = group.sort_values(by="pred_prob", ascending=False)
        best_row = sorted_group.iloc[0]
        cls_only_arcs.append(int(best_row["arc_id"]))

        cand_list = [
            (int(r["arc_id"]), float(r["pred_prob"]))
            for _, r in sorted_group.iterrows()
        ]

        if gps_track is not None and obs_idx < len(gps_track):
            lon = float(gps_track[obs_idx, 0])
            lat = float(gps_track[obs_idx, 1])
            ts = float(gps_track[obs_idx, 2])
        else:
            lon = float(road_graph.nodes[0, 0])
            lat = float(road_graph.nodes[0, 1])
            ts = float(obs_idx)

        observations.append({
            "obs_idx": int(obs_idx),
            "timestamp": ts,
            "lon": lon,
            "lat": lat,
            "candidates": cand_list,
        })
    cls_runtime_sec = time.perf_counter() - t0_cls

    # Run Viterbi decoding
    t0_hmm = time.perf_counter()
    hmm_result = decoder.decode(observations, road_graph)
    hmm_runtime_sec = time.perf_counter() - t0_hmm

    hmm_arcs = hmm_result["predicted_arc_ids"]
    hmm_collapsed = hmm_result["collapsed_route"]

    # Compute Route Continuity
    cls_continuity = compute_route_continuity(cls_only_arcs, road_graph, max_step_dist_m=100.0)
    hmm_continuity = compute_route_continuity(hmm_arcs, road_graph, max_step_dist_m=100.0)

    # Compute Pointwise Accuracy if ground-truth is_match labels exist
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

    # Compute Sequence-level metrics vs .route
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
            "top1_accuracy_gain": round(hmm_top1 - cls_top1, 4) if (hmm_top1 is not None and cls_top1 is not None) else None,
            "seq_f1_gain": round(
                hmm_seq_metrics.get("sequence_f1", 0.0) - cls_seq_metrics.get("sequence_f1", 0.0), 4
            ) if ground_truth_route else None,
        },
    }


def run_benchmark_on_parquet(
    parquet_path: str | Path,
    student_test_dir: str | Path,
    hmm_config: Optional[HMMConfig] = None,
    output_csv_path: Optional[str | Path] = None,
    output_summary_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run full Classifier-Only vs. HMM/Viterbi benchmark across all trips in a parquet file."""
    df_all = pd.read_parquet(parquet_path)
    test_dir = Path(student_test_dir)
    trips = sorted(df_all["trip_id"].unique().tolist())

    per_trip_rows = []
    config = hmm_config or HMMConfig()

    print(f"Running benchmark on {len(trips)} trips from {parquet_path}...")
    for trip_id in trips:
        t_dir = test_dir / trip_id
        if not t_dir.exists():
            continue

        df_trip = df_all[df_all["trip_id"] == trip_id]
        road_graph = RoadGraph.from_trip_dir(str(t_dir), trip_id=trip_id)
        gps_track = load_trip_track(t_dir, trip_id)
        gt_route = load_trip_route(t_dir, trip_id)

        res = evaluate_trip_predictions(
            trip_id=trip_id,
            df_trip_candidates=df_trip,
            road_graph=road_graph,
            gps_track=gps_track,
            ground_truth_route=gt_route,
            hmm_config=config,
        )

        row = {
            "trip_id": trip_id,
            "n_observations": res["n_observations"],
            "cls_continuity": res["classifier_only"]["route_continuity"],
            "hmm_continuity": res["hmm_viterbi"]["route_continuity"],
            "continuity_gain": res["improvement"]["continuity_gain"],
            "cls_top1": res["classifier_only"]["top1_accuracy"],
            "hmm_top1": res["hmm_viterbi"]["top1_accuracy"],
            "top1_gain": res["improvement"]["top1_accuracy_gain"],
            "cls_seq_f1": res["classifier_only"]["sequence_metrics"].get("sequence_f1"),
            "hmm_seq_f1": res["hmm_viterbi"]["sequence_metrics"].get("sequence_f1"),
            "seq_f1_gain": res["improvement"]["seq_f1_gain"],
            "hmm_runtime_sec": res["hmm_viterbi"]["runtime_sec"],
        }
        per_trip_rows.append(row)
        print(f"  Trip {trip_id}: Continuity {row['cls_continuity']*100:.1f}% -> {row['hmm_continuity']*100:.1f}% (+{row['continuity_gain']*100:.1f}%), Top1 {row['cls_top1']*100:.1f}% -> {row['hmm_top1']*100:.1f}%")

    df_res = pd.DataFrame(per_trip_rows)
    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df_res.to_csv(output_csv_path, index=False)
        print(f"Saved per-trip results to {output_csv_path}")

    summary = {
        "n_trips": len(df_res),
        "mean_cls_continuity": round(float(df_res["cls_continuity"].mean()), 4),
        "mean_hmm_continuity": round(float(df_res["hmm_continuity"].mean()), 4),
        "mean_continuity_gain": round(float(df_res["continuity_gain"].mean()), 4),
        "mean_cls_top1": round(float(df_res["cls_top1"].mean()), 4),
        "mean_hmm_top1": round(float(df_res["hmm_top1"].mean()), 4),
        "mean_top1_gain": round(float(df_res["top1_gain"].mean()), 4),
        "mean_cls_seq_f1": round(float(df_res["cls_seq_f1"].mean()), 4),
        "mean_hmm_seq_f1": round(float(df_res["hmm_seq_f1"].mean()), 4),
        "mean_seq_f1_gain": round(float(df_res["seq_f1_gain"].mean()), 4),
        "total_hmm_runtime_sec": round(float(df_res["hmm_runtime_sec"].sum()), 4),
        "best_improvement_trip": str(df_res.sort_values(by="continuity_gain", ascending=False).iloc[0]["trip_id"]),
    }

    if output_summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_summary_json)), exist_ok=True)
        with open(output_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {output_summary_json}")

    return {"summary": summary, "per_trip": df_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMM decoder on prediction parquet")
    parser.add_argument("--parquet", type=str, default="outputs/models/classical/logreg_student_test_preds.parquet")
    parser.add_argument("--test_dir", type=str, default="Student folder/student_test")
    parser.add_argument("--out_csv", type=str, default="outputs/hmm/classifier_vs_hmm_per_trip.csv")
    parser.add_argument("--out_json", type=str, default="outputs/hmm/classifier_vs_hmm_summary.json")
    args = parser.parse_args()

    run_benchmark_on_parquet(
        parquet_path=args.parquet,
        student_test_dir=args.test_dir,
        output_csv_path=args.out_csv,
        output_summary_json=args.out_json,
    )
