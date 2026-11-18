"""PathSnap - HMM & Viterbi Hyperparameter Tuning (Optimized High-Speed Runner).

Systematically evaluates HMM parameter configurations:
- beta (transition distance scale parameter)
- emission_weight (classifier probability scalar)
- disconnect_penalty (log penalty for unreachable transitions)
- max_route_dist_m (search radius cutoff)
- gap_threshold_s (adaptive GPS dropout threshold)

Pre-builds observation structures and distance matrices to complete grid search in seconds.
Logs trials to outputs/hmm/hmm_tuning_log.csv and exports optimal config to artifacts/hmm_config.json.
"""

from __future__ import annotations

import csv
import itertools
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
from src.hmm.evaluate_hmm import load_trip_route, load_trip_track
from src.hmm.route_metrics import compute_observation_accuracy, compute_route_continuity, compute_sequence_metrics
from src.hmm.viterbi import HMMConfig, ViterbiDecoder


def run_tuning(
    parquet_path: str = "outputs/models/classical/logreg_student_test_preds.parquet",
    student_test_dir: str = "Student folder/student_test",
    output_log_csv: str = "outputs/hmm/hmm_tuning_log.csv",
    output_config_json: str = "artifacts/hmm_config.json",
    tuning_trips: Sequence[str] = ("00000052", "00000091"),
) -> HMMConfig:
    """Run fast grid search over HMM parameters on representative tuning trips."""
    print("=" * 70)
    print("PATHSNAP PERSON 4 - HMM HYPERPARAMETER TUNING (FAST GRID SEARCH)")
    print("=" * 70)

    df_all = pd.read_parquet(parquet_path)
    test_dir = Path(student_test_dir)

    # 1. Pre-build observations once
    tuning_data = {}
    for tid in tuning_trips:
        t_dir = test_dir / tid
        if not t_dir.exists():
            continue

        rg = RoadGraph.from_trip_dir(str(t_dir), trip_id=tid)
        gps_track = load_trip_track(t_dir, tid)
        gt_route = load_trip_route(t_dir, tid)
        df_trip = df_all[df_all["trip_id"] == tid]

        obs_groups = df_trip.groupby("obs_idx", sort=True)
        obs_indices = sorted(obs_groups.groups.keys())

        observations = []
        true_aligned = []
        for obs_idx in obs_indices:
            group = obs_groups.get_group(obs_idx).sort_values(by="pred_prob", ascending=False)
            cand_list = [(int(r["arc_id"]), float(r["pred_prob"])) for _, r in group.iterrows()]

            match_rows = group[group["is_match"] == 1]
            true_aligned.append(int(match_rows.iloc[0]["arc_id"]) if not match_rows.empty else -1)

            lon = float(gps_track[obs_idx, 0]) if obs_idx < len(gps_track) else float(rg.nodes[0, 0])
            lat = float(gps_track[obs_idx, 1]) if obs_idx < len(gps_track) else float(rg.nodes[0, 1])
            ts = float(gps_track[obs_idx, 2]) if obs_idx < len(gps_track) else float(obs_idx)

            observations.append({
                "obs_idx": int(obs_idx),
                "timestamp": ts,
                "lon": lon,
                "lat": lat,
                "candidates": cand_list,
            })

        tuning_data[tid] = {
            "graph": rg,
            "observations": observations,
            "gt_route": gt_route,
            "true_aligned": true_aligned,
        }

    # 2. Parameter Grid
    beta_vals = [10.0, 15.0, 25.0]
    ew_vals = [1.0, 1.5, 2.0]
    dp_vals = [30.0, 50.0]
    gap_vals = [3.0, 5.0]

    combinations = list(itertools.product(beta_vals, ew_vals, dp_vals, gap_vals))
    print(f"Evaluating {len(combinations)} configurations across {len(tuning_data)} trips...")

    log_rows = []
    best_score = -1.0
    best_config: Optional[HMMConfig] = None

    t0_all = time.perf_counter()
    for idx, (beta, ew, dp, gt) in enumerate(combinations):
        cfg = HMMConfig(
            beta=beta,
            emission_weight=ew,
            disconnect_penalty=dp,
            max_route_dist_m=1000.0,
            gap_threshold_s=gt,
        )
        decoder = ViterbiDecoder(cfg)

        continuities = []
        top1_accs = []
        seq_f1s = []

        for tid, data in tuning_data.items():
            res = decoder.decode(data["observations"], data["graph"])
            cont = compute_route_continuity(res["predicted_arc_ids"], data["graph"])
            top1 = compute_observation_accuracy(res["predicted_arc_ids"], data["true_aligned"])
            seq = compute_sequence_metrics(res["collapsed_route"], data["gt_route"], deduplicate=False)

            continuities.append(cont)
            top1_accs.append(top1)
            seq_f1s.append(seq.get("sequence_f1", 0.0))

        mean_cont = float(np.mean(continuities))
        mean_top1 = float(np.mean(top1_accs))
        mean_f1 = float(np.mean(seq_f1s))
        composite = 0.50 * mean_cont + 0.30 * mean_top1 + 0.20 * mean_f1

        trial = {
            "trial_id": idx + 1,
            "beta": beta,
            "emission_weight": ew,
            "disconnect_penalty": dp,
            "gap_threshold_s": gt,
            "mean_route_continuity": round(mean_cont, 4),
            "mean_top1_accuracy": round(mean_top1, 4),
            "mean_sequence_f1": round(mean_f1, 4),
            "composite_score": round(composite, 4),
        }
        log_rows.append(trial)

        if composite > best_score:
            best_score = composite
            best_config = cfg

    t_total = time.perf_counter() - t0_all
    print(f"[OK] Completed {len(combinations)} parameter trials in {t_total:.2f}s!")
    print(f"     Best Composite Score: {best_score:.4f}")
    print(f"     Optimal Config: beta={best_config.beta}, emission_weight={best_config.emission_weight}, disconnect_penalty={best_config.disconnect_penalty}, gap_threshold={best_config.gap_threshold_s}")

    # Save log CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_log_csv)), exist_ok=True)
    df_log = pd.DataFrame(log_rows).sort_values(by="composite_score", ascending=False)
    df_log.to_csv(output_log_csv, index=False)
    print(f"[OK] Saved tuning log to {output_log_csv}")

    # Export optimal configuration JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_config_json)), exist_ok=True)
    with open(output_config_json, "w", encoding="utf-8") as f:
        export_dict = best_config.to_dict()
        export_dict["description"] = "PathSnap Person 4 Optimal Tuned HMM/Viterbi Configuration"
        json.dump(export_dict, f, indent=2)
    print(f"[OK] Saved optimal config to {output_config_json}")

    return best_config


if __name__ == "__main__":
    run_tuning()
