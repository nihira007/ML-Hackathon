"""Evaluate the Final Tuned XGBoost + HMM/Viterbi Model on `evaluator_test/`.

Runs zero-leakage end-to-end inference (`predict_trip` using ONLY `.track`, `.nodes`, `.arcs`)
followed by strictly separated server-side evaluation (`evaluate_trip_server_side` against `.route`)
across all 10 trips in `evaluator_test/`:
  00000015, 00000017, 00000023, 00000030, 00000049,
  00000059, 00000067, 00000073, 00000087, 00000094

Outputs:
  - `outputs/evaluator_test/evaluator_test_per_trip.csv`
  - `outputs/evaluator_test/evaluator_test_summary.json`
  - `outputs/evaluator_test/evaluator_test_preds.parquet`
  - `outputs/maps/trip_<trip_id>_reconstruction.html`
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.data.loader import load_trip
from src.data.spatial import ENUProjector
from src.features.candidates import align_ground_truth
from src.pipeline.inference import evaluate_trip_server_side, predict_trip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_TEST_DIR = PROJECT_ROOT / "evaluator_test"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluator_test"
MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"


def run_evaluator_test_benchmark() -> Dict[str, Any]:
    if not EVALUATOR_TEST_DIR.exists():
        raise FileNotFoundError(f"evaluator_test directory not found at {EVALUATOR_TEST_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted(
        [
            d
            for d in EVALUATOR_TEST_DIR.iterdir()
            if d.is_dir() and (d / f"{d.name}.track").exists()
        ]
    )

    print(f"Running Final Tuned XGBoost + HMM/Viterbi Model on {len(trip_dirs)} evaluator_test trips...")
    print("-" * 104)
    print(
        f"{'Trip ID':<10} | {'Obs':>5} | {'Cands':>6} | {'Cls Top-1':>9} | {'HMM Top-1':>9} | "
        f"{'Cls Cont':>9} | {'HMM Cont':>9} | {'Seq F1':>8} | {'Time (s)':>8}"
    )
    print("-" * 104)

    per_trip_rows: List[Dict[str, Any]] = []
    pred_dfs: List[pd.DataFrame] = []
    t_all_start = time.perf_counter()

    for t_dir in trip_dirs:
        trip_id = t_dir.name
        map_path = MAPS_DIR / f"trip_{trip_id}_reconstruction.html"

        # 1. Zero-leakage inference (strictly .track, .nodes, .arcs)
        pred = predict_trip(
            t_dir,
            artifacts_dir=ARTIFACTS_DIR,
            generate_map=False,
        )

        # 2. Server-side evaluation against .route (and save interactive Folium map with GT comparison)
        eval_res = evaluate_trip_server_side(
            pred,
            t_dir / f"{trip_id}.route",
            trip_dir=t_dir,
            include_map_html=True,
        )

        if eval_res.get("folium_html"):
            map_path.write_text(eval_res["folium_html"], encoding="utf-8")

        # 3. Attach aligned ground-truth `is_match` to candidate_predictions for UI & log-loss audit
        trip_obj = load_trip(t_dir, load_route=True)
        proj = ENUProjector.from_track(trip_obj.track)
        p_xy = proj.project(trip_obj.track[:, :2])
        node_xy = proj.project(trip_obj.nodes)
        dt_obs = np.r_[0.0, np.diff(trip_obj.track[:, 2])]
        alignment = align_ground_truth(p_xy, node_xy, trip_obj.arcs, trip_obj.route, dt_obs)
        gt_map = {int(i): int(a) for i, a in enumerate(alignment.arc_id)}

        cdf = pred["candidate_predictions"].copy()
        cdf["true_arc_id"] = cdf["obs_idx"].map(gt_map)
        cdf["is_match"] = (cdf["arc_id"] == cdf["true_arc_id"]).astype(int)
        pred_dfs.append(cdf)

        cand_logloss = float(
            log_loss(cdf["is_match"].to_numpy(), np.clip(cdf["pred_prob"].to_numpy(), 1e-7, 1 - 1e-7), labels=[0, 1])
        )
        cand_macro_f1 = float(
            f1_score(cdf["is_match"].to_numpy(), (cdf["pred_prob"].to_numpy() >= 0.5).astype(int), average="macro")
        )

        row = {
            "trip_id": trip_id,
            "partition": "evaluator_test",
            "num_observations": int(pred["num_observations"]),
            "num_candidates": int(pred["num_candidates"]),
            "num_gt_route_arcs": int(eval_res["num_gt_route_arcs"]),
            "num_pred_route_arcs": int(eval_res["num_pred_route_arcs"]),
            "cls_top1": float(eval_res["classifier_only_top1_accuracy"]),
            "hmm_top1": float(eval_res["hmm_top1_accuracy"]),
            "top1_gain": float(eval_res["top1_accuracy_gain"]),
            "cls_continuity": float(eval_res["classifier_only_continuity"]),
            "hmm_continuity": float(eval_res["hmm_continuity"]),
            "continuity_gain": float(eval_res["continuity_gain"]),
            "cls_seq_f1": float(eval_res["classifier_only_seq_f1"]),
            "hmm_seq_f1": float(eval_res["hmm_seq_f1"]),
            "seq_f1_gain": float(eval_res["seq_f1_gain"]),
            "candidate_macro_f1": round(cand_macro_f1, 6),
            "candidate_log_loss": round(cand_logloss, 6),
            "feature_extraction_time_sec": round(float(pred["feature_extraction_time_sec"]), 4),
            "classifier_inference_time_sec": round(float(pred["classifier_inference_time_sec"]), 4),
            "hmm_runtime_sec": round(float(pred["hmm_decode_time_sec"]), 4),
            "total_inference_time_sec": round(float(pred["inference_time_sec"]), 4),
        }
        per_trip_rows.append(row)

        print(
            f"{trip_id:<10} | {row['num_observations']:>5} | {row['num_candidates']:>6} | "
            f"{row['cls_top1']*100:>8.2f}% | {row['hmm_top1']*100:>8.2f}% | "
            f"{row['cls_continuity']*100:>8.2f}% | {row['hmm_continuity']*100:>8.2f}% | "
            f"{row['hmm_seq_f1']:>8.4f} | {row['total_inference_time_sec']:>7.3f}s"
        )

    total_wall_sec = time.perf_counter() - t_all_start
    df_trips = pd.DataFrame(per_trip_rows)
    all_preds_df = pd.concat(pred_dfs, ignore_index=True)

    # Pooled candidate-level metrics across all 10 evaluator_test trips
    pooled_macro_f1 = float(
        f1_score(
            all_preds_df["is_match"].to_numpy(),
            (all_preds_df["pred_prob"].to_numpy() >= 0.5).astype(int),
            average="macro",
        )
    )
    pooled_log_loss = float(
        log_loss(
            all_preds_df["is_match"].to_numpy(),
            np.clip(all_preds_df["pred_prob"].to_numpy(), 1e-7, 1 - 1e-7),
            labels=[0, 1],
        )
    )

    # Observation-weighted & trip-mean metrics
    total_obs = int(df_trips["num_observations"].sum())
    weighted_cls_top1 = float((df_trips["cls_top1"] * df_trips["num_observations"]).sum() / total_obs)
    weighted_hmm_top1 = float((df_trips["hmm_top1"] * df_trips["num_observations"]).sum() / total_obs)

    summary = {
        "partition": "evaluator_test",
        "model": "XGBoost (Optuna Tuned) + HMM/Viterbi",
        "artifact_path": "artifacts/final_model.joblib",
        "hmm_config_path": "artifacts/hmm_config.json",
        "n_trips": int(len(df_trips)),
        "total_observations": total_obs,
        "total_candidate_rows": int(len(all_preds_df)),
        "mean_classifier_top1": round(float(df_trips["cls_top1"].mean()), 4),
        "mean_hmm_top1": round(float(df_trips["hmm_top1"].mean()), 4),
        "mean_top1_gain": round(float(df_trips["top1_gain"].mean()), 4),
        "obs_weighted_classifier_top1": round(weighted_cls_top1, 4),
        "obs_weighted_hmm_top1": round(weighted_hmm_top1, 4),
        "obs_weighted_top1_gain": round(weighted_hmm_top1 - weighted_cls_top1, 4),
        "mean_classifier_continuity": round(float(df_trips["cls_continuity"].mean()), 4),
        "mean_hmm_continuity": round(float(df_trips["hmm_continuity"].mean()), 4),
        "mean_continuity_gain": round(float(df_trips["continuity_gain"].mean()), 4),
        "mean_classifier_seq_f1": round(float(df_trips["cls_seq_f1"].mean()), 4),
        "mean_hmm_seq_f1": round(float(df_trips["hmm_seq_f1"].mean()), 4),
        "mean_seq_f1_gain": round(float(df_trips["seq_f1_gain"].mean()), 4),
        "candidate_macro_f1": round(pooled_macro_f1, 4),
        "candidate_log_loss": round(pooled_log_loss, 4),
        "total_feature_extraction_sec": round(float(df_trips["feature_extraction_time_sec"].sum()), 4),
        "total_classifier_inference_sec": round(float(df_trips["classifier_inference_time_sec"].sum()), 4),
        "total_hmm_runtime_sec": round(float(df_trips["hmm_runtime_sec"].sum()), 4),
        "total_pipeline_time_sec": round(float(df_trips["total_inference_time_sec"].sum()), 4),
        "wall_clock_sec": round(float(total_wall_sec), 4),
    }

    print("-" * 104)
    print(
        f"{'MEAN / TOT':<10} | {total_obs:>5} | {len(all_preds_df):>6} | "
        f"{summary['mean_classifier_top1']*100:>8.2f}% | {summary['mean_hmm_top1']*100:>8.2f}% | "
        f"{summary['mean_classifier_continuity']*100:>8.2f}% | {summary['mean_hmm_continuity']*100:>8.2f}% | "
        f"{summary['mean_hmm_seq_f1']:>8.4f} | {summary['total_pipeline_time_sec']:>7.3f}s"
    )
    print("-" * 104)

    csv_path = OUTPUT_DIR / "evaluator_test_per_trip.csv"
    json_path = OUTPUT_DIR / "evaluator_test_summary.json"
    pq_path = OUTPUT_DIR / "evaluator_test_preds.parquet"

    df_trips.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_preds_df.to_parquet(pq_path, index=False)

    print(f"Saved per-trip CSV:  {csv_path}")
    print(f"Saved summary JSON:  {json_path}")
    print(f"Saved preds parquet: {pq_path}")
    return summary


if __name__ == "__main__":
    run_evaluator_test_benchmark()
