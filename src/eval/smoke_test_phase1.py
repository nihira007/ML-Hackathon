"""
Synthetic smoke test for Person 2 Phase 1: Shared Evaluation & Logging Harness.

Verifies:
1. Shared feature parquet schema validation (`trip_id, obs_idx, timestamp, arc_id, cand_rank, is_match, <features>`).
2. Candidate-level metrics (`Accuracy`, `Precision`, `Recall`, `Macro F1`, `Log Loss`, `Confusion Matrix`).
3. Observation-level Top-1 Accuracy grouping by `(trip_id, obs_idx)` and selecting highest `pred_prob`.
4. 5-fold `GroupKFold` strictly grouped by `trip_id` with zero trip overlap between train and validation splits.
5. Wall-clock timing utilities (`Timer`, `measure_callable`).
6. `ExperimentLogger` saving standardized CSV/JSON records, confusion matrix PNG plots, and prediction parquets.

Run with:
    python3 -m src.eval.smoke_test_phase1
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.eval.logger import ExperimentLogger
from src.eval.metrics import (
    PREDICTION_COLUMNS,
    Timer,
    compute_candidate_metrics,
    compute_per_trip_metrics,
    compute_top1_accuracy,
    compute_topk_accuracy,
    evaluate_predictions,
    iter_trip_cv_splits,
    measure_callable,
    run_group_kfold_cv,
    save_prediction_parquet,
    validate_feature_dataframe,
    validate_prediction_dataframe,
)

SYNTHETIC_FEATURE_COLS: List[str] = [
    "dist_to_arc_m",
    "heading_diff_deg",
    "speed_mps",
    "arc_length_m",
    "cand_dist_ratio",
]


def generate_synthetic_candidate_df(
    num_trips: int = 10,
    obs_per_trip: int = 30,
    cands_per_obs: int = 5,
    seed: int = 42,
    trip_prefix: str = "trip",
) -> pd.DataFrame:
    """Generate a realistic synthetic candidate DataFrame adhering to the shared parquet schema."""
    rng = np.random.default_rng(seed)
    rows = []

    for t_idx in range(num_trips):
        trip_id = f"{trip_prefix}_{t_idx:03d}"
        base_ts = 1700000000.0 + t_idx * 10000.0
        for obs_idx in range(obs_per_trip):
            ts = base_ts + float(obs_idx)
            # Pick exactly one matching candidate per observation (weighted toward rank 0/1)
            match_rank = int(rng.choice(cands_per_obs, p=[0.60, 0.22, 0.10, 0.05, 0.03][:cands_per_obs]))

            raw_dists = np.sort(rng.uniform(2.0, 65.0, size=cands_per_obs))
            for cand_rank in range(cands_per_obs):
                is_match = 1 if cand_rank == match_rank else 0
                # Ground-truth arc tends to have smaller distance & heading difference
                dist_m = float(raw_dists[cand_rank] * (0.45 if is_match else 1.15))
                heading_diff = float(
                    np.clip(
                        rng.normal(12.0 if is_match else 55.0, 15.0),
                        0.0,
                        180.0,
                    )
                )
                speed_mps = float(rng.uniform(4.0, 18.0))
                arc_len_m = float(rng.uniform(25.0, 220.0))
                dist_ratio = float(dist_m / max(raw_dists[0], 1e-3))

                rows.append(
                    {
                        "trip_id": trip_id,
                        "obs_idx": int(obs_idx),
                        "timestamp": ts,
                        "arc_id": int(1000 * t_idx + 10 * obs_idx + cand_rank),
                        "cand_rank": int(cand_rank),
                        "is_match": int(is_match),
                        "dist_to_arc_m": dist_m,
                        "heading_diff_deg": heading_diff,
                        "speed_mps": speed_mps,
                        "arc_length_m": arc_len_m,
                        "cand_dist_ratio": dist_ratio,
                    }
                )

    return pd.DataFrame(rows)


def test_top1_exact_known_case() -> None:
    """Verify Top-1 Accuracy calculation on a deterministic hand-crafted mini table."""
    mini_df = pd.DataFrame(
        [
            # Observation 0: highest prob is arc 10 (is_match=1) -> HIT
            {"trip_id": "t1", "obs_idx": 0, "arc_id": 10, "cand_rank": 0, "is_match": 1, "pred_prob": 0.85},
            {"trip_id": "t1", "obs_idx": 0, "arc_id": 11, "cand_rank": 1, "is_match": 0, "pred_prob": 0.15},
            # Observation 1: highest prob is arc 21 (is_match=0) -> MISS
            {"trip_id": "t1", "obs_idx": 1, "arc_id": 20, "cand_rank": 0, "is_match": 1, "pred_prob": 0.30},
            {"trip_id": "t1", "obs_idx": 1, "arc_id": 21, "cand_rank": 1, "is_match": 0, "pred_prob": 0.70},
            # Observation 2 (trip t2): highest prob is arc 31 (is_match=1) -> HIT
            {"trip_id": "t2", "obs_idx": 0, "arc_id": 30, "cand_rank": 0, "is_match": 0, "pred_prob": 0.25},
            {"trip_id": "t2", "obs_idx": 0, "arc_id": 31, "cand_rank": 1, "is_match": 1, "pred_prob": 0.75},
        ]
    )
    top1 = compute_top1_accuracy(mini_df)
    top2 = compute_topk_accuracy(mini_df, k=2)
    assert np.isclose(top1, 2.0 / 3.0), f"Expected Top-1 Acc = 2/3, got {top1}"
    assert np.isclose(top2, 1.0), f"Expected Top-2 Acc = 1.0, got {top2}"


def run_smoke_test() -> Tuple[pd.DataFrame, dict]:
    """Run end-to-end smoke test of metrics.py and logger.py on synthetic Parquet data."""
    test_top1_exact_known_case()

    train_df_raw = generate_synthetic_candidate_df(
        num_trips=15, obs_per_trip=25, cands_per_obs=5, seed=42, trip_prefix="train_trip"
    )
    test_df_raw = generate_synthetic_candidate_df(
        num_trips=5, obs_per_trip=20, cands_per_obs=5, seed=99, trip_prefix="student_test_trip"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        train_parquet_path = tmp_path / "train_features.parquet"
        test_parquet_path = tmp_path / "student_test_features.parquet"

        train_df_raw.to_parquet(train_parquet_path, index=False)
        test_df_raw.to_parquet(test_parquet_path, index=False)

        # Reload from Parquet to verify full schema round-trip
        train_df = pd.read_parquet(train_parquet_path)
        test_df = pd.read_parquet(test_parquet_path)

        validate_feature_dataframe(train_df, feature_cols=SYNTHETIC_FEATURE_COLS, require_timestamp=True)
        validate_feature_dataframe(test_df, feature_cols=SYNTHETIC_FEATURE_COLS, require_timestamp=True)

        # 1. Verify 5-fold GroupKFold trip separation
        seen_val_trips = set()
        for fold_idx, train_idx, val_idx in iter_trip_cv_splits(
            train_df, feature_cols=SYNTHETIC_FEATURE_COLS, n_splits=5
        ):
            train_trips = set(train_df.iloc[train_idx]["trip_id"])
            val_trips = set(train_df.iloc[val_idx]["trip_id"])
            assert train_trips.isdisjoint(val_trips), f"Fold {fold_idx} has trip leakage!"
            seen_val_trips.update(val_trips)
        assert seen_val_trips == set(train_df["trip_id"].unique()), "Not all trips covered in GroupKFold!"

        # 2. Run 5-fold GroupKFold CV with a baseline LogisticRegression
        model_params = {"C": 1.0, "class_weight": "balanced", "max_iter": 200, "random_state": 42}
        cv_results = run_group_kfold_cv(
            model_fn=lambda: LogisticRegression(**model_params),
            df=train_df,
            feature_cols=SYNTHETIC_FEATURE_COLS,
            n_splits=5,
            scale_features=True,
        )
        assert 0.0 <= cv_results["cv_macro_f1_mean"] <= 1.0
        assert 0.0 <= cv_results["cv_top1_accuracy_mean"] <= 1.0
        assert len(cv_results["folds"]) == 5

        # 3. Fit full model & time training and inference on synthetic student_test
        X_train = train_df[SYNTHETIC_FEATURE_COLS].to_numpy(dtype=np.float64)
        y_train = train_df["is_match"].to_numpy(dtype=np.int64)
        X_test = test_df[SYNTHETIC_FEATURE_COLS].to_numpy(dtype=np.float64)

        clf = LogisticRegression(**model_params)
        _, train_time_sec = measure_callable(clf.fit, X_train, y_train)

        with Timer() as infer_timer:
            y_test_prob = clf.predict_proba(X_test)[:, 1]
        inference_time_sec = infer_timer.elapsed_sec

        # 4. Evaluate candidate-level and observation-level metrics
        cand_metrics = compute_candidate_metrics(test_df["is_match"], y_test_prob)
        test_metrics = evaluate_predictions(test_df, y_prob=y_test_prob)
        per_trip_df = compute_per_trip_metrics(test_df.assign(pred_prob=y_test_prob))

        assert len(per_trip_df) == test_df["trip_id"].nunique()
        assert cand_metrics["confusion_matrix"] == test_metrics["confusion_matrix"]
        assert 0.0 <= test_metrics["top1_accuracy"] <= 1.0

        # 5. Test prediction Parquet export & validation
        pred_parquet_path = tmp_path / "smoke_student_test_preds.parquet"
        save_prediction_parquet(test_df, y_test_prob, pred_parquet_path)
        reloaded_preds = pd.read_parquet(pred_parquet_path)
        validate_prediction_dataframe(reloaded_preds)
        assert list(reloaded_preds.columns) == PREDICTION_COLUMNS

        # 6. Test ExperimentLogger (CSV, JSON, and Confusion Matrix PNG)
        output_dir = Path("outputs/models/classical")
        logger = ExperimentLogger(
            output_dir=output_dir,
            csv_filename="smoke_test_log.csv",
        )
        logged_payload = logger.log_experiment(
            model_name="SmokeTest_LogisticRegression",
            algorithm_family="Linear / Probabilistic (Synthetic Smoke Test)",
            key_hyperparameters=model_params,
            random_seed=42,
            cv_results=cv_results,
            student_test_metrics=test_metrics,
            train_time_sec=train_time_sec,
            inference_time_sec=inference_time_sec,
            model_slug="smoke_test",
            save_cm_plot=True,
        )

        csv_path = output_dir / "smoke_test_log.csv"
        json_path = output_dir / "smoke_test_metrics.json"
        cm_png_path = output_dir / "smoke_test_confusion_matrix.png"

        assert csv_path.exists(), "Experiment CSV was not created."
        assert json_path.exists(), "Experiment JSON was not created."
        assert cm_png_path.exists() and cm_png_path.stat().st_size > 0, "Confusion Matrix PNG was not created."

        logged_csv_df = pd.read_csv(csv_path)
        with json_path.open("r", encoding="utf-8") as f:
            logged_json = json.load(f)

        assert logged_json["summary_row"]["model_name"] == "SmokeTest_LogisticRegression"
        assert len(logged_csv_df) == 1

    return logged_csv_df, logged_payload


def test_phase1_harness_end_to_end() -> None:
    """Pytest entry point for the full Phase 1 synthetic smoke test."""
    csv_df, payload = run_smoke_test()
    assert len(csv_df) == 1
    assert payload["summary_row"]["cv_macro_f1_mean"] > 0.70
    assert payload["summary_row"]["student_test_top1_accuracy"] > 0.70


if __name__ == "__main__":
    csv_df, payload = run_smoke_test()
    summary = payload["summary_row"]
    print("=== Person 2 Phase 1 Smoke Test PASSED ===")
    print(f"Model:                  {summary['model_name']}")
    print(f"5-Fold CV Macro F1:     {summary['cv_macro_f1_mean']:.4f} +/- {summary['cv_macro_f1_std']:.4f}")
    print(f"5-Fold CV Top-1 Acc:    {summary['cv_top1_accuracy_mean']:.4f} +/- {summary['cv_top1_accuracy_std']:.4f}")
    print(f"Test Top-1 Accuracy:    {summary['student_test_top1_accuracy']:.4f}")
    print(f"Test Macro F1:          {summary['student_test_macro_f1']:.4f}")
    print(f"Test Precision / Recall:{summary['student_test_precision']:.4f} / {summary['student_test_recall']:.4f}")
    print(f"Test Log Loss:          {summary['student_test_log_loss']:.4f}")
    print(f"Train / Inference Time: {summary['train_time_sec']:.4f}s / {summary['inference_time_sec']:.4f}s")
    print(f"Artifacts saved to:     {payload['artifacts']}")

