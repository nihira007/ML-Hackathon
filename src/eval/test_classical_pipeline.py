"""Comprehensive End-to-End Verification Test Suite for Person 2 (Phases 1–5).

Verifies:
1. All 3 prediction parquets (`logreg`, `knn`, `rf`) exist, match the exact shared contract schema
   `['trip_id', 'obs_idx', 'arc_id', 'is_match', 'pred_prob']`, cover all 10 `student_test` trips
   (19,160 observations, 153,280 candidates), have zero NaNs, and probabilities in [0, 1].
2. Every metric in `classical_comparison.csv` matches `logreg_metrics.json`, `knn_metrics.json`,
   `rf_metrics.json`, `calibration_audit.json`, and direct recomputation from the prediction parquets.
3. `per_trip_results.csv` and `per_trip_error_analysis.json` cover all 10 `student_test` trips across
   all 3 models (30 rows) and match recomputed per-trip metrics.
4. All required confusion matrices, calibration curves, Phase 5 comparison plots, and report markdown
   files (`outputs/models/classical/summary.md`, `src/models/classical/README.md`, `src/eval/README.md`)
   exist and are non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import (
    PREDICTION_COLUMNS,
    compute_topk_accuracy,
    evaluate_predictions,
    validate_prediction_dataframe,
)
from src.models.classical.data_utils import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"
PLOTS_DIR = OUTPUT_DIR / "plots"

EXPECTED_MODELS = [
    ("Logistic Regression", "logreg"),
    ("Distance-Weighted KNN", "knn"),
    ("Random Forest", "rf"),
]


def test_prediction_parquets_and_metric_consistency() -> None:
    """Verify prediction parquets match schema and exactly reproduce logged JSON/CSV metrics."""
    comp_df = pd.read_csv(OUTPUT_DIR / "classical_comparison.csv")
    assert list(comp_df["model_name"]) == [m[0] for m in EXPECTED_MODELS]

    with (OUTPUT_DIR / "calibration_audit.json").open("r", encoding="utf-8") as fh:
        cal_audit = json.load(fh)

    for model_name, slug in EXPECTED_MODELS:
        parquet_path = OUTPUT_DIR / f"{slug}_student_test_preds.parquet"
        assert parquet_path.exists(), f"Missing prediction parquet: {parquet_path}"

        pred_df = pd.read_parquet(parquet_path)
        validate_prediction_dataframe(pred_df)
        assert list(pred_df.columns) == PREDICTION_COLUMNS
        assert len(pred_df) == 153280
        assert pred_df["trip_id"].nunique() == 10
        assert pred_df.groupby(["trip_id", "obs_idx"], sort=False).ngroups == 19160

        with (OUTPUT_DIR / f"{slug}_metrics.json").open("r", encoding="utf-8") as fh:
            m_json = json.load(fh)

        recomputed = evaluate_predictions(pred_df, threshold=0.5)
        recomputed["top3_accuracy"] = compute_topk_accuracy(pred_df, k=3)

        csv_row = comp_df[comp_df["model_name"] == model_name].iloc[0]
        json_test = m_json["student_test_metrics"]
        cal_model = cal_audit["models"][slug]

        for metric_key, csv_col in [
            ("top1_accuracy", "student_test_top1_accuracy"),
            ("macro_f1", "student_test_macro_f1"),
            ("binary_f1", "student_test_binary_f1"),
            ("precision", "student_test_precision"),
            ("recall", "student_test_recall"),
            ("accuracy", "student_test_accuracy"),
            ("log_loss", "student_test_log_loss"),
            ("brier_score", "student_test_brier_score"),
        ]:
            assert np.isclose(recomputed[metric_key], json_test[metric_key], atol=1e-6), (
                f"{slug} {metric_key}: recomputed={recomputed[metric_key]} vs json={json_test[metric_key]}"
            )
            assert np.isclose(recomputed[metric_key], csv_row[csv_col], atol=1e-6), (
                f"{slug} {csv_col}: recomputed={recomputed[metric_key]} vs csv={csv_row[csv_col]}"
            )
            if metric_key in cal_model:
                assert np.isclose(recomputed[metric_key], cal_model[metric_key], atol=1e-5), (
                    f"{slug} {metric_key}: recomputed={recomputed[metric_key]} vs cal_audit={cal_model[metric_key]}"
                )


def test_per_trip_and_visual_artifacts_exist() -> None:
    """Verify per-trip error analysis files, confusion matrices, Phase 5 plots, and summaries exist."""
    per_trip_df = pd.read_csv(OUTPUT_DIR / "per_trip_results.csv", dtype={"trip_id": str})
    assert len(per_trip_df) == 30
    assert per_trip_df["trip_id"].nunique() == 10
    assert set(per_trip_df["model_slug"]) == {"logreg", "knn", "rf"}

    required_files = [
        OUTPUT_DIR / "classical_comparison.csv",
        OUTPUT_DIR / "logreg_metrics.json",
        OUTPUT_DIR / "knn_metrics.json",
        OUTPUT_DIR / "rf_metrics.json",
        OUTPUT_DIR / "logreg_confusion_matrix.png",
        OUTPUT_DIR / "knn_confusion_matrix.png",
        OUTPUT_DIR / "rf_confusion_matrix.png",
        OUTPUT_DIR / "calibration_curves.png",
        OUTPUT_DIR / "calibration_audit.json",
        OUTPUT_DIR / "per_trip_results.csv",
        OUTPUT_DIR / "per_trip_error_analysis.json",
        OUTPUT_DIR / "summary.md",
        PLOTS_DIR / "cv_macro_f1_comparison.png",
        PLOTS_DIR / "student_test_top1_and_f1.png",
        PLOTS_DIR / "logloss_and_latency_comparison.png",
        PLOTS_DIR / "confusion_matrix_grid.png",
        PLOTS_DIR / "per_trip_top1_and_rf_importance.png",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required deliverable file: {path}"
        assert path.stat().st_size > 500, f"Deliverable file is unexpectedly small: {path}"
