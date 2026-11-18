"""Unified runner for Person 2 Classical Models 1–3 on identical v2 features and 5-fold GroupKFold splits.

Executes in order:
1. Model 1 — Logistic Regression (`src/models/classical/train_logreg.py`)
2. Model 2 — Distance-Weighted KNN (`src/models/classical/train_knn.py`)
3. Model 3 — Random Forest Classifier (`src/models/classical/train_rf.py`)

Guarantees that all 3 models are evaluated on the exact same 21 v2 domain features and identical
5-fold `GroupKFold` trip splits across the 80 training trips, and exports the consolidated table to
`outputs/models/classical/classical_comparison.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.classical.data_utils import PROJECT_ROOT, load_feature_datasets
from src.models.classical.train_knn import run_knn_pipeline
from src.models.classical.train_logreg import run_logreg_pipeline
from src.models.classical.train_rf import run_rf_pipeline

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"


def run_all_classical_models(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Run Models 1, 2, and 3 on identical v2 features and return the comparison DataFrame."""
    print("=" * 84)
    print("PathSnap Person 2 — Unified Classical & Ensemble Benchmark (Models 1–3 on v2 Features)")
    print("=" * 84)

    # Ensure v2 feature datasets are built/loaded once before running Models 1–3
    train_df, test_df, feature_cols = load_feature_datasets(require_v2=True)
    print(
        f"Dataset ready: {len(train_df):,} train rows ({train_df['trip_id'].nunique()} trips), "
        f"{len(test_df):,} student_test rows ({test_df['trip_id'].nunique()} trips), "
        f"{len(feature_cols)} v2 features.\n"
    )

    records: dict[str, Any] = {}
    records["logreg"] = run_logreg_pipeline(output_dir=output_dir)
    print("-" * 84)
    records["knn"] = run_knn_pipeline(output_dir=output_dir)
    print("-" * 84)
    records["rf"] = run_rf_pipeline(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        output_dir=output_dir,
    )
    print("=" * 84)

    csv_path = output_dir / "classical_comparison.csv"
    comparison_df = pd.read_csv(csv_path)
    display_cols = [
        "model_name",
        "cv_top1_accuracy_mean",
        "cv_macro_f1_mean",
        "train_macro_f1_mean",
        "student_test_top1_accuracy",
        "student_test_macro_f1",
        "student_test_log_loss",
        "student_test_brier_score",
        "train_time_sec",
        "inference_time_sec",
    ]
    print("\nConsolidated Benchmark Summary (outputs/models/classical/classical_comparison.csv):")
    print(comparison_df[display_cols].to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    run_all_classical_models()
