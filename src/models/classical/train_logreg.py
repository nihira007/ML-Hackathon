"""Model 1 (Linear / Probabilistic Baseline): Logistic Regression for Candidate Road Segment Classification.

Trains an L2-regularized Logistic Regression classifier (`class_weight='balanced'`, `random_state=42`)
with `StandardScaler` fit strictly inside each 5-fold `GroupKFold` training split (grouped by `trip_id`
across the 80 training trips), then refits on all 80 training trips and evaluates on the 10
`student_test` trips.

Outputs:
- `outputs/models/classical/logreg_metrics.json`
- `outputs/models/classical/logreg_confusion_matrix.png`
- `outputs/models/classical/classical_comparison.csv`
- `outputs/models/classical/logreg_student_test_preds.parquet`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval import (
    ExperimentLogger,
    Timer,
    compute_topk_accuracy,
    evaluate_predictions,
    run_group_kfold_cv,
    save_prediction_parquet,
)
from src.models.classical.data_utils import PROJECT_ROOT, load_feature_datasets

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"
PREDS_PARQUET_PATH = OUTPUT_DIR / "logreg_student_test_preds.parquet"

LOGREG_HYPERPARAMETERS: dict[str, Any] = {
    "C": 1.0,
    "penalty": "l2",
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 500,
    "random_state": 42,
}


def build_logreg_estimator() -> LogisticRegression:
    """Create a fresh LogisticRegression instance with fixed baseline hyperparameters."""
    return LogisticRegression(
        C=LOGREG_HYPERPARAMETERS["C"],
        class_weight=LOGREG_HYPERPARAMETERS["class_weight"],
        solver=LOGREG_HYPERPARAMETERS["solver"],
        max_iter=LOGREG_HYPERPARAMETERS["max_iter"],
        random_state=LOGREG_HYPERPARAMETERS["random_state"],
    )


def run_logreg_pipeline(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Execute 5-fold GroupKFold CV on 80 train trips and evaluate on 10 student_test trips."""
    print("[Logistic Regression] Loading train (80 trips) and student_test (10 trips) features...")
    train_df, test_df, feature_cols = load_feature_datasets()
    print(
        f"[Logistic Regression] Loaded {len(train_df):,} train candidates "
        f"({train_df['trip_id'].nunique()} trips) and {len(test_df):,} student_test candidates "
        f"({test_df['trip_id'].nunique()} trips) across {len(feature_cols)} features: {feature_cols}"
    )

    # 1. 5-fold GroupKFold CV grouped strictly by trip_id on the 80 training trips
    print("[Logistic Regression] Running 5-fold GroupKFold CV by trip_id...")
    cv_results = run_group_kfold_cv(
        model_fn=build_logreg_estimator,
        df=train_df,
        feature_cols=feature_cols,
        n_splits=5,
        scale_features=True,
        random_state=42,
    )
    print(
        f"[Logistic Regression] 5-Fold CV Top-1 Acc: {cv_results['cv_top1_accuracy_mean']:.4f} "
        f"(±{cv_results['cv_top1_accuracy_std']:.4f}) | "
        f"Macro F1: {cv_results['cv_macro_f1_mean']:.4f} | "
        f"Log Loss: {cv_results['cv_log_loss_mean']:.4f}"
    )

    # 2. Full training fit on all 80 training trips (strictly no student_test leakage)
    x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64, copy=True)
    y_train = train_df["is_match"].to_numpy(dtype=np.int64, copy=True)
    x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float64, copy=True)

    train_medians = np.nanmedian(x_train_raw, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    for col_idx in range(x_train_raw.shape[1]):
        nan_train = np.isnan(x_train_raw[:, col_idx])
        if np.any(nan_train):
            x_train_raw[nan_train, col_idx] = train_medians[col_idx]
        nan_test = np.isnan(x_test_raw[:, col_idx])
        if np.any(nan_test):
            x_test_raw[nan_test, col_idx] = train_medians[col_idx]

    scaler = StandardScaler()
    model = build_logreg_estimator()

    with Timer() as fit_timer:
        x_train_scaled = scaler.fit_transform(x_train_raw)
        model.fit(x_train_scaled, y_train)
    train_time_sec = fit_timer.elapsed_sec

    # 3. Inference on 10 student_test trips
    with Timer() as infer_timer:
        x_test_scaled = scaler.transform(x_test_raw)
        test_pred_prob = model.predict_proba(x_test_scaled)[:, 1]
    inference_time_sec = infer_timer.elapsed_sec

    test_metrics = evaluate_predictions(test_df, test_pred_prob, threshold=0.5)
    eval_df = test_df.copy()
    eval_df["pred_prob"] = test_pred_prob
    test_metrics["top3_accuracy"] = compute_topk_accuracy(eval_df, k=3)
    print(
        f"[Logistic Regression] student_test Top-1 Acc: {test_metrics['top1_accuracy']:.4f} | "
        f"Top-3 Acc: {test_metrics['top3_accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | Binary F1: {test_metrics['binary_f1']:.4f} | "
        f"Log Loss: {test_metrics['log_loss']:.4f} | "
        f"Train: {train_time_sec:.3f}s | Infer: {inference_time_sec:.3f}s"
    )

    # 4. Save standardized prediction parquet and log metrics + confusion matrix
    preds_path = output_dir / "logreg_student_test_preds.parquet"
    save_prediction_parquet(test_df, test_pred_prob, preds_path)

    feature_coefficients = {
        col: round(float(coef), 6)
        for col, coef in zip(feature_cols, model.coef_.ravel())
    }

    logger = ExperimentLogger(output_dir=output_dir)
    record = logger.log_experiment(
        model_name="Logistic Regression",
        algorithm_family="Linear / Probabilistic",
        key_hyperparameters=LOGREG_HYPERPARAMETERS,
        random_seed=42,
        cv_results=cv_results,
        student_test_metrics=test_metrics,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        model_slug="logreg",
        extra_metadata={
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "n_train_trips": int(train_df["trip_id"].nunique()),
            "n_train_candidates": int(len(train_df)),
            "n_test_trips": int(test_df["trip_id"].nunique()),
            "n_test_candidates": int(len(test_df)),
            "intercept": round(float(model.intercept_[0]), 6),
            "standardized_coefficients": feature_coefficients,
            "predictions_parquet": str(preds_path),
        },
    )
    return record


if __name__ == "__main__":
    run_logreg_pipeline()
