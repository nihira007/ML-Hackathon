"""Model 3 (Tree / Bagging Ensemble): Random Forest Classifier for Candidate Road Segment Classification.

Trains a regularized Random Forest ensemble (`n_estimators=120`, `max_depth=14`, `min_samples_leaf=25`,
`max_features='sqrt'`, `class_weight='balanced_subsample'`, `random_state=42`) evaluated with 5-fold
`GroupKFold` CV grouped strictly by `trip_id` across the 80 training trips, then refits on the 80
training trips and evaluates on the 10 `student_test` trips.

Uses deterministic observation-grouped subsampling (`max_train_samples=320_000`, `40,000` complete
observations with all 8 candidates per observation) so 5-fold CV and full training complete rapidly
in single-process execution while capturing rich non-linear interactions across all 80 trips.

Outputs:
- `outputs/models/classical/rf_metrics.json`
- `outputs/models/classical/rf_confusion_matrix.png`
- `outputs/models/classical/classical_comparison.csv`
- `outputs/models/classical/rf_student_test_preds.parquet`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
PREDS_PARQUET_PATH = OUTPUT_DIR / "rf_student_test_preds.parquet"

RF_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 14,
    "min_samples_leaf": 25,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "max_train_samples": 200000,
    "random_state": 42,
}


def build_rf_estimator() -> RandomForestClassifier:
    """Create a fresh RandomForestClassifier instance with depth/leaf regularization."""
    return RandomForestClassifier(
        n_estimators=RF_HYPERPARAMETERS["n_estimators"],
        criterion=RF_HYPERPARAMETERS["criterion"],
        max_depth=RF_HYPERPARAMETERS["max_depth"],
        min_samples_leaf=RF_HYPERPARAMETERS["min_samples_leaf"],
        max_features=RF_HYPERPARAMETERS["max_features"],
        class_weight=RF_HYPERPARAMETERS["class_weight"],
        random_state=RF_HYPERPARAMETERS["random_state"],
    )


def run_rf_pipeline(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Execute 5-fold GroupKFold CV on 80 train trips and evaluate on 10 student_test trips."""
    if train_df is None or test_df is None or feature_cols is None:
        print("[Random Forest] Loading train (80 trips) and student_test (10 trips) v2 features...")
        train_df, test_df, feature_cols = load_feature_datasets(require_v2=True)

    print(
        f"[Random Forest] Loaded {len(train_df):,} train candidates "
        f"({train_df['trip_id'].nunique()} trips) and {len(test_df):,} student_test candidates "
        f"({test_df['trip_id'].nunique()} trips) across {len(feature_cols)} features."
    )

    max_train_samples = int(RF_HYPERPARAMETERS["max_train_samples"])

    # 1. 5-fold GroupKFold CV grouped strictly by trip_id on the 80 training trips
    print("[Random Forest] Running 5-fold GroupKFold CV by trip_id...")
    cv_results = run_group_kfold_cv(
        model_fn=build_rf_estimator,
        df=train_df,
        feature_cols=feature_cols,
        n_splits=5,
        scale_features=False,
        max_train_samples_per_fold=max_train_samples,
        random_state=42,
    )
    print(
        f"[Random Forest] 5-Fold CV Top-1 Acc: {cv_results['cv_top1_accuracy_mean']:.4f} "
        f"(±{cv_results['cv_top1_accuracy_std']:.4f}) | "
        f"Macro F1: {cv_results['cv_macro_f1_mean']:.4f} | "
        f"Log Loss: {cv_results['cv_log_loss_mean']:.4f}"
    )

    # 2. Full training fit on the 80 training trips (strictly no student_test leakage)
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

    model = build_rf_estimator()
    rng = np.random.default_rng(42)

    with Timer() as fit_timer:
        if len(x_train_raw) > max_train_samples:
            obs_codes = train_df.groupby(["trip_id", "obs_idx"], sort=False).ngroup().to_numpy()
            n_obs_total = int(obs_codes.max()) + 1
            avg_cands = max(1.0, len(x_train_raw) / float(n_obs_total))
            target_obs = max(1, min(n_obs_total, int(max_train_samples / avg_cands)))
            chosen_obs = rng.choice(n_obs_total, size=target_obs, replace=False)
            chosen_idx = np.nonzero(np.isin(obs_codes, chosen_obs))[0]
            x_fit = x_train_raw[chosen_idx]
            y_fit = y_train[chosen_idx]
        else:
            x_fit = x_train_raw
            y_fit = y_train
        model.fit(x_fit, y_fit)
    train_time_sec = fit_timer.elapsed_sec

    # 3. Inference on 10 student_test trips
    with Timer() as infer_timer:
        test_pred_prob = model.predict_proba(x_test_raw)[:, 1]
    inference_time_sec = infer_timer.elapsed_sec

    test_metrics = evaluate_predictions(test_df, test_pred_prob, threshold=0.5)
    eval_df = test_df.copy()
    eval_df["pred_prob"] = test_pred_prob
    test_metrics["top3_accuracy"] = compute_topk_accuracy(eval_df, k=3)
    print(
        f"[Random Forest] student_test Top-1 Acc: {test_metrics['top1_accuracy']:.4f} | "
        f"Top-3 Acc: {test_metrics['top3_accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | Binary F1: {test_metrics['binary_f1']:.4f} | "
        f"Log Loss: {test_metrics['log_loss']:.4f} | "
        f"Train: {train_time_sec:.3f}s | Infer: {inference_time_sec:.3f}s"
    )

    # 4. Save standardized prediction parquet and log metrics + confusion matrix + feature importances
    preds_path = output_dir / "rf_student_test_preds.parquet"
    save_prediction_parquet(test_df, test_pred_prob, preds_path)

    feature_importances = {
        col: round(float(imp), 6)
        for col, imp in sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    }

    logger = ExperimentLogger(output_dir=output_dir)
    record = logger.log_experiment(
        model_name="Random Forest",
        algorithm_family="Bagging / Tree Ensemble",
        key_hyperparameters=RF_HYPERPARAMETERS,
        random_seed=42,
        cv_results=cv_results,
        student_test_metrics=test_metrics,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        model_slug="rf",
        extra_metadata={
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "n_train_trips": int(train_df["trip_id"].nunique()),
            "n_train_candidates_total": int(len(train_df)),
            "n_train_candidates_fitted": int(len(x_fit)),
            "n_test_trips": int(test_df["trip_id"].nunique()),
            "n_test_candidates": int(len(test_df)),
            "feature_importances_gini": feature_importances,
            "predictions_parquet": str(preds_path),
        },
    )
    return record


if __name__ == "__main__":
    run_rf_pipeline()
