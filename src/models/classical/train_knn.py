"""Model 2 (Instance-based / Non-parametric Baseline): Distance-Weighted KNN for Candidate Road Segment Classification.

Trains a Distance-Weighted K-Nearest Neighbors classifier (`n_neighbors=15`, `weights='distance'`,
`algorithm='kd_tree'`) on `StandardScaler`-normalized spatial features fit strictly inside each
5-fold `GroupKFold` training split (grouped by `trip_id` across the 80 training trips) and
evaluates on the 10 `student_test` trips.

Uses deterministic subsampling of the training index (`max_train_samples=150_000` with
`random_state=42`) to bound KD-tree memory and query latency while preserving positive/negative class
structure across all 80 training trips.

Outputs:
- `outputs/models/classical/knn_metrics.json`
- `outputs/models/classical/knn_confusion_matrix.png`
- `outputs/models/classical/classical_comparison.csv`
- `outputs/models/classical/knn_student_test_preds.parquet`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
PREDS_PARQUET_PATH = OUTPUT_DIR / "knn_student_test_preds.parquet"

KNN_HYPERPARAMETERS: dict[str, Any] = {
    "n_neighbors": 15,
    "weights": "distance",
    "algorithm": "kd_tree",
    "leaf_size": 40,
    "metric": "minkowski",
    "p": 2,
    "max_train_samples": 150000,
    "random_state": 42,
}


def build_knn_estimator() -> KNeighborsClassifier:
    """Create a fresh KNeighborsClassifier instance with distance weighting and KD-tree indexing."""
    return KNeighborsClassifier(
        n_neighbors=KNN_HYPERPARAMETERS["n_neighbors"],
        weights=KNN_HYPERPARAMETERS["weights"],
        algorithm=KNN_HYPERPARAMETERS["algorithm"],
        leaf_size=KNN_HYPERPARAMETERS["leaf_size"],
        metric=KNN_HYPERPARAMETERS["metric"],
        p=KNN_HYPERPARAMETERS["p"],
    )


def run_knn_pipeline(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Execute 5-fold GroupKFold CV on 80 train trips and evaluate on 10 student_test trips."""
    print("[Distance-Weighted KNN] Loading train (80 trips) and student_test (10 trips) features...")
    train_df, test_df, feature_cols = load_feature_datasets()
    print(
        f"[Distance-Weighted KNN] Loaded {len(train_df):,} train candidates "
        f"({train_df['trip_id'].nunique()} trips) and {len(test_df):,} student_test candidates "
        f"({test_df['trip_id'].nunique()} trips) across {len(feature_cols)} features."
    )

    max_train_samples = int(KNN_HYPERPARAMETERS["max_train_samples"])

    # 1. 5-fold GroupKFold CV grouped strictly by trip_id on the 80 training trips
    print("[Distance-Weighted KNN] Running 5-fold GroupKFold CV by trip_id...")
    cv_results = run_group_kfold_cv(
        model_fn=build_knn_estimator,
        df=train_df,
        feature_cols=feature_cols,
        n_splits=5,
        scale_features=True,
        max_train_samples_per_fold=max_train_samples,
        random_state=42,
    )
    print(
        f"[Distance-Weighted KNN] 5-Fold CV Top-1 Acc: {cv_results['cv_top1_accuracy_mean']:.4f} "
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

    scaler = StandardScaler()
    model = build_knn_estimator()

    rng = np.random.default_rng(42)
    with Timer() as fit_timer:
        x_train_scaled = scaler.fit_transform(x_train_raw)
        if len(x_train_scaled) > max_train_samples:
            obs_codes = train_df.groupby(["trip_id", "obs_idx"], sort=False).ngroup().to_numpy()
            n_obs_total = int(obs_codes.max()) + 1
            avg_cands = max(1.0, len(x_train_scaled) / float(n_obs_total))
            target_obs = max(1, min(n_obs_total, int(max_train_samples / avg_cands)))
            chosen_obs = rng.choice(n_obs_total, size=target_obs, replace=False)
            chosen_idx = np.nonzero(np.isin(obs_codes, chosen_obs))[0]
            x_fit = x_train_scaled[chosen_idx]
            y_fit = y_train[chosen_idx]
        else:
            x_fit = x_train_scaled
            y_fit = y_train
        model.fit(x_fit, y_fit)
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
        f"[Distance-Weighted KNN] student_test Top-1 Acc: {test_metrics['top1_accuracy']:.4f} | "
        f"Top-3 Acc: {test_metrics['top3_accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | Binary F1: {test_metrics['binary_f1']:.4f} | "
        f"Log Loss: {test_metrics['log_loss']:.4f} | "
        f"Train: {train_time_sec:.3f}s | Infer: {inference_time_sec:.3f}s"
    )

    # 4. Save standardized prediction parquet and log metrics + confusion matrix
    preds_path = output_dir / "knn_student_test_preds.parquet"
    save_prediction_parquet(test_df, test_pred_prob, preds_path)

    logger = ExperimentLogger(output_dir=output_dir)
    record = logger.log_experiment(
        model_name="Distance-Weighted KNN",
        algorithm_family="Instance-based / Non-parametric",
        key_hyperparameters=KNN_HYPERPARAMETERS,
        random_seed=42,
        cv_results=cv_results,
        student_test_metrics=test_metrics,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        model_slug="knn",
        extra_metadata={
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "n_train_trips": int(train_df["trip_id"].nunique()),
            "n_train_candidates_total": int(len(train_df)),
            "n_train_candidates_indexed": int(len(x_fit)),
            "n_test_trips": int(test_df["trip_id"].nunique()),
            "n_test_candidates": int(len(test_df)),
            "predictions_parquet": str(preds_path),
        },
    )
    return record


if __name__ == "__main__":
    run_knn_pipeline()
