"""
Shared evaluation harness and experiment logging utilities for PathSnap (Person 2).

Owned by branch: feat/p2-models-classical
"""

from src.eval.metrics import (
    REQUIRED_ID_COLUMNS,
    PREDICTION_COLUMNS,
    Timer,
    measure_callable,
    compute_candidate_metrics,
    compute_top1_accuracy,
    compute_topk_accuracy,
    compute_per_trip_metrics,
    evaluate_predictions,
    get_trip_group_kfold,
    iter_trip_cv_splits,
    run_group_kfold_cv,
    format_prediction_df,
    save_prediction_parquet,
    validate_feature_dataframe,
    validate_prediction_dataframe,
)
from src.eval.logger import (
    STANDARD_LOG_COLUMNS,
    ExperimentLogger,
    format_hyperparameters,
    save_metrics_json,
    append_experiment_csv,
    plot_confusion_matrix,
)

__all__ = [
    "REQUIRED_ID_COLUMNS",
    "PREDICTION_COLUMNS",
    "STANDARD_LOG_COLUMNS",
    "Timer",
    "measure_callable",
    "compute_candidate_metrics",
    "compute_top1_accuracy",
    "compute_topk_accuracy",
    "compute_per_trip_metrics",
    "evaluate_predictions",
    "get_trip_group_kfold",
    "iter_trip_cv_splits",
    "run_group_kfold_cv",
    "format_prediction_df",
    "save_prediction_parquet",
    "validate_feature_dataframe",
    "validate_prediction_dataframe",
    "ExperimentLogger",
    "format_hyperparameters",
    "save_metrics_json",
    "append_experiment_csv",
    "plot_confusion_matrix",
]
