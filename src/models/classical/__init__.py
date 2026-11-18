"""Person 2 classical and ensemble machine learning models for candidate road segment classification."""

from src.models.classical.calibration_and_error_analysis import (
    compute_ece_and_bins,
    plot_calibration_curves,
    run_calibration_and_error_analysis,
)
from src.models.classical.data_utils import V1_SPATIAL_FEATURE_COLUMNS, V2_FEATURE_COLUMNS, load_feature_datasets
from src.models.classical.run_all_classical import run_all_classical_models
from src.models.classical.train_knn import build_knn_estimator, run_knn_pipeline
from src.models.classical.train_logreg import build_logreg_estimator, run_logreg_pipeline
from src.models.classical.train_rf import build_rf_estimator, run_rf_pipeline

__all__ = [
    "V1_SPATIAL_FEATURE_COLUMNS",
    "V2_FEATURE_COLUMNS",
    "load_feature_datasets",
    "build_logreg_estimator",
    "run_logreg_pipeline",
    "build_knn_estimator",
    "run_knn_pipeline",
    "build_rf_estimator",
    "run_rf_pipeline",
    "run_all_classical_models",
    "compute_ece_and_bins",
    "plot_calibration_curves",
    "run_calibration_and_error_analysis",
]
