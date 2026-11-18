"""
Standardized experiment logging (CSV/JSON) and confusion matrix plotting utilities.

Owned by Person 2 (branch: feat/p2-models-classical).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

STANDARD_LOG_COLUMNS: Tuple[str, ...] = (
    "model_name",
    "algorithm_family",
    "key_hyperparameters",
    "random_seed",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "cv_top1_accuracy_mean",
    "cv_top1_accuracy_std",
    "cv_log_loss_mean",
    "train_macro_f1_mean",
    "train_top1_accuracy_mean",
    "student_test_top1_accuracy",
    "student_test_macro_f1",
    "student_test_binary_f1",
    "student_test_precision",
    "student_test_recall",
    "student_test_accuracy",
    "student_test_log_loss",
    "student_test_brier_score",
    "train_time_sec",
    "inference_time_sec",
    "timestamp_utc",
)


def format_hyperparameters(params: Union[str, Mapping[str, Any]]) -> str:
    """Format a dictionary of key hyperparameters into a deterministic, human-readable string."""
    if isinstance(params, str):
        return params
    items = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return ", ".join(items)


def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars/arrays and Paths cleanly to JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_metrics_json(metrics_data: Mapping[str, Any], output_path: Union[str, Path]) -> Path:
    """Save a metrics dictionary as formatted JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dict(metrics_data), f, indent=2, default=_json_default)
    return out_path


def append_experiment_csv(
    row_dict: Mapping[str, Any],
    csv_path: Union[str, Path],
    upsert_by: Optional[str] = "model_name",
) -> pd.DataFrame:
    """Append or upsert a standardized experiment row into a CSV log file.

    Preserves column ordering starting with `STANDARD_LOG_COLUMNS`.
    """
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_row_df = pd.DataFrame([dict(row_dict)])

    if out_path.exists() and out_path.stat().st_size > 0:
        existing_df = pd.read_csv(out_path)
        if upsert_by and upsert_by in existing_df.columns and upsert_by in new_row_df.columns:
            key_val = new_row_df.iloc[0][upsert_by]
            existing_df = existing_df[existing_df[upsert_by] != key_val]
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    ordered_cols = [c for c in STANDARD_LOG_COLUMNS if c in combined_df.columns]
    extra_cols = [c for c in combined_df.columns if c not in ordered_cols]
    combined_df = combined_df[ordered_cols + extra_cols]
    combined_df.to_csv(out_path, index=False)
    return combined_df


def plot_confusion_matrix(
    cm_or_y_true: Union[np.ndarray, Sequence[Sequence[int]], Sequence[int]],
    y_pred: Optional[Sequence[int]] = None,
    model_name: str = "Model",
    output_path: Union[str, Path] = "outputs/models/classical/confusion_matrix.png",
    class_names: Tuple[str, str] = ("Non-Match (0)", "Match (1)"),
    subtitle: Optional[str] = None,
    dpi: int = 160,
) -> Path:
    """Render and save a confusion matrix PNG with both raw counts and row-normalized percentages.

    Parameters
    ----------
    cm_or_y_true:
        Either a precomputed 2x2 confusion matrix `[[TN, FP], [FN, TP]]` or 1D `y_true` array.
    y_pred:
        1D binary predictions array if `cm_or_y_true` is `y_true`.
    model_name:
        Model display name for the plot title.
    output_path:
        Destination PNG file path.
    class_names:
        Display labels for negative (0) and positive (1) candidate classes.
    subtitle:
        Optional subtitle string (e.g., Top-1 Accuracy & Macro F1 summary).
    dpi:
        Output image resolution.
    """
    arr = np.asarray(cm_or_y_true)
    if y_pred is not None:
        cm = confusion_matrix(arr.astype(np.int64), np.asarray(y_pred, dtype=np.int64), labels=[0, 1])
    else:
        if arr.shape != (2, 2):
            raise ValueError(
                f"Expected a 2x2 confusion matrix when y_pred is None, got shape {arr.shape}."
            )
        cm = arr.astype(np.int64)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums > 0,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=dpi)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-Normalized Proportion", rotation=270, labelpad=14, fontsize=9)

    title_text = f"Confusion Matrix — {model_name}"
    if subtitle:
        title_text += f"\n{subtitle}"
    ax.set_title(title_text, fontsize=11, fontweight="bold", pad=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=9.5)
    ax.set_yticklabels(class_names, fontsize=9.5)
    ax.set_xlabel("Predicted Candidate Label", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground-Truth Candidate Label", fontsize=10, fontweight="bold", labelpad=8)

    quadrant_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            count_val = cm[i, j]
            pct_val = cm_norm[i, j] * 100.0
            q_label = quadrant_labels[i][j]
            text_color = "white" if cm_norm[i, j] > 0.55 else "#1a1a1a"
            cell_text = f"{q_label}\n{count_val:,}\n({pct_val:.1f}%)"
            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


class ExperimentLogger:
    """Standardized experiment logger for PathSnap candidate classifiers."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/models/classical",
        csv_filename: str = "classical_comparison.csv",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_filename

    def log_experiment(
        self,
        model_name: str,
        algorithm_family: str,
        key_hyperparameters: Union[str, Mapping[str, Any]],
        random_seed: int,
        cv_results: Mapping[str, Any],
        student_test_metrics: Mapping[str, Any],
        train_time_sec: float,
        inference_time_sec: float,
        model_slug: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        save_cm_plot: bool = True,
    ) -> Dict[str, Any]:
        """Record a complete model evaluation run to CSV, JSON, and Confusion Matrix PNG.

        Parameters
        ----------
        model_name:
            Human-readable model name (e.g., `"Logistic Regression"`).
        algorithm_family:
            Model family description (e.g., `"Linear / Probabilistic"`).
        key_hyperparameters:
            Dictionary or formatted string of key hyperparameters.
        random_seed:
            Random seed used for training/CV.
        cv_results:
            Output dictionary from `run_group_kfold_cv` (or dict containing `cv_macro_f1_mean`, `cv_macro_f1_std`).
        student_test_metrics:
            Output dictionary from `evaluate_predictions` on `student_test`.
        train_time_sec:
            Wall-clock training time on the full training set in seconds.
        inference_time_sec:
            Wall-clock inference time on the `student_test` set in seconds.
        model_slug:
            Short filename prefix (e.g., `"logreg"`, `"knn"`, `"rf"`).
        extra_metadata:
            Optional additional fields to persist in the model's detailed JSON file.
        save_cm_plot:
            If True and `confusion_matrix` is in `student_test_metrics`, saves `<slug>_confusion_matrix.png`.
        """
        slug = (
            model_slug
            if model_slug is not None
            else model_name.lower().replace(" ", "_").replace("-", "_")
        )
        formatted_params = format_hyperparameters(key_hyperparameters)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        row: Dict[str, Any] = {
            "model_name": model_name,
            "algorithm_family": algorithm_family,
            "key_hyperparameters": formatted_params,
            "random_seed": int(random_seed),
            "cv_macro_f1_mean": float(cv_results.get("cv_macro_f1_mean", np.nan)),
            "cv_macro_f1_std": float(cv_results.get("cv_macro_f1_std", np.nan)),
            "cv_top1_accuracy_mean": float(cv_results.get("cv_top1_accuracy_mean", np.nan)),
            "cv_top1_accuracy_std": float(cv_results.get("cv_top1_accuracy_std", np.nan)),
            "cv_log_loss_mean": float(cv_results.get("cv_log_loss_mean", np.nan)),
            "train_macro_f1_mean": float(cv_results.get("train_macro_f1_mean", np.nan)),
            "train_top1_accuracy_mean": float(cv_results.get("train_top1_accuracy_mean", np.nan)),
            "student_test_top1_accuracy": float(student_test_metrics.get("top1_accuracy", np.nan)),
            "student_test_macro_f1": float(student_test_metrics.get("macro_f1", np.nan)),
            "student_test_binary_f1": float(student_test_metrics.get("binary_f1", np.nan)),
            "student_test_precision": float(student_test_metrics.get("precision", np.nan)),
            "student_test_recall": float(student_test_metrics.get("recall", np.nan)),
            "student_test_accuracy": float(student_test_metrics.get("accuracy", np.nan)),
            "student_test_log_loss": float(student_test_metrics.get("log_loss", np.nan)),
            "student_test_brier_score": float(student_test_metrics.get("brier_score", np.nan)),
            "train_time_sec": float(train_time_sec),
            "inference_time_sec": float(inference_time_sec),
            "timestamp_utc": now_utc,
        }

        append_experiment_csv(row, self.csv_path, upsert_by="model_name")

        cm_plot_path: Optional[Path] = None
        if save_cm_plot and "confusion_matrix" in student_test_metrics:
            cm_plot_path = self.output_dir / f"{slug}_confusion_matrix.png"
            subtitle = (
                f"Top-1 Acc: {row['student_test_top1_accuracy']:.4f} | "
                f"Macro F1: {row['student_test_macro_f1']:.4f} | "
                f"Log Loss: {row['student_test_log_loss']:.4f}"
            )
            plot_confusion_matrix(
                cm_or_y_true=student_test_metrics["confusion_matrix"],
                model_name=model_name,
                output_path=cm_plot_path,
                subtitle=subtitle,
            )

        cv_serializable = {
            k: v for k, v in cv_results.items() if k != "oof_predictions"
        }
        detailed_payload: Dict[str, Any] = {
            "summary_row": row,
            "hyperparameters_raw": (
                dict(key_hyperparameters)
                if isinstance(key_hyperparameters, Mapping)
                else key_hyperparameters
            ),
            "cv_results": cv_serializable,
            "student_test_metrics": dict(student_test_metrics),
            "artifacts": {
                "csv_log_path": str(self.csv_path),
                "confusion_matrix_png": str(cm_plot_path) if cm_plot_path else None,
            },
        }
        if extra_metadata:
            detailed_payload["extra_metadata"] = dict(extra_metadata)

        json_path = self.output_dir / f"{slug}_metrics.json"
        save_metrics_json(detailed_payload, json_path)
        detailed_payload["artifacts"]["metrics_json_path"] = str(json_path)

        return detailed_payload
