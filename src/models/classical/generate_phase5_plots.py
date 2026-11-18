"""Phase 5 — Publication-Ready Comparison Visualisations for Classical & Ensemble Models 1–3.

Generates real experimental comparison plots from `outputs/models/classical/classical_comparison.csv`,
`logreg_metrics.json`, `knn_metrics.json`, `rf_metrics.json`, and `per_trip_results.csv` into
`outputs/models/classical/plots/`:

1. `cv_macro_f1_comparison.png` — 5-Fold GroupKFold CV Macro F1 & Top-1 Accuracy (with fold std error bars) + Train vs Val Generalization Gap.
2. `student_test_top1_and_f1.png` — Student-Test Top-1 Accuracy, Top-3 Accuracy, Macro F1, and Binary F1 grouped bar chart.
3. `logloss_and_latency_comparison.png` — Student-Test Log Loss, Brier Score, Training Time, and Inference Latency comparison.
4. `confusion_matrix_grid.png` — Side-by-side 1x3 Confusion Matrix grid for Models 1–3 on the 10 `student_test` trips (153,280 candidates).
5. `per_trip_top1_comparison.png` — Per-trip Top-1 Accuracy across all 10 `student_test` trips alongside Random Forest top Gini feature importances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.classical.data_utils import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"
PLOTS_DIR = OUTPUT_DIR / "plots"

MODEL_ORDER: list[tuple[str, str, str]] = [
    ("Logistic Regression", "logreg", "#2563eb"),
    ("Distance-Weighted KNN", "knn", "#d97706"),
    ("Random Forest", "rf", "#059669"),
]


def _load_artifacts(output_dir: Path = OUTPUT_DIR) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """Load real experimental CSV and JSON artifacts for Models 1–3."""
    comp_df = pd.read_csv(output_dir / "classical_comparison.csv")
    per_trip_df = pd.read_csv(output_dir / "per_trip_results.csv", dtype={"trip_id": str})

    metrics_by_slug: dict[str, dict[str, Any]] = {}
    for _, slug, _ in MODEL_ORDER:
        with (output_dir / f"{slug}_metrics.json").open("r", encoding="utf-8") as fh:
            metrics_by_slug[slug] = json.load(fh)

    return comp_df, metrics_by_slug, per_trip_df


def plot_cv_macro_f1_comparison(
    metrics_by_slug: dict[str, dict[str, Any]],
    plots_dir: Path = PLOTS_DIR,
    dpi: int = 170,
) -> Path:
    """Plot 5-Fold GroupKFold CV Macro F1 & Top-1 Accuracy with std error bars and Train vs Val gap."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "cv_macro_f1_comparison.png"

    names = [m[0] for m in MODEL_ORDER]
    slugs = [m[1] for m in MODEL_ORDER]
    colors = [m[2] for m in MODEL_ORDER]

    cv_f1_means = [metrics_by_slug[s]["cv_results"]["cv_macro_f1_mean"] for s in slugs]
    cv_f1_stds = [metrics_by_slug[s]["cv_results"]["cv_macro_f1_std"] for s in slugs]
    train_f1_means = [metrics_by_slug[s]["cv_results"]["train_macro_f1_mean"] for s in slugs]

    cv_top1_means = [metrics_by_slug[s]["cv_results"]["cv_top1_accuracy_mean"] for s in slugs]
    cv_top1_stds = [metrics_by_slug[s]["cv_results"]["cv_top1_accuracy_std"] for s in slugs]
    train_top1_means = [metrics_by_slug[s]["cv_results"]["train_top1_accuracy_mean"] for s in slugs]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=dpi)
    x = np.arange(len(names))
    width = 0.35

    # Panel A: CV Macro F1 vs Train Macro F1
    ax1 = axes[0]
    b_train1 = ax1.bar(
        x - width / 2,
        train_f1_means,
        width,
        color="#94a3b8",
        edgecolor="#334155",
        linewidth=0.9,
        label="Train Fold Mean Macro F1",
    )
    b_val1 = ax1.bar(
        x + width / 2,
        cv_f1_means,
        width,
        yerr=cv_f1_stds,
        capsize=5,
        color=colors,
        edgecolor="#1e293b",
        linewidth=0.9,
        error_kw={"elinewidth": 1.4, "ecolor": "#0f172a"},
        label="5-Fold CV Val Macro F1 (±1 SD)",
    )
    for rect in b_train1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color="#334155")
    for idx, rect in enumerate(b_val1):
        h = rect.get_height()
        std_v = cv_f1_stds[idx]
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            h + std_v + 0.015,
            f"{h:.4f}\n(±{std_v:.3f})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax1.set_title("A. 5-Fold GroupKFold Macro F1 (Train vs. Validation)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Macro F1-Score", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 1.14)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Panel B: CV Top-1 Accuracy vs Train Top-1 Accuracy
    ax2 = axes[1]
    b_train2 = ax2.bar(
        x - width / 2,
        train_top1_means,
        width,
        color="#94a3b8",
        edgecolor="#334155",
        linewidth=0.9,
        label="Train Fold Mean Top-1 Acc",
    )
    b_val2 = ax2.bar(
        x + width / 2,
        cv_top1_means,
        width,
        yerr=cv_top1_stds,
        capsize=5,
        color=colors,
        edgecolor="#1e293b",
        linewidth=0.9,
        error_kw={"elinewidth": 1.4, "ecolor": "#0f172a"},
        label="5-Fold CV Val Top-1 Acc (±1 SD)",
    )
    for rect in b_train2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color="#334155")
    for idx, rect in enumerate(b_val2):
        h = rect.get_height()
        std_v = cv_top1_stds[idx]
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            h + std_v + 0.015,
            f"{h:.4f}\n(±{std_v:.3f})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax2.set_title("B. 5-Fold GroupKFold Observation Top-1 Accuracy", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Observation Top-1 Accuracy", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 1.14)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "PathSnap Models 1–3 — 5-Fold Trip-Grouped Cross-Validation & Generalization Gap (80 Train Trips)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def plot_student_test_top1_and_f1(
    metrics_by_slug: dict[str, dict[str, Any]],
    plots_dir: Path = PLOTS_DIR,
    dpi: int = 170,
) -> Path:
    """Plot Student-Test Top-1 Accuracy, Top-3 Accuracy, Macro F1, and Binary F1 across Models 1–3."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "student_test_top1_and_f1.png"

    names = [m[0] for m in MODEL_ORDER]
    slugs = [m[1] for m in MODEL_ORDER]

    top1_vals = [metrics_by_slug[s]["student_test_metrics"]["top1_accuracy"] for s in slugs]
    top3_vals = [metrics_by_slug[s]["student_test_metrics"]["top3_accuracy"] for s in slugs]
    macro_f1_vals = [metrics_by_slug[s]["student_test_metrics"]["macro_f1"] for s in slugs]
    bin_f1_vals = [metrics_by_slug[s]["student_test_metrics"]["binary_f1"] for s in slugs]

    x = np.arange(len(names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=dpi)
    b1 = ax.bar(x - 1.5 * width, top1_vals, width, label="Observation Top-1 Accuracy", color="#1d4ed8", edgecolor="#0f172a", linewidth=0.8)
    b2 = ax.bar(x - 0.5 * width, top3_vals, width, label="Observation Top-3 Accuracy", color="#38bdf8", edgecolor="#0f172a", linewidth=0.8)
    b3 = ax.bar(x + 0.5 * width, macro_f1_vals, width, label="Candidate Macro F1", color="#059669", edgecolor="#0f172a", linewidth=0.8)
    b4 = ax.bar(x + 1.5 * width, bin_f1_vals, width, label="Candidate Positive-Class F1", color="#f59e0b", edgecolor="#0f172a", linewidth=0.8)

    # Nearest-segment baseline line (0.5668 weighted across 19,160 student_test observations)
    ax.axhline(
        0.5668,
        linestyle="--",
        color="#dc2626",
        linewidth=1.5,
        label="Nearest-Segment Distance Baseline Top-1 (56.68%)",
    )

    for bars in (b1, b2, b3, b4):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.012,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )

    ax.set_title(
        "Out-of-Sample Evaluation on student_test (10 Trips, 19,160 GPS Observations, 153,280 Candidates)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.0, 1.09)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", ncol=2, fontsize=8.8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def plot_logloss_and_latency(
    metrics_by_slug: dict[str, dict[str, Any]],
    plots_dir: Path = PLOTS_DIR,
    dpi: int = 170,
) -> Path:
    """Plot Probability Quality (Log Loss & Brier Score) and Runtime Latency (Train & Inference Time)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "logloss_and_latency_comparison.png"

    names = [m[0] for m in MODEL_ORDER]
    slugs = [m[1] for m in MODEL_ORDER]

    logloss_vals = [metrics_by_slug[s]["student_test_metrics"]["log_loss"] for s in slugs]
    brier_vals = [metrics_by_slug[s]["student_test_metrics"]["brier_score"] for s in slugs]
    train_times = [metrics_by_slug[s]["summary_row"]["train_time_sec"] for s in slugs]
    infer_times = [metrics_by_slug[s]["summary_row"]["inference_time_sec"] for s in slugs]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=dpi)
    x = np.arange(len(names))
    width = 0.34

    # Panel A: Log Loss & Brier Score (Lower is Better)
    ax1 = axes[0]
    b_ll = ax1.bar(x - width / 2, logloss_vals, width, color="#7c3aed", edgecolor="#1e1b4b", linewidth=0.9, label="Binary Log Loss (↓ better)")
    b_br = ax1.bar(x + width / 2, brier_vals, width, color="#ec4899", edgecolor="#500724", linewidth=0.9, label="Brier Score (↓ better)")

    for bars in (b_ll, b_br):
        for rect in bars:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax1.set_title("A. Probability Quality on student_test (Lower is Better)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Error / Loss", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 0.42)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.8, frameon=True)

    # Panel B: Wall-Clock Train Time & Inference Time (Log Scale)
    ax2 = axes[1]
    b_tr = ax2.bar(x - width / 2, train_times, width, color="#0284c7", edgecolor="#0c4a6e", linewidth=0.9, label="Full Train Fit Time (s)")
    b_in = ax2.bar(x + width / 2, infer_times, width, color="#16a34a", edgecolor="#052e16", linewidth=0.9, label="10-Trip Test Inference Time (s)")

    for bars in (b_tr, b_in):
        for rect in bars:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, h * 1.18, f"{h:.3f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_title("B. Wall-Clock Training & Test Inference Latency (Log Scale)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Wall-Clock Time in Seconds (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(0.005, 120.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True)

    fig.suptitle(
        "PathSnap Models 1–3 — Emission Probability Calibration Loss & Computational Latency",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def plot_confusion_matrix_grid(
    metrics_by_slug: dict[str, dict[str, Any]],
    plots_dir: Path = PLOTS_DIR,
    dpi: int = 170,
) -> Path:
    """Plot a 1x3 side-by-side Confusion Matrix grid for Models 1–3 on student_test."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "confusion_matrix_grid.png"

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), dpi=dpi)
    class_names = ("Non-Match (0)", "Match (1)")
    quadrant_labels = [["TN", "FP"], ["FN", "TP"]]

    for idx, (name, slug, _) in enumerate(MODEL_ORDER):
        ax = axes[idx]
        m = metrics_by_slug[slug]["student_test_metrics"]
        cm = np.asarray(m["confusion_matrix"], dtype=np.int64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{name}\nTop-1: {m['top1_accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f}",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        if idx == 0:
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_ylabel("Ground-Truth Candidate Label", fontsize=9.5, fontweight="bold")
        else:
            ax.set_yticklabels(["", ""])
        ax.set_xlabel("Predicted Candidate Label", fontsize=9.5, fontweight="bold")

        for i in range(2):
            for j in range(2):
                count_val = cm[i, j]
                pct_val = cm_norm[i, j] * 100.0
                q_label = quadrant_labels[i][j]
                text_color = "white" if cm_norm[i, j] > 0.55 else "#111827"
                ax.text(
                    j,
                    i,
                    f"{q_label}\n{count_val:,}\n({pct_val:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9.5,
                    fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("Row-Normalized Proportion", rotation=270, labelpad=14, fontsize=9.5)

    fig.suptitle(
        "PathSnap Models 1–3 — Candidate Classification Confusion Matrices on student_test (153,280 Candidates)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def plot_per_trip_and_feature_importance(
    metrics_by_slug: dict[str, dict[str, Any]],
    per_trip_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    dpi: int = 170,
) -> Path:
    """Plot per-trip Top-1 accuracy across all 10 student_test trips and Random Forest feature importances."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "per_trip_top1_and_rf_importance.png"

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=dpi)

    # Panel A: Per-Trip Top-1 Accuracy across all 10 student_test trips
    ax1 = axes[0]
    trips = sorted(per_trip_df["trip_id"].unique())
    x = np.arange(len(trips))
    width = 0.20

    nearest_vals = [
        float(per_trip_df[(per_trip_df["trip_id"] == t) & (per_trip_df["model_slug"] == "logreg")]["nearest_segment_top1_acc"].iloc[0])
        for t in trips
    ]
    ax1.bar(x - 1.5 * width, nearest_vals, width, color="#94a3b8", edgecolor="#334155", linewidth=0.7, label="Nearest-Arc Baseline")

    for offset_idx, (name, slug, color) in enumerate(MODEL_ORDER):
        vals = [
            float(per_trip_df[(per_trip_df["trip_id"] == t) & (per_trip_df["model_slug"] == slug)]["top1_accuracy"].iloc[0])
            for t in trips
        ]
        pos = x + (-0.5 + offset_idx) * width
        ax1.bar(pos, vals, width, color=color, edgecolor="#0f172a", linewidth=0.7, label=name)

    ax1.set_title("A. Per-Trip Top-1 Accuracy Across All 10 student_test Trips", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(trips, rotation=30, ha="right", fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Observation Top-1 Accuracy", fontsize=10, fontweight="bold")
    ax1.set_xlabel("student_test Trip ID", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0.35, 1.03)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8.2, ncol=2, frameon=True)

    # Panel B: Top 12 Random Forest Gini Feature Importances
    ax2 = axes[1]
    rf_imp: dict[str, float] = metrics_by_slug["rf"]["extra_metadata"]["feature_importances_gini"]
    top_items = list(rf_imp.items())[:12][::-1]
    feat_names = [k for k, _ in top_items]
    feat_vals = [v for _, v in top_items]

    bars = ax2.barh(feat_names, feat_vals, color="#059669", edgecolor="#064e3b", linewidth=0.8)
    for rect in bars:
        w = rect.get_width()
        ax2.text(w + 0.003, rect.get_y() + rect.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=8.3, fontweight="bold")

    ax2.set_title("B. Model 3 (Random Forest) Top-12 Gini Feature Importances", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Normalized Gini Importance", fontsize=10, fontweight="bold")
    ax2.set_xlim(0.0, 0.23)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "PathSnap Models 1–3 — Per-Trip Generalization Breakdown & Ensemble Feature Attribution",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def generate_all_phase5_plots(output_dir: Path = OUTPUT_DIR, plots_dir: Path = PLOTS_DIR) -> list[Path]:
    """Generate all Phase 5 comparison figures from verified experimental logs."""
    _, metrics_by_slug, per_trip_df = _load_artifacts(output_dir=output_dir)
    generated = [
        plot_cv_macro_f1_comparison(metrics_by_slug, plots_dir=plots_dir),
        plot_student_test_top1_and_f1(metrics_by_slug, plots_dir=plots_dir),
        plot_logloss_and_latency(metrics_by_slug, plots_dir=plots_dir),
        plot_confusion_matrix_grid(metrics_by_slug, plots_dir=plots_dir),
        plot_per_trip_and_feature_importance(metrics_by_slug, per_trip_df, plots_dir=plots_dir),
    ]
    for p in generated:
        print(f"[Phase 5 Plots] Generated -> {p}")
    return generated


if __name__ == "__main__":
    generate_all_phase5_plots()
