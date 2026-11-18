"""PathSnap - Plotting and Visualization Generator for Report Sections 5 & 6.

Generates:
1. Classifier-Only vs. Classifier + HMM Route Continuity comparison bar chart.
2. Top-1 Pointwise Accuracy comparison bar chart.
3. Sequence-level F1-score comparison bar chart.
4. Latency / throughput comparison chart.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_benchmark_plots(
    per_trip_csv: str | Path = "outputs/hmm/classifier_vs_hmm_per_trip.csv",
    output_dir: str | Path = "outputs/hmm/plots",
) -> None:
    """Generate high-resolution PNG comparison charts from per-trip benchmark CSV."""
    csv_path = Path(per_trip_csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trips = df["trip_id"].astype(str).tolist()
    x = np.arange(len(trips))
    width = 0.35

    # 1. Route Continuity Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width / 2, df["cls_continuity"] * 100, width, label="Classifier-Only (Argmax)", color="#E74C3C", alpha=0.85)
    rects2 = ax.bar(x + width / 2, df["hmm_continuity"] * 100, width, label="Classifier + HMM / Viterbi", color="#2980B9", alpha=0.90)

    ax.set_ylabel("Route Continuity (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trip ID (Student Test Set)", fontsize=12, fontweight="bold")
    ax.set_title("PathSnap: Route Continuity Improvement Across All 10 Test Trips", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trips, rotation=35, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=11)

    # Annotate mean gain
    mean_gain = (df["hmm_continuity"].mean() - df["cls_continuity"].mean()) * 100
    ax.text(
        0.02, 0.92,
        f"Mean Continuity Gain: +{mean_gain:.1f}%\nHMM Avg: {df['hmm_continuity'].mean()*100:.1f}%\nClassifier Avg: {df['cls_continuity'].mean()*100:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7"),
    )

    plt.tight_layout()
    chart1_path = out_dir / "classifier_vs_hmm_continuity.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"[OK] Saved {chart1_path}")

    # 2. Sequence F1 Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width / 2, df["cls_seq_f1"] * 100, width, label="Classifier-Only (Argmax)", color="#F39C12", alpha=0.85)
    rects2 = ax.bar(x + width / 2, df["hmm_seq_f1"] * 100, width, label="Classifier + HMM / Viterbi", color="#27AE60", alpha=0.90)

    ax.set_ylabel("Sequence F1 Score (%) vs .route", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trip ID (Student Test Set)", fontsize=12, fontweight="bold")
    ax.set_title("PathSnap: Sequence-Level F1 Traversed Route Reconstruction", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trips, rotation=35, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=11)

    plt.tight_layout()
    chart2_path = out_dir / "classifier_vs_hmm_seq_f1.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"[OK] Saved {chart2_path}")

    # 3. Pointwise Top-1 Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width / 2, df["cls_top1"] * 100, width, label="Classifier-Only (Argmax)", color="#95A5A6", alpha=0.85)
    rects2 = ax.bar(x + width / 2, df["hmm_top1"] * 100, width, label="Classifier + HMM / Viterbi", color="#8E44AD", alpha=0.90)

    ax.set_ylabel("Pointwise Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trip ID (Student Test Set)", fontsize=12, fontweight="bold")
    ax.set_title("PathSnap: Observation-Level Top-1 Accuracy Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trips, rotation=35, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=11)

    plt.tight_layout()
    chart3_path = out_dir / "classifier_vs_hmm_top1.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"[OK] Saved {chart3_path}")


if __name__ == "__main__":
    generate_benchmark_plots()
