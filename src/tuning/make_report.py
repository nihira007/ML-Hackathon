"""Plot baseline diagnostics from measured model outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.models.advanced.common import NAMES

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def importance(csv, title, path, xlabel):
    table = pd.read_csv(csv).dropna(subset=["importance"]).nlargest(15, "importance").iloc[::-1]
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.barh(table.iloc[:, 0], table.importance, color="#00796b")
    ax.set(title=title, xlabel=xlabel)
    save(fig, path)


def confusion(record, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(record["student_test"]["confusion_matrix"])
    ConfusionMatrixDisplay(matrix, display_labels=["Other arc", "Match"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    save(fig, path)


def loss_curve(models, title, path):
    history = pd.read_csv(models / "mlp_history.csv").groupby("epoch").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history.index, history.train_loss, label="Training trips")
    ax.plot(history.index, history.validation_loss, label="Validation trips")
    ax.set(title=f"MLP loss across five folds · {title}", xlabel="Epoch", ylabel="Candidate log loss")
    ax.legend()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("outputs/models/advanced"))
    args = parser.parse_args()
    for slug in NAMES:
        record_path = args.models / f"{slug}_metrics.json"
        if not record_path.exists():
            continue
        record = read_json(record_path)
        title = f"{record['model_name']} · {record['feature_version']}"
        if record["smoke"]:
            title += " · synthetic"
        confusion(record, title, args.models / f"{slug}_confusion_matrix.png")
        importance(args.models / f"{slug}_feature_importance.csv", title,
                   args.models / f"{slug}_feature_importance.png", "Log-loss increase after shuffling")
        if slug == "mlp":
            loss_curve(args.models, title, args.models / "mlp_loss_curve.png")


if __name__ == "__main__":
    main()
