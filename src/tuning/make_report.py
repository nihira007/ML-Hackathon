"""Render plots and draft Sections 3.2 and 4 from measured results."""
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

RATIONALE = {
    "lgbm": ("Leaf-wise boosting captures interactions between cues such as distance and heading. It is efficient on tabular inputs, and unconstrained leaves can overfit.", "Binary log loss"),
    "xgb": ("Regularised boosting combines several weak cues. Depth and shrinkage control complexity, and large ensembles may increase latency.", "Binary logistic loss"),
    "mlp": ("A feed-forward network can learn smooth nonlinear combinations of geometry and movement. It is sensitive to scaling and training budget.", "Binary cross-entropy with an L2 penalty"),
}
METRICS = {
    "cv_macro_f1_mean": "CV Macro F1",
    "student_test_top1_accuracy": "Student-test Top-1",
    "student_test_macro_f1": "Student-test Macro F1",
    "student_test_log_loss": "Student-test log loss",
    "inference_time_sec": "Inference seconds",
}


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


def comparisons(table, labels, fields, title, path):
    fig, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 4.3), squeeze=False)
    for ax, field in zip(axes.flat, fields, strict=True):
        ax.bar(labels, table[field], color="#00796b",
               yerr=table.cv_macro_f1_std if field == "cv_macro_f1_mean" else None)
        ax.set_title(METRICS[field])
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(title)
    save(fig, path)


def loss_curve(models, title, path):
    history = pd.read_csv(models / "mlp_history.csv").groupby("epoch").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history.index, history.train_loss, label="Training trips")
    ax.plot(history.index, history.validation_loss, label="Validation trips")
    ax.set(title=f"MLP loss across five folds · {title}", xlabel="Epoch", ylabel="Candidate log loss")
    ax.legend()
    save(fig, path)


def make_report(models, tuning):
    baselines = [read_json(models / f"{slug}_metrics.json") for slug in NAMES]
    summary = read_json(tuning / "tuning_summary.json")
    if any(r["data_signature"] != summary["data_signature"] or r["smoke"] != summary["smoke"] for r in baselines):
        raise ValueError("Report inputs come from different runs")
    title = "Synthetic verification" if summary["smoke"] else "PathSnap experiments"
    plots = tuning / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    lines = [f"# Person 3 · {title}", ""]
    if summary["smoke"]:
        lines += ["These results measure generated candidates and verify the software workflow.", ""]
    lines += ["## 3.2 Advanced classifiers", ""]
    for record in baselines:
        slug, name = record["model"], record["model_name"]
        fit, objective = RATIONALE[slug]
        student = record["student_test"]
        train_f1, cv_f1 = record["train_macro_f1_mean"], record["cv_macro_f1_mean"]
        gap = train_f1 - cv_f1
        if gap > 0.03:
            assessment = "The gap suggests some overfitting to training trips."
        elif train_f1 < 0.7 and cv_f1 < 0.7:
            assessment = "Both scores are low. Limited model capacity and difficult features are possible causes."
        else:
            assessment = "The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion."
        lines += [
            f"### {name}", "",
            f"**A. Model understanding.** {fit}", "",
            f"**B. Training and configuration.** {objective}. Initial settings {json.dumps(record['initial_params'])}."
            + (f" The network made {record['epochs']} passes per fold." if slug == "mlp" else ""), "",
            f"**C. Results.** Student-test Top-1 {student['top1_accuracy']:.2%}, Macro F1 {student['macro_f1']:.4f}, "
            f"log loss {student['log_loss']:.4f}, precision {student['precision']:.4f}, recall {student['recall']:.4f}. "
            f"Mean training-fold Macro F1 {train_f1:.4f} and validation Macro F1 "
            f"{cv_f1:.4f} ± {record['cv_macro_f1_std']:.4f} give a gap of {gap:+.4f}. {assessment}", "",
        ]
        confusion(record, f"{name} confusion matrix · {title}", plots / f"{slug}_confusion_matrix.png")
        importance(models / f"{slug}_feature_importance.csv", f"{name} · {title}",
                   plots / f"{slug}_feature_importance.png", "Log-loss increase after shuffling")

    loss_curve(models, title, plots / "mlp_loss_curve.png")

    selection = pd.read_csv(tuning / "model_selection.csv")
    comparisons(selection, selection.model_name, list(METRICS)[:4], title, plots / "baseline_comparison.png")
    trials = pd.read_csv(tuning / "optuna_trials.csv")
    complete = trials[trials.state.eq("COMPLETE")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(complete.number, complete.value, label="Trial")
    ax.step(complete.number, complete.value.cummin(), where="post", label="Best so far")
    ax.set(title=f"Optuna history · {title}", xlabel="Trial", ylabel="Mean CV log loss")
    ax.legend()
    save(fig, plots / "optuna_history.png")
    importance(tuning / "optuna_parameter_importance.csv", f"Optuna parameter importance · {title}",
               plots / "optuna_parameter_importance.png", "Relative importance")

    comparison = pd.read_csv(tuning / "untuned_vs_tuned.csv")
    comparisons(comparison, comparison.version, METRICS, title, plots / "untuned_vs_tuned.png")
    confusion(read_json(tuning / "tuned_metrics.json"), f"Tuned classifier · {title}", plots / "tuned_confusion_matrix.png")
    importance(tuning / "tuned_feature_importance.csv", f"Tuned classifier · {title}",
               plots / "tuned_feature_importance.png", "Log-loss increase after shuffling")

    lines += [
        "## 4. Bayesian hyperparameter optimisation", "",
        f"**Model selection.** {summary['selected_model_name']} led the ranking of {summary['baseline_models_compared']} "
        "baselines. The rank weights were 40% CV Macro F1, 30% student-test Top-1, 20% student-test log loss and 10% inference time.", "",
        f"**Objective.** {summary['objective']}. Log loss penalises overconfident errors in probabilities used as HMM emission scores.", "",
        f"**Trials and compute cost.** Seeded TPE completed {summary['completed_trials']} trials. "
        f"Trial {summary['best_trial']} achieved the lowest mean CV log loss ({summary['best_cv_log_loss']:.5f}). "
        f"Search took {summary['tuning_wall_seconds']:.2f} wall-clock seconds.", "",
        f"**Best estimator parameters.** {json.dumps(summary['best_params'])}.", "",
        "| Parameter | Distribution | Settings |", "| --- | --- | --- |",
    ]
    for name, dist in summary["search_space"].items():
        lines.append(f"| {name} | {dist['name']} | {json.dumps(dist['attributes'])} |")
    lines += ["", "The search varies capacity, learning, regularisation and sampling where applicable.", "",
              "| Metric | Untuned | Tuned | Change |", "| --- | ---: | ---: | ---: |"]
    for field, label in METRICS.items():
        before, after = comparison[field]
        lines.append(f"| {label} | {before:.5f} | {after:.5f} | {after - before:+.5f} |")
    before, after = comparison.student_test_log_loss
    direction = "improved" if after < before else "worsened" if after > before else "was unchanged"
    lines += ["", f"Student-test log loss {direction}. Interpret that result alongside the changes in ranking quality "
              "and latency. The tuning CV scores were used to choose the hyperparameters.", ""]
    (tuning / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("outputs/models/advanced"))
    parser.add_argument("--tuning", type=Path, default=Path("outputs/tuning"))
    parser.add_argument("--baseline-plots", action="store_true")
    args = parser.parse_args()
    if args.baseline_plots:
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
        return
    make_report(args.models, args.tuning)


if __name__ == "__main__":
    main()
