"""End-to-End Verification & Master 6-Model + HMM Consolidation Script (`prompts/05_POST_MERGE_INTEGRATION.md`).

1. Runs Person 4's HMM/Viterbi benchmark on the final tuned classifier probabilities
   (`outputs/tuning/tuned_student_test_preds.parquet`) and computes Route Continuity for all 6 baseline
   models (`Logistic Regression`, `Distance-Weighted KNN`, `Random Forest`, `LightGBM`, `XGBoost`, `MLP`)
   plus `XGBoost (Tuned)` and `XGBoost (Tuned) + HMM/Viterbi`.
2. Saves `outputs/hmm/baseline_continuity.csv`, `outputs/hmm/classifier_vs_hmm_per_trip.csv`,
   `outputs/hmm/classifier_vs_hmm_summary.json`, and regenerates `outputs/hmm/plots/*`.
3. Builds the consolidated Report Section 3.1 master table `outputs/master_model_comparison.csv`
   and `outputs/master_plots/master_6model_comparison.png`.
4. Verifies `predict_trip()` and `evaluate_trip_server_side()` on `Student folder/student_test/`
   with `.route` strictly isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.road_graph import RoadGraph
from src.hmm.evaluate_hmm import load_trip_route, load_trip_track, run_benchmark_on_parquet
from src.hmm.plot_results import generate_benchmark_plots
from src.hmm.route_metrics import compute_observation_accuracy, compute_route_continuity
from src.hmm.viterbi import HMMConfig
from src.pipeline.inference import evaluate_trip_server_side, predict_trip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "Student folder" / "student_test"


def compute_fast_continuity_for_parquet(
    parquet_path: Path,
    road_graphs: dict[str, RoadGraph],
) -> tuple[float, float]:
    """Compute mean Classifier-Only Top-1 Accuracy and Route Continuity across all 10 student_test trips."""
    df = pd.read_parquet(parquet_path)
    conts = []
    top1s = []
    for trip_id, g in df.groupby("trip_id", sort=True):
        rg = road_graphs[str(trip_id)]
        sorted_g = g.sort_values(["obs_idx", "pred_prob"], ascending=[True, False], kind="mergesort")
        best = sorted_g.drop_duplicates(subset=["obs_idx"], keep="first").sort_values("obs_idx")
        pred_arcs = [int(a) for a in best["arc_id"].to_numpy()]
        conts.append(compute_route_continuity(pred_arcs, rg, max_step_dist_m=100.0))
        top1s.append(float((best["is_match"].to_numpy() == 1).mean()))
    return float(np.mean(conts)), float(np.mean(top1s))


def run_master_consolidation() -> None:
    print("1. Pre-loading road graphs for 10 student_test trips...")
    road_graphs: dict[str, RoadGraph] = {}
    for t_dir in sorted(TEST_DIR.iterdir()):
        if t_dir.is_dir():
            road_graphs[t_dir.name] = RoadGraph.from_trip_dir(str(t_dir), trip_id=t_dir.name)

    # 2. Run full HMM + Viterbi benchmark on the final Tuned XGBoost predictions
    hmm_cfg_path = PROJECT_ROOT / "artifacts" / "hmm_config.json"
    hmm_cfg_dict = json.loads(hmm_cfg_path.read_text(encoding="utf-8"))
    valid_cfg = {k: v for k, v in hmm_cfg_dict.items() if k in HMMConfig.__dataclass_fields__}
    hmm_config = HMMConfig(**valid_cfg)

    tuned_parquet = PROJECT_ROOT / "outputs" / "tuning" / "tuned_student_test_preds.parquet"
    hmm_out_dir = PROJECT_ROOT / "outputs" / "hmm"
    hmm_res = run_benchmark_on_parquet(
        parquet_path=tuned_parquet,
        student_test_dir=TEST_DIR,
        hmm_config=hmm_config,
        output_csv_path=hmm_out_dir / "classifier_vs_hmm_per_trip.csv",
        output_summary_json=hmm_out_dir / "classifier_vs_hmm_summary.json",
    )
    generate_benchmark_plots(
        per_trip_csv=hmm_out_dir / "classifier_vs_hmm_per_trip.csv",
        output_dir=hmm_out_dir / "plots",
    )
    hmm_summary = hmm_res["summary"]

    # 3. Compute Route Continuity across all 6 models + Tuned model
    classical_df = pd.read_csv(PROJECT_ROOT / "outputs" / "models" / "classical" / "classical_comparison.csv")
    advanced_df = pd.read_csv(PROJECT_ROOT / "outputs" / "models" / "advanced" / "advanced_comparison.csv")
    untuned_vs_tuned = pd.read_csv(PROJECT_ROOT / "outputs" / "tuning" / "untuned_vs_tuned.csv")
    tuned_row = untuned_vs_tuned[untuned_vs_tuned["version"] == "tuned"].iloc[0]

    model_Editable = [
        (
            "Logistic Regression",
            "Linear / Probabilistic",
            "C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs'",
            PROJECT_ROOT / "outputs" / "models" / "classical" / "logreg_student_test_preds.parquet",
            classical_df[classical_df["model_name"] == "Logistic Regression"].iloc[0],
        ),
        (
            "Distance-Weighted KNN",
            "Instance-based / Non-parametric",
            "n_neighbors=15, weights='distance', metric='euclidean'",
            PROJECT_ROOT / "outputs" / "models" / "classical" / "knn_student_test_preds.parquet",
            classical_df[classical_df["model_name"] == "Distance-Weighted KNN"].iloc[0],
        ),
        (
            "Random Forest",
            "Bagging / Tree Ensemble",
            "n_estimators=100, max_depth=14, min_samples_leaf=25, class_weight='balanced_subsample'",
            PROJECT_ROOT / "outputs" / "models" / "classical" / "rf_student_test_preds.parquet",
            classical_df[classical_df["model_name"] == "Random Forest"].iloc[0],
        ),
        (
            "LightGBM",
            "Gradient Boosted Decision Trees (Leaf-wise)",
            "n_estimators=160, learning_rate=0.06, num_leaves=31, min_child_samples=30",
            PROJECT_ROOT / "outputs" / "models" / "advanced" / "lgbm_student_test_preds.parquet",
            advanced_df[advanced_df["model_name"] == "LightGBM"].iloc[0],
        ),
        (
            "XGBoost",
            "Gradient Boosted Decision Trees (Depth-wise)",
            "n_estimators=160, learning_rate=0.06, max_depth=5, tree_method='hist'",
            PROJECT_ROOT / "outputs" / "models" / "advanced" / "xgb_student_test_preds.parquet",
            advanced_df[advanced_df["model_name"] == "XGBoost"].iloc[0],
        ),
        (
            "MLP Neural Network",
            "Feed-Forward Multi-Layer Perceptron",
            "hidden_layer_sizes=(64, 32), activation='relu', solver='adam', epochs=20",
            PROJECT_ROOT / "outputs" / "models" / "advanced" / "mlp_student_test_preds.parquet",
            advanced_df[advanced_df["model_name"] == "MLP"].iloc[0],
        ),
        (
            "XGBoost (Optuna Tuned)",
            "Bayesian-Optimized GBDT (35 TPE Trials)",
            "TPE-tuned XGBoost (artifacts/final_model.joblib)",
            tuned_parquet,
            tuned_row,
        ),
    ]

    master_rows = []
    continuity_rows = []
    for m_name, family, config_str, pq_path, row in model_Editable:
        cls_cont, _ = compute_fast_continuity_for_parquet(pq_path, road_graphs)
        master_rows.append(
            {
                "Model": m_name,
                "Algorithm Family": family,
                "Key Configuration": config_str,
                "CV Macro F1-score": round(float(row["cv_macro_f1_mean"]), 4),
                "CV Macro F1 Std": round(float(row["cv_macro_f1_std"]), 4),
                "Top-1 Accuracy": round(float(row["student_test_top1_accuracy"]), 4),
                "Route Continuity": round(float(cls_cont), 4),
                "Student-Test Macro F1-score": round(float(row["student_test_macro_f1"]), 4),
                "Log Loss": round(float(row["student_test_log_loss"]), 4),
                "Inference Time (s)": round(float(row["inference_time_sec"]), 4),
            }
        )
        continuity_rows.append(
            {
                "model_name": m_name,
                "algorithm_family": family,
                "mean_classifier_continuity": round(float(cls_cont), 4),
                "mean_hmm_continuity": round(float(hmm_summary["mean_hmm_continuity"]), 4),
                "continuity_gain": round(float(hmm_summary["mean_hmm_continuity"] - cls_cont), 4),
                "student_test_top1": round(float(row["student_test_top1_accuracy"]), 4),
            }
        )

    # Add final row for Tuned Classifier + HMM/Viterbi
    master_rows.append(
        {
            "Model": "XGBoost (Tuned) + HMM/Viterbi",
            "Algorithm Family": "Probabilistic Sequence Decoder + GBDT",
            "Key Configuration": f"Tuned XGBoost emissions + Viterbi (beta={hmm_config.beta}, max_dist={hmm_config.max_route_dist_m}m)",
            "CV Macro F1-score": round(float(tuned_row["cv_macro_f1_mean"]), 4),
            "CV Macro F1 Std": round(float(tuned_row["cv_macro_f1_std"]), 4),
            "Top-1 Accuracy": round(float(hmm_summary["mean_hmm_top1"]), 4),
            "Route Continuity": round(float(hmm_summary["mean_hmm_continuity"]), 4),
            "Student-Test Macro F1-score": round(float(tuned_row["student_test_macro_f1"]), 4),
            "Log Loss": round(float(tuned_row["student_test_log_loss"]), 4),
            "Inference Time (s)": round(
                float(tuned_row["inference_time_sec"]) + float(hmm_summary["total_hmm_runtime_sec"]), 4
            ),
        }
    )

    pd.DataFrame(continuity_rows).to_csv(hmm_out_dir / "baseline_continuity.csv", index=False)
    master_df = pd.DataFrame(master_rows)
    master_csv = PROJECT_ROOT / "outputs" / "master_model_comparison.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\nSaved {master_csv}:")
    print(master_df.to_string(index=False))

    # 4. Plot Master 6-Model + HMM Comparison Figure
    master_plots_dir = PROJECT_ROOT / "outputs" / "master_plots"
    master_plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=170)

    labels = master_df["Model"].tolist()
    x = np.arange(len(labels))
    w = 0.28

    ax1 = axes[0]
    ax1.bar(x - w, master_df["Top-1 Accuracy"], w, label="Student-Test Top-1 Accuracy", color="#2563eb")
    ax1.bar(x, master_df["Student-Test Macro F1-score"], w, label="Student-Test Macro F1", color="#059669")
    ax1.bar(x + w, master_df["Route Continuity"], w, label="Route Continuity", color="#d97706")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0.65, 1.05)
    ax1.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax1.set_title("A. Accuracy, Macro F1 & Route Continuity Across All 6 Models + HMM", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2 = axes[1]
    ax2.bar(x, master_df["Log Loss"], width=0.45, color="#7c3aed", label="Student-Test Binary Log Loss (↓ better)")
    for idx, val in enumerate(master_df["Log Loss"]):
        ax2.text(idx, val + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0.0, 0.30)
    ax2.set_ylabel("Log Loss", fontsize=10, fontweight="bold")
    ax2.set_title("B. Candidate Probability Calibration (Log Loss ↓)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("PathSnap Master Benchmark — Classical, Boosting, Neural Network & HMM/Viterbi", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(master_plots_dir / "master_6model_comparison.png", bbox_inches="tight", dpi=170)
    plt.close(fig)

    # 5. Verify end-to-end single-trip inference (`predict_trip` + `evaluate_trip_server_side`) on sample trip 00000052
    sample_trip_dir = TEST_DIR / "00000052"
    pred_out = predict_trip(sample_trip_dir, artifacts_dir=PROJECT_ROOT / "artifacts")
    eval_out = evaluate_trip_server_side(pred_out, sample_trip_dir / "00000052.route", trip_dir=sample_trip_dir)
    print("\nVerified End-to-End Single-Trip Pipeline on trip 00000052 (.route isolated during inference):")
    print(json.dumps(eval_out, indent=2))


if __name__ == "__main__":
    run_master_consolidation()
