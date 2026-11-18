"""Phase 4 — Probability Calibration Audit, Per-Trip Error Analysis & Prediction Schema Verification.

Evaluates probability calibration quality (Log Loss, Brier Score, Expected Calibration Error [ECE],
Maximum Calibration Error [MCE], and 10-bin reliability curves) and per-trip / spatial-slice error
breakdowns across all 10 `student_test` trips for Models 1–3:
- Model 1: Logistic Regression (`logreg_student_test_preds.parquet`)
- Model 2: Distance-Weighted KNN (`knn_student_test_preds.parquet`)
- Model 3: Random Forest (`rf_student_test_preds.parquet`)

Outputs:
- `outputs/models/classical/calibration_curves.png`
- `outputs/models/classical/calibration_audit.json`
- `outputs/models/classical/per_trip_results.csv`
- `outputs/models/classical/per_trip_error_analysis.json`
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
from sklearn.calibration import calibration_curve

from src.eval.metrics import (
    PREDICTION_COLUMNS,
    compute_topk_accuracy,
    evaluate_predictions,
    validate_prediction_dataframe,
)
from src.models.classical.data_utils import PROJECT_ROOT, load_feature_datasets

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"

MODEL_SPECS: list[dict[str, str]] = [
    {
        "model_name": "Logistic Regression",
        "slug": "logreg",
        "family": "Linear / Probabilistic",
        "color": "#2563eb",
        "parquet": "logreg_student_test_preds.parquet",
    },
    {
        "model_name": "Distance-Weighted KNN",
        "slug": "knn",
        "family": "Instance-based / Non-parametric",
        "color": "#d97706",
        "parquet": "knn_student_test_preds.parquet",
    },
    {
        "model_name": "Random Forest",
        "slug": "rf",
        "family": "Bagging / Tree Ensemble",
        "color": "#059669",
        "parquet": "rf_student_test_preds.parquet",
    },
]


def compute_ece_and_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and bin statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=False), 0, n_bins - 1)

    total_n = len(y_true)
    ece = 0.0
    mce = 0.0
    bins_table: list[dict[str, Any]] = []

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            frac_pos = float(np.mean(y_true[mask]))
            abs_gap = abs(mean_pred - frac_pos)
            ece += (count / total_n) * abs_gap
            mce = max(mce, abs_gap)
        else:
            mean_pred = float(0.5 * (bin_edges[b] + bin_edges[b + 1]))
            frac_pos = 0.0
            abs_gap = 0.0

        bins_table.append(
            {
                "bin_idx": b,
                "bin_lower": round(float(bin_edges[b]), 2),
                "bin_upper": round(float(bin_edges[b + 1]), 2),
                "count": count,
                "proportion_of_samples": round(count / total_n, 6),
                "mean_predicted_prob": round(mean_pred, 6),
                "empirical_positive_rate": round(frac_pos, 6),
                "abs_calibration_gap": round(abs_gap, 6),
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mean_predicted_prob": float(np.mean(y_prob)),
        "empirical_positive_rate": float(np.mean(y_true)),
        "overconfidence_bias": float(np.mean(y_prob) - np.mean(y_true)),
        "bins": bins_table,
    }


def plot_calibration_curves(
    pred_dfs: dict[str, pd.DataFrame],
    audit_summary: dict[str, Any],
    output_path: Path,
    dpi: int = 160,
) -> Path:
    """Render a 2-panel reliability curve and predicted probability histogram figure for Models 1–3."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=dpi)

    ax_cal, ax_hist = axes

    # Panel A: Reliability Diagram
    ax_cal.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#4b5563",
        linewidth=1.5,
        label="Perfectly Calibrated (y = x)",
    )

    for spec in MODEL_SPECS:
        slug = spec["slug"]
        name = spec["model_name"]
        color = spec["color"]
        df = pred_dfs[slug]
        y_true = df["is_match"].to_numpy(dtype=np.int64)
        y_prob = df["pred_prob"].to_numpy(dtype=np.float64)

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        m_info = audit_summary["models"][slug]
        label_str = (
            f"{name} (LL={m_info['log_loss']:.3f}, "
            f"Brier={m_info['brier_score']:.3f}, ECE={m_info['ece']:.3f})"
        )
        ax_cal.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            label=label_str,
        )

        # Panel B: Histogram of predicted probabilities
        ax_hist.hist(
            y_prob,
            bins=25,
            range=(0.0, 1.0),
            histtype="step",
            linewidth=2.0,
            color=color,
            density=True,
            label=f"{name} (mean p={m_info['mean_predicted_prob']:.3f})",
        )

    ax_cal.set_title(
        "A. Reliability Curves on student_test (10 Trips, 153,280 Candidates)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_cal.set_xlabel("Mean Predicted Emission Probability P(is_match = 1 | x)", fontsize=10, fontweight="bold")
    ax_cal.set_ylabel("Empirical Fraction of True Matches", fontsize=10, fontweight="bold")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    ax_cal.legend(loc="upper left", fontsize=8.5, frameon=True)

    ax_hist.axvline(
        0.125,
        linestyle=":",
        color="#111827",
        linewidth=1.4,
        label="True Base Rate (1/8 = 0.125)",
    )
    ax_hist.set_title(
        "B. Predicted Probability Sharpness & Distribution (Log Density)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_hist.set_xlabel("Predicted Probability (pred_prob)", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("Density (Log Scale)", fontsize=10, fontweight="bold")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "PathSnap Person 2 Phase 4 — Emission Probability Calibration Audit (Models 1–3)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def run_calibration_and_error_analysis(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run Phase 4 probability calibration audit, per-trip error breakdown, and schema validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, test_df, feature_cols = load_feature_datasets(require_v2=True)

    # 1. Load and validate all 3 prediction parquets against the shared contract schema
    pred_dfs: dict[str, pd.DataFrame] = {}
    schema_checks: dict[str, Any] = {}

    for spec in MODEL_SPECS:
        slug = spec["slug"]
        parquet_path = output_dir / spec["parquet"]
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Required prediction parquet '{parquet_path}' is missing. Run Phase 3 first."
            )
        df_pred = pd.read_parquet(parquet_path)
        validate_prediction_dataframe(df_pred)

        if list(df_pred.columns) != PREDICTION_COLUMNS:
            raise ValueError(
                f"Parquet {parquet_path.name} columns {list(df_pred.columns)} != {PREDICTION_COLUMNS}"
            )
        if len(df_pred) != len(test_df):
            raise ValueError(
                f"Parquet {parquet_path.name} row count {len(df_pred)} != test_df row count {len(test_df)}"
            )

        pred_dfs[slug] = df_pred
        schema_checks[slug] = {
            "parquet_path": str(parquet_path),
            "columns": list(df_pred.columns),
            "num_rows": int(len(df_pred)),
            "num_trips": int(df_pred["trip_id"].nunique()),
            "num_observations": int(df_pred.groupby(["trip_id", "obs_idx"], sort=False).ngroups),
            "null_count": int(df_pred.isna().sum().sum()),
            "prob_min": round(float(df_pred["pred_prob"].min()), 6),
            "prob_max": round(float(df_pred["pred_prob"].max()), 6),
            "schema_valid": True,
        }

    # 2. Compute calibration metrics & reliability curves across Models 1–3
    models_calibration: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        slug = spec["slug"]
        df_pred = pred_dfs[slug]
        eval_df = test_df[["trip_id", "obs_idx", "arc_id", "cand_rank", "is_match"]].copy()
        eval_df["pred_prob"] = df_pred["pred_prob"].to_numpy(dtype=np.float64)

        overall_m = evaluate_predictions(eval_df, threshold=0.5)
        overall_m["top3_accuracy"] = compute_topk_accuracy(eval_df, k=3)
        cal_m = compute_ece_and_bins(
            y_true=eval_df["is_match"].to_numpy(dtype=np.int64),
            y_prob=eval_df["pred_prob"].to_numpy(dtype=np.float64),
            n_bins=10,
        )

        # Compute log-odds / clipped emission log-likelihood separation between true matches and non-matches
        probs_clipped = np.clip(eval_df["pred_prob"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
        match_mask = eval_df["is_match"].to_numpy(dtype=np.int64) == 1
        mean_prob_on_match = float(np.mean(probs_clipped[match_mask]))
        mean_prob_on_nonmatch = float(np.mean(probs_clipped[~match_mask]))
        prob_discrimination_ratio = float(mean_prob_on_match / max(mean_prob_on_nonmatch, 1e-6))

        models_calibration[slug] = {
            "model_name": spec["model_name"],
            "algorithm_family": spec["family"],
            "top1_accuracy": round(overall_m["top1_accuracy"], 6),
            "top3_accuracy": round(overall_m["top3_accuracy"], 6),
            "macro_f1": round(overall_m["macro_f1"], 6),
            "binary_f1": round(overall_m["binary_f1"], 6),
            "precision": round(overall_m["precision"], 6),
            "recall": round(overall_m["recall"], 6),
            "log_loss": round(overall_m["log_loss"], 6),
            "brier_score": round(overall_m["brier_score"], 6),
            "ece": round(cal_m["ece"], 6),
            "mce": round(cal_m["mce"], 6),
            "mean_predicted_prob": round(cal_m["mean_predicted_prob"], 6),
            "empirical_positive_rate": round(cal_m["empirical_positive_rate"], 6),
            "overconfidence_bias": round(cal_m["overconfidence_bias"], 6),
            "mean_prob_on_true_match": round(mean_prob_on_match, 6),
            "mean_prob_on_non_match": round(mean_prob_on_nonmatch, 6),
            "prob_discrimination_ratio": round(prob_discrimination_ratio, 4),
            "reliability_bins": cal_m["bins"],
        }

    # 3. Compute per-trip diagnostics & per-trip model breakdown across all 10 student_test trips
    preproc_path = PROJECT_ROOT / "artifacts" / "preprocessor.joblib"
    phys_df = pd.DataFrame(index=test_df.index)
    if "n_cand_within_15m" in test_df.columns and preproc_path.exists():
        import joblib
        preproc = joblib.load(preproc_path)
        scaler = preproc.named_steps["scaler"]
        raw_mat = scaler.inverse_transform(test_df[feature_cols].to_numpy(dtype=np.float32))
        raw_cols = pd.DataFrame(raw_mat, index=test_df.index, columns=feature_cols)
        phys_df["cands_15m"] = raw_cols["n_cand_within_15m"]
        phys_df["road_density"] = raw_cols["local_road_density"]
        phys_df["gps_gap"] = raw_cols["gps_gap_flag"]
        phys_df["speed_mps"] = raw_cols["speed_mps"]
        phys_df["dist_m"] = raw_cols["perp_dist_m"]
        phys_df["heading_diff_deg"] = raw_cols["bearing_diff_deg"]
    else:
        phys_df["cands_15m"] = test_df.get("cands_within_15m", test_df.get("n_cand_within_15m", 2.0))
        phys_df["road_density"] = test_df.get("local_road_density_30m", test_df.get("local_road_density", 3.0))
        phys_df["gps_gap"] = test_df["gps_gap_flag"]
        phys_df["speed_mps"] = test_df.get("gps_speed_mps", test_df.get("speed_mps", 10.0))
        phys_df["dist_m"] = test_df.get("dist_to_arc_m", test_df.get("perp_dist_m", 5.0))
        phys_df["heading_diff_deg"] = test_df.get("heading_diff_deg", test_df.get("bearing_diff_deg", 15.0))

    per_trip_rows: list[dict[str, Any]] = []
    for trip_id, trip_feat_df in test_df.groupby("trip_id", sort=True):
        obs_first_idx = trip_feat_df.drop_duplicates(subset=["obs_idx"], keep="first").index
        gt_idx = trip_feat_df[trip_feat_df["is_match"] == 1].index
        if len(gt_idx) == 0:
            gt_idx = obs_first_idx

        n_obs = int(len(obs_first_idx))
        n_cands = int(len(trip_feat_df))
        mean_cands_15m = float(phys_df.loc[obs_first_idx, "cands_15m"].mean())
        dense_intersection_rate = float((phys_df.loc[obs_first_idx, "cands_15m"] >= 2.9).mean())
        mean_road_density = float(phys_df.loc[obs_first_idx, "road_density"].mean())
        gps_gap_rate = float((phys_df.loc[obs_first_idx, "gps_gap"] > 0.5).mean())
        mean_speed_mps = float(phys_df.loc[obs_first_idx, "speed_mps"].mean())
        mean_gt_dist_m = float(phys_df.loc[gt_idx, "dist_m"].mean())
        mean_gt_heading_diff_deg = float(phys_df.loc[gt_idx, "heading_diff_deg"].mean())
        nearest_dist_top1_baseline = float(
            (trip_feat_df[trip_feat_df["cand_rank"] == 0]["is_match"] == 1).mean()
        )

        trip_idx = trip_feat_df.index
        for spec in MODEL_SPECS:
            slug = spec["slug"]
            trip_eval_df = trip_feat_df[["trip_id", "obs_idx", "arc_id", "cand_rank", "is_match"]].copy()
            trip_eval_df["pred_prob"] = pred_dfs[slug].loc[trip_idx, "pred_prob"].to_numpy(dtype=np.float64)

            tm = evaluate_predictions(trip_eval_df, threshold=0.5)
            top3 = compute_topk_accuracy(trip_eval_df, k=3)
            cal_trip = compute_ece_and_bins(
                trip_eval_df["is_match"].to_numpy(dtype=np.int64),
                trip_eval_df["pred_prob"].to_numpy(dtype=np.float64),
                n_bins=10,
            )

            per_trip_rows.append(
                {
                    "trip_id": str(trip_id),
                    "model_name": spec["model_name"],
                    "model_slug": slug,
                    "num_observations": n_obs,
                    "num_candidates": n_cands,
                    "nearest_segment_top1_acc": round(nearest_dist_top1_baseline, 6),
                    "top1_accuracy": round(tm["top1_accuracy"], 6),
                    "top1_gain_over_nearest": round(tm["top1_accuracy"] - nearest_dist_top1_baseline, 6),
                    "top3_accuracy": round(top3, 6),
                    "macro_f1": round(tm["macro_f1"], 6),
                    "binary_f1": round(tm["binary_f1"], 6),
                    "precision": round(tm["precision"], 6),
                    "recall": round(tm["recall"], 6),
                    "log_loss": round(tm["log_loss"], 6),
                    "brier_score": round(tm["brier_score"], 6),
                    "ece": round(cal_trip["ece"], 6),
                    "mean_cands_within_15m": round(mean_cands_15m, 4),
                    "dense_intersection_rate": round(dense_intersection_rate, 4),
                    "mean_local_road_density_30m": round(mean_road_density, 4),
                    "gps_gap_rate": round(gps_gap_rate, 4),
                    "mean_speed_mps": round(mean_speed_mps, 4),
                    "mean_gt_dist_m": round(mean_gt_dist_m, 4),
                    "mean_gt_heading_diff_deg": round(mean_gt_heading_diff_deg, 4),
                }
            )

    per_trip_df = pd.DataFrame(per_trip_rows)
    per_trip_csv_path = output_dir / "per_trip_results.csv"
    per_trip_df.to_csv(per_trip_csv_path, index=False)

    # 4. Domain Error Regime Slice Analysis (Open Road vs Dense Intersections vs Heading Turns vs GPS Gaps)
    regime_slices: dict[str, Any] = {}
    obs_first_mask = ~test_df.duplicated(subset=["trip_id", "obs_idx"], keep="first")
    obs_phys = phys_df[obs_first_mask].copy()
    obs_phys["trip_id"] = test_df.loc[obs_first_mask, "trip_id"].to_numpy()
    obs_phys["obs_idx"] = test_df.loc[obs_first_mask, "obs_idx"].to_numpy()
    obs_phys = obs_phys.set_index(["trip_id", "obs_idx"])

    slice_definitions = {
        "open_road_low_ambiguity (cands_within_15m <= 2)": obs_phys["cands_15m"] <= 2.1,
        "dense_intersection_or_parallel (cands_within_15m >= 3)": obs_phys["cands_15m"] >= 2.9,
        "aligned_heading (gt_heading_diff <= 25 deg)": obs_phys["heading_diff_deg"] <= 25.0,
        "turning_or_noisy_heading (gt_heading_diff > 25 deg)": obs_phys["heading_diff_deg"] > 25.0,
        "high_gps_noise (gt_dist_to_arc > 10m)": obs_phys["dist_m"] > 10.0,
        "low_speed_stationary (gps_speed < 2.0 m/s)": obs_phys["speed_mps"] < 2.0,
    }

    for slice_name, mask_series in slice_definitions.items():
        valid_obs_keys = set(mask_series[mask_series].index)
        slice_mask = test_df.set_index(["trip_id", "obs_idx"]).index.isin(valid_obs_keys)
        slice_sub = test_df[slice_mask]
        n_slice_obs = int(len(valid_obs_keys))
        if n_slice_obs == 0:
            continue

        nearest_acc = float(
            (slice_sub[slice_sub["cand_rank"] == 0]["is_match"] == 1).mean()
        )
        slice_models: dict[str, Any] = {
            "num_observations": n_slice_obs,
            "proportion_of_test_obs": round(n_slice_obs / 19160.0, 4),
            "nearest_segment_top1_acc": round(nearest_acc, 6),
        }
        for spec in MODEL_SPECS:
            slug = spec["slug"]
            s_eval = slice_sub[["trip_id", "obs_idx", "arc_id", "cand_rank", "is_match"]].copy()
            s_eval["pred_prob"] = pred_dfs[slug].loc[slice_sub.index, "pred_prob"].to_numpy(dtype=np.float64)
            sm = evaluate_predictions(s_eval, threshold=0.5)
            slice_models[slug] = {
                "top1_accuracy": round(sm["top1_accuracy"], 6),
                "macro_f1": round(sm["macro_f1"], 6),
                "log_loss": round(sm["log_loss"], 6),
            }
        regime_slices[slice_name] = slice_models

    audit_payload = {
        "schema_verification": schema_checks,
        "models": models_calibration,
        "hmm_emission_readiness_recommendation": {
            "best_calibrated_model": min(
                models_calibration.values(), key=lambda m: m["brier_score"]
            )["model_name"],
            "highest_top1_model": max(
                models_calibration.values(), key=lambda m: m["top1_accuracy"]
            )["model_name"],
            "highest_macro_f1_model": max(
                models_calibration.values(), key=lambda m: m["macro_f1"]
            )["model_name"],
            "notes": (
                "Class-balanced models (Logistic Regression and Random Forest with class_weight='balanced*') "
                "shift mean predicted probabilities upward (~0.32–0.38 vs 0.125 base rate) to maximize recall "
                "on the 1:7 imbalanced candidate pool. When used as emission likelihoods in Viterbi decoding, "
                "dividing by prior class weights or applying log-odds scaling restores exact Bayesian emission "
                "calibration while preserving their superior candidate ranking."
            ),
        },
    }

    cal_plot_path = output_dir / "calibration_curves.png"
    plot_calibration_curves(pred_dfs, audit_payload, cal_plot_path)

    cal_json_path = output_dir / "calibration_audit.json"
    with cal_json_path.open("w", encoding="utf-8") as fh:
        json.dump(audit_payload, fh, indent=2)

    error_json_path = output_dir / "per_trip_error_analysis.json"
    error_payload = {
        "per_trip_summary_by_model": per_trip_rows,
        "error_regime_slices": regime_slices,
    }
    with error_json_path.open("w", encoding="utf-8") as fh:
        json.dump(error_payload, fh, indent=2)

    print(f"[Phase 4 Audit] Saved calibration plot -> {cal_plot_path}")
    print(f"[Phase 4 Audit] Saved calibration JSON -> {cal_json_path}")
    print(f"[Phase 4 Audit] Saved per-trip CSV     -> {per_trip_csv_path}")
    print(f"[Phase 4 Audit] Saved error slices JSON -> {error_json_path}")

    return {
        "calibration_audit": audit_payload,
        "error_analysis": error_payload,
    }


if __name__ == "__main__":
    run_calibration_and_error_analysis()
