"""Benchmark 5 High-Accuracy Models (LightGBM, XGBoost, Optuna-Tuned XGBoost, HistGradientBoosting, Soft-Voting Ensemble) + HMM/Viterbi.

Evaluates all 5 models on the exact same 33-feature v2 dataset (`student_test_features.parquet`,
10 trips, 19,160 GPS observations, 157,866 candidates) and pairs each with Person 4's
HMM/Viterbi sequence decoder (`artifacts/hmm_config.json`), exporting:
- `outputs/models/advanced/high_accuracy_5models.csv`
- `outputs/models/advanced/hgb_student_test_preds.parquet`
- `outputs/models/advanced/ensemble_student_test_preds.parquet`
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.graph.road_graph import RoadGraph
from src.hmm.evaluate_hmm import evaluate_trip_predictions, load_trip_route, load_trip_track
from src.hmm.viterbi import HMMConfig
from src.models.advanced.common import (
    SEED,
    candidate_predictions,
    feature_names,
    grouped_cv,
    load_data,
    metrics,
    trip_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = PROJECT_ROOT / "Student folder" / "student_test"


def evaluate_with_hmm(
    pred_df: pd.DataFrame,
    road_graphs: dict[str, RoadGraph],
    tracks: dict[str, np.ndarray],
    routes: dict[str, list[int]],
    hmm_config: HMMConfig,
) -> tuple[float, float, float]:
    """Run HMM/Viterbi across all 10 student_test trips and return (mean_top1, weighted_top1, mean_continuity)."""
    per_trip_top1 = []
    per_trip_cont = []
    total_obs = 0
    total_correct = 0

    for tid in sorted(road_graphs.keys()):
        g = pred_df[pred_df["trip_id"] == tid]
        res = evaluate_trip_predictions(
            trip_id=tid,
            df_trip_candidates=g,
            road_graph=road_graphs[tid],
            gps_track=tracks[tid],
            ground_truth_route=routes[tid],
            hmm_config=hmm_config,
        )
        n_obs = res["n_observations"]
        top1 = float(res["hmm_viterbi"]["top1_accuracy"])
        cont = float(res["hmm_viterbi"]["route_continuity"])
        per_trip_top1.append(top1)
        per_trip_cont.append(cont)
        total_obs += n_obs
        total_correct += int(round(top1 * n_obs))

    return float(np.mean(per_trip_top1)), float(total_correct / total_obs), float(np.mean(per_trip_cont))


def run_5_high_accuracy_models() -> pd.DataFrame:
    features = feature_names(PROJECT_ROOT / "artifacts" / "feature_columns.json")
    train, test = load_data(
        PROJECT_ROOT / "data" / "processed" / "train_features.parquet",
        PROJECT_ROOT / "data" / "processed" / "student_test_features.parquet",
        features,
    )

    adv_dir = PROJECT_ROOT / "outputs" / "models" / "advanced"
    tuning_dir = PROJECT_ROOT / "outputs" / "tuning"

    # 1. Train HistGradientBoostingClassifier (Model 4 of 5) with 5-fold GroupKFold CV
    print("[1/2] Training HistGradientBoostingClassifier (5-fold GroupKFold CV + full train)...")
    hgb_pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas"),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    )

    X_tr, y_tr = train[features], train["is_match"]
    rng = np.random.default_rng(SEED)
    hgb_cv_f1 = []
    for train_idx, val_idx in trip_splits(train):
        sub_idx = train_idx
        if len(sub_idx) > 220000:
            sub_idx = np.sort(rng.choice(sub_idx, size=220000, replace=False))
        from sklearn.base import clone
        m_fold = clone(hgb_pipe)
        m_fold.fit(X_tr.iloc[sub_idx], y_tr.iloc[sub_idx])
        p_val = m_fold.predict_proba(X_tr.iloc[val_idx])[:, 1]
        hgb_cv_f1.append(metrics(train.iloc[val_idx], p_val)["macro_f1"])

    full_idx = np.sort(rng.choice(len(train), size=min(280000, len(train)), replace=False))
    t0 = perf_counter()
    hgb_pipe.fit(X_tr.iloc[full_idx], y_tr.iloc[full_idx])
    hgb_train_sec = perf_counter() - t0
    t0 = perf_counter()
    hgb_probs = hgb_pipe.predict_proba(test[features])[:, 1]
    hgb_infer_sec = perf_counter() - t0
    hgb_test_m = metrics(test, hgb_probs)
    hgb_pq = adv_dir / "hgb_student_test_preds.parquet"
    candidate_predictions(test, hgb_probs).to_parquet(hgb_pq, index=False)

    # 2. Build Soft-Voting Ensemble (LightGBM + Tuned XGBoost + HistGB + MLP)
    print("[2/2] Building Soft-Voting Ensemble (LightGBM + Tuned XGBoost + HistGB + MLP)...")
    lgbm_df = pd.read_parquet(adv_dir / "lgbm_student_test_preds.parquet")
    xgb_df = pd.read_parquet(adv_dir / "xgb_student_test_preds.parquet")
    mlp_df = pd.read_parquet(adv_dir / "mlp_student_test_preds.parquet")
    tuned_df = pd.read_parquet(tuning_dir / "tuned_student_test_preds.parquet")

    ens_probs = (
        0.35 * tuned_df["pred_prob"].to_numpy()
        + 0.30 * lgbm_df["pred_prob"].to_numpy()
        + 0.25 * hgb_probs
        + 0.10 * mlp_df["pred_prob"].to_numpy()
    )
    ens_test_m = metrics(test, ens_probs)
    ens_pq = adv_dir / "ensemble_student_test_preds.parquet"
    candidate_predictions(test, ens_probs).to_parquet(ens_pq, index=False)

    # 3. Pre-load RoadGraphs, Tracks, Routes & evaluate all 5 models + HMM/Viterbi
    road_graphs = {}
    tracks = {}
    routes = {}
    for t_dir in sorted(TEST_DIR.iterdir()):
        if t_dir.is_dir():
            tid = t_dir.name
            road_graphs[tid] = RoadGraph.from_trip_dir(str(t_dir), trip_id=tid)
            tracks[tid] = load_trip_track(t_dir, tid)
            routes[tid] = load_trip_route(t_dir, tid)

    hmm_cfg_dict = json.loads((PROJECT_ROOT / "artifacts" / "hmm_config.json").read_text(encoding="utf-8"))
    valid_cfg = {k: v for k, v in hmm_cfg_dict.items() if k in HMMConfig.__dataclass_fields__}
    hmm_config = HMMConfig(**valid_cfg)

    adv_comp = pd.read_csv(adv_dir / "advanced_comparison.csv").set_index("model_name")
    untuned_vs_tuned = pd.read_csv(tuning_dir / "untuned_vs_tuned.csv").set_index("version")

    specs = [
        (
            "1. Soft-Voting Ensemble (LGBM+XGB+HGB+MLP) + HMM",
            ens_pq,
            float(np.mean(hgb_cv_f1)),
            ens_test_m["accuracy"],
            ens_test_m["macro_f1"],
            ens_test_m["top1_accuracy"],
            ens_test_m["log_loss"],
        ),
        (
            "2. LightGBM + HMM/Viterbi",
            adv_dir / "lgbm_student_test_preds.parquet",
            float(adv_comp.loc["LightGBM", "cv_macro_f1_mean"]),
            float(adv_comp.loc["LightGBM", "student_test_accuracy"]),
            float(adv_comp.loc["LightGBM", "student_test_macro_f1"]),
            float(adv_comp.loc["LightGBM", "student_test_top1_accuracy"]),
            float(adv_comp.loc["LightGBM", "student_test_log_loss"]),
        ),
        (
            "3. XGBoost (Optuna Tuned) + HMM/Viterbi",
            tuning_dir / "tuned_student_test_preds.parquet",
            float(untuned_vs_tuned.loc["tuned", "cv_macro_f1_mean"]),
            float(untuned_vs_tuned.loc["tuned", "student_test_accuracy"]),
            float(untuned_vs_tuned.loc["tuned", "student_test_macro_f1"]),
            float(untuned_vs_tuned.loc["tuned", "student_test_top1_accuracy"]),
            float(untuned_vs_tuned.loc["tuned", "student_test_log_loss"]),
        ),
        (
            "4. HistGradientBoosting + HMM/Viterbi",
            hgb_pq,
            float(np.mean(hgb_cv_f1)),
            hgb_test_m["accuracy"],
            hgb_test_m["macro_f1"],
            hgb_test_m["top1_accuracy"],
            hgb_test_m["log_loss"],
        ),
        (
            "5. XGBoost (Baseline) + HMM/Viterbi",
            adv_dir / "xgb_student_test_preds.parquet",
            float(adv_comp.loc["XGBoost", "cv_macro_f1_mean"]),
            float(adv_comp.loc["XGBoost", "student_test_accuracy"]),
            float(adv_comp.loc["XGBoost", "student_test_macro_f1"]),
            float(adv_comp.loc["XGBoost", "student_test_top1_accuracy"]),
            float(adv_comp.loc["XGBoost", "student_test_log_loss"]),
        ),
    ]

    rows = []
    for name, pq, cv_f1, cand_acc, test_f1, cls_top1, ll in specs:
        df_p = pd.read_parquet(pq)
        hmm_mean_top1, hmm_w_top1, hmm_cont = evaluate_with_hmm(
            df_p, road_graphs, tracks, routes, hmm_config
        )
        rows.append(
            {
                "Model": name,
                "CV Macro F1": round(cv_f1, 4),
                "Candidate Accuracy": round(cand_acc, 4),
                "Student-Test Macro F1": round(test_f1, 4),
                "Pointwise Top-1 Acc": round(cls_top1, 4),
                "HMM Mean Trip Top-1 Acc": round(hmm_mean_top1, 4),
                "HMM Route Continuity": round(hmm_cont, 4),
                "Log Loss": round(ll, 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = adv_dir / "high_accuracy_5models.csv"
    out_df.to_csv(out_csv, index=False)
    print("\n=== 5 High-Accuracy Models Summary ===")
    print(out_df.to_string(index=False))
    return out_df


if __name__ == "__main__":
    run_5_high_accuracy_models()
