"""Compare all six baselines, tune the winner, and export its probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import distribution_to_json
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.models.advanced.common import (
    NAMES, SEED, candidate_predictions, comparison_row, data_arguments, data_signature,
    evaluate_model, feature_names, fit_model, grouped_cv, importance_table, load_data,
    package_versions, trip_splits, write_json,
)

CLASSICAL_TYPES = {
    "logreg": (LogisticRegression,),
    "knn": (KNeighborsClassifier,),
    "rf": (RandomForestClassifier, ExtraTreesClassifier),
}
RANK_METRICS = {
    "cv_macro_f1_mean": (0.40, False),
    "student_test_top1_accuracy": (0.30, False),
    "student_test_log_loss": (0.20, True),
    "inference_time_sec": (0.10, True),
}


def classical_slug(name):
    name = name.lower()
    if "logistic" in name or name == "logreg":
        return "logreg"
    if "knn" in name or "neighbor" in name or "neighbour" in name:
        return "knn"
    if "forest" in name or "extra tree" in name or name == "rf":
        return "rf"
    raise ValueError(f"Unexpected model in Person 2's comparison: {name}")


def choose_model(advanced, classical_path=None):
    table = pd.DataFrame([{"slug": r["model"], **comparison_row(r)} for r in advanced])
    if classical_path is not None:
        classical = pd.read_csv(classical_path)
        classical["slug"] = classical.model_name.map(classical_slug)
        if len(classical) != 3 or set(classical.slug) != set(CLASSICAL_TYPES):
            raise ValueError("Expected exactly the three real Person 2 v2 model rows")
        if classical.astype(str).apply(lambda c: c.str.contains("synthetic|smoke", case=False)).any().any():
            raise ValueError("Synthetic Person 2 results cannot be used for the official model selection")
        if "feature_version" in classical and not classical.feature_version.eq("v2").all():
            raise ValueError("The classical comparison must use v2 features")
        table = pd.concat([table, classical], ignore_index=True, sort=False)
    table["weighted_rank"] = 0.0
    for metric, (weight, ascending) in RANK_METRICS.items():
        values = pd.to_numeric(table[metric], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"All models need a finite {metric} for selection")
        table["weighted_rank"] += weight * values.rank(ascending=ascending)
    return table.sort_values(["weighted_rank", "cv_macro_f1_mean", "slug"],
                             ascending=[True, False, True]).reset_index(drop=True)


def proposed_params(trial, name, prefix="", *, max_neighbors=31):
    if name == "mlp":
        architecture = trial.suggest_categorical("architecture", ["32", "64x32", "96x48"])
        return {
            "hidden_layer_sizes": tuple(map(int, architecture.split("x"))),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
        }
    if name == "logreg":
        return {prefix + "C": trial.suggest_float("C", 0.01, 100, log=True),
                prefix + "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "knn":
        return {prefix + "n_neighbors": trial.suggest_int("n_neighbors", min(3, max_neighbors), max_neighbors),
                prefix + "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                prefix + "p": trial.suggest_int("p", 1, 2)}
    if name == "rf":
        return {prefix + "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
                prefix + "max_depth": trial.suggest_int("max_depth", 5, 24),
                prefix + "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 24),
                prefix + "max_features": trial.suggest_float("max_features", 0.45, 1.0)}

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 320, step=40),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.20, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    if name == "lgbm":
        return params | {"num_leaves": trial.suggest_int("num_leaves", 15, 95),
                         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                         "subsample_freq": 1}
    return params | {"max_depth": trial.suggest_int("max_depth", 3, 8),
                     "min_child_weight": trial.suggest_int("min_child_weight", 1, 12)}


def tune(train, features, name, *, trials, epochs, estimator=None):
    # The objective receives only training rows and validation trip indices.
    X, y = train[features], train.is_match
    splits = trip_splits(train)
    prefix = estimator.steps[-1][0] + "__" if isinstance(estimator, Pipeline) else ""
    max_neighbors = min(31, min(len(fit) for fit, _ in splits))

    def objective(trial):
        params = proposed_params(trial, name, prefix, max_neighbors=max_neighbors)
        scores = []
        for fit, val in splits:
            model, _ = fit_model(name, X.iloc[fit], y.iloc[fit], params=params,
                                 epochs=epochs, estimator=estimator, max_rows=100000)
            probabilities = model.predict_proba(X.iloc[val])[:, 1].astype("float64")
            scores.append(log_loss(y.iloc[val], probabilities, labels=[0, 1]))
        trial.set_user_attr("fold_log_loss", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, n_jobs=1)
    best_params = proposed_params(optuna.trial.FixedTrial(study.best_params), name, prefix,
                                  max_neighbors=max_neighbors)
    return study, best_params


def run_tuning(train, test, features, advanced, output, artifacts, *, trials, epochs,
               signature, classical_path=None, classical_estimator=None, smoke=False):
    if not smoke and classical_path is None:
        raise ValueError("Person 2's real v2 comparison CSV is required before official tuning")
    if trials < (2 if smoke else 35):
        raise ValueError("Use at least 35 trials for the official run or two for a smoke run")
    if sorted(record["model"] for record in advanced) != sorted(NAMES):
        raise ValueError("Expected exactly one v2 baseline for each of LightGBM, XGBoost, and MLP")
    for record in advanced:
        if record["data_signature"] != signature or record["feature_version"] != "v2":
            raise ValueError("Re-run all advanced baselines on these exact v2 inputs before tuning")
        if record["smoke"] != smoke or record["epochs"] != epochs:
            raise ValueError("Baseline run mode and epoch count must match the tuning run")
    selection = choose_model(advanced, classical_path)
    winner = selection.iloc[0]
    name = winner.slug
    estimator = None
    if name in CLASSICAL_TYPES:
        if classical_estimator is None:
            raise ValueError(
                f"{winner.model_name} led the six-model selection. Supply its original trusted "
                "Person 2 estimator or sklearn pipeline with --classical-estimator.",
            )
        estimator = joblib.load(classical_estimator)
        classifier = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        if not isinstance(classifier, CLASSICAL_TYPES[name]):
            raise ValueError(f"The supplied estimator does not match the selected {winner.model_name}.")

    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output / "model_selection.csv", index=False)
    start = perf_counter()
    study, best_params = tune(train, features, name, trials=trials, epochs=epochs, estimator=estimator)
    search_seconds = perf_counter() - start
    trials_frame = study.trials_dataframe(attrs=("number", "value", "duration", "params", "state", "user_attrs"))
    trials_frame["duration_seconds"] = trials_frame.pop("duration").dt.total_seconds()
    trials_frame.to_csv(output / "optuna_trials.csv", index=False)
    importance = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED),
    ) if trials >= 5 else {}
    pd.DataFrame(list(importance.items()), columns=["parameter", "importance"]).to_csv(
        output / "optuna_parameter_importance.csv", index=False,
    )

    comparison = []
    for label, params in (("untuned", None), ("tuned", best_params)):
        cv, _ = grouped_cv(train, features, name, params=params, epochs=epochs, estimator=estimator)
        model, probabilities, record = evaluate_model(
            train, test, features, name, cv, params=params, epochs=epochs, estimator=estimator,
        )
        record["model_name"] = str(winner.model_name)
        comparison.append({"version": label, **comparison_row(record)})
    write_json(output / "tuned_metrics.json", record)
    pd.DataFrame(comparison).to_csv(output / "untuned_vs_tuned.csv", index=False)
    importance_table(model, test, features).to_csv(output / "tuned_feature_importance.csv", index=False)

    # Verify the complete prediction batch before publishing the model at its handoff path.
    model_path = artifacts / "final_model.joblib"
    staged = model_path.with_suffix(".tmp")
    joblib.dump(model, staged, compress=3)
    reloaded = joblib.load(staged)
    if not np.array_equal(probabilities, reloaded.predict_proba(test[features])[:, 1]):
        staged.unlink()
        raise RuntimeError("Reloaded model changed the exported probabilities")
    staged.replace(model_path)
    candidate_predictions(test, probabilities).to_parquet(output / "tuned_student_test_preds.parquet", index=False)

    summary = {
        "smoke": smoke, "feature_version": "v2", "data_signature": signature,
        "selected_model": name, "selected_model_name": str(winner.model_name),
        "baseline_models_compared": len(selection),
        "selection_rank_weights": {metric: settings[0] for metric, settings in RANK_METRICS.items()},
        "objective": "Minimize mean candidate log loss across five trip-grouped training folds",
        "sampler": "TPESampler", "seed": SEED, "trials": len(study.trials),
        "completed_trials": int(trials_frame.state.eq("COMPLETE").sum()),
        "best_trial": study.best_trial.number, "best_cv_log_loss": study.best_value,
        "best_params": best_params, "search_space": {
            key: json.loads(distribution_to_json(distribution))
            for key, distribution in study.best_trial.distributions.items()
        },
        "tuning_wall_seconds": search_seconds, "summed_trial_seconds": float(trials_frame.duration_seconds.sum()),
        "train_trips": train.trip_id.nunique(), "student_test_trips": test.trip_id.nunique(),
    }
    write_json(artifacts / "final_model_meta.json", {
        **summary, "model": name, "feature_columns": features, "match_class": 1,
        "threshold": 0.5, "software": package_versions(), "exact_reload_verified": True,
    })
    write_json(output / "tuning_summary.json", summary)
    print(f"{name}: completed {summary['completed_trials']} trials; best CV log loss "
          f"{study.best_value:.5f} (trial {study.best_trial.number})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_arguments(parser)
    parser.add_argument("--advanced", type=Path, default=Path("outputs/models/advanced"))
    parser.add_argument("--classical", type=Path, default=Path("outputs/models/classical/classical_comparison.csv"))
    parser.add_argument("--classical-estimator", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/tuning"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--trials", type=int, default=35)
    args = parser.parse_args()
    if args.epochs < 1 or args.trials < 35:
        parser.error("Use a positive epoch count and at least 35 Optuna trials.")
    if not args.classical.is_file():
        parser.error(f"Person 2's real v2 comparison is missing: {args.classical}")
    features = feature_names(args.features)
    train, test = load_data(args.train, args.test, features)
    advanced = [json.loads((args.advanced / f"{name}_metrics.json").read_text(encoding="utf-8")) for name in NAMES]
    run_tuning(train, test, features, advanced, args.output, args.artifacts, trials=args.trials,
               epochs=args.epochs, signature=data_signature(args.train, args.test, args.features),
               classical_path=args.classical, classical_estimator=args.classical_estimator)


if __name__ == "__main__":
    main()
