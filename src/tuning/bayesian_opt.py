"""Seeded Optuna search on trip-grouped training folds."""
import numpy as np
import optuna
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from src.models.advanced.common import SEED, fit_model, trip_splits


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
                                 epochs=epochs, estimator=estimator)
            probabilities = model.predict_proba(X.iloc[val])[:, 1].astype("float64")
            scores.append(log_loss(y.iloc[val], probabilities, labels=[0, 1]))
        trial.set_user_attr("fold_log_loss", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, n_jobs=1)
    best_params = proposed_params(optuna.trial.FixedTrial(study.best_params), name, prefix,
                                  max_neighbors=max_neighbors)
    return study, best_params
