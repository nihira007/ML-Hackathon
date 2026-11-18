"""PathSnap training, trip-grouped validation, and the prediction handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
KEY = ["trip_id", "obs_idx", "arc_id"]
PRED_COLS = [*KEY, "is_match", "pred_prob"]
METADATA = [*KEY, "timestamp", "cand_rank", "is_match"]
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP"}
DEFAULTS = {
    "lgbm": {"n_estimators": 160, "learning_rate": 0.06, "num_leaves": 31,
             "min_child_samples": 30, "importance_type": "gain"},
    "xgb": {"n_estimators": 160, "learning_rate": 0.06, "max_depth": 5,
            "tree_method": "hist", "importance_type": "gain", "eval_metric": "logloss"},
    "mlp": {"hidden_layer_sizes": (64, 32), "alpha": 0.0001, "learning_rate_init": 0.001,
            "batch_size": 512, "solver": "adam", "activation": "relu"},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def feature_names(path):
    names = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(names, dict):
        names = next((names[key] for key in ("feature_columns", "features", "columns") if key in names), None)
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"Expected an ordered list of feature names in {path}")
    # Rank is available at inference. Route labels and row identifiers are not features.
    forbidden = set(METADATA) - {"cand_rank"}
    if len(set(names)) != len(names) or forbidden.intersection(names):
        raise ValueError("Feature manifest has duplicate names or includes a protected metadata column")
    return names


def data_signature(*paths):
    return [file_sha256(path) for path in paths]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frame(path, features):
    df = pd.read_parquet(path, columns=list(dict.fromkeys([*METADATA, *features])))
    if df.empty or df[METADATA].isna().any().any():
        raise ValueError(f"{path} has no candidate rows or contains missing IDs, ranks, timestamps, or labels")
    df["trip_id"] = df["trip_id"].astype(str)
    if not df["is_match"].isin([0, 1]).all() or df.duplicated(KEY).any():
        raise ValueError(f"{path} contains invalid labels or duplicate candidate rows")
    positives = df.groupby(["trip_id", "obs_idx"], sort=False)["is_match"].sum()
    if positives.gt(1).any():
        raise ValueError(f"{path} contains multiple labeled matches for a single observation")
    for col in features:
        values = pd.to_numeric(df[col], errors="raise")
        if np.isinf(values).any():
            raise ValueError(f"{path} contains infinite values in {col}. Use NaN to encode missing features.")
        if col != "cand_rank":
            df[col] = values.astype("float32")
    return df


def load_data(train_path, test_path, features, *, smoke=False):
    train = load_frame(train_path, features)
    test = load_frame(test_path, features)
    if not set(train.trip_id).isdisjoint(test.trip_id):
        raise ValueError("Training and student-test trips overlap")
    if not smoke and (train.trip_id.nunique(), test.trip_id.nunique()) != (80, 10):
        raise ValueError("The official run requires 80 training trips and 10 student-test trips")
    if train.is_match.nunique() != 2:
        raise ValueError("Training requires at least one example of each candidate class")
    return train, test


def candidate_predictions(frame, probabilities):
    pred = frame[PRED_COLS[:-1]].copy()
    pred["pred_prob"] = probabilities
    return pred


def metrics(frame, probabilities):
    y = frame.is_match.to_numpy()
    p = np.asarray(probabilities, dtype="float64")
    if len(p) != len(frame) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected one finite probability in [0, 1] per candidate")
    hard = p >= 0.5
    ranked = frame[[*KEY, "cand_rank", "is_match"]].assign(pred_prob=p).sort_values(
        ["trip_id", "obs_idx", "pred_prob", "cand_rank", "arc_id"],
        ascending=[True, True, False, True, True],
    )
    best = ranked.drop_duplicates(["trip_id", "obs_idx"])
    can_match = ranked.groupby(["trip_id", "obs_idx"], sort=False).is_match.max()
    return {
        "accuracy": float(accuracy_score(y, hard)),
        "macro_f1": float(f1_score(y, hard, average="macro", labels=[0, 1], zero_division=0)),
        "precision": float(precision_score(y, hard, zero_division=0)),
        "recall": float(recall_score(y, hard, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top1_accuracy": float(best.is_match.mean()),
        "candidate_miss_rate": float(1 - can_match.mean()),
        "observations": len(best),
        "candidates": len(frame),
        "confusion_matrix": confusion_matrix(y, hard, labels=[0, 1]).tolist(),
    }


def build_model(name, params=None):
    config = DEFAULTS[name] | (params or {})
    # A named output keeps feature identity through preprocessing and export.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return make_pipeline(imputer, LGBMClassifier(
            random_state=SEED, verbosity=-1, n_jobs=2, deterministic=True, force_col_wise=True, **config,
        ))
    if name == "xgb":
        from xgboost import XGBClassifier
        return make_pipeline(imputer, XGBClassifier(random_state=SEED, n_jobs=2, **config))
    scaler = StandardScaler().set_output(transform="pandas")
    model = MLPClassifier(random_state=np.random.RandomState(SEED), **config)
    return make_pipeline(imputer, scaler, model)


def fit_model(name, X, y, *, params=None, epochs=20, validation=None, estimator=None):
    if estimator is None:
        model = build_model(name, params)
    else:
        model = clone(estimator).set_params(**(params or {}))
    if name != "mlp" or estimator is not None:
        model.fit(X, y)
        return model, []

    # partial_fit performs one pass and preserves Adam's optimizer state between passes.
    scaling = model[:-1]
    net = model.steps[-1][1]
    X_train = scaling.fit_transform(X)
    history = []
    if validation is not None:
        X_val, y_val = validation
        X_val = scaling.transform(X_val)
    with threadpool_limits(limits=2):
        for epoch in range(1, epochs + 1):
            net.partial_fit(X_train, y, classes=[0, 1])
            if validation is not None:
                history.append({
                    "epoch": epoch,
                    "train_loss": float(log_loss(y, net.predict_proba(X_train)[:, 1], labels=[0, 1])),
                    "validation_loss": float(log_loss(y_val, net.predict_proba(X_val)[:, 1], labels=[0, 1])),
                })
    return model, history


def trip_splits(train):
    splits = list(GroupKFold(n_splits=5).split(train, train.is_match, train.trip_id))
    for train_idx, val_idx in splits:
        if not set(train.trip_id.iloc[train_idx]).isdisjoint(train.trip_id.iloc[val_idx]):
            raise RuntimeError("A trip appears on both sides of a GroupKFold split")
        if train.is_match.iloc[train_idx].nunique() != 2:
            raise ValueError("A grouped training fold has fewer than two candidate classes")
    return splits


def grouped_cv(train, features, name, *, params=None, epochs=20, estimator=None):
    X, y = train[features], train.is_match
    scores, histories = [], []
    start = perf_counter()
    for fold, (train_idx, val_idx) in enumerate(trip_splits(train), 1):
        X_fit, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fit, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model, history = fit_model(name, X_fit, y_fit, params=params, epochs=epochs,
                                   validation=(X_val, y_val), estimator=estimator)
        val = metrics(train.iloc[val_idx], model.predict_proba(X_val)[:, 1])
        train_f1 = f1_score(y_fit, model.predict_proba(X_fit)[:, 1] >= 0.5,
                            labels=[0, 1], average="macro", zero_division=0)
        scores.append({"fold": fold, "validation_trips": sorted(train.trip_id.iloc[val_idx].unique()),
                       "train_macro_f1": float(train_f1), **val})
        histories.extend({"fold": fold, **item} for item in history)
    rows = pd.DataFrame(scores)
    return {
        "cv_macro_f1_mean": float(rows.macro_f1.mean()),
        "cv_macro_f1_std": float(rows.macro_f1.std(ddof=0)),
        "cv_top1_accuracy_mean": float(rows.top1_accuracy.mean()),
        "cv_log_loss_mean": float(rows.log_loss.mean()),
        "train_macro_f1_mean": float(rows.train_macro_f1.mean()),
        "cv_time_sec": perf_counter() - start,
        "fold_metrics": scores,
    }, pd.DataFrame(histories)


def cli(name=None):
    parser = argparse.ArgumentParser(description="Check the advanced models on generated data")
    parser.parse_args()
    from src.models.advanced.smoke import check_models
    check_models([name] if name else NAMES)


if __name__ == "__main__":
    cli()
