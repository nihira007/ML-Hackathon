"""
Shared candidate-level and observation-level evaluation metrics, trip-grouped
cross-validation (GroupKFold), prediction schema validators, and timing utilities.

Owned by Person 2 (branch: feat/p2-models-classical).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Shared contract column constants
REQUIRED_ID_COLUMNS: List[str] = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "arc_id",
    "cand_rank",
]
CORE_CANDIDATE_COLUMNS: List[str] = [
    "trip_id",
    "obs_idx",
    "arc_id",
    "cand_rank",
    "is_match",
]
PREDICTION_COLUMNS: List[str] = [
    "trip_id",
    "obs_idx",
    "arc_id",
    "is_match",
    "pred_prob",
]

EPSILON: float = 1e-15


class Timer:
    """Context manager for high-resolution wall-clock timing using time.perf_counter()."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_sec: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_sec = float(self.end_time - self.start_time)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_sec * 1000.0


def measure_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Execute `fn(*args, **kwargs)` and return `(result, elapsed_seconds)`."""
    with Timer() as timer:
        result = fn(*args, **kwargs)
    return result, timer.elapsed_sec


def validate_feature_dataframe(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    require_timestamp: bool = False,
    require_label: bool = True,
) -> None:
    """Validate that a feature DataFrame satisfies the shared PathSnap parquet schema."""
    required = ["trip_id", "obs_idx", "arc_id", "cand_rank"]
    if require_timestamp:
        required.append("timestamp")
    if require_label:
        required.append("is_match")
    if feature_cols is not None:
        required.extend(list(feature_cols))

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Feature DataFrame is missing required schema columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if len(df) == 0:
        raise ValueError("Feature DataFrame is empty.")


def validate_prediction_dataframe(df: pd.DataFrame) -> None:
    """Validate that a prediction DataFrame matches `['trip_id', 'obs_idx', 'arc_id', 'is_match', 'pred_prob']`."""
    missing = [col for col in PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction DataFrame is missing required columns {missing}. "
            f"Expected exact columns: {PREDICTION_COLUMNS}"
        )
    if df["pred_prob"].isna().any():
        raise ValueError("Prediction DataFrame contains NaN values in 'pred_prob'.")
    probs = df["pred_prob"].to_numpy(dtype=np.float64)
    if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
        raise ValueError("Prediction probabilities 'pred_prob' must lie in [0.0, 1.0].")


def compute_candidate_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute all candidate-level classification and probability quality metrics.

    Metrics included:
      - accuracy: overall candidate classification accuracy
      - precision: positive-class (is_match=1) precision
      - recall: positive-class (is_match=1) recall
      - binary_f1: positive-class (is_match=1) F1-score
      - macro_f1: unweighted mean F1-score across classes 0 and 1
      - macro_precision: unweighted mean precision across classes 0 and 1
      - macro_recall: unweighted mean recall across classes 0 and 1
      - log_loss: binary cross-entropy loss on clipped probabilities
      - brier_score: mean squared probability error
      - confusion_matrix: 2x2 matrix [[TN, FP], [FN, TP]]
      - tn, fp, fn, tp: individual confusion matrix counts
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.shape[0]} rows, "
            f"y_prob has {y_prob_arr.shape[0]} rows."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("Cannot compute candidate metrics on empty arrays.")

    y_prob_clipped = np.clip(y_prob_arr, EPSILON, 1.0 - EPSILON)
    y_pred_arr = (y_prob_clipped >= threshold).astype(np.int64)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_pos = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    ll = float(log_loss(y_true_arr, y_prob_clipped, labels=[0, 1]))
    brier = float(brier_score_loss(y_true_arr, np.clip(y_prob_arr, 0.0, 1.0)))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "accuracy": acc,
        "precision": prec_pos,
        "recall": rec_pos,
        "binary_f1": f1_pos,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "log_loss": ll,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_candidates": int(y_true_arr.shape[0]),
        "positive_rate": float(np.mean(y_true_arr)),
    }


def compute_top1_accuracy(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    label_col: str = "is_match",
    group_cols: Tuple[str, str] = ("trip_id", "obs_idx"),
) -> float:
    """Compute Observation-level Top-1 Accuracy.

    Groups candidate rows by `(trip_id, obs_idx)`, selects the candidate with the
    highest `pred_prob` (breaking ties deterministically by `cand_rank` ascending if
    present, else `arc_id` ascending), and checks if `is_match == 1`.
    """
    for col in (*group_cols, prob_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required to compute Top-1 Accuracy.")
    if len(df) == 0:
        raise ValueError("Cannot compute Top-1 Accuracy on an empty DataFrame.")

    sort_cols: List[str] = list(group_cols) + [prob_col]
    ascending: List[bool] = [True] * len(group_cols) + [False]

    if "cand_rank" in df.columns:
        sort_cols.append("cand_rank")
        ascending.append(True)

    sorted_df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    top1_df = sorted_df.drop_duplicates(subset=list(group_cols), keep="first")

    return float((top1_df[label_col].to_numpy(dtype=np.int64) == 1).mean())


def compute_topk_accuracy(
    df: pd.DataFrame,
    k: int = 1,
    prob_col: str = "pred_prob",
    label_col: str = "is_match",
    group_cols: Tuple[str, str] = ("trip_id", "obs_idx"),
) -> float:
    """Compute Observation-level Top-K Accuracy (whether any top-k candidate has `is_match == 1`)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k == 1:
        return compute_top1_accuracy(df, prob_col=prob_col, label_col=label_col, group_cols=group_cols)

    sort_cols: List[str] = list(group_cols) + [prob_col]
    ascending: List[bool] = [True] * len(group_cols) + [False]
    if "cand_rank" in df.columns:
        sort_cols.append("cand_rank")
        ascending.append(True)

    sorted_df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    topk_df = sorted_df.groupby(list(group_cols), sort=False).head(k)
    obs_hits = topk_df.groupby(list(group_cols), sort=False)[label_col].max()
    return float((obs_hits.to_numpy(dtype=np.int64) == 1).mean())


def evaluate_predictions(
    df: pd.DataFrame,
    y_prob: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    threshold: float = 0.5,
    prob_col: str = "pred_prob",
    label_col: str = "is_match",
) -> Dict[str, Any]:
    """Compute both candidate-level and observation-level metrics on a candidate DataFrame."""
    eval_df = df.copy() if y_prob is not None else df
    if y_prob is not None:
        eval_df[prob_col] = np.asarray(y_prob, dtype=np.float64)

    if prob_col not in eval_df.columns:
        raise ValueError(f"Column '{prob_col}' not found in DataFrame and `y_prob` was not provided.")
    if label_col not in eval_df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    cand_metrics = compute_candidate_metrics(
        y_true=eval_df[label_col].to_numpy(dtype=np.int64),
        y_prob=eval_df[prob_col].to_numpy(dtype=np.float64),
        threshold=threshold,
    )
    top1_acc = compute_top1_accuracy(
        eval_df,
        prob_col=prob_col,
        label_col=label_col,
        group_cols=("trip_id", "obs_idx"),
    )

    num_obs = int(eval_df.groupby(["trip_id", "obs_idx"], sort=False).ngroups)
    num_trips = int(eval_df["trip_id"].nunique())

    return {
        **cand_metrics,
        "top1_accuracy": top1_acc,
        "num_observations": num_obs,
        "num_trips": num_trips,
    }


def compute_per_trip_metrics(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    label_col: str = "is_match",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-trip evaluation breakdown across all trips in `df`."""
    rows: List[Dict[str, Any]] = []
    for trip_id, trip_df in df.groupby("trip_id", sort=True):
        m = evaluate_predictions(
            trip_df,
            threshold=threshold,
            prob_col=prob_col,
            label_col=label_col,
        )
        rows.append(
            {
                "trip_id": str(trip_id),
                "num_observations": m["num_observations"],
                "num_candidates": m["num_candidates"],
                "top1_accuracy": m["top1_accuracy"],
                "macro_f1": m["macro_f1"],
                "binary_f1": m["binary_f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "accuracy": m["accuracy"],
                "log_loss": m["log_loss"],
                "brier_score": m["brier_score"],
            }
        )
    return pd.DataFrame(rows)


def get_trip_group_kfold(n_splits: int = 5) -> GroupKFold:
    """Return a `GroupKFold` splitter configured for trip-grouped cross-validation."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    return GroupKFold(n_splits=n_splits)


def iter_trip_cv_splits(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    label_col: str = "is_match",
    group_col: str = "trip_id",
    n_splits: int = 5,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(fold_idx, train_indices, val_indices)` grouped strictly by `trip_id`.

    Verifies on every fold that zero `trip_id` values overlap between train and validation sets.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    unique_trips = df[group_col].nunique()
    if unique_trips < n_splits:
        raise ValueError(
            f"Dataset has only {unique_trips} unique '{group_col}' values, "
            f"which is fewer than n_splits={n_splits}."
        )

    gkf = get_trip_group_kfold(n_splits=n_splits)
    groups = df[group_col].to_numpy()
    y = df[label_col].to_numpy()
    X_placeholder = np.zeros((len(df), 1), dtype=np.float32) if feature_cols is None else df[list(feature_cols)]

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_placeholder, y, groups=groups)):
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        overlap = train_trips.intersection(val_trips)
        if overlap:
            raise RuntimeError(
                f"Data leakage detected in fold {fold_idx}: overlapping trip_ids={overlap}"
            )
        yield fold_idx, train_idx, val_idx


def _predict_probabilities(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Extract positive-class (`is_match == 1`) probabilities from a fitted estimator."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=np.float64)
        return np.asarray(proba, dtype=np.float64).ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
    preds = np.asarray(model.predict(X), dtype=np.float64).ravel()
    return np.clip(preds, 0.0, 1.0)


def run_group_kfold_cv(
    model_fn: Callable[[], Any],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_match",
    group_col: str = "trip_id",
    n_splits: int = 5,
    scale_features: bool = False,
    threshold: float = 0.5,
    max_train_samples_per_fold: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run 5-fold `GroupKFold` cross-validation strictly grouped by `trip_id`.

    Parameters
    ----------
    model_fn:
        Zero-argument factory returning a fresh unfitted estimator for each fold.
    df:
        Training candidate DataFrame with `trip_id, obs_idx, arc_id, cand_rank, is_match` and `feature_cols`.
    feature_cols:
        Ordered list of numeric feature column names.
    label_col:
        Target binary column name (`is_match`).
    group_col:
        Grouping column name (`trip_id`).
    n_splits:
        Number of GroupKFold splits (default 5).
    scale_features:
        If True, fits a `StandardScaler` strictly on each fold's training split before transforming
        train and validation splits (preventing any cross-fold leakage).
    threshold:
        Probability threshold for binary candidate predictions (default 0.5).
    max_train_samples_per_fold:
        Optional cap on fold training samples (e.g. for instance-based KNN speed if needed),
        sampled deterministically using `random_state`.
    random_state:
        Random seed used when subsampling fold training rows.

    Returns
    -------
    dict with fold metrics, mean/std summaries (both train and validation for overfitting analysis),
    and out-of-fold (`oof_predictions`) DataFrame.
    """
    validate_feature_dataframe(df, feature_cols=feature_cols, require_label=True)

    X_all = df[list(feature_cols)].to_numpy(dtype=np.float64)
    y_all = df[label_col].to_numpy(dtype=np.int64)
    oof_probs = np.zeros(len(df), dtype=np.float64)

    fold_records: List[Dict[str, Any]] = []
    total_fit_sec = 0.0
    total_val_infer_sec = 0.0

    for fold_idx, train_idx, val_idx in iter_trip_cv_splits(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        group_col=group_col,
        n_splits=n_splits,
    ):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val = X_all[val_idx]

        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        fit_train_idx = np.arange(len(train_idx))
        if max_train_samples_per_fold is not None and len(train_idx) > max_train_samples_per_fold:
            rng = np.random.default_rng(random_state + fold_idx)
            fold_train_sub = df.iloc[train_idx]
            obs_codes = fold_train_sub.groupby(["trip_id", "obs_idx"], sort=False).ngroup().to_numpy()
            n_obs_total = int(obs_codes.max()) + 1
            avg_cands_per_obs = max(1.0, len(train_idx) / float(n_obs_total))
            target_obs = max(1, min(n_obs_total, int(max_train_samples_per_fold / avg_cands_per_obs)))
            chosen_obs = rng.choice(n_obs_total, size=target_obs, replace=False)
            fit_train_idx = np.nonzero(np.isin(obs_codes, chosen_obs))[0]

        model = model_fn()
        with Timer() as fit_timer:
            model.fit(X_train[fit_train_idx], y_train[fit_train_idx])
        fold_fit_sec = fit_timer.elapsed_sec
        total_fit_sec += fold_fit_sec

        with Timer() as val_timer:
            val_prob = _predict_probabilities(model, X_val)
        fold_val_infer_sec = val_timer.elapsed_sec
        total_val_infer_sec += fold_val_infer_sec

        oof_probs[val_idx] = val_prob

        # Evaluate on validation fold
        val_fold_df = df.iloc[val_idx][["trip_id", "obs_idx", "arc_id", "cand_rank", label_col]].copy()
        val_fold_df["pred_prob"] = val_prob
        val_metrics = evaluate_predictions(val_fold_df, threshold=threshold)

        # Evaluate on training fold (on fitted subset) to quantify overfitting vs validation
        train_prob = _predict_probabilities(model, X_train[fit_train_idx])
        train_fold_df = (
            df.iloc[train_idx[fit_train_idx]][["trip_id", "obs_idx", "arc_id", "cand_rank", label_col]].copy()
        )
        train_fold_df["pred_prob"] = train_prob
        train_metrics = evaluate_predictions(train_fold_df, threshold=threshold)

        fold_records.append(
            {
                "fold": fold_idx,
                "num_train_trips": int(df.iloc[train_idx][group_col].nunique()),
                "num_val_trips": int(df.iloc[val_idx][group_col].nunique()),
                "num_train_candidates": int(len(fit_train_idx)),
                "num_val_candidates": int(len(val_idx)),
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "train_top1_accuracy": train_metrics["top1_accuracy"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "train_log_loss": train_metrics["log_loss"],
                "val_log_loss": val_metrics["log_loss"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_accuracy": val_metrics["accuracy"],
                "val_brier_score": val_metrics["brier_score"],
                "fit_time_sec": fold_fit_sec,
                "val_inference_time_sec": fold_val_infer_sec,
            }
        )

    oof_df = format_prediction_df(df, oof_probs)
    oof_overall = evaluate_predictions(oof_df, threshold=threshold)

    val_macro_f1s = [r["val_macro_f1"] for r in fold_records]
    val_top1s = [r["val_top1_accuracy"] for r in fold_records]
    val_lls = [r["val_log_loss"] for r in fold_records]
    val_precs = [r["val_precision"] for r in fold_records]
    val_recs = [r["val_recall"] for r in fold_records]

    train_macro_f1s = [r["train_macro_f1"] for r in fold_records]
    train_top1s = [r["train_top1_accuracy"] for r in fold_records]
    train_lls = [r["train_log_loss"] for r in fold_records]

    return {
        "n_splits": n_splits,
        "cv_macro_f1_mean": float(np.mean(val_macro_f1s)),
        "cv_macro_f1_std": float(np.std(val_macro_f1s, ddof=0)),
        "cv_top1_accuracy_mean": float(np.mean(val_top1s)),
        "cv_top1_accuracy_std": float(np.std(val_top1s, ddof=0)),
        "cv_log_loss_mean": float(np.mean(val_lls)),
        "cv_log_loss_std": float(np.std(val_lls, ddof=0)),
        "cv_precision_mean": float(np.mean(val_precs)),
        "cv_recall_mean": float(np.mean(val_recs)),
        "train_macro_f1_mean": float(np.mean(train_macro_f1s)),
        "train_macro_f1_std": float(np.std(train_macro_f1s, ddof=0)),
        "train_top1_accuracy_mean": float(np.mean(train_top1s)),
        "train_log_loss_mean": float(np.mean(train_lls)),
        "macro_f1_generalization_gap": float(np.mean(train_macro_f1s) - np.mean(val_macro_f1s)),
        "top1_generalization_gap": float(np.mean(train_top1s) - np.mean(val_top1s)),
        "total_cv_fit_time_sec": float(total_fit_sec),
        "total_cv_val_inference_time_sec": float(total_val_infer_sec),
        "oof_metrics": oof_overall,
        "folds": fold_records,
        "oof_predictions": oof_df,
    }


def format_prediction_df(
    df: pd.DataFrame,
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    prob_col: str = "pred_prob",
) -> pd.DataFrame:
    """Create a standardized prediction DataFrame matching the shared contract schema:
    `['trip_id', 'obs_idx', 'arc_id', 'is_match', 'pred_prob']`.
    """
    required_base = ["trip_id", "obs_idx", "arc_id", "is_match"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build prediction DataFrame; missing columns: {missing}")

    probs = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(probs) != len(df):
        raise ValueError(f"y_prob length ({len(probs)}) does not match df length ({len(df)}).")

    pred_df = pd.DataFrame(
        {
            "trip_id": df["trip_id"].astype(str).to_numpy(),
            "obs_idx": df["obs_idx"].astype(np.int64).to_numpy(),
            "arc_id": df["arc_id"].astype(np.int64).to_numpy(),
            "is_match": df["is_match"].astype(np.int64).to_numpy(),
            prob_col: probs,
        }
    )
    validate_prediction_dataframe(pred_df)
    return pred_df


def save_prediction_parquet(
    df: pd.DataFrame,
    y_prob: Optional[Union[np.ndarray, pd.Series, Sequence[float]]],
    output_path: Union[str, Path],
) -> Path:
    """Validate and save candidate predictions to a Parquet file following the shared contract."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if y_prob is not None:
        pred_df = format_prediction_df(df, y_prob)
    else:
        pred_df = df[PREDICTION_COLUMNS].copy()
        validate_prediction_dataframe(pred_df)

    pred_df.to_parquet(out_path, index=False)
    return out_path
