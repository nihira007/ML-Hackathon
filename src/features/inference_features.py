"""Phase 4 - inference-ready single-trip feature extractor.

    from src.features.inference_features import extract_trip_features
    X_df = extract_trip_features(trip_dir, "artifacts/preprocessor.joblib", "artifacts/feature_columns.json")
    probs = model.predict_proba(X_df[feature_cols])[:, 1]

Reads ONLY ``<id>.track``, ``<id>.nodes`` and ``<id>.arcs`` - ``.route`` is never opened, listed or required
(``load_trip(..., load_route=False)``). The returned frame has the identifier columns
``trip_id, obs_idx, timestamp, arc_id, cand_rank`` followed by the feature columns in EXACTLY the order of
``feature_columns.json``, already passed through the train-fitted preprocessor (float32), i.e. identical in
layout to ``data/processed/*_features.parquet`` (minus the label columns ``is_match`` / ``label_ok``).

Robustness (every observation gets at least one row):
* GPS rows with NaN / out-of-range coordinates are linearly interpolated by the loader (obs_idx stays aligned);
* an observation with no arc within the search radius (outlier / sparse area) falls back to its exact nearest
  K arcs at any distance, flagged ``is_fallback_cand = 1``;
* a trip whose road graph has no arcs at all yields one explicit marker row per observation with
  ``arc_id = -1`` (features imputed by the preprocessor) so the HMM can treat it as "no candidate";
* time gaps and ~stationary fixes produce NaN speed / heading which the (train-fitted) imputer fills.

CLI benchmark of all student_test trips:  python -m src.features.inference_features --benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import REPO_ROOT, TEST_DIR, list_trips, load_trip  # noqa: E402
from src.features.build_features import ID_COLS, apply_preprocessor, compute_trip_features  # noqa: E402
from src.features.candidates import K_DEFAULT, RADIUS_M  # noqa: E402


def extract_trip_features(trip_dir: str, preprocessor_path: str, feature_cols_path: str) -> pd.DataFrame:
    """Candidate feature frame for one trip, ready for ``model.predict_proba`` and the HMM decoder."""
    cols: List[str] = json.loads(Path(feature_cols_path).read_text())
    preproc = joblib.load(preprocessor_path)
    trip = load_trip(trip_dir, load_route=False)                         # .route is never read
    feature_set = "v2" if len(cols) > 7 else "v1"
    raw, _ = compute_trip_features(trip, with_labels=False, feature_set=feature_set, k=K_DEFAULT, radius=RADIUS_M)
    out = apply_preprocessor(raw, preproc, cols)
    return out[ID_COLS + cols].reset_index(drop=True)


def benchmark(out_csv: Path, artifacts: Path) -> pd.DataFrame:
    rows = []
    for d in list_trips(TEST_DIR):
        t0 = time.perf_counter()
        df = extract_trip_features(str(d), str(artifacts / "preprocessor.joblib"), str(artifacts / "feature_columns.json"))
        dt = time.perf_counter() - t0
        n_obs = int(df.obs_idx.nunique())
        rows.append({"trip_id": d.name, "n_obs": n_obs, "n_candidate_rows": len(df),
                     "n_obs_without_candidate": int(n_obs - df.loc[df.arc_id >= 0, "obs_idx"].nunique()),
                     "seconds_total": round(dt, 3), "ms_per_obs": round(1000 * dt / max(n_obs, 1), 3)})
        print(f"{d.name}: {n_obs} obs -> {len(df)} rows in {dt:.2f}s")
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"trip_id": "ALL", "n_obs": res.n_obs.sum(), "n_candidate_rows": res.n_candidate_rows.sum(),
                         "n_obs_without_candidate": res.n_obs_without_candidate.sum(),
                         "seconds_total": round(res.seconds_total.sum(), 3),
                         "ms_per_obs": round(1000 * res.seconds_total.sum() / res.n_obs.sum(), 3)}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true", help="time all student_test trips -> outputs/eda/feature_extraction_timing.csv")
    ap.add_argument("--trip-dir"); ap.add_argument("--artifacts", default=str(REPO_ROOT / "artifacts"))
    a = ap.parse_args()
    if a.benchmark:
        benchmark(REPO_ROOT / "outputs" / "eda" / "feature_extraction_timing.csv", Path(a.artifacts))
    elif a.trip_dir:
        print(extract_trip_features(a.trip_dir, f"{a.artifacts}/preprocessor.joblib", f"{a.artifacts}/feature_columns.json").head())
    else:
        ap.print_help()
