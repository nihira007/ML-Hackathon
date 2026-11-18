"""Phase 5 - verification of Person 1's data / feature pipeline.

    python -m pytest src/features/test_data_pipeline.py -q        # or:  python src/features/test_data_pipeline.py

Requires ``python -m src.features.build_features`` to have been run (parquets + artifacts exist).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import REPO_ROOT, TEST_DIR, TRAIN_DIR, list_trips, load_trip  # noqa: E402
from src.features.inference_features import extract_trip_features  # noqa: E402
from src.features.table_io import load_table  # noqa: E402

ART = REPO_ROOT / "artifacts"
PROC = REPO_ROOT / "data" / "processed"
PREPROC, COLS_JSON = ART / "preprocessor.joblib", ART / "feature_columns.json"
ID_COLS = ["trip_id", "obs_idx", "timestamp", "arc_id", "cand_rank"]


@pytest.fixture(scope="module")
def cols():
    return json.loads(COLS_JSON.read_text())


@pytest.fixture(scope="module")
def train_df():
    return load_table(PROC / "train_features.parquet")


@pytest.fixture(scope="module")
def test_df():
    return load_table(PROC / "student_test_features.parquet")


# ---------------------------------------------------------------- counts / leakage
def test_trip_counts_and_no_overlap(train_df, test_df):
    tr = {d.name for d in list_trips(TRAIN_DIR)}
    te = {d.name for d in list_trips(TEST_DIR)}
    assert len(tr) == 80 and len(te) == 10
    assert tr.isdisjoint(te)
    assert set(train_df.trip_id) <= tr and set(test_df.trip_id) == te
    assert set(train_df.trip_id).isdisjoint(set(test_df.trip_id))


def test_preprocessor_fit_on_train_only(train_df, cols):
    import joblib
    pre = joblib.load(PREPROC)
    audit = json.loads((REPO_ROOT / "outputs" / "eda" / "leakage_audit.json").read_text())
    assert audit["passed"] and audit["overlap_count"] == 0
    assert int(pre.named_steps["scaler"].n_samples_seen_) == len(train_df)   # exactly the train rows, nothing else
    # train features are standardised by that scaler -> ~zero mean / unit std on train, NOT on test
    assert np.allclose(train_df[cols].mean().to_numpy(), 0, atol=1e-2)


# ---------------------------------------------------------------- contract / finiteness
def test_schema_and_order(train_df, test_df, cols):
    for df in (train_df, test_df):
        assert list(df.columns) == ID_COLS + ["is_match", "label_ok"] + cols
        assert df.is_match.isin([0, 1]).all()
        assert (df.groupby(["trip_id", "obs_idx"]).is_match.sum() <= 1).all()


def test_no_nan_or_inf(train_df, test_df, cols):
    for name, df in (("train", train_df), ("student_test", test_df)):
        X = df[cols].to_numpy(dtype=np.float64)
        assert np.isfinite(X).all(), f"non-finite values in {name} features"


def test_every_observation_has_candidates(test_df):
    for tid, g in test_df.groupby("trip_id"):
        n_obs = len(np.loadtxt(TEST_DIR / tid / f"{tid}.track"))
        assert g.obs_idx.nunique() == n_obs


# ---------------------------------------------------------------- inference path
@pytest.mark.parametrize("trip", [d.name for d in list_trips(TEST_DIR)])
def test_extract_without_route_matches_parquet(trip, tmp_path, test_df, cols):
    src = TEST_DIR / trip
    dst = tmp_path / trip
    dst.mkdir()
    for ext in ("track", "nodes", "arcs"):                       # .route deliberately NOT copied
        shutil.copy(src / f"{trip}.{ext}", dst / f"{trip}.{ext}")
    assert not (dst / f"{trip}.route").exists()
    out = extract_trip_features(str(dst), str(PREPROC), str(COLS_JSON))
    ref = test_df[test_df.trip_id == trip].reset_index(drop=True)
    assert list(out.columns) == ID_COLS + cols
    assert len(out) == len(ref)
    for c in ID_COLS:
        assert (out[c].to_numpy() == ref[c].to_numpy()).all(), c
    assert np.array_equal(out[cols].to_numpy(), ref[cols].to_numpy()), "features differ from student_test_features"
    assert np.isfinite(out[cols].to_numpy(dtype=np.float64)).all()


def test_loader_never_opens_route(tmp_path):
    trip = list_trips(TEST_DIR)[0]
    t = load_trip(trip)                                          # default: load_route=False
    assert t.route is None
    assert load_trip(trip, load_route=True).route is not None


# ---------------------------------------------------------------- fallbacks
def _write_trip(d: Path, track, nodes, arcs):
    d.mkdir(parents=True)
    tid = d.name
    np.savetxt(d / f"{tid}.track", np.asarray(track), fmt="%.6f")
    (d / f"{tid}.nodes").write_text("\n".join(f"{a:.6f} {b:.6f}" for a, b in nodes) + ("\n" if nodes else ""))
    (d / f"{tid}.arcs").write_text("\n".join(f"{a} {b}" for a, b in arcs) + ("\n" if arcs else ""))


def test_outlier_gps_gets_fallback_candidates(tmp_path, cols):
    nodes = [(0.0, 51.0), (0.001, 51.0), (0.002, 51.0), (0.003, 51.0)]
    arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    track = [(0.0005, 51.00001, 1.0), (0.0015, 51.00001, 2.0), (0.5, 51.5, 3.0), (0.0025, 51.00001, 4.0)]   # row 2: ~60 km away
    d = tmp_path / "T0001"
    _write_trip(d, track, nodes, arcs)
    out = extract_trip_features(str(d), str(PREPROC), str(COLS_JSON))
    assert out.obs_idx.nunique() == 4
    assert (out.arc_id >= 0).all()
    assert np.isfinite(out[cols].to_numpy(dtype=np.float64)).all()
    assert (out[out.obs_idx == 2].shape[0]) >= 1


def test_empty_graph_gives_explicit_marker(tmp_path, cols):
    d = tmp_path / "T0002"
    _write_trip(d, [(0.0, 51.0, 1.0), (0.0001, 51.0, 2.0), (0.0002, 51.0, 3.0)], [], [])
    out = extract_trip_features(str(d), str(PREPROC), str(COLS_JSON))
    assert len(out) == 3 and (out.arc_id == -1).all()
    assert np.isfinite(out[cols].to_numpy(dtype=np.float64)).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
