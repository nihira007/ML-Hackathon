"""Phases 2 + 3 - candidate feature pipeline (v1 baseline and v2 kinematic/topological).

    python -m src.features.build_features                 # v2 (final)  -> parquets + artifacts
    python -m src.features.build_features --feature-set v1  # Phase 2 baseline (7 spatial features)

One row per ``(trip_id, obs_idx, candidate arc)``. Output contract (see prompts/00_*):

    trip_id, obs_idx, timestamp, arc_id, cand_rank, is_match, label_ok, <feature columns...>

* feature columns are in the exact order stored in ``artifacts/feature_columns.json``;
* feature values in the parquets are the OUTPUT OF THE PREPROCESSOR (median imputer + StandardScaler)
  fitted strictly on the 80 training trips -> ``artifacts/preprocessor.joblib``;
* ``is_match`` = 1 iff the candidate is the ground-truth arc aligned to that GPS row;
* ``label_ok`` = 1 iff the aligned ground-truth arc lies within ``LABEL_MAX_DIST_M`` of the fix. Rows
  with ``label_ok = 0`` have an unreliable label (the vehicle was far from the recorded route, e.g. trip
  00000033). They are DROPPED from train and kept (flagged) in student_test so evaluation is not edited.

Only ``.track``, ``.nodes`` and ``.arcs`` are used for features; ``.route`` is used solely for labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import REPO_ROOT, TEST_DIR, TRAIN_DIR, Trip, list_trips, load_trip  # noqa: E402
from src.data.spatial import ENUProjector, angular_diff_deg, bearing_deg, segment_bearing  # noqa: E402
from src.features.candidates import (K_DEFAULT, K_MAX, RADIUS_M, ArcIndex, align_ground_truth,  # noqa: E402
                                     generate_candidates, gt_rank, truncate)
from src.features.table_io import save_table  # noqa: E402

LABEL_MAX_DIST_M = 60.0     # alignment distance above which a label is considered unreliable
LINK_S = 10.0               # fixes further apart than this are not "linked" for speed / heading
GAP_FLAG_S = 2.0            # dt above this -> gps_gap_flag
ID_COLS = ["trip_id", "obs_idx", "timestamp", "arc_id", "cand_rank"]

V1_FEATURES = ["perp_dist_m", "dist_to_start_m", "dist_to_end_m", "proj_t", "arc_len_m", "dist_rank",
               "dist_diff_from_min_m"]
V2_EXTRA = [
    # kinematic
    "speed_mps", "dt_s", "accel_mps2", "gps_gap_flag", "bearing_diff_deg", "cos_bearing_diff",
    # relative / context
    "is_fallback_cand", "dist_ratio_to_min", "dist_zscore", "n_cand_within_15m", "n_cand_within_30m",
    "same_as_prev_nearest", "same_as_next_nearest", "connected_prev_nearest", "connected_next_nearest",
    # topological
    "from_out_degree", "to_out_degree", "to_in_degree", "has_reverse_arc", "local_road_density",
    # multi-step trajectory & margin context (zero-leakage)
    "bearing_diff_w5_deg", "cos_bearing_diff_w5",
    "connected_prev2_nearest", "connected_next2_nearest",
    "cand_freq_window5", "perp_dist_ratio_to_2nd",
]
V2_FEATURES = V1_FEATURES + V2_EXTRA
FEATURE_SETS = {"v1": V1_FEATURES, "v2": V2_FEATURES}


# ----------------------------------------------------------------------- observation-level kinematics
def kinematics(P: np.ndarray, ts: np.ndarray) -> Dict[str, np.ndarray]:
    """Speed / dt / acceleration / smoothed GPS heading per fix. Fixes > LINK_S apart are not linked."""
    n = len(P)
    idx = np.arange(n)
    dtp = np.r_[np.nan, np.diff(ts)]                       # dt to previous fix
    dtn = np.r_[np.diff(ts), np.nan]                       # dt to next fix
    p = np.where(np.nan_to_num(dtp, nan=1e9) <= LINK_S, idx - 1, idx)
    q = np.where(np.nan_to_num(dtn, nan=1e9) <= LINK_S, idx + 1, idx)
    span = ts[q] - ts[p]
    disp = np.linalg.norm(P[q] - P[p], axis=1)
    speed = np.where(span > 0, disp / np.maximum(span, 1e-6), np.nan)            # central difference
    # smoothed heading: one more fix on each side where linked
    p2 = np.where((p < idx) & (p > 0) & (ts[p] - ts[np.maximum(p - 1, 0)] <= LINK_S), p - 1, p)
    q2 = np.where((q > idx) & (q < n - 1) & (ts[np.minimum(q + 1, n - 1)] - ts[q] <= LINK_S), q + 1, q)
    d = P[q2] - P[p2]
    moved = np.linalg.norm(d, axis=1) >= 2.0                                       # heading undefined when ~stationary
    heading = np.where(moved, bearing_deg(d[:, 0], d[:, 1]), np.nan)

    # wider +/-5 step smoothed heading for low-speed / noisy turns
    p5 = p2.copy()
    q5 = q2.copy()
    for _ in range(3):
        p5 = np.where((p5 > 0) & (ts[idx] - ts[np.maximum(p5 - 1, 0)] <= 12.0), p5 - 1, p5)
        q5 = np.where((q5 < n - 1) & (ts[np.minimum(q5 + 1, n - 1)] - ts[idx] <= 12.0), q5 + 1, q5)
    d5 = P[q5] - P[p5]
    moved5 = np.linalg.norm(d5, axis=1) >= 3.0
    heading_w5 = np.where(moved5, bearing_deg(d5[:, 0], d5[:, 1]), heading)

    prev_speed = np.r_[np.nan, speed[:-1]]
    linked_prev = np.nan_to_num(dtp, nan=1e9) <= LINK_S
    accel = np.where(linked_prev, (speed - prev_speed) / np.maximum(np.nan_to_num(dtp, nan=1.0), 1e-6), np.nan)
    dt_first = dtn[0] if n > 1 else 1.0
    dt_s = np.where(np.isnan(dtp), dt_first, dtp)
    return {
        "speed_mps": np.clip(speed, 0, 70.0),
        "dt_s": np.clip(dt_s, 0, 60.0),
        "accel_mps2": np.clip(accel, -10.0, 10.0),
        "gps_gap_flag": (np.nan_to_num(dtp, nan=0.0) > GAP_FLAG_S).astype(np.float64),
        "heading": heading,
        "heading_w5": heading_w5,
        "dt_raw": np.nan_to_num(dt_s, nan=1.0),
    }


# ----------------------------------------------------------------------- per-trip feature computation
def compute_trip_features(trip: Trip, with_labels: bool, feature_set: str = "v2", k: int = K_DEFAULT,
                          radius: float = RADIUS_M, k_eval: Optional[int] = None) -> tuple[pd.DataFrame, dict]:
    """Raw (un-preprocessed) candidate feature table for ONE trip + a stats dict.

    ``with_labels=True`` needs ``trip.route`` (training / development only); ``False`` is the inference path
    and never touches the route. ``k_eval`` (> k) additionally evaluates the miss-rate sensitivity.
    """
    cols = FEATURE_SETS[feature_set]
    proj = ENUProjector.from_track(trip.track)
    P = proj.project(trip.track[:, :2])
    node_xy = proj.project(trip.nodes) if len(trip.nodes) else np.empty((0, 2))
    ts = trip.track[:, 2]
    n = len(P)
    index = ArcIndex(node_xy, trip.arcs)
    cand_full = generate_candidates(index, P, max(k, k_eval or 0), radius)
    cand = truncate(cand_full, k)
    stats: dict = {"trip_id": trip.trip_id, "n_obs": n, "n_cand_rows": int(len(cand["obs_idx"]))}

    o = cand["obs_idx"].astype(np.int64)
    arc = cand["arc_id"].astype(np.int64)
    perp = cand["perp_dist"]
    M = len(o)
    kin = kinematics(P, ts)
    empty = index.n_arcs == 0
    a_safe = np.where(arc >= 0, arc, 0) if not empty else arc

    f: Dict[str, np.ndarray] = {}
    starts = np.flatnonzero(cand["cand_rank"] == 0)             # first (nearest) candidate of each obs
    counts = np.diff(np.r_[starts, M])
    if not empty:
        A, B = index.A[a_safe], index.B[a_safe]
        dmin = np.repeat(perp[starts], counts)
        f["perp_dist_m"] = perp
        f["dist_to_start_m"] = np.linalg.norm(P[o] - A, axis=1)
        f["dist_to_end_m"] = np.linalg.norm(P[o] - B, axis=1)
        f["proj_t"] = cand["t"]
        f["arc_len_m"] = index.length[a_safe]
        f["dist_rank"] = cand["cand_rank"].astype(np.float64)
        f["dist_diff_from_min_m"] = perp - dmin
        if feature_set == "v2":
            f["is_fallback_cand"] = cand["is_fallback"].astype(np.float64)
            f["dist_ratio_to_min"] = (perp + 1.0) / (dmin + 1.0)
            mean = np.repeat(np.add.reduceat(perp, starts) / counts, counts)
            var = np.repeat(np.add.reduceat((perp - mean) ** 2, starts) / counts, counts)
            f["dist_zscore"] = np.where(var > 1e-9, (perp - mean) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
            f["n_cand_within_15m"] = np.repeat(np.add.reduceat((perp <= 15).astype(np.float64), starts), counts)
            f["n_cand_within_30m"] = np.repeat(np.add.reduceat((perp <= 30).astype(np.float64), starts), counts)
            near = arc[starts]                                       # nearest arc per observation (N,)
            prev_near = np.r_[-1, near[:-1]]
            next_near = np.r_[near[1:], -1]
            pn, nn = prev_near[o], next_near[o]
            frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
            same_p, same_n = arc == pn, arc == nn
            f["same_as_prev_nearest"] = same_p.astype(np.float64)
            f["same_as_next_nearest"] = same_n.astype(np.float64)
            f["connected_prev_nearest"] = (same_p | ((pn >= 0) & (frm[a_safe] == to[np.maximum(pn, 0)]))).astype(np.float64)
            f["connected_next_nearest"] = (same_n | ((nn >= 0) & (to[a_safe] == frm[np.maximum(nn, 0)]))).astype(np.float64)
            # bearings
            arc_b = segment_bearing(A, B)
            diff = angular_diff_deg(kin["heading"][o], arc_b)       # NaN when heading undefined
            f["bearing_diff_deg"] = diff
            f["cos_bearing_diff"] = np.cos(np.radians(diff))
            # topology
            n_nodes = len(trip.nodes)
            out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes).astype(np.float64)
            in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes).astype(np.float64)
            f["from_out_degree"] = out_deg[frm[a_safe]]
            f["to_out_degree"] = out_deg[to[a_safe]]
            f["to_in_degree"] = in_deg[to[a_safe]]
            keys = frm.astype(np.int64) * n_nodes + to
            rev = to[a_safe].astype(np.int64) * n_nodes + frm[a_safe]
            f["has_reverse_arc"] = np.isin(rev, keys).astype(np.float64)
            uniq, inv = np.unique(a_safe, return_inverse=True)
            f["local_road_density"] = index.road_density(uniq)[inv]
            for key in ("speed_mps", "dt_s", "accel_mps2", "gps_gap_flag"):
                f[key] = kin[key][o]

            # multi-step trajectory & margin context
            diff_w5 = angular_diff_deg(kin["heading_w5"][o], arc_b)
            f["bearing_diff_w5_deg"] = diff_w5
            f["cos_bearing_diff_w5"] = np.cos(np.radians(diff_w5))

            prev2_near = np.r_[-1, -1, near[:-2]] if n >= 2 else np.full(n, -1, dtype=np.int64)
            next2_near = np.r_[near[2:], -1, -1] if n >= 2 else np.full(n, -1, dtype=np.int64)
            p2n, n2n = prev2_near[o], next2_near[o]
            f["connected_prev2_nearest"] = (
                (arc == p2n) | ((p2n >= 0) & (frm[a_safe] == to[np.maximum(p2n, 0)]))
            ).astype(np.float64)
            f["connected_next2_nearest"] = (
                (arc == n2n) | ((n2n >= 0) & (to[a_safe] == frm[np.maximum(n2n, 0)]))
            ).astype(np.float64)

            second_idx = np.where(counts > 1, starts + 1, starts)
            d2nd = np.repeat(perp[second_idx], counts)
            f["perp_dist_ratio_to_2nd"] = (perp + 1.0) / (d2nd + 1.0)

            top3_mask = cand["cand_rank"] < 3
            top3_arcs_per_obs: List[set[int]] = [set() for _ in range(n)]
            for oi, ai in zip(o[top3_mask], arc[top3_mask]):
                if ai >= 0:
                    top3_arcs_per_obs[int(oi)].add(int(ai))
            freq_w5 = np.zeros(M, dtype=np.float64)
            for row_i in range(M):
                oi = int(o[row_i])
                ai = int(arc[row_i])
                if ai >= 0:
                    lo_w = max(0, oi - 2)
                    hi_w = min(n, oi + 3)
                    hits = sum(1 for w_i in range(lo_w, hi_w) if ai in top3_arcs_per_obs[w_i])
                    freq_w5[row_i] = hits / float(hi_w - lo_w)
            f["cand_freq_window5"] = freq_w5
    else:                                                            # no road graph at all: explicit empty marker
        for c in cols:
            f[c] = np.full(M, np.nan)
        if feature_set == "v2":
            for key in ("speed_mps", "dt_s", "accel_mps2", "gps_gap_flag"):
                f[key] = kin[key][o]

    df = pd.DataFrame({
        "trip_id": np.full(M, trip.trip_id, dtype=object),
        "obs_idx": o.astype(np.int32), "timestamp": ts[o].astype(np.float64),
        "arc_id": arc.astype(np.int32), "cand_rank": cand["cand_rank"].astype(np.int32),
    })
    if with_labels:
        if trip.route is None or len(trip.route) == 0:
            raise ValueError(f"trip {trip.trip_id}: labels requested but no .route available")
        dt_obs = np.r_[0.0, np.diff(ts)]
        al = align_ground_truth(P, node_xy, trip.arcs, trip.route, dt_obs)
        df["is_match"] = (arc == al.arc_id[o]).astype(np.int8)
        df["label_ok"] = (al.dist[o] <= LABEL_MAX_DIST_M).astype(np.int8)
        rk = gt_rank(cand_full, al.arc_id, n)
        ok = al.dist <= LABEL_MAX_DIST_M
        stats.update({
            "n_obs_label_ok": int(ok.sum()),
            "align_dist_p50_m": float(np.median(al.dist)), "align_dist_p95_m": float(np.percentile(al.dist, 95)),
            "align_dist_max_m": float(al.dist.max()),
            "route_arcs": int(len(trip.route)), "route_arcs_with_obs": int(len(np.unique(al.route_pos))),
            "_rank": rk, "_ok": ok,
        })
    for c in cols:
        df[c] = f[c].astype(np.float32)
    return df, stats


# ----------------------------------------------------------------------- preprocessing
def make_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def apply_preprocessor(df: pd.DataFrame, preproc: Pipeline, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = preproc.transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32)
    return df


# ----------------------------------------------------------------------- partition build + reports
def build_partition(part: str, pdir: Path, feature_set: str, k: int, radius: float, limit: Optional[int]):
    dirs = list_trips(pdir)[:limit]
    frames, stats = [], []
    t0 = time.time()
    for i, d in enumerate(dirs):
        trip = load_trip(d, load_route=True)
        df, st = compute_trip_features(trip, True, feature_set, k, radius, k_eval=K_MAX)
        frames.append(df); stats.append(st)
        if (i + 1) % 10 == 0 or i + 1 == len(dirs):
            print(f"  [{part}] {i + 1}/{len(dirs)} trips  ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True), stats


def miss_rate_tables(stats_by_part: Dict[str, List[dict]], k: int, out_dir: Path) -> None:
    rows, sens = [], []
    for part, stats in stats_by_part.items():
        for st in stats:
            rk, ok = st["_rank"], st["_ok"]
            miss = rk >= k
            rows.append({"partition": part, "trip_id": st["trip_id"], "n_obs": st["n_obs"], "n_obs_label_ok": st["n_obs_label_ok"],
                         "n_missed": int(miss.sum()), "miss_rate_pct": 100 * miss.mean(),
                         "n_missed_label_ok": int((miss & ok).sum()),
                         "miss_rate_pct_label_ok": 100 * (miss & ok).sum() / max(int(ok.sum()), 1)})
        allrk = np.concatenate([s["_rank"] for s in stats]); allok = np.concatenate([s["_ok"] for s in stats])
        miss = allrk >= k
        rows.append({"partition": part, "trip_id": "ALL", "n_obs": len(allrk), "n_obs_label_ok": int(allok.sum()),
                     "n_missed": int(miss.sum()), "miss_rate_pct": 100 * miss.mean(),
                     "n_missed_label_ok": int((miss & allok).sum()),
                     "miss_rate_pct_label_ok": 100 * (miss & allok).sum() / max(int(allok.sum()), 1)})
        for kk in (1, 3, 5, 10, 15, 20):
            m = allrk >= kk
            sens.append({"partition": part, "K": kk, "radius_m": RADIUS_M, "miss_rate_pct_all_obs": 100 * m.mean(),
                         "miss_rate_pct_label_ok": 100 * (m & allok).sum() / max(int(allok.sum()), 1)})
    pd.DataFrame(rows).to_csv(out_dir / "candidate_miss_rate.csv", index=False)
    pd.DataFrame(sens).to_csv(out_dir / "candidate_miss_rate_by_K.csv", index=False)
    pd.DataFrame([{k_: v for k_, v in s.items() if not k_.startswith("_")} | {"partition": p}
                  for p, ss in stats_by_part.items() for s in ss]).to_csv(out_dir / "alignment_quality.csv", index=False)


def leakage_audit(train_dirs, test_dirs, train_raw: pd.DataFrame, preproc: Pipeline, cols: List[str], out_dir: Path) -> dict:
    tr_ids = {d.name for d in train_dirs}; te_ids = {d.name for d in test_dirs}
    refit = make_preprocessor().fit(train_raw[cols].to_numpy(dtype=np.float32))
    sc, rc = preproc.named_steps["scaler"], refit.named_steps["scaler"]
    audit = {
        "n_train_trips": len(tr_ids), "n_student_test_trips": len(te_ids),
        "trip_id_overlap": sorted(tr_ids & te_ids), "overlap_count": len(tr_ids & te_ids),
        "preprocessor_fit_rows": int(sc.n_samples_seen_),
        "train_rows_available_for_fit": int(len(train_raw)),
        "preprocessor_fit_on_train_only": bool(int(sc.n_samples_seen_) == len(train_raw)
                                              and np.allclose(sc.mean_, rc.mean_) and np.allclose(sc.scale_, rc.scale_)),
        "train_trip_ids_in_fit": int(train_raw.trip_id.nunique()),
        "test_trip_ids_in_fit": int(train_raw.trip_id.isin(te_ids).sum()),
        "route_used_for_features": False,
    }
    audit["passed"] = bool(audit["overlap_count"] == 0 and audit["preprocessor_fit_on_train_only"] and audit["test_trip_ids_in_fit"] == 0)
    (out_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def feature_plots(raw: pd.DataFrame, cols: List[str], out_dir: Path, max_rows: int = 300_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = out_dir / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    s = raw.sample(min(max_rows, len(raw)), random_state=0)
    X = s[cols].astype(np.float64)
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(max(8, .42 * len(cols) + 3), max(7, .4 * len(cols) + 2)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.04); ax.set_title("Feature correlation (train, raw features)")
    fig.tight_layout(); fig.savefig(pdir / "feature_correlation.png", dpi=130); plt.close(fig)
    # point-biserial style correlation with the label
    lab = s["is_match"].to_numpy(dtype=float)
    cl = pd.Series({c: np.corrcoef(np.nan_to_num(X[c].to_numpy()), lab)[0, 1] if X[c].std() > 0 else 0.0 for c in cols}).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, .27 * len(cols))))
    ax.barh(cl.index, cl.values, color=np.where(cl.values >= 0, "#2ca02c", "#d62728")); ax.set_xlabel("corr with is_match")
    ax.set_title("Feature vs. label correlation (train)"); fig.tight_layout(); fig.savefig(pdir / "feature_label_correlation.png", dpi=130); plt.close(fig)
    nc = 5; nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.4 * nr))
    for ax, c in zip(axes.ravel(), cols):
        v = X[c].dropna(); lo, hi = np.percentile(v, [0.5, 99.5]) if len(v) else (0, 1)
        if hi <= lo: hi = lo + 1
        bins = np.linspace(lo, hi, 40)
        ax.hist(v[s.loc[v.index, "is_match"] == 0], bins=bins, alpha=.6, density=True, label="non-match", color="#7f7f7f")
        ax.hist(v[s.loc[v.index, "is_match"] == 1], bins=bins, alpha=.6, density=True, label="match", color="#1f77b4")
        ax.set_title(c, fontsize=8); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle("Feature distributions by label (train, raw, 0.5-99.5 pctl)", y=1.0)
    fig.tight_layout(); fig.savefig(pdir / "feature_distributions.png", dpi=110); plt.close(fig)
    desc = X.describe().T; desc["nan_frac"] = X.isna().mean()
    desc.to_csv(out_dir / "feature_summary.csv")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="v2")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--radius", type=float, default=RADIUS_M)
    ap.add_argument("--limit", type=int, default=None, help="only first N trips per partition (debug)")
    ap.add_argument("--out-suffix", default="", help="e.g. _v1 to avoid overwriting the final v2 files")
    args = ap.parse_args(argv)
    cols = FEATURE_SETS[args.feature_set]
    out_dir = REPO_ROOT / "outputs" / "eda"; out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = REPO_ROOT / "data" / "processed"; art_dir = REPO_ROOT / "artifacts"; art_dir.mkdir(exist_ok=True)
    sfx = args.out_suffix

    print(f"feature set {args.feature_set} ({len(cols)} features), K={args.k}, radius={args.radius} m")
    train_raw, tr_stats = build_partition("train", TRAIN_DIR, args.feature_set, args.k, args.radius, args.limit)
    test_raw, te_stats = build_partition("student_test", TEST_DIR, args.feature_set, args.k, args.radius, args.limit)

    miss_rate_tables({"train": tr_stats, "student_test": te_stats}, args.k, out_dir)

    n_before = train_raw.obs_idx.size
    train_fit = train_raw[train_raw.label_ok == 1].reset_index(drop=True)          # drop unreliable training labels
    print(f"train rows {n_before:,} -> {len(train_fit):,} after dropping unreliable-label observations")

    preproc = make_preprocessor().fit(train_fit[cols].to_numpy(dtype=np.float32))   # TRAIN ONLY
    audit = leakage_audit(list_trips(TRAIN_DIR)[:args.limit], list_trips(TEST_DIR)[:args.limit], train_fit, preproc, cols, out_dir)
    print("leakage audit:", {k_: audit[k_] for k_ in ("overlap_count", "preprocessor_fit_on_train_only", "passed")})
    assert audit["passed"], "leakage audit failed"

    feature_plots(train_fit, cols, out_dir)
    train_out = apply_preprocessor(train_fit, preproc, cols)
    test_out = apply_preprocessor(test_raw, preproc, cols)
    lead = ID_COLS + ["is_match", "label_ok"]
    p1 = save_table(train_out[lead + cols], proc_dir / f"train_features{sfx}.parquet")
    p2 = save_table(test_out[lead + cols], proc_dir / f"student_test_features{sfx}.parquet")
    joblib.dump(preproc, art_dir / f"preprocessor{sfx}.joblib")
    (art_dir / f"feature_columns{sfx}.json").write_text(json.dumps(cols, indent=2))
    scaler = preproc.named_steps["scaler"]
    (out_dir / f"preprocessor_meta{sfx}.json").write_text(json.dumps({
        "feature_set": args.feature_set, "K": args.k, "radius_m": args.radius, "label_max_dist_m": LABEL_MAX_DIST_M,
        "fit_rows": int(scaler.n_samples_seen_), "fit_trips": int(train_fit.trip_id.nunique()),
        "imputer": "median", "scaler": "StandardScaler", "feature_columns": cols,
        "train_rows_dropped_unreliable_label": int(n_before - len(train_fit)),
    }, indent=2))
    print("wrote", p1.name, p2.name, f"(train {len(train_out):,} rows, test {len(test_out):,} rows)")


if __name__ == "__main__":
    main()
