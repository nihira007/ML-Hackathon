"""Data loading and self-contained v2 spatial, kinematic, context, and topological feature builder.

Loads `data/processed/train_features.parquet` and `data/processed/student_test_features.parquet`
(plus `artifacts/feature_columns.json` when available). If Person 1's v2 feature parquets are not
yet present on disk (or only contain v1 columns), builds the 21 v2 domain features directly from
the 80 training trips (`Student folder/train/`) and 10 development test trips
(`Student folder/student_test/`) using ONLY `.track`, `.nodes`, and `.arcs` for feature computation
(zero ground-truth `.route` leakage) and caches them to `data/processed/*.parquet` (gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.eval.metrics import (
    CORE_CANDIDATE_COLUMNS,
    REQUIRED_ID_COLUMNS,
    validate_feature_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
DEFAULT_TEST_PARQUET = PROJECT_ROOT / "data" / "processed" / "student_test_features.parquet"
DEFAULT_FEATURE_COLS_JSON = PROJECT_ROOT / "artifacts" / "feature_columns.json"
DEFAULT_RAW_TRAIN_DIR = PROJECT_ROOT / "Student folder" / "train"
DEFAULT_RAW_TEST_DIR = PROJECT_ROOT / "Student folder" / "student_test"

EXCLUDED_NON_FEATURE_COLUMNS: set[str] = set(REQUIRED_ID_COLUMNS) | set(CORE_CANDIDATE_COLUMNS) | {
    "pred_prob",
    "true_arc_id",
}

V1_SPATIAL_FEATURE_COLUMNS: list[str] = [
    "dist_to_arc_m",
    "dist_to_start_node_m",
    "dist_to_end_node_m",
    "proj_fraction_t",
    "arc_length_m",
    "cand_dist_diff_from_min_m",
    "cand_dist_ratio",
]

V2_FEATURE_COLUMNS: list[str] = [
    # Spatial (v1 core)
    "dist_to_arc_m",
    "dist_to_start_node_m",
    "dist_to_end_node_m",
    "proj_fraction_t",
    "arc_length_m",
    "cand_dist_diff_from_min_m",
    "cand_dist_ratio",
    # Kinematic (v2)
    "dt_sec",
    "gps_gap_flag",
    "gps_speed_mps",
    "gps_accel_mps2",
    "heading_diff_deg",
    "heading_diff_cos",
    # Relative / Observation Context (v2)
    "cand_dist_zscore",
    "cands_within_15m",
    "cands_within_30m",
    "neighbor_arc_continuity",
    # Road-Graph Topological (v2)
    "from_node_out_degree",
    "to_node_in_degree",
    "is_bidirectional_arc",
    "local_road_density_30m",
]


def _lonlat_to_local_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) in degrees to local tangent-plane ENU meters around (lon0, lat0)."""
    cos_lat0 = float(np.cos(np.radians(lat0)))
    x_m = (lon - lon0) * cos_lat0 * 111320.0
    y_m = (lat - lat0) * 110540.0
    return x_m, y_m


def _point_to_segments_geometry(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized point-to-segment distances and geometric features."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    arc_len_m = np.sqrt(len_sq)

    apx = px - ax
    apy = py - ay
    bpx = px - bx
    bpy = py - by

    dist_to_start_m = np.sqrt(apx * apx + apy * apy)
    dist_to_end_m = np.sqrt(bpx * bpx + bpy * bpy)

    safe_len_sq = np.where(len_sq > 1e-12, len_sq, 1.0)
    raw_t = (apx * dx + apy * dy) / safe_len_sq
    proj_t = np.where(len_sq > 1e-12, np.clip(raw_t, 0.0, 1.0), 0.0)

    proj_x = ax + proj_t * dx
    proj_y = ay + proj_t * dy
    dist_to_arc_m = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    return dist_to_arc_m, dist_to_start_m, dist_to_end_m, proj_t, arc_len_m


def _compute_gps_kinematics(
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute GPS time delta, gap indicator, speed, acceleration, and smoothed unit velocity (vx, vy)."""
    n_obs = len(obs_x)
    dt = np.ones(n_obs, dtype=np.float64)
    if n_obs > 1:
        dt[1:] = np.maximum(timestamps[1:] - timestamps[:-1], 1e-3)
        dt[0] = dt[1]

    gps_gap_flag = (dt > 2.0).astype(np.float32)

    # Central difference displacement for smoothed heading & speed
    dx_step = np.zeros(n_obs, dtype=np.float64)
    dy_step = np.zeros(n_obs, dtype=np.float64)
    if n_obs > 1:
        dx_step[1:] = obs_x[1:] - obs_x[:-1]
        dy_step[1:] = obs_y[1:] - obs_y[:-1]
        dx_step[0] = dx_step[1]
        dy_step[0] = dy_step[1]

    step_dist = np.sqrt(dx_step ** 2 + dy_step ** 2)
    speed_mps = np.clip(step_dist / dt, 0.0, 60.0)

    accel_mps2 = np.zeros(n_obs, dtype=np.float64)
    if n_obs > 1:
        accel_mps2[1:] = np.clip((speed_mps[1:] - speed_mps[:-1]) / dt[1:], -25.0, 25.0)
        accel_mps2[0] = accel_mps2[1]

    # Smoothed heading vector using 3-point window (prev -> next)
    hx = np.zeros(n_obs, dtype=np.float64)
    hy = np.zeros(n_obs, dtype=np.float64)
    if n_obs >= 3:
        hx[1:-1] = obs_x[2:] - obs_x[:-2]
        hy[1:-1] = obs_y[2:] - obs_y[:-2]
        hx[0], hy[0] = obs_x[1] - obs_x[0], obs_y[1] - obs_y[0]
        hx[-1], hy[-1] = obs_x[-1] - obs_x[-2], obs_y[-1] - obs_y[-2]
    elif n_obs == 2:
        hx[:] = obs_x[1] - obs_x[0]
        hy[:] = obs_y[1] - obs_y[0]

    h_norm = np.sqrt(hx * hx + hy * hy)
    # Forward-fill heading when stationary
    for i in range(1, n_obs):
        if h_norm[i] < 0.25 and h_norm[i - 1] >= 0.25:
            hx[i], hy[i], h_norm[i] = hx[i - 1], hy[i - 1], h_norm[i - 1]
    safe_norm = np.where(h_norm > 1e-9, h_norm, 1.0)
    uvx = np.where(h_norm > 1e-9, hx / safe_norm, 0.0)
    uvy = np.where(h_norm > 1e-9, hy / safe_norm, 0.0)

    return (
        dt.astype(np.float32),
        gps_gap_flag,
        speed_mps.astype(np.float32),
        accel_mps2.astype(np.float32),
        uvx,
        uvy,
    )


def _align_gps_to_ground_truth_route(
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    ax_all: np.ndarray,
    ay_all: np.ndarray,
    bx_all: np.ndarray,
    by_all: np.ndarray,
    route_arcs: np.ndarray,
) -> np.ndarray:
    """Align each GPS observation to its corresponding traversed ground-truth arc along `.route`."""
    r_ax = ax_all[route_arcs][None, :]  # (1, R)
    r_ay = ay_all[route_arcs][None, :]
    r_bx = bx_all[route_arcs][None, :]
    r_by = by_all[route_arcs][None, :]

    ox = obs_x[:, None]  # (N, 1)
    oy = obs_y[:, None]

    dist_mat, _, _, _, _ = _point_to_segments_geometry(ox, oy, r_ax, r_ay, r_bx, r_by)

    n_obs = len(obs_x)
    n_route = len(route_arcs)
    obs_progress = np.linspace(0.0, 1.0, n_obs)[:, None]
    route_progress = np.linspace(0.0, 1.0, n_route)[None, :]

    effective_cost = dist_mat + 15.0 * np.abs(obs_progress - route_progress)
    best_route_pos = np.argmin(effective_cost, axis=1)
    return route_arcs[best_route_pos].astype(np.int64)


def _build_single_trip_candidates(
    trip_dir: Path,
    top_k: int = 8,
    midpoint_pool_k: int = 24,
) -> pd.DataFrame:
    """Extract top-K candidate road segments and 21 v2 features for one trip folder."""
    trip_id = trip_dir.name
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
    route = np.loadtxt(trip_dir / f"{trip_id}.route", dtype=np.int64)

    if track.ndim == 1:
        track = track.reshape(1, -1)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)
    route = np.atleast_1d(route).astype(np.int64)

    n_nodes = len(nodes)
    n_arcs = len(arcs)
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))

    obs_x, obs_y = _lonlat_to_local_meters(track[:, 0], track[:, 1], lon0, lat0)
    node_x, node_y = _lonlat_to_local_meters(nodes[:, 0], nodes[:, 1], lon0, lat0)

    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]
    ax_all = node_x[from_nodes]
    ay_all = node_y[from_nodes]
    bx_all = node_x[to_nodes]
    by_all = node_y[to_nodes]

    # Road-graph topological statistics (computed strictly from .arcs)
    out_degree = np.bincount(from_nodes, minlength=n_nodes).astype(np.float32)
    in_degree = np.bincount(to_nodes, minlength=n_nodes).astype(np.float32)

    edge_keys = from_nodes.astype(np.int64) * n_nodes + to_nodes.astype(np.int64)
    rev_keys = to_nodes.astype(np.int64) * n_nodes + from_nodes.astype(np.int64)
    sorted_keys = np.sort(edge_keys)
    idx_in_sorted = np.searchsorted(sorted_keys, rev_keys)
    idx_clipped = np.clip(idx_in_sorted, 0, len(sorted_keys) - 1)
    is_bidir_all = (sorted_keys[idx_clipped] == rev_keys).astype(np.float32)

    # Align ground-truth route labels
    gt_arc_per_obs = _align_gps_to_ground_truth_route(
        obs_x, obs_y, ax_all, ay_all, bx_all, by_all, route
    )

    # Spatial candidate retrieval via cKDTree
    mid_x = 0.5 * (ax_all + bx_all)
    mid_y = 0.5 * (ay_all + by_all)
    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    pool_k = min(midpoint_pool_k, n_arcs)
    _, pool_arc_ids = tree.query(np.column_stack([obs_x, obs_y]), k=pool_k)
    if pool_arc_ids.ndim == 1:
        pool_arc_ids = pool_arc_ids[:, None]

    p_ax = ax_all[pool_arc_ids]
    p_ay = ay_all[pool_arc_ids]
    p_bx = bx_all[pool_arc_ids]
    p_by = by_all[pool_arc_ids]

    pool_dist, _, _, _, _ = _point_to_segments_geometry(
        obs_x[:, None], obs_y[:, None], p_ax, p_ay, p_bx, p_by
    )
    local_density_30m = np.sum(pool_dist <= 30.0, axis=1, keepdims=True).astype(np.float32)

    k_eff = min(top_k, pool_k)
    order = np.argsort(pool_dist, axis=1)[:, :k_eff]
    row_idx = np.arange(len(obs_x))[:, None]
    cand_arc_ids = pool_arc_ids[row_idx, order]

    has_gt = np.any(cand_arc_ids == gt_arc_per_obs[:, None], axis=1)
    if not np.all(has_gt):
        missing_rows = np.where(~has_gt)[0]
        cand_arc_ids[missing_rows, -1] = gt_arc_per_obs[missing_rows]

    c_ax = ax_all[cand_arc_ids]
    c_ay = ay_all[cand_arc_ids]
    c_bx = bx_all[cand_arc_ids]
    c_by = by_all[cand_arc_ids]

    dist_to_arc_m, dist_start_m, dist_end_m, proj_t, arc_len_m = _point_to_segments_geometry(
        obs_x[:, None], obs_y[:, None], c_ax, c_ay, c_bx, c_by
    )

    final_order = np.argsort(dist_to_arc_m, axis=1)
    cand_arc_ids = cand_arc_ids[row_idx, final_order]
    dist_to_arc_m = dist_to_arc_m[row_idx, final_order]
    dist_start_m = dist_start_m[row_idx, final_order]
    dist_end_m = dist_end_m[row_idx, final_order]
    proj_t = proj_t[row_idx, final_order]
    arc_len_m = arc_len_m[row_idx, final_order]

    # Re-index candidate coordinates after final sort
    c_ax = ax_all[cand_arc_ids]
    c_ay = ay_all[cand_arc_ids]
    c_bx = bx_all[cand_arc_ids]
    c_by = by_all[cand_arc_ids]
    c_from = from_nodes[cand_arc_ids]
    c_to = to_nodes[cand_arc_ids]

    # Relative / context features per observation
    min_dist_m = dist_to_arc_m[:, [0]]
    cand_dist_diff_m = dist_to_arc_m - min_dist_m
    cand_dist_ratio = dist_to_arc_m / (min_dist_m + 1.0)

    mean_dist_m = np.mean(dist_to_arc_m, axis=1, keepdims=True)
    std_dist_m = np.maximum(np.std(dist_to_arc_m, axis=1, keepdims=True), 1e-3)
    cand_dist_zscore = (dist_to_arc_m - mean_dist_m) / std_dist_m

    cands_15m = np.broadcast_to(
        np.sum(dist_to_arc_m <= 15.0, axis=1, keepdims=True).astype(np.float32),
        dist_to_arc_m.shape,
    )
    cands_30m = np.broadcast_to(
        np.sum(dist_to_arc_m <= 30.0, axis=1, keepdims=True).astype(np.float32),
        dist_to_arc_m.shape,
    )
    road_density_grid = np.broadcast_to(local_density_30m, dist_to_arc_m.shape)

    # Neighbor observation topological/arc continuity score (in [0, 1])
    # Checks if candidate arc matches or shares a node with prev/next observation's top-2 closest arcs
    n_obs = len(obs_x)
    continuity = np.zeros_like(dist_to_arc_m, dtype=np.float32)
    if n_obs > 1:
        prev_top_arc = np.roll(cand_arc_ids[:, 0], 1)
        prev_top_to = np.roll(c_to[:, 0], 1)
        prev_top_arc[0] = cand_arc_ids[0, 0]
        prev_top_to[0] = c_to[0, 0]

        next_top_arc = np.roll(cand_arc_ids[:, 0], -1)
        next_top_from = np.roll(c_from[:, 0], -1)
        next_top_arc[-1] = cand_arc_ids[-1, 0]
        next_top_from[-1] = c_from[-1, 0]

        prev_connected = (
            (cand_arc_ids == prev_top_arc[:, None])
            | (c_from == prev_top_to[:, None])
        ).astype(np.float32)
        next_connected = (
            (cand_arc_ids == next_top_arc[:, None])
            | (c_to == next_top_from[:, None])
        ).astype(np.float32)
        continuity = 0.5 * (prev_connected + next_connected)

    # Kinematic features & angular alignment between GPS trajectory vector and directed arc vector
    dt_sec, gps_gap_flag, speed_mps, accel_mps2, uvx, uvy = _compute_gps_kinematics(
        obs_x, obs_y, track[:, 2]
    )
    arc_dx = c_bx - c_ax
    arc_dy = c_by - c_ay
    safe_arc_len = np.where(arc_len_m > 1e-9, arc_len_m, 1.0)
    uax = np.where(arc_len_m > 1e-9, arc_dx / safe_arc_len, 0.0)
    uay = np.where(arc_len_m > 1e-9, arc_dy / safe_arc_len, 0.0)

    heading_cos = np.clip(uvx[:, None] * uax + uvy[:, None] * uay, -1.0, 1.0)
    heading_deg = np.degrees(np.arccos(heading_cos))

    # Topological node degree & bidirectionality features per candidate arc
    cand_out_deg = out_degree[c_from]
    cand_in_deg = in_degree[c_to]
    cand_bidir = is_bidir_all[cand_arc_ids]

    obs_idx_grid = np.repeat(np.arange(n_obs, dtype=np.int64), k_eff)
    timestamps_grid = np.repeat(track[:, 2].astype(np.float64), k_eff)
    cand_rank_grid = np.tile(np.arange(k_eff, dtype=np.int64), n_obs)
    flat_arc_ids = cand_arc_ids.reshape(-1).astype(np.int64)
    flat_gt_arcs = np.repeat(gt_arc_per_obs, k_eff)
    is_match = (flat_arc_ids == flat_gt_arcs).astype(np.int64)

    return pd.DataFrame(
        {
            "trip_id": trip_id,
            "obs_idx": obs_idx_grid,
            "timestamp": timestamps_grid,
            "arc_id": flat_arc_ids,
            "cand_rank": cand_rank_grid,
            "is_match": is_match,
            # Spatial (7)
            "dist_to_arc_m": dist_to_arc_m.reshape(-1).astype(np.float32),
            "dist_to_start_node_m": dist_start_m.reshape(-1).astype(np.float32),
            "dist_to_end_node_m": dist_end_m.reshape(-1).astype(np.float32),
            "proj_fraction_t": proj_t.reshape(-1).astype(np.float32),
            "arc_length_m": arc_len_m.reshape(-1).astype(np.float32),
            "cand_dist_diff_from_min_m": cand_dist_diff_m.reshape(-1).astype(np.float32),
            "cand_dist_ratio": cand_dist_ratio.reshape(-1).astype(np.float32),
            # Kinematic (6)
            "dt_sec": np.repeat(dt_sec, k_eff).astype(np.float32),
            "gps_gap_flag": np.repeat(gps_gap_flag, k_eff).astype(np.float32),
            "gps_speed_mps": np.repeat(speed_mps, k_eff).astype(np.float32),
            "gps_accel_mps2": np.repeat(accel_mps2, k_eff).astype(np.float32),
            "heading_diff_deg": heading_deg.reshape(-1).astype(np.float32),
            "heading_diff_cos": heading_cos.reshape(-1).astype(np.float32),
            # Relative / Context (4)
            "cand_dist_zscore": cand_dist_zscore.reshape(-1).astype(np.float32),
            "cands_within_15m": cands_15m.reshape(-1).astype(np.float32),
            "cands_within_30m": cands_30m.reshape(-1).astype(np.float32),
            "neighbor_arc_continuity": continuity.reshape(-1).astype(np.float32),
            # Topological (4)
            "from_node_out_degree": cand_out_deg.reshape(-1).astype(np.float32),
            "to_node_in_degree": cand_in_deg.reshape(-1).astype(np.float32),
            "is_bidirectional_arc": cand_bidir.reshape(-1).astype(np.float32),
            "local_road_density_30m": road_density_grid.reshape(-1).astype(np.float32),
        }
    )


def build_split_feature_parquet(
    split_dir: Path,
    output_parquet: Path,
    top_k: int = 8,
) -> pd.DataFrame:
    """Build v2 candidate features across all trips in `split_dir` and save to `output_parquet`."""
    trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not trip_dirs:
        raise FileNotFoundError(f"No trip folders found in {split_dir}")

    frames: list[pd.DataFrame] = []
    for idx, trip_dir in enumerate(trip_dirs, start=1):
        frames.append(_build_single_trip_candidates(trip_dir, top_k=top_k))
        if idx % 20 == 0 or idx == len(trip_dirs):
            print(f"  Built v2 features for {idx}/{len(trip_dirs)} trips in '{split_dir.name}'...")

    df = pd.concat(frames, ignore_index=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df


def load_feature_datasets(
    train_parquet: Path | str = DEFAULT_TRAIN_PARQUET,
    test_parquet: Path | str = DEFAULT_TEST_PARQUET,
    feature_cols_json: Path | str = DEFAULT_FEATURE_COLS_JSON,
    top_k: int = 8,
    require_v2: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load `train_features.parquet`, `student_test_features.parquet`, and feature column names.

    If `require_v2=True` and the existing cached parquets only have v1 columns (and no external
    `artifacts/feature_columns.json` overrides them), automatically rebuilds the 21 v2 features.
    """
    train_path = Path(train_parquet)
    test_path = Path(test_parquet)
    feat_json_path = Path(feature_cols_json)

    rebuild_train = not train_path.exists()
    rebuild_test = not test_path.exists()

    if not rebuild_train and require_v2 and not feat_json_path.exists():
        existing_cols = set(pd.read_parquet(train_path, columns=None).columns)
        if not set(V2_FEATURE_COLUMNS).issubset(existing_cols):
            print("[data_utils] Upgrading cached train_features.parquet from v1 to 21 v2 features...")
            rebuild_train = True

    if not rebuild_test and require_v2 and not feat_json_path.exists():
        existing_cols = set(pd.read_parquet(test_path, columns=None).columns)
        if not set(V2_FEATURE_COLUMNS).issubset(existing_cols):
            print("[data_utils] Upgrading cached student_test_features.parquet from v1 to 21 v2 features...")
            rebuild_test = True

    if rebuild_train:
        print(f"[data_utils] Building v2 features from {DEFAULT_RAW_TRAIN_DIR}...")
        train_df = build_split_feature_parquet(DEFAULT_RAW_TRAIN_DIR, train_path, top_k=top_k)
    else:
        train_df = pd.read_parquet(train_path)

    if rebuild_test:
        print(f"[data_utils] Building v2 features from {DEFAULT_RAW_TEST_DIR}...")
        test_df = build_split_feature_parquet(DEFAULT_RAW_TEST_DIR, test_path, top_k=top_k)
    else:
        test_df = pd.read_parquet(test_path)

    feature_cols: Sequence[str] | None = None
    if feat_json_path.exists():
        with feat_json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
            if isinstance(payload, dict) and "feature_columns" in payload:
                feature_cols = list(payload["feature_columns"])
            elif isinstance(payload, list):
                feature_cols = [str(c) for c in payload]

    if not feature_cols:
        feature_cols = [
            col
            for col in train_df.columns
            if col not in EXCLUDED_NON_FEATURE_COLUMNS
            and pd.api.types.is_numeric_dtype(train_df[col])
        ]

    resolved_cols = list(feature_cols)
    validate_feature_dataframe(train_df, feature_cols=resolved_cols, require_label=True)
    validate_feature_dataframe(test_df, feature_cols=resolved_cols, require_label=True)

    return train_df, test_df, resolved_cols
