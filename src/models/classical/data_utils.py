"""Data loading and self-contained v1 spatial candidate feature builder for Person 2 classical models.

Loads `data/processed/train_features.parquet` and `data/processed/student_test_features.parquet`
(plus `artifacts/feature_columns.json` when available). If Person 1's processed parquets are not
yet present on disk, builds the v1 spatial candidate feature tables directly from the 80 training
trips (`Student folder/train/`) and 10 development test trips (`Student folder/student_test/`) and
caches them to `data/processed/*.parquet` (which is gitignored, preserving strict branch isolation).
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
    """Extract top-K candidate road segments and v1 spatial features for one trip folder."""
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

    gt_arc_per_obs = _align_gps_to_ground_truth_route(
        obs_x, obs_y, ax_all, ay_all, bx_all, by_all, route
    )

    mid_x = 0.5 * (ax_all + bx_all)
    mid_y = 0.5 * (ay_all + by_all)
    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    pool_k = min(midpoint_pool_k, len(arcs))
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

    min_dist_m = dist_to_arc_m[:, [0]]
    cand_dist_diff_m = dist_to_arc_m - min_dist_m
    cand_dist_ratio = dist_to_arc_m / (min_dist_m + 1.0)

    n_obs = len(obs_x)
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
            "dist_to_arc_m": dist_to_arc_m.reshape(-1).astype(np.float32),
            "dist_to_start_node_m": dist_start_m.reshape(-1).astype(np.float32),
            "dist_to_end_node_m": dist_end_m.reshape(-1).astype(np.float32),
            "proj_fraction_t": proj_t.reshape(-1).astype(np.float32),
            "arc_length_m": arc_len_m.reshape(-1).astype(np.float32),
            "cand_dist_diff_from_min_m": cand_dist_diff_m.reshape(-1).astype(np.float32),
            "cand_dist_ratio": cand_dist_ratio.reshape(-1).astype(np.float32),
        }
    )


def build_split_feature_parquet(
    split_dir: Path,
    output_parquet: Path,
    top_k: int = 8,
) -> pd.DataFrame:
    """Build v1 candidate features across all trips in `split_dir` and save to `output_parquet`."""
    trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not trip_dirs:
        raise FileNotFoundError(f"No trip folders found in {split_dir}")

    frames: list[pd.DataFrame] = []
    for idx, trip_dir in enumerate(trip_dirs, start=1):
        frames.append(_build_single_trip_candidates(trip_dir, top_k=top_k))
        if idx % 20 == 0 or idx == len(trip_dirs):
            print(f"  Built v1 features for {idx}/{len(trip_dirs)} trips in '{split_dir.name}'...")

    df = pd.concat(frames, ignore_index=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df


def load_feature_datasets(
    train_parquet: Path | str = DEFAULT_TRAIN_PARQUET,
    test_parquet: Path | str = DEFAULT_TEST_PARQUET,
    feature_cols_json: Path | str = DEFAULT_FEATURE_COLS_JSON,
    top_k: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load `train_features.parquet`, `student_test_features.parquet`, and feature column names."""
    train_path = Path(train_parquet)
    test_path = Path(test_parquet)
    feat_json_path = Path(feature_cols_json)

    if not train_path.exists():
        print(f"[data_utils] '{train_path}' not found; building v1 features from {DEFAULT_RAW_TRAIN_DIR}...")
        train_df = build_split_feature_parquet(DEFAULT_RAW_TRAIN_DIR, train_path, top_k=top_k)
    else:
        train_df = pd.read_parquet(train_path)

    if not test_path.exists():
        print(f"[data_utils] '{test_path}' not found; building v1 features from {DEFAULT_RAW_TEST_DIR}...")
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
