"""Raw trip loader for PathSnap.

A trip directory ``<root>/<id>/`` contains (headerless, whitespace separated):

* ``<id>.track``  : ``lon lat timestamp``      (~1 Hz GPS)
* ``<id>.nodes``  : ``lon lat``                (row index == node_id, 0-based)
* ``<id>.arcs``   : ``from_node_id to_node_id`` (row index == arc_id, 0-based)
* ``<id>.route``  : ``arc_id`` per row         (ground truth, chronological; TRAINING ONLY)

``.route`` is never needed at inference time: ``load_trip(..., load_route=False)``
(the default) does not even open the file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", REPO_ROOT / "Student folder"))
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "student_test"


@dataclass
class Trip:
    trip_id: str
    trip_dir: Path
    track: np.ndarray          # (N, 3) float64: lon, lat, timestamp  (invalid coords repaired)
    nodes: np.ndarray          # (M, 2) float64: lon, lat
    arcs: np.ndarray           # (A, 2) int64:   from_node_id, to_node_id
    route: Optional[np.ndarray] = None   # (R,) int64 arc ids, or None
    n_invalid_coords: int = 0  # GPS rows that had to be repaired (NaN / out of range)
    meta: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.track)


def list_trips(partition_dir: os.PathLike) -> List[Path]:
    """Sorted trip directories of a partition (``train`` or ``student_test``)."""
    p = Path(partition_dir)
    return sorted(d for d in p.iterdir() if d.is_dir())


def read_table(path: os.PathLike, ncols: int, dtype=np.float64) -> np.ndarray:
    """Read a headerless whitespace separated numeric file -> (rows, ncols). Empty file -> (0, ncols)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64, engine="c")
    arr = df.to_numpy()
    if arr.shape[1] < ncols:
        raise ValueError(f"{path}: expected {ncols} columns, found {arr.shape[1]}")
    return arr[:, :ncols].astype(dtype, copy=False)


def _repair_track(track: np.ndarray) -> tuple[np.ndarray, int]:
    """Linearly interpolate (in time) GPS rows whose coordinates are NaN / out of range.

    Keeps one row per observation so downstream ``obs_idx`` always lines up with the file.
    """
    lon, lat = track[:, 0], track[:, 1]
    bad = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return track, 0
    track = track.copy()
    good = ~bad
    if good.sum() == 0:
        return track, n_bad
    idx = np.arange(len(track))
    for c in (0, 1):
        track[bad, c] = np.interp(idx[bad], idx[good], track[good, c])
    return track, n_bad


def load_trip(trip_dir: os.PathLike, load_route: bool = False) -> Trip:
    """Load one trip. ``load_route=False`` (default) guarantees ``.route`` is never read."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = read_table(trip_dir / f"{tid}.track", 3)
    track, n_bad = _repair_track(track)
    nodes = read_table(trip_dir / f"{tid}.nodes", 2)
    arcs = read_table(trip_dir / f"{tid}.arcs", 2).astype(np.int64)
    route = None
    if load_route:
        rpath = trip_dir / f"{tid}.route"
        if rpath.exists():
            route = read_table(rpath, 1).astype(np.int64).ravel()
    return Trip(tid, trip_dir, track, nodes, arcs, route, n_bad)
