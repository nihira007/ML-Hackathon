"""Phase 2 - candidate arc generation + ground-truth alignment.

Everything works in the per-trip metric ENU frame (see ``src.data.spatial``).

Candidate generation
--------------------
Arcs are straight segments between two nodes. To find the arcs near a GPS fix quickly (trips
have up to ~165k+ arcs) every arc is densified into sample points <= ``SPACING_M`` apart and
indexed in a ``scipy.spatial.cKDTree`` (Shapely's STRtree is not required). A ball query of
radius ``radius + spacing/2`` is guaranteed to return every arc whose true distance is
<= ``radius`` (the nearest sample on any such arc is at most spacing/2 from the nearest point
of the arc); exact point-to-segment distances are then computed and used for ranking.

For each GPS observation: top-``K`` arcs (by exact perpendicular distance, ties broken by arc_id)
within ``RADIUS_M``. If NO arc is inside the radius (GPS outlier / sparse area) we fall back to
the exact nearest-``K`` arcs at any distance and flag them ``is_fallback=1``. Only if the trip
has no arcs at all is an explicit empty marker (``arc_id = -1``) emitted.

Ground-truth alignment (``align_ground_truth``)
-----------------------------------------------
``.route`` is the chronological list of traversed arcs (e.g. 149 arcs for 1240 GPS rows) so it is
not 1-to-1 with ``.track``. We assign every GPS row to one *position in the route* with a
monotone dynamic program (a Viterbi over route positions that uses ground truth, so it is only
ever run on training / development trips, never at inference):

    cost[i, k]  = min(perp. distance of GPS i to route arc k, CAP_M)
    D[i, k]     = cost[i, k] + min_{j in [k - J, k]} D[i-1, j]          (non-decreasing k)

* monotone in k  -> time order of the route is respected, so a road that is passed twice
  (loops / repeated arcs: ~11% of trips) is resolved by position-in-route, not by arc_id, and a
  directed arc is never confused with its geometrically identical reverse arc;
* jump limit ``J`` = ``MAX_JUMP`` arcs per 1 s step; after a time gap (dt > ``GAP_FREE_S``) the jump is
  unlimited (the vehicle kept moving while no fix was recorded);
* any start / end position is allowed (the trace may begin before or end after the route);
* the cost is capped at ``CAP_M`` so a single far outlier cannot drag the path;
* edge cases: GPS rows before the first / after the last route arc are clamped onto the first /
  last arc; route arcs that no GPS row maps to (short arcs driven between two fixes) simply get no
  observation; breaks in the route chain (route not fully continuous in ~12% of trips) are not an
  issue because continuity is never assumed - only order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from src.data.spatial import point_to_segment

K_DEFAULT = 10          # candidates per observation used for features
K_MAX = 20              # evaluated for the miss-rate sensitivity table
RADIUS_M = 100.0        # candidate search radius
SPACING_M = 20.0        # arc densification spacing for the KD-tree
MAX_JUMP = 15           # max route arcs advanced between two consecutive 1 s fixes
GAP_FREE_S = 2.0        # dt above this -> unlimited forward jump
CAP_M = 100.0           # alignment cost cap


class ArcIndex:
    """Spatial index over the arcs of one trip (all coordinates in meters)."""

    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray, spacing: float = SPACING_M):
        self.node_xy = node_xy
        self.arcs = arcs
        self.n_arcs = len(arcs)
        self.A = node_xy[arcs[:, 0]] if self.n_arcs else np.empty((0, 2))
        self.B = node_xy[arcs[:, 1]] if self.n_arcs else np.empty((0, 2))
        self.mid = (self.A + self.B) / 2.0
        self.length = np.linalg.norm(self.B - self.A, axis=1) if self.n_arcs else np.empty(0)
        self._mid_tree: Optional[cKDTree] = None
        self.tree = None
        if self.n_arcs == 0:
            return
        counts = np.maximum(2, np.ceil(self.length / spacing).astype(np.int64) + 1)
        total = int(counts.sum())
        self.arc_of = np.repeat(np.arange(self.n_arcs), counts)
        start = np.cumsum(counts) - counts
        off = np.arange(total) - np.repeat(start, counts)
        frac = off / (np.repeat(counts, counts) - 1)
        pts = self.A[self.arc_of] + frac[:, None] * (self.B - self.A)[self.arc_of]
        self.tree = cKDTree(pts)
        self.slack = float(np.max(self.length / (counts - 1)) / 2.0) + 1e-6   # <= spacing/2

    # -- helpers
    def exact(self, p: np.ndarray, arc_ids: np.ndarray):
        return point_to_segment(p, self.A[arc_ids], self.B[arc_ids])

    def road_density(self, arc_ids: np.ndarray, radius: float = 100.0) -> np.ndarray:
        """# arcs whose midpoint lies within ``radius`` of each given arc's midpoint."""
        if self._mid_tree is None:
            self._mid_tree = cKDTree(self.mid)
        return self._mid_tree.query_ball_point(self.mid[arc_ids], r=radius, return_length=True).astype(np.float64)

    def _nearest_k(self, p: np.ndarray, k: int):
        """Exact nearest-k arcs at any distance (fallback for outliers / sparse areas)."""
        kk = min(64, len(self.arc_of))
        while True:
            _, idx = self.tree.query(p, k=kk)
            arcs = np.unique(self.arc_of[np.atleast_1d(idx)])
            if len(arcs) >= min(k, self.n_arcs) or kk >= len(self.arc_of):
                break
            kk = min(kk * 4, len(self.arc_of))
        d, _ = self.exact(p, arcs)
        dk = np.sort(d)[min(k, len(d)) - 1]
        ids = np.unique(self.arc_of[self.tree.query_ball_point(p, r=dk + self.slack)])   # exact set
        d, t = self.exact(p, ids)
        return ids, d, t


def generate_candidates(index: ArcIndex, P: np.ndarray, k: int = K_DEFAULT, radius: float = RADIUS_M) -> Dict[str, np.ndarray]:
    """Top-k candidate arcs for every row of ``P`` (N, 2 meters).

    Returns flat arrays sorted by (obs_idx, rank): obs_idx, arc_id, perp_dist, t, cand_rank, is_fallback.
    """
    n = len(P)
    if index.tree is None:      # explicit empty-candidate marker
        return {"obs_idx": np.arange(n, dtype=np.int32), "arc_id": np.full(n, -1, dtype=np.int32),
                "perp_dist": np.full(n, np.nan), "t": np.full(n, np.nan),
                "cand_rank": np.zeros(n, dtype=np.int32), "is_fallback": np.ones(n, dtype=np.int8)}
    lists = index.tree.query_ball_point(P, r=radius + index.slack)
    o_l, a_l, d_l, t_l, f_l = [], [], [], [], []
    for i in range(n):
        fb = 0
        ids = lists[i]
        if len(ids):
            arcs = np.unique(index.arc_of[ids])
            d, t = index.exact(P[i], arcs)
            m = d <= radius
            arcs, d, t = arcs[m], d[m], t[m]
        else:
            arcs = np.empty(0, dtype=np.int64)
        if len(arcs) == 0:
            arcs, d, t = index._nearest_k(P[i], k)
            fb = 1
        order = np.lexsort((arcs, d))[:k]
        o_l.append(np.full(len(order), i, dtype=np.int32)); a_l.append(arcs[order]); d_l.append(d[order]); t_l.append(t[order])
        f_l.append(np.full(len(order), fb, dtype=np.int8))
    cnt = np.array([len(x) for x in a_l])
    rank = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return {"obs_idx": np.concatenate(o_l), "arc_id": np.concatenate(a_l).astype(np.int32),
            "perp_dist": np.concatenate(d_l), "t": np.concatenate(t_l),
            "cand_rank": rank.astype(np.int32), "is_fallback": np.concatenate(f_l)}


def truncate(cand: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    """Keep the first ``k`` candidates per observation (identical to generating with that k)."""
    m = cand["cand_rank"] < k
    return {key: v[m] for key, v in cand.items()}


# ------------------------------------------------------------------ ground truth alignment
@dataclass
class Alignment:
    arc_id: np.ndarray       # (N,) ground-truth arc per GPS observation
    route_pos: np.ndarray    # (N,) index into the route (non-decreasing)
    dist: np.ndarray         # (N,) true perpendicular distance GPS -> aligned arc (m)


def align_ground_truth(P: np.ndarray, node_xy: np.ndarray, arcs: np.ndarray, route: np.ndarray,
                       dt: Optional[np.ndarray] = None, max_jump: int = MAX_JUMP, cap: float = CAP_M) -> Alignment:
    """Monotone DP alignment of GPS rows to positions of the ground-truth route (see module doc)."""
    n, r = len(P), len(route)
    a, b = node_xy[arcs[route, 0]], node_xy[arcs[route, 1]]
    dist = np.empty((n, r), dtype=np.float32)
    for s in range(0, n, 1000):                       # chunked: avoids a (N, R, 2) float64 blob
        d, _ = point_to_segment(P[s:s + 1000, None, :], a[None], b[None])
        dist[s:s + 1000] = d
    cost = np.minimum(dist, cap)
    if dt is None:
        dt = np.ones(n)
    jmax = min(max_jump, r - 1)
    ar = np.arange(r)
    D = cost[0].copy()
    back = np.zeros((n, r), dtype=np.int32)
    inf = np.float32(np.inf)
    for i in range(1, n):
        if dt[i] > GAP_FREE_S or jmax >= r - 1:       # unlimited forward jump: prefix minimum
            run = np.minimum.accumulate(D)
            src = np.maximum.accumulate(np.where(D <= run, ar, 0))
            best = run
        else:                                          # bounded jump: sliding window of width jmax+1
            stack = np.full((jmax + 1, r), inf, dtype=np.float32)
            for j in range(jmax + 1):
                stack[j, j:] = D[:r - j]
            jb = stack.argmin(0)                       # smallest jump wins ties
            best = stack[jb, ar]
            src = ar - jb
        D = cost[i] + best
        back[i] = src
    pos = np.empty(n, dtype=np.int64)
    k = int(np.argmin(D))
    for i in range(n - 1, -1, -1):
        pos[i] = k
        if i:
            k = int(back[i, k])
    return Alignment(route[pos], pos, dist[np.arange(n), pos].astype(np.float64))


def gt_rank(cand: Dict[str, np.ndarray], gt_arc: np.ndarray, n_obs: int, missing: int = 10 ** 6) -> np.ndarray:
    """Rank (0 = nearest) of the ground-truth arc inside each observation's candidate list, ``missing`` if absent."""
    hit = cand["arc_id"] == gt_arc[cand["obs_idx"]]
    out = np.full(n_obs, missing, dtype=np.int64)
    out[cand["obs_idx"][hit]] = cand["cand_rank"][hit]
    return out
