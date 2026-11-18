"""PathSnap - Road Network Graph Engine.

Loads trip road network (.nodes and .arcs) into a metric-weighted directed graph
using NetworkX, providing fast shortest-path lookups, spatial projection,
and graph connectivity diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def haversine_distance(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute great-circle distance between coordinates in meters using the Haversine formula."""
    r_earth = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return r_earth * c


def project_latlon_to_enu(
    lons: np.ndarray,
    lats: np.ndarray,
    origin_lon: Optional[float] = None,
    origin_lat: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Project (lon, lat) arrays to local East-North-Up (ENU) tangent plane in meters.

    Args:
        lons: Longitude array in degrees.
        lats: Latitude array in degrees.
        origin_lon: Projection center longitude. Defaults to mean longitude.
        origin_lat: Projection center latitude. Defaults to mean latitude.

    Returns:
        Tuple of (x_east_m, y_north_m, origin_lon, origin_lat).
    """
    if origin_lon is None:
        origin_lon = float(np.mean(lons))
    if origin_lat is None:
        origin_lat = float(np.mean(lats))

    r_earth = 6371000.0
    lat0_rad = math.radians(origin_lat)
    delta_lon_rad = np.radians(lons - origin_lon)
    delta_lat_rad = np.radians(lats - origin_lat)

    x_east = r_earth * delta_lon_rad * math.cos(lat0_rad)
    y_north = r_earth * delta_lat_rad
    return x_east, y_north, origin_lon, origin_lat


@dataclass
class GraphDiagnostics:
    """Summary diagnostics for a road network graph."""

    n_nodes: int
    n_arcs: int
    n_isolated_nodes: int
    n_weakly_connected_components: int
    n_strongly_connected_components: int
    mean_arc_length_m: float
    min_arc_length_m: float
    max_arc_length_m: float
    total_network_length_km: float

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_isolated_nodes": self.n_isolated_nodes,
            "n_weakly_connected_components": self.n_weakly_connected_components,
            "n_strongly_connected_components": self.n_strongly_connected_components,
            "mean_arc_length_m": round(self.mean_arc_length_m, 2),
            "min_arc_length_m": round(self.min_arc_length_m, 2),
            "max_arc_length_m": round(self.max_arc_length_m, 2),
            "total_network_length_km": round(self.total_network_length_km, 2),
        }


class RoadGraph:
    """Directed Road Network Graph for map matching and route reconstruction."""

    def __init__(
        self,
        nodes: np.ndarray,
        arcs: np.ndarray,
        trip_id: str = "unknown",
    ) -> None:
        """Initialize RoadGraph.

        Args:
            nodes: (N_nodes, 2) array of [lon, lat]. Row index corresponds to node_id.
            arcs: (N_arcs, 2) array of [from_node_id, to_node_id]. Row index is arc_id.
            trip_id: Identifier of the trip / city network.
        """
        self.trip_id = trip_id
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.arcs = np.asarray(arcs, dtype=np.int32)
        self.n_nodes = len(self.nodes)
        self.n_arcs = len(self.arcs)

        # Metric local ENU projection
        self.x, self.y, self.origin_lon, self.origin_lat = project_latlon_to_enu(
            self.nodes[:, 0], self.nodes[:, 1]
        )

        # Compute metric lengths of all arcs (in meters)
        from_x, from_y = self.x[self.arcs[:, 0]], self.y[self.arcs[:, 0]]
        to_x, to_y = self.x[self.arcs[:, 1]], self.y[self.arcs[:, 1]]
        self.arc_lengths = np.hypot(to_x - from_x, to_y - from_y)

        # Build NetworkX DiGraph
        self.G = nx.DiGraph()
        # Add all nodes with attributes
        for node_id in range(self.n_nodes):
            self.G.add_node(
                node_id,
                lon=float(self.nodes[node_id, 0]),
                lat=float(self.nodes[node_id, 1]),
                x=float(self.x[node_id]),
                y=float(self.y[node_id]),
            )

        # Add directed edges with arc_id and weight in meters
        edges_with_data = []
        for arc_id in range(self.n_arcs):
            u = int(self.arcs[arc_id, 0])
            v = int(self.arcs[arc_id, 1])
            w = float(self.arc_lengths[arc_id])
            edges_with_data.append((u, v, {"arc_id": arc_id, "weight": w}))

        self.G.add_edges_from(edges_with_data)

        # Fast shortest-path Dijkstra cache: source_node -> {target_node: distance_m}
        self._dijkstra_cache: Dict[int, Dict[int, float]] = {}
        self._cache_cutoff: float = 1500.0

    @classmethod
    def from_files(
        cls,
        nodes_path: str,
        arcs_path: str,
        trip_id: str = "unknown",
    ) -> RoadGraph:
        """Load nodes and arcs from text files."""
        nodes = np.loadtxt(nodes_path, dtype=np.float64)
        arcs = np.loadtxt(arcs_path, dtype=np.int32)
        return cls(nodes=nodes, arcs=arcs, trip_id=trip_id)

    @classmethod
    def from_trip_dir(cls, trip_dir: str, trip_id: Optional[str] = None) -> RoadGraph:
        """Load road graph directly from trip directory containing <trip_id>.nodes and <trip_id>.arcs."""
        import os

        base_name = os.path.basename(os.path.normpath(trip_dir))
        tid = trip_id or base_name
        nodes_path = os.path.join(trip_dir, f"{tid}.nodes")
        arcs_path = os.path.join(trip_dir, f"{tid}.arcs")

        if not os.path.exists(nodes_path) or not os.path.exists(arcs_path):
            # Fallback search for any .nodes and .arcs
            nodes_candidates = [f for f in os.listdir(trip_dir) if f.endswith(".nodes")]
            arcs_candidates = [f for f in os.listdir(trip_dir) if f.endswith(".arcs")]
            if nodes_candidates and arcs_candidates:
                nodes_path = os.path.join(trip_dir, nodes_candidates[0])
                arcs_path = os.path.join(trip_dir, arcs_candidates[0])
                tid = nodes_candidates[0].replace(".nodes", "")
            else:
                raise FileNotFoundError(f"Missing .nodes or .arcs in {trip_dir}")

        return cls.from_files(nodes_path, arcs_path, trip_id=tid)

    def get_diagnostics(self) -> GraphDiagnostics:
        """Compute network diagnostics for the graph."""
        isolated = list(nx.isolates(self.G))
        wcc_count = nx.number_weakly_connected_components(self.G)
        scc_count = nx.number_strongly_connected_components(self.G)

        return GraphDiagnostics(
            n_nodes=self.n_nodes,
            n_arcs=self.n_arcs,
            n_isolated_nodes=len(isolated),
            n_weakly_connected_components=wcc_count,
            n_strongly_connected_components=scc_count,
            mean_arc_length_m=float(np.mean(self.arc_lengths)),
            min_arc_length_m=float(np.min(self.arc_lengths)),
            max_arc_length_m=float(np.max(self.arc_lengths)),
            total_network_length_km=float(np.sum(self.arc_lengths)) / 1000.0,
        )

    def get_arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        """Return (from_node_id, to_node_id) for given arc_id."""
        return int(self.arcs[arc_id, 0]), int(self.arcs[arc_id, 1])

    def get_arc_coords(self, arc_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((from_lon, from_lat), (to_lon, to_lat)) for given arc_id."""
        u, v = self.arcs[arc_id, 0], self.arcs[arc_id, 1]
        return (
            (float(self.nodes[u, 0]), float(self.nodes[u, 1])),
            (float(self.nodes[v, 0]), float(self.nodes[v, 1])),
        )

    def get_shortest_path_distance(
        self,
        source_node: int,
        target_node: int,
        cutoff: float = 1500.0,
    ) -> float:
        """Compute shortest path distance in meters from source_node to target_node with caching."""
        if source_node == target_node:
            return 0.0

        # Query cache
        if source_node not in self._dijkstra_cache or cutoff > self._cache_cutoff:
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.G, source_node, cutoff=cutoff, weight="weight"
                )
                self._dijkstra_cache[source_node] = distances
                self._cache_cutoff = max(self._cache_cutoff, cutoff)
            except Exception:
                self._dijkstra_cache[source_node] = {source_node: 0.0}

        source_cache = self._dijkstra_cache[source_node]
        if target_node in source_cache:
            return float(source_cache[target_node])

        return float("inf")

    def get_arc_transition_distance(
        self,
        arc_prev: int,
        arc_curr: int,
        cutoff: float = 1500.0,
    ) -> float:
        """Compute road network route distance in meters from arc_prev to arc_curr.

        - If arc_prev == arc_curr: returns 0.0 (continuous movement on same segment).
        - If arc_prev directly connects to arc_curr: returns 0.0.
        - Otherwise: computes shortest network path from to_node(arc_prev) to from_node(arc_curr).
        """
        if arc_prev == arc_curr:
            return 0.0

        u_from, u_to = self.get_arc_endpoints(arc_prev)
        v_from, v_to = self.get_arc_endpoints(arc_curr)

        if u_to == v_from:
            return 0.0

        return self.get_shortest_path_distance(u_to, v_from, cutoff=cutoff)
