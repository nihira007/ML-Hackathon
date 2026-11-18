"""PathSnap - Person 4 Phase 1 Smoke Test & Benchmark.

Verifies:
1. RoadGraph construction, metric ENU projection, edge lengths, NetworkX DiGraph on 2 real trips.
2. Network diagnostics (nodes, arcs, isolated nodes, WCC, SCC).
3. Dijkstra shortest-path caching and radius cutoff speed.
4. Route Continuity and sequence metrics on ground-truth .route.
5. Interactive Folium/Leaflet map generation saved to outputs/maps/.
6. Structured logging to outputs/hmm/smoke_test_metrics.json and outputs/hmm/smoke_test_log.csv.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.road_graph import RoadGraph
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.viz.folium_map import create_folium_map


def run_smoke_test() -> None:
    print("=" * 70)
    print("PATHSNAP PERSON 4 - PHASE 1 ROAD GRAPH & CONTINUITY SMOKE TEST")
    print("=" * 70)

    train_dir = PROJECT_ROOT / "Student folder" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    test_trips = ["00000001", "00000002"]
    out_hmm_dir = PROJECT_ROOT / "outputs" / "hmm"
    out_maps_dir = PROJECT_ROOT / "outputs" / "maps"
    out_hmm_dir.mkdir(parents=True, exist_ok=True)
    out_maps_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for trip_id in test_trips:
        trip_path = train_dir / trip_id
        print(f"\n--- Testing Trip: {trip_id} ---")

        # 1. Load RoadGraph
        t0 = time.perf_counter()
        rg = RoadGraph.from_trip_dir(str(trip_path), trip_id=trip_id)
        t_load = time.perf_counter() - t0
        print(f"[OK] RoadGraph loaded in {t_load:.3f}s: {rg.n_nodes:,} nodes, {rg.n_arcs:,} arcs")

        # 2. Graph Diagnostics
        t0 = time.perf_counter()
        diag = rg.get_diagnostics()
        t_diag = time.perf_counter() - t0
        print(f"[OK] Diagnostics computed in {t_diag:.3f}s:")
        print(f"     - Isolated nodes: {diag.n_isolated_nodes}")
        print(f"     - Weakly Connected Components: {diag.n_weakly_connected_components}")
        print(f"     - Strongly Connected Components: {diag.n_strongly_connected_components}")
        print(f"     - Mean arc length: {diag.mean_arc_length_m:.1f} m")
        print(f"     - Total network length: {diag.total_network_length_km:.1f} km")

        # 3. Dijkstra Caching Benchmark
        t0 = time.perf_counter()
        sample_node = int(rg.arcs[0, 1])
        # First call (cold)
        dists_cold = rg.get_shortest_path_distance(sample_node, int(rg.arcs[1, 0]), cutoff=1000.0)
        t_cold = (time.perf_counter() - t0) * 1000.0

        # Second call (warm cache)
        t0 = time.perf_counter()
        dists_warm = rg.get_shortest_path_distance(sample_node, int(rg.arcs[1, 0]), cutoff=1000.0)
        t_warm = (time.perf_counter() - t0) * 1000.0
        print(f"[OK] Dijkstra lookup: cold = {t_cold:.2f}ms, warm (cached) = {t_warm:.4f}ms")

        # 4. Load Ground Truth Route & Track
        route_file = trip_path / f"{trip_id}.route"
        track_file = trip_path / f"{trip_id}.track"
        gt_route = np.loadtxt(str(route_file), dtype=np.int32).tolist()
        gps_track = np.loadtxt(str(track_file), dtype=np.float64)

        # 5. Route Continuity on Ground Truth (Expected ~1.0)
        continuity = compute_route_continuity(gt_route, rg, max_step_dist_m=100.0)
        print(f"[OK] Ground-Truth Route Continuity: {continuity * 100:.2f}% (Expected ~100%)")
        assert continuity >= 0.90, f"Ground-truth continuity too low: {continuity}"

        # 6. Sequence Metrics on Ground Truth (Self-Comparison Expected 1.0)
        seq_metrics = compute_sequence_metrics(gt_route, gt_route)
        print(f"[OK] Ground-Truth Sequence F1: {seq_metrics['sequence_f1']} (Expected 1.0)")
        assert seq_metrics["sequence_f1"] == 1.0

        # 7. Generate Interactive Folium / Leaflet Map
        map_path = out_maps_dir / f"trip_{trip_id}_reconstruction.html"
        t0 = time.perf_counter()
        create_folium_map(
            trip_id=trip_id,
            gps_points=gps_track,
            road_graph=rg,
            ground_truth_route=gt_route,
            predicted_route=gt_route,
            output_html_path=str(map_path),
        )
        t_map = time.perf_counter() - t0
        print(f"[OK] Interactive Map exported in {t_map:.2f}s: {map_path.name}")
        assert map_path.exists() and map_path.stat().st_size > 1000

        results.append({
            "trip_id": trip_id,
            "n_nodes": rg.n_nodes,
            "n_arcs": rg.n_arcs,
            "n_track_points": len(gps_track),
            "n_route_arcs": len(gt_route),
            "graph_load_sec": round(t_load, 4),
            "diagnostics": diag.to_dict(),
            "dijkstra_cold_ms": round(t_cold, 2),
            "dijkstra_warm_ms": round(t_warm, 4),
            "route_continuity": round(continuity, 4),
            "sequence_f1": seq_metrics["sequence_f1"],
            "map_file": str(map_path.name),
        })

    # Save metrics JSON
    metrics_path = out_hmm_dir / "smoke_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"smoke_test_timestamp": time.time(), "trips": results}, f, indent=2)
    print(f"\n[OK] Saved metrics to {metrics_path}")

    # Save CSV summary
    csv_path = out_hmm_dir / "smoke_test_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trip_id",
            "n_nodes",
            "n_arcs",
            "n_track_points",
            "n_route_arcs",
            "graph_load_sec",
            "route_continuity",
            "sequence_f1",
            "dijkstra_cold_ms",
            "dijkstra_warm_ms",
            "map_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[OK] Saved CSV log to {csv_path}")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
