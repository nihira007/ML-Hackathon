"""PathSnap - Generate Mandatory Folium Maps for Student Test Trips.

Renders interactive maps for at least 3 representative trips:
1. Trip 00000052: Best improvement case (+21.5% continuity gain)
2. Trip 00000091: High GPS dropout and ambiguity case (+14.5% continuity gain)
3. Trip 00000000: Standard dense urban trajectory
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.road_graph import RoadGraph
from src.hmm.evaluate_hmm import load_trip_route, load_trip_track
from src.hmm.viterbi import HMMConfig, ViterbiDecoder
from src.viz.folium_map import create_folium_map


def generate_representative_maps() -> None:
    tuned_pq = Path("outputs/tuning/tuned_student_test_preds.parquet")
    parquet_path = str(tuned_pq) if tuned_pq.exists() else "outputs/models/classical/logreg_student_test_preds.parquet"
    test_dir = Path("Student folder/student_test")
    maps_out = Path("outputs/maps")
    maps_out.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_parquet(parquet_path)
    target_trips = ["00000000", "00000052", "00000091"]

    config = HMMConfig()
    decoder = ViterbiDecoder(config)

    for trip_id in target_trips:
        t_dir = test_dir / trip_id
        if not t_dir.exists():
            continue

        print(f"Generating map for Trip {trip_id}...")
        df_trip = df_all[df_all["trip_id"] == trip_id]
        road_graph = RoadGraph.from_trip_dir(str(t_dir), trip_id=trip_id)
        gps_track = load_trip_track(t_dir, trip_id)
        gt_route = load_trip_route(t_dir, trip_id)

        # Build classifier-only sequence and observations for HMM
        obs_groups = df_trip.groupby("obs_idx", sort=True)
        obs_indices = sorted(obs_groups.groups.keys())

        cls_only_arcs = []
        observations = []
        for obs_idx in obs_indices:
            group = obs_groups.get_group(obs_idx).sort_values(by="pred_prob", ascending=False)
            best_row = group.iloc[0]
            cls_only_arcs.append(int(best_row["arc_id"]))

            cand_list = [(int(r["arc_id"]), float(r["pred_prob"])) for _, r in group.iterrows()]
            lon = float(gps_track[obs_idx, 0]) if obs_idx < len(gps_track) else float(road_graph.nodes[0, 0])
            lat = float(gps_track[obs_idx, 1]) if obs_idx < len(gps_track) else float(road_graph.nodes[0, 1])
            ts = float(gps_track[obs_idx, 2]) if obs_idx < len(gps_track) else float(obs_idx)

            observations.append({
                "obs_idx": int(obs_idx),
                "timestamp": ts,
                "lon": lon,
                "lat": lat,
                "candidates": cand_list,
            })

        # Run HMM decoding
        hmm_res = decoder.decode(observations, road_graph)
        hmm_collapsed = hmm_res["collapsed_route"]

        # Save HTML map
        out_html = maps_out / f"trip_{trip_id}_reconstruction.html"
        create_folium_map(
            trip_id=trip_id,
            gps_points=gps_track,
            road_graph=road_graph,
            ground_truth_route=gt_route,
            predicted_route=hmm_collapsed,
            classifier_only_route=cls_only_arcs,
            output_html_path=str(out_html),
        )
        print(f"  [OK] Saved {out_html.name} ({out_html.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    generate_representative_maps()
