"""PathSnap - Interactive Map Visualizer (Folium & Standalone Leaflet Engine).

Visualizes GPS trajectories, road network geometry, ground-truth reference routes,
classifier-only predictions, and HMM/Viterbi reconstructed routes with interactive
layer toggling, auto-fitted bounds, tooltips, and custom legend.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.graph.road_graph import RoadGraph


def create_folium_map(
    trip_id: str,
    gps_points: np.ndarray,
    road_graph: RoadGraph,
    ground_truth_route: Optional[Sequence[int]] = None,
    predicted_route: Optional[Sequence[int]] = None,
    classifier_only_route: Optional[Sequence[int]] = None,
    output_html_path: Optional[str] = None,
    include_local_network: bool = True,
    max_network_arcs: int = 1500,
) -> str:
    """Generate an interactive HTML map for a trip.

    Args:
        trip_id: Trip identifier.
        gps_points: (N, 2) or (N, 3) array of [lon, lat, (optional timestamp)].
        road_graph: RoadGraph instance containing nodes and arcs.
        ground_truth_route: Optional list of ground truth arc IDs.
        predicted_route: Optional list of HMM reconstructed arc IDs.
        classifier_only_route: Optional list of classifier-only argmax arc IDs.
        output_html_path: Path to save the interactive HTML map.
        include_local_network: Whether to render local road arcs in the background.
        max_network_arcs: Maximum nearby arcs to render to avoid browser slowdown.

    Returns:
        HTML string of the map.
    """
    gps_arr = np.asarray(gps_points)
    lons = gps_arr[:, 0]
    lats = gps_arr[:, 1]
    timestamps = gps_arr[:, 2] if gps_arr.shape[1] >= 3 else np.arange(len(gps_arr))

    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))
    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))

    # Helper to extract polyline for an arc sequence
    def extract_route_polyline(arc_ids: Sequence[int]) -> List[List[float]]:
        coords: List[List[float]] = []
        for arc_id in arc_ids:
            if 0 <= arc_id < road_graph.n_arcs:
                (u_lon, u_lat), (v_lon, v_lat) = road_graph.get_arc_coords(arc_id)
                if not coords or coords[-1] != [u_lat, u_lon]:
                    coords.append([u_lat, u_lon])
                coords.append([v_lat, v_lon])
        return coords

    gt_polyline = extract_route_polyline(ground_truth_route) if ground_truth_route is not None else []
    hmm_polyline = extract_route_polyline(predicted_route) if predicted_route is not None else []
    cls_polyline = extract_route_polyline(classifier_only_route) if classifier_only_route is not None else []

    # Try using folium if installed
    try:
        import folium
        from folium import Element, FeatureGroup, LayerControl, PolyLine

        m = folium.Map(
            location=[mean_lat, mean_lon],
            zoom_start=14,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # 1. Local road network layer
        if include_local_network:
            fg_net = FeatureGroup(name="Local Road Network (Context)", show=False)
            # Find arcs within bounding box + margin
            margin = 0.015
            node_mask = (
                (road_graph.nodes[:, 0] >= min_lon - margin)
                & (road_graph.nodes[:, 0] <= max_lon + margin)
                & (road_graph.nodes[:, 1] >= min_lat - margin)
                & (road_graph.nodes[:, 1] <= max_lat + margin)
            )
            valid_nodes = set(np.where(node_mask)[0])
            count = 0
            for arc_id in range(road_graph.n_arcs):
                u, v = road_graph.arcs[arc_id, 0], road_graph.arcs[arc_id, 1]
                if u in valid_nodes and v in valid_nodes:
                    (u_lon, u_lat), (v_lon, v_lat) = road_graph.get_arc_coords(arc_id)
                    PolyLine(
                        locations=[[u_lat, u_lon], [v_lat, v_lon]],
                        color="#888888",
                        weight=1.5,
                        opacity=0.45,
                        tooltip=f"Arc {arc_id}",
                    ).add_to(fg_net)
                    count += 1
                    if count >= max_network_arcs:
                        break
            fg_net.add_to(m)

        # 2. Raw GPS observations
        fg_gps = FeatureGroup(name="Raw GPS Trajectory", show=True)
        # GPS polyline
        gps_coords = [[float(lats[i]), float(lons[i])] for i in range(len(lats))]
        PolyLine(
            locations=gps_coords,
            color="#E67E22",
            weight=2,
            opacity=0.6,
            dash_array="5, 5",
        ).add_to(fg_gps)
        # Sample points to keep map responsive
        step = max(1, len(lats) // 150)
        for i in range(0, len(lats), step):
            folium.CircleMarker(
                location=[float(lats[i]), float(lons[i])],
                radius=3.5,
                color="#D35400",
                fill=True,
                fill_color="#E67E22",
                fill_opacity=0.85,
                tooltip=f"GPS #{i} | t={timestamps[i]:.1f}s",
            ).add_to(fg_gps)
        fg_gps.add_to(m)

        # 3. Ground Truth Route
        if gt_polyline:
            fg_gt = FeatureGroup(name="Ground Truth Route (.route)", show=True)
            PolyLine(
                locations=gt_polyline,
                color="#27AE60",
                weight=5,
                opacity=0.85,
                tooltip=f"Ground Truth ({len(ground_truth_route)} arcs)",
            ).add_to(fg_gt)
            fg_gt.add_to(m)

        # 4. Classifier-only Route
        if cls_polyline:
            fg_cls = FeatureGroup(name="Classifier-Only (Argmax)", show=False)
            PolyLine(
                locations=cls_polyline,
                color="#E74C3C",
                weight=3.5,
                opacity=0.75,
                dash_array="6, 6",
                tooltip="Classifier-Only Argmax",
            ).add_to(fg_cls)
            fg_cls.add_to(m)

        # 5. HMM Reconstructed Route
        if hmm_polyline:
            fg_hmm = FeatureGroup(name="HMM/Viterbi Reconstructed Route", show=True)
            PolyLine(
                locations=hmm_polyline,
                color="#2980B9",
                weight=4.5,
                opacity=0.92,
                tooltip=f"HMM/Viterbi Route ({len(predicted_route)} arcs)",
            ).add_to(fg_hmm)
            fg_hmm.add_to(m)

        # Legend HTML
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 30px; left: 30px; width: 280px; height: auto; 
                    background-color: white; z-index:9999; font-size:12px;
                    border: 2px solid #aaa; border-radius: 8px; padding: 12px;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.25);">
            <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #333;">
                Trip {trip_id} Map-Matching
            </h4>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background-color: #27AE60; width: 22px; height: 5px; display: inline-block; margin-right: 8px;"></span>
                <span><b>Ground Truth Route</b></span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background-color: #2980B9; width: 22px; height: 5px; display: inline-block; margin-right: 8px;"></span>
                <span><b>HMM / Viterbi Route</b></span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="border-top: 3px dashed #E74C3C; width: 22px; display: inline-block; margin-right: 8px;"></span>
                <span>Classifier-Only (Argmax)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background-color: #E67E22; width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 15px; margin-left: 7px;"></span>
                <span>Raw GPS Trajectory</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #888888; width: 22px; height: 2px; display: inline-block; margin-right: 8px;"></span>
                <span>Road Network Context</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(Element(legend_html))
        LayerControl(collapsed=False).add_to(m)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        html_content = m.get_root().render()
        if output_html_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
            m.save(output_html_path)

        return html_content

    except ImportError:
        # Fallback to self-contained interactive Leaflet.js HTML
        return _generate_standalone_leaflet_html(
            trip_id=trip_id,
            lats=lats,
            lons=lons,
            timestamps=timestamps,
            gt_polyline=gt_polyline,
            hmm_polyline=hmm_polyline,
            cls_polyline=cls_polyline,
            road_graph=road_graph,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            output_html_path=output_html_path,
        )


def _generate_standalone_leaflet_html(
    trip_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    timestamps: np.ndarray,
    gt_polyline: List[List[float]],
    hmm_polyline: List[List[float]],
    cls_polyline: List[List[float]],
    road_graph: RoadGraph,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    output_html_path: Optional[str] = None,
) -> str:
    """Generate self-contained Leaflet.js HTML without requiring folium."""
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    # Downsample GPS markers for responsive rendering
    step = max(1, len(lats) // 150)
    gps_markers = [
        {"lat": float(lats[i]), "lon": float(lons[i]), "idx": int(i), "t": float(timestamps[i])}
        for i in range(0, len(lats), step)
    ]
    gps_path = [[float(lats[i]), float(lons[i])] for i in range(len(lats))]

    # Gather local road segments
    margin = 0.015
    local_arcs = []
    count = 0
    for arc_id in range(road_graph.n_arcs):
        (u_lon, u_lat), (v_lon, v_lat) = road_graph.get_arc_coords(arc_id)
        if (
            min_lon - margin <= u_lon <= max_lon + margin
            and min_lat - margin <= u_lat <= max_lat + margin
        ):
            local_arcs.append([[u_lat, u_lon], [v_lat, v_lon]])
            count += 1
            if count >= 1200:
                break

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>PathSnap - Trip {trip_id} Route Reconstruction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        .legend-card {{
            background: white;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.18);
            border: 1px solid #ddd;
            line-height: 1.6;
            font-size: 13px;
        }}
        .legend-card h4 {{ margin: 0 0 10px 0; font-size: 15px; color: #2c3e50; font-weight: 700; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; }}
        .legend-bar {{ width: 24px; height: 5px; margin-right: 10px; border-radius: 2px; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);

        const tileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri, HERE, Garmin, OpenStreetMap',
            maxZoom: 19
        }}).addTo(map);

        // 1. Context Road Network
        const roadLayer = L.layerGroup();
        const localArcs = {json.dumps(local_arcs)};
        localArcs.forEach(coords => {{
            L.polyline(coords, {{ color: '#888888', weight: 1.5, opacity: 0.4 }}).addTo(roadLayer);
        }});

        // 2. GPS Observations
        const gpsLayer = L.layerGroup().addTo(map);
        const gpsPath = {json.dumps(gps_path)};
        L.polyline(gpsPath, {{ color: '#E67E22', weight: 2, dashArray: '5, 5', opacity: 0.6 }}).addTo(gpsLayer);

        const gpsMarkers = {json.dumps(gps_markers)};
        gpsMarkers.forEach(pt => {{
            L.circleMarker([pt.lat, pt.lon], {{
                radius: 4,
                color: '#D35400',
                fillColor: '#E67E22',
                fillOpacity: 0.85,
                weight: 1
            }}).bindTooltip(`GPS #${{pt.idx}} | t=${{pt.t.toFixed(1)}}s`).addTo(gpsLayer);
        }});

        // 3. Ground Truth Route
        const gtLayer = L.layerGroup().addTo(map);
        const gtCoords = {json.dumps(gt_polyline)};
        if (gtCoords.length > 0) {{
            L.polyline(gtCoords, {{ color: '#27AE60', weight: 5, opacity: 0.85 }}).bindTooltip('Ground Truth Route').addTo(gtLayer);
        }}

        // 4. Classifier-Only Route
        const clsLayer = L.layerGroup();
        const clsCoords = {json.dumps(cls_polyline)};
        if (clsCoords.length > 0) {{
            L.polyline(clsCoords, {{ color: '#E74C3C', weight: 3.5, dashArray: '6, 6', opacity: 0.75 }}).bindTooltip('Classifier-Only (Argmax)').addTo(clsLayer);
        }}

        // 5. HMM / Viterbi Reconstructed Route
        const hmmLayer = L.layerGroup().addTo(map);
        const hmmCoords = {json.dumps(hmm_polyline)};
        if (hmmCoords.length > 0) {{
            L.polyline(hmmCoords, {{ color: '#2980B9', weight: 4.5, opacity: 0.92 }}).bindTooltip('HMM / Viterbi Reconstructed Route').addTo(hmmLayer);
        }}

        // Layer Controls
        const overlays = {{
            "HMM / Viterbi Route": hmmLayer,
            "Ground Truth Route": gtLayer,
            "Raw GPS Observations": gpsLayer,
            "Classifier-Only (Argmax)": clsLayer,
            "Road Network (Context)": roadLayer
        }};
        L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

        // Custom Legend
        const legend = L.control({{ position: 'bottomleft' }});
        legend.onAdd = function() {{
            const div = L.DomUtil.create('div', 'legend-card');
            div.innerHTML = `
                <h4>Trip {trip_id} Map-Matching</h4>
                <div class="legend-item"><div class="legend-bar" style="background:#27AE60;"></div><b>Ground Truth Route</b></div>
                <div class="legend-item"><div class="legend-bar" style="background:#2980B9;"></div><b>HMM / Viterbi Route</b></div>
                <div class="legend-item"><div class="legend-bar" style="border-top:3px dashed #E74C3C; height:0;"></div>Classifier-Only (Argmax)</div>
                <div class="legend-item"><div class="legend-dot" style="background:#E67E22;"></div>Raw GPS Observations</div>
                <div class="legend-item"><div class="legend-bar" style="background:#888888; height:2px;"></div>Road Network Context</div>
            `;
            return div;
        }};
        legend.addTo(map);

        map.fitBounds([[{min_lat}, {min_lon}], [{max_lat}, {max_lon}]]);
    </script>
</body>
</html>"""

    if output_html_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
