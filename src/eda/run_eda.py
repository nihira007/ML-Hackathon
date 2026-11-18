"""Phase 1 - full EDA audit of the PathSnap trips.

Run from the repo root:   python -m src.eda.run_eda

Audits ``Student folder/train`` (80 trips) and ``Student folder/student_test`` (10 trips)
separately and writes everything to ``outputs/eda/``:

* inventory.csv                  per-trip counts / data-quality flags (column ``partition``)
* summary_stats.json             per-partition aggregate statistics
* plots/gps_gap_hist.png, plots/points_per_trip.png, plots/trajectory_samples.png
* maps/eda_<trip>.html           interactive maps (raw GPS + ground-truth route), train trips only

Only real numbers computed from the data are written - nothing is hard-coded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import REPO_ROOT, TEST_DIR, TRAIN_DIR, list_trips, load_trip, read_table  # noqa: E402
from src.data.spatial import ENUProjector  # noqa: E402

OUT = REPO_ROOT / "outputs" / "eda"
GAP_WARN_S, GAP_BIG_S = 2.0, 10.0


# --------------------------------------------------------------------------- audit
def audit_trip(trip_dir: Path, partition: str) -> tuple[dict, np.ndarray]:
    """Audit one trip from the *raw* files (no repair), return (inventory row, dt array)."""
    tid = trip_dir.name
    raw = read_table(trip_dir / f"{tid}.track", 3)
    nodes = read_table(trip_dir / f"{tid}.nodes", 2)
    arcs = read_table(trip_dir / f"{tid}.arcs", 2)
    route = read_table(trip_dir / f"{tid}.route", 1).ravel() if (trip_dir / f"{tid}.route").exists() else np.empty(0)

    lon, lat, ts = raw[:, 0], raw[:, 1], raw[:, 2]
    nan_rows = int((~np.isfinite(raw)).any(axis=1).sum())
    invalid_coord = int((~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)).sum())
    dts = np.diff(ts)
    dup_ts = int((dts == 0).sum())
    nonmono = int((dts < 0).sum())

    n_nodes, n_arcs = len(nodes), len(arcs)
    arc_oob = int(((arcs < 0) | (arcs >= n_nodes)).any(axis=1).sum()) if n_arcs else 0
    route_i = route[np.isfinite(route)].astype(np.int64)
    route_oob = int(((route_i < 0) | (route_i >= n_arcs)).sum())
    route_cont = np.nan
    if len(route_i) > 1 and route_oob == 0:
        arcs_i = arcs.astype(np.int64)
        route_cont = float(np.mean(arcs_i[route_i[:-1], 1] == arcs_i[route_i[1:], 0]))

    # trajectory length in meters (valid rows only)
    ok = np.isfinite(lon) & np.isfinite(lat)
    traj_len = np.nan
    if ok.sum() > 1:
        xy = ENUProjector(lon[ok].mean(), lat[ok].mean()).to_enu(lon[ok], lat[ok])
        traj_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    row = {
        "partition": partition, "trip_id": tid,
        "n_gps": len(raw), "n_nodes": n_nodes, "n_arcs": n_arcs, "n_route_arcs": len(route_i),
        "has_route_file": (trip_dir / f"{tid}.route").exists(),
        "size_track_kb": round((trip_dir / f"{tid}.track").stat().st_size / 1024, 1),
        "size_nodes_kb": round((trip_dir / f"{tid}.nodes").stat().st_size / 1024, 1),
        "size_arcs_kb": round((trip_dir / f"{tid}.arcs").stat().st_size / 1024, 1),
        "size_route_kb": round(((trip_dir / f"{tid}.route").stat().st_size / 1024) if (trip_dir / f"{tid}.route").exists() else 0, 1),
        "nan_rows_track": nan_rows, "invalid_coords": invalid_coord,
        "duplicate_timestamps": dup_ts, "non_monotonic_timestamps": nonmono,
        "arcs_node_out_of_range": arc_oob, "route_arc_out_of_range": route_oob,
        "route_continuity_frac": route_cont,
        "route_repeated_arcs": int(len(route_i) - len(np.unique(route_i))),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "traj_length_m": traj_len,
        "dt_median_s": float(np.median(dts)) if len(dts) else np.nan,
        "dt_max_s": float(dts.max()) if len(dts) else np.nan,
        "n_gaps_gt2s": int((dts > GAP_WARN_S).sum()), "n_gaps_gt10s": int((dts > GAP_BIG_S).sum()),
        "lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)),
        "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat)),
    }
    row.update(graph_connectivity(n_nodes, arcs, arc_oob))
    return row, dts


def graph_connectivity(n_nodes: int, arcs: np.ndarray, arc_oob: int) -> dict:
    """Weak / strong connectivity of the (directed) road graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if n_nodes == 0 or len(arcs) == 0 or arc_oob:
        return {"n_weak_components": np.nan, "largest_weak_frac": np.nan, "largest_strong_frac": np.nan,
                "mean_out_degree": np.nan, "frac_arcs_with_reverse": np.nan}
    a = arcs.astype(np.int64)
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    keys = a[:, 0] * n_nodes + a[:, 1]
    rev = a[:, 1] * n_nodes + a[:, 0]
    return {
        "n_weak_components": int(nw),
        "largest_weak_frac": float(np.bincount(lw).max() / n_nodes),
        "largest_strong_frac": float(np.bincount(ls).max() / n_nodes),
        "mean_out_degree": float(len(a) / n_nodes),
        "frac_arcs_with_reverse": float(np.isin(rev, keys).mean()),
    }


def summarize(df: pd.DataFrame, dts_all: np.ndarray) -> dict:
    def dist(s):
        s = s.dropna()
        return {k: float(v) for k, v in {"min": s.min(), "p25": s.quantile(.25), "median": s.median(),
                                         "mean": s.mean(), "p75": s.quantile(.75), "max": s.max()}.items()} if len(s) else {}
    n = len(dts_all)
    return {
        "n_trips": int(len(df)),
        "total_gps_points": int(df.n_gps.sum()),
        "gps_points_per_trip": dist(df.n_gps),
        "route_arcs_per_trip": dist(df.n_route_arcs),
        "nodes_per_trip": dist(df.n_nodes), "arcs_per_trip": dist(df.n_arcs),
        "trajectory_length_m": dist(df.traj_length_m), "duration_s": dist(df.duration_s),
        "sampling_interval": {
            "n_intervals": int(n),
            "frac_eq_1s": float(np.mean(dts_all == 1)) if n else None,
            "frac_le_1s": float(np.mean(dts_all <= 1)) if n else None,
            "frac_gt_2s": float(np.mean(dts_all > GAP_WARN_S)) if n else None,
            "frac_gt_10s": float(np.mean(dts_all > GAP_BIG_S)) if n else None,
            "n_gt_2s": int((dts_all > GAP_WARN_S).sum()), "n_gt_10s": int((dts_all > GAP_BIG_S).sum()),
            "max_gap_s": float(dts_all.max()) if n else None,
            "median_s": float(np.median(dts_all)) if n else None,
            "value_counts_top": {str(k): int(v) for k, v in zip(*np.unique(dts_all, return_counts=True))
                                 if v >= max(1, 0.0005 * n)},
        },
        "data_quality": {
            "trips_with_nan_rows": int((df.nan_rows_track > 0).sum()),
            "trips_with_invalid_coords": int((df.invalid_coords > 0).sum()),
            "duplicate_timestamps_total": int(df.duplicate_timestamps.sum()),
            "non_monotonic_timestamps_total": int(df.non_monotonic_timestamps.sum()),
            "trips_missing_route_file": int((~df.has_route_file).sum()),
            "trips_with_out_of_range_arcs": int((df.arcs_node_out_of_range > 0).sum()),
            "trips_with_out_of_range_route": int((df.route_arc_out_of_range > 0).sum()),
            "trips_with_gaps_gt2s": int((df.n_gaps_gt2s > 0).sum()),
            "trips_with_gaps_gt10s": int((df.n_gaps_gt10s > 0).sum()),
            "route_fully_continuous_trips": int((df.route_continuity_frac == 1.0).sum()),
            "trips_with_repeated_route_arcs": int((df.route_repeated_arcs > 0).sum()),
        },
        "bounding_box_overall": {"lon_min": float(df.lon_min.min()), "lon_max": float(df.lon_max.max()),
                                 "lat_min": float(df.lat_min.min()), "lat_max": float(df.lat_max.max())},
        "road_graph": {
            "nodes_total": int(df.n_nodes.sum()), "arcs_total": int(df.n_arcs.sum()),
            "max_nodes_trip": int(df.n_nodes.max()), "max_arcs_trip": int(df.n_arcs.max()),
            "trips_single_weak_component": int((df.n_weak_components == 1).sum()),
            "median_largest_weak_frac": float(df.largest_weak_frac.median()),
            "median_largest_strong_frac": float(df.largest_strong_frac.median()),
            "median_mean_out_degree": float(df.mean_out_degree.median()),
            "median_frac_arcs_with_reverse": float(df.frac_arcs_with_reverse.median()),
        },
    }


# --------------------------------------------------------------------------- plots
def make_plots(inv: pd.DataFrame, dts: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = OUT / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#1f77b4", "student_test": "#ff7f0e"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part, d in dts.items():
        vals, cnt = np.unique(np.clip(d, 0, 30), return_counts=True)
        ax[0].bar(vals + (0.2 if part == "student_test" else -0.2), cnt / cnt.sum(), width=0.4,
                  label=f"{part} (n={len(d):,})", color=colors[part])
    ax[0].set_yscale("log"); ax[0].set_xlabel("sampling interval dt (s, clipped at 30)"); ax[0].set_ylabel("fraction of intervals")
    ax[0].set_title("GPS sampling interval (log scale)"); ax[0].axvline(2.5, ls="--", c="grey"); ax[0].legend()
    big = {p: d[d > 1] for p, d in dts.items()}
    for part, d in big.items():
        ax[1].hist(np.clip(d, 0, 60), bins=np.arange(1.5, 62, 1), alpha=.6, label=f"{part} (n={len(d):,} intervals > 1 s)", color=colors[part])
    ax[1].set_yscale("log"); ax[1].set_xlabel("gap length (s, clipped at 60)"); ax[1].set_title("Irregular gaps only (dt > 1 s)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(pdir / "gps_gap_hist.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part in ("train", "student_test"):
        ax[0].hist(inv[inv.partition == part].n_gps, bins=25, alpha=.6, label=part, color=colors[part])
    ax[0].set_xlabel("GPS points per trip"); ax[0].set_ylabel("trips"); ax[0].set_title("Points-per-trip distribution"); ax[0].legend()
    for part in ("train", "student_test"):
        s = inv[inv.partition == part]
        ax[1].scatter(s.n_gps, s.n_route_arcs, s=18, alpha=.7, label=part, color=colors[part])
    ax[1].set_xlabel("GPS points"); ax[1].set_ylabel("ground-truth route arcs"); ax[1].set_title("GPS points vs route arcs (not 1-to-1)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(pdir / "points_per_trip.png", dpi=130); plt.close(fig)

    # trajectory samples (6 train trips, GPS + GT route in local ENU)
    sample_ids = inv[inv.partition == "train"].sort_values("n_gps").trip_id.iloc[np.linspace(0, 79, 6).astype(int)].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, tid in zip(axes.ravel(), sample_ids):
        t = load_trip(TRAIN_DIR / tid, load_route=True)
        pr = ENUProjector.from_track(t.track)
        xy, nxy = pr.project(t.track[:, :2]), pr.project(t.nodes)
        a, b = nxy[t.arcs[t.route, 0]], nxy[t.arcs[t.route, 1]]
        for p, q in zip(a, b):
            ax.plot([p[0], q[0]], [p[1], q[1]], c="#2ca02c", lw=1.2)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#d62728", alpha=.5)
        ax.set_aspect("equal"); ax.set_title(f"train/{tid}  ({t.n_obs} GPS, {len(t.route)} route arcs)", fontsize=9)
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.suptitle("Raw GPS (red) vs ground-truth route (green), local ENU frame")
    fig.tight_layout(); fig.savefig(pdir / "trajectory_samples.png", dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- maps
def write_map(trip_dir: Path, out_html: Path, max_points: int = 4000) -> None:
    """Interactive map: raw GPS points + ground-truth route. Folium if installed, else plain Leaflet."""
    t = load_trip(trip_dir, load_route=True)
    step = max(1, len(t.track) // max_points)
    pts = t.track[::step]
    seg = [[[float(t.nodes[a, 1]), float(t.nodes[a, 0])], [float(t.nodes[b, 1]), float(t.nodes[b, 0])]]
           for a, b in t.arcs[t.route]]
    centre = [float(np.mean(t.track[:, 1])), float(np.mean(t.track[:, 0]))]
    title = f"train/{t.trip_id}: {t.n_obs} GPS points (every {step}th shown), {len(t.route)} ground-truth route arcs"
    try:
        import folium
        m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")
        for s in seg:
            folium.PolyLine(s, color="#2ca02c", weight=5, opacity=.8).add_to(m)
        for lon, lat, ts in pts:
            folium.CircleMarker([lat, lon], radius=2, color="#d62728", fill=True, weight=0,
                                tooltip=f"t={ts:.0f}s").add_to(m)
        m.get_root().html.add_child(folium.Element(f"<div style='position:fixed;top:8px;left:60px;z-index:9999;background:white;padding:4px 8px'>{title}</div>"))
        m.save(str(out_html))
    except ImportError:
        pj = json.dumps([[float(la), float(lo), float(ts)] for lo, la, ts in pts])
        out_html.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0}}#t{{position:fixed;top:8px;left:60px;z-index:9999;background:#fff;padding:4px 8px;font:13px sans-serif}}</style></head>
<body><div id="t">{title}<br><span style="color:#2ca02c">&#9632;</span> ground-truth route &nbsp; <span style="color:#d62728">&#9679;</span> raw GPS</div><div id="m"></div>
<script>
var m=L.map('m').setView({json.dumps(centre)},14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap'}}).addTo(m);
var segs={json.dumps(seg)};segs.forEach(function(s){{L.polyline(s,{{color:'#2ca02c',weight:5,opacity:.8}}).addTo(m);}});
var pts={pj};pts.forEach(function(p){{L.circleMarker([p[0],p[1]],{{radius:2,color:'#d62728',weight:0,fillOpacity:.8}}).bindTooltip('t='+p[2]).addTo(m);}});
</script></body></html>""")


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, dts, summaries = [], {}, {}
    for part, pdir in (("train", TRAIN_DIR), ("student_test", TEST_DIR)):
        part_rows, part_dts = [], []
        for d in list_trips(pdir):
            r, dt = audit_trip(d, part)
            part_rows.append(r); part_dts.append(dt)
        dts[part] = np.concatenate(part_dts)
        pdf = pd.DataFrame(part_rows)
        summaries[part] = summarize(pdf, dts[part])
        rows += part_rows
        print(f"[{part}] {len(pdf)} trips, {int(pdf.n_gps.sum()):,} GPS points")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    with open(OUT / "summary_stats.json", "w") as f:
        json.dump(summaries, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
    make_plots(inv, dts)

    # >= 2 interactive maps: a short, a median-sized and the largest-road-graph train trip
    tr = inv[inv.partition == "train"].sort_values("n_gps")
    picks = [tr.trip_id.iloc[0], tr.trip_id.iloc[len(tr) // 2], tr.sort_values("n_arcs").trip_id.iloc[-1]]
    (OUT / "maps").mkdir(exist_ok=True)
    for tid in dict.fromkeys(picks):
        write_map(TRAIN_DIR / tid, OUT / "maps" / f"eda_{tid}.html")
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
