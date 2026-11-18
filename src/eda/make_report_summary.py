"""Phase 5 - generate ``outputs/eda/eda_report_summary.md`` (Technical Report Section 2.2 material).

    python -m src.eda.make_report_summary
    
Every number is read from the artifacts produced by run_eda / build_features / inference_features
(or recomputed from the data here); only the surrounding prose is static.
Run after:  run_eda  ->  build_features  ->  inference_features --benchmark
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import REPO_ROOT, TEST_DIR, TRAIN_DIR, list_trips, load_trip  # noqa: E402
from src.data.spatial import ENUProjector, haversine  # noqa: E402
from src.features.build_features import V1_FEATURES, V2_EXTRA  # noqa: E402
from src.features.table_io import load_table  # noqa: E402

OUT = REPO_ROOT / "outputs" / "eda"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def crs_check() -> dict:
    """ENU-vs-haversine agreement on every consecutive GPS step + lon/lat anisotropy."""
    diffs, lats = [], []
    for pdir in (TRAIN_DIR, TEST_DIR):
        for d in list_trips(pdir):
            t = load_trip(d)
            pr = ENUProjector.from_track(t.track)
            xy = pr.project(t.track[:, :2])
            e = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            h = haversine(t.track[:-1, 0], t.track[:-1, 1], t.track[1:, 0], t.track[1:, 1])
            m = h > 1.0
            diffs.append(np.abs(e[m] - h[m]) / h[m]); lats.append(t.track[:, 1].mean())
    rel = np.concatenate(diffs)
    return {"p50_rel": float(np.median(rel)), "p99_rel": float(np.percentile(rel, 99)), "max_rel": float(rel.max()),
            "lat_min": float(min(lats)), "lat_max": float(max(lats)),
            "cos_lat_min": float(np.cos(np.radians(max(lats)))), "cos_lat_max": float(np.cos(np.radians(min(lats))))}


def main() -> None:
    S = json.loads((OUT / "summary_stats.json").read_text())
    inv = pd.read_csv(OUT / "inventory.csv", dtype={"trip_id": str})
    miss = pd.read_csv(OUT / "candidate_miss_rate.csv", dtype={"trip_id": str})
    missk = pd.read_csv(OUT / "candidate_miss_rate_by_K.csv")
    align = pd.read_csv(OUT / "alignment_quality.csv", dtype={"trip_id": str})
    audit = json.loads((OUT / "leakage_audit.json").read_text())
    meta = json.loads((OUT / "preprocessor_meta.json").read_text())
    timing = pd.read_csv(OUT / "feature_extraction_timing.csv")
    fsum = pd.read_csv(OUT / "feature_summary.csv", index_col=0)
    cols = json.loads((REPO_ROOT / "artifacts" / "feature_columns.json").read_text())
    train = load_table(REPO_ROOT / "data" / "processed" / "train_features.parquet")
    test = load_table(REPO_ROOT / "data" / "processed" / "student_test_features.parquet")
    crs = crs_check()

    # measured KD-tree build time on the trip with the largest road graph
    import time
    from src.features.candidates import ArcIndex
    big = inv.sort_values("n_arcs").iloc[-1]
    bt = load_trip((TRAIN_DIR if big.partition == "train" else TEST_DIR) / big.trip_id)
    bp = ENUProjector.from_track(bt.track)
    t0 = time.perf_counter()
    ArcIndex(bp.project(bt.nodes), bt.arcs)
    index_s = time.perf_counter() - t0

    tr, te = S["train"], S["student_test"]
    align["bad"] = align.n_obs - align.n_obs_label_ok
    align["bad_frac"] = align.bad / align.n_obs
    worst = align[align.bad_frac > 0.2].sort_values("bad_frac", ascending=False)
    ma = miss[miss.trip_id == "ALL"].set_index("partition")
    pos = train[train.is_match == 1]
    rank_share = pos.cand_rank.value_counts(normalize=True).sort_index()
    inv_tr = inv[inv.partition == "train"]
    disc = int((inv_tr.route_continuity_frac < 1).sum()), int((inv[inv.partition == "student_test"].route_continuity_frac < 1).sum())
    big_gap_trip = inv.loc[inv.dt_max_s.idxmax()]

    L = []
    w = L.append
    w("# PathSnap - Section 2.2 Data, EDA and Feature Engineering (auto-generated)\n")
    w("_All numbers below were computed by the pipeline from the provided data (no hard-coded results). "
      "Regenerate with `python -m src.eda.make_report_summary`._\n")

    w("## 1. Dataset overview\n")
    w("| | train | student_test |\n|---|---|---|")
    w(f"| trips | {tr['n_trips']} | {te['n_trips']} |")
    w(f"| GPS points | {tr['total_gps_points']:,} | {te['total_gps_points']:,} |")
    w(f"| GPS points / trip (min / median / max) | {tr['gps_points_per_trip']['min']:.0f} / {tr['gps_points_per_trip']['median']:.0f} / {tr['gps_points_per_trip']['max']:.0f} | {te['gps_points_per_trip']['min']:.0f} / {te['gps_points_per_trip']['median']:.0f} / {te['gps_points_per_trip']['max']:.0f} |")
    w(f"| ground-truth route arcs / trip (min / median / max) | {tr['route_arcs_per_trip']['min']:.0f} / {tr['route_arcs_per_trip']['median']:.0f} / {tr['route_arcs_per_trip']['max']:.0f} | {te['route_arcs_per_trip']['min']:.0f} / {te['route_arcs_per_trip']['median']:.0f} / {te['route_arcs_per_trip']['max']:.0f} |")
    w(f"| road graph per trip, max nodes / max arcs | {tr['road_graph']['max_nodes_trip']:,} / {tr['road_graph']['max_arcs_trip']:,} | {te['road_graph']['max_nodes_trip']:,} / {te['road_graph']['max_arcs_trip']:,} |")
    w(f"| trajectory length (m), median | {tr['trajectory_length_m']['median']:,.0f} | {te['trajectory_length_m']['median']:,.0f} |")
    w("\n`.track` rows are **not** 1-to-1 with `.route` rows (many GPS fixes per traversed arc), so ground-truth labels per GPS fix must be derived by alignment (Section 3).\n")

    w("## 2. Data-quality findings\n")
    si, sj = tr["sampling_interval"], te["sampling_interval"]
    w(f"* **Sampling rate:** {pct(si['frac_eq_1s'])} of train intervals and {pct(sj['frac_eq_1s'])} of student_test intervals are exactly 1 s (median {si['median_s']:.0f} s) - the data is ~1 Hz.")
    w(f"* **Irregular gaps:** train has {si['n_gt_2s']:,} intervals > 2 s ({pct(si['frac_gt_2s'])}) and {si['n_gt_10s']} > 10 s, in {tr['data_quality']['trips_with_gaps_gt2s']}/{tr['n_trips']} trips; student_test has {sj['n_gt_2s']} > 2 s and {sj['n_gt_10s']} > 10 s, in {te['data_quality']['trips_with_gaps_gt2s']}/{te['n_trips']} trips. "
      f"The longest gap is {big_gap_trip.dt_max_s:,.0f} s (trip {big_gap_trip.trip_id}, {big_gap_trip.partition}). Gap-aware kinematics: fixes more than 10 s apart are never used for speed / heading, and `dt_s` is clipped at 60 s.")
    dq = tr["data_quality"]
    w(f"* **Missing / invalid values:** trips with NaN rows: {dq['trips_with_nan_rows']}; invalid coordinates: {dq['trips_with_invalid_coords']}; duplicate timestamps: {dq['duplicate_timestamps_total']}; non-monotonic timestamps: {dq['non_monotonic_timestamps_total']}; out-of-range arc / route ids: {dq['trips_with_out_of_range_arcs']} / {dq['trips_with_out_of_range_route']} (train); student_test: all zero as well." if (dq['trips_with_nan_rows'] + dq['trips_with_invalid_coords'] + dq['duplicate_timestamps_total'] + dq['non_monotonic_timestamps_total'] == 0 and sum(te['data_quality'][k] for k in ('trips_with_nan_rows', 'trips_with_invalid_coords', 'duplicate_timestamps_total', 'non_monotonic_timestamps_total')) == 0)
      else f"* **Missing / invalid values:** train {dq}; student_test {te['data_quality']}. The loader repairs invalid coordinates by time-interpolation so `obs_idx` stays aligned.")
    w(f"* **Route structure:** {disc[0]}/{tr['n_trips']} train and {disc[1]}/{te['n_trips']} student_test routes are not a perfectly connected arc chain; {dq['trips_with_repeated_route_arcs']} train trips and {te['data_quality']['trips_with_repeated_route_arcs']} student_test trips traverse some arc more than once (loops). The alignment therefore never assumes continuity and resolves repeats by position in the route, not by arc id.")
    rg, rgt = tr["road_graph"], te["road_graph"]
    w(f"* **Road graph:** median mean out-degree {rg['median_mean_out_degree']:.2f}; median share of arcs that have a reverse arc {pct(rg['median_frac_arcs_with_reverse'], 1)} (two-way streets are stored as two directed arcs with identical geometry); the largest weakly connected component holds a median {pct(rg['median_largest_weak_frac'], 1)} of the nodes (strongly connected: {pct(rg['median_largest_strong_frac'], 1)}), and only {rg['trips_single_weak_component']}/{tr['n_trips']} train graphs are a single component.")
    if len(worst):
        w(f"* **Ground-truth inconsistent with GPS (important):** in trip(s) " + ", ".join(f"{r.trip_id} ({pct(r.bad_frac, 0)} of fixes)" for r in worst.itertuples())
          + f" the vehicle is far (> 60 m, median {worst.align_dist_p50_m.max():.0f} m) from the recorded route for long stretches, so no arc of the route explains those fixes. Overall {int(align[align.partition == 'train'].bad.sum()):,} of {int(align[align.partition == 'train'].n_obs.sum()):,} train fixes ({pct(align[align.partition == 'train'].bad.sum() / align[align.partition == 'train'].n_obs.sum(), 1)}) and {int(align[align.partition == 'student_test'].bad.sum())} of {int(align[align.partition == 'student_test'].n_obs.sum()):,} student_test fixes have such an unreliable label (`label_ok = 0`). They are dropped from the training set and only flagged in student_test.")
    w("")

    w("## 3. Spatial CRS choice and ground-truth alignment\n")
    w(f"* **CRS:** every trip is projected to its own local East-North-Up tangent plane (WGS84-exact geodetic -> ECEF -> ENU, origin = mean GPS lat/lon of the trip), so all distances, bearings and projections are in metres. Trips lie at latitudes {crs['lat_min']:.1f} - {crs['lat_max']:.1f} deg where one degree of longitude is only {crs['cos_lat_max']:.2f} - {crs['cos_lat_min']:.2f} of a degree of latitude, so Euclidean distance in degrees would be anisotropic by a factor of {1 / crs['cos_lat_min']:.1f}-{1 / crs['cos_lat_max']:.1f}. Over all GPS steps ENU agrees with haversine to a median {pct(crs['p50_rel'], 3)} / p99 {pct(crs['p99_rel'], 3)} (max {pct(crs['max_rel'], 2)}) relative difference (the residual is the sphere-vs-ellipsoid difference), i.e. far below GPS noise. Per-trip frames avoid UTM-zone issues for trips in different cities and need no external dependency.")
    w(f"* **Spatial index:** each arc is densified to <= 20 m samples and indexed in a `cKDTree`; a ball query of radius `R + 10 m` provably returns every arc within `R`; exact point-to-segment distances rank the candidates. The largest road graph ({int(big.n_arcs):,} arcs, trip {big.trip_id}) is indexed in {index_s:.2f} s (measured when this report was generated).")
    w("* **Ground-truth alignment:** monotone dynamic program over route positions (cost = capped perpendicular distance to the route arc; non-decreasing route position; jump <= 15 arcs per 1 s step, unlimited after a gap > 2 s; any start / end). It respects driving order, resolves loops and never confuses an arc with its geometrically identical reverse arc. "
      f"Alignment quality (train, all fixes): median over trips of the per-trip median distance fix -> aligned arc = {align[(align.partition == 'train')].align_dist_p50_m.median():.1f} m; across all {len(align)} trips {pct(align.route_arcs_with_obs.sum() / align.route_arcs.sum(), 1)} of route arcs receive at least one GPS fix (the others are traversed between two consecutive fixes).\n")

    w("## 4. Candidate generation and Candidate Miss Rate\n")
    w(f"Top-K=10 arcs within 100 m per GPS fix (exact nearest-K fallback with `is_fallback_cand=1` if none is in range). Miss rate = % of GPS fixes whose aligned ground-truth arc is not in the candidate set.\n")
    w("| partition | K | miss rate, all fixes | miss rate, label-ok fixes |\n|---|---|---|---|")
    for r in missk.itertuples():
        w(f"| {r.partition} | {r.K} | {r.miss_rate_pct_all_obs:.2f}% | {r.miss_rate_pct_label_ok:.2f}% |")
    w(f"\n**Reported (K=10):** student_test {ma.loc['student_test', 'miss_rate_pct']:.2f}% of fixes ({int(ma.loc['student_test', 'n_missed']):,}/{int(ma.loc['student_test', 'n_obs']):,}); train {ma.loc['train', 'miss_rate_pct']:.2f}% on all fixes, **{ma.loc['train', 'miss_rate_pct_label_ok']:.2f}% on label-ok fixes** ({int(ma.loc['train', 'n_missed_label_ok']):,}/{int(ma.loc['train', 'n_obs_label_ok']):,}). "
      "The all-fix train figure is dominated by the inconsistent-label trips above (the vehicle is nowhere near the recorded route there), not by the candidate generator. "
      "The candidate miss rate is a hard ceiling on per-fix accuracy of any classifier; increasing K from 10 to 20 changes it only marginally (table), so K=10 is kept for speed. "
      "Because every two-way street contributes two directed candidates with identical distance, only "
      f"{pct(rank_share.get(0, 0), 1)} of true arcs are the nearest candidate and {pct(rank_share.get(1, 0), 1)} are the 2nd nearest (the reverse twin ties) - distance alone cannot pick the direction, which motivates the bearing and connectivity features.\n")

    w("## 5. Feature engineering rationale\n")
    w(f"{len(cols)} numeric features per (fix, candidate arc), built only from `.track`, `.nodes`, `.arcs`:\n")
    w(f"* **Core spatial (v1 baseline, {len(V1_FEATURES)}):** `" + "`, `".join(V1_FEATURES) + "` - how close the fix is to the arc, where along it, and how it ranks among rivals.")
    w("* **Kinematic:** `speed_mps`, `dt_s`, `accel_mps2`, `gps_gap_flag` (dt > 2 s) and the heading features `bearing_diff_deg`, `cos_bearing_diff` (GPS heading from fixes +-2 samples away vs the arc's bearing). Heading disambiguates the two directions of a two-way street; speed / gap tell the model how much to trust heading and position. Undefined values (stationary, after a gap) are NaN and imputed with the train median.")
    w("* **Relative / context:** `dist_ratio_to_min`, `dist_zscore`, `n_cand_within_15m/30m` (local ambiguity), and `same_as_prev/next_nearest`, `connected_prev/next_nearest` (temporal consistency with the neighbouring fixes' nearest arcs - uses only GPS, no labels).")
    w("* **Topological:** `from_out_degree`, `to_out_degree`, `to_in_degree`, `has_reverse_arc`, `local_road_density` (junction complexity and urban density).")
    w(f"* **Preprocessing:** median imputer + StandardScaler fitted on the **{meta['fit_trips']} training trips only** ({meta['fit_rows']:,} rows), applied unchanged to student_test and to inference. Rows are dropped from the fit only for unreliable labels ({meta['train_rows_dropped_unreliable_label']:,} candidate rows).")
    w(f"* **Output sizes:** train {len(train):,} rows ({train.trip_id.nunique()} trips, positive rate {pct(train.is_match.mean(), 1)}); student_test {len(test):,} rows ({test.trip_id.nunique()} trips, positive rate {pct(test.is_match.mean(), 1)}).\n")

    w("## 6. Leakage audit\n")
    w(f"* train / student_test trip-id overlap: **{audit['overlap_count']}** ({audit['n_train_trips']} / {audit['n_student_test_trips']} trips).")
    w(f"* preprocessor fitted on exactly the {audit['preprocessor_fit_rows']:,} train rows (refit check identical: {audit['preprocessor_fit_on_train_only']}); student_test trip ids inside the fit set: {audit['test_trip_ids_in_fit']}.")
    w("* `.route` is used only to create labels (`is_match`, `label_ok`) for train / development; `extract_trip_features()` never reads it (verified by tests with the file physically absent).\n")

    w("## 7. Inference timing\n")
    t_all = timing[timing.trip_id == "ALL"].iloc[0]
    t_trip = timing[timing.trip_id != "ALL"]
    w(f"`extract_trip_features` on the {len(t_trip)} student_test trips: {t_all.seconds_total:.2f} s in total ({t_all.ms_per_obs:.3f} ms per GPS fix; slowest trip {t_trip.seconds_total.max():.2f} s for {int(t_trip.loc[t_trip.seconds_total.idxmax(), 'n_obs']):,} fixes); fixes without any candidate: {int(t_all.n_obs_without_candidate)}.\n")
    (OUT / "eda_report_summary.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "eda_report_summary.md")


if __name__ == "__main__":
    main()


