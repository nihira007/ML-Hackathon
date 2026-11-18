# PathSnap - Section 2.2 Data, EDA and Feature Engineering (auto-generated)

_All numbers below were computed by the pipeline from the provided data (no hard-coded results). Regenerate with `python -m src.eda.make_report_summary`._

## 1. Dataset overview

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| GPS points / trip (min / median / max) | 361 / 1529 / 19313 | 262 / 1420 / 3958 |
| ground-truth route arcs / trip (min / median / max) | 59 / 280 / 2266 | 95 / 250 / 1323 |
| road graph per trip, max nodes / max arcs | 155,003 / 307,392 | 26,635 / 56,027 |
| trajectory length (m), median | 24,038 | 20,967 |

`.track` rows are **not** 1-to-1 with `.route` rows (many GPS fixes per traversed arc), so ground-truth labels per GPS fix must be derived by alignment (Section 3).

## 2. Data-quality findings

* **Sampling rate:** 96.50% of train intervals and 98.62% of student_test intervals are exactly 1 s (median 1 s) - the data is ~1 Hz.
* **Irregular gaps:** train has 2,937 intervals > 2 s (1.41%) and 590 > 10 s, in 50/80 trips; student_test has 138 > 2 s and 27 > 10 s, in 3/10 trips. The longest gap is 153,064 s (trip 00000090, train). Gap-aware kinematics: fixes more than 10 s apart are never used for speed / heading, and `dt_s` is clipped at 60 s.
* **Missing / invalid values:** trips with NaN rows: 0; invalid coordinates: 0; duplicate timestamps: 0; non-monotonic timestamps: 0; out-of-range arc / route ids: 0 / 0 (train); student_test: all zero as well.
* **Route structure:** 10/80 train and 1/10 student_test routes are not a perfectly connected arc chain; 9 train trips and 1 student_test trips traverse some arc more than once (loops). The alignment therefore never assumes continuity and resolves repeats by position in the route, not by arc id.
* **Road graph:** median mean out-degree 2.03; median share of arcs that have a reverse arc 95.1% (two-way streets are stored as two directed arcs with identical geometry); the largest weakly connected component holds a median 98.1% of the nodes (strongly connected: 96.4%), and only 1/80 train graphs are a single component.
* **Ground-truth inconsistent with GPS (important):** in trip(s) 00000054 (88% of fixes), 00000033 (52% of fixes) the vehicle is far (> 60 m, median 140 m) from the recorded route for long stretches, so no arc of the route explains those fixes. Overall 22,207 of 208,417 train fixes (10.7%) and 25 of 19,160 student_test fixes have such an unreliable label (`label_ok = 0`). They are dropped from the training set and only flagged in student_test.

## 3. Spatial CRS choice and ground-truth alignment

* **CRS:** every trip is projected to its own local East-North-Up tangent plane (WGS84-exact geodetic -> ECEF -> ENU, origin = mean GPS lat/lon of the trip), so all distances, bearings and projections are in metres. Trips lie at latitudes -34.6 - 61.7 deg where one degree of longitude is only 0.82 - 0.47 of a degree of latitude, so Euclidean distance in degrees would be anisotropic by a factor of 2.1-1.2. Over all GPS steps ENU agrees with haversine to a median 0.195% / p99 0.487% (max 0.56%) relative difference (the residual is the sphere-vs-ellipsoid difference), i.e. far below GPS noise. Per-trip frames avoid UTM-zone issues for trips in different cities and need no external dependency.
* **Spatial index:** each arc is densified to <= 20 m samples and indexed in a `cKDTree`; a ball query of radius `R + 10 m` provably returns every arc within `R`; exact point-to-segment distances rank the candidates. The largest road graph (307,392 arcs, trip 00000013) is indexed in 0.33 s (measured when this report was generated).
* **Ground-truth alignment:** monotone dynamic program over route positions (cost = capped perpendicular distance to the route arc; non-decreasing route position; jump <= 15 arcs per 1 s step, unlimited after a gap > 2 s; any start / end). It respects driving order, resolves loops and never confuses an arc with its geometrically identical reverse arc. Alignment quality (train, all fixes): median over trips of the per-trip median distance fix -> aligned arc = 4.6 m; across all 90 trips 90.1% of route arcs receive at least one GPS fix (the others are traversed between two consecutive fixes).

## 4. Candidate generation and Candidate Miss Rate

Top-K=10 arcs within 100 m per GPS fix (exact nearest-K fallback with `is_fallback_cand=1` if none is in range). Miss rate = % of GPS fixes whose aligned ground-truth arc is not in the candidate set.

| partition | K | miss rate, all fixes | miss rate, label-ok fixes |
|---|---|---|---|
| train | 1 | 50.34% | 44.80% |
| train | 3 | 14.99% | 5.33% |
| train | 5 | 12.34% | 2.44% |
| train | 10 | 10.48% | 1.05% |
| train | 15 | 10.19% | 0.84% |
| train | 20 | 9.94% | 0.61% |
| student_test | 1 | 46.00% | 45.93% |
| student_test | 3 | 4.96% | 4.83% |
| student_test | 5 | 1.75% | 1.65% |
| student_test | 10 | 0.21% | 0.12% |
| student_test | 15 | 0.14% | 0.05% |
| student_test | 20 | 0.10% | 0.01% |

**Reported (K=10):** student_test 0.21% of fixes (40/19,160); train 10.48% on all fixes, **1.05% on label-ok fixes** (1,952/186,210). The all-fix train figure is dominated by the inconsistent-label trips above (the vehicle is nowhere near the recorded route there), not by the candidate generator. The candidate miss rate is a hard ceiling on per-fix accuracy of any classifier; increasing K from 10 to 20 changes it only marginally (table), so K=10 is kept for speed. Because every two-way street contributes two directed candidates with identical distance, only 55.8% of true arcs are the nearest candidate and 36.7% are the 2nd nearest (the reverse twin ties) - distance alone cannot pick the direction, which motivates the bearing and connectivity features.

## 5. Feature engineering rationale

27 numeric features per (fix, candidate arc), built only from `.track`, `.nodes`, `.arcs`:

* **Core spatial (v1 baseline, 7):** `perp_dist_m`, `dist_to_start_m`, `dist_to_end_m`, `proj_t`, `arc_len_m`, `dist_rank`, `dist_diff_from_min_m` - how close the fix is to the arc, where along it, and how it ranks among rivals.
* **Kinematic:** `speed_mps`, `dt_s`, `accel_mps2`, `gps_gap_flag` (dt > 2 s) and the heading features `bearing_diff_deg`, `cos_bearing_diff` (GPS heading from fixes +-2 samples away vs the arc's bearing). Heading disambiguates the two directions of a two-way street; speed / gap tell the model how much to trust heading and position. Undefined values (stationary, after a gap) are NaN and imputed with the train median.
* **Relative / context:** `dist_ratio_to_min`, `dist_zscore`, `n_cand_within_15m/30m` (local ambiguity), and `same_as_prev/next_nearest`, `connected_prev/next_nearest` (temporal consistency with the neighbouring fixes' nearest arcs - uses only GPS, no labels).
* **Topological:** `from_out_degree`, `to_out_degree`, `to_in_degree`, `has_reverse_arc`, `local_road_density` (junction complexity and urban density).
* **Preprocessing:** median imputer + StandardScaler fitted on the **80 training trips only** (1,510,241 rows), applied unchanged to student_test and to inference. Rows are dropped from the fit only for unreliable labels (117,416 candidate rows).
* **Output sizes:** train 1,510,241 rows (80 trips, positive rate 12.2%); student_test 157,866 rows (10 trips, positive rate 12.1%).

## 6. Leakage audit

* train / student_test trip-id overlap: **0** (80 / 10 trips).
* preprocessor fitted on exactly the 1,510,241 train rows (refit check identical: True); student_test trip ids inside the fit set: 0.
* `.route` is used only to create labels (`is_match`, `label_ok`) for train / development; `extract_trip_features()` never reads it (verified by tests with the file physically absent).

## 7. Inference timing

`extract_trip_features` on the 10 student_test trips: 1.75 s in total (0.092 ms per GPS fix; slowest trip 0.34 s for 3,958 fixes); fixes without any candidate: 0.

