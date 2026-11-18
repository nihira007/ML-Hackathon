# Person 1 — Data, Spatial Utils, EDA & Feature Engineering

- **Branch:** `feat/p1-data-features`
- **Owned Folders:** `src/data/`, `src/eda/`, `src/features/`, `outputs/eda/`, `data/processed/`, `artifacts/preprocessor.joblib`, `artifacts/feature_columns.json`
- **Rule:** Do NOT edit other members' folders. Commit after every phase on `feat/p1-data-features`.

---

## Phase 1 Prompt — Branch Setup, Raw Loader & Full EDA

```markdown
You are Person 1 working on PathSnap (GPS Map-Matching & Route Reconstruction).
Git: Create and checkout branch `feat/p1-data-features` from `main`. Only edit files inside `src/data/`, `src/eda/`, `src/features/`, `outputs/eda/`, and `artifacts/`.

Task (Phase 1 — Raw Loader & Complete EDA):
1. Build `src/data/loader.py` to load any trip directory containing `<id>.track` (`lon lat timestamp`), `<id>.nodes` (`lon lat`, 0-based `node_id`), `<id>.arcs` (`from_node_id to_node_id`, 0-based `arc_id`), and optionally `<id>.route` (chronological `arc_id` sequence). Never require `.route` during inference.
2. Build `src/data/spatial.py` implementing a fast per-trip metric coordinate projection (local tangent plane ENU in meters centered on trip mean lat/lon, or per-trip UTM via pyproj) plus Haversine distance, point-to-segment perpendicular distance, projected position along segment `t in [0,1]`, and segment/GPS bearings.
3. Build `src/eda/run_eda.py` to audit all 80 `Student folder/train/` trips and 10 `Student folder/student_test/` trips separately (clearly labeling partitions). Compute and save to `outputs/eda/`:
   - `inventory.csv`: per-trip counts of GPS points, nodes, arcs, route arcs, file sizes, missing/NaN/duplicate timestamps, invalid coordinates.
   - `summary_stats.json`: trajectory lengths, sampling interval distribution (confirm ~1 Hz & flag irregular GPS gaps > 2s/10s), geographical bounding boxes, road-graph connectivity summary.
   - Real matplotlib plots (`outputs/eda/plots/`): GPS gap histogram, points-per-trip distribution, coordinate scatter / trajectory samples.
   - At least 2 interactive Folium EDA maps (`outputs/eda/maps/`) showing raw GPS points and ground-truth `.route` overlay for sample training trips.
4. Commit your work cleanly on `feat/p1-data-features` with message `feat(p1): phase 1 raw loader, metric spatial utils, and full EDA`.
```

---

## Phase 2 Prompt — Candidate Generation, GT Alignment & Feature Pipeline v1

```markdown
Continue on branch `feat/p1-data-features` (PathSnap — Person 1).

Task (Phase 2 — Candidate Generation, Ground-Truth Alignment & Features v1):
1. Build `src/features/candidates.py`:
   - Using projected metric coordinates and a fast spatial index (`scipy.spatial.cKDTree` or Shapely `STRtree` over arc midpoints/segments) so trips with 165k+ arcs run in seconds, generate top-K candidate arcs (e.g., K=10 within a 50–100m radius, with fallback to nearest K if empty) for every GPS observation.
   - Implement ground-truth alignment (`align_ground_truth`): since `.route` is a chronological sequence of distinct traversed `arc_id`s (fewer rows than `.track`), align each GPS observation monotonically/spatially to its true `arc_id` from `.route` (document exact alignment logic & edge cases).
   - Compute and log exact **Candidate Miss Rate** (% of GPS observations whose aligned ground-truth `arc_id` is absent from the candidate set) separately for `train` (80 trips) and `student_test` (10 trips) into `outputs/eda/candidate_miss_rate.csv`.
2. Build `src/features/build_features.py` (v1 baseline features per `(trip_id, obs_idx, candidate_arc_id)`):
   - Core spatial features: perpendicular distance to arc (m), distance to arc start/end nodes, projection fraction `t`, arc length (m), candidate distance rank, distance difference from closest candidate.
   - Fit any preprocessing (`StandardScaler` / imputer) **strictly on the 80 training trips only** (zero data leakage!).
   - Export `data/processed/train_features.parquet`, `data/processed/student_test_features.parquet`, `artifacts/preprocessor.joblib`, and `artifacts/feature_columns.json` following the shared contract (`trip_id, obs_idx, timestamp, arc_id, cand_rank, is_match, <features>`).
3. Commit on `feat/p1-data-features` with message `feat(p1): phase 2 candidate generation, GT alignment, and v1 feature pipeline`.
```

---

## Phase 3 Prompt — Feature Engineering v2 (Kinematic & Topological Features)

```markdown
Continue on branch `feat/p1-data-features` (PathSnap — Person 1).

Task (Phase 3 — Feature Engineering v2 & Leakage Audit):
1. Upgrade `src/features/build_features.py` to add rich domain features while strictly using only `.track`, `.nodes`, and `.arcs`:
   - **Kinematic features:** GPS instantaneous speed (m/s), time delta `dt` (s), GPS heading/bearing (using smoothed prev/next points), angular difference `|gps_bearing - arc_bearing|` (in degrees and `cos(diff)`), acceleration, GPS gap flag (`dt > 2s`).
   - **Relative/context features:** ratio of candidate distance to min candidate distance, z-score of distance among candidates for that observation, number of candidates within 15m/30m, candidate agreement with previous/next observation's nearest arc.
   - **Topological features:** `from_node` out-degree, `to_node` in-degree, bidirectional/reverse-arc indicator, local road density.
2. Verify zero data leakage: confirm `train` (80 trips) and `student_test` (10 trips) have zero overlapping `trip_id`s and that `preprocessor.joblib` is fitted exclusively on `train`.
3. Regenerate `data/processed/train_features.parquet`, `data/processed/student_test_features.parquet`, `artifacts/preprocessor.joblib`, and `artifacts/feature_columns.json`.
4. Save feature correlation & distribution summary plots to `outputs/eda/plots/`.
5. Commit on `feat/p1-data-features` with message `feat(p1): phase 3 kinematic and topological v2 features with leakage check`.
```

---

## Phase 4 Prompt — Inference-Ready Single-Trip Feature Extractor

```markdown
Continue on branch `feat/p1-data-features` (PathSnap — Person 1).

Task (Phase 4 — Inference-Ready Single-Trip Pipeline):
1. Create `src/features/inference_features.py` exposing a clean function:
   `extract_trip_features(trip_dir: str, preprocessor_path: str, feature_cols_path: str) -> pd.DataFrame`
   - Must read ONLY `.track`, `.nodes`, and `.arcs` (must NEVER read or require `.route`).
   - Must return the exact candidate DataFrame with aligned `feature_columns.json` columns ready to feed directly into `model.predict_proba(X)` and the HMM decoder.
   - Benchmark and log single-trip feature extraction runtime across all 10 `student_test` trips (`outputs/eda/feature_extraction_timing.csv`).
2. Add a fallback handler for extreme GPS outliers or sparse road areas so every observation safely gets candidates (or an explicit empty-candidate marker handled cleanly by HMM).
3. Commit on `feat/p1-data-features` with message `feat(p1): phase 4 inference-ready single-trip feature extractor`.
```

---

## Phase 5 Prompt — Verification Tests & EDA Report Artifacts

```markdown
Continue on branch `feat/p1-data-features` (PathSnap — Person 1).

Task (Phase 5 — Unit Tests & Report Section 2.2 Summary):
1. Add `src/features/test_data_pipeline.py` (pytest/script) verifying:
   - Exact trip counts (80 train, 10 student_test) and zero trip overlap.
   - Zero NaNs/Infs in transformed features.
   - `extract_trip_features()` works when `.route` is hidden/absent and matches `student_test_features.parquet` output identically.
2. Generate `outputs/eda/eda_report_summary.md` containing actual numbers, data-quality findings, Candidate Miss Rate analysis, spatial CRS justification, and feature engineering rationale directly ready for Technical Report Section 2.2.
3. Run all checks, ensure clean `git status`, and commit on `feat/p1-data-features` with message `feat(p1): phase 5 pipeline tests and EDA report artifacts`.
```
