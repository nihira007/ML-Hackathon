# Final Post-Merge Integration Prompt (Run on `main` After All 4 Branches Finish Phase 5)

Use this prompt once `feat/p1-data-features`, `feat/p2-models-classical`, `feat/p3-models-boosting-tuning`, and `feat/p4-hmm-viterbi-viz` have all completed Phase 5 and been pushed to GitHub.

```markdown
You are the Integration Lead merging all 4 team branches of **PathSnap** into `main` to produce a single, verified, reproducible end-to-end ML + HMM/Viterbi route reconstruction system (deployment FastAPI/UI will be built on top of this in the next stage).

### Step 1 — Clean Sequential Merge into `main`
1. Checkout `main` and merge branches in dependency order:
   - `git merge --no-ff feat/p1-data-features`
   - `git merge --no-ff feat/p2-models-classical`
   - `git merge --no-ff feat/p3-models-boosting-tuning`
   - `git merge --no-ff feat/p4-hmm-viterbi-viz`
2. Resolve any minor shared-file conflicts (`requirements.txt`, `.gitignore`, shared utils) cleanly without dropping any member's code or experimental outputs.

### Step 2 — Build the Unified End-to-End Inference & Server-Side Evaluation Pipeline
Create `src/pipeline/inference.py` with TWO strictly separated functions (essential for future Laptop 1 FastAPI deployment & zero-leakage compliance):
1. `predict_trip(trip_dir: str, artifacts_dir: str = "artifacts") -> dict`:
   - Reads ONLY `.track`, `.nodes`, and `.arcs` (MUST NEVER read `.route`).
   - Runs P1's `extract_trip_features` -> P3's `artifacts/final_model.joblib` -> P4's HMM/Viterbi decoder using `artifacts/hmm_config.json`.
   - Returns: `trip_id`, `predicted_arc_ids_per_obs`, `predicted_route_sequence`, `predicted_coordinates` (lat/lon polyline), `gps_coordinates`, `route_continuity`, `inference_time_sec`, and optional generated Folium HTML string/path.
2. `evaluate_trip_server_side(prediction: dict, route_file_path: str) -> dict`:
   - Strictly separate from `predict_trip` so ground-truth `.route` labels never enter the model or UI.
   - Computes observation-level Top-1 accuracy, sequence-level F1/precision/recall, and route continuity against the reference `.route`.

### Step 3 — Full End-to-End Verification & Master Report Consolidation
1. Run `src/pipeline/verify_end_to_end.py` across ALL 10 trips in `Student folder/student_test/`:
   - Confirm `predict_trip()` runs cleanly on all 10 trips with `.route` untouched, logs total inference time per trip, and matches the standalone benchmark metrics.
   - Verify all required artifacts exist and load cleanly: `artifacts/preprocessor.joblib`, `artifacts/feature_columns.json`, `artifacts/final_model.joblib`, `artifacts/hmm_config.json`.
2. Consolidate all 6 models (P2's Models 1–3 + P3's Models 4–6 + P4's Route Continuity per model) into a single master table `outputs/master_model_comparison.csv` and master comparison charts (`outputs/master_plots/`):
   - Columns matching Report Section 3.1: `Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time`.
3. Audit repository compliance:
   - Confirm zero data leakage, zero hardcoded predictions, no secrets/credentials, and reproducible `requirements.txt`.
   - Update root `README.md` with clear folder structure, setup/install steps, commands to reproduce EDA -> 6-model training -> Bayesian tuning -> HMM/Viterbi -> single-trip inference, and summary results table (noting API/UI deployment hooks ready in `src/pipeline/inference.py`).
4. Commit the verified integrated pipeline on `main` with message `feat(main): post-merge end-to-end PathSnap pipeline, master comparisons, and verification` and push to `origin main`.
```
