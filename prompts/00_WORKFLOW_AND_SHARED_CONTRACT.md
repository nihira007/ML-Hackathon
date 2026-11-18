# PathSnap — 4-Member, 5-Phase Workflow & Shared Contract

This document coordinates all 4 team members up to a **complete, verified end-to-end ML + HMM/Viterbi model** (FastAPI + UI deployment will be done in a subsequent stage).

---

## Critical Dataset Facts (Verified on `Student folder/`)

- **Partitions:**
  - `Student folder/train/`: **80 trips** (training, preprocessing fitting, trip-grouped CV, Bayesian optimization)
  - `Student folder/student_test/`: **10 trips** (development-stage comparison & HMM tuning/validation)
  - **Evaluator test (10 reserved trips):** Untouched during development; `.route` ground truth stays strictly server-side.
- **File Schema (headerless, whitespace-separated):**
  - `<id>.track`: `longitude latitude timestamp` (~1 Hz; ~97% are 1 s apart)
  - `<id>.nodes`: `longitude latitude` (0-based row index = `node_id`)
  - `<id>.arcs`: `from_node_id to_node_id` (0-based row index = `arc_id`)
  - `<id>.route`: `arc_id` (chronological sequence of ground-truth traversed road segments)
- **Important Non-1-to-1 Alignment:** `.track` rows (e.g., 1,315–7,243 GPS points) are **NOT** 1-to-1 with `.route` rows (e.g., 79–525 chronological traversed arcs). GPS points must be properly aligned to `.route`.
- **Large Road Graphs & Metric Projection:** Trips can have up to **88,809 nodes / 165,790 arcs** across different cities. Always use **per-trip metric projection** (local ENU tangent plane or per-trip UTM) + **spatial indexing** (`cKDTree` / Shapely `STRtree`), never raw degree Euclidean distance.

---

## Branch Ownership (Zero Merge Conflicts)

Each member commits ONLY on their dedicated branch and edits ONLY inside their owned folders:

| Member | Branch Name | Owned Folders | Prompt File |
| :--- | :--- | :--- | :--- |
| **Person 1** | `feat/p1-data-features` | `src/data/`, `src/eda/`, `src/features/`, `outputs/eda/`, `data/processed/`, `artifacts/preprocessor.joblib`, `artifacts/feature_columns.json` | `prompts/person1_data_features.md` |
| **Person 2** | `feat/p2-models-classical` | `src/eval/`, `src/models/classical/`, `outputs/models/classical/` | `prompts/person2_models_classical.md` |
| **Person 3** | `feat/p3-models-boosting-tuning` | `src/models/advanced/`, `src/tuning/`, `outputs/models/advanced/`, `outputs/tuning/`, `artifacts/final_model.*` | `prompts/person3_models_boosting_tuning.md` |
| **Person 4** | `feat/p4-hmm-viterbi-viz` | `src/graph/`, `src/hmm/`, `src/viz/`, `outputs/hmm/`, `outputs/maps/`, `artifacts/hmm_config.json` | `prompts/person4_hmm_viterbi_viz.md` |

---

## Shared Data & Artifact Contracts

1. **Feature Parquets (`data/processed/train_features.parquet`, `data/processed/student_test_features.parquet`):**
   - Required ID columns: `trip_id` (str), `obs_idx` (int), `timestamp` (float), `arc_id` (int), `cand_rank` (int)
   - Required label column: `is_match` (int: `1` if aligned ground-truth arc, `0` otherwise)
   - Numeric feature columns: exact ordered list stored in `artifacts/feature_columns.json`
2. **Model Prediction Parquets (`*_student_test_preds.parquet`):**
   - Exact columns: `trip_id`, `obs_idx`, `arc_id`, `is_match`, `pred_prob`
3. **Exported Model & Decoder Artifacts (`artifacts/`):**
   - `artifacts/preprocessor.joblib` + `artifacts/feature_columns.json` (Person 1)
   - `artifacts/final_model.joblib` + `artifacts/final_model_meta.json` (Person 3)
   - `artifacts/hmm_config.json` (Person 4)

---

## Git Commands & Team Sync Cadence

```bash
# Start of project (each member):
git checkout main && git pull origin main
git checkout -b <your-branch-name>

# After each phase:
git add <your-owned-folders>
git commit -m "feat(pX): phase Y <short description>"
git push -u origin <your-branch-name>
```

### Sync Checkpoints
- **After Phase 2 & Phase 3:** Person 1 pushes `feat/p1-data-features` with updated feature-generation code; Persons 2 & 3 pull/run P1's script locally to generate `data/processed/*.parquet` (kept out of Git via `.gitignore` to avoid large-file issues) so everyone trains on identical features.
- **After Phase 4:** Person 3 pushes `outputs/tuning/tuned_student_test_preds.parquet`; Person 4 pulls it to tune and benchmark the HMM/Viterbi decoder on the final tuned classifier probabilities.
- **After Phase 5:** Merge all 4 branches into `main` using `prompts/05_POST_MERGE_INTEGRATION.md`.

---

## Hard Academic Integrity & Engineering Rules
- **Zero data leakage:** Fit all learned preprocessors/scalers strictly on the 80 training trips; use 5-fold `GroupKFold` grouped by `trip_id`; never tune Optuna on `student_test`; never expose `.route` to inference.
- **Real results only:** No fabricated metrics, no artificial epoch curves for non-iterative models, no hardcoded predictions.
- **Viva readiness:** Every team member must be able to explain all code, design choices, and recorded metrics.
