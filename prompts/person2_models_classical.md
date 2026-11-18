# Person 2 — Shared Evaluation Harness & Classical/Ensemble Models 1–3

- **Branch:** `feat/p2-models-classical`
- **Owned Folders:** `src/eval/`, `src/models/classical/`, `outputs/models/classical/`
- **Rule:** Do NOT edit other members' folders. Commit after every phase on `feat/p2-models-classical`.

---

## Phase 1 Prompt — Branch Setup & Shared Evaluation Harness

```markdown
You are Person 2 working on PathSnap (GPS Map-Matching & Route Reconstruction).
Git: Create and checkout branch `feat/p2-models-classical` from `main`. Only edit files inside `src/eval/`, `src/models/classical/`, and `outputs/models/classical/`.

Task (Phase 1 — Shared Evaluation & Experiment Logging Harness):
1. Build `src/eval/metrics.py` implementing all required candidate-level and observation-level metrics:
   - Candidate-level: Accuracy, Precision, Recall, Macro F1-score, Log Loss, Confusion Matrix.
   - **Observation-level Top-1 Accuracy:** Group candidate rows by `(trip_id, obs_idx)`, pick the candidate with highest `pred_prob`, and check if `is_match == 1`.
   - **Trip-grouped Cross-Validation helper:** `GroupKFold(n_splits=5)` grouped strictly by `trip_id` so observations from the same trip never split across train/val folds.
   - Timing utilities for wall-clock training time and inference time.
2. Build `src/eval/logger.py` to save standardized experiment rows (model name, algorithm family, key hyperparameters, random seed, CV Macro F1 mean/std, Student-Test Top-1 Accuracy, Student-Test Macro F1, Precision, Recall, Log Loss, Train Time, Inference Time) into CSV/JSON and save confusion matrix PNG plots.
3. Write a synthetic smoke test confirming `metrics.py` and `logger.py` work against the shared parquet schema (`trip_id, obs_idx, arc_id, cand_rank, is_match, <features>`).
4. Commit on `feat/p2-models-classical` with message `feat(p2): phase 1 shared GroupKFold evaluation and logging harness`.
```

---

## Phase 2 Prompt — Train & Evaluate Model 1 (Logistic Regression) & Model 2 (Distance-Weighted KNN)

```markdown
Continue on branch `feat/p2-models-classical` (PathSnap — Person 2).
Sync first: Ensure you have Person 1's v1 `data/processed/train_features.parquet`, `student_test_features.parquet`, and `artifacts/feature_columns.json` (by running P1's pipeline or pulling `feat/p1-data-features` data scripts).

Task (Phase 2 — Baseline Classical Models 1 & 2):
1. Build `src/models/classical/train_logreg.py` (**Model 1 — Linear/Probabilistic: Logistic Regression** with `class_weight='balanced'` or tuned L2 regularization, fixed seed=42):
   - Run 5-fold `GroupKFold` CV by `trip_id` on the 80 training trips.
   - Refit on all 80 training trips and evaluate on the 10 `student_test` trips.
   - Save metrics to `outputs/models/classical/logreg_metrics.json`, confusion matrix PNG, and candidate predictions `outputs/models/classical/logreg_student_test_preds.parquet`.
2. Build `src/models/classical/train_knn.py` (**Model 2 — Instance-based: Distance-Weighted KNN** using scaled features, e.g., `weights='distance'`, `n_neighbors=15`, `n_jobs=-1`, with subsampling/KDTree if needed for speed — document exact config):
   - Run 5-fold `GroupKFold` CV on the 80 training trips and evaluate on the 10 `student_test` trips.
   - Save metrics, confusion matrix PNG, and `outputs/models/classical/knn_student_test_preds.parquet`.
3. Commit on `feat/p2-models-classical` with message `feat(p2): phase 2 train and evaluate Logistic Regression and KNN baselines`.
```

---

## Phase 3 Prompt — Train Model 3 (Random Forest / Extra Trees) & Re-run on v2 Features

```markdown
Continue on branch `feat/p2-models-classical` (PathSnap — Person 2).
Sync first: Pull/update to Person 1's v2 features (`train_features.parquet`, `student_test_features.parquet`, `artifacts/feature_columns.json`).

Task (Phase 3 — Model 3 Ensemble & Full Classical Re-evaluation on v2 Features):
1. Build `src/models/classical/train_rf.py` (**Model 3 — Tree Ensemble: Random Forest Classifier** — or Extra Trees, fixed seed=42, balanced class weights/tuned depth to control overfitting).
2. Re-run **all 3 classical models** (Logistic Regression, Distance-Weighted KNN, Random Forest) on the exact same v2 feature set and identical 5-fold `GroupKFold` splits so comparisons are 100% fair.
3. Also record **training vs. validation** scores for each model to quantify overfitting/underfitting evidence (required for Report Section 3.2C).
4. Export a unified runner `src/models/classical/run_all_classical.py` and consolidated CSV `outputs/models/classical/classical_comparison.csv`.
5. Commit on `feat/p2-models-classical` with message `feat(p2): phase 3 train Random Forest and re-evaluate Models 1-3 on v2 features`.
```

---

## Phase 4 Prompt — Probability Calibration Check, Error Analysis & Predictions Export

```markdown
Continue on branch `feat/p2-models-classical` (PathSnap — Person 2).

Task (Phase 4 — Error Analysis & Probability Quality Audit for Models 1–3):
1. Analyze probability quality (Log Loss, Brier score / calibration curve plot) for Models 1–3 since the downstream HMM relies on well-calibrated emission probabilities.
2. Save per-trip performance breakdown across all 10 `student_test` trips (`outputs/models/classical/per_trip_results.csv`) to identify which trips cause errors (e.g., dense intersections, parallel roads, GPS gaps).
3. Ensure the saved `student_test` prediction parquets (`logreg_student_test_preds.parquet`, `knn_student_test_preds.parquet`, `rf_student_test_preds.parquet`) strictly match the shared schema (`trip_id, obs_idx, arc_id, is_match, pred_prob`) so Person 4 can compute Route Continuity for every classical model.
4. Commit on `feat/p2-models-classical` with message `feat(p2): phase 4 calibration audit and per-trip error analysis for Models 1-3`.
```

---

## Phase 5 Prompt — Classical Visualisations & Report Section 3.2 Summaries

```markdown
Continue on branch `feat/p2-models-classical` (PathSnap — Person 2).

Task (Phase 5 — Final Charts & Report Section 3.2 Documentation for Models 1–3):
1. Generate publication-ready comparison charts (from real CSV results only — no artificial epoch curves!) in `outputs/models/classical/plots/`:
   - CV Macro F1 comparison bar chart (with fold std error bars).
   - Student-Test Top-1 Accuracy & Macro F1 grouped bar chart.
   - Log Loss & Inference Time comparison plots + Confusion Matrix grid for Models 1–3.
2. Write `outputs/models/classical/summary.md` answering all compulsory Report Section 3.2 questions (A: suitability, strengths/limitations; B: initial hyperparameters & loss/objective; C: actual student-test Top-1/Macro F1/Log Loss & overfitting/underfitting evidence) grounded in your real experimental numbers for Models 1, 2, and 3.
3. Verify scripts run cleanly and commit on `feat/p2-models-classical` with message `feat(p2): phase 5 final plots and report summary for Models 1-3`.
```
