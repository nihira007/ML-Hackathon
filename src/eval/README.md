# PathSnap Shared Evaluation & Experiment Logging Harness

**Owner:** Person 2 (`feat/p2-models-classical`)  
**Scope:** Core evaluation metrics, GroupKFold validation, observation-level Top-1 accuracy, standardized experiment logging, and prediction schema contracts.

---

## 1. Overview & Architectural Role

In PathSnap (GPS Map-Matching & Route Reconstruction), multiple machine learning models (Classical Models 1–3 by Person 2, Gradient Boosted Trees / Ensembles by Person 3, and HMM / Viterbi Post-processing by Person 4) are trained, evaluated, and compared.

To guarantee **100% fair comparisons**, **zero data leakage**, and **consistent reporting across the entire team**, Person 2 owns the shared evaluation and logging harness in `src/eval/`.

```
src/eval/
├── __init__.py               # Public API re-exports
├── metrics.py                # Candidate & observation-level metrics, GroupKFold CV, timers, schema validation
├── logger.py                 # Standardized experiment logger (CSV/JSON) & confusion matrix plotting
├── smoke_test_phase1.py      # Automated synthetic smoke test verifying the full pipeline
└── README.md                 # Evaluation harness documentation & contract specification
```

---

## 2. Core Modules & Public API

### `src/eval/metrics.py`

1. **Candidate-Level Classification Metrics:**
   - `compute_candidate_metrics(y_true, y_prob, threshold=0.5)`
   - Computes: Accuracy, Precision, Recall, Macro F1, Binary F1, Log Loss, Brier Score, and Confusion Matrix.

2. **Observation-Level Top-1 & Top-K Accuracy:**
   - `compute_top1_accuracy(df_preds)`
   - Groups candidate rows strictly by `(trip_id, obs_idx)`.
   - Selects the candidate segment with highest predicted probability (`pred_prob`).
   - Evaluates whether that top predicted arc has `is_match == 1`.
   - Also supports `compute_topk_accuracy(df_preds, k=3)`.

3. **Trip-Grouped Cross-Validation (Zero Leakage):**
   - `get_trip_group_kfold(n_splits=5, shuffle=True, random_state=42)`
   - `iter_trip_cv_splits(df, n_splits=5, trip_col="trip_id")`
   - `run_group_kfold_cv(model, df_train, feature_cols, ...)`
   - Strictly enforces `GroupKFold(n_splits=5)` grouped by `trip_id` so that GPS observations from the same trip never leak across train and validation folds.

4. **Timing Utilities:**
   - `Timer` (context manager with high-resolution `time.perf_counter()`).
   - `measure_callable(fn, *args, **kwargs)` returning `(result, elapsed_seconds)`.

5. **Shared Schema Validators:**
   - `validate_feature_dataframe(df, feature_cols)`: validates required ID columns (`trip_id`, `obs_idx`, `timestamp`, `arc_id`, `cand_rank`, `is_match`) and feature types.
   - `validate_prediction_dataframe(df)`: ensures `trip_id, obs_idx, arc_id, is_match, pred_prob` are present with valid probability ranges $[0, 1]$.

---

### `src/eval/logger.py`

1. **Standardized Experiment Table:**
   - Logs experiment rows conforming to the agreed schema:
     `model_name`, `algorithm_family`, `key_hyperparameters`, `random_seed`, `cv_macro_f1_mean`, `cv_macro_f1_std`, `cv_top1_accuracy_mean`, `cv_top1_accuracy_std`, `cv_log_loss_mean`, `train_macro_f1_mean`, `train_top1_accuracy_mean`, `student_test_top1_accuracy`, `student_test_macro_f1`, `student_test_binary_f1`, `student_test_precision`, `student_test_recall`, `student_test_accuracy`, `student_test_log_loss`, `student_test_brier_score`, `train_time_sec`, `inference_time_sec`, `timestamp_utc`.

2. **Automated Artifact Generation:**
   - `ExperimentLogger.log_model_run(...)`: saves metrics to formatted JSON (`save_metrics_json`), appends rows to a master comparison CSV (`append_experiment_csv`), and plots publication-quality confusion matrix heatmaps (`plot_confusion_matrix`).

---

## 3. Running the Smoke Test

To verify the evaluation and logging harness:

```bash
# Direct execution (with virtual environment)
.venv/bin/python src/eval/smoke_test_phase1.py

# Or via pytest
.venv/bin/pytest src/eval/smoke_test_phase1.py
```

### Verification Results:
- **5-Fold CV Macro F1:** $0.9436 \pm 0.0077$
- **5-Fold CV Top-1 Accuracy:** $0.9707 \pm 0.0155$
- **Student-Test Top-1 Accuracy:** $0.9900$
- **Student-Test Macro F1:** $0.9493$
- **Student-Test Precision / Recall:** $0.8673 / 0.9800$
- **Student-Test Log Loss:** $0.0951$
- **Generated Artifacts:**
  - `outputs/models/classical/smoke_test_log.csv`
  - `outputs/models/classical/smoke_test_metrics.json`
  - `outputs/models/classical/smoke_test_confusion_matrix.png`
