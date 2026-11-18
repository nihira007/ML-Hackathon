# Person 3 — Boosting & Neural Net Models 4–6 + Bayesian Optimization

- **Branch:** `feat/p3-models-boosting-tuning`
- **Owned Folders:** `src/models/advanced/`, `src/tuning/`, `outputs/models/advanced/`, `outputs/tuning/`, `artifacts/final_model.*`
- **Rule:** Do NOT edit other members' folders. Commit after every phase on `feat/p3-models-boosting-tuning`.

---

## Phase 1 Prompt — Branch Setup & Advanced Model + Optuna Scaffolds

```markdown
You are Person 3 working on PathSnap (GPS Map-Matching & Route Reconstruction).
Git: Create and checkout branch `feat/p3-models-boosting-tuning` from `main`. Only edit files inside `src/models/advanced/`, `src/tuning/`, `outputs/models/advanced/`, `outputs/tuning/`, and `artifacts/final_model.*`.

Task (Phase 1 — Advanced Models & Bayesian Tuning Scaffolds):
1. Verify installed libraries (`lightgbm`, `xgboost`, `torch` or `scikit-learn` MLP, `optuna`, `joblib`) and document versions in `outputs/models/advanced/env_versions.txt`.
2. Create clean, modular training scripts ready for the shared parquet schema (`trip_id, obs_idx, arc_id, cand_rank, is_match, <features>`):
   - `src/models/advanced/train_lgbm.py` (**Model 4 — LightGBM**)
   - `src/models/advanced/train_xgb.py` (**Model 5 — XGBoost**)
   - `src/models/advanced/train_mlp.py` (**Model 6 — Neural Network MLP** using PyTorch or MLPClassifier with genuine epoch-wise train & validation log-loss history logged per epoch!)
3. Create `src/tuning/bayesian_opt.py` scaffold using **Optuna** (Bayesian TPE sampler) with 5-fold `GroupKFold` CV grouped by `trip_id`.
4. Run a fast synthetic-data smoke test verifying all 3 model scripts and the Optuna loop run end-to-end without errors.
5. Commit on `feat/p3-models-boosting-tuning` with message `feat(p3): phase 1 advanced model scripts and Optuna Bayesian tuning scaffold`.
```

---

## Phase 2 Prompt — Train & Evaluate Model 4 (LightGBM) & Model 5 (XGBoost) on v1 Features

```markdown
Continue on branch `feat/p3-models-boosting-tuning` (PathSnap — Person 3).
Sync first: Ensure you have Person 1's v1 `data/processed/train_features.parquet`, `student_test_features.parquet`, and `artifacts/feature_columns.json`.

Task (Phase 2 — Train & Evaluate LightGBM and XGBoost Baselines):
1. Train **Model 4 (LightGBM)** with fixed seed=42 using 5-fold `GroupKFold` CV by `trip_id` on the 80 training trips, then refit on all 80 training trips and evaluate on the 10 `student_test` trips.
   - Log CV Macro F1 (mean/std), Student-Test Top-1 Accuracy, Macro F1, Precision, Recall, Log Loss, Train Time, Inference Time, Confusion Matrix, and feature importances.
   - Save `outputs/models/advanced/lgbm_student_test_preds.parquet`.
2. Train **Model 5 (XGBoost)** under the exact same 5-fold `GroupKFold` splits and preprocessing; log identical metrics, confusion matrix, feature importances, and save `outputs/models/advanced/xgb_student_test_preds.parquet`.
3. Commit on `feat/p3-models-boosting-tuning` with message `feat(p3): phase 2 train and evaluate LightGBM and XGBoost on v1 features`.
```

---

## Phase 3 Prompt — Train Model 6 (MLP with Epoch Curves) & Re-run Models 4–6 on v2 Features

```markdown
Continue on branch `feat/p3-models-boosting-tuning` (PathSnap — Person 3).
Sync first: Pull/update to Person 1's v2 features (`train_features.parquet`, `student_test_features.parquet`, `artifacts/feature_columns.json`).

Task (Phase 3 — Model 6 MLP Neural Network & Full Advanced Suite on v2 Features):
1. Train **Model 6 (MLP Neural Network)** on scaled v2 features using 5-fold `GroupKFold` CV on the 80 training trips:
   - Log **real epoch-wise training loss and validation loss** to `outputs/models/advanced/mlp_history.csv` and plot the actual learning curve PNG (remember: report policy strictly requires real training curves only for epoch-based models!).
   - Evaluate on `student_test` and save `outputs/models/advanced/mlp_student_test_preds.parquet`.
2. Re-run **Model 4 (LightGBM)** and **Model 5 (XGBoost)** on the v2 features using the exact same `GroupKFold` splits; record both train and validation scores to check overfitting/underfitting.
3. Save consolidated `outputs/models/advanced/advanced_comparison.csv` and feature importance bar plots.
4. Commit on `feat/p3-models-boosting-tuning` with message `feat(p3): phase 3 train MLP with real epoch loss curves and re-run Models 4-6 on v2 features`.
```

---

## Phase 4 Prompt — Model Selection & Bayesian Hyperparameter Optimization via Optuna

```markdown
Continue on branch `feat/p3-models-boosting-tuning` (PathSnap — Person 3).

Task (Phase 4 — Bayesian Hyperparameter Optimization & Final Model Export):
1. Compare all models (your Models 4–6 plus Person 2's Models 1–3 CSV metrics) and select the single strongest candidate model based on CV Macro F1, Student-Test Top-1 Accuracy, Log Loss (probability quality for HMM), and inference speed.
2. Run **Bayesian Hyperparameter Optimization** (`src/tuning/bayesian_opt.py`) using **Optuna (`TPESampler(seed=42)`)** with **≥35 trials** and 5-fold `GroupKFold` CV strictly on the **80 training trips** (NEVER touch `student_test` inside the Optuna objective!):
   - Define and justify a clear search space (e.g., for LightGBM/XGBoost: `learning_rate`, `num_leaves`/`max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `n_estimators`).
   - Optimize a justified objective (e.g., CV Macro F1 or CV Log Loss + Top-1 Accuracy — explain why).
   - Record every trial's parameters, objective score, and duration to `outputs/tuning/optuna_trials.csv` and total compute cost to `outputs/tuning/tuning_summary.json`.
3. Refit the chosen model on all 80 training trips with the best hyperparameters, evaluate on the 10 `student_test` trips, and produce `outputs/tuning/untuned_vs_tuned.csv` comparing CV Macro F1, Student-Test Top-1 Accuracy, Student-Test Macro F1, Log Loss, and Inference Time.
4. Export the final tuned model to `artifacts/final_model.joblib` (with `artifacts/final_model_meta.json`) and save its `student_test` candidate probabilities to `outputs/tuning/tuned_student_test_preds.parquet` for Person 4's HMM tuning.
5. Commit on `feat/p3-models-boosting-tuning` with message `feat(p3): phase 4 Bayesian hyperparameter optimization and final tuned model export`.
```

---

## Phase 5 Prompt — Tuning Visualisations & Report Sections 3.2 + 4 Summaries

```markdown
Continue on branch `feat/p3-models-boosting-tuning` (PathSnap — Person 3).

Task (Phase 5 — Tuning Plots & Report Documentation):
1. Generate real experimental plots in `outputs/tuning/plots/`:
   - MLP genuine epoch-wise train vs. validation loss plot (`mlp_loss_curve.png`).
   - Optuna optimization history plot (trial number vs. CV score) and hyperparameter importance plot.
   - Untuned vs. Tuned model side-by-side metric comparison chart.
   - Confusion matrices & feature importance charts for Models 4–6 and the final tuned model.
2. Write `outputs/tuning/summary.md` containing:
   - Compulsory Report Section 3.2 answers (A/B/C) for Models 4, 5, and 6 grounded in actual metrics.
   - Complete Report Section 4 answers (hyperparameters searched, distributions/ranges, objective function, number of trials, best params, compute cost, and tuned vs. untuned comparison table).
3. Verify `artifacts/final_model.joblib` reloads cleanly and produces deterministic probabilities, then commit on `feat/p3-models-boosting-tuning` with message `feat(p3): phase 5 tuning plots and report sections 3.2 and 4 summaries`.
```
