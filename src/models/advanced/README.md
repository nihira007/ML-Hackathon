# Run Person 3's models

Run the commands from the repository root. In VS Code, select the same Python interpreter for the editor and terminal.

```powershell
python -m pip install -r requirements.txt
```

The smoke runner uses generated candidate roads. Its model and predictions are for software verification. To reproduce the 35-trial verification from a terminal:

```powershell
python -m src.models.advanced.smoke --trials 35
python -m pytest src/models/advanced/test_input_contract.py -q
```

The generated run writes its verification record, metrics, model, predictions, and plots under `outputs/tuning/synthetic_smoke`. The model plots are in `tuning/plots` inside that directory.

## Run with the team's data

Generate raw features with the pinned Person 1 source. This uses up to ten road candidates per fix, excludes unreliable labels from training, and retains every student-test fix. The v1 manifest selects seven spatial features and v2 selects 27 features. Each model fits its imputer and any scaler inside the current training fold. The dataset comprises 80 training trips and 10 student-test trips.

```powershell
git worktree add --detach ../pathsnap-features 10c442dbfc01576001beec371d42604965d18c64
python -m src.models.advanced.prepare_real_features --feature-source ../pathsnap-features --output ../pathsnap-real-data
$v1 = '../pathsnap-real-data/v1'
$v2 = '../pathsnap-real-data/v2'
$classical = 'outputs/models/advanced/shared_classical_v2'
python -m src.models.advanced.common --feature-version v1 --train "$v1/train_features.parquet" --test "$v1/student_test_features.parquet" --features "$v1/feature_columns.json"
python -m src.tuning.make_report --baseline-plots --models outputs/models/advanced/v1
python -m src.models.advanced.common --feature-version v2 --train "$v2/train_features.parquet" --test "$v2/student_test_features.parquet" --features "$v2/feature_columns.json"
python -m src.tuning.make_report --baseline-plots
python -m src.models.advanced.benchmark_classical --train "$v2/train_features.parquet" --test "$v2/student_test_features.parquet" --features "$v2/feature_columns.json"
python -m src.tuning.bayesian_opt --train "$v2/train_features.parquet" --test "$v2/student_test_features.parquet" --features "$v2/feature_columns.json" --classical "$classical/classical_comparison.csv" --trials 35 --feature-representation raw
python -m src.tuning.make_report
```

Individual runs use python -m src.models.advanced.train_lgbm, train_xgb or train_mlp as full module names under src.models.advanced. Use the consolidated runner to generate advanced_comparison.csv.

The classical rerun uses Person 2's estimator settings and preserves the KNN and Random Forest fit budgets by sampling whole GPS observations within each training fold. All six models are evaluated over the same complete validation and student-test candidate rows. The selector ranks CV Macro F1, student-test Top-1, student-test log loss, and inference time. If a classical model wins your run, pass `--classical-estimator` with its saved estimator from the shared v2 output folder. The TPE objective evaluates only held-out folds of the training partition. Use `--fold-workers` to run tree-model CV folds concurrently when the machine has sufficient resources.

Each v2 baseline stores a fingerprint of both Parquets and the feature manifest. Use the same epoch count for training and tuning. The default is 20. The candidate rows must use unique (trip_id, obs_idx, arc_id) keys. Every trip stays in one fold. Rank is eligible if named in the manifest. Route labels, label-quality flags, time stamps and ID columns are protected from accidental inclusion. Use numeric candidate features and NaN for missing values.

## Outputs and handoff

- outputs/models/advanced/v1 contains the Model 4 and 5 v1 benchmarks.
- outputs/models/advanced contains all v2 baselines, measured MLP loss, per-fold metrics, feature permutation scores and candidate probabilities.
- outputs/tuning contains the six-model scorecard, Optuna trials, tuned versus untuned comparison, plots and a draft for report Sections 3.2 and 4. Read the individual fold logs before writing your team's interpretation.
- artifacts/final_model.joblib is a reusable sklearn-style object. Pass the raw v2 feature columns in the order listed in final_model_meta.json. The fitted pipeline carries its own imputer and any scaling it requires.
- outputs/tuning/tuned_student_test_preds.parquet has trip_id, obs_idx, arc_id, is_match and pred_prob. Person 4 uses pred_prob as the match score for that candidate. The inference pipeline generates these scores without needing the label column.
