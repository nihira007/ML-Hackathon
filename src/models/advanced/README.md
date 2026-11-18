# Run Person 3 in VS Code

Open the cloned ML-Hackathon folder in VS Code and check out feat/p3-models-boosting-tuning. Copy this package's src, outputs and .vscode folders into the clone. Install the repository's requirements in the interpreter selected with **Python: Select Interpreter**.

```powershell
python -m pip install -r requirements.txt
```

Open **Run and Debug**, select **PathSnap Person 3 synthetic verification**, and press F5. The smoke runner uses generated rows and saves its artifacts inside outputs/tuning/synthetic_smoke. Its model and predictions are for software verification.

## Run with the team's data

Keep v1 and v2 in separate snapshots. Each snapshot needs train_features.parquet, student_test_features.parquet and the matching feature_columns.json. Copy or generate v2 at the standard paths shown in the repository's shared contract. The example assumes the v1 snapshot is at data/processed/v1. Use the same v2 features and five trip-grouped folds as Person 2.

```powershell
python -m src.models.advanced.common --feature-version v1 --train data/processed/v1/train_features.parquet --test data/processed/v1/student_test_features.parquet --features data/processed/v1/feature_columns.json
python -m src.tuning.make_report --baseline-plots --models outputs/models/advanced/v1
python -m src.models.advanced.common --feature-version v2
python -m src.tuning.make_report --baseline-plots
python -m src.tuning.bayesian_opt --trials 35
python -m src.tuning.make_report
```

Individual runs use python -m src.models.advanced.train_lgbm, train_xgb or train_mlp as full module names under src.models.advanced. Use the consolidated runner to generate advanced_comparison.csv.

The tuning command expects Person 2's real v2 outputs/models/classical/classical_comparison.csv. The model selection requires CV Macro F1, student-test Top-1, student-test log loss and inference time from all six models. The report also uses CV F1 variability and student-test Macro F1. If a classical model wins, export its exact sklearn classifier or preprocessing pipeline to a trusted local joblib file and pass --classical-estimator PATH. Measure classical timing on the same machine and batch where possible. The student-test results inform development-stage selection. TPE scores its trials using training groups only.

Each v2 baseline stores a fingerprint of both Parquets and the feature manifest. Use the same epoch count for training and tuning. The default is 20. The candidate rows must use unique (trip_id, obs_idx, arc_id) keys. Every trip stays in one fold. Rank is eligible if named in the manifest. Route labels, time stamps and ID columns are protected from accidental inclusion. Use numeric candidate features and NaN for missing values.

## Outputs and handoff

- outputs/models/advanced/v1 contains the Model 4 and 5 v1 benchmarks.
- outputs/models/advanced contains all v2 baselines, measured MLP loss, per-fold metrics, feature permutation scores and candidate probabilities.
- outputs/tuning contains the six-model scorecard, Optuna trials, tuned versus untuned comparison, plots and a draft for report Sections 3.2 and 4. Read the individual fold logs before writing your team's interpretation.
- artifacts/final_model.joblib is a reusable sklearn-style object. Pass the same feature representation as the v2 Parquet in the order listed in final_model_meta.json. The fitted pipeline already carries its own imputer and any scaling it requires.
- outputs/tuning/tuned_student_test_preds.parquet has trip_id, obs_idx, arc_id, is_match and pred_prob. Person 4 uses pred_prob as the match score for that candidate. The inference pipeline generates these scores without needing the label column.
