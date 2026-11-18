# Person 3 advanced models

Install the repository requirements, then run the grouped LightGBM and XGBoost
benchmarks on Person 1's v1 features and matching feature manifest.

    python -m src.models.advanced.common --feature-version v1 --train data/processed/v1/train_features.parquet --test data/processed/v1/student_test_features.parquet --features data/processed/v1/feature_columns.json

Metrics, feature importance and the candidate probability Parquets go under
outputs/models/advanced/v1. The generated-data smoke check runs independently
with python -m src.models.advanced.smoke.
