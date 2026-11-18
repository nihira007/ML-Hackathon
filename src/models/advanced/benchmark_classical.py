"""Rerun the three classical baselines on the advanced models' raw v2 snapshot."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models.advanced.common import (
    comparison_row, data_arguments, data_signature, evaluate_model, feature_names,
    grouped_cv, load_data, write_json,
)
from src.models.classical.train_knn import KNN_HYPERPARAMETERS, build_knn_estimator
from src.models.classical.train_logreg import build_logreg_estimator
from src.models.classical.train_rf import RF_HYPERPARAMETERS, build_rf_estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("outputs/models/advanced/shared_classical_v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features = feature_names(args.features)
    train, test = load_data(args.train, args.test, features)
    signature = data_signature(args.train, args.test, args.features)
    baselines = [
        ("logreg", "Logistic Regression", build_logreg_estimator(), None, True),
        ("knn", "Distance-Weighted KNN", build_knn_estimator(),
         KNN_HYPERPARAMETERS["max_train_samples"], True),
        ("rf", "Random Forest", build_rf_estimator(),
         RF_HYPERPARAMETERS["max_train_samples"], False),
    ]
    rows = []
    with threadpool_limits(limits=4):
        for slug, name, classifier, cap, scale in baselines:
            preprocessing = [SimpleImputer(strategy="median", keep_empty_features=True)]
            if scale:
                preprocessing.append(StandardScaler())
            pipeline = make_pipeline(*preprocessing, classifier)
            cv, _ = grouped_cv(train, features, slug, estimator=pipeline, max_train_rows=cap)
            fitted, _, record = evaluate_model(
                train, test, features, slug, cv, estimator=pipeline, max_train_rows=cap,
            )
            record.update(model=slug, model_name=name, feature_version="v2",
                          max_train_rows=cap, data_signature=signature)
            rows.append(dict(comparison_row(record), feature_version="v2", max_train_rows=cap,
                             data_signature=":".join(signature)))
            write_json(args.output / f"{slug}_metrics.json", record)
            joblib.dump(fitted, args.output / f"{slug}_estimator.joblib", compress=3)
    pd.DataFrame(rows).to_csv(args.output / "classical_comparison.csv", index=False)


if __name__ == "__main__":
    main()
