"""Verify all five Person 3 stages using generated candidate rows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.advanced.common import (
    NAMES, PRED_COLS, SEED, comparison_row, data_signature, load_data,
    package_versions, run_baseline, write_json,
)
from src.tuning.bayesian_opt import run_tuning
from src.tuning.make_report import make_report

V1 = ["distance_m", "arc_length_m", "cand_rank"]
V2 = [*V1, "heading_error_deg", "speed_mps"]


def generate(trip_ids, rng):
    rows = []
    for trip_id in trip_ids:
        for observation in range(55):
            distance = rng.uniform(0.5, 30, 3)
            heading = rng.uniform(0, 180, 3)
            score = -distance / 15 - heading / 100 + rng.normal(0, 0.42, 3)
            correct = int(np.argmax(score))
            for rank, index in enumerate(np.argsort(distance), 1):
                rows.append(
                    (trip_id, observation, observation * 1.0, int(index), rank,
                     int(index == correct and observation % 19 != 0),
                     distance[index], rng.uniform(15, 250), heading[index], rng.uniform(1, 22)),
                )
    return pd.DataFrame(rows, columns=[
        "trip_id", "obs_idx", "timestamp", "arc_id", "cand_rank", "is_match",
        "distance_m", "arc_length_m", "heading_error_deg", "speed_mps",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("outputs/tuning/synthetic_smoke"))
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    root = args.work
    root.mkdir(parents=True, exist_ok=True)
    train_path, test_path = root / "train.parquet", root / "student_test.parquet"
    generate([f"synthetic-train-{i}" for i in range(10)], rng).to_parquet(train_path, index=False)
    generate([f"synthetic-test-{i}" for i in range(3)], rng).to_parquet(test_path, index=False)
    epochs = 4
    for label, features, names in (("v1", V1, ["lgbm", "xgb"]), ("v2", V2, list(NAMES))):
        manifest = root / f"{label}_features.json"
        write_json(manifest, features)
        signature = data_signature(train_path, test_path, manifest)
        train, test = load_data(train_path, test_path, features, smoke=True)
        out = root / label
        records = [
            run_baseline(name, train, test, features, out, feature_version=label,
                         signature=signature, epochs=epochs, smoke=True) for name in names
        ]
        pd.DataFrame([comparison_row(r) for r in records]).to_csv(out / "advanced_comparison.csv", index=False)
    tuning = root / "tuning"
    result = run_tuning(train, test, V2, records, tuning, root / "artifacts", trials=args.trials,
                        epochs=epochs, signature=signature, smoke=True)
    make_report(root / "v2", tuning)
    handoff = pd.read_parquet(tuning / "tuned_student_test_preds.parquet")
    assert list(handoff.columns) == PRED_COLS
    assert len(handoff) == len(test) and handoff.pred_prob.between(0, 1).all()
    assert handoff.trip_id.str.startswith("synthetic-test-").all()
    history = pd.read_csv(root / "v2" / "mlp_history.csv")
    assert len(history) == 5 * epochs and history.validation_loss.nunique() > 1
    write_json(root / "verification.json", {
        "mode": "synthetic", "v1_models": 2, "v2_models": 3, "mlp_fold_epochs": len(history),
        "optuna_trials": result["completed_trials"], "prediction_rows": len(handoff),
        "prediction_schema": list(handoff.columns), "exact_reloaded_probabilities": True,
        "software": package_versions(),
    })
    print(f"Synthetic verification passed. See {root / 'verification.json'}.", flush=True)


if __name__ == "__main__":
    main()
