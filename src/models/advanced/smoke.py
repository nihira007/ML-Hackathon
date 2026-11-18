"""Fit each classifier and exercise the training-only tuning loop on generated rows."""
import numpy as np
import pandas as pd

from src.models.advanced.common import NAMES, SEED, grouped_cv
from src.tuning.bayesian_opt import tune


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


def check_models(models):
    frame = generate([f"synthetic-train-{i}" for i in range(10)], np.random.default_rng(SEED))
    for name in models:
        metrics, history = grouped_cv(frame, V2, name, epochs=2)
        assert len(metrics["fold_metrics"]) == 5
        if name == "mlp":
            assert len(history) == 10
        print(f"{name}: synthetic grouped validation passed")
    study, _ = tune(frame, V2, "xgb", trials=2, epochs=2)
    assert len(study.trials) == 2
    print("Two synthetic Optuna trials passed")


if __name__ == "__main__":
    check_models(NAMES)
