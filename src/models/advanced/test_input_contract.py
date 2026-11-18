import json

import numpy as np
import pandas as pd
import pytest

from src.models.advanced.common import feature_names, sample_fit_rows
from src.tuning.bayesian_opt import run_tuning


def test_manifest_allows_candidate_rank(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps(["perp_dist_m", "cand_rank"]), encoding="utf-8")

    assert feature_names(manifest) == ["perp_dist_m", "cand_rank"]


@pytest.mark.parametrize("feature", ["label_ok", "is_match", "arc_id", "timestamp"])
def test_manifest_rejects_label_metadata_and_identifiers(tmp_path, feature):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps(["perp_dist_m", feature]), encoding="utf-8")

    with pytest.raises(ValueError, match="protected metadata"):
        feature_names(manifest)


@pytest.mark.parametrize("models", [("xgb", "mlp"), ("xgb", "mlp", "mlp")])
def test_tuning_requires_one_record_per_advanced_model(tmp_path, models):
    with pytest.raises(ValueError, match="exactly one v2 baseline"):
        run_tuning(
            None, None, [], [{"model": name} for name in models],
            tmp_path / "tuning", tmp_path / "artifacts",
            trials=2, epochs=4, signature=[], smoke=True,
        )


def test_classical_budget_keeps_observations_intact():
    frame = pd.DataFrame({"trip_id": ["a", "a", "a", "a", "b", "b"],
                          "obs_idx": [0, 0, 1, 1, 0, 0]})
    rows = np.arange(len(frame))

    chosen = sample_fit_rows(frame, rows, 2)

    assert chosen.tolist() in ([0, 1], [2, 3], [4, 5])
    assert np.array_equal(chosen, sample_fit_rows(frame, rows, 2))
