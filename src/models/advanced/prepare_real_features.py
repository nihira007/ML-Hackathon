"""Generate raw v1 and v2 snapshots using the Person 1 feature implementation."""

import argparse
from pathlib import Path

import src

from src.models.advanced.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-source", type=Path, required=True,
                        help="Checkout containing Person 1's src/features and src/data")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    source = args.feature_source.resolve()
    src.__path__.insert(0, str(source / "src"))
    from src.data.loader import TEST_DIR, TRAIN_DIR
    from src.features.build_features import FEATURE_SETS, ID_COLS, K_DEFAULT, RADIUS_M, build_partition

    for version, columns in FEATURE_SETS.items():
        target = args.output / version
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "feature_columns.json", columns)

    for partition, directory in (("train", TRAIN_DIR), ("student_test", TEST_DIR)):
        frame, _ = build_partition(partition, directory, "v2", K_DEFAULT, RADIUS_M, None)
        if partition == "train":
            frame = frame.loc[frame.label_ok.eq(1)]
        for version, columns in FEATURE_SETS.items():
            target = args.output / version / f"{partition}_features.parquet"
            frame[ID_COLS + ["is_match", "label_ok"] + columns].to_parquet(target, index=False)
        print(f"{partition}: {len(frame):,} candidate rows across {frame.trip_id.nunique()} trips", flush=True)


if __name__ == "__main__":
    main()
