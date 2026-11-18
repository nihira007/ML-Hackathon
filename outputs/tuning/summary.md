# Person 3 experiment status

The advanced-model runner, TPE search, model export and report generator are implemented. The [synthetic verification](synthetic_verification.md) records an end-to-end run on generated candidate rows, including model reload and probability checks.

The real-data comparison for Sections 3.2 and 4 remains pending. Run the v1 baselines using Person 1's v1 feature snapshot. For v2, rerun the three classical and three advanced baselines on the same candidate rows, feature columns and trip folds. The existing classical comparison used 21 columns and eight candidates per observation. Person 1's current v2 manifest has 27 columns and its default builder keeps up to ten candidates.

After ranking and tuning the models on the shared data, use the measured outputs for the report. The resulting model and candidate probabilities are the handoff to the decoder. See the [run guide](../../src/models/advanced/README.md) for the commands.
