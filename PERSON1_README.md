# Person 1 - files by phase  (branch `feat/p1-data-features`)

Copy `src/`, `outputs/eda/` and `artifacts/` into your clone (they land inside your owned folders only), then run from the repo root:

```
pip install -r requirements.txt            # make sure pyarrow is in there, otherwise parquets fall back to .pkl
python -m src.eda.run_eda                  # Phase 1
python -m src.features.build_features      # Phases 2+3  (--feature-set v1 for the Phase-2 baseline)
python -m src.features.inference_features --benchmark   # Phase 4
python -m src.eda.make_report_summary      # Phase 5 report
python -m pytest src/features/test_data_pipeline.py -q  # Phase 5 tests
```

| Phase | Files | Commit message |
|---|---|---|
| 1 | `src/data/{__init__,loader,spatial}.py`, `src/eda/{__init__,run_eda}.py`, `outputs/eda/{inventory.csv,summary_stats.json,plots/{gps_gap_hist,points_per_trip,trajectory_samples}.png,maps/*.html}` | `feat(p1): phase 1 raw loader, metric spatial utils, and full EDA` |
| 2 | `src/features/{__init__,candidates,build_features,table_io}.py` (v1 = first 7 features), `outputs/eda/{candidate_miss_rate.csv,candidate_miss_rate_by_K.csv,alignment_quality.csv}`, `artifacts/{preprocessor.joblib,feature_columns.json}` | `feat(p1): phase 2 candidate generation, GT alignment, and v1 feature pipeline` |
| 3 | `src/features/build_features.py` (v2, 27 features, default), `outputs/eda/{leakage_audit.json,preprocessor_meta.json,feature_summary.csv,plots/feature_*.png}` | `feat(p1): phase 3 kinematic and topological v2 features with leakage check` |
| 4 | `src/features/inference_features.py`, `outputs/eda/feature_extraction_timing.csv` | `feat(p1): phase 4 inference-ready single-trip feature extractor` |
| 5 | `src/features/test_data_pipeline.py`, `src/eda/make_report_summary.py`, `outputs/eda/eda_report_summary.md` | `feat(p1): phase 5 pipeline tests and EDA report artifacts` |

`data/processed/*.parquet` are generated (about 1.5M and 158k rows) and gitignored - not in this bundle.
