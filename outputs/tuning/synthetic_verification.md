# Person 3 synthetic verification

This run used generated candidate rows from 10 training trips and three development trips. It exercised two v1 models, all three v2 models and 35 Optuna trials. The MLP logged four epochs across each of five folds. The exported model reproduced the same probabilities after reload for all 495 development candidate rows.

## Synthetic v2 results

| Model | CV Macro F1 | Development Top-1 | Development log loss |
| --- | ---: | ---: | ---: |
| LightGBM | 0.6861 | 68.48% | 0.5199 |
| XGBoost | 0.7098 | 70.91% | 0.4809 |
| MLP | 0.6801 | 65.45% | 0.5713 |
| Tuned XGBoost | 0.7242 | 71.52% | 0.4350 |

XGBoost led the weighted ranking. The lowest average CV log loss was 0.471616 at trial 27. The selected configuration used 120 trees, a maximum depth of 3 and a learning rate of approximately 0.028.

```powershell
python -m pytest src/models/advanced/test_input_contract.py -q
python -m src.models.advanced.smoke --trials 35
```

The first command passed all seven checks. The second writes the full verification record, metrics and plots under `outputs/tuning/synthetic_smoke`. Use common real-data feature snapshots and trip folds to choose the classifier for PathSnap.
