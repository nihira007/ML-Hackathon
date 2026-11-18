# Person 3 · PathSnap experiments

The benchmark uses 80 real training trips and 10 real development trips. Five-fold validation keeps whole trips together. Development metrics inform model selection, and Optuna scores its trials using the training partition only.

## 3.1 Baseline comparison

| Model | CV Macro F1 | Student-test Top-1 | Student-test log loss | Inference (s) | Rank score |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9286 | 88.43% | 0.0735 | 0.1574 | 1.80 |
| LightGBM | 0.9283 | 88.72% | 0.0752 | 0.4559 | 2.20 |
| MLP | 0.9207 | 88.82% | 0.0773 | 0.0779 | 2.30 |
| Random Forest | 0.9137 | 87.23% | 0.1164 | 0.9495 | 4.10 |
| Logistic Regression | 0.8549 | 86.72% | 0.1826 | 0.0540 | 5.00 |
| Distance-Weighted KNN | 0.8805 | 79.39% | 0.2700 | 3.1698 | 5.60 |

## 3.2 Advanced classifiers

### LightGBM

**A. Model understanding.** Leaf-wise boosting captures interactions between cues such as distance and heading. It is efficient on tabular inputs, and unconstrained leaves can overfit.

**B. Training and configuration.** Binary log loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 30, "importance_type": "gain"}.

**C. Results.** Student-test Top-1 88.72%, Macro F1 0.9323, log loss 0.0752, precision 0.8876, recall 0.8741. Mean training-fold Macro F1 0.9423 and validation Macro F1 0.9283 ± 0.0186 give a gap of +0.0140. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### XGBoost

**A. Model understanding.** Regularised boosting combines several weak cues. Depth and shrinkage control complexity, and large ensembles may increase latency.

**B. Training and configuration.** Binary logistic loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "max_depth": 5, "tree_method": "hist", "importance_type": "gain", "eval_metric": "logloss"}.

**C. Results.** Student-test Top-1 88.43%, Macro F1 0.9325, log loss 0.0735, precision 0.8898, recall 0.8726. Mean training-fold Macro F1 0.9366 and validation Macro F1 0.9286 ± 0.0188 give a gap of +0.0080. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### MLP

**A. Model understanding.** A feed-forward network can learn smooth nonlinear combinations of geometry and movement. It is sensitive to scaling and training budget.

**B. Training and configuration.** Binary cross-entropy with an L2 penalty. Initial settings {"hidden_layer_sizes": [64, 32], "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 512, "solver": "adam", "activation": "relu"}. The network made 20 passes per fold.

**C. Results.** Student-test Top-1 88.82%, Macro F1 0.9319, log loss 0.0773, precision 0.9021, recall 0.8587. Mean training-fold Macro F1 0.9427 and validation Macro F1 0.9207 ± 0.0192 give a gap of +0.0220. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

The mean validation loss was lowest at epoch 3 (0.0877). At epoch 20, mean training and validation loss were 0.0642 and 0.0925.

## 4. Bayesian hyperparameter optimisation

**Model selection.** XGBoost led the ranking of 6 baselines. The rank weights were 40% CV Macro F1, 30% student-test Top-1, 20% student-test log loss and 10% inference time.

**Objective.** Minimize mean candidate log loss across five trip-grouped training folds. Log loss penalises overconfident errors in probabilities used as HMM emission scores.

**Trials and compute cost.** Seeded TPE completed 35 trials. Trial 27 achieved the lowest mean CV log loss (0.07738). Search took 1584.84 wall-clock seconds.

**Best estimator parameters.** {"n_estimators": 160, "learning_rate": 0.053026243475099384, "subsample": 0.6802891554540573, "colsample_bytree": 0.9697753779477274, "reg_alpha": 0.7079704662923327, "reg_lambda": 0.35442871419835503, "max_depth": 8, "min_child_weight": 6}.

| Parameter | Distribution | Settings |
| --- | --- | --- |
| n_estimators | IntDistribution | {"log": false, "step": 40, "low": 80, "high": 320} |
| learning_rate | FloatDistribution | {"step": null, "low": 0.015, "high": 0.2, "log": true} |
| subsample | FloatDistribution | {"step": null, "low": 0.65, "high": 1.0, "log": false} |
| colsample_bytree | FloatDistribution | {"step": null, "low": 0.65, "high": 1.0, "log": false} |
| reg_alpha | FloatDistribution | {"step": null, "low": 1e-05, "high": 5.0, "log": true} |
| reg_lambda | FloatDistribution | {"step": null, "low": 0.001, "high": 10.0, "log": true} |
| max_depth | IntDistribution | {"log": false, "step": 1, "low": 3, "high": 8} |
| min_child_weight | IntDistribution | {"log": false, "step": 1, "low": 1, "high": 12} |

The search varies capacity, learning, regularisation and sampling where applicable.

| Metric | Untuned | Tuned | Change |
| --- | ---: | ---: | ---: |
| CV Macro F1 | 0.92864 | 0.93021 | +0.00157 |
| Student-test Top-1 | 0.88429 | 0.89139 | +0.00710 |
| Student-test Macro F1 | 0.93246 | 0.93648 | +0.00401 |
| Student-test log loss | 0.07349 | 0.06976 | -0.00373 |
| Inference seconds | 0.13150 | 0.21750 | +0.08600 |

Student-test log loss improved. Interpret that result alongside the changes in ranking quality and latency. The tuning CV scores were used to choose the hyperparameters.
