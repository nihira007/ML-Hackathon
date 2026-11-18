# Person 3 · PathSnap experiments

## 3.2 Advanced classifiers

### LightGBM

**A. Model understanding.** Leaf-wise boosting captures interactions between cues such as distance and heading. It is efficient on tabular inputs, and unconstrained leaves can overfit.

**B. Training and configuration.** Binary log loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 30, "importance_type": "gain"}.

**C. Results.** Student-test Top-1 88.85%, Macro F1 0.9344, log loss 0.0718, precision 0.8954, recall 0.8738. Mean training-fold Macro F1 0.9403 and validation Macro F1 0.9267 ± 0.0195 give a gap of +0.0136. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### XGBoost

**A. Model understanding.** Regularised boosting combines several weak cues. Depth and shrinkage control complexity, and large ensembles may increase latency.

**B. Training and configuration.** Binary logistic loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "max_depth": 5, "tree_method": "hist", "importance_type": "gain", "eval_metric": "logloss"}.

**C. Results.** Student-test Top-1 88.07%, Macro F1 0.9310, log loss 0.0746, precision 0.8903, recall 0.8668. Mean training-fold Macro F1 0.9357 and validation Macro F1 0.9287 ± 0.0189 give a gap of +0.0070. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### MLP

**A. Model understanding.** A feed-forward network can learn smooth nonlinear combinations of geometry and movement. It is sensitive to scaling and training budget.

**B. Training and configuration.** Binary cross-entropy with an L2 penalty. Initial settings {"hidden_layer_sizes": [64, 32], "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 512, "solver": "adam", "activation": "relu"}. The network made 20 passes per fold.

**C. Results.** Student-test Top-1 87.45%, Macro F1 0.9238, log loss 0.0822, precision 0.8853, recall 0.8469. Mean training-fold Macro F1 0.9312 and validation Macro F1 0.9191 ± 0.0186 give a gap of +0.0121. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

## 4. Bayesian hyperparameter optimisation

**Model selection.** XGBoost led the ranking of 6 baselines. The rank weights were 40% CV Macro F1, 30% student-test Top-1, 20% student-test log loss and 10% inference time.

**Objective.** Minimize mean candidate log loss across five trip-grouped training folds. Log loss penalises overconfident errors in probabilities used as HMM emission scores.

**Trials and compute cost.** Seeded TPE completed 35 trials. Trial 10 achieved the lowest mean CV log loss (0.07978). Search took 333.19 wall-clock seconds.

**Best estimator parameters.** {"n_estimators": 200, "learning_rate": 0.04649128682881624, "subsample": 0.9313938915639582, "colsample_bytree": 0.7958334304404576, "reg_alpha": 3.058050323757543e-05, "reg_lambda": 2.544769802734904, "max_depth": 7, "min_child_weight": 3}.

| Parameter | Distribution | Settings |
| --- | --- | --- |
| n_estimators | IntDistribution | {"low": 80, "high": 320, "step": 40, "log": false} |
| learning_rate | FloatDistribution | {"low": 0.015, "high": 0.2, "log": true, "step": null} |
| subsample | FloatDistribution | {"low": 0.65, "high": 1.0, "log": false, "step": null} |
| colsample_bytree | FloatDistribution | {"low": 0.65, "high": 1.0, "log": false, "step": null} |
| reg_alpha | FloatDistribution | {"low": 1e-05, "high": 5.0, "log": true, "step": null} |
| reg_lambda | FloatDistribution | {"low": 0.001, "high": 10.0, "log": true, "step": null} |
| max_depth | IntDistribution | {"low": 3, "high": 8, "step": 1, "log": false} |
| min_child_weight | IntDistribution | {"low": 1, "high": 12, "step": 1, "log": false} |

The search varies capacity, learning, regularisation and sampling where applicable.

| Metric | Untuned | Tuned | Change |
| --- | ---: | ---: | ---: |
| CV Macro F1 | 0.92869 | 0.92944 | +0.00075 |
| Student-test Top-1 | 0.88074 | 0.88575 | +0.00501 |
| Student-test Macro F1 | 0.93096 | 0.93290 | +0.00194 |
| Student-test log loss | 0.07462 | 0.07171 | -0.00291 |
| Inference seconds | 0.13140 | 0.21893 | +0.08753 |

Student-test log loss improved. Interpret that result alongside the changes in ranking quality and latency. The tuning CV scores were used to choose the hyperparameters.
