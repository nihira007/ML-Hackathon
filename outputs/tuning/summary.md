# Person 3 · PathSnap experiments

## 3.2 Advanced classifiers

### LightGBM

**A. Model understanding.** Leaf-wise boosting captures interactions between cues such as distance and heading. It is efficient on tabular inputs, and unconstrained leaves can overfit.

**B. Training and configuration.** Binary log loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 30, "importance_type": "gain"}.

**C. Results.** Student-test Top-1 90.14%, Macro F1 0.9418, log loss 0.0654, precision 0.9067, recall 0.8887. Mean training-fold Macro F1 0.9532 and validation Macro F1 0.9390 ± 0.0190 give a gap of +0.0142. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### XGBoost

**A. Model understanding.** Regularised boosting combines several weak cues. Depth and shrinkage control complexity, and large ensembles may increase latency.

**B. Training and configuration.** Binary logistic loss. Initial settings {"n_estimators": 160, "learning_rate": 0.06, "max_depth": 5, "tree_method": "hist", "importance_type": "gain", "eval_metric": "logloss"}.

**C. Results.** Student-test Top-1 89.76%, Macro F1 0.9391, log loss 0.0672, precision 0.9022, recall 0.8837. Mean training-fold Macro F1 0.9471 and validation Macro F1 0.9391 ± 0.0189 give a gap of +0.0080. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

### MLP

**A. Model understanding.** A feed-forward network can learn smooth nonlinear combinations of geometry and movement. It is sensitive to scaling and training budget.

**B. Training and configuration.** Binary cross-entropy with an L2 penalty. Initial settings {"hidden_layer_sizes": [64, 32], "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 512, "solver": "adam", "activation": "relu"}. The network made 20 passes per fold.

**C. Results.** Student-test Top-1 89.23%, Macro F1 0.9335, log loss 0.0748, precision 0.9081, recall 0.8585. Mean training-fold Macro F1 0.9425 and validation Macro F1 0.9313 ± 0.0171 give a gap of +0.0112. The gap gives limited evidence of overfitting. Check the individual folds before drawing a conclusion.

## 4. Bayesian hyperparameter optimisation

**Model selection.** XGBoost led the ranking of 6 baselines. The rank weights were 40% CV Macro F1, 30% student-test Top-1, 20% student-test log loss and 10% inference time.

**Objective.** Minimize mean candidate log loss across five trip-grouped training folds. Log loss penalises overconfident errors in probabilities used as HMM emission scores.

**Trials and compute cost.** Seeded TPE completed 35 trials. Trial 12 achieved the lowest mean CV log loss (0.06941). Search took 585.63 wall-clock seconds.

**Best estimator parameters.** {"n_estimators": 200, "learning_rate": 0.03217754414868047, "subsample": 0.7736497298411852, "colsample_bytree": 0.9075739048232317, "reg_alpha": 0.761761659649017, "reg_lambda": 0.9836238771046582, "max_depth": 8, "min_child_weight": 1}.

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
| CV Macro F1 | 0.93909 | 0.93990 | +0.00081 |
| Student-test Top-1 | 0.89760 | 0.90188 | +0.00428 |
| Student-test Macro F1 | 0.93915 | 0.94271 | +0.00356 |
| Student-test log loss | 0.06722 | 0.06443 | -0.00279 |
| Inference seconds | 0.24933 | 0.65404 | +0.40471 |

Student-test log loss improved. Interpret that result alongside the changes in ranking quality and latency. The tuning CV scores were used to choose the hyperparameters.
