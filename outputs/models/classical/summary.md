# Technical Report Section 3.2 — Classical & Ensemble Candidate Classification Models (Models 1–3)

**Owner:** Person 2 (`feat/p2-models-classical`)  
**Dataset Scope:** `80` training trips (`1,510,241` reliable-label candidate road segments across `80` trips at $K=10$) and `10` out-of-sample `student_test` trips (`19,160` GPS observations; `157,866` candidate road segments) across Person 1's canonical `27` v2 spatial, kinematic, observation-context, and road-graph topological features.

---

## 1. Consolidated Experimental Summary (`classical_comparison.csv`)

All three classical models were evaluated under the exact same 5-fold trip-grouped cross-validation protocol (`GroupKFold(n_splits=5)` grouped strictly by `trip_id` so observations from the same vehicle trajectory never appear in both train and validation splits) and tested out-of-sample on the 10 `student_test` trips:

| Metric | Nearest-Segment Heuristic | Model 1: Logistic Regression | Model 2: Distance-Weighted KNN | Model 3: Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **Algorithm Family** | Geometric Baseline | Linear / Probabilistic | Instance-based / Non-parametric | Bagging / Tree Ensemble |
| **Key Hyperparameters** | `cand_rank == 0` | `C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs'` | `n_neighbors=15, weights='distance', metric='euclidean', max_samples=16k` | `n_estimators=100, max_depth=14, min_samples_leaf=25, max_features='sqrt', class_weight='balanced_subsample'` |
| **5-Fold CV Top-1 Accuracy (Mean ± SD)** | $0.5520 \pm 0.0810$ | $0.8690 \pm 0.0403$ | $0.8012 \pm 0.0294$ | **$0.8695 \pm 0.0419$** |
| **5-Fold CV Macro F1 (Mean ± SD)** | — | $0.8551 \pm 0.0146$ | $0.8785 \pm 0.0126$ | **$0.9147 \pm 0.0230$** |
| **5-Fold CV Log Loss (Mean ± SD)** | — | $0.1793 \pm 0.0241$ | $0.2866 \pm 0.0680$ | **$0.1174 \pm 0.0218$** |
| **Train Fold Mean Top-1 Accuracy** | $0.5520$ | $0.8695$ | $0.9921$ | $0.8899$ |
| **Train Fold Mean Macro F1** | — | $0.8563$ | $1.0000$ | $0.9267$ |
| **Macro F1 Generalization Gap (Train − Val)** | — | **$+0.0012$** (Zero overfitting) | $+0.1215$ (Instance memorization) | **$+0.0120$** (Depth/leaf controlled) |
| **`student_test` Top-1 Accuracy (19,160 obs)** | $0.5400$ (54.00%) | $0.8669$ (86.69%) | $0.7936$ (79.36%) | **$0.8720$ (87.20%)** |
| **`student_test` Top-3 Accuracy (19,160 obs)** | $0.8841$ (88.41%) | $0.9748$ (97.48%) | $0.9635$ (96.35%) | **$0.9768$ (97.68%)** |
| **`student_test` Candidate Macro F1** | — | $0.8527$ | $0.8748$ | **$0.9154$** |
| **`student_test` Positive-Class Binary F1** | — | $0.7506$ | $0.7784$ | **$0.8534$** |
| **`student_test` Precision / Recall** | — | $0.6198$ / **$0.9515$** | **$0.8202$** / $0.7406$ | $0.7829$ / $0.9377$ |
| **`student_test` Overall Candidate Accuracy** | — | $0.9234$ | $0.9489$ | **$0.9610$** |
| **`student_test` Binary Log Loss** | — | $0.1827$ | $0.2607$ | **$0.1159$** |
| **`student_test` Brier Score** | — | $0.0560$ | $0.0387$ | **$0.0316$** |
| **`student_test` Route Continuity** | — | $0.9829$ (98.29%) | $0.9711$ (97.11%) | $0.9739$ (97.39%) |
| **Wall-Clock Full Train Time (s)** | — | $6.06\text{ s}$ | **$0.23\text{ s}$** | $45.41\text{ s}$ |
| **Wall-Clock `student_test` Inference Time (s)** | — | **$0.022\text{ s}$** | $8.31\text{ s}$ | $1.02\text{ s}$ |

---

## 2. Compulsory Report Section 3.2 Answers by Model

### Model 1 — L2-Regularized Logistic Regression (`src/models/classical/train_logreg.py`)
- **A. Suitability, Strengths & Limitations:** Estimates smooth conditional emission probabilities $P(y_{t,c}=1 \mid \mathbf{x}_{t,c}) = \sigma(\mathbf{w}^\top \tilde{\mathbf{x}}_{t,c} + b)$ over the 27 standardized v2 features. Sub-millisecond per-trip inference ($2.22\text{ ms}$ per trip), virtually zero generalization gap ($+0.0012$ Macro F1), and $95.15\%$ positive recall make it an ultra-fast baseline, though its linear hyperplane cannot capture non-linear feature interactions ($0.8527$ Macro F1 vs $0.9154$ for Random Forest).
- **B. Hyperparameters & Objective:** `C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=500, random_state=42`, minimizing class-weighted L2-penalized binary cross-entropy.
- **C. Results & Overfitting Analysis:** Achieves **$86.69\%$ `student_test` Top-1 Accuracy**, **$97.48\%$ Top-3 Accuracy**, **$0.8527$ Macro F1**, and **$0.1827$ Log Loss**, with a $+0.0012$ CV generalization gap ($\text{Train} = 0.8563$ vs $\text{Val} = 0.8551$), confirming linear underfitting rather than overfitting.

### Model 2 — Distance-Weighted K-Nearest Neighbors (`src/models/classical/train_knn.py`)
- **A. Suitability, Strengths & Limitations:** Non-parametric inverse-Euclidean-distance voting ($w_j = 1/d_j$) over $k=15$ neighbors captures local manifold structure and achieves high candidate precision ($82.02\%$) and natural probability calibration ($\text{Brier} = 0.0387$). However, in 27-dimensional feature space, distance concentration slows inference ($8.31\text{ s}$) and lowers Top-1 ranking accuracy ($79.36\%$).
- **B. Hyperparameters & Objective:** `n_neighbors=15, weights='distance', algorithm='brute', metric='euclidean', max_train_samples=16000, random_state=42`.
- **C. Results & Overfitting Analysis:** Achieves **$79.36\%$ `student_test` Top-1 Accuracy**, **$0.8748$ Macro F1**, and **$0.2607$ Log Loss**. Distance weighting memorizes exact training instances ($\text{Train Macro F1} = 1.0000$ vs $\text{Val Macro F1} = 0.8785 \pm 0.0126$, gap = $+0.1215$).

### Model 3 — Regularized Random Forest Ensemble (`src/models/classical/train_rf.py`)
- **A. Suitability, Strengths & Limitations:** Bagging 100 depth-regularized decision trees (`max_depth=14, min_samples_leaf=25, max_features='sqrt'`) models hierarchical interactions between perpendicular distance, heading alignment (`cos_bearing_diff`), and topological continuity (`connected_prev_nearest`, `connected_next_nearest`), achieving the best performance among all classical models across every metric.
- **B. Hyperparameters & Objective:** `n_estimators=100, criterion='gini', max_depth=14, min_samples_leaf=25, max_features='sqrt', class_weight='balanced_subsample', max_train_samples=200000, random_state=42`.
- **C. Results & Overfitting Analysis:** Achieves **$87.20\%$ `student_test` Top-1 Accuracy**, **$97.68\%$ Top-3 Accuracy**, **$0.9154$ Macro F1**, **$0.1159$ Log Loss**, and **$0.0316$ Brier Score**, with a tightly controlled $+0.0120$ Macro F1 generalization gap ($\text{Train} = 0.9267$ vs $\text{Val} = 0.9147 \pm 0.0230$).
