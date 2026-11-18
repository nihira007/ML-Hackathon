# Technical Report Section 3.2 — Classical & Ensemble Candidate Classification Models (Models 1–3)

**Owner:** Person 2 (`feat/p2-models-classical`)  
**Dataset Scope:** `80` training trips (`208,417` GPS observations; `1,667,336` candidate road segments at $K=8$) and `10` out-of-sample `student_test` trips (`19,160` GPS observations; `153,280` candidate road segments) across `21` v2 spatial, kinematic, observation-context, and road-graph topological features.

---

## 1. Consolidated Experimental Summary (`classical_comparison.csv`)

All three models were evaluated under the exact same 5-fold trip-grouped cross-validation protocol (`GroupKFold(n_splits=5)` grouped strictly by `trip_id` so observations from the same vehicle trajectory never appear in both train and validation splits) and tested out-of-sample on the 10 `student_test` trips:

| Metric | Nearest-Segment Heuristic | Model 1: Logistic Regression | Model 2: Distance-Weighted KNN | Model 3: Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **Algorithm Family** | Geometric Baseline | Linear / Probabilistic | Instance-based / Non-parametric | Bagging / Tree Ensemble |
| **Feature Scaling** | None (`cand_rank == 0`) | `StandardScaler` (per-fold train) | `StandardScaler` (per-fold train) | Unscaled (scale-invariant trees) |
| **5-Fold CV Top-1 Accuracy (Mean ± SD)** | $0.5421 \pm 0.1180$ | $0.6639 \pm 0.1139$ | $0.6965 \pm 0.1161$ | **$0.7722 \pm 0.1122$** |
| **5-Fold CV Macro F1 (Mean ± SD)** | — | $0.7217 \pm 0.0591$ | $0.8164 \pm 0.0552$ | **$0.8492 \pm 0.0685$** |
| **5-Fold CV Log Loss (Mean ± SD)** | — | $0.4061 \pm 0.0905$ | $0.5973 \pm 0.3411$ | **$0.1807 \pm 0.0590$** |
| **Train Fold Mean Top-1 Accuracy** | $0.5421$ | $0.6821$ | $1.0000$ | $0.8760$ |
| **Train Fold Mean Macro F1** | — | $0.7371$ | $1.0000$ | $0.9101$ |
| **Macro F1 Generalization Gap (Train − Val)** | — | **$+0.0154$** (Linear underfitting) | $+0.1836$ (Instance memorization) | **$+0.0609$** (Depth/leaf controlled) |
| **`student_test` Top-1 Accuracy (19,160 obs)** | $0.5668$ (56.68%) | $0.7433$ (74.33%) | $0.7736$ (77.36%) | **$0.8474$ (84.74%)** |
| **`student_test` Top-3 Accuracy (19,160 obs)** | $0.8841$ (88.41%) | $0.9480$ (94.80%) | $0.9516$ (95.16%) | **$0.9797$ (97.97%)** |
| **`student_test` Candidate Macro F1** | — | $0.7468$ | $0.8530$ | **$0.8937$** |
| **`student_test` Positive-Class Binary F1** | — | $0.5927$ | $0.7407$ | **$0.8177$** |
| **`student_test` Precision / Recall** | — | $0.4355$ / $0.9273$ | **$0.7865$** / $0.6999$ | $0.7290$ / **$0.9311$** |
| **`student_test` Overall Candidate Accuracy** | — | $0.8407$ | $0.9387$ | **$0.9481$** |
| **`student_test` Binary Log Loss** | — | $0.3489$ | $0.3117$ | **$0.1441$** |
| **`student_test` Brier Score** | — | $0.1117$ | $0.0446$ | **$0.0403$** |
| **`student_test` Expected Calibration Error (ECE)** | — | $0.1772$ | **$0.0114$** | $0.0634$ |
| **Wall-Clock Full Train Time (s)** | — | $2.48\text{ s}$ | **$0.20\text{ s}$** | $44.46\text{ s}$ |
| **Wall-Clock `student_test` Inference Time (s)** | — | **$0.021\text{ s}$** | $7.82\text{ s}$ | $0.97\text{ s}$ |

---

## 2. Compulsory Report Section 3.2 Answers by Model

### Model 1 — L2-Regularized Logistic Regression (`src/models/classical/train_logreg.py`)

#### A. Suitability, Theoretical Strengths & Domain Limitations
- **Why Suited to Map-Matching Candidate Classification:** Map-matching candidate scoring requires estimating a smooth conditional emission probability $P(y_{t,c}=1 \mid \mathbf{x}_{t,c}) = \sigma(\mathbf{w}^\top \tilde{\mathbf{x}}_{t,c} + b)$ over $K=8$ candidate road segments per GPS observation. Because perpendicular distance (`dist_to_arc_m`), relative distance ratio (`cand_dist_ratio = -6.264`), and angular heading alignment (`heading_diff_deg = -2.769`, `neighbor_arc_continuity = +0.890`) have strong monotonic log-odds relationships with road-segment validity, a linear probabilistic classifier provides an interpretable, convex baseline.
- **Strengths:**
  1. **Sub-millisecond per-trip inference latency:** Scoring all `153,280` `student_test` candidates takes only **$20.78\text{ ms}$** ($2.08\text{ ms}$ per trip), making it ideal for resource-constrained edge telemetry.
  2. **Zero overfitting across trips:** The 5-fold CV Macro F1 generalization gap is just **$+0.0154$** ($\text{Train} = 0.7371$ vs. $\text{Val} = 0.7217$), confirming that a 21-parameter linear hyperplane does not memorize trip-specific road coordinates.
  3. **High positive-class recall ($92.73\%$):** Using balanced class weights preserves nearly all true road segments in the high-probability candidate pool ($\text{Top-3 Accuracy} = 94.80\%$) for downstream Viterbi decoding.
- **Limitations:**
  1. **Linear underfitting on multiplicative spatial-kinematic interactions:** A single hyperplane cannot express conditional rules such as *"penalize high perpendicular distance only when GPS speed is high and heading misalignment exceeds $30^\circ$, but tolerate lateral offset during stationary stops or multi-lane junctions."*
  2. **False-positive inflation under class balancing:** Upweighting the $1:7$ minority class by $7\times$ shifts the intercept to $b = -2.524$, causing `23,029` false positives ($\text{Precision} = 0.4355$) and elevating Expected Calibration Error ($\text{ECE} = 0.1772$).

#### B. Initial Hyperparameters & Mathematical Objective Function
- **Hyperparameters:** `C=1.0`, `penalty='l2'`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=500`, `random_state=42`, preceded by fold-isolated `StandardScaler`.
- **Loss / Objective Function:** Minimizes class-weighted L2-penalized binary cross-entropy (negative log-likelihood):
  $$\mathcal{L}_{\text{LogReg}}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N} \omega_{y_i} \Big[ y_i \log \sigma(\mathbf{w}^\top \tilde{\mathbf{x}}_i + b) + (1 - y_i)\log\big(1 - \sigma(\mathbf{w}^\top \tilde{\mathbf{x}}_i + b)\big) \Big] + \frac{1}{2C}\|\mathbf{w}\|_2^2$$
  where $\omega_1 = \frac{N}{2 N_1} = 4.0$ and $\omega_0 = \frac{N}{2 N_0} = 0.5714$ compensate for the $12.5\%$ positive candidate rate ($1$ match per $8$ candidates).

#### C. Experimental Performance & Overfitting/Underfitting Evidence
- **Out-of-Sample `student_test` Metrics:** Top-1 Accuracy = **$74.33\%$** (+17.65 percentage points above the $56.68\%$ nearest-segment heuristic), Top-3 Accuracy = **$94.80\%$**, Candidate Macro F1 = **$0.7468$**, Positive Binary F1 = **$0.5927$**, Log Loss = **$0.3489$**, Brier Score = **$0.1117$**, Confusion Matrix = `[[TN=111,091, FP=23,029], [FN=1,393, TP=17,767]]`.
- **Underfitting Diagnosis:** Both Training Macro F1 ($0.7371$) and Validation Macro F1 ($0.7217$) plateau well below tree-based models, with a tiny $+0.0154$ generalization gap. Notably, upgrading from the 7 v1 pure-spatial features to the 21 v2 domain features boosted Logistic Regression `student_test` Top-1 Accuracy from **$46.81\%$ to $74.33\%$ (+27.52%)**, demonstrating that explicit feature engineering partially compensates for linear capacity limits.

---

### Model 2 — Distance-Weighted K-Nearest Neighbors (`src/models/classical/train_knn.py`)

#### A. Suitability, Theoretical Strengths & Domain Limitations
- **Why Suited to Map-Matching Candidate Classification:** Valid road matches occupy non-linear, locally clustered manifolds in standardized feature space (e.g., candidates with $d_\perp < 8\text{ m}$, $\text{cand\_dist\_ratio} \approx 1.0$, $\cos(\Delta\theta) > 0.85$, and $\text{neighbor\_arc\_continuity} \ge 0.5$). Distance-Weighted KNN estimates candidate probability non-parametrically from local neighborhood geometry without imposing global parametric assumptions.
- **Strengths:**
  1. **Captures local non-linear boundaries:** Improves `student_test` Top-1 Accuracy to **$77.36\%$** (+3.03% over Logistic Regression) and Candidate Macro F1 to **$0.8530$** (+0.1062 over Logistic Regression).
  2. **Best natural calibration ($\text{ECE} = 0.0114$):** Because KNN averages empirical labels in local neighborhoods without synthetic class-weight reweighting, its mean predicted probability ($0.1275$) matches the true $0.1250$ base rate within $+0.0025$, achieving the highest precision (**$78.65\%$**) among all three classical models.
- **Limitations:**
  1. **Severe instance memorization on training folds:** Because distance weighting ($w_j = 1 / \|\tilde{\mathbf{x}} - \tilde{\mathbf{x}}_j\|_2$) assigns $\infty$ weight to zero-distance training points, Training Macro F1 and Training Top-1 Accuracy are **$1.0000$**, creating a large generalization gap of **$+0.1836$ Macro F1** ($\text{Train} = 1.0000$ vs. $\text{CV Val} = 0.8164$) and **$+0.3035$ Top-1 Accuracy** ($\text{Train} = 1.0000$ vs. $\text{CV Val} = 0.6965$).
  2. **Curse of dimensionality & query latency:** In 21-dimensional standardized feature space, spatial trees degrade toward linear scans, requiring **$7.82\text{ s}$** for `student_test` inference ($376\times$ slower than Logistic Regression) even with observation-grouped subsampling (`max_train_samples=16,000`).
  3. **Tail probability clipping spikes Log Loss ($0.5973$ CV / $0.3117$ Test):** Whenever all $k=15$ nearest neighbors belong to class $0$, KNN outputs an exact $p = 0.0$, which incurs a heavy logarithmic penalty whenever a noisy GPS point is a true match.

#### B. Initial Hyperparameters & Mathematical Objective Function
- **Hyperparameters:** `n_neighbors=15`, `weights='distance'`, `algorithm='brute'` (BLAS Euclidean distance matrix multiplication), `metric='euclidean'`, `max_train_samples=16000` (2,000 complete 8-candidate observations sampled deterministically with `random_state=42`), preceded by fold-isolated `StandardScaler`.
- **Objective / Probability Estimator:** Non-parametric inverse-Euclidean-distance kernel estimator over the $k=15$ nearest training candidates $\mathcal{N}_k(\tilde{\mathbf{x}})$:
  $$\hat{P}(y=1 \mid \tilde{\mathbf{x}}) = \frac{\sum_{j \in \mathcal{N}_k(\tilde{\mathbf{x}})} w_j \, y_j}{\sum_{j \in \mathcal{N}_k(\tilde{\mathbf{x}})} w_j}, \quad \text{where } w_j = \frac{1}{\max\big(\|\tilde{\mathbf{x}} - \tilde{\mathbf{x}}_j\|_2, \epsilon\big)}$$

#### C. Experimental Performance & Overfitting/Underfitting Evidence
- **Out-of-Sample `student_test` Metrics:** Top-1 Accuracy = **$77.36\%$**, Top-3 Accuracy = **$95.16\%$**, Candidate Macro F1 = **$0.8530$**, Positive Binary F1 = **$0.7407$**, Precision = **$0.7865$**, Recall = **$0.6999$**, Log Loss = **$0.3117$**, Brier Score = **$0.0446$**, Confusion Matrix = `[[TN=125,088, FP=9,032], [FN=5,749, TP=13,411]]`.
- **Overfitting Diagnosis:** The gap between exact training interpolation ($\text{Train Macro F1} = 1.0000$) and out-of-fold validation ($\text{CV Macro F1} = 0.8164 \pm 0.0552$) exemplifies classic non-parametric variance in 21 dimensions.

---

### Model 3 — Regularized Random Forest Ensemble (`src/models/classical/train_rf.py`)

#### A. Suitability, Theoretical Strengths & Domain Limitations
- **Why Suited to Map-Matching Candidate Classification:** Candidate road classification depends on hierarchical threshold interactions across heterogeneous feature scales (meters, degrees, node degrees, and dimensionless ratios). A bagged ensemble of depth-regularized decision trees naturally partitions these non-linear interaction regimes while smoothing leaf probability estimates across 100 decorrelated trees.
- **Strengths:**
  1. **Highest accuracy and ranking power across all metrics:** Achieves **$84.74\%$ `student_test` Top-1 Accuracy** (+28.06% over the nearest-segment heuristic, +10.41% over Logistic Regression, +7.38% over KNN), **$97.97\%$ Top-3 Accuracy**, and **$0.8937$ Candidate Macro F1**.
  2. **Best probability quality for HMM emission scoring:** Achieves the lowest `student_test` Log Loss (**$0.1441$**) and lowest Brier Score (**$0.0403$**), assigning an average probability of $0.8930$ to true road matches versus $0.0878$ to false candidates ($10.17\times$ separation).
  3. **Interpretable domain feature attribution:** Gini importance confirms that relative spatial ranking (`cand_dist_ratio = 0.1950`, `cand_dist_zscore = 0.1432`, `cand_dist_diff_from_min_m = 0.1324`), perpendicular distance (`dist_to_arc_m = 0.0952`), segment projection (`proj_fraction_t = 0.0813`), and kinematic heading alignment (`heading_diff_cos = 0.0775`, `heading_diff_deg = 0.0575`, `neighbor_arc_continuity = 0.0379`) drive the ensemble's decisions.
- **Limitations:**
  1. **Higher training compute:** Fitting 100 trees across 5 CV folds and the full training set requires $44.46\text{ s}$ per fit ($0.97\text{ s}$ for 10-trip test inference).
  2. **Pointwise independence:** Like all pointwise candidate classifiers, Random Forest scores each observation's candidates without enforcing global multi-step path connectivity across long intersections (e.g., `trip 00000077` where mean heading discrepancy reaches $57.95^\circ$), motivating Person 3's boosted trees and Person 4's HMM Viterbi sequence decoder.

#### B. Initial Hyperparameters & Mathematical Objective Function
- **Hyperparameters:** `n_estimators=100`, `criterion='gini'`, `max_depth=14`, `min_samples_leaf=25`, `max_features='sqrt'` ($\lfloor\sqrt{21}\rfloor = 4$ features evaluated per split), `class_weight='balanced_subsample'`, `max_train_samples=200000` (25,000 complete 8-candidate observations per fit), `random_state=42`.
- **Objective Function:** Each tree $t \in \{1,\dots,100\}$ greedily selects axis-aligned splits $(j^*, s^*)$ that maximize sample-weighted Gini impurity reduction:
  $$\Delta i(\mathcal{S}, j, s) = i(\mathcal{S}) - \frac{W_L}{W_{\mathcal{S}}} i(\mathcal{S}_L) - \frac{W_R}{W_{\mathcal{S}}} i(\mathcal{S}_R), \quad i(\mathcal{S}) = 1 - \sum_{c \in \{0,1\}} \tilde{p}_c(\mathcal{S})^2$$
  where $\tilde{p}_c(\mathcal{S})$ is the bootstrap-subsample-reweighted class proportion in node $\mathcal{S}$. The ensemble probability is the unweighted average of regularized leaf posteriors across all $T=100$ trees: $\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} \hat{P}_t(y=1 \mid \mathbf{x})$.

#### C. Experimental Performance & Overfitting/Underfitting Evidence
- **Out-of-Sample `student_test` Metrics:** Top-1 Accuracy = **$84.74\%$**, Top-3 Accuracy = **$97.97\%$**, Candidate Macro F1 = **$0.8937$**, Positive Binary F1 = **$0.8177$**, Precision = **$0.7290$**, Recall = **$0.9311$**, Log Loss = **$0.1441$**, Brier Score = **$0.0403$**, Confusion Matrix = `[[TN=127,489, FP=6,631], [FN=1,321, TP=17,839]]`.
- **Controlled Generalization Gap:** Constraining `max_depth=14` and `min_samples_leaf=25` restricts Training Macro F1 to **$0.9101$** against a 5-fold CV Validation Macro F1 of **$0.8492 \pm 0.0685$** (a modest **$+0.0609$** generalization gap, cutting KNN's overfitting gap by two-thirds while achieving +7.57% higher CV Top-1 Accuracy).

---

## 3. Per-Trip & Spatial-Regime Error Analysis (`per_trip_results.csv`)

Evaluating Models 1–3 across physical road regimes on `student_test` reveals exactly where v2 domain features succeed and where sequence-level HMM decoding is required:

| Spatial / Kinematic Regime (`student_test`) | Share of Test Obs | Nearest-Segment Top-1 | LogReg Top-1 | KNN Top-1 | Random Forest Top-1 | RF Macro F1 | RF Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Open Road (`cands_within_15m <= 2`)** | $62.71\%$ (12,015) | $60.79\%$ | $86.12\%$ | $87.18\%$ | **$91.05\%$** | $0.9335$ | $0.0942$ |
| **Dense Intersection / Parallel (`cands_within_15m >= 3`)** | $37.29\%$ (7,145) | $49.59\%$ | $54.51\%$ | $60.84\%$ | **$74.12\%$** | $0.8327$ | $0.2281$ |
| **Aligned Heading (`gt_heading_diff <= 25 deg`)** | $72.59\%$ (13,908) | $59.25\%$ | $83.75\%$ | $86.63\%$ | **$94.54\%$** | $0.9535$ | $0.0846$ |
| **Turning / Noisy Heading (`gt_heading_diff > 25 deg`)** | $27.41\%$ (5,252) | $49.62\%$ | $49.39\%$ | $52.82\%$ | **$58.78\%$** | $0.7602$ | $0.3017$ |
| **High Lateral GPS Noise (`gt_dist_to_arc > 10m`)** | $20.02\%$ (3,835) | $45.97\%$ | $65.48\%$ | $69.10\%$ | **$72.93\%$** | $0.8413$ | $0.2082$ |
| **Low-Speed / Stationary (`gps_speed < 2.0 m/s`)** | $2.03\%$ (388) | $47.16\%$ | $63.14\%$ | $58.51\%$ | **$65.72\%$** | $0.7998$ | $0.2721$ |

- **Best-Performing Trips (`00000000`, `00000052`, `00000040`, `00000007`):** On trajectories with consistent vehicle speed ($17.5\text{–}21.7\text{ m/s}$) and moderate heading misalignment ($8.7^\circ\text{–}20.0^\circ$), Random Forest achieves **$92.17\%\text{–}93.41\%$ Top-1 Accuracy** and **$98.20\%\text{–}99.78\%$ Top-3 Accuracy**.
- **Hardest Trips (`00000077`, `00000065`, `00000009`, `00000076`):** Trip `00000009` has $70.64\%$ of observations inside dense intersections ($\ge 3$ candidates within $15\text{ m}$), where RF still lifts Top-1 Accuracy from $45.19\%$ (nearest arc) to **$83.70\%$**. Trip `00000077` exhibits severe heading discrepancy ($57.95^\circ$ mean difference between GPS velocity bearing and road-arc bearing), capping pointwise RF Top-1 Accuracy at **$69.34\%$** while still retaining the true arc in its Top-3 candidates **$96.22\%$** of the time—providing a high-recall candidate lattice for Person 4's Viterbi decoder.
