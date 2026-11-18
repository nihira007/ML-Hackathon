# Classical & Ensemble Models 1–3 (Person 2 — Phases 2, 3 & 4)

**Owner:** Person 2 (`feat/p2-models-classical`)  
**Scope:** Model 1 (Logistic Regression), Model 2 (Distance-Weighted KNN), and Model 3 (Random Forest Ensemble) training on 21 v2 domain features, 5-fold `GroupKFold` cross-validation by `trip_id`, probability calibration audit, per-trip error analysis across all 10 `student_test` trips, and prediction parquets export for HMM / Viterbi decoding.

---

## 1. Directory & Module Architecture

```
src/models/classical/
├── __init__.py                         # Package exports for Models 1–3 and Phase 4 audit utilities
├── data_utils.py                       # 21-feature v2 spatial/kinematic/context/topological builder & loader
├── train_logreg.py                     # Model 1: L2 Logistic Regression (balanced class weights)
├── train_knn.py                        # Model 2: Distance-Weighted KNN (k=15, Euclidean BLAS)
├── train_rf.py                         # Model 3: Regularized Random Forest Ensemble (100 trees, depth=14)
├── run_all_classical.py                # Unified runner evaluating Models 1–3 on identical 5-fold splits
├── calibration_and_error_analysis.py   # Phase 4 emission calibration curves & per-trip error analysis
└── README.md                           # Documentation, benchmarks, overfitting, and error analysis
```

---

## 2. Phase 3 Unified Benchmark on 21 v2 Domain Features (`classical_comparison.csv`)

All 3 models were trained and evaluated on the exact same 21 v2 features (`7` spatial, `6` kinematic, `4` observation-context, `4` road-graph topological) across `1,667,336` training candidates (`80` trips, `208,417` GPS observations) using identical 5-fold `GroupKFold(n_splits=5)` splits grouped strictly by `trip_id`, and evaluated out-of-sample on the `10` `student_test` trips (`153,280` candidates, `19,160` GPS observations):

| Metric | Model 1: Logistic Regression | Model 2: Distance-Weighted KNN | Model 3: Random Forest |
| :--- | :--- | :--- | :--- |
| **Algorithm Family** | Linear / Probabilistic | Instance-based / Non-parametric | Bagging / Tree Ensemble |
| **Key Hyperparameters** | `C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs'` | `n_neighbors=15, weights='distance', metric='euclidean', max_samples=16k` | `n_estimators=100, max_depth=14, min_samples_leaf=25, max_features='sqrt', class_weight='balanced_subsample'` |
| **CV Macro F1 (Mean ± Std)** | $0.7217 \pm 0.0591$ | $0.8164 \pm 0.0552$ | **$0.8492 \pm 0.0685$** |
| **CV Top-1 Accuracy (Mean ± Std)** | $0.6639 \pm 0.1139$ | $0.6965 \pm 0.1161$ | **$0.7722 \pm 0.1122$** |
| **Train Macro F1 (Mean)** | $0.7371$ | $1.0000$ | $0.9101$ |
| **Train Top-1 Accuracy (Mean)** | $0.6821$ | $1.0000$ | $0.8760$ |
| **Generalization Gap (Macro F1)** | **$0.0154$** (Linear underfitting) | $0.1836$ (Instance memorization) | **$0.0609$** (Controlled by depth=14 & leaf=25) |
| **Student-Test Top-1 Accuracy** | $0.7433$ (74.33%) | $0.7736$ (77.36%) | **$0.8474$ (84.74%)** |
| **Student-Test Top-3 Accuracy** | $0.9480$ (94.80%) | $0.9516$ (95.16%) | **$0.9797$ (97.97%)** |
| **Student-Test Macro F1** | $0.7468$ | $0.8530$ | **$0.8937$** |
| **Student-Test Binary F1** | $0.5927$ | $0.7407$ | **$0.8177$** |
| **Student-Test Precision / Recall** | $0.4355$ / $0.9273$ | **$0.7865$** / $0.6999$ | $0.7290$ / **$0.9311$** |
| **Student-Test Log Loss** | $0.3489$ | $0.3117$ | **$0.1441$** |
| **Student-Test Brier Score** | $0.1117$ | $0.0446$ | **$0.0403$** |
| **Wall-Clock Train Time** | $2.48$ s | **$0.34$ s** | $44.46$ s |
| **Student-Test Inference Time** | **$0.021$ s** | $7.80$ s | $0.97$ s |

### Impact of v1 $\rightarrow$ v2 Feature Engineering
Upgrading from the 7 v1 pure-distance features to the 21 v2 domain features (adding heading alignment `heading_diff_cos`, `heading_diff_deg`, observation-relative `cand_dist_zscore`, temporal continuity `neighbor_arc_continuity`, and road-graph node degrees) boosted Student-Test Top-1 Accuracy by **+27.52 percentage points** for Logistic Regression ($46.81\% \rightarrow 74.33\%$), **+23.03 percentage points** for Distance-Weighted KNN ($54.33\% \rightarrow 77.36\%$), and enabled Random Forest to reach **$84.74\%$ Top-1 Accuracy** and **$97.97\%$ Top-3 Accuracy** prior to Viterbi sequence decoding.

---

## 3. Phase 4 Probability Calibration Audit (`calibration_audit.json` & `calibration_curves.png`)

Because Person 4's Hidden Markov Model uses candidate classifier outputs as emission likelihoods $P(y_{t,c}=1 \mid \mathbf{x}_{t,c})$, Phase 4 audits probability calibration across all `153,280` `student_test` candidates:

- **Random Forest (`rf`):** Achieves the lowest Log Loss (**$0.1441$**) and lowest Brier Score (**$0.0403$**), with a $10.17\times$ probability discrimination ratio between true road matches (mean $p = 0.8930$) and non-matching segments (mean $p = 0.0878$). Because `class_weight='balanced_subsample'` upweights positive candidates in each bootstrap tree, mid-range probabilities exhibit a mild upward shift ($\text{ECE} = 0.0634$, mean predicted probability $0.1884$ vs $0.1250$ empirical base rate).
- **Distance-Weighted KNN (`knn`):** Because it uses natural training class frequencies without artificial class-weight reweighting, its empirical mean predicted probability ($0.1275$) closely matches the true $1/8 = 0.1250$ candidate base rate ($\text{ECE} = 0.0114$, $\text{Brier} = 0.0446$), though boundary sparsity when all 15 neighbors are negative yields a higher Log Loss ($0.3117$).
- **Logistic Regression (`logreg`):** `class_weight='balanced'` shifts the decision intercept to prioritize recall ($92.73\%$), resulting in $\text{ECE} = 0.1772$, $\text{Brier} = 0.1117$, and $\text{Log Loss} = 0.3489$.

---

## 4. Phase 4 Per-Trip & Regime Error Analysis (`per_trip_results.csv` & `per_trip_error_analysis.json`)

### Per-Trip Performance Across All 10 `student_test` Trips

| Trip ID | Obs Count | Nearest-Arc Baseline Top-1 | LogReg Top-1 | KNN Top-1 | Random Forest Top-1 | RF Macro F1 | RF Log Loss | Dense Intersection Rate ($\ge 3$ cands $\le 15\text{m}$) | Mean GT Heading Diff ($^\circ$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 52.02% | 88.01% | 85.54% | **93.41%** | 0.9535 | 0.0744 | 16.30% | $11.87^\circ$ |
| `00000007` | 1,354 | 48.01% | 87.08% | 87.30% | **92.17%** | 0.9497 | 0.0747 | 14.40% | $20.00^\circ$ |
| `00000009` | 1,049 | 45.19% | 64.06% | 72.16% | **83.70%** | 0.8962 | 0.1643 | **70.64%** | $21.71^\circ$ |
| `00000040` | 2,995 | 57.53% | 85.41% | 84.91% | **92.75%** | 0.9409 | 0.0836 | 23.51% | $19.74^\circ$ |
| `00000052` | 262 | 79.01% | 85.11% | 87.79% | **93.13%** | 0.9580 | 0.0810 | 35.88% | $8.71^\circ$ |
| `00000056` | 3,958 | 71.91% | 68.87% | 77.06% | **86.91%** | 0.8984 | 0.1470 | 45.05% | $22.01^\circ$ |
| `00000065` | 768 | 70.96% | 71.22% | 73.31% | **79.69%** | 0.8679 | 0.1812 | 44.40% | $36.71^\circ$ |
| `00000076` | 1,486 | 44.75% | 72.48% | 76.99% | **83.78%** | 0.8751 | 0.1713 | 54.10% | $20.08^\circ$ |
| `00000077` | 3,911 | 48.56% | 62.26% | 65.76% | **69.34%** | 0.8155 | 0.2351 | 40.30% | **$57.95^\circ$** |
| `00000091` | 874 | 61.33% | 71.17% | 73.23% | **84.78%** | 0.8894 | 0.1544 | 57.09% | $26.38^\circ$ |

### Key Error Regimes Identified
1. **High Heading Misalignment / Complex Turning (`trip 00000077`, `gt_heading_diff > 25°`):** On aligned straight segments (`72.59%` of test observations), Random Forest achieves **$94.54\%$ Top-1 Accuracy** ($\text{Macro F1} = 0.9535$). During sharp intersections, stop-and-go jitter, or reverse-heading noise (`27.41%` of observations, most prominent in `00000077` where mean heading discrepancy is $57.95^\circ$), single-point Top-1 accuracy drops to $58.78\%$ for RF ($49.39\%$ for LogReg), motivating global HMM Viterbi transition decoding.
2. **Dense Urban Intersections & Parallel Roads (`trip 00000009`, `00000076`, `00000091` where `cands_within_15m >= 3`):** On open roads ($\le 2$ candidates within $15\text{m}$), RF reaches **$91.05\%$ Top-1 Accuracy**. In dense intersections ($\ge 3$ candidates within $15\text{m}$, accounting for $70.64\%$ of `00000009`), nearest-segment baseline collapses to $49.59\%$, whereas RF maintains **$74.12\%$ Top-1 Accuracy** (+24.53 percentage points over nearest segment).
3. **GPS Sampling Gaps & High Lateral Noise (`trip 00000076` with $8.68\%$ gap rate, `gt_dist_to_arc > 10m`):** When GPS drift exceeds $10\text{m}$ from the true road (`20.02%` of observations), nearest-segment accuracy is $45.97\%$, while RF achieves **$72.93\%$ Top-1 Accuracy** by leveraging road-graph continuity and heading consistency.
