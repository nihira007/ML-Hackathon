# Classical & Baseline Models (Person 2 — Phase 2)

**Owner:** Person 2 (`feat/p2-models-classical`)  
**Scope:** Model 1 (Logistic Regression) & Model 2 (Distance-Weighted KNN) training, 5-fold `GroupKFold` cross-validation by `trip_id`, student-test evaluation, error analysis, and prediction parquets export.

---

## 1. Overview & Objectives

In Phase 2 of PathSnap, Person 2 establishes the classical machine learning baselines for candidate road segment classification:
- **Model 1 (Linear / Probabilistic):** L2-regularized Logistic Regression with balanced class weights.
- **Model 2 (Instance-Based / Non-Parametric):** Distance-Weighted K-Nearest Neighbors ($k=15$, $w=1/d$, KD-Tree indexed).

Both models are strictly evaluated using the shared 5-fold `GroupKFold` harness on the 80 training trips (grouped strictly by `trip_id` to guarantee zero trip data leakage) and evaluated out-of-sample on the 10 `student_test` trips.

```
src/models/classical/
├── __init__.py           # Package exports for classical models
├── data_utils.py         # Spatial candidate feature builder & dataset loader (local ENU + KDTree)
├── train_logreg.py       # Model 1 training, CV, and evaluation runner
├── train_knn.py          # Model 2 training, CV, and evaluation runner
└── README.md             # Documentation, hyperparameters, and experimental analysis
```

---

## 2. Experimental Results Summary

Grounded in real experimental runs logged to `outputs/models/classical/classical_comparison.csv`:

| Metric | Model 1: Logistic Regression | Model 2: Distance-Weighted KNN |
| :--- | :--- | :--- |
| **Algorithm Family** | Linear / Probabilistic | Instance-based / Non-parametric |
| **Key Hyperparameters** | $C=1.0$, `class_weight='balanced'`, `penalty='l2'`, `solver='lbfgs'` | $k=15$, `weights='distance'`, `algorithm='kd_tree'`, `max_samples=150k` |
| **CV Macro F1 (Mean ± Std)** | $0.6230 \pm 0.0290$ | **$0.7052 \pm 0.0320$** (+0.0822) |
| **CV Top-1 Accuracy (Mean ± Std)** | $0.4439 \pm 0.0695$ | **$0.5071 \pm 0.0628$** (+0.0632) |
| **Train Macro F1 (Mean)** | $0.6281$ | $0.9999$ |
| **Generalization Gap (Train - Val)** | **$0.0050$** (No overfitting, linear underfitting) | **$0.2948$** (Substantial memorization) |
| **Student-Test Top-1 Accuracy** | $0.4681$ (46.81%) | **$0.5433$ (54.33%)** (+7.52%) |
| **Student-Test Macro F1** | $0.6155$ | **$0.7185$** |
| **Student-Test Precision / Recall** | $0.2864$ / **$0.9755$** | **$0.5099$** / $0.5043$ |
| **Student-Test Log Loss** | $0.5118$ | **$0.3186$** |
| **Student-Test Brier Score** | $0.1772$ | **$0.0727$** |
| **Wall-Clock Train Time** | $1.34$ s | **$0.21$ s** (KD-Tree build) |
| **Student-Test Inference Time** | **$3.35$ ms** (Vectorized dot-product) | $2.04$ s (15-NN spatial tree query) |

---

## 3. Comparative Analysis & Report Section 3.2 Insights

### Model 1: Logistic Regression
- **Strengths:** Extremely fast inference ($3.35$ ms for the entire 10-trip test set), smooth well-calibrated probabilities, highly conservative recall ($97.55\%$ of true candidate arcs are flagged positive), and virtually zero overfitting gap ($0.0050$).
- **Weaknesses:** Substantial underfitting on complex spatial non-linearities. A single linear hyper-plane cannot model non-linear road geometries (e.g., parallel roads, overpasses, junction forks), capping Top-1 Accuracy at $46.81\%$.

### Model 2: Distance-Weighted KNN
- **Strengths:** Captures non-linear local manifold structures in feature space, significantly outperforming Logistic Regression in Student-Test Top-1 Accuracy ($54.33\%$ vs $46.81\%$) and Macro F1 ($0.7185$ vs $0.6155$).
- **Weaknesses:** Classical non-parametric memorization leading to a wide generalization gap ($0.2948$), higher inference latency ($2.04$ s due to distance queries), and memory consumption scaling with dataset size.

---

## 4. Exported Prediction Parquets for Person 4 (HMM / Viterbi)

Both models export standardized prediction parquets strictly matching the shared contract schema:
- `outputs/models/classical/logreg_student_test_preds.parquet`
- `outputs/models/classical/knn_student_test_preds.parquet`

Columns: `trip_id` (str), `obs_idx` (int), `arc_id` (int), `is_match` (int), `pred_prob` (float).  
These parquets are immediately consumable by Person 4 for Route Continuity and Viterbi decoding benchmarking.
