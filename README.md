# PathSnap: Distributed GPS Map-Matching & Route Reconstruction

**PES University — UE24CS352A: Machine Learning Hackathon**

PathSnap is an end-to-end spatial machine learning and probabilistic sequence decoding system that reconstructs continuous vehicle routes on directed urban road networks from noisy ~1 Hz GPS trajectories (`.track`, `.nodes`, `.arcs`).

It combines **exact WGS84-to-ENU metric projection**, **cKDTree spatial candidate generation (K=10)**, **33 engineered domain features (v2)**, **6 benchmarked ML classifiers + Bayesian Optuna TPE hyperparameter optimization**, and a **Newson-Krumm Log-Space Hidden Markov Model (Viterbi) route decoder**, deployed across a **2-Laptop FastAPI Backend + Interactive Evaluator UI** over the same Wi-Fi network.

## 1. Repository Structure (Mapped to Submission Deliverables)

Every folder and script in this repository maps directly to the required hackathon submission components:

```text
ML-Hackathon/
├── README.md                                  # Installation, results, and 2-laptop Wi-Fi deployment guide
├── requirements.txt                           # Python dependencies for training, API, UI, and Folium
├── verify_trips_api.py                        # Live CLI verification script (defaults to evaluator_test)
├── Student folder/
│   ├── train/                                 # 80 training trips (.track, .nodes, .arcs, .route)
│   └── student_test/                          # 10 development test trips
├── evaluator_test/                            # 10 reserved evaluator test trips (00000015 .. 00000094)
├── artifacts/                                 # Saved production model weights & configurations
│   ├── preprocessor.joblib                    # Train-only median imputer + StandardScaler
│   ├── feature_columns.json                   # Exact ordered list of 33 v2 engineered features
│   ├── final_model.joblib                     # Optuna-tuned XGBoost production classifier
│   ├── final_model_meta.json                  # Hyperparameters, Optuna search space, and data signatures
│   └── hmm_config.json                        # Optimal HMM/Viterbi transition & emission parameters
├── src/
│   ├── data/                                  # [Data Processing] Raw dataset reading, cleaning & projection
│   │   ├── loader.py                          # Reads .track, .nodes, .arcs, .route with integrity checks
│   │   └── spatial.py                         # Per-trip WGS84 -> ECEF -> ENU tangent-plane metric projection
│   ├── eda/                                   # [Data Processing & Visualisations] EDA & leakage audits
│   │   ├── run_eda.py                         # Missing-value checks, sampling gaps, trajectory plots & Folium maps
│   │   └── make_report_summary.py             # Generates verified dataset statistics and candidate miss-rate tables
│   ├── features/                              # [Feature Engineering] Candidate search, GT alignment & 33 features
│   │   ├── candidates.py                      # cKDTree spatial indexing (Top-10 within 100m) + monotone DP GT alignment
│   │   ├── build_features.py                  # Full 33-feature extraction pipeline & train-only preprocessor fitting
│   │   ├── inference_features.py              # Zero-leakage single-trip inference feature extractor (ignores .route)
│   │   ├── table_io.py                        # Parquet / CSV feature table I/O helpers
│   │   └── test_data_pipeline.py              # Unit & anti-leakage tests for data & feature pipeline
│   ├── models/                                # [Model Training] All 6 experimented model architectures
│   │   ├── classical/                         # Models 1-3: Logistic Regression, Distance-Weighted KNN, Random Forest
│   │   │   ├── train_logreg.py                # Model 1: L2 Logistic Regression
│   │   │   ├── train_knn.py                   # Model 2: Distance-Weighted KNN (k=15)
│   │   │   ├── train_rf.py                    # Model 3: Random Forest (100 trees, max_depth=14)
│   │   │   ├── run_all_classical.py           # Unified runner for classical baselines
│   │   │   ├── calibration_and_error_analysis.py # Reliability curves & per-trip error diagnostics
│   │   │   └── generate_phase5_plots.py       # Classical model comparison & diagnostic plots
│   │   └── advanced/                          # Models 4-6: LightGBM, XGBoost, MLP Neural Network
│   │       ├── common.py                      # Shared training, 5-fold GroupKFold CV & plotting utilities
│   │       ├── train_lgbm.py                  # Model 4: LightGBM (leaf-wise GBDT)
│   │       ├── train_xgb.py                   # Model 5: XGBoost (depth-wise histogram GBDT)
│   │       └── train_mlp.py                   # Model 6: Feed-Forward MLP Neural Network (64x32, 20 tracked epochs)
│   ├── eval/                                  # [Model Evaluation] Metrics, GroupKFold CV & confusion matrices
│   │   ├── metrics.py                         # Observation Top-1 accuracy, Macro F1, Log Loss, ROC-AUC, PR-AUC
│   │   ├── logger.py                          # Standardized JSON/CSV experiment logger & plot generator
│   │   └── test_classical_pipeline.py         # Unit tests for evaluation metrics and GroupKFold isolation
│   ├── tuning/                                # [Hyperparameter Optimisation] Bayesian Optuna TPE optimization
│   │   ├── bayesian_opt.py                    # 35-trial Optuna TPESampler optimizing 5-fold GroupKFold CV Log Loss
│   │   └── make_report.py                     # Generates untuned vs. tuned comparison tables & Optuna plots
│   ├── graph/                                 # [HMM & Viterbi] Directed road network topology
│   │   └── road_graph.py                      # Directed adjacency lists, haversine edge weights & cached Dijkstra
│   ├── hmm/                                   # [HMM & Viterbi] Log-space Viterbi route reconstruction
│   │   ├── viterbi.py                         # Log-space Viterbi decoder with adaptive gap scaling & disconnect bridging
│   │   ├── route_metrics.py                   # Route continuity, sequence Precision/Recall/F1, LCS ratio & Jaccard
│   │   ├── tune_hmm.py                        # Grid/Bayesian calibration of beta, emission_weight & disconnect_penalty
│   │   ├── evaluate_hmm.py                    # Classifier-only vs. Classifier+HMM benchmark runner
│   │   ├── plot_results.py                    # Continuity & Top-1 accuracy improvement charts
│   │   └── test_viterbi.py                    # Unit tests for road graph shortest paths and Viterbi decoding
│   ├── viz/                                   # [Visualisations] Interactive Folium map generation
│   │   ├── folium_map.py                      # Multi-layer Folium HTML map builder with custom legend & LayerControl
│   │   └── generate_student_test_maps.py      # Batch generator for interactive route reconstruction maps
│   ├── pipeline/                              # [Unified Inference & Evaluator Test Benchmark]
│   │   ├── inference.py                       # Zero-leakage predict_trip() and separated evaluate_trip_server_side()
│   │   ├── evaluate_evaluator_test.py         # Evaluates final model on all 10 trips in evaluator_test/
│   │   └── verify_end_to_end.py               # Consolidates 6-model + Tuned + HMM master comparison table & plots
│   ├── api/                                   # [API] Dedicated Laptop 1 REST API server
│   │   └── server.py                          # FastAPI backend (/api/health, /api/trips, /api/predict, /api/benchmarks)
│   ├── server/                                # [API + Telemetry HUD] Unified FastAPI backend & live map HUD
│   │   ├── app.py                             # FastAPI server serving both REST endpoints and live Leaflet/Folium UI
│   │   ├── backend.py                         # Model store supporting both evaluator_test and student_test
│   │   └── templates/index.html               # Interactive dark-mode telemetry HUD & live API console
│   └── ui/                                    # [UI] Dedicated Laptop 2 Evaluator Interface
│       ├── server.py                          # Lightweight HTTP server on port 8501 for Laptop 2
│       └── index.html                         # Evaluator dashboard connecting remotely to Laptop 1 IP over Wi-Fi
└── outputs/                                   # [Supporting Results] Verified CSVs, JSON logs, PNG plots & Folium maps
    ├── master_model_comparison.csv            # Consolidated 8-row benchmark across all 6 models + Tuned + HMM
    ├── master_plots/master_6model_comparison.png
    ├── evaluator_test/                        # Final model results on the 10 reserved evaluator_test trips
    │   ├── evaluator_test_per_trip.csv
    │   ├── evaluator_test_summary.json
    │   └── evaluator_test_preds.parquet
    ├── eda/                                   # Dataset inventory, candidate miss rates, leakage audit & EDA plots/maps
    ├── models/
    │   ├── classical/                         # LogReg, KNN, RF metrics, confusion matrices, calibration curves
    │   └── advanced/                          # LightGBM, XGBoost, MLP metrics, confusion matrices, MLP epoch loss curve
    ├── tuning/                                # Optuna 35-trial logs, parameter importance, untuned vs. tuned CSV/plots
    ├── hmm/                                   # Classifier vs. HMM per-trip CSV, continuity ablation & plots
    └── maps/                                  # 20 interactive Folium HTML maps (10 evaluator_test + 10 student_test)
```

## 2. Final Experimental Results

### 2.1 Final Model Performance on `evaluator_test` (10 Reserved Evaluation Trips)

We evaluated our production pipeline (**Optuna-Tuned XGBoost** in `artifacts/final_model.joblib` + **Viterbi HMM Decoder** in `artifacts/hmm_config.json`) on all 10 trips in `evaluator_test/` (`00000015`–`00000094`, totaling **19,674 GPS observations** and **167,790 candidate rows**). During inference, `predict_trip()` stages only `.track`, `.nodes`, and `.arcs` in an isolated directory so `.route` is never accessed until server-side evaluation.

* **Reproduce command:** `python -m src.pipeline.evaluate_evaluator_test`
* **Summary artifact:** `outputs/evaluator_test/evaluator_test_summary.json`
* **Per-trip CSV:** `outputs/evaluator_test/evaluator_test_per_trip.csv`

| Trip ID | GPS Obs | Candidate Rows | Classifier-Only Top-1 Acc | Tuned XGBoost + HMM Top-1 Acc | Top-1 Gain | Classifier-Only Continuity | Tuned XGBoost + HMM Continuity | Continuity Gain | Sequence F1 | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,854 | 57,750 | 85.83% | **85.83%** | +0.00% | 99.30% | **100.00%** | +0.70% | 0.6667 | 5.778s |
| `00000017` | 1,346 | 13,126 | 93.31% | **93.61%** | +0.30% | 98.36% | **99.70%** | +1.34% | 0.8347 | 2.712s |
| `00000023` | 458 | 2,852 | 94.98% | **94.76%** | -0.22% | 98.91% | **99.34%** | +0.44% | 0.8833 | 0.575s |
| `00000030` | 654 | 6,506 | 81.19% | **85.17%** | **+3.98%** | 96.78% | **99.85%** | **+3.06%** | 0.7866 | 1.188s |
| `00000049` | 2,642 | 17,381 | 99.28% | **99.62%** | +0.34% | 99.43% | **100.00%** | +0.57% | 0.9777 | 2.319s |
| `00000059` | 1,031 | 9,940 | 96.61% | **96.70%** | +0.10% | 99.22% | **100.00%** | +0.78% | 0.8369 | 1.546s |
| `00000067` | 916 | 7,520 | 96.29% | **96.29%** | +0.00% | 99.56% | **100.00%** | +0.44% | 0.9427 | 0.832s |
| `00000073` | 1,144 | 10,763 | 98.60% | **98.86%** | +0.26% | 99.56% | **99.91%** | +0.35% | 0.9677 | 1.282s |
| `00000087` | 1,399 | 12,337 | 95.78% | **99.64%** | **+3.86%** | 97.42% | **100.00%** | **+2.58%** | 0.9810 | 1.955s |
| `00000094` | 3,230 | 29,615 | 94.74% | **95.17%** | +0.43% | 99.54% | **99.97%** | +0.43% | 0.9405 | 3.282s |
| **Mean / Total** | **19,674** | **167,790** | **93.66%** | **94.57%** | **+0.90%** | **98.81%** | **99.88%** | **+1.07%** | **0.8818** | **21.468s** |

* **Candidate-Level Binary Macro F1 (`evaluator_test`):** `0.9529`
* **Candidate-Level Binary Log Loss (`evaluator_test`):** `0.0577`
* **6 out of 10 `evaluator_test` trips** achieve **100.00% physical route continuity** after Viterbi decoding.

### 2.2 Master Model Comparison Table (`train` 5-Fold GroupKFold CV & `student_test`)

All 6 baseline models + Bayesian-tuned XGBoost + HMM/Viterbi were trained on the 80 training trips (`1,510,241` label-verified candidate rows) using **5-Fold `GroupKFold(n_splits=5)` grouped by `trip_id`** (zero trip overlap) and evaluated on `student_test` (`10` trips, `19,160` GPS observations, `157,866` candidate rows).

* **Source file:** `outputs/master_model_comparison.csv`

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy (`student_test`) | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time (10 trips) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Logistic Regression** | Linear / Probabilistic | `C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs'` | 0.8707 (±0.0112) | 88.15% | 98.68% | 0.8631 | 0.1662 | 0.0115s |
| **Model 2: Distance-Weighted KNN** | Instance-based / Non-parametric | `n_neighbors=15, weights='distance', metric='euclidean'` | 0.8917 (±0.0146) | 81.66% | 98.12% | 0.8843 | 0.2722 | 8.3207s |
| **Model 3: Random Forest** | Bagging / Tree Ensemble | `n_estimators=100, max_depth=14, min_samples_leaf=25` | 0.9265 (±0.0216) | 88.70% | 97.54% | 0.9241 | 0.1039 | 2.0540s |
| **Model 4: LightGBM** | GBDT (Leaf-wise) | `n_estimators=160, lr=0.06, num_leaves=31, min_child_samples=30` | 0.9390 (±0.0190) | 90.14% | 98.48% | 0.9418 | 0.0654 | 0.7806s |
| **Model 5: XGBoost (Untuned)** | GBDT (Depth-wise Histogram) | `n_estimators=160, lr=0.06, max_depth=5, tree_method='hist'` | 0.9391 (±0.0189) | 89.76% | 98.44% | 0.9391 | 0.0672 | 0.1948s |
| **Model 6: MLP Neural Network** | Feed-Forward Neural Network | `hidden_layer_sizes=(64, 32), relu, adam, epochs=20` | 0.9313 (±0.0171) | 89.23% | 98.26% | 0.9335 | 0.0748 | 0.0483s |
| **XGBoost (Optuna Tuned)** | Bayesian-Optimized GBDT (35 Trials) | `n_estimators=200, lr=0.0322, max_depth=8, subsample=0.774, colsample=0.908` | **0.9399** (±0.0181) | **90.19%** | 98.45% | **0.9427** | **0.0644** | 0.6540s |
| **XGBoost (Tuned) + HMM/Viterbi** | Probabilistic Sequence Decoder + GBDT | Tuned XGBoost emissions + Viterbi (`beta=25.0m, w=2.0, max_dist=1000m`) | **0.9399** (±0.0181) | **92.47%** | **99.89%** | **0.9427** | **0.0644** | 5.2211s |

### 2.3 Bayesian Hyperparameter Optimization Summary (Optuna TPE)

* **Source files:** `src/tuning/bayesian_opt.py`, `outputs/tuning/tuning_summary.json`, `outputs/tuning/untuned_vs_tuned.csv`
* **Objective:** Minimize 5-fold GroupKFold cross-validation binary Log Loss (ensuring sharp, well-calibrated emission probabilities for the downstream Viterbi decoder).
* **Sampler & Trials:** Optuna `TPESampler(seed=42)`, **35 completed trials** (`585.6s` total wall time).
* **Best Trial (#12):** Reduced CV Log Loss from `0.0746` to **`0.0694`**, reduced `student_test` Log Loss from `0.0672` to **`0.0644`**, improved `student_test` Macro F1 from `0.9391` to **`0.9427`**, and improved pointwise Top-1 Accuracy from `89.76%` to **`90.19%`**.

### 2.4 HMM & Viterbi Route Reconstruction Summary

* **Source files:** `src/hmm/viterbi.py`, `outputs/hmm/classifier_vs_hmm_per_trip.csv`, `outputs/hmm/classifier_vs_hmm_summary.json`
* **Formulation:**
  * **Hidden States:** Directed road arcs in the Top-10 spatial candidate pool at timestamp t.
  * **Log-Emission Score:** `w_emit * log(max(p_pred, eps))` using calibrated Tuned XGBoost probabilities (`w_emit=2.0`).
  * **Log-Transition Score:** `-|d_route(a_prev -> a_curr) - d_gps(z_prev, z_curr)| / beta_eff` with `beta=25.0m` (dynamically scaled by `sqrt(dt)` when GPS dropout gaps exceed `3.0s`) and `disconnect_penalty = 30.0` beyond `1000.0m`.
* **Impact across Test Partitions:**
  * On `student_test` (10 trips): Route Continuity improves from **98.45% to 99.89%** (`+1.44%`), Top-1 Accuracy improves from **91.18% to 92.47%** (`+1.29%`), and Sequence F1 improves from **0.8979 to 0.9158** (`+1.78%`).
  * On `evaluator_test` (10 trips): Route Continuity improves from **98.81% to 99.88%** (`+1.07%`), Top-1 Accuracy improves from **93.66% to 94.57%** (`+0.90%`), and Sequence F1 improves from **0.8749 to 0.8818** (`+0.69%`).

## 3. Environment Setup & Reproducing All Pipeline Stages

### 3.1 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Running the Full Pipeline or Individual Stages

All artifacts (`artifacts/*`) and results (`outputs/*`) are already generated and committed. To re-run any stage from scratch:

```bash
# 1. Run Exploratory Data Analysis & Leakage / Candidate Miss-Rate Audits
.venv/bin/python -m src.eda.run_eda
.venv/bin/python -m src.eda.make_report_summary

# 2. Extract 33 Engineered Features (v2) & Fit Train-Only Preprocessor
.venv/bin/python -m src.features.build_features

# 3. Train & Evaluate Classical Models (Models 1-3: LogReg, KNN, Random Forest)
.venv/bin/python -m src.models.classical.run_all_classical
.venv/bin/python -m src.models.classical.calibration_and_error_analysis
.venv/bin/python -m src.models.classical.generate_phase5_plots

# 4. Train & Evaluate Advanced Models (Models 4-6: LightGBM, XGBoost, MLP)
.venv/bin/python -m src.models.advanced.train_lgbm
.venv/bin/python -m src.models.advanced.train_xgb
.venv/bin/python -m src.models.advanced.train_mlp

# 5. Run Bayesian Hyperparameter Optimization (35 Optuna TPE Trials)
.venv/bin/python -m src.tuning.bayesian_opt
.venv/bin/python -m src.tuning.make_report

# 6. Consolidate Master 6-Model + HMM Benchmark & Verify End-to-End Inference
.venv/bin/python -m src.pipeline.verify_end_to_end

# 7. Evaluate Final Tuned Model + HMM on the 10 Reserved evaluator_test Trips
.venv/bin/python -m src.pipeline.evaluate_evaluator_test

# 8. Run Automated Unit & Anti-Leakage Test Suite (31 tests)
.venv/bin/python -m pytest src/ -q
```

## 4. How to Run on 2 Separate Laptops Over the Same Wi-Fi Network

During live evaluation, **Laptop 1** acts as the **Backend Model & HMM/Viterbi Inference Server**, and **Laptop 2** acts as the **Evaluator Client Interface** over the same Wi-Fi / mobile hotspot LAN.

### Step 1 — Connect Both Laptops to the Same Wi-Fi / Hotspot
Ensure Laptop 1 and Laptop 2 are connected to the same Wi-Fi network (or a phone hotspot so peer-to-peer client isolation is disabled).

### Step 2 — On Laptop 1 (Backend Model & API Server)

1. **Find Laptop 1's Local Wi-Fi IP Address:**
   * **macOS:** Run `ipconfig getifaddr en0`
   * **Linux:** Run `hostname -I`
   * **Windows (PowerShell / CMD):** Run `ipconfig` and look for the IPv4 address under Wireless LAN adapter Wi-Fi.

2. **Start the FastAPI Backend Server Bound to Port 8000:**
   ```bash
   source .venv/bin/activate
   .venv/bin/python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000
   ```
   *(Alternative dedicated API entrypoint: `.venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000` — both expose the exact same `/api/health`, `/api/trips`, and `POST /api/predict` contracts.)*

3. **Keep Laptop 1's terminal visible:** Every request arriving from Laptop 2 logs a status line in Laptop 1's console showing the client IP, request ID, endpoint, HTTP status (`STATUS 200 OK`), and latency in milliseconds (for the Section 7 deliverable screenshot).

### Step 3 — On Laptop 2 (Evaluator UI & CLI Verification)

Use the Wi-Fi IPv4 address of Laptop 1 from Step 2 (referred to as `LAPTOP_1_WIFI_IP` below).

#### Option A: Open the Dedicated Laptop 2 Evaluator UI (`src/ui/`)
If Laptop 2 has this repository cloned:
```bash
python3 -m src.ui.server --port 8501
```
1. Open the local UI server on port `8501` in Laptop 2's browser.
2. In the top bar (**`Laptop 1 API:`** input box), enter Laptop 1's address on port `8000` (format: `http` + `://` + `LAPTOP_1_WIFI_IP` + `:8000`).
3. Ensure **`evaluator_test (10 Reserved Trips)`** is selected in the partition dropdown and click **`Connect & Load Trips`**.
4. Click any of the 10 trips (`00000015` through `00000094`) on the left sidebar to trigger live feature extraction + Tuned XGBoost + Viterbi HMM decoding on Laptop 1 and view the interactive Folium map, continuity gain, sequence F1, and raw JSON response on Laptop 2.

#### Option B: Zero-Install Browser Access from Laptop 2
Even without Python or git installed on Laptop 2, open any web browser on Laptop 2 and navigate directly to Laptop 1's address on port `8000`.
This loads the full **PathSnap Live Telemetry & Map-Matching HUD** directly from Laptop 1, including all 10 `evaluator_test` trips (`00000015`–`00000094`) and 10 `student_test` trips, real-time candidate probability inspection, interactive Folium maps (`Sec 6 Folium Map`), the 6-model leaderboard (`Sec 3.1 Leaderboard`), and an embedded live REST API tester (`Sec 7 API/LAN`).

#### Option C: Terminal / Command-Line API Verification from Laptop 2
Run the automated verification script from Laptop 2's terminal to test all 10 `evaluator_test` trips over Wi-Fi:
```bash
python3 verify_trips_api.py
```
*(Pass Laptop 1's URL as the first argument and `evaluator_test` as the second argument when running from Laptop 2.)*
