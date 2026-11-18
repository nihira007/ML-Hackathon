# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — UE24CS352A: Machine Learning Hackathon**

End-to-end machine learning + Hidden Markov Model (Viterbi) system for reconstructing continuous vehicle routes from noisy ~1 Hz GPS trajectories against urban road networks.

---

## Repository Structure

```text
ML-Hackathon/
├── PathSnap_student_materials/   # Official hackathon guidelines & rubrics (PDFs)
├── Problem Statement.mp4         # Problem statement overview video
├── Student folder/
│   ├── train/                    # 80 training trips (.track, .nodes, .arcs, .route)
│   └── student_test/             # 10 development-stage test trips
├── prompts/                      # 4-member, 5-phase branch prompts + post-merge prompt
│   ├── 00_WORKFLOW_AND_SHARED_CONTRACT.md
│   ├── person1_data_features.md
│   ├── person2_models_classical.md
│   ├── person3_models_boosting_tuning.md
│   ├── person4_hmm_viterbi_viz.md
│   └── 05_POST_MERGE_INTEGRATION.md
├── src/
│   ├── data/                     # Person 1: raw loader & metric spatial utilities
│   ├── eda/                      # Person 1: EDA scripts & audit checks
│   ├── features/                 # Person 1: candidate generation, GT alignment & features
│   ├── eval/                     # Person 2: GroupKFold CV & shared metric/logging harness
│   ├── models/
│   │   ├── classical/            # Person 2: Models 1-3 (LogReg, KNN, Random Forest)
│   │   └── advanced/             # Person 3: Models 4-6 (LightGBM, XGBoost, MLP)
│   ├── tuning/                   # Person 3: Bayesian Hyperparameter Optimization (Optuna)
│   ├── graph/                    # Person 4: road-network directed graph & shortest paths
│   ├── hmm/                      # Person 4: HMM + Viterbi decoder & route continuity metrics
│   ├── viz/                      # Person 4: interactive Folium map visualisations
│   └── pipeline/                 # Post-merge: unified predict_trip & server-side evaluation
├── data/processed/               # Generated train/test feature parquets (gitignored)
├── artifacts/                    # Exported preprocessor, final tuned model & HMM config
└── outputs/                      # Real CSV logs, plots, confusion matrices, Folium HTML maps
```

---

## 4-Member Branch Workflow

Read [`prompts/00_WORKFLOW_AND_SHARED_CONTRACT.md`](prompts/00_WORKFLOW_AND_SHARED_CONTRACT.md) first, then each teammate creates their own branch from `main` and executes Phases 1–5 from their dedicated prompt file:

| Member | Branch | Prompt File | Owned Scope |
| :--- | :--- | :--- | :--- |
| **Person 1** | `feat/p1-data-features` | [`prompts/person1_data_features.md`](prompts/person1_data_features.md) | Loader, Spatial CRS, EDA, Candidate Gen, GT Alignment, Features |
| **Person 2** | `feat/p2-models-classical` | [`prompts/person2_models_classical.md`](prompts/person2_models_classical.md) | GroupKFold Eval Harness + Models 1–3 (LogReg, KNN, RF) |
| **Person 3** | `feat/p3-models-boosting-tuning` | [`prompts/person3_models_boosting_tuning.md`](prompts/person3_models_boosting_tuning.md) | Models 4–6 (LightGBM, XGBoost, MLP) + Bayesian Optuna Tuning |
| **Person 4** | `feat/p4-hmm-viterbi-viz` | [`prompts/person4_hmm_viterbi_viz.md`](prompts/person4_hmm_viterbi_viz.md) | Road Graph, HMM + Viterbi Decoder, Continuity & Folium Maps |
| **Post-Merge** | `main` | [`prompts/05_POST_MERGE_INTEGRATION.md`](prompts/05_POST_MERGE_INTEGRATION.md) | Merge all 4 branches, unified inference pipeline & full verification |

---

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Deployment (Laptop 1 FastAPI backend + Laptop 2 Evaluator UI) will be added after the post-merge ML + HMM pipeline on `main` is verified.
