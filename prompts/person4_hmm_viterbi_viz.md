# Person 4 — Road Graph, HMM & Viterbi Decoder, Continuity & Folium Maps

- **Branch:** `feat/p4-hmm-viterbi-viz`
- **Owned Folders:** `src/graph/`, `src/hmm/`, `src/viz/`, `outputs/hmm/`, `outputs/maps/`, `artifacts/hmm_config.json`
- **Rule:** Do NOT edit other members' folders. Commit after every phase on `feat/p4-hmm-viterbi-viz`.

---

## Phase 1 Prompt — Branch Setup, Road Graph Engine & Continuity Metrics

```markdown
You are Person 4 working on PathSnap (GPS Map-Matching & Route Reconstruction).
Git: Create and checkout branch `feat/p4-hmm-viterbi-viz` from `main`. Only edit files inside `src/graph/`, `src/hmm/`, `src/viz/`, `outputs/hmm/`, `outputs/maps/`, and `artifacts/hmm_config.json`.

Task (Phase 1 — Road-Network Graph Engine, Continuity Metrics & Folium Base):
1. Build `src/graph/road_graph.py`:
   - Loads `<trip>.nodes` and `<trip>.arcs` into a directed NetworkX graph (`DiGraph`) with metric edge weights (Haversine/projected length in meters) and an `arc_id -> (from_node, to_node, length_m)` lookup table.
   - Computes road-network diagnostics (node count, arc count, isolated nodes, weakly/strongly connected components) for the EDA/HMM report.
   - Implements fast shortest-path distance lookup between candidate arc endpoints (`to_node(arc_i) -> from_node(arc_j)`) with a radius cutoff (e.g., max route distance) and caching so Viterbi runs quickly even on 165k-arc graphs!
2. Build `src/hmm/route_metrics.py`:
   - **Route Continuity:** fraction of consecutive predicted arcs `(a_{t-1}, a_t)` that are either the same arc (`a_{t-1} == a_t`) or directly connected / reachable within a short valid path in the road graph.
   - **Sequence-level Performance:** deduplicated predicted arc sequence vs. ground-truth `.route` sequence (sequence Precision, Recall, F1 / Jaccard / LCS-based accuracy) + observation-level pointwise accuracy.
3. Build `src/viz/folium_map.py` (base visualizer) that plots raw GPS observations, local road-network geometry around the trajectory, ground-truth reference route, and predicted route with distinct colors, legend, and `folium.LayerControl()`.
4. Smoke-test on 2 sample trips from `Student folder/train/` and commit on `feat/p4-hmm-viterbi-viz` with message `feat(p4): phase 1 road graph engine, route continuity metrics, and Folium base`.
```

---

## Phase 2 Prompt — Core HMM & Viterbi Route Reconstruction Engine

```markdown
Continue on branch `feat/p4-hmm-viterbi-viz` (PathSnap — Person 4).

Task (Phase 2 — Core HMM & Viterbi Decoder Implementation):
1. Build `src/hmm/viterbi.py` implementing a log-space Hidden Markov Model + Viterbi decoder:
   - **Hidden States:** Candidate road arcs `a_k` at each GPS observation `t`, strictly preserving candidate-to-`arc_id` correspondence.
   - **Emission Log-Probability:** Derived consistently from the ML classifier's predicted probabilities `log(clip(p(a_k | x_t), eps, 1.0))` (weighted by configurable `emission_weight`).
   - **Transition Log-Probability:** Based on Newson-Krumm / road-network topology:
     - Same arc (`a_{t-1} == a_t`): small/zero penalty proportional to movement.
     - Connected/reachable via shortest road path `d_route`: `log P(trans) = -|d_route - d_gps| / beta` (exponential distribution over route vs. great-circle distance discrepancy).
   - Full Viterbi forward DP table + backpointer reconstruction returning the optimal sequence of `arc_id`s (one per relevant GPS observation) plus the collapsed traversed route sequence.
2. Test `viterbi.py` end-to-end on a sample trip using either P2/P3's early prediction parquet or distance-based mock probabilities, verifying runtime and backtrace correctness.
3. Commit on `feat/p4-hmm-viterbi-viz` with message `feat(p4): phase 2 core log-space HMM and Viterbi decoder`.
```

---

## Phase 3 Prompt — Edge-Case Hardening & Classifier-Only vs. HMM Benchmark Runner

```markdown
Continue on branch `feat/p4-hmm-viterbi-viz` (PathSnap — Person 4).

Task (Phase 3 — Edge Cases, GPS Gaps & Benchmark Runner):
1. Harden `src/hmm/viterbi.py` for all real-world failure modes required by the rubric:
   - **Missing candidates:** What happens if an observation has 0 candidate arcs (carry forward / skip observation safely without crashing Viterbi).
   - **Disconnected transitions:** When no valid road path exists within cutoff between `a_{t-1}` and `a_t`, assign a configurable finite `disconnect_penalty` (or trigger a Viterbi segment break/re-initialization when all states are unreachable) so the decoder never collapses to `-inf`.
   - **Large GPS gaps (`dt > gap_threshold`):** Relax transition distance penalty / increase `beta` dynamically proportional to `dt` or split trajectory across extreme dropouts.
2. Build `src/hmm/evaluate_hmm.py` that takes ANY model's `student_test` prediction parquet (`trip_id, obs_idx, arc_id, is_match, pred_prob`), runs both:
   - **Classifier-only (argmax per observation)**
   - **Classifier + HMM/Viterbi**
   and outputs per-trip and aggregate Pointwise Top-1 Accuracy, Route Continuity, Sequence F1 (vs. `.route`), and decoding latency.
3. Also compute **Route Continuity** for all baseline models from P2 & P3 so the consolidated 6-model table has exact Route Continuity values for every model.
4. Commit on `feat/p4-hmm-viterbi-viz` with message `feat(p4): phase 3 HMM edge-case handling and classifier-vs-HMM benchmark runner`.
```

---

## Phase 4 Prompt — HMM Parameter Tuning on Tuned Classifier & Ablation Results

```markdown
Continue on branch `feat/p4-hmm-viterbi-viz` (PathSnap — Person 4).
Sync first: Pull/use Person 3's `outputs/tuning/tuned_student_test_preds.parquet` (and train OOF predictions if available).

Task (Phase 4 — HMM Parameter Tuning & Quantitative Comparison):
1. Tune HMM decoder parameters (`beta`, `emission_weight`, `disconnect_penalty`, `max_route_dist_m`, `gap_threshold_s`) strictly on training/student-test data (never touching evaluator data) and log the search table to `outputs/hmm/hmm_tuning_log.csv`.
2. Save the optimal decoder configuration to `artifacts/hmm_config.json`.
3. Run the complete **Classifier-only vs. Classifier + HMM/Viterbi** evaluation across all 10 `student_test` trips using the final tuned classifier probabilities:
   - Save per-trip comparison (`outputs/hmm/classifier_vs_hmm_per_trip.csv`) and summary metrics (`outputs/hmm/classifier_vs_hmm_summary.json`).
   - Explicitly identify both trips where HMM significantly improved continuity/accuracy AND any failure/edge cases where HMM did not help (required for Report Section 5).
4. Commit on `feat/p4-hmm-viterbi-viz` with message `feat(p4): phase 4 HMM parameter tuning, config export, and classifier-vs-HMM ablation`.
```

---

## Phase 5 Prompt — Mandatory Folium Interactive Maps & Report Sections 5–6 Summaries

```markdown
Continue on branch `feat/p4-hmm-viterbi-viz` (PathSnap — Person 4).

Task (Phase 5 — Mandatory Interactive Folium Maps & Report Documentation):
1. Upgrade `src/viz/folium_map.py` and generate interactive HTML maps in `outputs/maps/` for at least 3 representative `student_test` trips (including best improvement case and a tricky/gap case). Every map MUST include:
   - Layer 1: Original noisy GPS observations (distinct markers/color).
   - Layer 2: Relevant local road-network geometry (subtle gray lines).
   - Layer 3: Ground-truth reference `.route` (e.g., green line).
   - Layer 4: Classifier-only predicted segments (e.g., orange/red dashed) AND final Classifier+HMM predicted route (e.g., bold blue line).
   - Clear HTML legend, appropriate auto-fitted zoom bounds, tooltips, and interactive `LayerControl`.
2. Export high-resolution static PNG screenshots of the Folium maps (and matplotlib Classifier-only vs. HMM bar charts) into `outputs/maps/screenshots/` and `outputs/hmm/plots/` for the PDF report.
3. Write `outputs/hmm/summary.md` answering all compulsory questions for Report Section 5 (hidden states, emissions, transitions, missing candidates/GPS gaps, disconnected segments, Classifier-only vs. HMM results + where HMM did not help) and Section 6 (Folium map explanations).
4. Commit on `feat/p4-hmm-viterbi-viz` with message `feat(p4): phase 5 mandatory Folium maps, plots, and report sections 5-6 summaries`.
```
