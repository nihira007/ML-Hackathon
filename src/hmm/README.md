# Hidden Markov Model & Viterbi Decoder (`src/hmm/`)

**Owner:** Person 4 (`feat/p4-hmm-viterbi-viz`)  
**Scope:** Log-space HMM, Newson-Krumm transition modeling, Viterbi dynamic programming decoder, route continuity metrics, and sequence-level evaluation.

---

## 1. Overview & Problem Formulation

Pointwise machine learning classifiers predict candidate road segments independently at each observation $t$. Because GPS noise causes noisy candidate flickers, classifier-only predictions produce disconnected paths with poor Route Continuity.

The HMM + Viterbi decoder resolves spatial-temporal sequence consistency by finding the globally most likely sequence of road arcs:

$$\hat{A} = \arg\max_{A} \left[ \sum_{t=0}^{T-1} \log P(y_t \mid a_t) + \sum_{t=1}^{T-1} \log P(a_t \mid a_{t-1}) \right]$$

---

## 2. Emission & Transition Probabilities

### Emission Log-Probability
Derived from the ML classifier's calibrated posterior probability $p(a_{t,k} \mid x_t)$:

$$\log P(y_t \mid a_{t,k}) = w_{\text{emission}} \cdot \log\left(\max(p(a_{t,k} \mid x_t), \epsilon)\right)$$

where $w_{\text{emission}} = 1.5$ and $\epsilon = 10^{-7}$.

### Transition Log-Probability (Newson & Krumm 2009)
Measures physical route plausibility between candidate arcs $a_{t-1}$ and $a_t$:
- If $a_{t-1} == a_t$: $\Delta = 0$ (vehicle remains on same segment).
- If $a_{t-1} \neq a_t$: $d_{\text{route}}$ is the shortest road-network path distance from $\text{to\_node}(a_{t-1})$ to $\text{from\_node}(a_t)$, and $d_{\text{gps}}$ is the great-circle displacement.

$$\log P(a_t \mid a_{t-1}) = -\frac{|d_{\text{route}} - d_{\text{gps}}|}{\beta_{\text{effective}}}$$

When no valid road path connects the segments within the radius cutoff:
$$\log P(a_t \mid a_{t-1}) = -\text{disconnect\_penalty}$$

---

## 3. Route Continuity Metric

A transition $(a_{t-1}, a_t)$ is continuous if:
1. $a_{t-1} == a_t$, OR
2. $\text{to\_node}(a_{t-1}) == \text{from\_node}(a_t)$, OR
3. Shortest road network distance from $\text{to\_node}(a_{t-1})$ to $\text{from\_node}(a_t) \le 100\,\text{m}$.

$$\text{Route Continuity} = \frac{\sum_{t=1}^{T-1} \mathbb{I}(\text{continuous}(a_{t-1}, a_t))}{T - 1}$$
