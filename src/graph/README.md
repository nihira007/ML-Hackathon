# Road Network Graph Engine (`src/graph/`)

**Owner:** Person 4 (`feat/p4-hmm-viterbi-viz`)  
**Scope:** Parsing urban road networks, local metric ENU projection, directed NetworkX graph representation, fast shortest-path querying, and graph connectivity diagnostics.

---

## 1. Overview & Architectural Role

In urban map-matching, road networks can contain up to **88,809 nodes and 165,790 arcs** per city. Raw Euclidean distance in degrees is invalid due to latitude distortion and cannot capture directed road topology.

The `RoadGraph` engine provides:
1. **Per-Trip Metric Projection:** Converts `(lon, lat)` coordinates into a local East-North-Up (ENU) tangent plane in meters centered at the trip centroid.
2. **Directed Metric Network:** Constructs a NetworkX `DiGraph` where edge weights represent real metric road distances in meters.
3. **High-Speed Shortest-Path Caching:** Implements single-source Dijkstra path length lookups with radius cutoffs (e.g., $1500\,\text{m}$) and per-trip LRU caching, reducing transition distance queries from seconds to under $0.01\,\text{ms}$.
4. **Network Diagnostics:** Audits node counts, arc counts, isolated nodes, and weakly/strongly connected components for report documentation.

---

## 2. Mathematical Formulation

### Local ENU Metric Tangent Projection
Given road network node coordinates $(\lambda_i, \phi_i)$ (longitude, latitude in degrees) and projection origin $(\lambda_0, \phi_0)$:

$$x_i = R_{\text{earth}} \cdot (\lambda_i - \lambda_0) \cdot \frac{\pi}{180} \cdot \cos\left(\phi_0 \cdot \frac{\pi}{180}\right)$$

$$y_i = R_{\text{earth}} \cdot (\phi_i - \phi_0) \cdot \frac{\pi}{180}$$

where $R_{\text{earth}} = 6,371,000\,\text{m}$.

### Arc Edge Weight
For directed road arc $a_k = (u, v)$ from node $u$ to node $v$:

$$w(a_k) = \sqrt{(x_v - x_u)^2 + (y_v - y_u)^2}$$

---

## 3. Public API

- `RoadGraph.from_trip_dir(trip_dir: str, trip_id: str) -> RoadGraph`
- `RoadGraph.get_diagnostics() -> GraphDiagnostics`
- `RoadGraph.get_shortest_path_distance(source_node: int, target_node: int, cutoff: float = 1500.0) -> float`
- `RoadGraph.get_arc_transition_distance(arc_prev: int, arc_curr: int, cutoff: float = 1500.0) -> float`
- `haversine_distance(lon1, lat1, lon2, lat2) -> float`
