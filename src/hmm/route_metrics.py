"""PathSnap - Route Continuity & Sequence Metrics for Map-Matching Evaluation.

Implements:
1. Route Continuity: measures physical drivability and topological consistency
   of consecutive predicted road arcs on the road network graph.
2. Sequence-Level Metrics: deduplicated traversed route sequence compared against
   ground-truth .route sequence (Precision, Recall, F1, LCS ratio, Jaccard).
3. Observation-Level Accuracy: pointwise top-1 exact match fraction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

from src.graph.road_graph import RoadGraph


def collapse_route_sequence(arc_ids: Sequence[int]) -> List[int]:
    """Collapse consecutive duplicate arcs to produce the traversed route sequence.

    Example:
        [101, 101, 101, 204, 204, 305] -> [101, 204, 305]
    """
    if not arc_ids:
        return []
    collapsed = [int(arc_ids[0])]
    for arc in arc_ids[1:]:
        arc = int(arc)
        if arc != collapsed[-1]:
            collapsed.append(arc)
    return collapsed


def compute_route_continuity(
    predicted_arcs: Sequence[int],
    road_graph: RoadGraph,
    max_step_dist_m: float = 100.0,
) -> float:
    """Compute the Route Continuity metric across a predicted arc sequence.

    A transition between consecutive observations (a_{t-1}, a_t) is defined as continuous if:
    1. a_{t-1} == a_t (vehicle remains on same segment), OR
    2. to_node(a_{t-1}) == from_node(a_t) (direct physical graph adjacency), OR
    3. shortest road-network path from to_node(a_{t-1}) to from_node(a_t) exists
       and length <= max_step_dist_m (short topological bridge).

    Returns:
        Fraction of continuous transitions in [0.0, 1.0].
        If fewer than 2 observations, returns 1.0.
    """
    if len(predicted_arcs) < 2:
        return 1.0

    n_transitions = len(predicted_arcs) - 1
    continuous_count = 0

    for t in range(1, len(predicted_arcs)):
        u = int(predicted_arcs[t - 1])
        v = int(predicted_arcs[t])

        if u == v:
            continuous_count += 1
            continue

        u_from, u_to = road_graph.get_arc_endpoints(u)
        v_from, v_to = road_graph.get_arc_endpoints(v)

        if u_to == v_from:
            continuous_count += 1
            continue

        # Check graph shortest path distance between u_to and v_from
        dist = road_graph.get_shortest_path_distance(u_to, v_from, cutoff=max_step_dist_m)
        if dist <= max_step_dist_m:
            continuous_count += 1

    return float(continuous_count / n_transitions)


def compute_lcs_length(seq1: Sequence[int], seq2: Sequence[int]) -> int:
    """Compute length of Longest Common Subsequence (LCS) between two integer sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0

    # Space-optimized DP with 2 rows
    dp = [0] * (m + 1)
    for x in seq1:
        new_dp = [0] * (m + 1)
        for j, y in enumerate(seq2, 1):
            if x == y:
                new_dp[j] = dp[j - 1] + 1
            else:
                new_dp[j] = max(dp[j], new_dp[j - 1])
        dp = new_dp

    return dp[m]


def compute_sequence_metrics(
    pred_arcs: Sequence[int],
    true_route_arcs: Sequence[int],
    deduplicate: bool = True,
) -> Dict[str, float]:
    """Compute sequence-level Precision, Recall, F1, LCS ratio, and Jaccard similarity.

    Args:
        pred_arcs: Sequence of predicted arc IDs (e.g., from Viterbi or argmax).
        true_route_arcs: Chronological sequence of ground-truth traversed arc IDs (.route).
        deduplicate: If True, collapses consecutive identical predicted arcs first.

    Returns:
        Dict with keys:
            sequence_precision, sequence_recall, sequence_f1,
            lcs_length, lcs_ratio, jaccard_similarity,
            n_pred_arcs, n_true_arcs
    """
    p_seq = collapse_route_sequence(pred_arcs) if deduplicate else list(pred_arcs)
    t_seq = list(true_route_arcs)

    len_p = len(p_seq)
    len_t = len(t_seq)

    if len_p == 0 and len_t == 0:
        return {
            "sequence_precision": 1.0,
            "sequence_recall": 1.0,
            "sequence_f1": 1.0,
            "lcs_length": 0,
            "lcs_ratio": 1.0,
            "jaccard_similarity": 1.0,
            "n_pred_arcs": 0,
            "n_true_arcs": 0,
        }

    if len_p == 0 or len_t == 0:
        return {
            "sequence_precision": 0.0,
            "sequence_recall": 0.0,
            "sequence_f1": 0.0,
            "lcs_length": 0,
            "lcs_ratio": 0.0,
            "jaccard_similarity": 0.0,
            "n_pred_arcs": len_p,
            "n_true_arcs": len_t,
        }

    set_p = set(p_seq)
    set_t = set(t_seq)

    # Precision & Recall based on distinct traversed segments
    intersection_count = len(set_p & set_t)
    precision = float(intersection_count / len(set_p)) if set_p else 0.0
    recall = float(intersection_count / len(set_t)) if set_t else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Jaccard index
    union_count = len(set_p | set_t)
    jaccard = float(intersection_count / union_count) if union_count > 0 else 0.0

    # Longest Common Subsequence (order-preserving metric)
    lcs_len = compute_lcs_length(p_seq, t_seq)
    lcs_ratio = float(2.0 * lcs_len / (len_p + len_t)) if (len_p + len_t) > 0 else 0.0

    return {
        "sequence_precision": round(precision, 4),
        "sequence_recall": round(recall, 4),
        "sequence_f1": round(f1, 4),
        "lcs_length": lcs_len,
        "lcs_ratio": round(lcs_ratio, 4),
        "jaccard_similarity": round(jaccard, 4),
        "n_pred_arcs": len_p,
        "n_true_arcs": len_t,
    }


def compute_observation_accuracy(
    predicted_arcs: Sequence[int],
    ground_truth_aligned_arcs: Sequence[int],
) -> float:
    """Compute observation-level pointwise Top-1 accuracy against aligned true arcs.

    Args:
        predicted_arcs: Array of predicted arc IDs, one per GPS observation.
        ground_truth_aligned_arcs: Array of true aligned arc IDs for each GPS observation.

    Returns:
        Top-1 accuracy in [0.0, 1.0].
    """
    if len(predicted_arcs) == 0:
        return 0.0
    preds = np.asarray(predicted_arcs)
    trues = np.asarray(ground_truth_aligned_arcs)
    return float(np.mean(preds == trues))
