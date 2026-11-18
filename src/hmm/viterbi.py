"""PathSnap - Log-Space Hidden Markov Model & Viterbi Decoder.

Implements the Newson-Krumm (2009) map-matching formulation in log-space:
1. Hidden States: Candidate road arcs at each GPS observation t.
2. Emission Probabilities: Derived from the ML classifier's predicted candidate probabilities.
3. Transition Probabilities: Exponential decay over the absolute discrepancy between
   road network route distance and great-circle GPS displacement.
4. Dynamic Programming: Log-space forward lattice + backtrace path reconstruction.
5. Edge Case Hardening: Handles missing candidates, disconnected graph components,
   and temporal GPS dropouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.graph.road_graph import RoadGraph, haversine_distance


@dataclass
class HMMConfig:
    """Configuration hyperparameters for HMM/Viterbi decoder."""

    beta: float = 15.0  # Transition distance scale parameter (meters)
    emission_weight: float = 1.5  # Weight scalar on classifier log-probabilities
    disconnect_penalty: float = 50.0  # Log-space penalty for topologically unreachable arcs
    max_route_dist_m: float = 1500.0  # Shortest-path Dijkstra radius cutoff (meters)
    gap_threshold_s: float = 5.0  # Time gap threshold (seconds) for dynamic beta scaling
    eps: float = 1e-7  # Small epsilon to prevent log(0)

    @classmethod
    def from_dict(cls, d: dict) -> HMMConfig:
        return cls(
            beta=float(d.get("beta", 15.0)),
            emission_weight=float(d.get("emission_weight", 1.5)),
            disconnect_penalty=float(d.get("disconnect_penalty", 50.0)),
            max_route_dist_m=float(d.get("max_route_dist_m", 1500.0)),
            gap_threshold_s=float(d.get("gap_threshold_s", 5.0)),
            eps=float(d.get("eps", 1e-7)),
        )

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "emission_weight": self.emission_weight,
            "disconnect_penalty": self.disconnect_penalty,
            "max_route_dist_m": self.max_route_dist_m,
            "gap_threshold_s": self.gap_threshold_s,
            "eps": self.eps,
        }


class ViterbiDecoder:
    """Log-space Hidden Markov Model and Viterbi decoder for route reconstruction."""

    def __init__(self, config: Optional[HMMConfig] = None) -> None:
        self.config = config or HMMConfig()

    def decode(
        self,
        observations: Sequence[Dict[str, Any]],
        road_graph: RoadGraph,
    ) -> Dict[str, Any]:
        """Run Viterbi decoding over a sequence of GPS observations with candidate arcs.

        Args:
            observations: List of observation dicts, each containing:
                - 'obs_idx': int
                - 'timestamp': float
                - 'lon': float
                - 'lat': float
                - 'candidates': List[Tuple[int, float]]  # (arc_id, pred_prob)
            road_graph: RoadGraph instance for network topology and shortest paths.

        Returns:
            Dict containing:
                - 'predicted_arc_ids': List[int] (one per observation)
                - 'collapsed_route': List[int] (deduplicated sequence of traversed arcs)
                - 'log_likelihood': float (optimal sequence log-probability)
                - 'n_observations': int
                - 'n_disconnects_handled': int
        """
        n_obs = len(observations)
        if n_obs == 0:
            return {
                "predicted_arc_ids": [],
                "collapsed_route": [],
                "log_likelihood": 0.0,
                "n_observations": 0,
                "n_disconnects_handled": 0,
            }

        # Filter out observations that have empty candidates safely (edge-case hardening)
        clean_obs = []
        last_valid_candidates = []
        for obs in observations:
            cands = obs.get("candidates", [])
            if not cands and last_valid_candidates:
                # Carry forward last known candidates with decayed probability
                cands = [(arc, prob * 0.5) for arc, prob in last_valid_candidates]
            clean_obs.append({**obs, "candidates": cands})
            if cands:
                last_valid_candidates = cands

        # DP lattices: V[t] is array of log probabilities for each candidate at time t
        # backpointers[t] records the index of best candidate at time t-1
        V: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []
        n_disconnects = 0

        # Step 0: Initial observation emission probabilities
        cands_0 = clean_obs[0]["candidates"]
        if not cands_0:
            # Fallback if first observation has no candidates
            V.append(np.array([0.0]))
            backpointers.append(np.array([-1]))
        else:
            probs_0 = np.array([max(p, self.config.eps) for _, p in cands_0], dtype=np.float64)
            # Prior: uniform prior + emission log-prob
            init_logp = self.config.emission_weight * np.log(probs_0)
            V.append(init_logp)
            backpointers.append(np.zeros(len(cands_0), dtype=np.int32) - 1)

        # Forward DP recursion: t = 1 to n_obs - 1
        for t in range(1, n_obs):
            prev_obs = clean_obs[t - 1]
            curr_obs = clean_obs[t]

            prev_cands = prev_obs.get("candidates", [])
            curr_cands = curr_obs.get("candidates", [])

            if not curr_cands:
                # If current has no candidates, carry forward state from t-1
                V.append(V[-1].copy())
                backpointers.append(np.arange(len(V[-1]), dtype=np.int32))
                continue

            if not prev_cands:
                # Prior observation had no candidates, re-initialize
                curr_probs = np.array([max(p, self.config.eps) for _, p in curr_cands], dtype=np.float64)
                V.append(self.config.emission_weight * np.log(curr_probs))
                backpointers.append(np.zeros(len(curr_cands), dtype=np.int32) - 1)
                continue

            # Great-circle GPS distance
            d_gps = float(
                haversine_distance(
                    prev_obs["lon"], prev_obs["lat"], curr_obs["lon"], curr_obs["lat"]
                )
            )

            # Adaptive beta scaling for GPS dropouts / gaps (Phase 3 hardening)
            dt = max(1.0, float(curr_obs.get("timestamp", 0) - prev_obs.get("timestamp", 0)))
            if dt > self.config.gap_threshold_s:
                # Relax transition penalty proportionally to time gap
                effective_beta = self.config.beta * math.sqrt(dt / 1.0)
            else:
                effective_beta = self.config.beta

            # Emission log-probabilities for all current candidates
            curr_probs = np.array([max(p, self.config.eps) for _, p in curr_cands], dtype=np.float64)
            log_emissions = self.config.emission_weight * np.log(curr_probs)

            n_prev = len(prev_cands)
            n_curr = len(curr_cands)

            curr_v = np.zeros(n_curr, dtype=np.float64)
            curr_bp = np.zeros(n_curr, dtype=np.int32)
            prev_v = V[-1]

            # Compute transition log-probabilities matrix: (n_curr, n_prev)
            for k in range(n_curr):
                curr_arc = int(curr_cands[k][0])
                best_score = -float("inf")
                best_prev_idx = 0

                for j in range(n_prev):
                    prev_arc = int(prev_cands[j][0])

                    # Road network route distance
                    d_route = road_graph.get_arc_transition_distance(
                        prev_arc, curr_arc, cutoff=self.config.max_route_dist_m
                    )

                    if math.isinf(d_route):
                        trans_logp = -self.config.disconnect_penalty
                        n_disconnects += 1
                    else:
                        discrepancy = abs(d_route - d_gps)
                        trans_logp = -discrepancy / max(effective_beta, 1e-3)

                    candidate_score = prev_v[j] + trans_logp
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_prev_idx = j

                curr_v[k] = best_score + log_emissions[k]
                curr_bp[k] = best_prev_idx

            V.append(curr_v)
            backpointers.append(curr_bp)

        # Backtracking optimal path
        best_last_idx = int(np.argmax(V[-1]))
        total_log_likelihood = float(V[-1][best_last_idx])

        optimal_cand_indices = [0] * n_obs
        optimal_cand_indices[-1] = best_last_idx

        for t in range(n_obs - 1, 0, -1):
            optimal_cand_indices[t - 1] = int(backpointers[t][optimal_cand_indices[t]])

        # Recover arc IDs
        predicted_arc_ids: List[int] = []
        for t in range(n_obs):
            cands = clean_obs[t].get("candidates", [])
            idx = optimal_cand_indices[t]
            if cands and idx < len(cands):
                predicted_arc_ids.append(int(cands[idx][0]))
            elif cands:
                predicted_arc_ids.append(int(cands[0][0]))
            else:
                predicted_arc_ids.append(0)

        # Deduplicate consecutive identical arcs for traversed route sequence
        collapsed_route = []
        for arc in predicted_arc_ids:
            if not collapsed_route or arc != collapsed_route[-1]:
                collapsed_route.append(arc)

        return {
            "predicted_arc_ids": predicted_arc_ids,
            "collapsed_route": collapsed_route,
            "log_likelihood": total_log_likelihood,
            "n_observations": n_obs,
            "n_disconnects_handled": n_disconnects,
        }
