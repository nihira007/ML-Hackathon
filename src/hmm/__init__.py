"""HMM & Viterbi Route Reconstruction module for PathSnap."""

from src.hmm.evaluate_hmm import evaluate_trip_predictions
from src.hmm.route_metrics import (
    collapse_route_sequence,
    compute_lcs_length,
    compute_observation_accuracy,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.viterbi import HMMConfig, ViterbiDecoder

__all__ = [
    "HMMConfig",
    "ViterbiDecoder",
    "collapse_route_sequence",
    "compute_lcs_length",
    "compute_observation_accuracy",
    "compute_route_continuity",
    "compute_sequence_metrics",
    "evaluate_trip_predictions",
]
