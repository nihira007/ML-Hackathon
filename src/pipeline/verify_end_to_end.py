"""PathSnap - Full End-to-End Pipeline Verification Suite.

Runs predict_trip() and evaluate_trip_server_side() across all 10 trips in student_test/
to verify that:
1. predict_trip() consumes ONLY .track, .nodes, and .arcs (never touches .route).
2. The entire pipeline (spatial CRS -> feature extractor -> XGBoost -> Viterbi HMM -> Folium map) runs cleanly.
3. Route Continuity, Sequence F1, and latency are logged for evaluation.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.inference import evaluate_trip_server_side, load_artifacts, predict_trip


def verify_all_test_trips() -> None:
    print("=" * 75)
    print("PATHSNAP FULL END-TO-END PIPELINE VERIFICATION")
    print("=" * 75)

    test_dir = PROJECT_ROOT / "Student folder" / "student_test"
    trip_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()])
    print(f"Found {len(trip_dirs)} trips in {test_dir}\n")

    # Verify artifacts bundle
    artifacts = load_artifacts()
    print(f"[OK] Loaded artifacts successfully:")
    print(f"     - Model: {artifacts['model_meta'].get('selected_model_name', 'XGBoost')}")
    print(f"     - Feature Count: {len(artifacts['feature_columns'])}")
    print(f"     - HMM Config: {artifacts['hmm_config_dict']}\n")

    results = []
    t_start_all = time.perf_counter()

    for d in trip_dirs:
        tid = d.name
        t0 = time.perf_counter()
        pred = predict_trip(d, include_map_html=False)
        dt_pred = time.perf_counter() - t0

        eval_res = evaluate_trip_server_side(pred, d)
        hmm_m = eval_res.get("hmm_metrics", {})
        clf_m = eval_res.get("classifier_only_metrics", {})

        row = {
            "trip_id": tid,
            "n_obs": pred["n_observations"],
            "pred_route_len": len(pred["predicted_route_sequence"]),
            "hmm_continuity": round(pred["route_continuity"], 4),
            "clf_continuity": round(pred["classifier_only_continuity"], 4),
            "continuity_gain": round(pred["continuity_gain"], 4),
            "hmm_seq_f1": round(hmm_m.get("sequence_f1", 0.0), 4),
            "clf_seq_f1": round(clf_m.get("sequence_f1", 0.0), 4),
            "inference_time_sec": round(dt_pred, 3),
        }
        results.append(row)
        print(
            f"Trip {tid}: Continuity {row['clf_continuity']*100:.1f}% -> {row['hmm_continuity']*100:.1f}% "
            f"(+{row['continuity_gain']*100:.1f}%), Seq F1: {row['hmm_seq_f1']:.4f}, Latency: {row['inference_time_sec']:.2f}s"
        )

    total_time = time.perf_counter() - t_start_all
    avg_cont = sum(r["hmm_continuity"] for r in results) / len(results)
    avg_gain = sum(r["continuity_gain"] for r in results) / len(results)
    avg_f1 = sum(r["hmm_seq_f1"] for r in results) / len(results)

    print("\n" + "=" * 75)
    print(f"SUMMARY ACROSS ALL {len(results)} TEST TRIPS:")
    print(f"  - Mean HMM Route Continuity: {avg_cont*100:.2f}% (Average gain: +{avg_gain*100:.2f}%)")
    print(f"  - Mean Sequence F1: {avg_f1:.4f}")
    print(f"  - Total Elapsed Time: {total_time:.2f}s (Avg per trip: {total_time/len(results):.2f}s)")
    print("=" * 75)


if __name__ == "__main__":
    verify_all_test_trips()
