"""Laptop 1 — PathSnap FastAPI Model, HMM/Viterbi & Server-Side Evaluation Backend."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.inference import (
    evaluate_trip_server_side,
    load_artifacts,
    predict_trip,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [LAPTOP-1-API] %(message)s",
)
logger = logging.getLogger("pathsnap.api")

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUTS_DIR = REPO_ROOT / "outputs"

PARTITION_CANDIDATES: Dict[str, List[Path]] = {
    "evaluator_test": [
        REPO_ROOT / "Student folder" / "evaluator_test",
        REPO_ROOT / "evaluator_test",
        REPO_ROOT / "data" / "evaluator_test",
    ],
    "student_test": [
        REPO_ROOT / "Student folder" / "student_test",
        REPO_ROOT / "student_test",
    ],
    "train": [
        REPO_ROOT / "Student folder" / "train",
        REPO_ROOT / "train",
    ],
}

app = FastAPI(
    title="PathSnap Distributed Map-Matching API (Laptop 1)",
    version="1.0.0",
    description="Hosts tuned XGBoost classifier, HMM/Viterbi route decoder, and server-side evaluation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = None


def _resolve_partition_dir(partition: Optional[str] = None) -> Tuple[str, Path]:
    """Find active directory for requested partition (falling back cleanly)."""
    if partition and partition in PARTITION_CANDIDATES:
        for cand in PARTITION_CANDIDATES[partition]:
            if cand.exists() and any(cand.iterdir()):
                return partition, cand

    for part_name in ("evaluator_test", "student_test", "train"):
        for cand in PARTITION_CANDIDATES[part_name]:
            if cand.exists() and any(cand.iterdir()):
                return part_name, cand

    raise FileNotFoundError("No dataset partition folder found under 'Student folder/' or 'evaluator_test/'.")


def _list_trip_dirs(part_dir: Path) -> List[Path]:
    return sorted([d for d in part_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()])


@app.on_event("startup")
def startup_event() -> None:
    bundle = load_artifacts(ARTIFACTS_DIR)
    logger.info(
        "Loaded model=%s (%d features) | HMM config=%s",
        bundle["model_meta"].get("selected_model_name", "XGBoost"),
        len(bundle["feature_columns"]),
        bundle["hmm_config_dict"],
    )


@app.get("/api/health")
def health_check(request: Request) -> Dict[str, Any]:
    bundle = load_artifacts(ARTIFACTS_DIR)
    available_partitions: Dict[str, int] = {}
    for part_name, paths in PARTITION_CANDIDATES.items():
        for p in paths:
            if p.exists():
                trips = _list_trip_dirs(p)
                if trips:
                    available_partitions[part_name] = len(trips)
                    break

    logger.info(
        "GET /api/health from client=%s | partitions=%s",
        request.client.host if request.client else "unknown",
        available_partitions,
    )
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_model": bundle["model_meta"].get("selected_model_name", "XGBoost (Optuna Tuned)"),
        "feature_count": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
        "model_meta": bundle["model_meta"],
        "hmm_config": bundle["hmm_config_dict"],
        "available_partitions": available_partitions,
    }


@app.get("/api/trips")
def list_available_trips(
    request: Request,
    partition: Optional[str] = Query(default=None, description="evaluator_test, student_test, or train"),
) -> Dict[str, Any]:
    part_name, part_dir = _resolve_partition_dir(partition)
    trip_dirs = _list_trip_dirs(part_dir)

    trips_info = []
    for d in trip_dirs:
        tid = d.name
        has_route = (d / f"{tid}.route").exists()
        trips_info.append(
            {
                "trip_id": tid,
                "partition": part_name,
                "has_server_ground_truth": has_route,
            }
        )

    logger.info(
        "GET /api/trips?partition=%s from client=%s -> returning %d trips from %s",
        partition,
        request.client.host if request.client else "unknown",
        len(trips_info),
        part_dir,
    )
    return {
        "partition": part_name,
        "partition_path": str(part_dir.relative_to(REPO_ROOT)),
        "trip_count": len(trips_info),
        "trips": trips_info,
    }


@app.get("/api/benchmarks")
def get_model_benchmarks() -> Dict[str, Any]:
    """Return recorded model comparison and HMM ablation summaries from outputs/."""
    result: Dict[str, Any] = {"models": [], "hmm_summary": {}, "per_trip_ablation": []}

    hmm_summary_path = OUTPUTS_DIR / "hmm" / "classifier_vs_hmm_summary.json"
    if hmm_summary_path.exists():
        result["hmm_summary"] = json.loads(hmm_summary_path.read_text(encoding="utf-8"))

    per_trip_path = OUTPUTS_DIR / "hmm" / "classifier_vs_hmm_per_trip.csv"
    if per_trip_path.exists():
        df_pt = pd.read_csv(per_trip_path)
        result["per_trip_ablation"] = df_pt.to_dict(orient="records")

    for cand_csv in [
        OUTPUTS_DIR / "master_model_comparison.csv",
        OUTPUTS_DIR / "tuning" / "model_selection.csv",
        OUTPUTS_DIR / "models" / "advanced" / "advanced_comparison.csv",
        OUTPUTS_DIR / "models" / "classical" / "classical_comparison.csv",
    ]:
        if cand_csv.exists():
            df_m = pd.read_csv(cand_csv)
            result["models"].extend(df_m.to_dict(orient="records"))

    return result


@app.post("/api/predict")
def predict_and_evaluate_trip(req: PredictRequest, request: Request) -> Dict[str, Any]:
    client_host = request.client.host if request.client else "unknown"
    part_name, part_dir = _resolve_partition_dir(req.partition)
    trip_dir = part_dir / req.trip_id

    if not trip_dir.exists():
        found = False
        for p_name, cands in PARTITION_CANDIDATES.items():
            for c in cands:
                if (c / req.trip_id).exists():
                    part_name = p_name
                    trip_dir = c / req.trip_id
                    found = True
                    break
            if found:
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found.")

    logger.info(
        ">>> INCOMING INFERENCE REQUEST from %s | trip_id=%s | partition=%s",
        client_host,
        req.trip_id,
        part_name,
    )

    # 1. Prediction pipeline (strictly uses only .track, .nodes, .arcs)
    pred = predict_trip(trip_dir, artifacts_dir=ARTIFACTS_DIR, include_map_html=True)

    # 2. Strictly separated server-side evaluation (uses internal .route if present on Laptop 1)
    eval_res = evaluate_trip_server_side(pred, trip_dir)

    map_html = eval_res.get("folium_html") or pred.get("folium_html", "")

    seq_f1_str = (
        f"{eval_res['hmm_metrics']['sequence_f1']:.4f}"
        if eval_res.get("has_ground_truth") and "hmm_metrics" in eval_res
        else "N/A"
    )

    logger.info(
        "<<< COMPLETED INFERENCE trip_id=%s in %.3fs | Obs=%d | PredArcs=%d | Continuity=%.4f | SeqF1=%s",
        req.trip_id,
        pred["inference_time_sec"],
        pred["n_observations"],
        len(pred["predicted_route_sequence"]),
        pred["route_continuity"],
        seq_f1_str,
    )

    return {
        "trip_id": pred["trip_id"],
        "partition": part_name,
        "n_observations": pred["n_observations"],
        "n_candidate_rows": pred["n_candidate_rows"],
        "n_nodes": pred["n_nodes"],
        "n_arcs": pred["n_arcs"],
        "predicted_route_sequence": pred["predicted_route_sequence"],
        "classifier_only_route_sequence": pred["classifier_only_route_sequence"],
        "route_continuity": pred["route_continuity"],
        "classifier_only_continuity": pred["classifier_only_continuity"],
        "continuity_gain": pred["continuity_gain"],
        "log_likelihood": pred["log_likelihood"],
        "n_disconnects_handled": pred["n_disconnects_handled"],
        "inference_time_sec": pred["inference_time_sec"],
        "timing_breakdown_sec": pred["timing_breakdown_sec"],
        "evaluation": {k: v for k, v in eval_res.items() if k != "folium_html"},
        "folium_html": map_html,
    }
