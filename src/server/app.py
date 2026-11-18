"""FastAPI Live Server for PathSnap GPS Map-Matching and Model Evaluation HUD."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.inference import (
    evaluate_trip_server_side,
    load_artifacts,
    predict_trip as run_pipeline_predict_trip,
)
from src.server.backend import MODEL_METADATA, store

app = FastAPI(
    title="PathSnap Live Telemetry & Model HUD (Laptop 1 Server)",
    description="Interactive evaluation, HMM/Viterbi route reconstruction, and 2-laptop REST API server.",
    version="1.0.0",
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


@app.middleware("http")
async def log_incoming_requests(request: Request, call_next):
    """Log every incoming request from client laptops with client IP, trip, and status."""
    req_id = uuid4().hex[:8]
    client_ip = request.client.host if request.client else "127.0.0.1"
    t0 = perf_counter()

    response = await call_next(request)
    duration_ms = (perf_counter() - t0) * 1000

    status_code = response.status_code
    status_label = f"STATUS {status_code} OK" if status_code == 200 else f"STATUS {status_code} ERROR"
    color = "\033[92m" if status_code == 200 else "\033[91m"
    reset = "\033[0m"

    path_query = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    print(
        f"{color}[{status_label}]{reset} Client: \033[96m{client_ip}\033[0m | "
        f"Req: \033[93m#{req_id}\033[0m | {request.method} {path_query} | \033[95m{duration_ms:.1f}ms{reset}"
    )

    response.headers["X-Request-ID"] = req_id
    return response


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    """Serve the interactive map and telemetry HUD."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html template not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint compatible with both HUD and Laptop 2 Evaluator UI."""
    bundle = load_artifacts(ARTIFACTS_DIR)
    eval_trips = store.list_trips(partition="evaluator_test")
    student_trips = store.list_trips(partition="student_test")
    return {
        "status": "ok",
        "service": "PathSnap Live Telemetry & Map-Matching Server",
        "port": "8000",
        "selected_model": bundle["model_meta"].get("selected_model_name", "XGBoost (Optuna Tuned)"),
        "feature_count": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
        "hmm_config": bundle["hmm_config_dict"],
        "available_partitions": {
            "evaluator_test": len(eval_trips),
            "student_test": len(student_trips),
        },
    }


@app.get("/api/trips")
async def list_trips(partition: Optional[str] = Query(default=None)) -> Any:
    """List benchmark test trips (supports evaluator_test and student_test)."""
    trips = store.list_trips(partition=partition)
    return trips


@app.get("/api/models")
async def list_models() -> Dict[str, Any]:
    """List available models and performance summaries."""
    return MODEL_METADATA


@app.get("/api/trip/{trip_id}")
async def get_trip(trip_id: str) -> Dict[str, Any]:
    """Get trip coordinates, ground-truth segments, and bounding box."""
    try:
        geom = store.get_trip_geometry(trip_id)
        return {
            "trip_id": geom["trip_id"],
            "partition": geom["partition"],
            "num_obs": geom["num_obs"],
            "gps_points": geom["gps_points"],
            "gt_segments": geom["gt_segments"],
            "bbox": geom["bbox"],
            "center": geom["center"],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/predict/{trip_id}")
async def predict_trip(
    trip_id: str,
    model: str = Query(default="tuned", pattern="^(tuned|lgbm|xgb|mlp|rf|knn|logreg|heuristic)$"),
) -> Dict[str, Any]:
    """Get model predictions, matched road segments, and live evaluation metrics."""
    try:
        return store.predict_trip(trip_id, model_id=model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict")
async def predict_trip_post(req: PredictRequest) -> Dict[str, Any]:
    """POST /api/predict endpoint used by Laptop 2 Evaluator UI (`src/ui/index.html`)."""
    try:
        geom = store.get_trip_geometry(req.trip_id)
        trip_dir = geom["trip_dir"]
        pred = run_pipeline_predict_trip(trip_dir, artifacts_dir=ARTIFACTS_DIR, include_map_html=True)
        eval_res = evaluate_trip_server_side(pred, trip_dir, include_map_html=True)
        map_html = eval_res.get("folium_html") or pred.get("folium_html", "")
        return {
            "trip_id": pred["trip_id"],
            "partition": geom["partition"],
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/benchmark")
async def get_benchmark() -> List[Dict[str, Any]]:
    """Return consolidated master comparison table."""
    return store.get_master_comparison()


@app.get("/api/evaluator_test")
async def get_evaluator_test_results() -> Dict[str, Any]:
    """Return full evaluation results of the final model on evaluator_test."""
    summary_file = OUTPUTS_DIR / "evaluator_test" / "evaluator_test_summary.json"
    csv_file = OUTPUTS_DIR / "evaluator_test" / "evaluator_test_per_trip.csv"
    res: Dict[str, Any] = {}
    if summary_file.exists():
        res["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))
    if csv_file.exists():
        import pandas as pd
        res["per_trip"] = pd.read_csv(csv_file).to_dict(orient="records")
    return res


@app.get("/api/feature_importances")
async def get_feature_importances() -> List[Dict[str, Any]]:
    """Return Gini feature importances for Random Forest."""
    return store.get_feature_importances()


@app.get("/api/folium_map/{trip_id}", response_class=HTMLResponse)
async def get_folium_map(trip_id: str) -> HTMLResponse:
    """Serve the interactive Folium map generated for Deliverables Section 6."""
    tid = str(trip_id).zfill(8)
    map_file = OUTPUTS_DIR / "maps" / f"trip_{tid}_reconstruction.html"
    if not map_file.exists():
        raise HTTPException(status_code=404, detail=f"Folium map for trip {tid} not found")
    return HTMLResponse(content=map_file.read_text(encoding="utf-8"))


@app.get("/api/tuning")
async def get_tuning_summary() -> Dict[str, Any]:
    """Return Bayesian Optuna hyperparameter optimization summary (Section 4)."""
    summary_file = OUTPUTS_DIR / "tuning" / "tuning_summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Tuning summary not found")
    return json.loads(summary_file.read_text(encoding="utf-8"))


@app.get("/api/hmm")
async def get_hmm_summary() -> Dict[str, Any]:
    """Return Classifier vs HMM route reconstruction metrics (Section 5)."""
    summary_file = OUTPUTS_DIR / "hmm" / "classifier_vs_hmm_summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="HMM summary not found")
    return json.loads(summary_file.read_text(encoding="utf-8"))


@app.get("/api/deliverables")
async def get_deliverables_summary() -> Dict[str, Any]:
    """Return verified metadata mapped directly to Deliverables.pdf Sections 1-7."""
    return {
        "title": "PathSnap ML Hackathon Technical Deliverables",
        "sections": {
            "sec2_eda": {
                "title": "2.2 Dataset Understanding & EDA",
                "missing_values": "0 across all 80 train + 10 student_test + 10 evaluator_test trips",
                "train_trips": 80,
                "student_test_trips": 10,
                "evaluator_test_trips": 10,
                "features_extracted": 33,
                "feature_version": "v2",
                "leakage_guard": "5-Fold GroupKFold by trip_id",
            },
            "sec3_models": {
                "title": "3. Model Experimentation & Comparison",
                "minimum_required": 5,
                "models_trained": 7,
                "evaluator_test_top1": "94.57% (HMM) / 93.66% (Classifier-Only)",
                "evaluator_test_continuity": "99.88%",
                "evaluator_test_macro_f1": 0.9529,
                "evaluator_test_log_loss": 0.0577,
            },
            "sec4_bayesian": {
                "title": "4. Bayesian Hyperparameter Optimisation",
                "sampler": "TPESampler (Optuna)",
                "trials": 35,
                "objective": "Minimize 5-fold CV log loss",
                "best_cv_log_loss": 0.0694,
            },
            "sec5_hmm": {
                "title": "5. HMM and Viterbi Route Reconstruction",
                "student_test_continuity": "98.45% -> 99.89% (+1.44%)",
                "student_test_top1": "91.18% -> 92.47% (+1.29%)",
                "evaluator_test_continuity": "98.81% -> 99.88% (+1.07%)",
                "evaluator_test_top1": "93.66% -> 94.57% (+0.90%)",
            },
            "sec6_folium": {
                "title": "6. Geographical Visualisation Using Folium",
                "mandatory": True,
                "total_maps": 20,
            },
            "sec7_deployment": {
                "title": "7. API, UI and Deployment",
                "framework": "FastAPI (REST) + Leaflet / Folium Evaluator UI",
                "port": 8000,
                "two_laptop_ready": True,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
