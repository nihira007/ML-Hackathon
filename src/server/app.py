"""FastAPI Live Server for PathSnap GPS Map-Matching and Model Evaluation HUD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.server.backend import MODEL_METADATA, store

app = FastAPI(
    title="PathSnap Live Telemetry & Model HUD",
    description="Interactive evaluation and route reconstruction server for classical ML models.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

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
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "PathSnap Live Telemetry Server", "port": "8000"}


@app.get("/api/trips")
async def list_trips() -> List[Dict[str, Any]]:
    """List all available benchmark test trips."""
    return store.list_trips()


@app.get("/api/models")
async def list_models() -> Dict[str, Any]:
    """List available classical models and performance summaries."""
    return MODEL_METADATA


@app.get("/api/trip/{trip_id}")
async def get_trip(trip_id: str) -> Dict[str, Any]:
    """Get trip coordinates, ground-truth segments, and bounding box."""
    try:
        geom = store.get_trip_geometry(trip_id)
        # Omit raw numpy arrays for clean JSON response
        return {
            "trip_id": geom["trip_id"],
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


@app.get("/api/benchmark")
async def get_benchmark() -> List[Dict[str, Any]]:
    """Return consolidated master comparison table."""
    return store.get_master_comparison()


@app.get("/api/feature_importances")
async def get_feature_importances() -> List[Dict[str, Any]]:
    """Return Gini feature importances for Random Forest."""
    return store.get_feature_importances()


@app.get("/api/folium_map/{trip_id}", response_class=HTMLResponse)
async def get_folium_map(trip_id: str) -> HTMLResponse:
    """Serve the interactive Folium map generated for Deliverables Section 6."""
    map_file = OUTPUTS_DIR / "maps" / f"trip_{trip_id}_reconstruction.html"
    if not map_file.exists():
        raise HTTPException(status_code=404, detail=f"Folium map for trip {trip_id} not found")
    with open(map_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/tuning")
async def get_tuning_summary() -> Dict[str, Any]:
    """Return Bayesian Optuna hyperparameter optimization summary (Section 4)."""
    summary_file = OUTPUTS_DIR / "tuning" / "tuning_summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Tuning summary not found")
    import json
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/hmm")
async def get_hmm_summary() -> Dict[str, Any]:
    """Return Classifier vs HMM route reconstruction metrics (Section 5)."""
    summary_file = OUTPUTS_DIR / "hmm" / "classifier_vs_hmm_summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="HMM summary not found")
    import json
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/deliverables")
async def get_deliverables_summary() -> Dict[str, Any]:
    """Return verified metadata mapped directly to Deliverables.pdf Sections 1-7."""
    return {
        "title": "PathSnap ML Hackathon Technical Deliverables",
        "sections": {
            "sec2_eda": {
                "title": "2.2 Dataset Understanding & EDA",
                "missing_values": "0 across all 80 train + 10 test trips",
                "train_trips": 80,
                "test_trips": 10,
                "features_extracted": 27,
                "feature_version": "v2",
                "leakage_guard": "5-Fold GroupKFold by trip_id"
            },
            "sec3_models": {
                "title": "3. Model Experimentation & Comparison",
                "minimum_required": 5,
                "models_trained": 7,
                "models": [
                    {"id": "logreg", "name": "Logistic Regression", "family": "Linear / Probabilistic", "cv_f1": 0.8707, "top1": "88.15%", "continuity": "98.68%", "test_f1": 0.8631, "loss": 0.1662, "latency": "11.5ms"},
                    {"id": "knn", "name": "Distance-Weighted KNN", "family": "Instance-based", "cv_f1": 0.8917, "top1": "81.66%", "continuity": "98.12%", "test_f1": 0.8843, "loss": 0.2722, "latency": "8.32s"},
                    {"id": "rf", "name": "Random Forest", "family": "Tree Ensemble", "cv_f1": 0.9265, "top1": "88.70%", "continuity": "97.54%", "test_f1": 0.9241, "loss": 0.1039, "latency": "2.05s"},
                    {"id": "lgbm", "name": "LightGBM", "family": "Leaf-wise GBDT", "cv_f1": 0.9390, "top1": "90.14%", "continuity": "98.48%", "test_f1": 0.9418, "loss": 0.0654, "latency": "0.78s"},
                    {"id": "xgb", "name": "XGBoost", "family": "Depth-wise GBDT", "cv_f1": 0.9391, "top1": "89.76%", "continuity": "98.44%", "test_f1": 0.9391, "loss": 0.0672, "latency": "0.19s"},
                    {"id": "mlp", "name": "MLP Neural Network", "family": "Feed-Forward Neural Net", "cv_f1": 0.9313, "top1": "89.23%", "continuity": "98.26%", "test_f1": 0.9335, "loss": 0.0748, "latency": "48.3ms"},
                    {"id": "tuned", "name": "XGBoost (Optuna Tuned) + HMM", "family": "Bayesian GBDT + Sequence Decoder", "cv_f1": 0.9399, "top1": "92.47%", "continuity": "99.89%", "test_f1": 0.9427, "loss": 0.0644, "latency": "5.22s"}
                ]
            },
            "sec4_bayesian": {
                "title": "4. Bayesian Hyperparameter Optimisation",
                "sampler": "TPESampler (Optuna)",
                "trials": 35,
                "objective": "Minimize 5-fold CV log loss",
                "best_cv_log_loss": 0.0694,
                "gain": "+0.0052 log loss reduction over default",
                "best_params": {
                    "n_estimators": 200,
                    "learning_rate": 0.0322,
                    "max_depth": 8,
                    "subsample": 0.7736,
                    "colsample_bytree": 0.9076,
                    "reg_alpha": 0.7618,
                    "reg_lambda": 0.9836,
                    "min_child_weight": 1
                }
            },
            "sec5_hmm": {
                "title": "5. HMM and Viterbi Route Reconstruction",
                "hidden_states": "Directed road network arcs",
                "emission_probs": "Calibrated model class probabilities",
                "transition_probs": "Shortest path network distance exponential decay",
                "continuity_raw": "98.45%",
                "continuity_hmm": "99.89%",
                "top1_gain": "+1.29% (91.18% -> 92.47%)",
                "continuity_gain": "+1.44% (98.45% -> 99.89%)"
            },
            "sec6_folium": {
                "title": "6. Geographical Visualisation Using Folium",
                "mandatory": True,
                "layers": ["Raw GPS Observations", "Directed Road Network Arcs", "Ground Truth Reference Route", "Predicted Route", "Candidate Arcs Pool"],
                "total_maps": 10
            },
            "sec7_deployment": {
                "title": "7. API, UI and Deployment",
                "framework": "FastAPI (REST) + Leaflet (Canvas Vector Rendering) + Tailwind + GSAP",
                "port": 8000,
                "two_laptop_ready": True
            }
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
