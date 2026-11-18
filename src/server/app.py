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

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
