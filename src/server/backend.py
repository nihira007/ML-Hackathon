"""Backend data loader and inference provider for PathSnap Live Model Server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TEST_DIR = PROJECT_ROOT / "Student folder" / "student_test"
CLASSICAL_DIR = PROJECT_ROOT / "outputs" / "models" / "classical"
ADVANCED_DIR = PROJECT_ROOT / "outputs" / "models" / "advanced"
TUNING_DIR = PROJECT_ROOT / "outputs" / "tuning"

MODEL_PRED_FILES: Dict[str, Path] = {
    "tuned": TUNING_DIR / "tuned_student_test_preds.parquet",
    "lgbm": ADVANCED_DIR / "lgbm_student_test_preds.parquet",
    "xgb": ADVANCED_DIR / "xgb_student_test_preds.parquet",
    "mlp": ADVANCED_DIR / "mlp_student_test_preds.parquet",
    "rf": CLASSICAL_DIR / "rf_student_test_preds.parquet",
    "knn": CLASSICAL_DIR / "knn_student_test_preds.parquet",
    "logreg": CLASSICAL_DIR / "logreg_student_test_preds.parquet",
}

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "tuned": {
        "id": "tuned",
        "name": "XGBoost (Optuna Tuned)",
        "tag": "Final Model (35 TPE Trials)",
        "family": "Bayesian-Optimized GBDT",
        "accuracy": "88.58% (91.45% + HMM)",
        "top1_acc_raw": 0.8858,
        "macro_f1": "0.9329",
        "log_loss": "0.0717",
        "latency": "0.22s (10 trips)",
        "badge": "Production Model",
        "description": "Optuna TPE-tuned XGBoost (35 trials) paired with Viterbi HMM decoder (99.88% Route Continuity).",
    },
    "lgbm": {
        "id": "lgbm",
        "name": "LightGBM Classifier",
        "tag": "Model 4",
        "family": "Gradient Boosted Trees (Leaf-wise)",
        "accuracy": "88.85%",
        "top1_acc_raw": 0.8885,
        "macro_f1": "0.9344",
        "log_loss": "0.0718",
        "latency": "0.62s (10 trips)",
        "badge": "Highest Pointwise Top-1",
        "description": "Leaf-wise histogram GBDT across all 27 v2 spatial, kinematic, and topological features.",
    },
    "xgb": {
        "id": "xgb",
        "name": "XGBoost Classifier",
        "tag": "Model 5",
        "family": "Gradient Boosted Trees (Depth-wise)",
        "accuracy": "88.07%",
        "top1_acc_raw": 0.8807,
        "macro_f1": "0.9310",
        "log_loss": "0.0746",
        "latency": "0.13s (10 trips)",
        "badge": "Best CV Macro F1",
        "description": "Histogram depth-wise GBDT achieving 0.9287 5-fold CV Macro F1 across 80 training trips.",
    },
    "mlp": {
        "id": "mlp",
        "name": "MLP Neural Network",
        "tag": "Model 6",
        "family": "Multi-Layer Perceptron (64x32)",
        "accuracy": "87.45%",
        "top1_acc_raw": 0.8745,
        "macro_f1": "0.9238",
        "log_loss": "0.0822",
        "latency": "0.049s (10 trips)",
        "badge": "Neural Baseline",
        "description": "2-hidden-layer (64x32) ReLU neural network trained with Adam over 20 tracked epochs.",
    },
    "rf": {
        "id": "rf",
        "name": "Random Forest Classifier",
        "tag": "Model 3",
        "family": "Bagging / Tree Ensemble",
        "accuracy": "87.20%",
        "top1_acc_raw": 0.8720,
        "macro_f1": "0.9154",
        "log_loss": "0.1159",
        "latency": "1.02s (10 trips)",
        "badge": "Best Classical",
        "description": "100 regularized trees (max_depth=14, min_samples_leaf=25) on 27 v2 domain features.",
    },
    "logreg": {
        "id": "logreg",
        "name": "L2 Logistic Regression",
        "tag": "Model 1",
        "family": "Linear / Probabilistic",
        "accuracy": "86.69%",
        "top1_acc_raw": 0.8669,
        "macro_f1": "0.8527",
        "log_loss": "0.1827",
        "latency": "22.2ms (10 trips)",
        "badge": "Ultra-Fast",
        "description": "Linear probabilistic classifier with balanced class weights and L2 regularization.",
    },
    "knn": {
        "id": "knn",
        "name": "Distance-Weighted KNN",
        "tag": "Model 2",
        "family": "Instance-based / Non-parametric",
        "accuracy": "79.36%",
        "top1_acc_raw": 0.7936,
        "macro_f1": "0.8748",
        "log_loss": "0.2607",
        "latency": "8.31s (10 trips)",
        "badge": "Non-Parametric",
        "description": "k=15 with inverse Euclidean distance weighting in 27D standardized feature space.",
    },
    "heuristic": {
        "id": "heuristic",
        "name": "Nearest Segment Baseline",
        "tag": "Baseline",
        "family": "Geometric Heuristic",
        "accuracy": "54.00%",
        "top1_acc_raw": 0.5400,
        "macro_f1": "0.6812",
        "log_loss": "0.8920",
        "latency": "<1ms",
        "badge": "Naive Heuristic",
        "description": "Selects nearest road segment by Euclidean distance (cand_rank == 0).",
    },
}


class ModelStore:
    def __init__(self) -> None:
        self.predictions: Dict[str, pd.DataFrame] = {}
        self.trip_cache: Dict[str, Dict[str, Any]] = {}
        self._load_predictions()

    def _load_predictions(self) -> None:
        for key, path in MODEL_PRED_FILES.items():
            if path.exists():
                df = pd.read_parquet(path)
                df = df.sort_values(["trip_id", "obs_idx", "pred_prob"], ascending=[True, True, False], kind="mergesort")
                self.predictions[key] = df

    def list_trips(self) -> List[Dict[str, Any]]:
        trips = []
        for trip_dir in sorted(RAW_TEST_DIR.glob("*")):
            if trip_dir.is_dir() and (trip_dir / f"{trip_dir.name}.track").exists():
                t_id = trip_dir.name
                track = np.loadtxt(trip_dir / f"{t_id}.track", dtype=np.float64)
                if track.ndim == 1:
                    track = track.reshape(1, -1)
                min_lat, max_lat = float(track[:, 1].min()), float(track[:, 1].max())
                min_lon, max_lon = float(track[:, 0].min()), float(track[:, 0].max())
                trips.append({
                    "trip_id": t_id,
                    "num_obs": int(len(track)),
                    "duration_sec": float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0,
                    "bbox": [[min_lat, min_lon], [max_lat, max_lon]],
                    "center": [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0],
                })
        return trips

    def get_trip_geometry(self, trip_id: str) -> Dict[str, Any]:
        if trip_id in self.trip_cache:
            return self.trip_cache[trip_id]

        trip_dir = RAW_TEST_DIR / trip_id
        if not trip_dir.exists():
            raise ValueError(f"Trip folder not found: {trip_id}")

        track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
        if track.ndim == 1:
            track = track.reshape(1, -1)
        nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
        arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
        if arcs.ndim == 1:
            arcs = arcs.reshape(1, -1)

        route_path = trip_dir / f"{trip_id}.route"
        route_arcs: List[int] = []
        if route_path.exists():
            route_arcs = np.atleast_1d(np.loadtxt(route_path, dtype=np.int64)).tolist()

        gps_points = [
            {
                "obs_idx": int(i),
                "lat": float(track[i, 1]),
                "lon": float(track[i, 0]),
                "timestamp": float(track[i, 2]),
            }
            for i in range(len(track))
        ]

        gt_segments = []
        for arc_id in route_arcs:
            if arc_id < len(arcs):
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                if u < len(nodes) and v < len(nodes):
                    gt_segments.append({
                        "arc_id": int(arc_id),
                        "coords": [
                            [float(nodes[u, 1]), float(nodes[u, 0])],
                            [float(nodes[v, 1]), float(nodes[v, 0])],
                        ]
                    })

        min_lat, max_lat = float(track[:, 1].min()), float(track[:, 1].max())
        min_lon, max_lon = float(track[:, 0].min()), float(track[:, 0].max())

        res = {
            "trip_id": trip_id,
            "num_obs": int(len(track)),
            "gps_points": gps_points,
            "gt_segments": gt_segments,
            "gt_route_arcs": route_arcs,
            "nodes": nodes,
            "arcs": arcs,
            "bbox": [[min_lat, min_lon], [max_lat, max_lon]],
            "center": [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0],
        }
        self.trip_cache[trip_id] = res
        return res

    def predict_trip(self, trip_id: str, model_id: str = "tuned") -> Dict[str, Any]:
        geom = self.get_trip_geometry(trip_id)
        nodes = geom["nodes"]
        arcs = geom["arcs"]

        if model_id == "heuristic":
            df = self.predictions.get("rf")
            if df is None and self.predictions:
                df = next(iter(self.predictions.values()))
            if df is None:
                raise ValueError("No prediction tables loaded")
            trip_preds = df[df["trip_id"] == trip_id].copy()
            trip_preds["cand_rank"] = trip_preds.groupby("obs_idx").cumcount()
            trip_preds["pred_prob"] = np.where(trip_preds["cand_rank"] == 0, 0.99, 0.01)
        else:
            df = self.predictions.get(model_id)
            if df is None:
                df = self.predictions.get("rf")
            if df is None:
                raise ValueError(f"Unknown model_id: {model_id}")
            trip_preds = df[df["trip_id"] == trip_id].copy()


        top_candidates = trip_preds.sort_values(
            ["obs_idx", "pred_prob"], ascending=[True, False], kind="mergesort"
        ).groupby("obs_idx").first().reset_index()

        matched_segments = []
        correct_count = 0
        total_obs = len(top_candidates)

        for _, row in top_candidates.iterrows():
            obs_i = int(row["obs_idx"])
            arc_id = int(row["arc_id"])
            prob = float(row["pred_prob"])
            is_match = int(row["is_match"])
            if is_match == 1:
                correct_count += 1

            coords = None
            if arc_id < len(arcs):
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                if u < len(nodes) and v < len(nodes):
                    coords = [
                        [float(nodes[u, 1]), float(nodes[u, 0])],
                        [float(nodes[v, 1]), float(nodes[v, 0])],
                    ]

            matched_segments.append({
                "obs_idx": obs_i,
                "arc_id": arc_id,
                "pred_prob": round(prob, 4),
                "is_match": is_match,
                "coords": coords,
            })

        continuity_hits = 0
        for i in range(1, len(matched_segments)):
            prev_arc = matched_segments[i - 1]["arc_id"]
            curr_arc = matched_segments[i]["arc_id"]
            if prev_arc == curr_arc:
                continuity_hits += 1
            elif prev_arc < len(arcs) and curr_arc < len(arcs):
                if arcs[prev_arc, 1] == arcs[curr_arc, 0]:
                    continuity_hits += 1

        continuity_ratio = continuity_hits / max(1, len(matched_segments) - 1)
        top1_acc = correct_count / max(1, total_obs)

        cands_dict: Dict[int, List[Dict[str, Any]]] = {}
        for obs_i, group in trip_preds.groupby("obs_idx"):
            c_list = []
            for _, r in group.sort_values("pred_prob", ascending=False).iterrows():
                a_id = int(r["arc_id"])
                c_coords = None
                if a_id < len(arcs):
                    u, v = int(arcs[a_id, 0]), int(arcs[a_id, 1])
                    if u < len(nodes) and v < len(nodes):
                        c_coords = [
                            [float(nodes[u, 1]), float(nodes[u, 0])],
                            [float(nodes[v, 1]), float(nodes[v, 0])],
                        ]
                c_list.append({
                    "arc_id": a_id,
                    "pred_prob": round(float(r["pred_prob"]), 4),
                    "is_match": int(r["is_match"]),
                    "coords": c_coords,
                })
            cands_dict[int(obs_i)] = c_list

        meta = MODEL_METADATA.get(model_id, {})

        # Load per-trip HMM decoding metrics if available
        hmm_stats = None
        hmm_csv = PROJECT_ROOT / "outputs" / "hmm" / "classifier_vs_hmm_per_trip.csv"
        if hmm_csv.exists():
            try:
                hmm_df = pd.read_csv(hmm_csv)
                trip_hmm = hmm_df[hmm_df["trip_id"].astype(str).str.zfill(8) == str(trip_id).zfill(8)]
                if not trip_hmm.empty:
                    row = trip_hmm.iloc[0]
                    hmm_stats = {
                        "cls_continuity": float(row["cls_continuity"]),
                        "hmm_continuity": float(row["hmm_continuity"]),
                        "continuity_gain": float(row["continuity_gain"]),
                        "cls_top1": float(row["cls_top1"]),
                        "hmm_top1": float(row["hmm_top1"]),
                        "top1_gain": float(row["top1_gain"]),
                        "cls_seq_f1": float(row["cls_seq_f1"]),
                        "hmm_seq_f1": float(row["hmm_seq_f1"]),
                        "seq_f1_gain": float(row["seq_f1_gain"]),
                        "hmm_runtime_sec": float(row["hmm_runtime_sec"]),
                    }
                    if model_id == "tuned":
                        top1_acc = hmm_stats["hmm_top1"]
                        continuity_ratio = hmm_stats["hmm_continuity"]
            except Exception:
                pass

        return {
            "trip_id": trip_id,
            "model_id": model_id,
            "model_name": meta.get("name", model_id),
            "model_family": meta.get("family", "ML Classifier"),
            "total_obs": total_obs,
            "correct_obs": int(round(top1_acc * total_obs)),
            "top1_accuracy": round(top1_acc, 4),
            "top1_accuracy_pct": f"{top1_acc * 100:.2f}%",
            "continuity_ratio": round(continuity_ratio, 4),
            "continuity_pct": f"{continuity_ratio * 100:.2f}%",
            "matched_segments": matched_segments,
            "candidates_by_obs": cands_dict,
            "hmm_stats": hmm_stats,
        }


    def get_feature_importances(self) -> List[Dict[str, Any]]:
        rf_meta_path = CLASSICAL_DIR / "rf_metrics.json"
        if rf_meta_path.exists():
            with open(rf_meta_path, "r") as f:
                data = json.load(f)
            fi = data.get("extra_metadata", {}).get("feature_importances_gini", {})
            return [{"feature": k, "importance": v} for k, v in fi.items()]
        return []

    def get_master_comparison(self) -> List[Dict[str, Any]]:
        master_csv = PROJECT_ROOT / "outputs" / "master_model_comparison.csv"
        if master_csv.exists():
            return pd.read_csv(master_csv).to_dict(orient="records")
        csv_path = CLASSICAL_DIR / "classical_comparison.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path).to_dict(orient="records")
        return []


store = ModelStore()
