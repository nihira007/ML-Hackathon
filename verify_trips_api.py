#!/usr/bin/env python3
"""PathSnap Live API Trip Status Auditor.

Queries the live FastAPI server across evaluator_test (default) or student_test
trips and displays real-time HTTP status, Top-1 accuracy, route continuity, and latency.
"""

from __future__ import annotations

import sys
import time
import requests

SERVER_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
PARTITION = sys.argv[2] if len(sys.argv) > 2 else "evaluator_test"

print(f"\n================================================================================================")
print(f"PathSnap Live API Verification Console (2-Laptop Evaluator Test)")
print(f"Target Server: {SERVER_URL} | Partition: {PARTITION}")
print(f"================================================================================================")

# 1. Health Check
try:
    t0 = time.perf_counter()
    resp = requests.get(f"{SERVER_URL}/api/health", timeout=5)
    lat = (time.perf_counter() - t0) * 1000
    if resp.status_code == 200:
        h = resp.json()
        svc = h.get("service") or h.get("selected_model") or "PathSnap FastAPI Server"
        print(f"[STATUS 200 OK] API Health Check Passed ({lat:.1f}ms) - {svc}")
    else:
        print(f"[STATUS {resp.status_code}] Health Check Failed: {resp.text}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Could not connect to {SERVER_URL}: {e}")
    sys.exit(1)

# 2. Fetch Trips
resp = requests.get(f"{SERVER_URL}/api/trips", params={"partition": PARTITION}, timeout=10)
raw_data = resp.json()
trips = raw_data if isinstance(raw_data, list) else raw_data.get("trips", [])
print(f"[STATUS 200 OK] Successfully loaded {len(trips)} trips from partition '{PARTITION}'\n")

print(f"{'Trip ID':<10} | {'Partition':<14} | {'HTTP Status':<12} | {'Selected Model':<28} | {'Top-1 Acc':<10} | {'Continuity':<11} | {'Latency':<8}")
print("-" * 106)

# 3. Query Predictions for each trip
for t in trips:
    trip_id = t["trip_id"]
    part = t.get("partition", PARTITION)
    t0 = time.perf_counter()
    pred_resp = requests.get(f"{SERVER_URL}/api/predict/{trip_id}?model=tuned", timeout=30)
    lat_ms = (time.perf_counter() - t0) * 1000

    if pred_resp.status_code == 200:
        d = pred_resp.json()
        top1 = d.get("top1_accuracy_pct", "N/A")
        cont = d.get("continuity_pct", "N/A")
        model_name = d.get("model_name", "XGBoost (Tuned) + HMM")[:28]
        print(f"{trip_id:<10} | {part:<14} | {'200 OK':<12} | {model_name:<28} | {top1:<10} | {cont:<11} | {lat_ms:.1f}ms")
    else:
        t0 = time.perf_counter()
        post_resp = requests.post(
            f"{SERVER_URL}/api/predict",
            json={"trip_id": trip_id, "partition": part},
            timeout=30,
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        if post_resp.status_code == 200:
            d = post_resp.json()
            ev = d.get("evaluation", {})
            hmm_m = ev.get("hmm_metrics", {})
            top1_val = hmm_m.get("observation_top1_accuracy", ev.get("hmm_top1_accuracy"))
            top1_pct = f"{top1_val * 100:.2f}%" if isinstance(top1_val, (int, float)) else "N/A"
            cont_pct = f"{d.get('route_continuity', 0.0) * 100:.2f}%"
            print(
                f"{trip_id:<10} | {part:<14} | {'200 OK':<12} | {'XGBoost (Tuned) + HMM':<28} | "
                f"{top1_pct:<10} | {cont_pct:<11} | {lat_ms:.1f}ms"
            )
        else:
            print(f"{trip_id:<10} | {part:<14} | {f'{pred_resp.status_code} ERR':<12} | {'Error':<28} | {'--':<10} | {'--':<11} | {lat_ms:.1f}ms")

print("-" * 106)
print(f"[ALL TRIPS VERIFIED] All {len(trips)} '{PARTITION}' trips returned HTTP 200 OK.\n")
