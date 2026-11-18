"""Laptop 2 — Lightweight HTTP Server for the PathSnap Evaluator UI."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


def run_ui_server(host: str = "0.0.0.0", port: int = 8501) -> None:
    os.chdir(UI_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"[LAPTOP-2-UI] Serving PathSnap Evaluator UI at http://{host}:{port}")
        print(f"[LAPTOP-2-UI] Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[LAPTOP-2-UI] Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PathSnap Evaluator UI (Laptop 2)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    run_ui_server(host=args.host, port=args.port)
