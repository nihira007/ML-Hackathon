"""Tiny table IO helper: parquet when an engine (pyarrow / fastparquet) exists, pickle fallback otherwise.

The shared contract specifies ``.parquet`` files. On machines with ``pyarrow`` installed (see
requirements.txt) real parquet is written. If no parquet engine is importable we write a
``.pkl`` next to the intended path and say so loudly instead of failing.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".pkl")
        df.to_pickle(alt)
        warnings.warn(f"No parquet engine installed (pip install pyarrow): wrote {alt.name} instead of {path.name}", stacklevel=2)
        return alt


def load_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix(".pkl")
    if alt.exists():
        return pd.read_pickle(alt)
    raise FileNotFoundError(f"{path} not found - run `python -m src.features.build_features` first")
