"""Project-wide paths, constants, and the canonical train/test split."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CURVES_DIR = ARTIFACTS_DIR / "curves"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

for _dir in (ARTIFACTS_DIR, CURVES_DIR, EMBEDDINGS_DIR, MLRUNS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _months_between(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


# IMPORTANT: the test period is fixed and must never be used for training or model selection.
TRAIN_START = "2021-01"
TRAIN_END = "2024-04"
TEST_START = "2024-05"
TEST_END = "2025-05"

TRAIN_MONTHS: list[str] = _months_between(TRAIN_START, TRAIN_END)
TEST_MONTHS: list[str] = _months_between(TEST_START, TEST_END)

# Curve/embedding conventions
MIN_POINTS_PER_AUCTION = 4

# 64 quantile points, trimmed of endpoints to avoid edge noise.
DEFAULT_QUANTITY_GRID: np.ndarray = np.linspace(0.02, 0.98, 64)

# MLflow defaults
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR.as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aemo-curves")


@dataclass(frozen=True)
class Split:
    """An alternative train/test recut of the cached embedding tables."""

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# Recut keys usable by core.datasets.load_features(split=...).
# "canonical" is NOT listed: the cached train/test pkls already encode it, so it
# needs no recut. "user2024" recuts the combined table at 2023-12-31 / 2024.
SPLITS: dict[str, Split] = {
    "user2024": Split(
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}
