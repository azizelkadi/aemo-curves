"""Fixed-length embedding of a price/quantity curve via interpolation on a quantile grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aemo_curves.config import DEFAULT_QUANTITY_GRID, EMBEDDINGS_DIR


def curve_to_embedding(
    curve_qty: np.ndarray,
    curve_price: np.ndarray,
    quantity_grid: np.ndarray = DEFAULT_QUANTITY_GRID,
) -> np.ndarray:
    """Normalise cumulative quantity to [0, 1] and interpolate price on the grid.

    Returns a fixed-length float vector with the same shape as `quantity_grid`.
    """
    max_qty = float(curve_qty[-1]) if len(curve_qty) else 0.0
    if max_qty <= 0:
        return np.zeros_like(quantity_grid, dtype=float)
    norm_qty = np.clip(curve_qty / max_qty, 0.0, 1.0)
    emb = np.interp(
        quantity_grid,
        norm_qty,
        curve_price,
        left=curve_price[0],
        right=curve_price[-1],
    )
    return emb.astype(float)


def add_curve_embeddings(
    curve_table: pd.DataFrame,
    quantity_grid: np.ndarray = DEFAULT_QUANTITY_GRID,
) -> pd.DataFrame:
    """Append an `embedding` column to a curve table."""
    out = curve_table.copy()
    out["embedding"] = out.apply(
        lambda row: curve_to_embedding(row["curve_qty"], row["curve_price"], quantity_grid),
        axis=1,
    )
    return out


def _embedding_path(name: str, side: str, directory: Path = EMBEDDINGS_DIR) -> Path:
    return directory / f"{name}_{side}.pkl"


def save_embedding_table(
    table: pd.DataFrame,
    name: str,
    side: str,
    directory: Path = EMBEDDINGS_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = _embedding_path(name, side, directory)
    table.to_pickle(path)
    return path


def load_embedding_table(
    name: str,
    side: str,
    directory: Path = EMBEDDINGS_DIR,
) -> pd.DataFrame:
    return pd.read_pickle(_embedding_path(name, side, directory))
