"""Shared retrieval primitives for the regression-retrieval family."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._features import hist_index

key_index = hist_index


def library_by_interval(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(i): sub.reset_index(drop=True)
        for i, sub in df.groupby("Interval Number")
    }


def l2_retrieve(library: pd.DataFrame, pred_emb: np.ndarray) -> tuple[int, np.ndarray]:
    candidates = np.vstack(library["embedding"].to_numpy())
    distances = np.linalg.norm(candidates - pred_emb.reshape(1, -1), axis=1)
    best = int(np.argmin(distances))
    return best, distances


def retrieval_row(r, interval, pred_emb, src, dist_best) -> dict:
    return {
        "Trading Date": r["Trading Date"],
        "Interval Number": interval,
        "true_embedding": r["embedding"],
        "pred_embedding": pred_emb,
        "true_curve_qty": r["curve_qty"],
        "true_curve_price": r["curve_price"],
        "pred_curve_qty": src["curve_qty"],
        "pred_curve_price": src["curve_price"],
        "retrieved_date": src["Trading Date"],
        "embedding_retrieval_distance": float(dist_best),
        "embedding_l2": float(np.linalg.norm(r["embedding"] - pred_emb)),
    }
