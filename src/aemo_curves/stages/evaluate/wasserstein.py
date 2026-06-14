"""Metrics for comparing predicted vs. true supply/demand curves.

The primary evaluation metric is the 1-D Wasserstein distance between the
price distributions implied by each curve, weighted by cumulative-quantity
increments. We also expose L2 distance between fixed-length embeddings as
a cheap secondary signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def _step_weights(curve_qty: np.ndarray) -> np.ndarray:
    """Per-step quantity increments, prepended with 0 so len == len(curve_qty)."""
    increments = np.diff(np.insert(np.asarray(curve_qty, dtype=float), 0, 0.0))
    return np.maximum(increments, 1e-9)


def curve_wasserstein_distance(
    true_qty: np.ndarray,
    true_price: np.ndarray,
    pred_qty: np.ndarray,
    pred_price: np.ndarray,
) -> float:
    wa = _step_weights(true_qty)
    wb = _step_weights(pred_qty)
    return float(
        wasserstein_distance(
            u_values=np.asarray(true_price, dtype=float),
            v_values=np.asarray(pred_price, dtype=float),
            u_weights=wa,
            v_weights=wb,
        )
    )


def embedding_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add per-row `curve_wasserstein` and (when embeddings present) `embedding_l2`.

    Expected columns:
      - true_curve_qty, true_curve_price
      - pred_curve_qty, pred_curve_price
      - (optional) true_embedding, pred_embedding
    """
    out = predictions.copy()
    out["curve_wasserstein"] = out.apply(
        lambda r: curve_wasserstein_distance(
            r["true_curve_qty"], r["true_curve_price"],
            r["pred_curve_qty"], r["pred_curve_price"],
        ),
        axis=1,
    )
    if "true_embedding" in out.columns and "pred_embedding" in out.columns:
        out["embedding_l2"] = out.apply(
            lambda r: embedding_l2_distance(r["true_embedding"], r["pred_embedding"]),
            axis=1,
        )
    return out


def summary_stats(scored: pd.DataFrame) -> dict[str, float]:
    s = scored["curve_wasserstein"]
    stats = {
        "n": float(len(s)),
        "wasserstein_mean": float(s.mean()),
        "wasserstein_median": float(s.median()),
        "wasserstein_p75": float(s.quantile(0.75)),
        "wasserstein_p95": float(s.quantile(0.95)),
        "wasserstein_std": float(s.std()),
    }
    if "baseline_wasserstein" in scored.columns:
        b = scored["baseline_wasserstein"]
        stats["baseline_wasserstein_mean"] = float(b.mean())
        stats["wasserstein_delta_vs_baseline"] = float((s - b).mean())
        stats["wasserstein_win_rate"] = float((s < b).mean())
    if "embedding_l2" in scored.columns:
        stats["embedding_l2_mean"] = float(scored["embedding_l2"].mean())
    return stats
