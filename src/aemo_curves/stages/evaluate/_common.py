"""Generic, metric-parameterized scoring.

The column names and summary-key names are derived from `metric.name`, so the
same functions score predictions under any `Metric`.
"""
from __future__ import annotations

import pandas as pd

from aemo_curves.core.metric import Metric
from aemo_curves.stages.evaluate.metrics import WASSERSTEIN

from .wasserstein import embedding_l2_distance


def score_predictions(predictions: pd.DataFrame, metric: Metric = WASSERSTEIN) -> pd.DataFrame:
    out = predictions.copy()
    out[metric.curve_col] = out.apply(
        lambda r: metric.distance(
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


def attach_baseline(
    scored: pd.DataFrame, baseline_scored: pd.DataFrame, metric: Metric = WASSERSTEIN
) -> pd.DataFrame:
    """Merge the baseline distance onto the model rows by (date, interval).

    The join normalises `Trading Date` on both sides, so a strategy that carries
    a `datetime64` date still aligns with one that carries a Python `date`.
    """
    b = baseline_scored[["Trading Date", "Interval Number", metric.curve_col]].rename(
        columns={metric.curve_col: metric.baseline_col}
    ).copy()
    left = scored.copy()
    left["_key_date"] = pd.to_datetime(left["Trading Date"]).dt.normalize()
    b["_key_date"] = pd.to_datetime(b["Trading Date"]).dt.normalize()
    b = b.drop(columns=["Trading Date"])
    merged = left.merge(b, on=["_key_date", "Interval Number"], how="left")
    return merged.drop(columns=["_key_date"])


def summary_stats(scored: pd.DataFrame, metric: Metric = WASSERSTEIN) -> dict[str, float]:
    n = metric.name
    s = scored[metric.curve_col]
    stats = {
        "n": float(len(s)),
        f"{n}_mean": float(s.mean()),
        f"{n}_median": float(s.median()),
        f"{n}_p75": float(s.quantile(0.75)),
        f"{n}_p95": float(s.quantile(0.95)),
        f"{n}_std": float(s.std()),
    }
    if metric.baseline_col in scored.columns:
        b = scored[metric.baseline_col]
        # Compare only where a baseline row matched, so a missing match cannot
        # silently deflate the win rate (NaN < x is False) or the delta.
        paired = b.notna()
        sp, bp = s[paired], b[paired]
        stats[f"baseline_{n}_mean"] = float(bp.mean())
        stats[f"{n}_delta_vs_baseline"] = float((sp - bp).mean())
        stats[f"{n}_win_rate"] = float((sp < bp).mean())
    if "embedding_l2" in scored.columns:
        stats["embedding_l2_mean"] = float(scored["embedding_l2"].mean())
    return stats
