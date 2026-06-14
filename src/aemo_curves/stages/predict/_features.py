"""Shared feature-engineering helpers for residual / corrector predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hist_index(df: pd.DataFrame) -> dict[tuple[int, object], int]:
    return {
        (int(r["Interval Number"]), r["Trading Date"]): i
        for i, (_, r) in enumerate(df.iterrows())
    }


def lag_row(
    hist: pd.DataFrame,
    index: dict[tuple[int, object], int],
    interval: int,
    trading_date,
    lag_days: int,
) -> pd.Series | None:
    lag_date = (pd.Timestamp(trading_date) - pd.Timedelta(days=lag_days)).date()
    idx = index.get((interval, lag_date))
    if idx is None:
        return None
    return hist.iloc[idx]


def calendar_features(trading_date) -> np.ndarray:
    ts = pd.Timestamp(trading_date)
    dow = ts.dayofweek
    month = ts.month
    return np.array(
        [
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            float(dow >= 5),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ],
        dtype=float,
    )


def feature_from_lags(
    interval: int,
    trading_date,
    hist: pd.DataFrame,
    hist_index: dict[tuple[int, object], int],
    lags: tuple[int, ...],
    use_calendar: bool,
    use_trend: bool,
    use_qty_total: bool = False,
    exog_provider=None,
) -> np.ndarray | None:
    ts = pd.Timestamp(trading_date)
    vecs: list[np.ndarray] = []
    lag_embeddings: list[np.ndarray] = []
    qty_total_first: float | None = None
    for lag in lags:
        idx = hist_index.get((interval, (ts - pd.Timedelta(days=lag)).date()))
        if idx is None:
            return None
        row = hist.iloc[idx]
        emb = np.asarray(row["embedding"], dtype=float)
        lag_embeddings.append(emb)
        vecs.append(emb)
        if use_qty_total and qty_total_first is None:
            qty = np.asarray(row["curve_qty"], dtype=float)
            qty_total_first = float(qty[-1]) if len(qty) else 0.0
    if use_calendar:
        vecs.append(calendar_features(trading_date))
    if use_trend and len(lag_embeddings) >= 2:
        vecs.append(lag_embeddings[0] - lag_embeddings[1])
    if use_qty_total:
        v = qty_total_first if qty_total_first is not None and qty_total_first > 0 else 1.0
        vecs.append(np.array([np.log(v)], dtype=float))
    if exog_provider is not None:
        ex = exog_provider(int(interval), trading_date)
        if ex is None:
            return None
        vecs.append(np.asarray(ex, dtype=float))
    return np.concatenate(vecs, axis=0)


def curve_stats(qty: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Cheap summary stats for a single curve.

    Returns [mean, qty_weighted_mean, total_qty, p25, p50, p75, std].
    """
    qty = np.asarray(qty, dtype=float)
    price = np.asarray(price, dtype=float)
    w = np.diff(np.insert(qty, 0, 0.0))
    w = np.maximum(w, 1e-9)
    wm = float((price * w).sum() / w.sum())
    return np.array(
        [
            float(price.mean()),
            wm,
            float(qty[-1]),
            float(np.percentile(price, 25)),
            float(np.percentile(price, 50)),
            float(np.percentile(price, 75)),
            float(price.std()),
        ],
        dtype=float,
    )


def build_daily_regime_stats(embeddings: pd.DataFrame) -> dict:
    """For each Trading Date, aggregate-level stats across all 48 intervals.

    Returns dict: date -> np.ndarray [mean_of_mean_prices, sum_of_total_qty, std_of_mean_prices].
    Useful as a regime indicator - the yesterday-wide price level is a strong
    predictor of today's shift.
    """
    agg = embeddings.groupby("Trading Date").agg(
        day_mean_price=("mean_price", "mean"),
        day_total_qty=("total_quantity", "sum"),
        day_std_price=("mean_price", "std"),
    )
    agg = agg.fillna(0.0)
    return {
        date: np.array(row.values, dtype=float) for date, row in agg.iterrows()
    }


def daily_regime(day_stats: dict, trading_date, lag_days: int) -> np.ndarray | None:
    key = (pd.Timestamp(trading_date) - pd.Timedelta(days=lag_days)).date()
    v = day_stats.get(key)
    if v is None:
        return None
    return v
