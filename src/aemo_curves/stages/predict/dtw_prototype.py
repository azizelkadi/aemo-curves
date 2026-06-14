"""Prototype-based predict strategies for the DTW track.

Each test curve is approximated by one of K learned medoid curves. `dtw_oracle`
picks the medoid closest to the true curve (the best a prototype method can do).
`dtw_predicted` fits a per-interval model that predicts the K-vector of
distances from lagged features and picks the medoid with the smallest predicted
distance. The runner scores the chosen medoid curve under DTW against the lag
baseline.

The feature tables are the kmedoids tables (`features: kmedoids`, with `k` in
the data block); the fitted medoid curves come from the cached embedding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.features import KMedoidsCurveEmbedding, default_embedding_path
from aemo_curves.stages.predict._features import calendar_features, hist_index, lag_row

LAGS_DEFAULT = (1, 2, 7, 14)


def _medoid_curves(ctx: PredictContext):
    k = ctx.extra.get("k")
    emb = KMedoidsCurveEmbedding.load(default_embedding_path(side=ctx.side, k=k))
    return emb.medoid_curves


def _row(test_row, qty, price) -> dict:
    return {
        "Trading Date": test_row["Trading Date"],
        "Interval Number": int(test_row["Interval Number"]),
        "true_curve_qty": test_row["curve_qty"],
        "true_curve_price": test_row["curve_price"],
        "pred_curve_qty": qty,
        "pred_curve_price": price,
    }


@register("predict", "dtw_oracle")
def dtw_oracle(ctx: PredictContext) -> pd.DataFrame:
    medoids = _medoid_curves(ctx)
    rows = []
    for _, r in ctx.test.iterrows():
        idx = int(np.asarray(r["kmedoids_embedding"], dtype=float).argmin())
        rows.append(_row(r, medoids[idx][0], medoids[idx][1]))
    return pd.DataFrame(rows)


def _features(interval, trading_date, table, idx, lags):
    parts = [calendar_features(trading_date)]
    for lag in lags:
        prev = lag_row(table, idx, interval, trading_date, lag)
        if prev is None:
            return None
        parts.append(np.asarray(prev["kmedoids_embedding"], dtype=float))
    return np.concatenate(parts, axis=0)


def _regressor(query: str, gbm_kwargs: dict):
    if query == "ridge":
        return Ridge(alpha=gbm_kwargs.get("alpha", 5.0))
    if query == "gbm":
        base = HistGradientBoostingRegressor(
            max_iter=gbm_kwargs.get("max_iter", 80),
            max_depth=gbm_kwargs.get("max_depth", 4),
            learning_rate=gbm_kwargs.get("learning_rate", 0.05),
            l2_regularization=gbm_kwargs.get("l2_regularization", 1.0),
            early_stopping=True,
            random_state=0,
        )
        return MultiOutputRegressor(base)
    raise ValueError(f"dtw_predicted query must be 'ridge' or 'gbm', got {query!r}")


@register("predict", "dtw_predicted")
def dtw_predicted(
    ctx: PredictContext,
    query: str = "ridge",
    lags: tuple = LAGS_DEFAULT,
    min_samples: int = 30,
    **gbm_kwargs,
) -> pd.DataFrame:
    lags = tuple(lags)
    medoids = _medoid_curves(ctx)
    train, test = ctx.train, ctx.test
    full = pd.concat([train, test], ignore_index=True)
    full_idx = hist_index(full)
    train_idx = hist_index(train)

    models: dict[int, dict] = {}
    for interval, sub in train.groupby("Interval Number"):
        xs, ys = [], []
        for _, row in sub.iterrows():
            feat = _features(int(interval), row["Trading Date"], train, train_idx, lags)
            if feat is None:
                continue
            xs.append(feat)
            ys.append(np.asarray(row["kmedoids_embedding"], dtype=float))
        if len(xs) < min_samples:
            continue
        X, Y = np.vstack(xs), np.vstack(ys)
        x_scaler = StandardScaler().fit(X)
        y_scaler = StandardScaler().fit(Y)
        reg = _regressor(query, gbm_kwargs)
        reg.fit(x_scaler.transform(X), y_scaler.transform(Y))
        models[int(interval)] = {"x_scaler": x_scaler, "y_scaler": y_scaler, "reg": reg}

    rows = []
    for _, r in test.iterrows():
        interval = int(r["Interval Number"])
        m = models.get(interval)
        if m is None:
            continue
        feat = _features(interval, r["Trading Date"], full, full_idx, lags)
        if feat is None:
            continue
        xs = m["x_scaler"].transform(feat.reshape(1, -1))
        pred = np.atleast_2d(m["reg"].predict(xs))
        kvec = m["y_scaler"].inverse_transform(pred)[0]
        idx = int(kvec.argmin())
        rows.append(_row(r, medoids[idx][0], medoids[idx][1]))
    return pd.DataFrame(rows)
