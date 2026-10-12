"""Tests for the recent pool and the median decoder of the stratified retrieval.

Every check compares `StratifiedKNNRetrievalPredictor` (and `rolling_refit`)
with a brute-force search over a small synthetic table. The query is the
oracle (the target's own embedding), so no model is fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict._orchestration import rolling_refit
from aemo_curves.stages.predict.lag_median import weighted_median_curve
from aemo_curves.stages.predict.soft_knn import StratifiedKNNRetrievalPredictor

LAG = 7


@pytest.fixture
def table():
    rng = np.random.default_rng(1)
    rows = []
    for date in pd.date_range("2021-01-01", periods=120, freq="D").date:
        for interval in (1, 2):
            qty = np.cumsum(rng.uniform(1, 3, size=6))
            price = np.sort(rng.uniform(-50, 100, size=6))[::-1].copy()
            rows.append({
                "Trading Date": date,
                "Interval Number": interval,
                "curve_qty": qty,
                "curve_price": price,
                "embedding": rng.normal(size=4),
            })
    hist = pd.DataFrame(rows)
    cut = pd.Timestamp("2021-03-01").date()
    train = hist[hist["Trading Date"] < cut].reset_index(drop=True)
    test = hist[hist["Trading Date"] >= cut].reset_index(drop=True)
    return train, test, hist


def _pool(hist, date, interval, first, last, same_dow=False):
    """Rows of `interval` dated from `date - first` to `date - last` days, in date order."""
    ts = pd.to_datetime(hist["Trading Date"])
    d = pd.Timestamp(date)
    keep = (hist["Interval Number"] == interval) & (ts >= d - pd.Timedelta(days=first)) & (ts <= d - pd.Timedelta(days=last))
    if same_dow:
        keep &= ts.dt.dayofweek == d.dayofweek
    return hist[keep].sort_values("Trading Date").reset_index(drop=True)


def _ranked(pool, emb):
    d = np.linalg.norm(np.vstack(pool["embedding"].to_numpy()) - emb.reshape(1, -1), axis=1)
    return np.argsort(d, kind="stable")


def _truth(test, date, interval):
    return test[(test["Trading Date"] == date) & (test["Interval Number"] == interval)].iloc[0]


def test_default_pool_is_the_fitted_rows_of_the_same_interval_and_weekday(table):
    train, test, hist = table
    model = StratifiedKNNRetrievalPredictor(k=1, query="oracle", lag_days=LAG).fit(train)
    out = model.predict(test, hist)
    assert len(out) == len(test)
    assert {"pred_embedding", "retrieved_date"} <= set(out.columns)
    last_train = train["Trading Date"].max()
    for _, row in out.iterrows():
        pool = _pool(train, row["Trading Date"], row["Interval Number"], 10_000, 0, same_dow=True)
        emb = _truth(test, row["Trading Date"], row["Interval Number"])["embedding"]
        assert row["retrieved_date"] == pool["Trading Date"].iloc[_ranked(pool, emb)[0]]
        assert row["retrieved_date"] <= last_train


def test_recent_pool_moves_with_the_target_date(table):
    train, test, hist = table
    model = StratifiedKNNRetrievalPredictor(
        k=1, query="oracle", bucket="interval", lag_days=LAG, pool_max_age_days=10,
    ).fit(train)
    out = model.predict(test, hist)
    assert len(out) == len(test)
    for _, row in out.iterrows():
        pool = _pool(hist, row["Trading Date"], row["Interval Number"], LAG + 9, LAG)
        assert row["bucket_size"] == len(pool) == 10
        emb = _truth(test, row["Trading Date"], row["Interval Number"])["embedding"]
        assert row["retrieved_date"] == pool["Trading Date"].iloc[_ranked(pool, emb)[0]]
    assert (out["retrieved_date"] > train["Trading Date"].max()).any()


@pytest.mark.parametrize("range_rule", ["median", "ref"])
def test_median_topk_is_the_weighted_median_of_lags_and_nearest_curves(table, range_rule):
    train, test, hist = table
    lags = [[7, 3], [14, 1], [21, 1]]
    model = StratifiedKNNRetrievalPredictor(
        k=1, query="oracle", bucket="interval_dow", lag_days=LAG, pool_max_age_days=28,
        decode="median_topk", top_k=3, topk_weight=2, extra_lags=lags, range_rule=range_rule,
    ).fit(train)
    out = model.predict(test, hist)
    assert len(out) == len(test)
    assert "pred_embedding" not in out.columns and "retrieved_date" not in out.columns
    for _, row in out.iterrows():
        date, interval = row["Trading Date"], row["Interval Number"]
        pool = _pool(hist, date, interval, LAG + 27, LAG, same_dow=True)
        assert len(pool) == 4
        top = pool.iloc[_ranked(pool, _truth(test, date, interval)["embedding"])[:3]]
        curves = [(r["curve_qty"], r["curve_price"]) for lag, _ in lags
                  for r in [_pool(hist, date, interval, lag, lag).iloc[0]]]
        curves += list(zip(top["curve_qty"], top["curve_price"], strict=True))
        qty, price = weighted_median_curve(curves, [3, 1, 1, 2, 2, 2], range_rule=range_rule)
        np.testing.assert_array_equal(row["pred_curve_qty"], qty)
        np.testing.assert_array_equal(row["pred_curve_price"], price)
        assert row["retrieved_dates"] == list(top["Trading Date"])
        assert row["n_lags_used"] == 3 and row["n_neighbors_used"] == 3


def test_ref_range_without_lags_is_the_range_of_the_nearest_curve(table):
    train, test, hist = table
    model = StratifiedKNNRetrievalPredictor(
        k=1, query="oracle", bucket="interval", lag_days=LAG, pool_max_age_days=14,
        decode="median_topk", top_k=5, range_rule="ref",
    ).fit(train)
    out = model.predict(test, hist)
    for _, row in out.iterrows():
        pool = _pool(hist, row["Trading Date"], row["Interval Number"], LAG + 13, LAG)
        best = pool.iloc[_ranked(pool, _truth(test, row["Trading Date"], row["Interval Number"])["embedding"])[0]]
        assert row["pred_curve_qty"][0] == best["curve_qty"][0]
        assert row["pred_curve_qty"][-1] == best["curve_qty"][-1]


def test_rolling_refit_passes_the_options(table):
    train, test, hist = table
    ctx = PredictContext(train=train, test=test, historical=hist, side="demand", lag_days=LAG)
    out = rolling_refit(
        ctx, query="oracle", bucket="interval", pool_max_age_days=28, decode="median_topk",
        top_k=5, extra_lags=[[7, 3], [8, 1], [14, 1]],
    )
    assert len(out) == len(test)
    assert (out["bucket_size"] == 28).all()
    assert (out["n_neighbors_used"] == 5).all() and (out["n_lags_used"] == 3).all()
    assert out["refit_date"].nunique() == 2


@pytest.mark.parametrize("kwargs, match", [
    ({"bucket": "weekday"}, "bucket"),
    ({"decode": "mean"}, "decode"),
    ({"pool_max_age_days": 0}, "pool_max_age_days"),
    ({"decode": "median_topk", "top_k": 0}, "top_k"),
    ({"decode": "median_topk", "extra_lags": [[3, 1]]}, "horizon"),
    ({"decode": "median_topk", "extra_lags": [[7, 0]]}, "weight"),
    ({"decode": "median_topk", "extra_lags": [[7, 1], [7, 2]]}, "distinct"),
    ({"decode": "median_topk", "range_rule": "max"}, "range_rule"),
    ({"extra_lags": [[7, 1]]}, "median_topk"),
    ({"range_rule": "ref"}, "median_topk"),
])
def test_invalid_options_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StratifiedKNNRetrievalPredictor(query="oracle", lag_days=LAG, **kwargs)
