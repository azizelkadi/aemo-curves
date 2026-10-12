"""Tests for the volume_shift strategy: cliff volume, shift, history rule, end to end."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict.volume_shift import (
    cliff_volume,
    shift_keep_first_edge,
    volume_features,
    volume_shift,
)


def _demand_curve(rng):
    n = int(rng.integers(4, 9))
    qty = np.cumsum(rng.uniform(5, 50, size=n))
    price = np.sort(rng.uniform(-500, 300, size=n))[::-1].copy()
    price[-1] = -1000.0
    return qty, price


@pytest.fixture
def history():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2021-01-01", periods=90, freq="D").date:
        for interval in (1, 2):
            qty, price = _demand_curve(rng)
            rows.append({"Trading Date": d, "Interval Number": interval,
                         "curve_qty": qty, "curve_price": price})
    return pd.DataFrame(rows)


def test_cliff_volume():
    qty = np.array([1.0, 5.0, 9.0, 20.0])
    assert cliff_volume(qty, np.array([300.0, 50.0, -999.0, -1000.0])) == 9.0
    assert cliff_volume(qty, np.full(4, -1000.0)) == 0.0


@pytest.mark.parametrize(
    ("delta", "qty", "price"),
    [
        (5.0, [2.0, 15.0, 35.0, 50.0], [500.0, 100.0, 0.0, -1000.0]),
        # The step pushed past the last edge goes away, the last step keeps its price.
        (25.0, [2.0, 35.0, 50.0], [500.0, 100.0, -1000.0]),
        # A step pushed onto the first edge gives its price to the first step.
        (-9.0, [2.0, 21.0, 50.0], [100.0, 0.0, -1000.0]),
        (-29.0, [2.0, 50.0], [0.0, -1000.0]),
    ],
)
def test_shift_keeps_first_and_last_edge(delta, qty, price):
    q = np.array([2.0, 10.0, 30.0, 50.0])
    p = np.array([500.0, 100.0, 0.0, -1000.0])
    sq, sp = shift_keep_first_edge(q, p, delta)
    np.testing.assert_array_equal(sq, qty)
    np.testing.assert_array_equal(sp, price)


def test_features_use_only_curves_dated_horizon_days_earlier(history):
    day = pd.to_datetime(history["Trading Date"])
    target = pd.Timestamp("2021-03-20")
    changed = history.copy()
    rng = np.random.default_rng(1)
    for i in np.flatnonzero((day > target - pd.Timedelta(days=7)).to_numpy()):
        qty, price = _demand_curve(rng)
        changed.at[i, "curve_qty"] = qty
        changed.at[i, "curve_price"] = price
    X, _, _, _ = volume_features(history, 7)
    X2, _, _, _ = volume_features(changed, 7)
    rows = (day == target).to_numpy()
    assert np.isfinite(X[rows]).all()
    np.testing.assert_array_equal(X[rows], X2[rows])
    next_day = (day == target + pd.Timedelta(days=1)).to_numpy()
    assert not np.array_equal(X[next_day], X2[next_day])


@pytest.mark.parametrize(("range_rule", "merge_equal"), [("median", True), ("median_minstart", False)])
def test_volume_shift_end_to_end(history, range_rule, merge_equal):
    day = pd.to_datetime(history["Trading Date"])
    split = day < pd.Timestamp("2021-03-01")
    train = history[split].reset_index(drop=True)
    test = history[~split].reset_index(drop=True)
    ctx = PredictContext(train=train, test=test, historical=pd.concat([train, test], ignore_index=True),
                         side="demand", lag_days=7)
    out = volume_shift(ctx, range_rule=range_rule, merge_equal=merge_equal, max_iter=5, min_samples_leaf=5)
    assert len(out) == len(test)
    assert out["v_hat"].notna().all()
    for q, p, v in zip(out["pred_curve_qty"], out["pred_curve_price"], out["v_hat"], strict=True):
        assert np.all(np.diff(q) > 0) and np.all(np.diff(p) <= 0)
        if q[0] < v < q[-1]:
            assert np.isclose(cliff_volume(q, p), v)


def test_volume_shift_checks_inputs(history):
    ctx = PredictContext(train=history, test=history, historical=history, side="supply", lag_days=7)
    with pytest.raises(ValueError, match="demand"):
        volume_shift(ctx)
    ctx.side = "demand"
    with pytest.raises(ValueError, match="horizon"):
        volume_shift(ctx, shape_lags=[6, 7])
