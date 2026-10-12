"""Tests for the weighted pointwise median of lagged curves (`lag_median`)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aemo_curves.catalog import PREDICTIONS
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.evaluate.hc_mae import _step_eval
from aemo_curves.stages.predict.lag_median import (
    cap_map_curve,
    lag_wmedian,
    weighted_median_curve,
)

# Three supply curves (prices do not fall with volume).
A = (np.array([10.0, 20.0, 30.0]), np.array([1.0, 5.0, 9.0]))
B = (np.array([15.0, 30.0]), np.array([2.0, 8.0]))
C = (np.array([5.0, 25.0, 40.0]), np.array([0.0, 4.0, 12.0]))


def test_median_range_rule_by_hand():
    # Range: median first edge 10, median last edge 30.
    qty, price = weighted_median_curve([A, B, C], [1, 1, 1], range_rule="median")
    np.testing.assert_array_equal(qty, [10.0, 15.0, 20.0, 25.0, 30.0])
    np.testing.assert_array_equal(price, [2.0, 4.0, 5.0, 8.0, 9.0])


def test_ref_range_rule_by_hand():
    qty, price = weighted_median_curve([A, B, C], [1, 1, 1], range_rule="ref", ref=1)
    np.testing.assert_array_equal(qty, [15.0, 20.0, 25.0, 30.0])
    np.testing.assert_array_equal(price, [4.0, 5.0, 8.0, 9.0])


def test_median_minstart_range_rule_by_hand():
    # Range: smallest first edge 5, median last edge 30.
    qty, price = weighted_median_curve([A, B, C], [1, 1, 1], range_rule="median_minstart")
    np.testing.assert_array_equal(qty, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    np.testing.assert_array_equal(price, [1.0, 2.0, 4.0, 5.0, 8.0, 9.0])


def test_curve_with_most_of_the_weight_is_returned():
    # C has weight 3 of 5, so the median is C at every volume, and the equal
    # steps merge back into C's own steps.
    qty, price = weighted_median_curve([A, B, C], [1, 1, 3], range_rule="ref", ref=2)
    np.testing.assert_array_equal(qty, C[0])
    np.testing.assert_array_equal(price, C[1])


def test_median_range_counts_each_curve_once():
    # With the weights counted, the median last edge would be 40.
    qty, _ = weighted_median_curve([A, B, C], [1, 1, 5], range_rule="median")
    assert (qty[0], qty[-1]) == (10.0, 30.0)


def _random_curve(rng, side: str):
    n = int(rng.integers(1, 30))
    qty = np.unique(np.round(np.cumsum(rng.uniform(0.5, 40.0, n)) + rng.uniform(0.0, 20.0), 1))
    price = np.sort(rng.choice(np.arange(-20.0, 300.0, 10.0), len(qty)))
    return qty, (price[::-1].copy() if side == "demand" else price)


@pytest.mark.parametrize("side", ["supply", "demand"])
@pytest.mark.parametrize("range_rule", ["median", "ref"])
def test_exact_weighted_median_and_monotone(side, range_rule):
    rng = np.random.default_rng(7)
    for _ in range(200):
        k = int(rng.integers(1, 11))
        curves = [_random_curve(rng, side) for _ in range(k)]
        weights = [int(w) for w in rng.integers(1, 4, k)]
        if sum(weights) % 2 == 0:
            weights[0] += 1
        ref = int(rng.integers(0, k))
        qty, price = weighted_median_curve(curves, weights, range_rule=range_rule, ref=ref)

        steps = np.diff(price)
        assert np.all(steps >= 0) if side == "supply" else np.all(steps <= 0)
        assert np.all(np.diff(qty) > 0)
        if range_rule == "ref":
            assert (qty[0], qty[-1]) == (curves[ref][0][0], curves[ref][0][-1])
        else:
            assert qty[0] == np.median([c[0][0] for c in curves])
            assert qty[-1] == np.median([c[0][-1] for c in curves])

        # At any volume in the range, the curve equals the plain median of the
        # candidate prices with each price repeated `weight` times.
        v = np.concatenate(([qty[0]], rng.uniform(qty[0], qty[-1], 50), qty))
        repeated = np.vstack([_step_eval(q, p, v) for (q, p), w in zip(curves, weights, strict=True) for _ in range(w)])
        np.testing.assert_array_equal(_step_eval(qty, price, v), np.median(repeated, axis=0))


def _plain_median_cut_at_median_last_edge(curves):
    # Plain median at every edge of every curve, kept below the median last
    # edge, which becomes the last edge.
    qty = np.unique(np.concatenate([q for q, _ in curves]))
    price = np.median(np.vstack([_step_eval(q, p, qty) for q, p in curves]), axis=0)
    hi = float(np.median([q[-1] for q, _ in curves]))
    last = min(int(np.searchsorted(qty, hi, side="left")), len(price) - 1)
    return np.append(qty[qty < hi], hi), np.append(price[qty < hi], price[last])


@pytest.mark.parametrize("side", ["supply", "demand"])
def test_median_minstart_without_merge(side):
    rng = np.random.default_rng(11)
    for _ in range(200):
        k = int(rng.integers(1, 15))
        curves = [_random_curve(rng, side) for _ in range(k)]
        qty, price = weighted_median_curve(curves, [1] * k, range_rule="median_minstart", merge_equal=False)
        want_qty, want_price = _plain_median_cut_at_median_last_edge(curves)
        np.testing.assert_array_equal(qty, want_qty)
        np.testing.assert_array_equal(price, want_price)
        assert qty[0] == min(q[0] for q, _ in curves)

        # The merge keeps the price at every volume and only removes edges.
        merged_qty, merged_price = weighted_median_curve(curves, [1] * k, range_rule="median_minstart")
        assert (merged_qty[0], merged_qty[-1]) == (qty[0], qty[-1])
        assert set(merged_qty) <= set(qty)
        v = np.concatenate((qty, rng.uniform(qty[0], qty[-1], 50)))
        np.testing.assert_array_equal(_step_eval(merged_qty, merged_price, v), _step_eval(qty, price, v))


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        weighted_median_curve([A, B], [1, 0])
    with pytest.raises(ValueError):
        weighted_median_curve([A, B], [1])
    with pytest.raises(ValueError):
        weighted_median_curve([A, B], [1, 2], range_rule="max")


def test_cap_map_curve():
    qty = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    price = np.array([10.0, 50.0, 299.4, 299.6, 300.0])
    q, p = cap_map_curve(qty, price, 300.0, 500.0)
    np.testing.assert_array_equal(q, qty)
    np.testing.assert_array_equal(p, [10.0, 50.0, 299.4, 500.0, 500.0])
    # When the cap falls, prices above the new cap are cut to it as well.
    _, p = cap_map_curve(qty, price, 300.0, 250.0)
    np.testing.assert_array_equal(p, [10.0, 50.0, 250.0, 250.0, 250.0])
    _, p = cap_map_curve(qty, price, 300.0, 300.0)
    assert p is price
    np.testing.assert_array_equal(price, [10.0, 50.0, 299.4, 299.6, 300.0])


def _context(side: str = "supply", drop=()):
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.date_range("2021-01-01", periods=45, freq="D").date:
        for interval in (1, 2):
            if (d, interval) in drop:
                continue
            qty = np.cumsum(rng.uniform(1.0, 3.0, 5))
            price = np.sort(rng.uniform(0.0, 100.0, 5))
            rows.append({"Trading Date": d, "Interval Number": interval,
                         "curve_qty": qty, "curve_price": price})
    df = pd.DataFrame(rows)
    cut = pd.Timestamp("2021-02-05").date()
    train = df[df["Trading Date"] < cut].reset_index(drop=True)
    test = df[df["Trading Date"] >= cut].reset_index(drop=True)
    hist = pd.concat([train, test], ignore_index=True)
    return PredictContext(train=train, test=test, historical=hist, side=side, lag_days=7)


LAGS = [[7, 2], [8, 1], [14, 1], [21, 1], [28, 1]]


def _lagged(hist: pd.DataFrame, day: date, interval: int, lag: int):
    src = hist[(hist["Trading Date"] == day - timedelta(days=lag)) & (hist["Interval Number"] == interval)]
    return (src["curve_qty"].iloc[0], src["curve_price"].iloc[0]) if len(src) else None


def test_strategy_matches_helper_and_drops_missing_lags():
    # (2021-02-01, 2) is the lag-7 curve of (2021-02-08, 2) and the lag-8 curve
    # of (2021-02-09, 2). Test days run from 2021-02-05 to 2021-02-14.
    ctx = _context(drop={(date(2021, 2, 1), 2)})
    preds = lag_wmedian(ctx, lags=LAGS, range_rule="median")
    PREDICTIONS.validate(preds)
    keys = set(zip(preds["Trading Date"], preds["Interval Number"], strict=True))
    assert len(preds) == len(ctx.test) - 1
    assert (date(2021, 2, 8), 2) not in keys
    assert (preds["n_lags_used"] == len(LAGS)).sum() == len(preds) - 1

    day = date(2021, 2, 9)
    row = preds[(preds["Trading Date"] == day) & (preds["Interval Number"] == 2)].iloc[0]
    assert row["n_lags_used"] == len(LAGS) - 1
    pairs = [(_lagged(ctx.historical, day, 2, lag), w) for lag, w in LAGS]
    curves = [c for c, _ in pairs if c is not None]
    weights = [w for c, w in pairs if c is not None]
    qty, price = weighted_median_curve(curves, weights, range_rule="median")
    np.testing.assert_array_equal(row["pred_curve_qty"], qty)
    np.testing.assert_array_equal(row["pred_curve_price"], price)


def test_strategy_ref_range_is_the_first_lag():
    ctx = _context()
    preds = lag_wmedian(ctx, lags=LAGS, range_rule="ref")
    assert len(preds) == len(ctx.test)
    for _, r in preds.iterrows():
        lag7_qty, _ = _lagged(ctx.historical, r["Trading Date"], r["Interval Number"], 7)
        assert (r["pred_curve_qty"][0], r["pred_curve_qty"][-1]) == (lag7_qty[0], lag7_qty[-1])


def test_strategy_rejects_bad_params():
    ctx = _context()
    with pytest.raises(ValueError, match="horizon"):
        lag_wmedian(ctx, lags=[[6, 1], [7, 1], [8, 1]])
    with pytest.raises(ValueError, match="supply"):
        lag_wmedian(_context(side="demand"), lags=LAGS, cap_map="always")
    with pytest.raises(ValueError):
        lag_wmedian(ctx, lags=LAGS, range_rule="outer")
