"""Market-clearing evaluation: crossing operator, per-row attach, and summary.

Style mirrors ``tests/test_curves_and_metrics.py`` (small hand-built curves with
exactly known crossings).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aemo_curves.stages.evaluate import attach_clearing, clearing_summary
from aemo_curves.stages.evaluate.clearing import (
    _clearing_price,
    _clearing_volume,
    _true_curve_lookup,
)


@pytest.fixture(autouse=True)
def _clear_curve_lookup_cache():
    """``_true_curve_lookup`` is ``lru_cache``d (a per-run memory win), so without
    this it would carry one test's monkeypatched loader curves into the next.
    Clear it around every test so the fakes stay isolated."""
    _true_curve_lookup.cache_clear()
    yield
    _true_curve_lookup.cache_clear()


def test_clearing_volume_and_price_known_crossing():
    # Demand falls 100 -> 50 over (0,10],(10,20]; supply rises 30 -> 70 over
    # (0,8],(8,25]. Demand >= supply up to V=10, below after -> V* = 10.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 50.0])
    sq, sp = np.array([8.0, 25.0]), np.array([30.0, 70.0])
    v_star = _clearing_volume(dq, dp, sq, sp)
    assert v_star == 10.0
    # Clearing price is read off the TRUE SUPPLY at V*: supply(10) = 70.
    assert _clearing_price(sq, sp, v_star) == 70.0


def test_clearing_volume_no_trade_is_zero():
    # Supply opens above demand (R[0] < 0) -> no trade -> V* = 0.
    dq, dp = np.array([10.0, 20.0]), np.array([20.0, 10.0])
    sq, sp = np.array([10.0, 20.0]), np.array([50.0, 60.0])
    assert _clearing_volume(dq, dp, sq, sp) == 0.0


def test_clearing_volume_clears_at_cap():
    # Demand exceeds supply on the whole shared support -> V* = v_max.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 90.0])
    sq, sp = np.array([10.0, 30.0]), np.array([10.0, 20.0])
    assert _clearing_volume(dq, dp, sq, sp) == 20.0  # v_max = min(20, 30)


def test_residual_zero_crossing():
    # Residual (falling) crosses zero between its 2nd and 3rd edge -> V* = 20.
    rq = np.array([10.0, 20.0, 30.0])
    rp = np.array([50.0, 10.0, -20.0])
    assert _clearing_volume(rq, rp) == 20.0


def _supply_loader(rows):
    """A ``load_curve_table`` stand-in returning fixed supply rows for any split."""
    table = pd.DataFrame(rows)

    def _loader(name, side, *args, **kwargs):
        if side == "supply":
            return table
        return table.iloc[0:0]  # demand not needed on the demand side

    return _loader


def test_attach_clearing_perfect_prediction(monkeypatch):
    # One demand row whose prediction equals the truth -> zero clearing error.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 50.0])
    sq, sp = np.array([8.0, 25.0]), np.array([30.0, 70.0])
    scored = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-01")],
            "Interval Number": [1],
            "true_curve_qty": [dq], "true_curve_price": [dp],
            "pred_curve_qty": [dq], "pred_curve_price": [dp],
        }
    )
    loader = _supply_loader(
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": sq, "curve_price": sp}]
    )
    monkeypatch.setattr("aemo_curves.stages.curves.load_curve_table", loader)

    out = attach_clearing(scored, None, side="demand", split="canonical")
    assert out["clearing_vol_true"].iloc[0] == 10.0
    assert out["clearing_price_true"].iloc[0] == 70.0
    assert out["clearing_vol_ae"].iloc[0] == 0.0
    assert out["clearing_price_ae"].iloc[0] == 0.0


def test_attach_clearing_with_baseline(monkeypatch):
    # Demand side, perfect model prediction (V*=10) and a baseline demand curve
    # that clears earlier (V*=8, priced at supply(8)=30) -> its ae is read off
    # the same true-supply ruler. Guards the pred_curve_* column lookup.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 50.0])
    sq, sp = np.array([8.0, 25.0]), np.array([30.0, 70.0])
    bq, bp = np.array([10.0, 20.0]), np.array([60.0, 40.0])
    scored = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-01")],
            "Interval Number": [1],
            "true_curve_qty": [dq], "true_curve_price": [dp],
            "pred_curve_qty": [dq], "pred_curve_price": [dp],
        }
    )
    baseline_preds = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-01")],
            "Interval Number": [1],
            "pred_curve_qty": [bq], "pred_curve_price": [bp],
        }
    )
    loader = _supply_loader(
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": sq, "curve_price": sp}]
    )
    monkeypatch.setattr("aemo_curves.stages.curves.load_curve_table", loader)

    out = attach_clearing(scored, baseline_preds, side="demand", split="canonical")
    assert out["baseline_clearing_vol_pred"].iloc[0] == 8.0
    assert out["baseline_clearing_price_pred"].iloc[0] == 30.0
    assert out["baseline_clearing_vol_ae"].iloc[0] == 2.0  # |8 - 10|
    assert out["baseline_clearing_price_ae"].iloc[0] == 40.0  # |30 - 70|


def test_clearing_summary_keys_and_values():
    scored = pd.DataFrame(
        {
            "clearing_vol_true": [10.0, 20.0],
            "clearing_vol_pred": [12.0, 18.0],
            "clearing_price_true": [70.0, 60.0],
            "clearing_price_pred": [75.0, 63.0],
            "clearing_vol_ae": [2.0, 2.0],
            "clearing_price_ae": [5.0, 3.0],
            "baseline_clearing_vol_pred": [15.0, 26.0],
            "baseline_clearing_price_pred": [80.0, 64.0],
            "baseline_clearing_vol_ae": [5.0, 6.0],
            "baseline_clearing_price_ae": [10.0, 4.0],
        }
    )
    s = clearing_summary(scored)
    expected_keys = {
        "clearing_vol_mae", "clearing_vol_medae", "clearing_vol_bias",
        "clearing_price_mae", "clearing_price_medae", "clearing_price_bias",
        "baseline_clearing_vol_mae", "baseline_clearing_price_mae",
        "clearing_vol_delta_vs_baseline", "clearing_price_delta_vs_baseline",
        "clearing_vol_win_rate", "clearing_price_win_rate",
    }
    assert set(s) == expected_keys
    assert s["clearing_vol_mae"] == 2.0
    assert s["clearing_vol_medae"] == 2.0
    assert s["clearing_vol_bias"] == 0.0  # +2 and -2 cancel
    assert s["clearing_price_mae"] == 4.0
    assert s["clearing_price_medae"] == 4.0
    assert s["clearing_price_bias"] == 4.0
    assert s["baseline_clearing_vol_mae"] == 5.5
    assert s["baseline_clearing_price_mae"] == 7.0
    assert s["clearing_vol_delta_vs_baseline"] == 2.0 - 5.5
    assert s["clearing_price_delta_vs_baseline"] == 4.0 - 7.0
    assert s["clearing_vol_win_rate"] == 1.0
    assert s["clearing_price_win_rate"] == 1.0


def _side_loader(supply_rows, demand_rows):
    """A ``load_curve_table`` stand-in returning fixed supply and demand rows."""
    sup, dem = pd.DataFrame(supply_rows), pd.DataFrame(demand_rows)

    def _loader(name, side, *args, **kwargs):
        return sup if side == "supply" else dem

    return _loader


def test_attach_clearing_missing_key_raises(monkeypatch):
    # A prediction row whose (date, interval) has no true-supply curve is a hard
    # error, never a silent drop.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 50.0])
    scored = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-02")],  # absent from the loader
            "Interval Number": [1],
            "true_curve_qty": [dq], "true_curve_price": [dp],
            "pred_curve_qty": [dq], "pred_curve_price": [dp],
        }
    )
    loader = _supply_loader(
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": np.array([8.0, 25.0]), "curve_price": np.array([30.0, 70.0])}]
    )
    monkeypatch.setattr("aemo_curves.stages.curves.load_curve_table", loader)
    with pytest.raises(ValueError, match="no matching"):
        attach_clearing(scored, None, side="demand", split="canonical")


def test_attach_clearing_supply_side(monkeypatch):
    # Supply experiment: cross the true demand with the in-frame SUPPLY curve;
    # price is still read off the true-supply ruler. Perfect prediction -> 0 error.
    dq, dp = np.array([10.0, 20.0]), np.array([100.0, 50.0])   # true demand (partner)
    sq, sp = np.array([8.0, 25.0]), np.array([30.0, 70.0])     # true supply (in-frame + ruler)
    scored = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-01")],
            "Interval Number": [1],
            "true_curve_qty": [sq], "true_curve_price": [sp],
            "pred_curve_qty": [sq], "pred_curve_price": [sp],
        }
    )
    loader = _side_loader(
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": sq, "curve_price": sp}],
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": dq, "curve_price": dp}],
    )
    monkeypatch.setattr("aemo_curves.stages.curves.load_curve_table", loader)
    out = attach_clearing(scored, None, side="supply", split="canonical")
    assert out["clearing_vol_true"].iloc[0] == 10.0
    assert out["clearing_price_true"].iloc[0] == 70.0
    assert out["clearing_vol_ae"].iloc[0] == 0.0
    assert out["clearing_price_ae"].iloc[0] == 0.0


def test_attach_clearing_residual_side(monkeypatch):
    # Residual experiment: V* = the residual's own zero-crossing; price is read
    # off the true-supply ruler at V*.
    tq, tp = np.array([10.0, 20.0, 30.0]), np.array([50.0, 10.0, -20.0])  # true resid, V*=20
    pq, pp = np.array([10.0, 20.0, 30.0]), np.array([40.0, -5.0, -30.0])  # pred resid, V*=10
    sq, sp = np.array([15.0, 40.0]), np.array([25.0, 60.0])               # true-supply ruler
    scored = pd.DataFrame(
        {
            "Trading Date": [pd.Timestamp("2024-05-01")],
            "Interval Number": [1],
            "true_curve_qty": [tq], "true_curve_price": [tp],
            "pred_curve_qty": [pq], "pred_curve_price": [pp],
        }
    )
    loader = _supply_loader(
        [{"Trading Date": pd.Timestamp("2024-05-01"), "Interval Number": 1,
          "curve_qty": sq, "curve_price": sp}]
    )
    monkeypatch.setattr("aemo_curves.stages.curves.load_curve_table", loader)
    out = attach_clearing(scored, None, side="residual", split="canonical")
    assert out["clearing_vol_true"].iloc[0] == 20.0
    assert out["clearing_vol_pred"].iloc[0] == 10.0
    assert out["clearing_vol_ae"].iloc[0] == 10.0
    assert out["clearing_price_true"].iloc[0] == 60.0  # true supply at V*=20
    assert out["clearing_price_pred"].iloc[0] == 25.0  # true supply at V*=10
    assert out["clearing_price_ae"].iloc[0] == 35.0
