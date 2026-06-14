"""Smoke tests for curve building, embedding, and the Wasserstein metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.stages.curves import build_demand_curve, build_supply_curve
from aemo_curves.stages.features.interp import curve_to_embedding
from aemo_curves.stages.evaluate.wasserstein import curve_wasserstein_distance


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Price ($/MWh)": [10.0, 20.0, 30.0, 40.0],
            "Quantity (MWh)": [5.0, 5.0, 5.0, 5.0],
        }
    )


def _bids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Price ($/MWh)": [50.0, 40.0, 30.0, 20.0],
            "Quantity (MWh)": [5.0, 5.0, 5.0, 5.0],
        }
    )


def test_supply_curve_is_monotone_increasing_in_price():
    qty, price = build_supply_curve(_offers())
    assert np.all(np.diff(price) >= 0), "supply prices must be non-decreasing"
    assert qty[-1] == 20.0


def test_demand_curve_is_monotone_decreasing_in_price():
    qty, price = build_demand_curve(_bids())
    assert np.all(np.diff(price) <= 0), "demand prices must be non-increasing"
    assert qty[-1] == 20.0


def test_embedding_is_fixed_length_and_preserves_range():
    qty, price = build_supply_curve(_offers())
    emb = curve_to_embedding(qty, price)
    assert emb.shape == (64,)
    # Interpolation should never exceed the curve's observed range.
    assert emb.min() >= price.min() - 1e-9
    assert emb.max() <= price.max() + 1e-9


def test_wasserstein_is_zero_for_identical_curves():
    qty, price = build_supply_curve(_offers())
    assert curve_wasserstein_distance(qty, price, qty, price) == 0.0


def test_wasserstein_increases_when_curves_diverge():
    qty, price = build_supply_curve(_offers())
    other = price + 15.0
    d1 = curve_wasserstein_distance(qty, price, qty, other)
    d2 = curve_wasserstein_distance(qty, price, qty, other + 10.0)
    assert d2 > d1 > 0
