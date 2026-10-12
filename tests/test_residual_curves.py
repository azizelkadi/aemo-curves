"""Properties of the residual (demand-minus-supply price spread) curve tables.

The residual curve of one auction is ``R(v) = P_demand(v) - P_supply(v)`` on the
shared volume axis. It is a monotone non-increasing step curve (demand price
falls, supply price rises), and it goes negative past the market-clearing volume.

These are checks on the CACHED tables built by
``scripts/build_residual_curves.py``; they skip gracefully when the tables are
absent (mirroring ``test_smoke_runner``'s artifact gate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CURVES = REPO / "artifacts" / "curves"
_TOL = 1e-6

# Recon-verified row counts (inner join is 100% aligned, zero orphans).
EXPECTED_ROWS = {"train": 58_368, "test": 17_232}


def _tables_ready() -> bool:
    return (CURVES / "train_residual.pkl").exists() and (CURVES / "test_residual.pkl").exists()


pytestmark = pytest.mark.skipif(not _tables_ready(), reason="residual curve tables not built")


def _load(split: str):
    import pandas as pd

    return pd.read_pickle(CURVES / f"{split}_residual.pkl")


@pytest.mark.parametrize("split", ["train", "test"])
def test_row_counts(split):
    df = _load(split)
    assert len(df) == EXPECTED_ROWS[split]


@pytest.mark.parametrize("split", ["train", "test"])
def test_curve_price_non_increasing(split):
    """Residual R is a falling (demand-like) step curve on every auction."""
    df = _load(split)
    for p in df["curve_price"]:
        arr = np.asarray(p, dtype=float)
        assert np.all(np.diff(arr) <= _TOL), "residual price must be non-increasing"


@pytest.mark.parametrize("split", ["train", "test"])
def test_curve_qty_strictly_increasing(split):
    df = _load(split)
    for q in df["curve_qty"]:
        arr = np.asarray(q, dtype=float)
        assert len(arr) >= 2, "each residual curve needs at least two grid points"
        assert np.all(np.diff(arr) > 0.0), "curve_qty must be strictly increasing"


@pytest.mark.parametrize("split", ["train", "test"])
def test_negatives_present(split):
    """The spread goes negative past the clearing volume in at least some curves."""
    df = _load(split)
    any_negative = any((np.asarray(p, dtype=float) < 0).any() for p in df["curve_price"])
    assert any_negative, "expected negative residuals in at least some curves"


@pytest.mark.parametrize("split", ["train", "test"])
def test_schema(split):
    """Columns match the curves-table schema the downstream stages expect."""
    df = _load(split)
    expected = {
        "Trading Date",
        "Interval Number",
        "curve_qty",
        "curve_price",
        "n_offers",
        "total_quantity",
        "mean_price",
    }
    assert expected <= set(df.columns)
    # total_quantity is the shared volume cap = last grid point.
    row = df.iloc[0]
    assert abs(float(row["total_quantity"]) - float(np.asarray(row["curve_qty"])[-1])) < _TOL
