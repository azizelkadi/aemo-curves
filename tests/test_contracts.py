"""The catalog contracts accept a valid frame and reject malformed ones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aemo_curves.catalog import PREDICTIONS, SCORES
from aemo_curves.core.contracts import ContractError


def _valid_predictions(n: int = 3) -> pd.DataFrame:
    arr = np.array([0.0, 1.0, 2.0])
    return pd.DataFrame(
        {
            "Trading Date": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-02"][:n]),
            "Interval Number": [1, 2, 1][:n],
            "true_curve_qty": [arr] * n,
            "true_curve_price": [arr] * n,
            "pred_curve_qty": [arr] * n,
            "pred_curve_price": [arr] * n,
        }
    )


def test_valid_predictions_pass():
    df = _valid_predictions()
    assert PREDICTIONS.validate(df) is df


def test_missing_required_column_raises():
    df = _valid_predictions().drop(columns=["pred_curve_price"])
    with pytest.raises(ContractError):
        PREDICTIONS.validate(df)


def test_duplicate_keys_raise():
    df = _valid_predictions()
    df.loc[2, "Trading Date"] = df.loc[0, "Trading Date"]
    df.loc[2, "Interval Number"] = df.loc[0, "Interval Number"]
    with pytest.raises(ContractError):
        PREDICTIONS.validate(df)


def test_empty_frame_raises():
    with pytest.raises(ContractError):
        SCORES.validate(_valid_predictions(0))
