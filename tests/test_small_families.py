"""Tests for the consolidated small model families (lag_blend / shift_diagnostics / fpca_snap).

Verifies that the merged classes are exported, that the `name` attributes are
unchanged, that fit/predict emit the required schema, and that the divergent
FPCA extra-key contracts are preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aemo_curves.config import DEFAULT_QUANTITY_GRID
from aemo_curves.stages.predict.fpca_snap import FoFSnapPredictor, LogDerivFPCAPredictor
from aemo_curves.stages.predict.lag_blend import (
    BlendedLagPredictor,
    EMABaselinePredictor,
    LearnedBlendPredictor,
    SameWeekdayAveragePredictor,
)
from aemo_curves.stages.predict.shift_diagnostics import (
    GridSearchOracleShift,
    OracleShiftPredictor,
    RealizedShiftPredictor,
)

EMB = len(DEFAULT_QUANTITY_GRID)
REQUIRED = {
    "Trading Date",
    "Interval Number",
    "true_curve_qty",
    "true_curve_price",
    "pred_curve_qty",
    "pred_curve_price",
}


@pytest.fixture
def split():
    rng = np.random.default_rng(0)

    def mkrow(date, interval):
        n = 5
        qty = np.cumsum(rng.uniform(1, 3, size=n))
        price = np.sort(rng.uniform(0, 100, size=n))
        emb = np.sort(rng.uniform(0, 100, size=EMB))
        return {
            "Trading Date": date,
            "Interval Number": interval,
            "curve_qty": qty,
            "curve_price": price,
            "embedding": emb,
            "mean_price": float(price.mean()),
            "total_quantity": float(qty[-1]),
        }

    dates = pd.date_range("2021-01-01", periods=60, freq="D").date
    rows = [mkrow(d, itv) for d in dates for itv in (1, 2)]
    df = pd.DataFrame(rows)
    cut = pd.Timestamp("2021-02-20").date()
    train = df[df["Trading Date"] < cut].reset_index(drop=True)
    test = df[df["Trading Date"] >= cut].reset_index(drop=True)
    hist = pd.concat([train, test], ignore_index=True)
    return train, test, hist


def test_merged_family_classes_are_exported():
    from aemo_curves.stages.predict import (
        BlendedLagPredictor as B,
        EMABaselinePredictor as E,
        GridSearchOracleShift as G,
        LearnedBlendPredictor as L,
        OracleShiftPredictor as O,
        RealizedShiftPredictor as R,
        SameWeekdayAveragePredictor as S,
    )

    assert (B, L, E, S, O, R, G) == (
        BlendedLagPredictor, LearnedBlendPredictor, EMABaselinePredictor,
        SameWeekdayAveragePredictor, OracleShiftPredictor, RealizedShiftPredictor,
        GridSearchOracleShift,
    )


def test_name_attributes_unchanged():
    assert BlendedLagPredictor.name == "blended_lag_predictor"
    assert LearnedBlendPredictor.name == "learned_blend"
    assert SameWeekdayAveragePredictor.name == "same_weekday_avg"
    assert EMABaselinePredictor.name == "ema_baseline"
    assert OracleShiftPredictor.name == "oracle_shift"
    assert RealizedShiftPredictor.name == "realized_shift"
    assert GridSearchOracleShift.name == "grid_oracle_shift"
    assert FoFSnapPredictor.name == "fof_snap"
    assert LogDerivFPCAPredictor.name == "log_deriv_fpca"


@pytest.mark.parametrize(
    "model",
    [
        BlendedLagPredictor(lag_alt=7),
        LearnedBlendPredictor(lags=(1, 2, 7), min_samples=1),
        SameWeekdayAveragePredictor(lags=(7, 14)),
        EMABaselinePredictor(),
        OracleShiftPredictor(),
        RealizedShiftPredictor(),
        GridSearchOracleShift(grid=np.arange(-5.0, 5.5, 1.0)),
        FoFSnapPredictor(n_components=3, lags=(7, 14), min_samples=1),
        LogDerivFPCAPredictor(n_components=3, lags=(7, 14), min_samples=1, grid_size=16),
    ],
)
def test_fit_predict_emits_required_schema(model, split):
    train, test, hist = split
    model.fit(train)
    out = model.predict(test, hist)
    assert len(out) > 0
    assert REQUIRED.issubset(out.columns)
    model.params()


def test_fpca_divergent_extra_keys_preserved(split):
    train, test, hist = split
    fof = FoFSnapPredictor(n_components=3, lags=(7, 14), min_samples=1).fit(train).predict(test, hist)
    assert "embedding_retrieval_distance" in fof.columns
    assert "pca_score_distance" in fof.columns

    ld = (
        LogDerivFPCAPredictor(n_components=3, lags=(7, 14), min_samples=1, grid_size=16)
        .fit(train)
        .predict(test, hist)
    )
    assert "score_distance" in ld.columns
    assert "pca_score_distance" not in ld.columns
    assert "embedding_retrieval_distance" not in ld.columns
