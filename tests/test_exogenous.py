"""Tests for the exogenous-features subsystem."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aemo_curves.exogenous import (
    ExogenousProvider,
    ExternalCsvSource,
    MarketStateProxySource,
    build_day_state_table,
    build_shifted_lookup,
    is_wa_holiday,
    wa_holiday_flags,
)
from aemo_curves.exogenous.proxies import DAY_STATE_COLUMNS


def _toy_embeddings(n_days: int = 40, n_intervals: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = dt.date(2024, 1, 1)
    for d in range(n_days):
        date = start + dt.timedelta(days=d)
        for i in range(1, n_intervals + 1):
            rows.append(
                {
                    "Trading Date": date,
                    "Interval Number": i,
                    "curve_qty": np.array([10.0, 50.0, 100.0]),
                    "curve_price": np.array([10.0 + rng.normal(), 30.0, 80.0]),
                    "n_offers": 12,
                    "total_quantity": 100.0,
                    "mean_price": 40.0 + rng.normal(),
                    "embedding": rng.normal(loc=40.0, scale=10.0, size=64),
                }
            )
    return pd.DataFrame(rows)


def test_wa_holiday_known_dates():
    assert is_wa_holiday(dt.date(2024, 1, 1)) is True
    assert is_wa_holiday(dt.date(2024, 4, 25)) is True
    assert is_wa_holiday(dt.date(2024, 3, 4)) is True  # WA Labour Day 2024
    assert is_wa_holiday(dt.date(2024, 5, 15)) is False


def test_wa_holiday_flags_shape_and_semantics():
    flags = wa_holiday_flags(dt.date(2024, 1, 1))
    assert flags.shape == (5,)
    assert flags[0] == 1.0  # is_holiday
    assert wa_holiday_flags(dt.date(2024, 5, 18))[3] == 1.0  # saturday => weekend


def test_day_state_table_structure():
    df = _toy_embeddings()
    tbl = build_day_state_table(df)
    assert list(tbl.columns) == DAY_STATE_COLUMNS
    assert len(tbl) == df["Trading Date"].nunique()


def test_shifted_lookup_respects_lag():
    df = _toy_embeddings()
    tbl = build_day_state_table(df)
    lookup = build_shifted_lookup(tbl, lag_days=1)
    # The value for target date d comes from d-1.
    first_day = min(tbl.index)
    # target = first_day + 1 day should match the first row of tbl
    target = first_day + dt.timedelta(days=1)
    np.testing.assert_allclose(lookup[target], tbl.loc[first_day].values)
    # target = first_day itself should be absent (no data for first_day - 1)
    assert first_day not in lookup


def test_proxy_source_returns_features_or_none():
    df = _toy_embeddings(n_days=10)
    src = MarketStateProxySource(embeddings=df, lag_days=1, include_calendar=True)
    # Day 0 has no yesterday → None
    first_day = min(df["Trading Date"])
    assert src.features(interval=1, trading_date=first_day) is None
    # Day 2 has yesterday available → non-None of correct length
    day2 = first_day + dt.timedelta(days=2)
    v = src.features(interval=1, trading_date=day2)
    assert v is not None
    assert v.shape == (src.dim,)


def test_provider_returns_none_if_any_source_is_missing(tmp_path):
    df = _toy_embeddings(n_days=10)
    proxy = MarketStateProxySource(embeddings=df, lag_days=1)
    # Write a tiny CSV that doesn't cover the requested date → None
    path = tmp_path / "tiny.csv"
    path.write_text("date,flag\n2020-12-31,1\n")
    csv_src = ExternalCsvSource(path=path, mode="daily", shift_days=1)
    provider = ExogenousProvider([proxy, csv_src])
    # Proxy needs at least 1 day of history; pick a date in toy range
    target = min(df["Trading Date"]) + dt.timedelta(days=2)
    v = provider(interval=1, trading_date=target)
    # proxy works, but csv_src can't cover target -> None
    assert v is None


def test_provider_concatenates_when_all_sources_have_data(tmp_path):
    df = _toy_embeddings(n_days=10)
    first_day = min(df["Trading Date"])
    # Make a CSV that covers the shifted dates (shift=1 → shifts to first_day+1)
    csv_path = tmp_path / "weather.csv"
    rows = ["date,t_max"]
    for d in range(10):
        rows.append(f"{first_day + dt.timedelta(days=d)},{20 + d}")
    csv_path.write_text("\n".join(rows))
    proxy = MarketStateProxySource(embeddings=df, lag_days=1)
    csv_src = ExternalCsvSource(path=csv_path, mode="daily", shift_days=1, name_prefix="w_")
    provider = ExogenousProvider([proxy, csv_src])
    target = first_day + dt.timedelta(days=3)
    v = provider(interval=1, trading_date=target)
    assert v is not None
    assert v.shape == (proxy.dim + csv_src.dim,)


def test_missing_csv_file_does_not_crash_loader():
    src = ExternalCsvSource(path="/tmp/this_definitely_does_not_exist.csv", mode="daily")
    assert src.dim == 0
    assert src.features(1, dt.date(2024, 1, 1)) is None
