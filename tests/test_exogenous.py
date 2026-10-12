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


# --- Real weather source (task 0.4): leakage-safe Perth weather -----------

def _perth_weather_source(shift_days: int) -> ExternalCsvSource:
    """The weather source the builder emits, at a given horizon shift."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER

    return ExternalCsvSource(
        path=_WEATHER,
        mode="daily",
        shift_days=shift_days,
        feature_cols=list(_PERTH_WEATHER_COLS),
        name_prefix="weather_",
    )


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_weather_source_shift_matches_horizon_no_leakage():
    """At the 7-day horizon the target date d must read weather from d - 7.

    Guards against the old bug where the weather source hardcoded shift_days=1,
    which at a 7-day horizon would leak data from inside the forecast gap.
    """
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER
    from aemo_curves.exogenous.loaders import load_external_daily_csv

    lag_days = 7
    src = _perth_weather_source(shift_days=lag_days)
    # Unshifted table: the ground-truth value observed *on* each calendar date.
    raw = load_external_daily_csv(
        _WEATHER, feature_cols=list(_PERTH_WEATHER_COLS), shift_days=0
    )
    # Pick a target date well inside the table so d - 7 exists.
    target = raw.index[100]
    source_date = target - dt.timedelta(days=lag_days)
    emitted = src.features(interval=1, trading_date=target)
    assert emitted is not None
    # The features served for `target` are exactly those observed on d - lag_days.
    np.testing.assert_allclose(emitted, raw.loc[source_date].values)
    # And they are NOT yesterday's (d - 1) values — the leaky behaviour.
    leaky_date = target - dt.timedelta(days=1)
    assert not np.allclose(emitted, raw.loc[leaky_date].values)


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_builder_weather_emits_perth_subset_only():
    """The builder's weather source exposes exactly the Perth column subset."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, build_exog_provider

    provider = build_exog_provider({"sources": ["weather"]}, side="demand", lag_days=7)
    assert provider is not None
    names = provider.feature_names
    # Every Perth column is present, prefixed, and nothing else leaks in.
    assert names == [f"weather_{c}" for c in _PERTH_WEATHER_COLS]
    assert len(names) == len(_PERTH_WEATHER_COLS)
    # All columns are Perth ("per_") columns — no other capital city.
    assert all(n.startswith("weather_per_") for n in names)


def test_builder_weather_missing_file_is_skipped(monkeypatch):
    """If the real weather file is absent, the weather source is skipped."""
    import aemo_curves.exogenous.builder as builder

    monkeypatch.setattr(
        builder, "_WEATHER", builder._EXTERNAL / "does_not_exist_weather.csv"
    )
    # Weather-only spec with a missing file -> no sources -> provider is None.
    provider = builder.build_exog_provider(
        {"sources": ["weather"]}, side="demand", lag_days=7
    )
    assert provider is None


# --- Simulated day-7 weather forecast source ------------------------------
#
# `weather_fc` = actual same-day weather + realistic seeded day-7 noise (honest,
# deployable). `weather_actual` = actual same-day weather, no noise (oracle).


def _forecast_weather_source(base_seed: int = 42):
    """The `weather_fc` source the builder emits, at a given base seed."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER
    from aemo_curves.exogenous.forecast_weather import ForecastPerturbedDailySource

    return ForecastPerturbedDailySource(
        path=_WEATHER,
        feature_cols=list(_PERTH_WEATHER_COLS),
        name_prefix="weather_fc_",
        base_seed=base_seed,
    )


def _synthetic_weather_table(n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """A daily table with the 10 Perth columns, with edge values that force clips.

    Precipitation and radiation are 0 (any negative noise must clip to 0) and
    humidity is 100 (any positive noise must clip to 100), so the clipping
    bounds are genuinely exercised rather than merely never violated.
    """
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS

    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=d) for d in range(n_days)]
    data = {
        "per_temperature_2m_max": rng.uniform(15, 40, n_days),
        "per_temperature_2m_min": rng.uniform(5, 25, n_days),
        "per_temperature_2m_mean": rng.uniform(10, 32, n_days),
        "per_apparent_temperature_max": rng.uniform(15, 42, n_days),
        "per_precipitation_sum": np.zeros(n_days),
        "per_windspeed_10m_max": rng.uniform(0, 3, n_days),
        "per_windspeed_10m_mean": rng.uniform(0, 2, n_days),
        "per_shortwave_radiation_sum": np.zeros(n_days),
        "per_relative_humidity_2m_mean": np.full(n_days, 100.0),
        "per_surface_pressure_mean": rng.uniform(1000, 1025, n_days),
    }
    return pd.DataFrame({c: data[c] for c in _PERTH_WEATHER_COLS}, index=dates)


def test_noise_spec_covers_every_perth_column():
    """The auditable NOISE_SPEC has a scheme for exactly the 10 Perth columns."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS
    from aemo_curves.exogenous.forecast_weather import NOISE_SPEC

    assert set(NOISE_SPEC) == set(_PERTH_WEATHER_COLS)


def test_forecast_weather_clipping_bounds():
    """Precip/radiation never go negative; humidity stays within [0, 100].

    Uses a synthetic table (no CSV needed) whose seed values sit exactly on the
    bounds so the clips must fire.
    """
    from aemo_curves.exogenous.forecast_weather import perturb_daily_table

    actual = _synthetic_weather_table(n_days=120)
    pert = perturb_daily_table(actual, base_seed=7)

    assert (pert["per_precipitation_sum"] >= 0).all()
    assert (pert["per_windspeed_10m_max"] >= 0).all()
    assert (pert["per_windspeed_10m_mean"] >= 0).all()
    assert (pert["per_shortwave_radiation_sum"] >= 0).all()
    humid = pert["per_relative_humidity_2m_mean"]
    assert (humid >= 0).all()
    assert (humid <= 100).all()
    # The clips genuinely bit: precip started at 0 and humidity at 100.
    assert (pert["per_precipitation_sum"] == 0).any()
    assert (pert["per_relative_humidity_2m_mean"] == 100).any()


def test_perturb_daily_table_requires_spec_for_every_column():
    """A column with no NoiseScheme is a silently-unperturbed (leaky) bug."""
    from aemo_curves.exogenous.forecast_weather import perturb_daily_table

    bad = pd.DataFrame(
        {"per_temperature_2m_max": [20.0], "not_in_spec": [1.0]},
        index=[dt.date(2024, 1, 1)],
    )
    with pytest.raises(ValueError, match="not_in_spec"):
        perturb_daily_table(bad)


def test_perturb_daily_table_deterministic_across_constructions():
    """Same seed -> byte-identical table; different seed -> different values."""
    from aemo_curves.exogenous.forecast_weather import perturb_daily_table

    actual = _synthetic_weather_table(n_days=30)
    a = perturb_daily_table(actual, base_seed=42)
    b = perturb_daily_table(actual, base_seed=42)
    assert np.array_equal(a.to_numpy(), b.to_numpy())  # byte-identical
    c = perturb_daily_table(actual, base_seed=99)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_forecast_weather_source_deterministic_and_seed_sensitive():
    """Two sources from the same CSV+seed agree byte-for-byte; a new seed differs.

    Also checks that all 48 intervals of a day share the one daily perturbation.
    """
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER
    from aemo_curves.exogenous.loaders import load_external_daily_csv

    raw = load_external_daily_csv(
        _WEATHER, feature_cols=list(_PERTH_WEATHER_COLS), shift_days=0
    )
    target = raw.index[100]

    a = _forecast_weather_source(base_seed=42)
    b = _forecast_weather_source(base_seed=42)
    va = a.features(interval=1, trading_date=target)
    vb = b.features(interval=1, trading_date=target)
    assert va is not None
    assert np.array_equal(va, vb)  # byte-identical across constructions

    # Every interval of the day shares the same date-seeded perturbation.
    assert np.array_equal(a.features(interval=48, trading_date=target), va)

    # A different base seed yields a different perturbation.
    c = _forecast_weather_source(base_seed=123)
    assert not np.allclose(c.features(interval=1, trading_date=target), va)


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_forecast_weather_differs_from_actual_and_from_stale_d7():
    """The forecast perturbs the actual same-day value and is not the stale d-7."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER
    from aemo_curves.exogenous.loaders import load_external_daily_csv

    raw = load_external_daily_csv(
        _WEATHER, feature_cols=list(_PERTH_WEATHER_COLS), shift_days=0
    )
    target = raw.index[100]
    fc = _forecast_weather_source(base_seed=42).features(interval=1, trading_date=target)

    # Perturbed: not equal to the actual same-day weather it was built from...
    assert not np.allclose(fc, raw.loc[target].values)
    # ...and not equal to the stale d-7 weather the plain "weather" source serves.
    stale = raw.loc[target - dt.timedelta(days=7)].values
    assert not np.allclose(fc, stale)


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_builder_weather_fc_provider_block():
    """proxy + weather_fc exposes the 10 prefixed Perth columns as its tail block."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, build_exog_provider

    provider = build_exog_provider(
        {"sources": ["proxy", "weather_fc"]}, side="demand", lag_days=7
    )
    assert provider is not None
    names = provider.feature_names
    expected = [f"weather_fc_{c}" for c in _PERTH_WEATHER_COLS]
    assert names[-len(expected):] == expected
    fc_names = [n for n in names if n.startswith("weather_fc_")]
    assert len(fc_names) == len(_PERTH_WEATHER_COLS)
    assert all(n.startswith("weather_fc_per_") for n in fc_names)
    assert provider.dim == len(names)


@pytest.mark.skipif(
    not (__import__("aemo_curves.exogenous.builder", fromlist=["_WEATHER"])._WEATHER).exists(),
    reason="real weather CSV not present",
)
def test_builder_weather_actual_provider_block():
    """proxy + weather_actual exposes the perfect same-day Perth columns (no noise)."""
    from aemo_curves.exogenous.builder import _PERTH_WEATHER_COLS, _WEATHER, build_exog_provider
    from aemo_curves.exogenous.loaders import load_external_daily_csv

    provider = build_exog_provider(
        {"sources": ["proxy", "weather_actual"]}, side="demand", lag_days=7
    )
    assert provider is not None
    names = provider.feature_names
    expected = [f"weather_actual_{c}" for c in _PERTH_WEATHER_COLS]
    assert names[-len(expected):] == expected
    actual_names = [n for n in names if n.startswith("weather_actual_")]
    assert len(actual_names) == len(_PERTH_WEATHER_COLS)

    # The ceiling source serves the actual same-day (unshifted) weather verbatim.
    raw = load_external_daily_csv(
        _WEATHER, feature_cols=list(_PERTH_WEATHER_COLS), shift_days=0
    )
    target = raw.index[100]
    weather_block = provider.sources[-1].features(interval=1, trading_date=target)
    np.testing.assert_allclose(weather_block, raw.loc[target].values)
