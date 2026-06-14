"""Proxy exogenous features derived from the STEM dataset itself.

These features act as stand-ins for true external data (weather, demand
forecasts) and are always available as long as cached embeddings exist.
The design principle is **no leakage**: features for `(interval, date)`
are built from data strictly before `date`.

The main class is `MarketStateProxySource`, which:
  1. On construction, builds a per-date table of daily market-state
     aggregates from the full embedding history (train + test).
  2. Exposes `.features(interval, trading_date) -> np.ndarray`. Internally
     the date is shifted by `lag_days` (default 1) before lookup.

Features currently encoded (per shifted day):
  - daily_mean_price           - mean across 48 intervals' mean prices
  - daily_median_price
  - daily_price_std            - volatility across the day
  - daily_total_qty_log        - log of total MWh cleared
  - daily_n_offers             - total offer rows in the day
  - daily_mean_emb_l2          - mean L2 norm of 64-d embeddings
  - daily_mean_emb_steepness   - mean of (emb.max - emb.min)
  - trailing_7d_mean_price     - 7-day moving average
  - trailing_28d_mean_price    - 28-day moving average
  - price_trend_7d             - today - 7 days ago
  - price_trend_28d            - today - 28 days ago
  - price_zscore_vs_28d        - (today - MA28) / STD28
  - qty_log_trend_7d           - log-ratio of total qty vs 7-day mean

Plus calendar features (always available, no dataset required):
  - sin/cos of day-of-year
  - is_holiday, is_day_before_holiday, is_day_after_holiday,
    is_weekend, is_bridge_day  (see `holidays.py`)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .holidays import wa_holiday_flag_names, wa_holiday_flags
from .provider import ExogenousSource


def _safe_log(x: float, floor: float = 1e-6) -> float:
    return float(np.log(max(float(x), floor)))


def build_day_state_table(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 48 intervals into per-day market-state features.

    Output columns match the constants `DAY_STATE_COLUMNS`. Indexed by
    Trading Date.
    """
    df = embeddings.copy()
    if "embedding" in df.columns:
        embs = np.vstack(df["embedding"].to_numpy())
        df = df.assign(
            _emb_l2=np.linalg.norm(embs, axis=1),
            _emb_steepness=embs.max(axis=1) - embs.min(axis=1),
        )
    else:
        df["_emb_l2"] = 0.0
        df["_emb_steepness"] = 0.0

    grp = df.groupby("Trading Date")
    daily = grp.agg(
        daily_mean_price=("mean_price", "mean"),
        daily_median_price=("mean_price", "median"),
        daily_price_std=("mean_price", "std"),
        daily_total_qty=("total_quantity", "sum"),
        daily_n_offers=("n_offers", "sum"),
        daily_mean_emb_l2=("_emb_l2", "mean"),
        daily_mean_emb_steepness=("_emb_steepness", "mean"),
    )
    daily["daily_total_qty_log"] = daily["daily_total_qty"].apply(_safe_log)
    daily = daily.fillna(0.0)

    # Rolling statistics computed with min_periods=1; leakage control is
    # done later by shifting the whole table by `lag_days` before lookup.
    ma7 = daily["daily_mean_price"].rolling(7, min_periods=1).mean()
    ma28 = daily["daily_mean_price"].rolling(28, min_periods=1).mean()
    std28 = daily["daily_mean_price"].rolling(28, min_periods=1).std().fillna(1.0)
    qty7 = daily["daily_total_qty"].rolling(7, min_periods=1).mean()

    daily["trailing_7d_mean_price"] = ma7
    daily["trailing_28d_mean_price"] = ma28
    daily["price_trend_7d"] = daily["daily_mean_price"] - ma7
    daily["price_trend_28d"] = daily["daily_mean_price"] - ma28
    daily["price_zscore_vs_28d"] = (daily["daily_mean_price"] - ma28) / np.maximum(std28, 1e-6)
    daily["qty_log_trend_7d"] = (
        daily["daily_total_qty_log"] - qty7.apply(_safe_log)
    )

    return daily[DAY_STATE_COLUMNS]


DAY_STATE_COLUMNS: list[str] = [
    "daily_mean_price",
    "daily_median_price",
    "daily_price_std",
    "daily_total_qty_log",
    "daily_n_offers",
    "daily_mean_emb_l2",
    "daily_mean_emb_steepness",
    "trailing_7d_mean_price",
    "trailing_28d_mean_price",
    "price_trend_7d",
    "price_trend_28d",
    "price_zscore_vs_28d",
    "qty_log_trend_7d",
]


def build_shifted_lookup(day_state: pd.DataFrame, lag_days: int = 1) -> dict[dt.date, np.ndarray]:
    """Return {date: feature_vector} for the day that is `lag_days` earlier.

    Given a day-state table indexed by calendar date and a `lag_days`, this
    builds a dict keyed by **target date** (the date to predict) whose value
    is the feature row from `target_date - lag_days`. Missing entries are
    omitted.
    """
    lookup: dict[dt.date, np.ndarray] = {}
    for d, row in day_state.iterrows():
        target = pd.Timestamp(d).date() + dt.timedelta(days=lag_days)
        lookup[target] = np.asarray(row.values, dtype=float)
    return lookup


@dataclass
class MarketStateProxySource(ExogenousSource):
    """Market-state proxy source.

    Parameters
    ----------
    embeddings:
        The full historical embedding table (train + test). Used only to
        build the per-day aggregate table; no per-row leakage because
        lookup is shifted by `lag_days`.
    lag_days:
        Features for prediction date `d` are pulled from day `d - lag_days`.
        Default is 1 (yesterday).
    include_calendar:
        If True, appends sin/cos of day-of-year and WA holiday flags. These
        are available for any date, including test dates.
    """

    embeddings: pd.DataFrame
    lag_days: int = 1
    include_calendar: bool = True

    def __post_init__(self) -> None:
        day_state = build_day_state_table(self.embeddings)
        self._lookup = build_shifted_lookup(day_state, lag_days=self.lag_days)
        self._n_state = len(DAY_STATE_COLUMNS)
        self._cal_names = ["sin_doy", "cos_doy", *wa_holiday_flag_names()]

    @property
    def feature_names(self) -> list[str]:
        names = list(DAY_STATE_COLUMNS)
        if self.include_calendar:
            names.extend(self._cal_names)
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def features(self, interval: int, trading_date) -> np.ndarray | None:
        date = pd.Timestamp(trading_date).date()
        state = self._lookup.get(date)
        if state is None:
            # no history available for this date; caller will drop the row
            return None
        if not self.include_calendar:
            return state
        doy = date.timetuple().tm_yday
        cal = np.concatenate(
            [
                np.array([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]),
                wa_holiday_flags(date),
            ]
        )
        return np.concatenate([state, cal], axis=0)
