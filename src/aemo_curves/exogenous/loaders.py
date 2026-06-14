"""CSV loaders for real external data (weather, demand forecast, fuel).

Two tabular formats are supported:

1. **Daily**: one row per date.

    ```
    date,feature_1,feature_2,...
    2021-01-01,32.1,4.5,...
    ```

2. **Interval**: one row per (date, interval) pair.

    ```
    Trading Date,Interval Number,feature_1,feature_2,...
    2021-01-01,1,1890.3,...
    ```

An `ExternalCsvSource` wraps either format and exposes the standard
`ExogenousSource` interface. Missing dates / intervals return `None`, so
training/prediction rows without external data are skipped rather than
faked with zeros. That's the safe default; override with `fill_missing=0.0`
if you want zero-fill.

All external files live under `data/external/`. See
`data/external/README.md` for the expected schemas and sources.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provider import ExogenousSource

log = logging.getLogger(__name__)


def load_external_daily_csv(
    path: str | Path,
    date_col: str = "date",
    feature_cols: list[str] | None = None,
    shift_days: int = 1,
) -> pd.DataFrame:
    """Load a per-day external feature table.

    The dataframe is shifted by `shift_days` so that the row indexed by
    date `d` holds the features needed to predict date `d`. This default
    (shift_days=1) yields yesterday's features for today's prediction.
    Set `shift_days=0` if the source already represents a *forecast* of
    that date (e.g., a day-ahead demand forecast published yesterday).
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df = df.set_index(date_col).sort_index()
    if feature_cols is not None:
        df = df[feature_cols]
    if shift_days != 0:
        df.index = pd.to_datetime(df.index) + pd.Timedelta(days=shift_days)
        df.index = df.index.date
    return df


def load_external_interval_csv(
    path: str | Path,
    date_col: str = "Trading Date",
    interval_col: str = "Interval Number",
    feature_cols: list[str] | None = None,
    shift_days: int = 0,
) -> pd.DataFrame:
    """Load a per-(date, interval) external feature table.

    Default shift is 0 because interval-resolution forecasts (AEMO
    day-ahead demand forecast) are typically already aligned with the
    target interval.
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[interval_col] = df[interval_col].astype(int)
    df = df.set_index([date_col, interval_col]).sort_index()
    if feature_cols is not None:
        df = df[feature_cols]
    if shift_days != 0:
        # Shift both levels of the MultiIndex.
        new_idx = [
            (pd.Timestamp(d) + pd.Timedelta(days=shift_days)).date() for d, _ in df.index
        ]
        df.index = pd.MultiIndex.from_arrays(
            [new_idx, df.index.get_level_values(interval_col)],
            names=df.index.names,
        )
    return df


@dataclass
class ExternalCsvSource(ExogenousSource):
    """Per-auction features loaded from an external CSV.

    Parameters
    ----------
    path:
        Path to a CSV file. Can be either daily- or interval-resolution.
    mode:
        'daily' (one row per date) or 'interval' (one row per date+interval).
    feature_cols:
        Which columns to expose as features. If None, all non-index columns.
    shift_days:
        How many days to shift the source forward before lookup. For daily
        files this defaults to 1 (yesterday's features used today). For
        interval files (assumed day-ahead forecasts) it defaults to 0.
    fill_missing:
        If set to a float, missing lookups return a vector of that value
        instead of None. Useful for dense training.
    name_prefix:
        Prefix added to every feature name (e.g., 'weather_'), so the
        merged feature_names are interpretable.
    """

    path: str | Path
    mode: str = "daily"
    feature_cols: list[str] | None = None
    shift_days: int | None = None
    fill_missing: float | None = None
    name_prefix: str = ""
    date_col: str = "date"
    interval_col: str = "Interval Number"
    _table: pd.DataFrame = field(init=False, default=None)
    _feature_names: list[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.exists():
            log.warning("External CSV not found: %s - this source will return None.", path)
            self._table = None
            self._feature_names = []
            return

        if self.mode == "daily":
            shift = 1 if self.shift_days is None else self.shift_days
            tbl = load_external_daily_csv(
                path,
                date_col=self.date_col,
                feature_cols=self.feature_cols,
                shift_days=shift,
            )
        elif self.mode == "interval":
            shift = 0 if self.shift_days is None else self.shift_days
            tbl = load_external_interval_csv(
                path,
                date_col="Trading Date",
                interval_col=self.interval_col,
                feature_cols=self.feature_cols,
                shift_days=shift,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._table = tbl
        self._feature_names = [self.name_prefix + c for c in tbl.columns.tolist()]

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def dim(self) -> int:
        return len(self._feature_names)

    def features(self, interval: int, trading_date) -> np.ndarray | None:
        if self._table is None or self._table.empty:
            if self.fill_missing is None:
                return None
            return np.full(self.dim, self.fill_missing, dtype=float)
        date = pd.Timestamp(trading_date).date()
        try:
            if self.mode == "daily":
                row = self._table.loc[date]
            else:
                row = self._table.loc[(date, int(interval))]
        except KeyError:
            if self.fill_missing is None:
                return None
            return np.full(self.dim, self.fill_missing, dtype=float)
        if isinstance(row, pd.DataFrame):
            # Duplicate keys in the source file
            row = row.iloc[0]
        return row.values.astype(float)
