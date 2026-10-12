"""Simulated day-7 weather *forecast* source for the 7-day-ahead horizon.

At a 7-day horizon a deployable model cannot know the target day's actual
weather. It would instead have a *forecast* of that weather, which carries
error. This module simulates such a forecast honestly:

    forecast(d) = actual(d) + realistic day-7 forecast error

The error is per-variable calibrated Gaussian noise (see ``NOISE_SPEC``),
NOT a flat percentage - a flat 10% is physically wrong (10% of surface
pressure is ~100 hPa). The magnitudes come from ECMWF/WeatherReal day-7
verification: temperature, mean wind and pressure are forecast well; daily
precipitation and the daily-max wind gust barely at all.

Contrast with the two neighbouring sources in ``builder.py``:

    * ``weather``        - actual weather SHIFTED by the horizon (d-7 stale
      proxy). Leakage-safe but uses week-old weather.
    * ``weather_actual`` - actual same-day weather, no noise. A perfect but
      NON-deployable ceiling (oracle) on what weather could ever buy.
    * ``weather_fc``     - this module: actual same-day weather + day-7 noise.
      Honest and strictly more informative than the stale d-7 proxy.

Determinism is a hard requirement: the whole pipeline is cached and must
reproduce byte-for-byte across runs and processes. The noise is therefore
seeded per calendar DATE - ``rng = np.random.default_rng(base_seed +
d.toordinal())`` - so a date always gets the same perturbation and all 48
intervals of the day share it. Python's builtin ``hash()`` is deliberately
avoided (it is salted per-process). The whole perturbed table is precomputed
once at construction; ``features()`` is then a plain daily lookup, exactly
like ``ExternalCsvSource`` in daily mode. The value is NOT shifted - the
"7 days ahead" lives entirely in the day-7-sized noise.

Honesty caveat: the noise is i.i.d. across days and variables, which is an
*optimistic* simulation of a real 7-day forecast. Real forecast errors are
temporally correlated (one mis-timed front biases several consecutive days the
same way), cross-variable correlated, and damped toward climatology - none of
which this captures - so a model that leans on multi-day weather aggregates may
look slightly more accurate here than it would in deployment. Precipitation is
additionally clipped at 0, so positive draws on dry days add a small one-sided
wet bias. Treat the ``weather_fc`` lift as an upper-ish estimate of realistic
weather value, bracketed below by ``weather`` (stale d-7) and above by
``weather_actual`` (perfect same-day, the non-deployable ceiling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import load_external_daily_csv
from .provider import ExogenousSource

log = logging.getLogger(__name__)

# Repo convention (see runner.group.BOOTSTRAP_SEED); every reproducible draw
# in the codebase seeds from 42.
DEFAULT_BASE_SEED = 42


@dataclass(frozen=True)
class NoiseScheme:
    """Per-variable day-7 forecast-error model: ``value' = value + sigma * z``.

    ``z`` is a standard normal. ``sigma`` is either constant (``kind
    == "additive"``) or scales with the magnitude of the value with a floor
    (``kind == "hybrid"``: ``sigma = max(sigma_floor, sigma_frac * |value|)``).
    The result is then clipped to ``[clip_min, clip_max]`` (``None`` = no bound
    on that side). Units: temps degC, precip mm/day, wind km/h, radiation
    MJ/m^2/day, humidity %RH, pressure hPa.
    """

    kind: str  # "additive" or "hybrid"
    sigma_floor: float  # constant sigma (additive) or lower floor (hybrid)
    sigma_frac: float = 0.0  # fraction of |value| (hybrid only)
    clip_min: float | None = None
    clip_max: float | None = None

    def sigma(self, value: np.ndarray) -> np.ndarray:
        """Return the per-element noise scale for a column of values."""
        if self.kind == "additive":
            return np.full_like(np.asarray(value, dtype=float), self.sigma_floor)
        if self.kind == "hybrid":
            return np.maximum(self.sigma_floor, self.sigma_frac * np.abs(value))
        raise ValueError(f"Unknown NoiseScheme kind: {self.kind!r}")


# Per-variable day-7 forecast-error spec, keyed by the Perth column name so it
# is auditable and testable. Kept module-level on purpose. See the class
# docstring for the maths. Sources: ECMWF/WeatherReal day-7 verification.
#
#   column                          kind      sigma                         clip
#   per_temperature_2m_max          additive  3.0                           none
#   per_temperature_2m_min          additive  3.0                           none
#   per_temperature_2m_mean         additive  2.5                           none
#   per_apparent_temperature_max    additive  3.5                           none
#   per_precipitation_sum           hybrid    max(2.0, 0.70*|v|)            min 0
#   per_windspeed_10m_max           hybrid    max(5.0, 0.35*|v|)            min 0
#   per_windspeed_10m_mean          hybrid    max(3.0, 0.35*|v|)            min 0
#   per_shortwave_radiation_sum     hybrid    max(0.0, 0.25*|v|)            min 0
#   per_relative_humidity_2m_mean   additive  10.0                        [0,100]
#   per_surface_pressure_mean       additive  4.0                           none
NOISE_SPEC: dict[str, NoiseScheme] = {
    "per_temperature_2m_max": NoiseScheme("additive", 3.0),
    "per_temperature_2m_min": NoiseScheme("additive", 3.0),
    "per_temperature_2m_mean": NoiseScheme("additive", 2.5),
    "per_apparent_temperature_max": NoiseScheme("additive", 3.5),
    "per_precipitation_sum": NoiseScheme("hybrid", 2.0, 0.70, clip_min=0.0),
    "per_windspeed_10m_max": NoiseScheme("hybrid", 5.0, 0.35, clip_min=0.0),
    "per_windspeed_10m_mean": NoiseScheme("hybrid", 3.0, 0.35, clip_min=0.0),
    "per_shortwave_radiation_sum": NoiseScheme("hybrid", 0.0, 0.25, clip_min=0.0),
    "per_relative_humidity_2m_mean": NoiseScheme("additive", 10.0, clip_min=0.0, clip_max=100.0),
    "per_surface_pressure_mean": NoiseScheme("additive", 4.0),
}


def perturb_daily_table(
    actual: pd.DataFrame,
    base_seed: int = DEFAULT_BASE_SEED,
    spec: dict[str, NoiseScheme] = NOISE_SPEC,
) -> pd.DataFrame:
    """Return a copy of ``actual`` with per-date, per-column day-7 noise added.

    ``actual`` is an unshifted (``shift_days=0``) daily table indexed by
    calendar date; each row holds the *ground-truth* weather observed on that
    date. For each date ``d`` a fresh generator ``np.random.default_rng(
    base_seed + d.toordinal())`` draws one standard normal per column, in the
    table's (fixed) column order, and the column's ``NoiseScheme`` is applied.
    Every column must have a scheme in ``spec`` - an unspecified column would
    be a silently un-perturbed (i.e. leaky) variable, so that is an error.
    """
    cols = list(actual.columns)
    missing = [c for c in cols if c not in spec]
    if missing:
        raise ValueError(f"No NoiseScheme for columns: {missing}")

    values = actual.to_numpy(dtype=float)  # (n_dates, n_cols)
    # Draw the standard normals date-by-date so every date is independently but
    # reproducibly seeded (and every interval of a day shares its date's draw).
    z = np.empty_like(values)
    for i, d in enumerate(actual.index):
        rng = np.random.default_rng(base_seed + d.toordinal())
        z[i] = rng.standard_normal(len(cols))

    out = values.copy()
    for j, col in enumerate(cols):
        scheme = spec[col]
        perturbed = values[:, j] + scheme.sigma(values[:, j]) * z[:, j]
        if scheme.clip_min is not None:
            perturbed = np.maximum(perturbed, scheme.clip_min)
        if scheme.clip_max is not None:
            perturbed = np.minimum(perturbed, scheme.clip_max)
        out[:, j] = perturbed

    return pd.DataFrame(out, index=actual.index, columns=actual.columns)


@dataclass
class ForecastPerturbedDailySource(ExogenousSource):
    """A simulated day-7 weather forecast: actual same-day weather + noise.

    Loads a daily weather CSV unshifted (``shift_days=0``), then precomputes a
    deterministically perturbed copy (see ``perturb_daily_table``) once at
    construction so ``features()`` is a plain daily lookup. Missing dates
    return ``None`` (or ``fill_missing`` if set), matching ``ExternalCsvSource``.

    Parameters
    ----------
    path:
        Path to a daily-resolution weather CSV.
    feature_cols:
        Which columns to expose. If None, all non-index columns (each must
        appear in ``spec``).
    name_prefix:
        Prefix added to every feature name (e.g. ``'weather_fc_'``), so the
        merged feature_names stay interpretable.
    base_seed:
        Base seed for the per-date generators. Default 42 (repo convention).
    spec:
        Per-column ``NoiseScheme`` table. Defaults to module-level
        ``NOISE_SPEC``.
    fill_missing:
        If set to a float, missing lookups return a vector of that value
        instead of None.
    """

    path: str | Path
    feature_cols: list[str] | None = None
    name_prefix: str = "weather_fc_"
    base_seed: int = DEFAULT_BASE_SEED
    spec: dict[str, NoiseScheme] = field(default_factory=lambda: NOISE_SPEC)
    fill_missing: float | None = None
    date_col: str = "date"
    _table: pd.DataFrame = field(init=False, default=None)
    _feature_names: list[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.exists():
            log.warning("Weather CSV not found: %s - this source will return None.", path)
            self._table = None
            self._feature_names = []
            return

        actual = load_external_daily_csv(
            path,
            date_col=self.date_col,
            feature_cols=self.feature_cols,
            shift_days=0,  # use day d's actual weather; the noise carries the horizon
        )
        self._table = perturb_daily_table(actual, base_seed=self.base_seed, spec=self.spec)
        self._feature_names = [self.name_prefix + c for c in self._table.columns.tolist()]

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
            row = self._table.loc[date]
        except KeyError:
            if self.fill_missing is None:
                return None
            return np.full(self.dim, self.fill_missing, dtype=float)
        if isinstance(row, pd.DataFrame):
            # Duplicate keys in the source file
            row = row.iloc[0]
        return row.values.astype(float)
