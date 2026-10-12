"""Build an ExogenousProvider from an experiment's small `exog` spec.

The spec lists which sources to combine, for example
`{"sources": ["proxy"]}` or `{"sources": ["proxy", "weather"]}`. The proxy
source is derived from the cached curves; weather and demand are optional CSVs
under `data/external/` and are skipped when the file is missing.
"""
from __future__ import annotations

import pandas as pd

from aemo_curves.config import PROJECT_ROOT
from aemo_curves.exogenous.forecast_weather import ForecastPerturbedDailySource
from aemo_curves.exogenous.loaders import ExternalCsvSource
from aemo_curves.exogenous.provider import ExogenousProvider
from aemo_curves.exogenous.proxies import MarketStateProxySource

_EXTERNAL = PROJECT_ROOT / "data" / "external"
# Real daily weather (Open-Meteo/ERA5, fetched by scripts/fetch_au_weather.py),
# restricted to the Perth columns because the market is the WA grid around Perth.
_WEATHER = _EXTERNAL / "weather_au_capitals.csv"
_DEMAND = _EXTERNAL / "demand_forecast_example.csv"

# The Perth daily variables we expose as weather features. All are standard
# demand/supply drivers: temperature (cooling/heating load), apparent
# temperature, precipitation, wind (wind generation), shortwave radiation
# (rooftop PV), humidity, and surface pressure (synoptic state).
_PERTH_WEATHER_COLS = [
    "per_temperature_2m_max",
    "per_temperature_2m_min",
    "per_temperature_2m_mean",
    "per_apparent_temperature_max",
    "per_precipitation_sum",
    "per_windspeed_10m_max",
    "per_windspeed_10m_mean",
    "per_shortwave_radiation_sum",
    "per_relative_humidity_2m_mean",
    "per_surface_pressure_mean",
]


def build_exog_provider(spec, side: str = "supply", lag_days: int = 1):
    """Return an `ExogenousProvider` for the spec, or None when nothing applies."""
    if not spec:
        return None
    sources_req = spec.get("sources", []) if isinstance(spec, dict) else list(spec)
    lag = spec.get("lag_days", lag_days) if isinstance(spec, dict) else lag_days

    sources: list = []
    if "proxy" in sources_req:
        from aemo_curves.core.datasets import load_features

        train, test = load_features(side, "interp", "canonical")
        full = pd.concat([train, test], ignore_index=True)
        sources.append(
            MarketStateProxySource(embeddings=full, lag_days=lag, include_calendar=True)
        )
    if "weather" in sources_req and _WEATHER.exists():
        # Shift by the forecast horizon (`lag`), not a hardcoded 1 day, so the
        # weather source never peeks inside the forecast gap. At a 7-day horizon
        # the target date d reads weather from d - 7.
        sources.append(
            ExternalCsvSource(
                path=_WEATHER,
                mode="daily",
                shift_days=lag,
                feature_cols=_PERTH_WEATHER_COLS,
                name_prefix="weather_",
            )
        )
    # Three views of the SAME Perth weather, for the 7-day horizon:
    #   * "weather"        (above): actual weather shifted by the horizon (d-7
    #     stale proxy) - deployable but week-old.
    #   * "weather_actual" (below): actual same-day weather, no noise - a
    #     perfect but NON-deployable ceiling/oracle on weather's value.
    #   * "weather_fc"     (below): actual same-day weather + realistic day-7
    #     forecast noise - honest and strictly more informative than the stale
    #     d-7 proxy. This is the deployable treatment.
    if "weather_actual" in sources_req and _WEATHER.exists():
        sources.append(
            ExternalCsvSource(
                path=_WEATHER,
                mode="daily",
                shift_days=0,
                feature_cols=_PERTH_WEATHER_COLS,
                name_prefix="weather_actual_",
            )
        )
    if "weather_fc" in sources_req and _WEATHER.exists():
        sources.append(
            ForecastPerturbedDailySource(
                path=_WEATHER,
                feature_cols=_PERTH_WEATHER_COLS,
                name_prefix="weather_fc_",
                base_seed=42,
            )
        )
    if "demand" in sources_req and _DEMAND.exists():
        sources.append(
            ExternalCsvSource(path=_DEMAND, mode="interval", shift_days=0, name_prefix="demand_")
        )
    sources = [s for s in sources if s.dim > 0]
    return ExogenousProvider(sources) if sources else None
