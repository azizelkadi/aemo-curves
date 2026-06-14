"""Build an ExogenousProvider from a recipe's small `exog` spec.

The spec lists which sources to combine, for example
`{"sources": ["proxy"]}` or `{"sources": ["proxy", "weather"]}`. The proxy
source is derived from the cached curves; weather and demand are optional CSVs
under `data/external/` and are skipped when the file is missing.
"""
from __future__ import annotations

import pandas as pd

from aemo_curves.config import PROJECT_ROOT
from aemo_curves.exogenous.loaders import ExternalCsvSource
from aemo_curves.exogenous.provider import ExogenousProvider
from aemo_curves.exogenous.proxies import MarketStateProxySource

_EXTERNAL = PROJECT_ROOT / "data" / "external"
_WEATHER = _EXTERNAL / "weather_example.csv"
_DEMAND = _EXTERNAL / "demand_forecast_example.csv"


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
        sources.append(
            ExternalCsvSource(path=_WEATHER, mode="daily", shift_days=1, name_prefix="weather_")
        )
    if "demand" in sources_req and _DEMAND.exists():
        sources.append(
            ExternalCsvSource(path=_DEMAND, mode="interval", shift_days=0, name_prefix="demand_")
        )
    sources = [s for s in sources if s.dim > 0]
    return ExogenousProvider(sources) if sources else None
