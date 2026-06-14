"""Exogenous-feature support for curve prediction models.

Two flavours of exogenous features live here:

1. **Proxy features** - derived from the STEM dataset itself (daily
   aggregates, volatility, curve-shape moments). These are always available
   as long as cached embeddings exist. See `proxies.py`.

2. **External features** - loaded from CSV files you supply under
   `data/external/` (weather from Bureau of Meteorology, AEMO demand
   forecast, fuel prices, etc.). Missing files fall back to zeros or to
   proxy features, so your experiments still run end-to-end. See
   `loaders.py` and `data/external/README.md`.

All sources are combined by `ExogenousProvider` (see `provider.py`) which
every model in `aemo_curves.stages.predict.correctors` can accept through the
`exog_provider=` constructor argument.

The key invariant is **no leakage**: every source returns features for
`(interval, trading_date)` using only data strictly before `trading_date`.
Proxy features are shifted by a configurable number of days (default 1)
for this reason.
"""

from .builder import build_exog_provider
from .holidays import is_wa_holiday, wa_holiday_flags
from .loaders import ExternalCsvSource, load_external_daily_csv, load_external_interval_csv
from .provider import ExogenousProvider, ExogenousSource
from .proxies import (
    MarketStateProxySource,
    build_day_state_table,
    build_shifted_lookup,
)

__all__ = [
    "ExogenousProvider",
    "ExogenousSource",
    "ExternalCsvSource",
    "MarketStateProxySource",
    "build_day_state_table",
    "build_exog_provider",
    "build_shifted_lookup",
    "is_wa_holiday",
    "load_external_daily_csv",
    "load_external_interval_csv",
    "wa_holiday_flags",
]
