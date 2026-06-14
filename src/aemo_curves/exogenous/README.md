# `aemo_curves.exogenous` - exogenous-feature support

This module joins extra signals into the per-auction feature vectors used by the
correctors. The signals come either from outside the STEM dataset or from
in-dataset proxies.

## Three families of features

| Module | Class / functions | Source |
|---|---|---|
| `proxies.py` | `MarketStateProxySource` | Per-day aggregates built from the cached embedding tables themselves: daily price mean / median / std, total quantity, embedding-shape moments, trailing 7/28-day moving averages, and calendar features. Always available. |
| `loaders.py` | `ExternalCsvSource` | A daily- or interval-resolution CSV file under `data/external/`. Real candidates: weather, AEMO demand forecast, fuel prices, macro series. |
| `holidays.py` | `is_wa_holiday`, `wa_holiday_flags` | Western Australia public-holiday flags for 2021-2026, hand-coded. |

Each source implements `ExogenousSource.features(interval, trading_date)` and
returns a fixed-length `np.ndarray`, or `None` when it has no features for that
key. The provider stitches the sources together:

```python
from aemo_curves.exogenous import (
    MarketStateProxySource, ExternalCsvSource, ExogenousProvider,
)

provider = ExogenousProvider([
    MarketStateProxySource(embeddings=full_embeddings, lag_days=1),
    ExternalCsvSource(path="data/external/weather.csv", mode="daily", shift_days=1,
                      name_prefix="weather_"),
    ExternalCsvSource(path="data/external/demand_forecast.csv", mode="interval",
                      shift_days=0, name_prefix="demand_"),
])

vec = provider(interval=12, trading_date=some_date)  # np.ndarray of fixed length
```

If any source returns `None` for a key, the provider returns `None` and the
caller drops that row.

## How experiments use the provider

Every corrector in `aemo_curves.stages.predict.correctors` accepts an optional
`exog_provider=` argument. When set, the provider is called once per training
and prediction row, and its vector is concatenated to the lag-based feature
vector inside `_row_features`.

Each exog group builds the provider with a small factory and exposes
`use_proxy`, `use_weather`, `use_demand` flags for ablation studies:

- [`horizon_1d_exog/__init__.py`](../../../experiments/horizon_1d_exog/__init__.py) (`build_provider`, proxy lag 1)
- [`horizon_7d_exog/__init__.py`](../../../experiments/horizon_7d_exog/__init__.py) (`build_provider_7d`, proxy lag 7)
- [`horizon_30d_exog/__init__.py`](../../../experiments/horizon_30d_exog/__init__.py) (proxy lag 30)

## No-leakage guarantees

- **Proxy features** for `(interval, d)` are read from the day-state table at
  `(d - lag_days)`. `lag_days` matches the horizon: 1, 7, or 30.
- **Daily external CSVs** default to `shift_days=1`: the row dated `d` holds
  features that are legal to use for predicting `d` (yesterday's observations).
- **Interval external CSVs** default to `shift_days=0`: they are assumed to be a
  day-ahead forecast, published the day before the target, and so legal to use.

The provider does not re-check leakage. It trusts the `shift_days` and
`lag_days` you set, so document your CSV provenance and match the shift to it.

## Replacing the example CSVs with real data

Both `data/external/weather_example.csv` and
`data/external/demand_forecast_example.csv` are synthetic. They let the plumbing
run end-to-end without internet access. To use real data:

1. Replace the CSV with one of the same schema (matching column names) and re-run
   the relevant experiment script. The loaders pick up the new file
   automatically.
2. See [`data/external/README.md`](../../../data/external/README.md) for
   recommended sources.
