"""The context a predict strategy receives.

It carries the resolved feature tables plus the data choices a strategy may
need (side, feature kind, split, the lag horizon, an optional exogenous
provider). Strategies read what they need and ignore the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PredictContext:
    train: pd.DataFrame
    test: pd.DataFrame
    historical: pd.DataFrame
    side: str = "supply"
    features: str = "interp"
    split: str = "canonical"
    lag_days: int = 1
    exog_provider: object | None = None
    extra: dict = field(default_factory=dict)
