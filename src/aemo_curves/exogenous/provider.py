"""Unified provider that joins multiple exogenous sources into one feature vector."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class ExogenousSource(ABC):
    """Abstract source that, given `(interval, trading_date)`, returns a feature vector.

    Return `None` to signal that no features are available for that key; the
    caller will drop the row from training / prediction.

    Sources must respect the no-leakage invariant: features for
    `(interval, d)` may only use data strictly before `d`.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]: ...

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @abstractmethod
    def features(self, interval: int, trading_date) -> np.ndarray | None: ...


class ExogenousProvider:
    """Concatenates the outputs of multiple `ExogenousSource`s.

    - If any source returns `None`, the provider as a whole returns `None`
      and the row is skipped.
    - If a source returns a vector shorter/longer than its declared `dim`,
      a ValueError is raised at first invocation (fail-fast).

    Usage:
        provider = ExogenousProvider([
            MarketStateProxySource(embeddings, lag_days=1),
            ExternalCsvSource(path="data/external/weather.csv"),
        ])
        feats = provider.features(interval=12, trading_date=some_date)
    """

    def __init__(self, sources: Iterable[ExogenousSource]) -> None:
        self.sources: list[ExogenousSource] = list(sources)
        self._dims = [s.dim for s in self.sources]
        self._names: list[str] = []
        for s in self.sources:
            self._names.extend(s.feature_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def dim(self) -> int:
        return sum(self._dims)

    def features(self, interval: int, trading_date) -> np.ndarray | None:
        parts = []
        for s, d in zip(self.sources, self._dims):
            v = s.features(interval, trading_date)
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            if v.shape != (d,):
                raise ValueError(
                    f"Source {type(s).__name__} returned shape {v.shape}, expected ({d},)"
                )
            parts.append(v)
        return np.concatenate(parts, axis=0) if parts else np.zeros(0)

    def __call__(self, interval: int, trading_date) -> np.ndarray | None:
        return self.features(interval, trading_date)
