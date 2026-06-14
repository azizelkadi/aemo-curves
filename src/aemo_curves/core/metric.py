"""Metric abstraction: wrap a curve pair-distance function with naming.

A `Metric` pairs a symmetric curve-distance function with a short name. The name
derives the per-row and summary column names used by the evaluate stage. The
metric instances themselves (wasserstein, dtw, hc_mae) are registered as
evaluate strategies in `stages/evaluate`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    name: str
    pair_fn: Callable
    default_kwargs: dict = field(default_factory=dict)

    def distance(self, true_qty, true_price, pred_qty, pred_price) -> float:
        return float(
            self.pair_fn(true_qty, true_price, pred_qty, pred_price, **self.default_kwargs)
        )

    @property
    def curve_col(self) -> str:
        return f"curve_{self.name}"

    @property
    def baseline_col(self) -> str:
        return f"baseline_{self.name}"
