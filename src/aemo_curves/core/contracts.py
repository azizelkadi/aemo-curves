"""Output contracts for the pipeline stages.

A `Contract` lists the columns a stage must produce. `validate()` raises
`ContractError` when a frame does not match, so a wrong stage output fails
loudly here instead of flowing silently into the next stage.

Element checks run on a small head sample only, because the curve columns hold
numpy arrays and checking every one of ~17k rows would be slow with no extra
safety.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised when a dataframe does not satisfy a Contract."""


def is_array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, list, tuple))


@dataclass(frozen=True)
class Column:
    name: str
    required: bool = True
    check: Callable[[object], bool] | None = None
    note: str = ""


@dataclass(frozen=True)
class Contract:
    name: str
    columns: tuple[Column, ...]
    keys: tuple[str, ...] = field(default=())

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def validate(self, df: pd.DataFrame, *, sample: int = 50) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ContractError(f"{self.name}: expected a DataFrame, got {type(df)!r}")
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ContractError(f"{self.name}: missing required columns {missing}")
        if len(df) == 0:
            raise ContractError(f"{self.name}: frame is empty")
        for key in self.keys:
            if df[key].isna().any():
                raise ContractError(f"{self.name}: key column {key!r} has nulls")
        if self.keys and df.duplicated(list(self.keys)).any():
            n = int(df.duplicated(list(self.keys)).sum())
            raise ContractError(f"{self.name}: {n} duplicate rows for keys {self.keys}")
        head = df.head(sample)
        for col in self.columns:
            if col.check is None or col.name not in df.columns:
                continue
            for value in head[col.name]:
                if value is None:
                    continue
                if not col.check(value):
                    detail = f" ({col.note})" if col.note else ""
                    raise ContractError(
                        f"{self.name}: column {col.name!r} failed its element check{detail}"
                    )
        return df
