"""Strategy registry: every stage maps strategy names to implementations.

An experiment names one strategy per stage; the runner looks it up here. Importing a
stage package registers its strategies as a side effect, so call `bootstrap()`
(or import the stage packages) before resolving names.
"""
from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, dict[str, object]] = {}


def register(stage: str, name: str) -> Callable:
    def deco(obj):
        _REGISTRY.setdefault(stage, {})[name] = obj
        return obj

    return deco


def get(stage: str, name: str):
    try:
        return _REGISTRY[stage][name]
    except KeyError:
        raise KeyError(
            f"No strategy {name!r} for stage {stage!r}. "
            f"Available: {available(stage)}"
        ) from None


def has(stage: str, name: str) -> bool:
    return name in _REGISTRY.get(stage, {})


def available(stage: str) -> list[str]:
    return sorted(_REGISTRY.get(stage, {}))


def all_strategies() -> dict[str, list[str]]:
    return {stage: sorted(names) for stage, names in _REGISTRY.items()}
