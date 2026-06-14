"""Run a whole comparable group and rank the results.

A group is one `experiments/<side>/<horizon>/<metric>/` folder. Every experiment
under it must share the side, the horizon (so the same lag baseline), and the
metric, so the results are directly comparable. `run_group` runs them all (or
loads them from the cache), checks that invariant, and returns a leaderboard
sorted by the metric (best first).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aemo_curves.config import PROJECT_ROOT
from aemo_curves.core.experiment import Experiment
from aemo_curves.runner import cache as _cache
from aemo_curves.runner.run import run_experiment

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _resolve_group_dir(group: str) -> Path:
    rel = Path(*group.split(".")) if "." in group else Path(group)
    directory = EXPERIMENTS_DIR / rel
    if not directory.is_dir():
        raise ValueError(f"No group folder at {directory} (expected <side>/<horizon>/<metric>)")
    return directory


def _check_comparable(directory: Path, experiments: list[Experiment]) -> tuple[str, int, str]:
    parts = directory.relative_to(EXPERIMENTS_DIR).parts
    if len(parts) != 3:
        raise ValueError(
            f"A group is a <side>/<horizon>/<metric> folder; got {'/'.join(parts)}"
        )
    side, horizon, metric = parts
    lag_days = int(horizon.rstrip("d"))
    baselines = set()
    for e in experiments:
        if e.data.get("side") != side:
            raise ValueError(f"{e.name}: side {e.data.get('side')!r} != folder {side!r}")
        if int(e.data.get("lag_days", -1)) != lag_days:
            raise ValueError(f"{e.name}: lag_days {e.data.get('lag_days')} != horizon {horizon}")
        if e.metric != metric:
            raise ValueError(f"{e.name}: metric {e.metric!r} != folder {metric!r}")
        baselines.add(repr(e.baseline))
    if len(baselines) > 1:
        raise ValueError(f"group {'/'.join(parts)} mixes baselines: {baselines}")
    return side, lag_days, metric


def run_group(
    group: str,
    *,
    use_cache: bool = True,
    log_to_mlflow: bool = False,
    subset: dict | None = None,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run every experiment in a group and return the ranked leaderboard.

    With `use_cache`, an experiment whose definition and input tables are
    unchanged is loaded from `artifacts/cache/` instead of being recomputed, so
    adding one experiment and re-running only computes the new one.
    """
    directory = _resolve_group_dir(group)
    paths = sorted(directory.rglob("*.yaml"))
    if not paths:
        raise ValueError(f"No experiments under {directory}")
    experiments = [Experiment.from_yaml(p) for p in paths]
    side, lag_days, metric = _check_comparable(directory, experiments)

    rows = []
    for e in experiments:
        key = _cache.cache_key(e, subset=subset)
        result = _cache.load(key) if use_cache else None
        if result is not None:
            hit = True
            log.info("cache hit: %s", e.name)
        else:
            hit = False
            log.info("running:   %s", e.name)
            result = run_experiment(e, log_to_mlflow=log_to_mlflow, subset=subset)
            if use_cache:
                _cache.save(key, result)
        m = result.metrics
        rows.append({
            "experiment": e.name.split(f".{metric}.")[-1],
            "n": int(m.get("n", 0)),
            metric: m.get(f"{metric}_mean"),
            "baseline": m.get(f"baseline_{metric}_mean"),
            "delta": m.get(f"{metric}_delta_vs_baseline"),
            "win_%": (100 * m[f"{metric}_win_rate"]) if f"{metric}_win_rate" in m else None,
            "cached": hit,
        })

    table = pd.DataFrame(rows).sort_values(metric, na_position="last").reset_index(drop=True)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_csv, index=False)
        log.info("wrote leaderboard to %s", save_csv)
    return table
