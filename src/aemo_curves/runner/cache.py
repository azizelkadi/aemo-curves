"""Result cache for the group runner.

A cache key is the hash of an experiment's result-affecting definition (data
block, predict, evaluate, exogenous spec), plus a fingerprint of the input
tables it reads (size and modification time), plus the subset used. A cached
result is reused only when all of those match, so adding one experiment to a
group re-runs only that one, and rebuilding the cached tables invalidates only
the experiments that read them.

Strategy code changes are not tracked. After changing a strategy's code, run
with cache off or clear the cache (`rm -rf artifacts/cache`); bump
`CACHE_VERSION` to invalidate every entry at once.
"""
from __future__ import annotations

import hashlib
import json

from aemo_curves.config import ARTIFACTS_DIR, CURVES_DIR, DATA_DIR, EMBEDDINGS_DIR
from aemo_curves.core.datasets import CANONICAL_ONLY_TABLES
from aemo_curves.core.experiment import Experiment
from aemo_curves.core.result import ExperimentResult

CACHE_DIR = ARTIFACTS_DIR / "cache"
CACHE_VERSION = 2


def _feature_files(data: dict) -> list:
    side = data.get("side", "supply")
    features = data.get("features", "interp")
    split = data.get("split", "canonical")
    k = data.get("k")
    if features == "kmedoids":
        names = [f"train_kmedoids_{side}_k{k}", f"test_kmedoids_{side}_k{k}"]
    elif features in CANONICAL_ONLY_TABLES:
        name_tr, name_te, sfx = CANONICAL_ONLY_TABLES[features]
        names = [f"{name_tr}_{side}_{sfx}", f"{name_te}_{side}_{sfx}"]
        if features.startswith("hcmae_medoids_"):
            # the medoid oracle also reads the medoid curves
            names.append(f"hcmae_medoids_{side}_{sfx}")
    elif features == "v_grid":
        sfx = side if split == "canonical" else f"{side}_{split}"
        names = [f"train_v_grid_{sfx}", f"test_v_grid_{sfx}"]
    else:
        names = [f"train_{side}", f"test_{side}"]
    return [EMBEDDINGS_DIR / f"{n}.pkl" for n in names]


def _input_files(experiment: Experiment) -> list:
    files = _feature_files(experiment.data)
    if experiment.predict.get("strategy") == "adaptive":
        # The adaptive family consumes a precomputed table; fingerprint it so
        # rebuilding the table invalidates every adaptive experiment.
        pc = experiment.predict.get("params", {}).get(
            "precompute", "adaptive_precompute_demand_7d.pkl"
        )
        files.append(EMBEDDINGS_DIR / pc)
    if experiment.extra_params.get("exog"):
        side = experiment.data.get("side", "supply")
        files += [EMBEDDINGS_DIR / f"train_{side}.pkl", EMBEDDINGS_DIR / f"test_{side}.pkl"]
        files += [DATA_DIR / "external" / "weather_au_capitals.csv",
                  DATA_DIR / "external" / "weather_example.csv",
                  DATA_DIR / "external" / "demand_forecast_example.csv"]
    # The market-clearing evaluation reads the true supply/demand curve tables on
    # every experiment, so a curve rebuild must invalidate the cached clearing
    # numbers.
    files += [CURVES_DIR / f"{n}.pkl"
              for n in ("train_supply", "test_supply", "train_demand", "test_demand")]
    return sorted(set(files), key=str)


def _data_fingerprint(experiment: Experiment) -> list:
    out = []
    for f in _input_files(experiment):
        if f.exists():
            st = f.stat()
            out.append([f.name, st.st_size, st.st_mtime_ns])
        else:
            out.append([f.name, "missing"])
    return out


def cache_key(experiment: Experiment, subset: dict | None = None) -> str:
    payload = {
        "version": CACHE_VERSION,
        "data": experiment.data,
        "predict": experiment.predict,
        "evaluate": experiment.evaluate,
        "exog": experiment.extra_params.get("exog"),
        "subset": subset,
        "inputs": _data_fingerprint(experiment),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load(key: str) -> ExperimentResult | None:
    entry = CACHE_DIR / key
    if (entry / "result.json").exists():
        return ExperimentResult.load(entry)
    return None


def save(key: str, result: ExperimentResult) -> None:
    result.save(CACHE_DIR / key)
