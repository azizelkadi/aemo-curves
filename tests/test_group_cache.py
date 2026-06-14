"""The cache keys results correctly and the group runner reuses them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aemo_curves.core.experiment import Experiment
from aemo_curves.core.result import ExperimentResult
from aemo_curves.runner import cache, run_group

REPO = Path(__file__).resolve().parents[1]

BASE = {
    "name": "demand.7d.hc_mae.vgrid.strat_knn",
    "data": {"side": "demand", "features": "v_grid", "split": "canonical", "lag_days": 7},
    "predict": {"strategy": "static_retrieval", "params": {"query": "ridge", "ridge_alpha": 200.0}},
    "evaluate": {"metric": "hc_mae", "baseline": {"strategy": "lag1", "params": {"lag_days": 7}}},
}


def _exp(**over):
    d = {**BASE, **over}
    return Experiment.from_dict(d)


def test_cache_key_is_stable_and_param_sensitive():
    assert cache.cache_key(_exp()) == cache.cache_key(_exp())
    # the name and notes do not affect the result, so they do not change the key
    assert cache.cache_key(_exp()) == cache.cache_key(_exp(name="renamed"))
    # a changed hyperparameter does change the key
    other = _exp(predict={"strategy": "static_retrieval", "params": {"query": "ridge", "ridge_alpha": 100.0}})
    assert cache.cache_key(_exp()) != cache.cache_key(other)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    arr = np.array([0.0, 1.0])
    preds = pd.DataFrame({
        "Trading Date": pd.to_datetime(["2024-05-01"]),
        "Interval Number": [1],
        "true_curve_qty": [arr], "true_curve_price": [arr],
        "pred_curve_qty": [arr], "pred_curve_price": [arr],
    })
    res = ExperimentResult("e", "lag1", {"side": "demand"}, {"hc_mae_mean": 1.5}, preds)
    key = "abc123"
    assert cache.load(key) is None
    cache.save(key, res)
    got = cache.load(key)
    assert got is not None and got.metrics["hc_mae_mean"] == 1.5


def _group_ready() -> bool:
    return (REPO / "artifacts" / "embeddings" / "test_v_grid_demand.pkl").exists()


@pytest.mark.skipif(not _group_ready(), reason="demand v_grid tables not built")
def test_group_runner_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    subset = {"test_days": 2, "max_train": 3000}
    first = run_group("demand.7d.hc_mae", use_cache=True, log_to_mlflow=False, subset=subset)
    assert len(first) == 6
    assert "hc_mae" in first.columns
    assert first["hc_mae"].is_monotonic_increasing  # ranked best first
    assert not first["cached"].any()
    second = run_group("demand.7d.hc_mae", use_cache=True, log_to_mlflow=False, subset=subset)
    assert second["cached"].all()  # every experiment served from cache
