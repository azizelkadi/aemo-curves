"""The runner produces a valid scored result on a few test days.

These are logic checks on a tiny subset, not real evaluations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aemo_curves.core.experiment import Experiment
from aemo_curves.runner import run_experiment

REPO = Path(__file__).resolve().parents[1]
SUBSET = {"test_days": 2, "max_train": 4000}

EXPERIMENTS = [
    "demand/7d/hc_mae/baseline/lag7.yaml",
    "demand/7d/hc_mae/weekly/same_weekday_avg.yaml",
]


def _artifacts_ready() -> bool:
    return (REPO / "artifacts" / "embeddings" / "test_v_grid_demand.pkl").exists()


@pytest.mark.skipif(not _artifacts_ready(), reason="cached embeddings not built")
@pytest.mark.parametrize("rel", EXPERIMENTS)
def test_runner_smoke(rel):
    experiment = Experiment.from_yaml(REPO / "experiments" / rel)
    result = run_experiment(experiment, log_to_mlflow=False, subset=SUBSET)
    assert result.metrics["n"] > 0
    assert experiment.metric + "_mean" in result.metrics
    cols = set(result.predictions.columns)
    assert {"pred_curve_qty", "pred_curve_price", f"curve_{experiment.metric}"} <= cols
