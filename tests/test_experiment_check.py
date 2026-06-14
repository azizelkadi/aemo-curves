"""Every experiment loads and names a registered strategy, metric, and baseline."""
from __future__ import annotations

from pathlib import Path

import pytest

from aemo_curves.core.experiment import Experiment
from aemo_curves.core.registry import has
from aemo_curves.runner import bootstrap

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((REPO / "experiments").rglob("*.yaml"))


def test_experiments_exist():
    assert EXPERIMENTS, "no experiments found under experiments/"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[p.stem for p in EXPERIMENTS])
def test_experiment_resolves(path):
    bootstrap()
    experiment = Experiment.from_yaml(path)
    assert has("predict", experiment.predict_strategy), experiment.predict_strategy
    assert has("evaluate", experiment.metric), experiment.metric
    if experiment.baseline is not None:
        assert has("predict", experiment.baseline["strategy"]), experiment.baseline["strategy"]
    assert "side" in experiment.data and "features" in experiment.data
