"""The runner produces a valid scored result on a few test days.

These are logic checks on a tiny subset, not real evaluations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aemo_curves.core.recipe import Recipe
from aemo_curves.runner import run_recipe

REPO = Path(__file__).resolve().parents[1]
SUBSET = {"test_days": 2, "max_train": 4000}

RECIPES = [
    "horizon_1d/baseline_lag1.yaml",
    "horizon_1d/ridge_retrieval.yaml",
]


def _artifacts_ready() -> bool:
    return (REPO / "artifacts" / "embeddings" / "train_supply.pkl").exists()


@pytest.mark.skipif(not _artifacts_ready(), reason="cached embeddings not built")
@pytest.mark.parametrize("rel", RECIPES)
def test_runner_smoke(rel):
    recipe = Recipe.from_yaml(REPO / "recipes" / rel)
    result = run_recipe(recipe, log_to_mlflow=False, subset=SUBSET)
    assert result.metrics["n"] > 0
    assert recipe.metric + "_mean" in result.metrics
    cols = set(result.predictions.columns)
    assert {"pred_curve_qty", "pred_curve_price", f"curve_{recipe.metric}"} <= cols
