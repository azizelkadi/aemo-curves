"""Every recipe loads and names a registered strategy, metric, and baseline."""
from __future__ import annotations

from pathlib import Path

import pytest

from aemo_curves.core.recipe import Recipe
from aemo_curves.core.registry import has
from aemo_curves.runner import bootstrap

REPO = Path(__file__).resolve().parents[1]
RECIPES = sorted((REPO / "recipes").rglob("*.yaml"))


def test_recipes_exist():
    assert RECIPES, "no recipes found under recipes/"


@pytest.mark.parametrize("path", RECIPES, ids=[p.stem for p in RECIPES])
def test_recipe_resolves(path):
    bootstrap()
    recipe = Recipe.from_yaml(path)
    assert has("predict", recipe.predict_strategy), recipe.predict_strategy
    assert has("evaluate", recipe.metric), recipe.metric
    if recipe.baseline is not None:
        assert has("predict", recipe.baseline["strategy"]), recipe.baseline["strategy"]
    assert "side" in recipe.data and "features" in recipe.data
