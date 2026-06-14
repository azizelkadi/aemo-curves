"""Run one recipe by its dotted <group>.<name>.

Usage:
    uv run python scripts/run_experiment.py horizon_1d.baseline_lag1
    uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval --no-mlflow
    uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval --subset-days 3

A recipe lives at recipes/<group>/<name>.yaml. --subset-days runs only the first
N test days for a fast logic check (not a real evaluation).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aemo_curves.core.recipe import Recipe
from aemo_curves.runner import run_recipe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a recipe by its dotted <group>.<name>.")
    parser.add_argument("name", help="Recipe, e.g. 'horizon_1d.baseline_lag1'")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--subset-days", type=int, default=None,
                        help="Run only the first N test days (fast logic check)")
    args = parser.parse_args()

    group, _, name = args.name.partition(".")
    path = REPO / "recipes" / group / f"{name}.yaml"
    if not path.exists():
        parser.error(f"No recipe at {path}")

    recipe = Recipe.from_yaml(path)
    subset = {"test_days": args.subset_days} if args.subset_days else None
    log.info("Running recipe %s", recipe.name)
    result = run_recipe(recipe, log_to_mlflow=not args.no_mlflow, subset=subset)

    print("\n=== Experiment summary ===")
    print(f"Experiment: {result.experiment_name}")
    print(f"Strategy:   {result.model_name}")
    print(f"MLflow run: {result.mlflow_run_id}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
