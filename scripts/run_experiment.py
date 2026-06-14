"""Run one experiment by its dotted <group>.<name>.

Usage:
    uv run python scripts/run_experiment.py horizon_1d.baseline_lag1
    uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval --no-mlflow
    uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval --subset-days 3

An experiment lives at experiments/<group>/<name>.yaml. --subset-days runs only the first
N test days for a fast logic check (not a real evaluation).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aemo_curves.core.experiment import Experiment
from aemo_curves.runner import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment by its dotted <group>.<name>.")
    parser.add_argument("name", help="Experiment, e.g. 'horizon_1d.baseline_lag1'")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--subset-days", type=int, default=None,
                        help="Run only the first N test days (fast logic check)")
    args = parser.parse_args()

    group, _, name = args.name.partition(".")
    path = REPO / "experiments" / group / f"{name}.yaml"
    if not path.exists():
        parser.error(f"No experiment at {path}")

    experiment = Experiment.from_yaml(path)
    subset = {"test_days": args.subset_days} if args.subset_days else None
    log.info("Running experiment %s", experiment.name)
    result = run_experiment(experiment, log_to_mlflow=not args.no_mlflow, subset=subset)

    print("\n=== Experiment summary ===")
    print(f"Experiment: {result.experiment_name}")
    print(f"Strategy:   {result.model_name}")
    print(f"MLflow run: {result.mlflow_run_id}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
