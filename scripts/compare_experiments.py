"""List / compare MLflow experiment runs.

Usage:
    uv run python scripts/compare_experiments.py list
    uv run python scripts/compare_experiments.py compare <run_id_1> <run_id_2> ...
"""
from __future__ import annotations

import argparse
import logging

from aemo_curves.tracking import get_experiment_runs, load_run_result
from aemo_curves.viz.comparison import compare_runs_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def cmd_list() -> None:
    runs = get_experiment_runs()
    if runs.empty:
        print("No MLflow runs yet.")
        return
    cols = [
        "run_id",
        "tags.experiment_name",
        "tags.model_name",
        "metrics.wasserstein_mean",
        "metrics.baseline_wasserstein_mean",
        "metrics.wasserstein_delta_vs_baseline",
        "metrics.wasserstein_win_rate",
        "metrics.n",
        "status",
        "start_time",
    ]
    keep = [c for c in cols if c in runs.columns]
    print(runs[keep].to_string(index=False))


def cmd_compare(run_ids: list[str]) -> None:
    results = {}
    for rid in run_ids:
        res = load_run_result(rid)
        label = res.experiment_name
        if label in results:
            label = f"{label}__{rid[:8]}"
        results[label] = res
    table = compare_runs_table(results)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List all runs.")
    c = sub.add_parser("compare", help="Compare runs by id.")
    c.add_argument("run_ids", nargs="+")
    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "compare":
        cmd_compare(args.run_ids)


if __name__ == "__main__":
    main()
