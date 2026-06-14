"""Run a whole comparable group and print the leaderboard.

A group is one `experiments/<side>/<horizon>/<metric>/` folder. Every experiment
under it is comparable (same side, horizon, metric, baseline), so this runs them
all and ranks them by the metric.

Usage:
    uv run python scripts/run_group.py demand.7d.hc_mae
    uv run python scripts/run_group.py demand.7d.hc_mae --no-cache
    uv run python scripts/run_group.py supply.7d.hc_mae --subset-days 3 --no-mlflow

By default results are cached under `artifacts/cache/`, so re-running a group
after adding one experiment only computes the new one. Pass --no-cache to force
a full recompute (do this after changing strategy code).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aemo_curves.runner import run_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a comparable group and rank it.")
    parser.add_argument("group", help="Group, e.g. 'demand.7d.hc_mae'")
    parser.add_argument("--no-cache", action="store_true", help="Recompute every experiment")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--subset-days", type=int, default=None, help="First N test days only")
    parser.add_argument("--save", default=None, help="Write the leaderboard CSV here")
    args = parser.parse_args()

    import pandas as pd

    subset = {"test_days": args.subset_days} if args.subset_days else None
    table = run_group(
        args.group,
        use_cache=not args.no_cache,
        log_to_mlflow=not args.no_mlflow,
        subset=subset,
        save_csv=args.save,
    )
    pd.set_option("display.width", 200)
    print(f"\n=== {args.group} (lower is better) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
