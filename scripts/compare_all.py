"""Load every MLflow run, rank by mean Wasserstein, print a single ranking table.

Usage:
    uv run python scripts/compare_all.py
    uv run python scripts/compare_all.py --top 20
"""
from __future__ import annotations

import argparse

import pandas as pd

from aemo_curves.tracking import get_experiment_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50, help="Rows to show")
    args = parser.parse_args()

    runs = get_experiment_runs()
    if runs.empty:
        print("No MLflow runs found.")
        return

    cols = {
        "tags.experiment_name": "experiment",
        "tags.model_name": "model",
        "metrics.wasserstein_mean": "mean_W",
        "metrics.wasserstein_median": "median_W",
        "metrics.baseline_wasserstein_mean": "baseline_W",
        "metrics.wasserstein_delta_vs_baseline": "Δ_vs_base",
        "metrics.wasserstein_win_rate": "win_rate",
        "metrics.n": "n",
        "run_id": "run_id",
    }
    keep = [c for c in cols if c in runs.columns]
    tbl = runs[keep].rename(columns={k: cols[k] for k in keep})
    tbl = tbl.sort_values("mean_W")
    # Keep only one run per experiment_name (the best)
    tbl = tbl.drop_duplicates(subset=["experiment"], keep="first")
    tbl = tbl.head(args.top)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 3)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
