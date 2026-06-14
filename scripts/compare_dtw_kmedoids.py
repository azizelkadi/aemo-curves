"""Build the comparison table + plots for the dtw_kmedoids experiment groups.

Reads MLflow runs whose experiment name starts with `dtwkm__`. Supports
filtering by side (``supply`` or ``demand``) so the supply and demand
groups produce separate, comparable artefacts:

  - ``--side supply`` (default): writes ``artifacts/comparison_dtw_kmedoids.{csv,png}``.
  - ``--side demand``: writes ``artifacts/comparison_dtw_kmedoids_demand.{csv,png}``.
  - ``--side all``: writes ``artifacts/comparison_dtw_kmedoids_all.{csv,png}`` (mixes
    sides; only useful for diagnostic scans).

Usage:
    uv run python scripts/compare_dtw_kmedoids.py --side supply
    uv run python scripts/compare_dtw_kmedoids.py --side demand
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aemo_curves.config import ARTIFACTS_DIR
from aemo_curves.tracking import get_experiment_runs

logging.basicConfig(level=logging.WARNING)


def _load(side: str) -> pd.DataFrame:
    runs = get_experiment_runs()
    keep = [
        "tags.experiment_name",
        "metrics.dtw_predicted_mean",
        "metrics.dtw_oracle_mean",
        "metrics.dtw_baseline_mean",
        "metrics.dtw_delta_vs_baseline",
        "metrics.dtw_win_rate_vs_baseline",
        "metrics.n",
        "params.k",
        "params.alpha",
    ]
    keep = [c for c in keep if c in runs.columns]
    df = (
        runs[keep]
        .rename(columns={
            "tags.experiment_name": "name",
            "metrics.dtw_predicted_mean": "predicted",
            "metrics.dtw_oracle_mean": "oracle",
            "metrics.dtw_baseline_mean": "baseline",
            "metrics.dtw_delta_vs_baseline": "delta",
            "metrics.dtw_win_rate_vs_baseline": "win_vs_base",
        })
        .dropna(subset=["name"])
    )
    df = df[df["name"].str.startswith("dtwkm__")].copy()
    if side != "all":
        df = df[df["name"].str.contains(f"_{side}_", regex=False)].copy()
    return df.drop_duplicates(subset=["name"], keep="first").sort_values("predicted").reset_index(drop=True)


def _suffix(side: str) -> str:
    return "" if side == "supply" else f"_{side}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["supply", "demand", "all"], default="supply")
    args = p.parse_args()

    df = _load(args.side)
    if df.empty:
        print(f"No dtwkm__ runs found for side={args.side}.")
        return

    print(df[["name", "predicted", "oracle", "baseline", "delta", "win_vs_base"]].to_string(index=False))
    suffix = _suffix(args.side)
    out_csv = ARTIFACTS_DIR / f"comparison_dtw_kmedoids{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(df)), 5))
    width = 0.27
    x = np.arange(len(df))
    ax.bar(x - width, df["predicted"], width, label="Predicted", color="steelblue")
    ax.bar(x, df["baseline"], width, label="Baseline (lag-1)", color="crimson")
    ax.bar(x + width, df["oracle"], width, label="Oracle (closest medoid to true)", color="seagreen")
    for i, (p_, b, o) in enumerate(zip(df["predicted"], df["baseline"], df["oracle"])):
        ax.text(i - width, p_, f"{p_:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, o, f"{o:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=20, ha="right")
    ax.set_ylabel("Mean DTW (lower = better)")
    ax.set_title(f"dtw_kmedoids ({args.side}): predicted vs baseline (lag-1) vs oracle")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out_png = ARTIFACTS_DIR / f"comparison_dtw_kmedoids{suffix}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
