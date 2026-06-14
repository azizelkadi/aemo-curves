"""Build a comparison table + plot for the 1-day-ahead exogenous-feature sweep.

Reads all MLflow runs whose experiment name starts with `1d_exog__`, groups
them by (model, exog_tag), and plots Δ vs lag-1 baseline + win-rate
ablations across sources.

Usage:
    uv run python scripts/compare_exog_1d.py
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aemo_curves.config import ARTIFACTS_DIR
from aemo_curves.tracking import get_experiment_runs

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def load_1d_exog_runs() -> pd.DataFrame:
    runs = get_experiment_runs()
    keep = [
        "tags.experiment_name",
        "metrics.wasserstein_mean",
        "metrics.wasserstein_delta_vs_baseline",
        "metrics.wasserstein_win_rate",
        "metrics.baseline_wasserstein_mean",
        "params.exog_tag",
        "params.exog_dim",
        "params.alpha",
        "params.shrinkage",
    ]
    keep = [c for c in keep if c in runs.columns]
    df = (
        runs[keep]
        .rename(columns={
            "tags.experiment_name": "name",
            "metrics.wasserstein_mean": "mean",
            "metrics.wasserstein_delta_vs_baseline": "delta",
            "metrics.wasserstein_win_rate": "win",
            "metrics.baseline_wasserstein_mean": "base_mean",
            "params.exog_tag": "exog_tag",
            "params.exog_dim": "exog_dim",
        })
        .dropna(subset=["name", "delta"])
    )
    df = df[df["name"].str.startswith("1d_exog__")].copy()
    # Deduplicate - keep first (most recent) run per experiment name.
    df = df.drop_duplicates(subset=["name"], keep="first").sort_values("delta").reset_index(drop=True)
    # Split name into model + exog_tag
    # Expected pattern: 1d_exog__<model>__<tag>
    parts = df["name"].str.split("__", n=2, expand=True)
    df["model"] = parts[1] if 1 in parts.columns else df["name"]
    df["tag"] = parts[2].fillna("none") if 2 in parts.columns else "none"
    return df


def build_pivot(df: pd.DataFrame, value: str = "delta") -> pd.DataFrame:
    tags_order = ["none", "proxy", "proxy+wx", "proxy+dem", "proxy+wx+dem"]
    piv = df.pivot_table(index="model", columns="tag", values=value, aggfunc="first")
    cols = [c for c in tags_order if c in piv.columns]
    return piv[cols]


def plot_heatmap(df: pd.DataFrame, out_path):
    piv = build_pivot(df, "delta")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(piv))))
    vmax = max(abs(piv.values.min()), abs(piv.values.max()))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, f"{v:+.3f}",
                ha="center", va="center", fontsize=9,
                color="black" if abs(v) < vmax * 0.6 else "white",
            )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ vs lag-1 baseline ($/MWh)  [green = better]")
    ax.set_title("1-day-ahead: Δ Wasserstein across (model × exogenous sources)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_winrate(df: pd.DataFrame, out_path):
    piv = build_pivot(df, "win")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(piv))))
    im = ax.imshow(piv.values * 100, cmap="Greens", vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=9,
                    color="black" if 100*v < 45 else "white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Win rate (%)")
    ax.set_title("1-day-ahead: % rows where model beats lag-1 (exog ablation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def main() -> None:
    df = load_1d_exog_runs()
    if df.empty:
        print("No 1d_exog__ runs found in MLflow.")
        return
    print(
        df[["name", "mean", "delta", "win", "exog_tag", "exog_dim"]]
        .to_string(index=False)
    )
    csv_path = ARTIFACTS_DIR / "comparison_1d_exog.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    plot_heatmap(df, ARTIFACTS_DIR / "comparison_1d_exog_heatmap.png")
    plot_winrate(df, ARTIFACTS_DIR / "comparison_1d_exog_winrate.png")
    print(f"Saved: {ARTIFACTS_DIR / 'comparison_1d_exog_heatmap.png'}")
    print(f"Saved: {ARTIFACTS_DIR / 'comparison_1d_exog_winrate.png'}")


if __name__ == "__main__":
    main()
