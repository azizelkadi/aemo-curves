"""Visualizations that compare multiple ExperimentResult objects against each other and the baseline."""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from aemo_curves.core.result import ExperimentResult


def _collect(results: Mapping[str, ExperimentResult]) -> dict[str, pd.DataFrame]:
    return {name: res.predictions for name, res in results.items()}


def compare_runs_table(results: Mapping[str, ExperimentResult]) -> pd.DataFrame:
    """Wide table: one row per run, one col per summary metric."""
    rows = []
    for name, res in results.items():
        row = {
            "experiment": name,
            "model": res.model_name,
            "mlflow_run_id": res.mlflow_run_id,
            "n": len(res.predictions),
        }
        row.update(res.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "wasserstein_mean" in df.columns:
        df = df.sort_values("wasserstein_mean").reset_index(drop=True)
    return df


def plot_run_means(
    results: Mapping[str, ExperimentResult],
    baseline_name: str | None = "baseline_lag1",
):
    """Bar chart of mean Wasserstein per run. If baseline is among results, highlight it."""
    df = compare_runs_table(results)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(df)), 4.5))
    colors = [
        "crimson" if baseline_name and name == baseline_name else "steelblue"
        for name in df["experiment"]
    ]
    bars = ax.bar(
        df["experiment"],
        df["wasserstein_mean"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )
    for bar, v in zip(bars, df["wasserstein_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Wasserstein ($/MWh)")
    ax.set_title("Run comparison - mean Wasserstein (lower = better)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_run_distributions(results: Mapping[str, ExperimentResult]):
    """Boxplot of curve_wasserstein across runs."""
    preds = _collect(results)
    data = [p["curve_wasserstein"].dropna().to_numpy() for p in preds.values()]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(preds)), 4.5))
    ax.boxplot(data, tick_labels=list(preds.keys()), patch_artist=True,
               medianprops={"color": "red", "linewidth": 2})
    ax.set_ylabel("Wasserstein distance ($/MWh)")
    ax.set_title("Wasserstein distribution per run")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_residuals_over_time(
    result: ExperimentResult,
    rolling_days: int = 14,
):
    """Plot per-day mean Wasserstein of a run against the lag-1 baseline if available."""
    df = result.predictions.copy()
    df["Trading Date"] = pd.to_datetime(df["Trading Date"])
    daily = df.groupby("Trading Date").agg(
        model=("curve_wasserstein", "mean"),
        baseline=("baseline_wasserstein", "mean")
        if "baseline_wasserstein" in df.columns
        else ("curve_wasserstein", "mean"),
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily["model"], label=result.model_name, color="steelblue", alpha=0.4)
    ax.plot(
        daily.index,
        daily["model"].rolling(rolling_days, min_periods=1).mean(),
        label=f"{result.model_name} ({rolling_days}d MA)",
        color="steelblue",
        linewidth=2,
    )
    if "baseline_wasserstein" in df.columns:
        ax.plot(daily.index, daily["baseline"], label="baseline", color="tomato", alpha=0.4)
        ax.plot(
            daily.index,
            daily["baseline"].rolling(rolling_days, min_periods=1).mean(),
            label=f"baseline ({rolling_days}d MA)",
            color="tomato",
            linewidth=2,
        )
    ax.set_title(f"Daily mean Wasserstein - {result.experiment_name}")
    ax.set_ylabel("Wasserstein ($/MWh)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
