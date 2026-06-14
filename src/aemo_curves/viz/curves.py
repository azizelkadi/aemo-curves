"""Visualizations that focus on single predictions (curve shapes, embeddings)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import DEFAULT_QUANTITY_GRID


def plot_curve_examples(
    scored_predictions: pd.DataFrame,
    n_examples: int = 6,
    seed: int = 42,
    title_suffix: str = "",
    ax_grid: tuple[int, int] | None = None,
    show_baseline: bool = True,
):
    """Plot a few true vs. predicted (and optionally baseline) curves side by side."""
    if len(scored_predictions) == 0:
        raise ValueError("empty predictions")
    n = min(n_examples, len(scored_predictions))
    sample = scored_predictions.sample(n=n, random_state=seed).reset_index(drop=True)
    ncols, nrows = ax_grid if ax_grid else (2, int(np.ceil(n / 2)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (_, row) in enumerate(sample.iterrows()):
        ax = axes[i]
        ax.step(row["true_curve_qty"], row["true_curve_price"], where="post",
                label="True", linewidth=2, color="steelblue")
        ax.step(row["pred_curve_qty"], row["pred_curve_price"], where="post",
                label=f"Pred ({row.get('retrieved_date', 'model')})",
                linewidth=2, linestyle="--", color="tomato")
        w = row.get("curve_wasserstein", float("nan"))
        title = f"{row['Trading Date']} | Int {row['Interval Number']} | W={w:.1f}{title_suffix}"
        if show_baseline and "baseline_wasserstein" in row:
            title += f" (baseline W={row['baseline_wasserstein']:.1f})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cumulative quantity (MWh)")
        ax.set_ylabel("Price ($/MWh)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    return fig


def plot_embedding_comparison(
    scored_predictions: pd.DataFrame,
    idx: int = 0,
    quantity_grid: np.ndarray = DEFAULT_QUANTITY_GRID,
):
    """Plot true vs predicted embedding (price @ normalized quantile)."""
    row = scored_predictions.iloc[idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(quantity_grid, row["true_embedding"], label="True", color="steelblue", linewidth=2)
    ax.plot(quantity_grid, row["pred_embedding"], label="Predicted",
            color="tomato", linewidth=2, linestyle="--")
    ax.set_xlabel("Normalized cumulative quantity")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title(
        f"{row['Trading Date']} | Interval {row['Interval Number']} | "
        f"L2 {row.get('embedding_l2', float('nan')):.1f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
