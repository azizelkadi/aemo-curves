"""Volume-axis bid-curve embedding, aligned with HC-MAE by construction.

The embedding samples the step price at K equispaced volume positions on
``[0, V_max_global]``. The Euclidean distance between two embeddings then
satisfies

    ‖z_a - z_b‖₂ ≈ √K · HC-MAE_global(a, b)

up to per-pair union-support discretisation, because HC-MAE is itself a
mean absolute error on a uniform volume grid (see
:mod:`aemo_curves.stages.evaluate.hc_mae`). The 64-d ``interp`` embedding instead
samples on a quantile grid and drops ``V_total``, so its L2 distance does
not track HC-MAE.

``V_max_global`` is computed once at fit time as a high quantile of the
training-set total cleared volumes (default 99 %). This stops one outlier
auction from stretching the grid and squashing all the signal into the
first few dimensions. Curves above this volume are clipped at the tail, and
the clipped region is treated as the curve's last-step price, matching
``hc_mae._step_eval``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.stages.evaluate.hc_mae import _step_eval

DEFAULT_V_GRID_DIM: int = 64
DEFAULT_V_MAX_QUANTILE: float = 0.99


def fit_v_max_global(
    curve_qty_list: list[np.ndarray] | pd.Series,
    quantile: float = DEFAULT_V_MAX_QUANTILE,
) -> float:
    """Return a robust upper bound for the global V grid.

    Parameters
    ----------
    curve_qty_list:
        Iterable of ``curve_qty`` arrays from the training set. The last
        element of each array is the curve's total cleared volume.
    quantile:
        Percentile of total cleared volumes to use as ``V_max_global``.
        Default 0.99 - robust to a single outlier auction.
    """
    totals = np.fromiter(
        (float(np.asarray(q, dtype=float)[-1]) for q in curve_qty_list),
        dtype=float,
    )
    if totals.size == 0:
        raise ValueError("curve_qty_list is empty")
    return float(np.quantile(totals, float(quantile)))


def curve_to_v_grid_embedding(
    curve_qty: np.ndarray,
    curve_price: np.ndarray,
    v_max_global: float,
    *,
    k: int = DEFAULT_V_GRID_DIM,
) -> np.ndarray:
    """Embed a single curve as the step price at K equispaced V positions.

    Parameters
    ----------
    curve_qty, curve_price:
        The step arrays of one auction's bid curve.
    v_max_global:
        Global volume cap, fitted on training data via
        :func:`fit_v_max_global`. The grid spans ``[0, v_max_global]``.
    k:
        Number of grid points (default 64 to match the dim of the existing
        ``interp`` embedding for drop-in interchangeability).

    Returns
    -------
    np.ndarray
        Shape ``(k,)`` float64 vector of step prices.
    """
    v_grid = np.linspace(0.0, float(v_max_global), int(k))
    qty = np.asarray(curve_qty, dtype=float)
    price = np.asarray(curve_price, dtype=float)
    return _step_eval(qty, price, v_grid).astype(float)
