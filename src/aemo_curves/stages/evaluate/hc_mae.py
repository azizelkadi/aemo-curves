"""Heterogeneous-Curves Mean Absolute Error (HC-MAE).

The metric proposed by Sinha & Lucheroni (Applied Energy 2025) to compare two
electricity bid curves that may have a different step structure. The definition
follows
[`docs/demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md`](../../../docs/demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md)
§2.7:

    1. ``V_u = sort(V_1 ∪ V_2)``  (union of step boundaries).
    2. ``V_common`` = ``n_grid`` (default 10 000) equispaced points spanning
       ``[min(V_u), max(V_u)]``.
    3. Step-interpolate ``P_1, P_2`` onto ``V_common``  → ``Π_1, Π_2``.
    4. ``HC-MAE = (1 / n_grid) · Σ |Π_1 - Π_2|``.

Properties (Sinha & Lucheroni §2.7): non-negative, identity-of-indiscernibles,
symmetric. Different from 1-D Wasserstein because the integration is over the
**volume axis** with uniform weighting, not over the price-quantile axis with
quantity weights.

Step convention (matches `metrics/wasserstein.py`): curve `k` covers the
half-open interval ``(qty[k-1], qty[k]]`` (with ``qty[-1] := 0`` implicit) and
has price ``price[k]``. The right-side ``np.searchsorted(side="left")`` lookup
matches this convention.
"""
from __future__ import annotations

import numpy as np


def _step_eval(qty: np.ndarray, price: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the step price at each volume in `v`.

    Out-of-range volumes are clipped to the nearest step's price.
    """
    idx = np.searchsorted(qty, v, side="left")
    idx = np.clip(idx, 0, len(price) - 1)
    return price[idx]


def curve_hc_mae_distance(
    qty_a: np.ndarray,
    price_a: np.ndarray,
    qty_b: np.ndarray,
    price_b: np.ndarray,
    *,
    n_grid: int = 10_000,
) -> float:
    """Heterogeneous-Curves MAE between two step curves (Sinha & Lucheroni 2025).

    Parameters
    ----------
    qty_a, price_a, qty_b, price_b:
        Cumulative-quantity (right-edge of step) and step-price arrays for the
        two curves.
    n_grid:
        Number of equispaced volume points on the union support. Default 10 000
        as in the paper.

    Returns
    -------
    float
        Mean absolute price difference over the common volume grid, in the same
        units as the input prices (typically $/MWh).
    """
    qa = np.asarray(qty_a, dtype=float)
    pa = np.asarray(price_a, dtype=float)
    qb = np.asarray(qty_b, dtype=float)
    pb = np.asarray(price_b, dtype=float)
    if len(qa) == 0 or len(qb) == 0 or len(pa) == 0 or len(pb) == 0:
        return float("nan")
    v_min = float(min(qa.min(), qb.min()))
    v_max = float(max(qa.max(), qb.max()))
    if v_max <= v_min:
        return 0.0
    v_common = np.linspace(v_min, v_max, int(n_grid))
    pi_a = _step_eval(qa, pa, v_common)
    pi_b = _step_eval(qb, pb, v_common)
    return float(np.mean(np.abs(pi_a - pi_b)))
