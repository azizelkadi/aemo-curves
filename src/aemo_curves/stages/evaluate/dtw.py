"""Dynamic Time Warping distance for supply / demand curves.

DTW is a more natural dissimilarity for matched market curves than 1-D
Wasserstein because:

- Step curves live on different supports `[0, E_t]` (a different total
  cleared energy per auction), and DTW handles non-uniform support by
  aligning points before measuring distance.
- DTW penalises shape differences along the curve, not only marginal
  distributional differences.

`curve_dtw_distance` wraps `dtaidistance`. `pairwise_dtw_matrix` is the
matrix variant used when building K-medoids prototypes.

The default is the multivariate form: each step is a `(price, qty)` point,
so both axes contribute to the alignment. The qty axis is optionally
rescaled to the price scale via `qty_weight`.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from dtaidistance import dtw as _dtw
    from dtaidistance import dtw_ndim as _dtw_ndim
except ImportError as e:  # pragma: no cover - import-time only
    raise ImportError(
        "DTW metric requires `dtaidistance`. Install with "
        "`uv pip install dtaidistance`."
    ) from e


def _normalise_qty(qty: np.ndarray) -> np.ndarray:
    qty = np.asarray(qty, dtype=np.float64)
    total = float(qty[-1]) if len(qty) else 1.0
    if total <= 0:
        return np.zeros_like(qty)
    return np.clip(qty / total, 0.0, 1.0)


def curve_dtw_distance(
    qty_a: np.ndarray,
    price_a: np.ndarray,
    qty_b: np.ndarray,
    price_b: np.ndarray,
    *,
    multivariate: bool = True,
    qty_weight: float = 50.0,
    normalise_qty: bool = True,
    window: int | None = None,
) -> float:
    """DTW between two step curves.

    Parameters
    ----------
    qty_a, price_a, qty_b, price_b:
        Cumulative-quantity and price arrays of the two step curves.
    multivariate:
        If True (default), each step is a 2-D point ``(price, qty * qty_weight)``
        and DTW is computed in the 2-D plane via `dtaidistance.dtw_ndim`.
        If False, DTW is computed on the price sequence alone.
    qty_weight:
        Scale applied to the (normalised) quantity axis so it contributes
        meaningfully to the alignment cost. The default 50.0 roughly matches
        the typical price-axis range when prices are in $/MWh.
    normalise_qty:
        If True, divide each qty array by its total before DTW so curves
        on different `[0, E_t]` supports become comparable.
    window:
        Optional Sakoe-Chiba band width. None = unconstrained DTW.
    """
    pa = np.asarray(price_a, dtype=np.float64)
    pb = np.asarray(price_b, dtype=np.float64)
    if not multivariate:
        kwargs = {} if window is None else {"window": int(window)}
        return float(_dtw.distance(pa, pb, **kwargs))
    qa = _normalise_qty(qty_a) if normalise_qty else np.asarray(qty_a, dtype=np.float64)
    qb = _normalise_qty(qty_b) if normalise_qty else np.asarray(qty_b, dtype=np.float64)
    a = np.column_stack([pa, qa * float(qty_weight)])
    b = np.column_stack([pb, qb * float(qty_weight)])
    kwargs = {} if window is None else {"window": int(window)}
    return float(_dtw_ndim.distance(a, b, **kwargs))


def pairwise_dtw_matrix(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    multivariate: bool = True,
    qty_weight: float = 50.0,
    normalise_qty: bool = True,
    window: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Symmetric matrix of pairwise DTW distances over a list of curves.

    Each entry of `curves` is a tuple ``(qty_array, price_array)``. Returns
    an ``(n, n)`` float array with zero on the diagonal.
    """
    n = len(curves)
    D = np.zeros((n, n), dtype=np.float64)
    iterator = range(n)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="pairwise DTW")
        except ImportError:
            pass
    for i in iterator:
        qa, pa = curves[i]
        for j in range(i + 1, n):
            qb, pb = curves[j]
            d = curve_dtw_distance(
                qa, pa, qb, pb,
                multivariate=multivariate,
                qty_weight=qty_weight,
                normalise_qty=normalise_qty,
                window=window,
            )
            D[i, j] = D[j, i] = d
    return D


def curve_dtw_to_set(
    target_qty: np.ndarray,
    target_price: np.ndarray,
    medoids: Sequence[tuple[np.ndarray, np.ndarray]],
    **kwargs,
) -> np.ndarray:
    """Distances from one curve to every medoid. Returns a (K,) vector.

    Used to (a) build the K-medoids embedding and (b) score predictions.
    """
    return np.array(
        [curve_dtw_distance(target_qty, target_price, mq, mp, **kwargs)
         for mq, mp in medoids],
        dtype=np.float64,
    )
