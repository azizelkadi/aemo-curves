"""Market-clearing evaluation: how far a predicted curve moves the clearing point.

Every auction clears where the demand (bid) curve crosses the supply (offer)
curve, giving a clearing **volume** ``V*`` (MWh) and a clearing **price** ``P*``
($/MWh). This module measures how far a *predicted* curve moves that clearing
point away from the true one, holding the *other* side at its true curve:

    * demand side:   cross the (true / predicted) demand curve with the true
      supply curve.
    * supply side:   cross the true demand curve with the (true / predicted)
      supply curve.
    * residual side: the residual ``R = P_demand - P_supply`` already encodes the
      spread, so the clearing volume is its own zero-crossing.

The clearing price is always read off the **true supply** curve at ``V*`` (the
uniform-price, marginal-offer reading), so a demand-side move and a supply-side
move are priced on the same ruler.

These numbers are reported *alongside* the ranking metric (HC-MAE); they never
sort the leaderboard. Volumes are MWh, prices $/MWh.

Step convention matches HC-MAE: step ``k`` covers ``(qty[k-1], qty[k]]`` with
price ``price[k]``; ``_step_eval`` (reused from ``hc_mae``) reads it. The crossing
grid mirrors ``_residual_curve`` in ``scripts/build_residual_curves.py``.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from aemo_curves.stages.evaluate.hc_mae import _step_eval

log = logging.getLogger(__name__)

_TOL = 1e-6

# Degenerate crossings (a curve with < 2 usable grid points) are clamped to
# V*=0.0. They are tallied so `attach_clearing` can log a single summary line.
_DEGENERATE_COUNT = 0


def _clearing_volume(
    a_qty: np.ndarray,
    a_price: np.ndarray,
    b_qty: np.ndarray | None = None,
    b_price: np.ndarray | None = None,
) -> float:
    """Clearing volume ``V*`` where falling curve A crosses rising curve B.

    ``A`` is a demand (or residual) curve, non-increasing in volume; ``B`` is a
    supply curve, non-decreasing. When ``b_qty is None`` the crossing is taken
    against the constant-zero curve, so ``V*`` is the zero-crossing of ``A``
    (the residual side).

    The grid is the union of both curves' step edges clipped to the shared
    support ``[0, v_max]`` with ``v_max = min(A_max, B_max)`` (for the residual,
    only its own edges on ``[0, A_max]``), mirroring ``_residual_curve``. Returns
    ``0.0`` on a degenerate curve or when supply opens above demand (no trade),
    and ``v_max`` when demand exceeds supply everywhere (clears at the shared
    cap). Ties (``R == 0``) count as clearing.
    """
    a_qty = np.asarray(a_qty, dtype=float)
    a_price = np.asarray(a_price, dtype=float)
    residual = b_qty is None
    if residual:
        v_max = float(a_qty[-1])
        v = np.unique(a_qty)
        v = v[(v >= -_TOL) & (v <= v_max + _TOL)]
    else:
        b_qty = np.asarray(b_qty, dtype=float)
        b_price = np.asarray(b_price, dtype=float)
        v_max = min(float(a_qty[-1]), float(b_qty[-1]))
        v = np.unique(np.concatenate([[0.0], a_qty, b_qty, [v_max]]))
        v = v[(v >= -_TOL) & (v <= v_max + _TOL)]

    if len(v) < 2 or len(a_qty) < 2:
        global _DEGENERATE_COUNT
        _DEGENERATE_COUNT += 1
        return 0.0

    if residual:
        r = _step_eval(a_qty, a_price, v)
    else:
        r = _step_eval(a_qty, a_price, v) - _step_eval(b_qty, b_price, v)

    if r[0] < 0:
        return 0.0  # no trade: supply opens above demand
    if not (r < 0).any():
        return float(v_max)  # excess demand everywhere: clears at the shared cap
    k = int(np.argmax(r < 0))  # first edge where the falling curve dips below
    return float(v[k - 1])  # last edge of the all-non-negative prefix


def _clearing_price(supply_qty: np.ndarray, supply_price: np.ndarray, v: float) -> float:
    """Marginal-offer clearing price: the true supply price at volume ``v``."""
    return float(
        _step_eval(
            np.asarray(supply_qty, dtype=float),
            np.asarray(supply_price, dtype=float),
            np.asarray([v], dtype=float),
        )[0]
    )


def _row_keys(df: pd.DataFrame) -> list[tuple]:
    """``(normalised Trading Date, int Interval Number)`` per row.

    Date normalisation matches ``attach_baseline`` so a frame carrying a
    ``datetime64`` date aligns with one carrying a Python ``date``. Both the
    lookup tables and the query frame are keyed through this one helper, so the
    tuple element types (``pd.Timestamp``, ``int``) always match.
    """
    dates = pd.to_datetime(df["Trading Date"]).dt.normalize()
    intervals = df["Interval Number"].astype(int)
    return [(d, int(i)) for d, i in zip(dates, intervals, strict=True)]


def _curve_lookup(
    table: pd.DataFrame,
    qty_col: str = "curve_qty",
    price_col: str = "curve_price",
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Map ``(date, interval) -> (qty, price)`` for a curve or predictions frame.

    Curve tables use ``curve_qty``/``curve_price``; a predictions frame carries
    the predicted curve under ``pred_curve_qty``/``pred_curve_price``.
    """
    keys = _row_keys(table)
    return {
        k: (q, p)
        for k, q, p in zip(keys, table[qty_col], table[price_col], strict=True)
    }


@lru_cache(maxsize=None)
def _true_curve_lookup(curve_side: str) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Lookup of the true ``curve_side`` curve over the union of train+test.

    Memoized per process: a group run scores dozens of experiments and every
    one needs the same true supply (and, for supply experiments, demand) curves.
    Loading the ~50 MB curve tables and rebuilding the ~52k-entry dict on each
    experiment peaked memory past the container limit and was pure waste; here
    the dict is built once and shared read-only across the whole run. The curve
    tables are static within a run, and a curve rebuild invalidates the result
    cache (see ``runner/cache.py``), so a stale lookup cannot survive.
    """
    from aemo_curves.stages.curves import load_curve_table

    table = pd.concat(
        [load_curve_table("train", curve_side), load_curve_table("test", curve_side)],
        ignore_index=True,
    )
    return _curve_lookup(table)


def _require_keys(keys: list[tuple], lut: dict, what: str) -> None:
    """Fail loud if any prediction key is absent from a required lookup."""
    missing = [k for k in keys if k not in lut]
    if missing:
        raise ValueError(
            f"clearing: {len(missing)} of {len(keys)} prediction rows have no matching "
            f"{what}; first missing (date, interval) = {missing[0]}"
        )


def attach_clearing(
    scored: pd.DataFrame,
    baseline_preds: pd.DataFrame | None,
    *,
    side: str,
    split: str,
) -> pd.DataFrame:
    """Add per-row market-clearing volume/price columns to a scored frame.

    The in-frame ``true_curve`` / ``pred_curve`` are that ``side``'s curves. The
    true opposite curve (supply side only) and the true-supply price ruler are
    loaded from the cached curve tables and matched by ``(date, interval)``; a
    missing ruler/partner match is a hard error (never silently dropped). When
    ``baseline_preds`` is given, its predicted curve is crossed the same way (same
    true partner, same true-supply ruler) so the leaderboard can report a
    model-vs-baseline gap. ``split`` is accepted for interface symmetry; the union
    of the train+test curve tables covers every split.
    """
    out = scored.copy()
    keys = _row_keys(out)

    supply_lut = _true_curve_lookup("supply")  # price ruler (and demand-side partner)
    demand_lut = _true_curve_lookup("demand") if side == "supply" else None  # supply partner

    _require_keys(keys, supply_lut, "true-supply curve (ruler)")
    if side == "supply":
        _require_keys(keys, demand_lut, "true-demand curve (partner)")

    baseline_lut = (
        _curve_lookup(baseline_preds, "pred_curve_qty", "pred_curve_price")
        if baseline_preds is not None
        else None
    )
    have_base = baseline_lut is not None

    true_q = out["true_curve_qty"].to_list()
    true_p = out["true_curve_price"].to_list()
    pred_q = out["pred_curve_qty"].to_list()
    pred_p = out["pred_curve_price"].to_list()

    n = len(out)
    vol_true = np.empty(n)
    vol_pred = np.empty(n)
    price_true = np.empty(n)
    price_pred = np.empty(n)
    base_vol = np.full(n, np.nan)
    base_price = np.full(n, np.nan)

    before = _DEGENERATE_COUNT
    for i, key in enumerate(keys):
        sup_q, sup_p = supply_lut[key]  # true supply = price ruler
        if side == "demand":
            # A = demand curve (in-frame), B = true supply.
            vt = _clearing_volume(true_q[i], true_p[i], sup_q, sup_p)
            vp = _clearing_volume(pred_q[i], pred_p[i], sup_q, sup_p)
            bvp = (
                _clearing_volume(*baseline_lut[key], sup_q, sup_p)
                if have_base and key in baseline_lut
                else None
            )
        elif side == "supply":
            dem_q, dem_p = demand_lut[key]  # true demand = partner A
            # A = true demand, B = supply curve (in-frame).
            vt = _clearing_volume(dem_q, dem_p, true_q[i], true_p[i])
            vp = _clearing_volume(dem_q, dem_p, pred_q[i], pred_p[i])
            if have_base and key in baseline_lut:
                b_q, b_p = baseline_lut[key]
                bvp = _clearing_volume(dem_q, dem_p, b_q, b_p)
            else:
                bvp = None
        else:  # residual: the in-frame curve is the residual; cross against zero.
            vt = _clearing_volume(true_q[i], true_p[i])
            vp = _clearing_volume(pred_q[i], pred_p[i])
            bvp = (
                _clearing_volume(*baseline_lut[key])
                if have_base and key in baseline_lut
                else None
            )

        vol_true[i] = vt
        vol_pred[i] = vp
        price_true[i] = _clearing_price(sup_q, sup_p, vt)
        price_pred[i] = _clearing_price(sup_q, sup_p, vp)
        if bvp is not None:
            base_vol[i] = bvp
            base_price[i] = _clearing_price(sup_q, sup_p, bvp)

    out["clearing_vol_true"] = vol_true
    out["clearing_vol_pred"] = vol_pred
    out["clearing_price_true"] = price_true
    out["clearing_price_pred"] = price_pred
    out["clearing_vol_ae"] = np.abs(vol_pred - vol_true)
    out["clearing_price_ae"] = np.abs(price_pred - price_true)

    if have_base:
        out["baseline_clearing_vol_pred"] = base_vol
        out["baseline_clearing_price_pred"] = base_price
        out["baseline_clearing_vol_ae"] = np.abs(base_vol - vol_true)
        out["baseline_clearing_price_ae"] = np.abs(base_price - price_true)

    degenerate = _DEGENERATE_COUNT - before
    if degenerate:
        log.warning("clearing: %d degenerate crossing(s) clamped to V*=0.0", degenerate)
    return out


def clearing_summary(scored: pd.DataFrame) -> dict[str, float]:
    """Summary market-clearing errors (reported alongside the ranking metric)."""
    vol_ae = scored["clearing_vol_ae"]
    price_ae = scored["clearing_price_ae"]
    stats = {
        "clearing_vol_mae": float(vol_ae.mean()),
        "clearing_vol_medae": float(vol_ae.median()),
        "clearing_vol_bias": float(
            (scored["clearing_vol_pred"] - scored["clearing_vol_true"]).mean()
        ),
        "clearing_price_mae": float(price_ae.mean()),
        "clearing_price_medae": float(price_ae.median()),
        "clearing_price_bias": float(
            (scored["clearing_price_pred"] - scored["clearing_price_true"]).mean()
        ),
    }
    if "baseline_clearing_vol_ae" in scored.columns:
        bva = scored["baseline_clearing_vol_ae"]
        bpa = scored["baseline_clearing_price_ae"]
        # Compare only where a baseline row matched: NaN < x is False and would
        # otherwise silently deflate the win rate (as in `summary_stats`).
        v_paired = bva.notna()
        p_paired = bpa.notna()
        stats["baseline_clearing_vol_mae"] = float(bva.mean())
        stats["baseline_clearing_price_mae"] = float(bpa.mean())
        stats["clearing_vol_delta_vs_baseline"] = (
            stats["clearing_vol_mae"] - stats["baseline_clearing_vol_mae"]
        )
        stats["clearing_price_delta_vs_baseline"] = (
            stats["clearing_price_mae"] - stats["baseline_clearing_price_mae"]
        )
        stats["clearing_vol_win_rate"] = float((vol_ae[v_paired] < bva[v_paired]).mean())
        stats["clearing_price_win_rate"] = float((price_ae[p_paired] < bpa[p_paired]).mean())
    return stats
