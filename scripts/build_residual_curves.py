"""Build the *residual* curve tables from the cached demand and supply curves.

The residual curve of one auction is the price spread on the volume axis:

    R(v) = P_demand(v) - P_supply(v)

Because the demand price is non-increasing in volume and the supply price is
non-decreasing, ``R`` is a monotone **non-increasing** step curve (falling,
demand-like). It becomes negative past the zero-crossing (the market-clearing
volume). Negatives are safe end to end - the curves / interp / v_grid / HC-MAE /
contract code carries no positivity assumptions.

Support: we intersect the two curves on ``[0, min(Vd_max, Vs_max)]`` (the shared
volume range). In this market ``min == Vd_max`` in every auction, so the residual
lives on the demand support.

This script reads the CACHED curve tables (no raw-CSV reprocessing):

    artifacts/curves/{train,test}_{demand,supply}.pkl

inner-joins them on ``(Trading Date, Interval Number)``, builds one residual
curve per pair, and writes them as another curve table (side ``residual``):

    artifacts/curves/{train,test}_residual.pkl

Emitted columns match the curves-table schema the downstream stages expect:
``curve_qty``, ``curve_price``, ``total_quantity`` (= Vd_max), ``mean_price``
(= mean of R, typically negative), ``n_offers`` (= number of grid points).

Usage:
    uv run python scripts/build_residual_curves.py
    uv run python scripts/build_residual_curves.py --splits train
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from aemo_curves.stages.curves import load_curve_table, save_curve_table
from aemo_curves.stages.evaluate.hc_mae import _step_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

_KEYS = ["Trading Date", "Interval Number"]
_FLOAT_TOL = 1e-6


def _residual_curve(
    demand_qty: np.ndarray,
    demand_price: np.ndarray,
    supply_qty: np.ndarray,
    supply_price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (V, R) for one auction, or None if fewer than 2 usable grid points.

    ``V`` is the union of both curves' step boundaries clipped to the shared
    support ``[0, min(Vd_max, Vs_max)]``; ``R = P_demand(V) - P_supply(V)`` using
    the same step-evaluation convention as HC-MAE.
    """
    dq = np.asarray(demand_qty, dtype=float)
    dp = np.asarray(demand_price, dtype=float)
    sq = np.asarray(supply_qty, dtype=float)
    sp = np.asarray(supply_price, dtype=float)

    v_max = min(float(dq[-1]), float(sq[-1]))
    v = np.unique(np.concatenate([dq, sq]))
    v = v[v <= v_max + _FLOAT_TOL]
    # Guarantee the shared cap and the origin are represented so the residual
    # spans the full [0, v_max] support both step functions agree on.
    v = np.unique(np.concatenate([[0.0], v, [v_max]]))
    v = v[(v >= -_FLOAT_TOL) & (v <= v_max + _FLOAT_TOL)]
    if len(v) < 2:
        return None

    r = _step_eval(dq, dp, v) - _step_eval(sq, sp, v)
    return v, r


def build_residual_table(demand: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    """Inner-join demand & supply curve tables on the keys and build residuals."""
    merged = demand.merge(supply, on=_KEYS, suffixes=("_d", "_s"), how="inner")
    log.info(
        "  inner join: demand=%d supply=%d -> matched=%d (demand orphans=%d, supply orphans=%d)",
        len(demand),
        len(supply),
        len(merged),
        len(demand) - len(merged),
        len(supply) - len(merged),
    )
    rows = []
    dropped = 0
    # Column-wise access by name (the keys contain spaces, so itertuples mangles them).
    dates = merged["Trading Date"].to_numpy()
    intervals = merged["Interval Number"].to_numpy()
    dq_col = merged["curve_qty_d"].to_numpy()
    dp_col = merged["curve_price_d"].to_numpy()
    sq_col = merged["curve_qty_s"].to_numpy()
    sp_col = merged["curve_price_s"].to_numpy()
    for i in range(len(merged)):
        built = _residual_curve(dq_col[i], dp_col[i], sq_col[i], sp_col[i])
        if built is None:
            dropped += 1
            continue
        v, res = built
        rows.append(
            {
                "Trading Date": dates[i],
                "Interval Number": int(intervals[i]),
                "curve_qty": v,
                "curve_price": res,
                "n_offers": int(len(v)),
                "total_quantity": float(v[-1]),  # = Vd_max (= shared cap)
                "mean_price": float(res.mean()),  # spread; negative on average is fine
            }
        )
    if dropped:
        log.info("  dropped %d pairs with < 2 usable grid points", dropped)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(_KEYS).reset_index(drop=True)

    # Fail loud if the residual invariants are violated (schema slip guard):
    # curve_qty strictly increasing (it is np.unique output), residual R
    # non-increasing (demand price falls, supply price rises).
    for rec in out.itertuples(index=False):
        q = np.asarray(rec.curve_qty, dtype=float)
        p = np.asarray(rec.curve_price, dtype=float)
        assert np.all(np.diff(q) > 0.0), "curve_qty not strictly increasing"
        assert np.all(np.diff(p) <= _FLOAT_TOL), "residual price not non-increasing"
    return out


def build_and_save(split: str) -> None:
    log.info("Loading cached demand / supply curves for %s", split)
    demand = load_curve_table(split, "demand")
    supply = load_curve_table(split, "supply")
    t0 = time.time()
    table = build_residual_table(demand, supply)
    log.info("  built %d residual curves in %.1fs", len(table), time.time() - t0)
    if not table.empty:
        neg_frac = float(
            np.mean([bool((np.asarray(p, dtype=float) < 0).any()) for p in table["curve_price"]])
        )
        log.info("  fraction of curves with any negative residual: %.3f", neg_frac)
        log.info("  mean of per-curve mean_price (spread): %.2f", table["mean_price"].mean())
    path = save_curve_table(table, name=split, side="residual")
    log.info("Saved %d residual curves -> %s", len(table), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test residual curve tables")
    parser.add_argument("--splits", nargs="*", choices=["train", "test"], default=["train", "test"])
    args = parser.parse_args()
    for split in args.splits:
        build_and_save(split)


if __name__ == "__main__":
    main()
