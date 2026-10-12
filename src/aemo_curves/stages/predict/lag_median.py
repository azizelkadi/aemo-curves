"""Weighted pointwise median of lagged curves, a forecast without training.

A step curve is a pair (qty, price). qty holds the right edge of each step
(cumulative MWh, increasing) and price holds the step price. Step k covers
(qty[k-1], qty[k]], and outside [qty[0], qty[-1]] the curve keeps its first or
last price. This is the convention of the official HC-MAE code, and
`_step_eval` is imported from it, so both read a curve the same way.

At each volume, the weighted median of K step curves depends only on their K
prices at that volume, so it is constant between two neighbouring edges of the
union of the K sets of edges. `weighted_median_curve` reads the candidates at
that union of edges, so the result is exact (no volume grid error). With an
odd total weight, the median at each volume is one of the K prices. The median
is built from order statistics of the K prices, so the median of
non-decreasing (supply) curves does not decrease, and the median of
non-increasing (demand, residual) curves does not increase.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.evaluate.hc_mae import _step_eval

RANGE_RULES = ("median", "median_minstart", "ref")
CAP_MAPS = (None, "always")
# A supply step priced at or above (cap of its date - CAP_TOL) is a cap step.
CAP_TOL = 0.5


def _output_range(firsts: np.ndarray, lasts: np.ndarray, range_rule: str, ref: int) -> tuple[float, float]:
    if range_rule == "median":
        return float(np.median(firsts)), float(np.median(lasts))
    if range_rule == "median_minstart":
        return float(np.min(firsts)), float(np.median(lasts))
    if range_rule == "ref":
        return float(firsts[ref]), float(lasts[ref])
    raise ValueError(f"unknown range_rule {range_rule!r}, expected one of {RANGE_RULES}")


def _weighted_median_columns(vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted median of each column of `vals` (shape K x G).

    When the cumulative weight hits exactly half of the total, the lower and
    upper weighted medians are averaged, so equal weights give the usual
    median. With an odd total of integer weights this never happens.
    """
    order = np.argsort(vals, axis=0, kind="stable")
    sorted_vals = np.take_along_axis(vals, order, axis=0)
    cum = np.cumsum(weights[order], axis=0)
    half = 0.5 * weights.sum()
    eps = 1e-9 * weights.sum()
    cols = np.arange(vals.shape[1])
    i_lo = np.argmax(cum >= half - eps, axis=0)
    tie = np.abs(cum[i_lo, cols] - half) <= eps
    i_hi = np.where(tie, np.minimum(i_lo + 1, vals.shape[0] - 1), i_lo)
    return 0.5 * (sorted_vals[i_lo, cols] + sorted_vals[i_hi, cols])


def _merge_equal_steps(qty: np.ndarray, price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge each run of equal prices into one step.

    The first edge is always kept because it sets the start of the curve's
    volume range, and HC-MAE reads that range.
    """
    n = len(qty)
    if n <= 2:
        return qty, price
    keep = np.ones(n, dtype=bool)
    keep[1:-1] = price[1:-1] != price[2:]
    return qty[keep], price[keep]


def weighted_median_curve(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: Sequence[int],
    range_rule: str = "median",
    ref: int = 0,
    merge_equal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise weighted median of step curves along the volume axis.

    Args:
        curves: the candidate step curves, each a (curve_qty, curve_price) pair.
        weights: one positive weight per curve. A curve with weight w counts
            w times in the median at each volume.
        range_rule: sets the volume range [lo, hi] of the output curve.
            "median": lo is the median of the curves' first edges and hi the
            median of their last edges. "median_minstart": lo is the smallest
            first edge and hi the median of the last edges, so the output also
            holds the steps below the median first edge. In both rules each
            curve counts once in the median of the edges, whatever its weight.
            "ref": the range of curves[ref].
        ref: index of the reference curve for range_rule="ref".
        merge_equal: merge each run of equal prices into one step. The merge
            keeps the curve's price at every volume, but it removes edges, so
            it changes the result of an operation that moves edges (for
            example `volume_shift.shift_keep_first_edge`).

    Returns:
        (qty, price) of the median curve. Its first edge is lo and its last
        edge is hi. Its edges are lo, hi and the candidate edges strictly
        between them, after merging runs of equal price when merge_equal is
        True.

    Raises:
        ValueError: on no curves, a weight count that does not match the
            curves, a weight that is not positive, an empty curve, or an
            unknown range_rule.
    """
    if len(curves) == 0:
        raise ValueError("weighted_median_curve needs at least one curve")
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(curves),):
        raise ValueError(f"got {len(curves)} curves but {w.size} weights")
    if not np.all(w > 0):
        raise ValueError("every weight must be positive")
    qs = [np.asarray(c[0], dtype=float) for c in curves]
    ps = [np.asarray(c[1], dtype=float) for c in curves]
    if any(len(q) == 0 or len(q) != len(p) for q, p in zip(qs, ps, strict=True)):
        raise ValueError("every curve needs the same number of qty and price values, at least one")
    firsts = np.array([q[0] for q in qs])
    lasts = np.array([q[-1] for q in qs])
    lo, hi = _output_range(firsts, lasts, range_rule, ref)
    if hi <= lo:
        grid = np.array([lo])
    else:
        allq = np.concatenate(qs)
        grid = np.unique(np.concatenate(([lo, hi], allq[(allq > lo) & (allq < hi)])))
    vals = np.vstack([_step_eval(q, p, grid) for q, p in zip(qs, ps, strict=True)])
    price = _weighted_median_columns(vals, w)
    return _merge_equal_steps(grid, price) if merge_equal else (grid, price)


def cap_map_curve(qty, price, cap_from: float, cap_to: float) -> tuple[np.ndarray, np.ndarray]:
    """Move a supply curve's cap steps from the cap of its own date to another cap.

    cap_from is the price cap of the curve's own trading date (in the
    `lag_wmedian` strategy: the highest offer price over all supply curves of
    that date). Every step priced at or above cap_from - 0.5 gets the price
    cap_to. When cap_to is below cap_from, every price above cap_to is also
    cut to cap_to, so the curve stays non-decreasing and at or below cap_to.
    The volumes do not change.

    Returns:
        (qty, price). When cap_from equals cap_to, the input arrays are
        returned as they are. Otherwise price is a new array.
    """
    q = np.asarray(qty, dtype=float)
    p = np.asarray(price, dtype=float)
    if cap_from == cap_to:
        return q, p
    out = p.copy()
    out[p >= cap_from - CAP_TOL] = cap_to
    if cap_to < cap_from:
        np.minimum(out, cap_to, out=out)
    return q, out


def _day_numbers(dates: pd.Series) -> np.ndarray:
    """Trading dates as whole days since 1970-01-01."""
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)


def _parse_lags(lags, horizon: int) -> list[tuple[int, int]]:
    pairs = []
    for item in lags:
        if len(item) != 2:
            raise ValueError(f"each entry of lags must be [lag_days, weight], got {item!r}")
        lag, weight = item
        if int(lag) != lag or int(weight) != weight:
            raise ValueError(f"lag_days and weight must be integers, got {item!r}")
        if lag < horizon:
            raise ValueError(f"lag {lag} is below the forecast horizon of {horizon} days")
        if weight < 1:
            raise ValueError(f"weight of lag {lag} must be at least 1, got {weight}")
        pairs.append((int(lag), int(weight)))
    if not pairs:
        raise ValueError("lags must not be empty")
    if len({lag for lag, _ in pairs}) != len(pairs):
        raise ValueError(f"lags must be distinct, got {[lag for lag, _ in pairs]}")
    return pairs


class _History:
    """Curves of the history table, looked up by (day number, interval)."""

    def __init__(self, hist: pd.DataFrame) -> None:
        day = _day_numbers(hist["Trading Date"])
        interval = hist["Interval Number"].to_numpy(dtype=np.int64)
        if interval.min() < 1:
            raise ValueError("Interval Number must be at least 1")
        self.day0 = int(day.min())
        n_days = int(day.max()) - self.day0 + 1
        self.row = np.full((n_days, int(interval.max()) + 1), -1, dtype=np.int64)
        self.row[day - self.day0, interval] = np.arange(len(hist))
        if (self.row >= 0).sum() != len(hist):
            raise ValueError("the history table has more than one curve for some (date, interval)")
        self.qty = hist["curve_qty"].to_numpy()
        self.price = hist["curve_price"].to_numpy()
        self._day = day

    def get(self, day: int, interval: int) -> tuple[np.ndarray, np.ndarray] | None:
        d = day - self.day0
        if d < 0 or d >= self.row.shape[0] or interval < 1 or interval >= self.row.shape[1]:
            return None
        r = self.row[d, interval]
        if r < 0:
            return None
        return self.qty[r], self.price[r]

    def caps(self) -> np.ndarray:
        """Highest price over all curves of each day (minus infinity on a day with no curve)."""
        caps = np.full(self.row.shape[0], -np.inf)
        row_max = np.array([np.max(p) if len(p) else -np.inf for p in self.price])
        np.maximum.at(caps, self._day - self.day0, row_max)
        return caps


@register("predict", "lag_wmedian")
def lag_wmedian(
    ctx: PredictContext,
    lags: Sequence[Sequence[int]],
    range_rule: str = "median",
    cap_map: str | None = None,
) -> pd.DataFrame:
    """Forecast each auction by the weighted pointwise median of its lagged curves.

    For the target (D, interval) the candidates are the curves of the same
    interval on the dates D - lag_days, read from train plus test. Every lag
    must be at least `ctx.lag_days`, so the forecast uses only curves dated
    D - ctx.lag_days or earlier. There is no training.

    Args:
        lags: a list of [lag_days, weight] pairs. The first pair is the
            reference curve: it sets the volume range for range_rule="ref",
            and a target without it gets no forecast (as in the lag baseline).
            A missing curve at any other lag is dropped with its weight.
        range_rule: one of RANGE_RULES, see `weighted_median_curve`.
        cap_map: None, or "always" (supply only). With "always", the cap of a
            date is the highest offer price over all supply curves of that
            date, and the cap known for target D is the cap of
            D - ctx.lag_days. Each candidate goes through `cap_map_curve`
            from the cap of its own date to the known cap before the median.
            When the date D - ctx.lag_days has no curve, the known cap is not
            defined and the candidates are used as they are.

    Returns:
        The predictions frame, plus `n_lags_used` (the number of candidate
        curves in the median of each row).
    """
    pairs = _parse_lags(lags, int(ctx.lag_days))
    if range_rule not in RANGE_RULES:
        raise ValueError(f"unknown range_rule {range_rule!r}, expected one of {RANGE_RULES}")
    if cap_map not in CAP_MAPS:
        raise ValueError(f"unknown cap_map {cap_map!r}, expected one of {CAP_MAPS}")
    if cap_map is not None and ctx.side != "supply":
        raise ValueError(f"cap_map={cap_map!r} is for the supply side, got side {ctx.side!r}")

    hist = _History(ctx.historical)
    caps = hist.caps() if cap_map is not None else None
    test = ctx.test.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
    days = _day_numbers(test["Trading Date"])
    intervals = test["Interval Number"].to_numpy(dtype=np.int64)
    rows = []
    for j in range(len(test)):
        day, interval = int(days[j]), int(intervals[j])
        if hist.get(day - pairs[0][0], interval) is None:
            continue
        known_cap = None
        if caps is not None:
            k = day - int(ctx.lag_days) - hist.day0
            if 0 <= k < len(caps) and np.isfinite(caps[k]):
                known_cap = float(caps[k])
        curves, weights = [], []
        for lag, weight in pairs:
            curve = hist.get(day - lag, interval)
            if curve is None:
                continue
            if known_cap is not None:
                curve = cap_map_curve(*curve, float(caps[day - lag - hist.day0]), known_cap)
            curves.append(curve)
            weights.append(weight)
        pred_qty, pred_price = weighted_median_curve(curves, weights, range_rule=range_rule, ref=0)
        rows.append(
            {
                "Trading Date": test["Trading Date"].iat[j],
                "Interval Number": interval,
                "true_curve_qty": test["curve_qty"].iat[j],
                "true_curve_price": test["curve_price"].iat[j],
                "pred_curve_qty": pred_qty,
                "pred_curve_price": pred_price,
                "n_lags_used": len(curves),
            }
        )
    return pd.DataFrame(rows)
