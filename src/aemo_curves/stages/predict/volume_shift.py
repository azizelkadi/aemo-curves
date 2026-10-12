"""Volume-shift forecast (demand side): a median shape moved to a forecast cliff volume.

For the target auction (D, i) and the horizon h = ctx.lag_days:

1. Shape: the pointwise median of the curves of interval i on the dates D - k
   for k in `shape_lags` (`weighted_median_curve` from `lag_median.py`, every
   weight 1).
2. Cliff volume V of a demand curve: the right edge of the last step priced at
   or above -999 $/MWh, so the volume of the bids above the -1,000 $/MWh price
   floor (`cliff_volume`).
3. Volume model: gradient-boosted trees with the absolute error loss (so the
   forecast is a median) predict V(D, i) - V(D - h, i) from the features of
   `volume_features`. The model is refit on the first day T of every test month
   with the rows dated on or before T - h.
4. Shift: the shape moves along the volume axis by V_hat - V(shape), with its
   first and last edges fixed (`shift_keep_first_edge`).

Every input of the row (D, i) comes from curves dated D - h or earlier, the
calendar of D and the experiment's exogenous provider, which the runner builds
with the same horizon.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from aemo_curves.core.registry import register
from aemo_curves.exogenous.holidays import wa_holiday_flag_names, wa_holiday_flags
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict._orchestration import _monthly_refit_dates
from aemo_curves.stages.predict.lag_median import RANGE_RULES, weighted_median_curve

log = logging.getLogger(__name__)

# The demand price floor is -1,000 $/MWh, so a step priced at or above -999 is above it.
CLIFF_MIN_PRICE = -999.0
N_VOLUME_LAGS = 28
TRAILING_WINDOWS = (7, 14, 28)
DAY_MEAN_WINDOW = 28


def cliff_volume(qty, price) -> float:
    """Right edge of the last step priced at or above CLIFF_MIN_PRICE, 0 when there is none."""
    q = np.asarray(qty, dtype=float)
    above = np.asarray(price, dtype=float) >= CLIFF_MIN_PRICE
    return float(q[above].max()) if above.any() else 0.0


def shift_keep_first_edge(qty, price, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Move every step edge of a curve except the first and the last by `delta` MWh.

    The moved edges are cut at the last edge and lifted to the first edge, so
    the volume range of the curve does not change. On a demand curve the first
    step holds the highest-priced bids, and with delta > 0 the extra volume
    goes in after it. A step that ends with zero width is dropped, so when
    several edges land on the first edge, the first step takes the price of the
    last of them. A first edge at or below 0 is dropped.
    """
    q = np.asarray(qty, dtype=float)
    p = np.asarray(price, dtype=float)
    if len(q) <= 1:
        return q, p
    moved = q.copy()
    moved[1:] = np.minimum(q[1:] + delta, q[-1])
    moved[-1] = q[-1]
    moved = np.maximum.accumulate(moved)
    keep = np.append(moved[1:] > moved[:-1], True)
    keep[0] = moved[0] > 0 and moved[0] < moved[1]
    return moved[keep], p[keep]


def _day_numbers(dates: pd.Series) -> np.ndarray:
    """Trading dates as whole days since 1970-01-01."""
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)


class _LagIndex:
    """Row of the same interval k days earlier, for every row of a table."""

    def __init__(self, day: np.ndarray, interval: np.ndarray) -> None:
        self.day = day
        self.interval = interval
        self.day0 = int(day.min())
        self.row = np.full((int(day.max()) - self.day0 + 1, int(interval.max()) + 1), -1,
                           dtype=np.int64)
        self.row[day - self.day0, interval] = np.arange(len(day))
        if (self.row >= 0).sum() != len(day):
            raise ValueError("the history table has more than one curve for some (date, interval)")

    def rows(self, day: np.ndarray, interval: np.ndarray) -> np.ndarray:
        """Row of each (day, interval) pair, -1 where the table has no such row."""
        d = day - self.day0
        ok = (d >= 0) & (d < self.row.shape[0])
        out = np.full(len(day), -1, dtype=np.int64)
        out[ok] = self.row[d[ok], interval[ok]]
        return out

    def lag(self, k: int) -> np.ndarray:
        return self.rows(self.day - int(k), self.interval)


def _calendar_block(dates: pd.Series, interval: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    ts = pd.to_datetime(dates)
    dow = ts.dt.weekday.to_numpy().astype(float)
    uniq, inv = np.unique(ts.dt.date.to_numpy(), return_inverse=True)
    flags = np.vstack([wa_holiday_flags(d) for d in uniq])[inv]
    doy = ts.dt.dayofyear.to_numpy().astype(float)
    cols = [interval.astype(float), dow, (dow >= 5).astype(float)]
    cols += [flags[:, j] for j in range(flags.shape[1])]
    cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
             ts.dt.month.to_numpy().astype(float)]
    names = ["interval", "dow", "weekend", *wa_holiday_flag_names(), "sin_doy", "cos_doy", "month"]
    return cols, names


def volume_features(hist: pd.DataFrame, horizon: int, exog_provider=None):
    """Features and cliff volumes of the volume model, for every row of `hist`.

    With h = horizon, V(k) is the cliff volume of the same interval k days
    earlier and E(k) the last edge of that curve. The columns are, in order:
    V(k) - V(h) for k = h+1 .. h+27, V(h), the mean and the median of V over
    the lags h .. h+6, h .. h+13 and h .. h+27 (each minus V(h)), V(h) - V(h+7),
    E(h), E(h) minus the mean of E over the lags h .. h+27, the mean V of the 48
    intervals of day D - h minus V(h), that mean minus its mean over the 28 days
    that end on D - h, then the calendar of D (interval, day of week, weekend,
    the 5 WA holiday flags, sin and cos of the day of year, month), then the
    columns of `exog_provider` whose names are not calendar names.

    Returns:
        (X, names, volume, v_ref). X has NaN where an input is missing. volume
        is the cliff volume of each row and v_ref is V(h), so the target of the
        model is volume - v_ref.
    """
    h = int(horizon)
    day = _day_numbers(hist["Trading Date"])
    interval = hist["Interval Number"].to_numpy(dtype=np.int64)
    qty = hist["curve_qty"].to_numpy()
    price = hist["curve_price"].to_numpy()
    n = len(hist)
    volume = np.array([cliff_volume(q, p) for q, p in zip(qty, price, strict=True)])
    last_edge = np.array([float(q[-1]) for q in qty])
    index = _LagIndex(day, interval)

    lags = list(range(h, h + N_VOLUME_LAGS))
    v_lag = np.full((n, len(lags)), np.nan)
    e_lag = np.full((n, len(lags)), np.nan)
    for c, k in enumerate(lags):
        j = index.lag(k)
        ok = j >= 0
        v_lag[ok, c] = volume[j[ok]]
        e_lag[ok, c] = last_edge[j[ok]]
    v_ref = v_lag[:, 0]

    cols: list[np.ndarray] = []
    names: list[str] = []
    for c, k in enumerate(lags[1:], start=1):
        cols.append(v_lag[:, c] - v_ref)
        names.append(f"v_lag{k}_minus_lag{h}")
    cols.append(v_ref)
    names.append(f"v_lag{h}")
    for w in TRAILING_WINDOWS:
        seg = v_lag[:, :w]
        cols += [seg.mean(axis=1) - v_ref, np.median(seg, axis=1) - v_ref]
        names += [f"v_mean_{w}d_minus_lag{h}", f"v_median_{w}d_minus_lag{h}"]
    cols.append(v_ref - v_lag[:, 7])
    names.append(f"v_lag{h}_minus_lag{h + 7}")
    cols += [e_lag[:, 0], e_lag[:, 0] - e_lag.mean(axis=1)]
    names += [f"edge_lag{h}", f"edge_lag{h}_minus_mean"]

    # Daily mean of V over the intervals of each day. A day without rows makes
    # every window that holds it missing.
    d = day - index.day0
    n_days = index.row.shape[0]
    count = np.bincount(d, minlength=n_days)
    day_mean = np.bincount(d, weights=volume, minlength=n_days) / np.maximum(count, 1)
    cum_mean = np.concatenate(([0.0], np.cumsum(day_mean)))
    cum_missing = np.concatenate(([0], np.cumsum(count == 0)))
    src = d - h
    ok = src >= DAY_MEAN_WINDOW - 1
    ok[ok] = cum_missing[src[ok] + 1] - cum_missing[src[ok] - DAY_MEAN_WINDOW + 1] == 0
    dm = np.full(n, np.nan)
    dm_window = np.full(n, np.nan)
    dm[ok] = day_mean[src[ok]]
    dm_window[ok] = (cum_mean[src[ok] + 1] - cum_mean[src[ok] - DAY_MEAN_WINDOW + 1]) / DAY_MEAN_WINDOW
    cols += [dm - v_ref, dm - dm_window]
    names += [f"daymean_lag{h}_minus_lag{h}", f"daymean_lag{h}_minus_{DAY_MEAN_WINDOW}d_mean"]

    cal_cols, cal_names = _calendar_block(hist["Trading Date"], interval)
    cols += cal_cols
    names += cal_names

    if exog_provider is not None:
        ex_names = list(exog_provider.feature_names)
        calendar = set(cal_names)
        keep = [j for j, nm in enumerate(ex_names) if nm not in calendar]
        ex = np.full((n, len(ex_names)), np.nan)
        dates = hist["Trading Date"].to_numpy()
        for r in range(n):
            v = exog_provider(int(interval[r]), dates[r])
            if v is not None:
                ex[r] = v
        cols += [ex[:, j] for j in keep]
        names += [ex_names[j] for j in keep]

    return np.column_stack(cols), names, volume, v_ref


@register("predict", "volume_shift")
def volume_shift(
    ctx: PredictContext,
    shape_lags: Sequence[int] = tuple(range(7, 21)),
    range_rule: str = "median",
    merge_equal: bool = True,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 200,
    l2_regularization: float = 1.0,
) -> pd.DataFrame:
    """Forecast each demand auction by its median shape moved to a forecast cliff volume.

    Args:
        shape_lags: the lags in days of the curves in the median shape. Every
            lag must be at least ctx.lag_days. The first lag is the reference
            curve for range_rule="ref".
        range_rule, merge_equal: the shape settings of `weighted_median_curve`.
        learning_rate, max_iter, max_leaf_nodes, min_samples_leaf,
            l2_regularization: settings of the HistGradientBoostingRegressor
            (absolute error loss, no early stopping, random_state 0).

    Returns:
        The predictions frame, plus `refit_date`, `v_shape` (the cliff volume of
        the shape) and `v_hat` (the forecast cliff volume). A row whose model
        features are not all present keeps the shape without a shift and has
        v_hat NaN. A row without any shape curve gets no forecast.
    """
    if ctx.side != "demand":
        raise ValueError(f"volume_shift is for the demand side, got side {ctx.side!r}")
    horizon = int(ctx.lag_days)
    lags = [int(k) for k in shape_lags]
    if not lags or len(set(lags)) != len(lags):
        raise ValueError(f"shape_lags must be distinct and not empty, got {list(shape_lags)}")
    if min(lags) < horizon:
        raise ValueError(f"shape lag {min(lags)} is below the forecast horizon of {horizon} days")
    if range_rule not in RANGE_RULES:
        raise ValueError(f"unknown range_rule {range_rule!r}, expected one of {RANGE_RULES}")

    t0 = time.time()
    hist = ctx.historical.reset_index(drop=True)
    X, names, volume, v_ref = volume_features(hist, horizon, ctx.exog_provider)
    target = volume - v_ref
    valid = np.isfinite(X).all(axis=1) & np.isfinite(target)
    hist_day = _day_numbers(hist["Trading Date"])
    index = _LagIndex(hist_day, hist["Interval Number"].to_numpy(dtype=np.int64))
    qty = hist["curve_qty"].to_numpy()
    price = hist["curve_price"].to_numpy()
    log.info("volume_shift: %d features on %d history rows (%d complete), %.0fs",
             len(names), len(hist), int(valid.sum()), time.time() - t0)

    test = ctx.test.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
    test_day = _day_numbers(test["Trading Date"])
    test_interval = test["Interval Number"].to_numpy(dtype=np.int64)
    test_row = index.rows(test_day, test_interval)
    if (test_row < 0).any():
        raise ValueError("some test rows are missing from the history table")

    v_hat = np.full(len(test), np.nan)
    refit_of: list[pd.Timestamp | None] = [None] * len(test)
    test_ts = pd.to_datetime(test["Trading Date"])
    refits = _monthly_refit_dates(test)
    window_ends = [*refits[1:], test_ts.max() + pd.Timedelta(days=1)]
    for T_refit, T_next in zip(refits, window_ends, strict=True):
        window = np.flatnonzero(((test_ts >= T_refit) & (test_ts < T_next)).to_numpy())
        if len(window) == 0:
            continue
        refit_day = int(np.datetime64(T_refit.date(), "D").astype(np.int64))
        train = valid & (hist_day <= refit_day - horizon)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=learning_rate,
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=0,
        ).fit(X[train], target[train])
        rows = test_row[window]
        ok = valid[rows]
        v_hat[window[ok]] = v_ref[rows[ok]] + model.predict(X[rows[ok]])
        for t in window:
            refit_of[t] = T_refit
        log.info("volume_shift refit %s: %d training rows, %d target rows, %.0fs",
                 T_refit.date(), int(train.sum()), int(ok.sum()), time.time() - t0)
    n_missing = int(np.isnan(v_hat).sum())
    if n_missing:
        log.warning("volume_shift: %d test rows have no volume forecast and keep the unshifted shape",
                    n_missing)

    out = []
    for t in range(len(test)):
        day, interval = int(test_day[t]), int(test_interval[t])
        cand_rows = index.rows(np.array([day - k for k in lags]), np.full(len(lags), interval))
        cand_rows = cand_rows[cand_rows >= 0]
        if len(cand_rows) == 0:
            continue
        shape_q, shape_p = weighted_median_curve(
            [(qty[j], price[j]) for j in cand_rows], [1] * len(cand_rows),
            range_rule=range_rule, ref=0, merge_equal=bool(merge_equal),
        )
        v_shape = cliff_volume(shape_q, shape_p)
        if np.isfinite(v_hat[t]):
            pred_q, pred_p = shift_keep_first_edge(shape_q, shape_p, float(v_hat[t] - v_shape))
        else:
            pred_q, pred_p = np.asarray(shape_q, dtype=float), np.asarray(shape_p, dtype=float)
        out.append(
            {
                "Trading Date": test["Trading Date"].iat[t],
                "Interval Number": interval,
                "true_curve_qty": test["curve_qty"].iat[t],
                "true_curve_price": test["curve_price"].iat[t],
                "pred_curve_qty": pred_q,
                "pred_curve_price": pred_p,
                "refit_date": refit_of[t],
                "v_shape": v_shape,
                "v_hat": v_hat[t],
            }
        )
    log.info("volume_shift: %d forecasts, %.0fs", len(out), time.time() - t0)
    return pd.DataFrame(out)
