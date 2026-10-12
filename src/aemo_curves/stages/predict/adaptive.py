"""Per-row SOFT blending of the lag-7 baseline curve with the champion's curve.

The adaptive family reads a precomputed table (built by
`scripts/build_adaptive_precompute.py`) that holds, for every warm-up and test
row, the two candidate curves (baseline lag-7 and the champion's retrieved curve),
their realized HC-MAE, and the champion's retrieval diagnostics. A meta-model
chooses a per-row blend weight `w in [0, 1]` and the strategy emits

    blended = w * baseline_curve + (1 - w) * champion_curve

as a step curve on the union of both curves' quantity break-points.

FAIRNESS INVARIANT (mandatory). The HC-MAE metric integrates on the *union* of
the two compared curves' volume supports. Re-expressing the baseline (or the
champion) on the union-of-both-endpoints grid can change that support and hence
the score. So at the endpoints we short-circuit and return the endpoint curve's
arrays VERBATIM:

    w >= 1 - 1e-12  ->  the baseline curve arrays exactly
    w <= 1e-12      ->  the champion curve arrays exactly

which guarantees `HC-MAE(emit(w=1)) == baseline_hc_mae` and
`HC-MAE(emit(w=0)) == curve_hc_mae` exactly.

WEIGHT MODES (one YAML each):
  * constant_blend  - one w per test month, grid-picked on trailing blended error.
  * stratum_blend   - w per (interval-block-of-4 x weekend) from trailing errors,
                      shrunk toward the global constant w.
  * prob_blend      - HistGradientBoostingClassifier on 1[baseline better], with
                      walk-forward isotonic calibration on the accumulated OOF
                      (p-hat, y) stream; w = calibrated p-hat.
  * prob_blend_raw  - same classifier, no calibration; w = raw p-hat.
  * regret_blend    - HistGradientBoostingRegressor on (champ_err - base_err);
                      w = sigmoid(delta-hat / tau), tau grid-picked on trailing OOF.
  * oracle_gate     - w in {0, 1} using the TRUTH (diagnostic; not deployable).
  * oracle_weight   - per-row best grid w using the TRUTH (diagnostic).

NO-LEAKAGE. Every meta-feature and every fitted model for test month T uses only
rows dated <= T_start - 7 days. Trailing realized-error features aggregate only
rows dated <= (refit-month-start - 7d).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.core.registry import register
from aemo_curves.exogenous.holidays import wa_holiday_flags, wa_holiday_flag_names
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.evaluate.hc_mae import _step_eval, curve_hc_mae_distance

log = logging.getLogger(__name__)

_EPS = 1e-12
WEIGHT_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 4)
# Cap on the number of trailing rows scored when grid-picking a blend weight.
# Scoring the blend at all 21 grid points for every training row at n_grid=10000
# is O(rows * grid) HC-MAE integrations; on the full ~24k-row training block,
# repeated per stratum per month, that is prohibitively slow. We approximate the
# trailing blended error on a deterministic random subsample of this size (the
# brief permits this; stated in the report). Blocks at or below the cap are exact.
_GRID_SELECT_MAX_ROWS = 3000


# --------------------------------------------------------------------------- #
# curve blending / emit
# --------------------------------------------------------------------------- #
def blend_curves(base_q, base_p, champ_q, champ_p, w: float):
    """Blend two step curves at weight `w` (baseline weight; champion = 1 - w).

    Endpoints short-circuit to the verbatim endpoint curve (fairness invariant).
    For an interior `w` the blended curve is defined on the union of both curves'
    quantity break-points, with price `w * step_eval(base) + (1-w) * step_eval(champ)`.
    """
    base_q = np.asarray(base_q, dtype=float)
    base_p = np.asarray(base_p, dtype=float)
    champ_q = np.asarray(champ_q, dtype=float)
    champ_p = np.asarray(champ_p, dtype=float)
    if w >= 1.0 - _EPS:
        return base_q, base_p
    if w <= _EPS:
        return champ_q, champ_p
    qty = np.unique(np.concatenate([base_q, champ_q]))
    price = w * _step_eval(base_q, base_p, qty) + (1.0 - w) * _step_eval(champ_q, champ_p, qty)
    return qty, price


# --------------------------------------------------------------------------- #
# meta-feature construction (leak-safe)
# --------------------------------------------------------------------------- #
_HOLIDAY_NAMES = wa_holiday_flag_names()


def _interval_block(iv: int) -> int:
    return (int(iv) - 1) // 4


def _static_features(table: pd.DataFrame) -> pd.DataFrame:
    """Per-row features that need no trailing aggregation (all leak-safe)."""
    dts = pd.to_datetime(table["Trading Date"])
    iv = table["Interval Number"].astype(int)
    feats = pd.DataFrame(index=table.index)
    feats["interval"] = iv
    feats["interval_block"] = iv.map(_interval_block)
    feats["dow"] = dts.dt.dayofweek
    feats["month"] = dts.dt.month
    feats["is_weekend"] = (dts.dt.dayofweek >= 5).astype(float)
    feats["retrieval_dist"] = table["retrieval_dist"].astype(float)
    feats["retrieval_dist_2"] = table["retrieval_dist_2"].astype(float)
    feats["bucket_size"] = table["bucket_size"].astype(float)
    feats["n_neighbors_used"] = table["n_neighbors_used"].astype(float)
    # L2 between the champion's predicted embedding and the lag-7 baseline embedding
    feats["l2_pred_lag7"] = [
        float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        for a, b in zip(table["pred_embedding"], table["baseline_embedding"])
    ]
    # lag7-vs-lag14 and lag7-vs-lag28 curve HC-MAE distances (baseline drift)
    feats["lag7_vs_lag14"] = table["lag7_vs_lag14"].astype(float)
    feats["lag7_vs_lag28"] = table["lag7_vs_lag28"].astype(float)
    hol = np.stack([wa_holiday_flags(d) for d in dts.dt.date])
    for j, nm in enumerate(_HOLIDAY_NAMES):
        feats[f"cal_{nm}"] = hol[:, j]
    return feats


def _trailing_error_features(
    table: pd.DataFrame,
    cutoff: pd.Timestamp,
    windows,
) -> pd.DataFrame:
    """Trailing realized mean errors of both models, per interval and per (interval,dow).

    For a test month whose predictions are made against `cutoff = T_start - 7d`,
    only rows dated <= `cutoff` contribute to the trailing aggregates. Windows are
    trailing day-counts (e.g. 7, 28) measured back from `cutoff`.
    """
    dts = pd.to_datetime(table["Trading Date"])
    hist = table[dts <= cutoff]
    hist_dts = pd.to_datetime(hist["Trading Date"])
    out = pd.DataFrame(index=table.index)
    iv = table["Interval Number"].astype(int)
    dow = pd.to_datetime(table["Trading Date"]).dt.dayofweek
    for wdays in windows:
        lo = cutoff - pd.Timedelta(days=int(wdays))
        win = hist[(hist_dts > lo) & (hist_dts <= cutoff)]
        for col, tag in (("baseline_hc_mae", "base"), ("curve_hc_mae", "champ")):
            by_iv = win.groupby(win["Interval Number"].astype(int))[col].mean()
            by_iv_dow = win.groupby(
                [win["Interval Number"].astype(int),
                 pd.to_datetime(win["Trading Date"]).dt.dayofweek]
            )[col].mean()
            gmean = float(win[col].mean()) if len(win) else np.nan
            out[f"trail{wdays}_{tag}_iv"] = iv.map(by_iv).astype(float).fillna(gmean)
            keys = list(zip(iv, dow))
            out[f"trail{wdays}_{tag}_ivdow"] = [
                float(by_iv_dow.get(k, np.nan)) for k in keys
            ]
            out[f"trail{wdays}_{tag}_ivdow"] = (
                out[f"trail{wdays}_{tag}_ivdow"].fillna(out[f"trail{wdays}_{tag}_iv"])
            )
    return out


def _build_meta(table: pd.DataFrame, cutoff: pd.Timestamp, windows) -> pd.DataFrame:
    static = _static_features(table)
    trailing = _trailing_error_features(table, cutoff, windows)
    return pd.concat([static, trailing], axis=1)


# --------------------------------------------------------------------------- #
# precompute loading + derived columns
# --------------------------------------------------------------------------- #
def _load_precompute(name: str) -> pd.DataFrame:
    path = EMBEDDINGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"adaptive precompute table not found: {path}")
    df = pd.read_pickle(path)
    df = df.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
    for c in ("lag7_vs_lag14", "lag7_vs_lag28"):
        if c not in df.columns:
            df[c] = 0.0
    return df


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
def _emit(table: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for (_, r), w in zip(table.iterrows(), weights):
        q, p = blend_curves(
            r["baseline_curve_qty"], r["baseline_curve_price"],
            r["champ_curve_qty"], r["champ_curve_price"], float(w),
        )
        rows.append({
            "Trading Date": r["Trading Date"],
            "Interval Number": int(r["Interval Number"]),
            "true_curve_qty": np.asarray(r["true_curve_qty"], dtype=float),
            "true_curve_price": np.asarray(r["true_curve_price"], dtype=float),
            "pred_curve_qty": q,
            "pred_curve_price": p,
            "true_embedding": np.asarray(r["true_embedding"], dtype=float),
            "pred_embedding": np.asarray(r["pred_embedding"], dtype=float),
            "retrieved_date": r["retrieved_date"],
            "blend_weight": float(w),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# per-row blended error over the grid (for weight selection)
# --------------------------------------------------------------------------- #
def _blend_errors_over_grid(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """(n_rows, n_grid) HC-MAE of the blended curve at each grid weight."""
    n = len(table)
    err = np.empty((n, len(grid)))
    for i, (_, r) in enumerate(table.iterrows()):
        tq, tp = np.asarray(r["true_curve_qty"]), np.asarray(r["true_curve_price"])
        for j, w in enumerate(grid):
            q, p = blend_curves(
                r["baseline_curve_qty"], r["baseline_curve_price"],
                r["champ_curve_qty"], r["champ_curve_price"], float(w),
            )
            err[i, j] = curve_hc_mae_distance(tq, tp, q, p)
    return err


# --------------------------------------------------------------------------- #
# walk-forward month machinery
# --------------------------------------------------------------------------- #
def _test_months(test_dates) -> list[pd.Timestamp]:
    ts = pd.to_datetime(pd.Series(test_dates))
    start = ts.min().normalize().replace(day=1)
    return list(pd.date_range(start=start, end=ts.max(), freq="MS"))


# --------------------------------------------------------------------------- #
# the strategy
# --------------------------------------------------------------------------- #
@register("predict", "adaptive")
def adaptive(
    ctx: PredictContext,
    precompute: str = "adaptive_precompute_demand_7d.pkl",
    weight_mode: str = "prob_blend",
    warmup_months: int = 6,  # documented; the precompute already fixes the warm-up span
    trailing_windows=(7, 28),
    tau_grid=(5.0, 15.0, 40.0, 100.0),
    shrink_min_rows: int = 50,
    **params,
):
    table = _load_precompute(precompute)

    # Restrict emitted predictions to ctx.test rows (respects --subset-days).
    test_keys = set(
        (int(i), pd.Timestamp(d).date())
        for d, i in zip(ctx.test["Trading Date"], ctx.test["Interval Number"])
    )
    tbl_keys = list(zip(table["Interval Number"].astype(int),
                        pd.to_datetime(table["Trading Date"]).dt.date))
    is_emit = np.array([k in test_keys for k in tbl_keys])
    table = table.reset_index(drop=True)
    windows = list(trailing_windows)

    dts = pd.to_datetime(table["Trading Date"])
    months = _test_months(pd.to_datetime(ctx.test["Trading Date"]))

    weights = np.full(len(table), np.nan)
    # OOF stream for prob_blend isotonic calibration / regret tau selection.
    oof_p: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    oof_dhat: list[np.ndarray] = []
    oof_dtrue: list[np.ndarray] = []

    for T in months:
        cutoff = T - pd.Timedelta(days=int(ctx.lag_days))
        T_next = (T + pd.offsets.MonthBegin(1))
        month_mask = (dts >= T) & (dts < T_next) & is_emit
        if not month_mask.any():
            continue
        month_tbl = table[month_mask]
        train_mask = dts <= cutoff
        train_tbl = table[train_mask]

        w_month = _fit_predict_month(
            weight_mode, train_tbl, month_tbl, cutoff, windows,
            tau_grid=tau_grid, shrink_min_rows=shrink_min_rows,
            oof_p=oof_p, oof_y=oof_y, oof_dhat=oof_dhat, oof_dtrue=oof_dtrue,
        )
        weights[np.where(month_mask.to_numpy())[0]] = w_month

    emit_idx = np.where(is_emit)[0]
    emit_tbl = table.iloc[emit_idx].reset_index(drop=True)
    emit_w = weights[emit_idx]
    if np.isnan(emit_w).any():  # months with no training history: default to champion
        emit_w = np.where(np.isnan(emit_w), 0.0, emit_w)
    return _emit(emit_tbl, emit_w)


def _fit_predict_month(
    mode, train_tbl, month_tbl, cutoff, windows, *,
    tau_grid, shrink_min_rows, oof_p, oof_y, oof_dhat, oof_dtrue,
):
    n_month = len(month_tbl)

    # ORACLE modes use the truth of the month rows directly (diagnostic).
    if mode == "oracle_gate":
        return np.where(
            month_tbl["baseline_hc_mae"].to_numpy() <= month_tbl["curve_hc_mae"].to_numpy(),
            1.0, 0.0,
        )
    if mode == "oracle_weight":
        err = _blend_errors_over_grid(month_tbl, WEIGHT_GRID)
        return WEIGHT_GRID[err.argmin(axis=1)]

    if len(train_tbl) == 0:
        return np.zeros(n_month)  # no history -> champion

    if mode == "constant_blend":
        w = _pick_constant_w(train_tbl)
        return np.full(n_month, w)

    if mode == "stratum_blend":
        return _stratum_weights(train_tbl, month_tbl, shrink_min_rows)

    # Learned modes need meta-features.
    Xtr = _build_meta(train_tbl, cutoff, windows)
    Xmo = _build_meta(month_tbl, cutoff, windows)
    feat_cols = list(Xtr.columns)
    Xtr_v = Xtr[feat_cols].to_numpy(dtype=float)
    Xmo_v = Xmo[feat_cols].to_numpy(dtype=float)

    if mode in ("prob_blend", "prob_blend_raw"):
        y = (train_tbl["baseline_hc_mae"].to_numpy()
             < train_tbl["curve_hc_mae"].to_numpy()).astype(int)
        clf = HistGradientBoostingClassifier(random_state=0)
        if len(np.unique(y)) < 2:
            p_month = np.full(n_month, float(y.mean()))
            p_oof = None
        else:
            clf.fit(Xtr_v, y)
            p_month = clf.predict_proba(Xmo_v)[:, 1]
            # OOF prediction on THIS month's training block is not available; we
            # accumulate the current month's (p_hat, y) as they become known for
            # the NEXT month's calibrator. To seed the calibrator we use prior OOF.
            p_oof = clf.predict_proba(Xtr_v)[:, 1]  # in-sample, only for accrual guard
        if mode == "prob_blend":
            # Calibrate p_month using the isotonic map fit on accumulated OOF.
            if oof_p and len(np.concatenate(oof_p)) >= 50:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(np.concatenate(oof_p), np.concatenate(oof_y))
                w_month = iso.predict(p_month)
            else:
                w_month = p_month  # not enough OOF yet -> raw
        else:
            w_month = p_month
        # Accrue this month's OOF (walk-forward): the month rows' realized labels
        # become the calibration data for future months.
        y_month = (month_tbl["baseline_hc_mae"].to_numpy()
                   < month_tbl["curve_hc_mae"].to_numpy()).astype(int)
        oof_p.append(np.asarray(p_month, dtype=float))
        oof_y.append(y_month)
        return np.clip(w_month, 0.0, 1.0)

    if mode == "regret_blend":
        delta = (train_tbl["curve_hc_mae"].to_numpy()
                 - train_tbl["baseline_hc_mae"].to_numpy())
        reg = HistGradientBoostingRegressor(random_state=0)
        reg.fit(Xtr_v, delta)
        dhat_month = reg.predict(Xmo_v)
        # pick tau on accumulated OOF (dhat, dtrue) minimizing blended-error proxy
        tau = _pick_tau(oof_dhat, oof_dtrue, tau_grid)
        w_month = 1.0 / (1.0 + np.exp(-dhat_month / tau))
        # accrue OOF for next month's tau selection
        dtrue_month = (month_tbl["curve_hc_mae"].to_numpy()
                       - month_tbl["baseline_hc_mae"].to_numpy())
        oof_dhat.append(np.asarray(dhat_month, dtype=float))
        oof_dtrue.append(dtrue_month)
        return np.clip(w_month, 0.0, 1.0)

    raise ValueError(f"unknown weight_mode {mode!r}")


def _subsample(tbl: pd.DataFrame, cap: int = _GRID_SELECT_MAX_ROWS) -> pd.DataFrame:
    """Deterministic subsample of a training block for grid scoring (approx)."""
    if len(tbl) <= cap:
        return tbl
    return tbl.sample(n=cap, random_state=0)


def _pick_constant_w(train_tbl: pd.DataFrame) -> float:
    """Grid-pick the single w minimizing mean blended HC-MAE on the training rows.

    Scores the blend at every grid w for each (subsampled) training row and takes
    the mean-minimizing w. Blocks larger than `_GRID_SELECT_MAX_ROWS` are scored
    on a deterministic random subsample (approximation; see the module note).
    """
    sub = _subsample(train_tbl)
    err = _blend_errors_over_grid(sub, WEIGHT_GRID)
    return float(WEIGHT_GRID[err.mean(axis=0).argmin()])


def _stratum_weights(train_tbl, month_tbl, shrink_min_rows: int) -> np.ndarray:
    """w per (interval-block-of-4 x weekend) from trailing realized errors.

    For each stratum, the best grid w on the training rows of that stratum,
    shrunk toward the global constant w; strata with < `shrink_min_rows` rows fall
    back to the global w.
    """
    global_w = _pick_constant_w(train_tbl)
    tr = train_tbl.copy()
    tr["_blk"] = tr["Interval Number"].astype(int).map(_interval_block)
    tr["_wknd"] = (pd.to_datetime(tr["Trading Date"]).dt.dayofweek >= 5).astype(int)
    strat_w: dict = {}
    for (blk, wknd), grp in tr.groupby(["_blk", "_wknd"]):
        if len(grp) < shrink_min_rows:
            strat_w[(blk, wknd)] = global_w
            continue
        err = _blend_errors_over_grid(_subsample(grp), WEIGHT_GRID)
        w_local = float(WEIGHT_GRID[err.mean(axis=0).argmin()])
        # shrink toward global by the count-based weight n / (n + shrink_min_rows)
        lam = len(grp) / (len(grp) + shrink_min_rows)
        strat_w[(blk, wknd)] = lam * w_local + (1 - lam) * global_w
    mo = month_tbl.copy()
    blks = mo["Interval Number"].astype(int).map(_interval_block)
    wknds = (pd.to_datetime(mo["Trading Date"]).dt.dayofweek >= 5).astype(int)
    return np.array([strat_w.get((b, w), global_w) for b, w in zip(blks, wknds)])


def _pick_tau(oof_dhat, oof_dtrue, tau_grid) -> float:
    """Pick tau on accumulated OOF minimizing a blended-error proxy.

    Proxy: with w = sigmoid(dhat/tau), the expected per-row error relative to the
    two endpoints is approximated as w * base_err + (1-w) * champ_err. We only
    have dtrue = champ_err - base_err, so minimize E[w * (-dtrue)] i.e. reward
    weight on the baseline when dtrue < 0 (baseline better). Equivalent to
    minimizing sum(w * dtrue_shifted); we use the linear surrogate below.
    """
    if not oof_dhat:
        return float(np.median(tau_grid))
    dhat = np.concatenate(oof_dhat)
    dtrue = np.concatenate(oof_dtrue)
    best_tau, best_score = tau_grid[0], np.inf
    for tau in tau_grid:
        w = 1.0 / (1.0 + np.exp(-dhat / tau))
        # blended error surrogate: base + w*0 ... use champ + w*(base-champ) = champ - w*dtrue
        score = float(np.mean(-w * dtrue))  # minimize -> maximize weight on baseline where dtrue<0
        if score < best_score:
            best_score, best_tau = score, tau
    return float(best_tau)
