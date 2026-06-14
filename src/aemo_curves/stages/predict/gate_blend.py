"""Gating predict strategy: blend a champion curve with the lag baseline.

For each test row the prediction is a convex blend in v_grid embedding space,
``alpha * champion + (1 - alpha) * baseline``, converted back to a curve. The
weight alpha is derived per `mode`, fitted on a held-out fit window:

- static_table: per (interval, day-of-week) alpha that minimises HC-MAE.
- distance_confidence: a sigmoid of the champion's retrieval distance.
- classifier_gate: P(champion beats baseline | features).
- error_stack: alpha = err_baseline / (err_baseline + err_champion).
- linear_meta: Ridge from features to the per-row optimal alpha.

It reads the champion and baseline from their saved runs, so those experiments
must have been logged first. Rows inside the fit window are returned with
is_fit=True and dropped before scoring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.features import load_embedding_table

KEY = ["Trading Date", "Interval Number"]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
FEATURE_NAMES = [
    "retrieval_dist", "retrieval_dist_2", "champ_l2", "champ_max", "champ_min",
    "champ_mean", "champ_steepness", "base_l2", "base_max", "base_min", "base_mean",
    "base_steepness", "champ_base_l2_diff", "interval", "sin_doy", "cos_doy",
    "is_weekend", *[f"dow_{i}" for i in range(7)],
]


def _v_grid(side: str):
    meta = json.loads((EMBEDDINGS_DIR / f"v_grid_meta_{side}.json").read_text())
    v_max = float(meta["v_max_global"])
    k = int(meta["k"])
    return np.linspace(0.0, v_max, k), v_max, k


def _vg_lookup(side: str) -> dict:
    train = load_embedding_table("train_v_grid", side)
    test = load_embedding_table("test_v_grid", side)
    full = pd.concat([train, test], ignore_index=True)
    full["Trading Date"] = pd.to_datetime(full["Trading Date"])
    return {
        (r["Trading Date"].date(), int(r["Interval Number"])): np.asarray(r["embedding"], dtype=float)
        for _, r in full.iterrows()
    }


def _features_for_row(row: pd.Series, base_emb: np.ndarray) -> np.ndarray:
    champ = np.asarray(row["champ_pred_embedding"], dtype=float)
    date = pd.Timestamp(row["Trading Date"])
    dow, doy = int(date.dayofweek), int(date.timetuple().tm_yday)
    vec = np.empty(len(FEATURE_NAMES), dtype=float)
    vec[0] = float(row.get("retrieval_dist", 0.0)) if pd.notna(row.get("retrieval_dist")) else 0.0
    vec[1] = float(row.get("retrieval_dist_2", 0.0)) if pd.notna(row.get("retrieval_dist_2")) else 0.0
    vec[2] = float(np.linalg.norm(champ)); vec[3] = float(champ.max()); vec[4] = float(champ.min())
    vec[5] = float(champ.mean()); vec[6] = float(champ.max() - champ.min())
    vec[7] = float(np.linalg.norm(base_emb)); vec[8] = float(base_emb.max())
    vec[9] = float(base_emb.min()); vec[10] = float(base_emb.mean())
    vec[11] = float(base_emb.max() - base_emb.min())
    vec[12] = float(np.linalg.norm(champ - base_emb))
    vec[13] = float(row["Interval Number"])
    vec[14] = float(np.sin(2 * np.pi * doy / 365.25)); vec[15] = float(np.cos(2 * np.pi * doy / 365.25))
    vec[16] = float(dow >= 5)
    for d in range(7):
        vec[17 + d] = float(dow == d)
    return vec


def _fit_weight_fn(mode, champ, vg, v_grid, lag_days, fit_window, params):
    start, end = pd.Timestamp(fit_window[0]), pd.Timestamp(fit_window[1])
    fit = champ[(champ["Trading Date"] >= start) & (champ["Trading Date"] < end)]

    def base_emb_for(row):
        bd = (pd.Timestamp(row["Trading Date"]) - pd.Timedelta(days=lag_days)).date()
        return vg.get((bd, int(row["Interval Number"])))

    if mode == "distance_confidence":
        d = fit["retrieval_dist"].dropna().to_numpy()
        d_star = float(np.median(d))
        s = max(float(np.quantile(d, 0.75) - np.quantile(d, 0.25)) / 2.0, 1e-3)
        def fn(row):
            dv = float(row.get("retrieval_dist", float("nan")))
            return 0.5 if np.isnan(dv) else 1.0 / (1.0 + np.exp(-(d_star - dv) / s))
        return fn

    if mode == "static_table":
        buckets: dict = {}
        for _, r in fit.iterrows():
            be = base_emb_for(r)
            if be is None:
                continue
            key = (int(r["Interval Number"]), int(pd.Timestamp(r["Trading Date"]).dayofweek))
            buckets.setdefault(key, []).append(
                (np.asarray(r["true_curve_qty"], float), np.asarray(r["true_curve_price"], float),
                 np.asarray(r["champ_pred_embedding"], float), be))
        table: dict = {}
        for key, items in buckets.items():
            best_a, best_hc = 1.0, float("inf")
            for a in ALPHA_GRID:
                hc = np.mean([curve_hc_mae_distance(tq, tp, v_grid, a * ce + (1 - a) * be)
                              for tq, tp, ce, be in items])
                if hc < best_hc:
                    best_hc, best_a = hc, float(a)
            table[key] = best_a
        def fn(row):
            key = (int(row["Interval Number"]), int(pd.Timestamp(row["Trading Date"]).dayofweek))
            return table.get(key, 1.0)
        return fn

    # the remaining modes fit a model on the feature vector
    X, y_ch, y_ba, y_opt, y_cls = [], [], [], [], []
    for _, r in fit.iterrows():
        be = base_emb_for(r)
        if be is None:
            continue
        tq = np.asarray(r["true_curve_qty"], float); tp = np.asarray(r["true_curve_price"], float)
        ce = np.asarray(r["champ_pred_embedding"], float)
        hc_ch = curve_hc_mae_distance(tq, tp, v_grid, ce)
        hc_ba = curve_hc_mae_distance(tq, tp, v_grid, be)
        X.append(_features_for_row(r, be)); y_ch.append(hc_ch); y_ba.append(hc_ba)
        y_cls.append(int(hc_ch < hc_ba))
        best_a, best_hc = 1.0, float("inf")
        for a in ALPHA_GRID:
            hc = curve_hc_mae_distance(tq, tp, v_grid, a * ce + (1 - a) * be)
            if hc < best_hc:
                best_hc, best_a = hc, float(a)
        y_opt.append(best_a)
    X = np.vstack(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if mode == "classifier_gate":
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         random_state=0).fit(Xs, np.asarray(y_cls))
        def fn(row):
            be = base_emb_for(row)
            if be is None:
                return 1.0
            return float(clf.predict_proba(scaler.transform(_features_for_row(row, be).reshape(1, -1)))[0, 1])
        return fn
    if mode == "error_stack":
        reg_ch = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                           random_state=0).fit(Xs, np.asarray(y_ch))
        reg_ba = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                           random_state=0).fit(Xs, np.asarray(y_ba))
        def fn(row):
            be = base_emb_for(row)
            if be is None:
                return 1.0
            xs = scaler.transform(_features_for_row(row, be).reshape(1, -1))
            ech = max(float(reg_ch.predict(xs)[0]), 1e-3); eba = max(float(reg_ba.predict(xs)[0]), 1e-3)
            return eba / (eba + ech)
        return fn
    if mode == "linear_meta":
        reg = Ridge(alpha=1.0).fit(Xs, np.asarray(y_opt))
        def fn(row):
            be = base_emb_for(row)
            if be is None:
                return 1.0
            return float(reg.predict(scaler.transform(_features_for_row(row, be).reshape(1, -1)))[0])
        return fn
    raise ValueError(f"unknown gate_blend mode {mode!r}")


@register("predict", "gate_blend")
def gate_blend(
    ctx: PredictContext,
    mode: str,
    champion: str,
    baseline_run: str,
    fit_window=("2024-05-01", "2024-07-01"),
    **params,
) -> pd.DataFrame:
    from aemo_curves.tracking import load_predictions_by_name

    v_grid, _, _ = _v_grid(ctx.side)
    vg = _vg_lookup(ctx.side)
    champ = load_predictions_by_name(champion).rename(
        columns={"pred_embedding": "champ_pred_embedding"}
    )
    weight_fn = _fit_weight_fn(mode, champ, vg, v_grid, ctx.lag_days, fit_window, params)

    fit_start, fit_end = pd.Timestamp(fit_window[0]), pd.Timestamp(fit_window[1])
    rows = []
    for _, r in champ.iterrows():
        interval = int(r["Interval Number"])
        bd = (pd.Timestamp(r["Trading Date"]) - pd.Timedelta(days=ctx.lag_days)).date()
        base_emb = vg.get((bd, interval))
        if base_emb is None:
            continue
        if fit_start <= pd.Timestamp(r["Trading Date"]) < fit_end:
            continue
        alpha = float(np.clip(weight_fn(r), 0.0, 1.0))
        blended = alpha * np.asarray(r["champ_pred_embedding"], float) + (1 - alpha) * base_emb
        rows.append({
            "Trading Date": r["Trading Date"],
            "Interval Number": interval,
            "true_curve_qty": r["true_curve_qty"],
            "true_curve_price": r["true_curve_price"],
            "pred_curve_qty": v_grid,
            "pred_curve_price": blended,
            "pred_embedding": blended,
            "alpha": alpha,
        })
    return pd.DataFrame(rows)
