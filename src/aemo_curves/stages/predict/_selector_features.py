"""Feature engineering for the lag1-vs-vgrid_daily selector.

The feature set is leakage-safe by construction: every feature is computable
from data with `Trading Date < D`
plus model predictions for day `D` (predictions don't use truth at D).

Features (14, plus 3 native HGB categoricals):
    - disagreement_hc            HC-MAE between lag1 and vgrid predictions
    - retrieval_dist             query-to-1NN L2 (vgrid model internals)
    - retrieval_dist_2           query-to-2NN L2
    - nn_ratio                   retrieval_dist / retrieval_dist_2 (Lowe-style confidence)
    - retrieved_age_days         D - retrieved_date
    - library_size_days          days of history available at refit
    - baseline_stability_hc      HC-MAE(truth[D-1, i], truth[D-2, i])
    - rolling_vgrid_hc_14d       per-interval 14d rolling HC-MAE of vgrid (ends D-1)
    - rolling_vgrid_minus_baseline_14d  rolling vgrid minus rolling baseline
    - interval_sin, interval_cos
    - dow_sin, dow_cos
    - doy_sin, doy_cos
    - (categorical) Interval Number, DoW, Month

is_holiday is omitted (no holidays dependency).

Public API:
    compute_lag1_predictions(history)  -> per-row lag1 predictions
    score_hc_mae(df, qa, pa, qb, pb)   -> per-row HC-MAE
    build_feature_table(vgrid_preds, history) -> selector training/inference table
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance

log = logging.getLogger(__name__)

KEY = ["Trading Date", "Interval Number"]

# Feature lists
CONTINUOUS_FEATURES = [
    "disagreement_hc",
    "retrieval_dist",
    "retrieval_dist_2",
    "nn_ratio",
    "retrieved_age_days",
    "library_size_days",
    "baseline_stability_hc",
    "rolling_vgrid_hc_14d",
    "rolling_vgrid_minus_baseline_14d",
    "interval_sin", "interval_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
]
CATEGORICAL_FEATURES = ["Interval Number", "DoW", "Month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


def _row_hc(args):
    return curve_hc_mae_distance(args[0], args[1], args[2], args[3])


def score_hc_mae(df: pd.DataFrame, qa: str, pa: str, qb: str, pb: str,
                 n_jobs: int = -1) -> np.ndarray:
    args = list(zip(df[qa], df[pa], df[qb], df[pb]))
    return np.asarray(Parallel(n_jobs=n_jobs)(delayed(_row_hc)(a) for a in args))


def compute_lag1_predictions(history: pd.DataFrame) -> pd.DataFrame:
    """Return a per-row lag1 prediction frame: for each (Trading Date, interval),
    the predicted curve is the truth at (Trading Date - 1, interval).

    History must contain `Trading Date`, `Interval Number`, `curve_qty`,
    `curve_price`. Returns a frame with the same keys plus
    `lag1_pred_qty/price`. Rows whose (Trading Date - 1, interval) is missing
    are dropped.
    """
    h = history.copy()
    h["Trading Date"] = pd.to_datetime(h["Trading Date"])
    h = h[KEY + ["curve_qty", "curve_price"]]
    h_idx = h.set_index(KEY)
    # For each (D, i), look up (D-1, i)
    out_rows = []
    for (D, i), row in h_idx.iterrows():
        D_prev = D - pd.Timedelta(days=1)
        try:
            prev = h_idx.loc[(D_prev, i)]
            out_rows.append({
                "Trading Date": D, "Interval Number": int(i),
                "lag1_pred_qty": prev["curve_qty"],
                "lag1_pred_price": prev["curve_price"],
                "true_curve_qty": row["curve_qty"],
                "true_curve_price": row["curve_price"],
            })
        except KeyError:
            pass
    return pd.DataFrame(out_rows)


def build_feature_table(vgrid_preds: pd.DataFrame,
                        history: pd.DataFrame) -> pd.DataFrame:
    """Build the selector feature + label table for the rows in `vgrid_preds`.

    `vgrid_preds` is the backfill output: per-row vgrid_daily predictions with
    `retrieval_dist`, `retrieval_dist_2`, `library_size_days`, `retrieved_date`,
    `pred_curve_qty/price`. Returns a long table with KEY, all features in
    `ALL_FEATURES`, the per-row HC-MAE for both models, and the label
    `delta_hc = hc_vgrid - hc_lag1`.

    `history` must contain `Trading Date`, `Interval Number`,
    `curve_qty`, `curve_price` (= the truth) for every (D, i) needed by the
    lag1 lookup and the rolling-error features.
    """
    log.info("computing lag1 predictions on %d rows…", len(vgrid_preds))
    lag1 = compute_lag1_predictions(history)
    log.info("  lag1 frame: %d rows", len(lag1))

    vg = vgrid_preds.copy()
    vg["Trading Date"] = pd.to_datetime(vg["Trading Date"])
    vg = vg.rename(columns={
        "pred_curve_qty": "vg_pred_qty",
        "pred_curve_price": "vg_pred_price",
    })
    keep_cols = KEY + ["vg_pred_qty", "vg_pred_price",
                       "retrieved_date", "retrieval_dist", "retrieval_dist_2",
                       "library_size_days", "n_neighbors_used", "bucket_size",
                       "refit_date"]
    vg = vg[keep_cols]

    df = lag1.merge(vg, on=KEY, how="inner")
    df = df.sort_values(KEY).reset_index(drop=True)
    log.info("  merged frame: %d rows", len(df))

    # ---- HC-MAE per row for both predictors (the labels)
    log.info("scoring HC-MAE for lag1 and vgrid_daily…")
    df["hc_lag1"] = score_hc_mae(df, "true_curve_qty", "true_curve_price",
                                 "lag1_pred_qty", "lag1_pred_price")
    df["hc_vgrid"] = score_hc_mae(df, "true_curve_qty", "true_curve_price",
                                  "vg_pred_qty", "vg_pred_price")

    # ---- Disagreement between predictions (leakage-safe - no truth)
    log.info("computing disagreement_hc (lag1 vs vgrid predictions)…")
    args = list(zip(df["lag1_pred_qty"], df["lag1_pred_price"],
                    df["vg_pred_qty"], df["vg_pred_price"]))
    df["disagreement_hc"] = np.asarray(Parallel(n_jobs=-1)(delayed(_row_hc)(a) for a in args))

    # ---- Retrieval features
    df["retrieved_date"] = pd.to_datetime(df["retrieved_date"])
    df["retrieved_age_days"] = (df["Trading Date"] - df["retrieved_date"]).dt.days
    df["nn_ratio"] = df["retrieval_dist"] / df["retrieval_dist_2"].replace(0, np.nan)

    # ---- Calendar
    df["DoW"] = df["Trading Date"].dt.dayofweek
    df["Month"] = df["Trading Date"].dt.month
    df["doy"] = df["Trading Date"].dt.dayofyear
    df["interval_sin"] = np.sin(2 * np.pi * df["Interval Number"] / 48)
    df["interval_cos"] = np.cos(2 * np.pi * df["Interval Number"] / 48)
    df["dow_sin"] = np.sin(2 * np.pi * df["DoW"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["DoW"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 366)

    # ---- baseline_stability_hc: HC-MAE(truth[D-1, i], truth[D-2, i])
    log.info("computing baseline_stability_hc (truth D-1 vs D-2)…")
    h_idx = history.copy()
    h_idx["Trading Date"] = pd.to_datetime(h_idx["Trading Date"])
    h_idx = h_idx[KEY + ["curve_qty", "curve_price"]].set_index(KEY)
    dates = df["Trading Date"].values
    intervals = df["Interval Number"].values
    d1 = pd.to_datetime(dates) - pd.Timedelta(days=1)
    d2 = pd.to_datetime(dates) - pd.Timedelta(days=2)
    stab_args = []
    has = np.zeros(len(df), dtype=bool)
    for k in range(len(df)):
        try:
            r1 = h_idx.loc[(d1[k], int(intervals[k]))]
            r2 = h_idx.loc[(d2[k], int(intervals[k]))]
            stab_args.append((r1["curve_qty"], r1["curve_price"],
                              r2["curve_qty"], r2["curve_price"]))
            has[k] = True
        except KeyError:
            stab_args.append((None, None, None, None))
    stab_vals = np.full(len(df), np.nan)
    valid_idx = np.where(has)[0]
    if len(valid_idx) > 0:
        valid_args = [stab_args[k] for k in valid_idx]
        valid_vals = np.asarray(Parallel(n_jobs=-1)(delayed(_row_hc)(a) for a in valid_args))
        stab_vals[valid_idx] = valid_vals
    df["baseline_stability_hc"] = stab_vals

    # ---- Rolling 14d per-interval HC-MAE of vgrid + baseline, shifted by 1
    log.info("computing 14d rolling per-interval HC-MAE features…")
    df = df.sort_values(KEY).reset_index(drop=True)
    for col, name in [("hc_lag1", "rolling_baseline_hc_14d"),
                      ("hc_vgrid", "rolling_vgrid_hc_14d")]:
        df[name] = (
            df.groupby("Interval Number")[col]
              .transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
        )
    df["rolling_vgrid_minus_baseline_14d"] = (
        df["rolling_vgrid_hc_14d"] - df["rolling_baseline_hc_14d"]
    )

    # ---- Label
    df["delta_hc"] = df["hc_vgrid"] - df["hc_lag1"]
    df["vgrid_better"] = (df["delta_hc"] < 0).astype(int)
    return df
