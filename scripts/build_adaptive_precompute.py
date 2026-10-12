"""Materialize the per-row precompute table for the `adaptive` family.

The adaptive family soft-blends the lag-7 baseline curve with the champion's
retrieved curve on a per-row basis. Both the meta-model and the walk-forward
selection need, for every row in a warm-up window (the last 6 train months) plus
the full test window, the two candidate curves and their realized HC-MAE, plus
the champion's retrieval diagnostics.

This script builds ONE table at
`artifacts/embeddings/adaptive_precompute_{side}_7d.pkl` covering

    WARM-UP  2023-11-01 .. 2024-04-30   (last 6 train months)
    TEST     2024-05-01 .. 2025-04-24

with, per (Trading Date, Interval Number):

  * baseline lag-7 curve arrays  (recovered by a direct date-7 lookup in the
    v_grid feature table; no model run needed), plus its v_grid embedding;
  * the champion's retrieved curve arrays + retrieval diagnostics + the
    champion's predicted embedding;
  * both models' realized HC-MAE against the true curve;
  * the true curve arrays and true embedding.

For the TEST window we REUSE the cached champion predictions (found by scanning
`artifacts/cache/*/result.json` for the side's champion experiment name). For the
WARM-UP window we run the champion pipeline fresh via a synthetic `PredictContext`:
`ctx.test` = the warm-up slice of the train feature table and `ctx.historical` =
concat(train, test). `rolling_refit` derives its monthly refit windows from
`ctx.test` and fits on `historical[date <= T - 7]`, so warm-up predictions only ever
see data at least 7 days older than the month being predicted -> leak-free.

The blend / meta code is direction-agnostic (a convex combination of two curves of
the same direction preserves that direction), so this builder is identical across
sides EXCEPT for the champion experiment name and the reported sanity means. Pass
`--side {demand,supply,residual}` to select which table to build.

Run:
    uv run python scripts/build_adaptive_precompute.py --side demand
    uv run python scripts/build_adaptive_precompute.py --side supply
    uv run python scripts/build_adaptive_precompute.py --side residual
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from aemo_curves.config import EMBEDDINGS_DIR, ARTIFACTS_DIR
from aemo_curves.core.datasets import historical_from, load_features
from aemo_curves.exogenous.builder import build_exog_provider
from aemo_curves.runner.context import PredictContext
from aemo_curves.runner.run import bootstrap
from aemo_curves.core.registry import get
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance

log = logging.getLogger("adaptive_precompute")

LAG_DAYS = 7
CHAMPION_PARAMS = dict(query="ridge", ridge_alpha=200.0, ridge_n_pca=20, pca_mode="split")
EXOG_SPEC = {"sources": ["proxy", "weather"]}

WARMUP_START = pd.Timestamp("2023-11-01")
WARMUP_END = pd.Timestamp("2024-04-30")

# Per-side sanity means for the TEST window (for the log line only). These are the
# cached champion / lag-7-baseline HC-MAE means the rebuilt table should reproduce.
SANITY = {
    "demand": (122.80, 148.97),
    "supply": (67.19, 54.83),
    "residual": (181.40, 191.90),
}


def _key_index(df: pd.DataFrame) -> dict:
    return {
        (int(r["Interval Number"]), r["Trading Date"]): i
        for i, (_, r) in enumerate(df.iterrows())
    }


def _find_champion_cache(experiment: str) -> Path:
    """Scan artifacts/cache/*/result.json for the side's champion experiment."""
    cache_root = ARTIFACTS_DIR / "cache"
    for entry in sorted(cache_root.iterdir()):
        rj = entry / "result.json"
        if not rj.exists():
            continue
        try:
            meta = json.load(open(rj))
        except Exception:
            continue
        if meta.get("experiment_name") == experiment:
            return entry
    raise RuntimeError(f"no cache entry found for experiment {experiment!r}")


def _load_cached_champion(experiment: str) -> pd.DataFrame:
    entry = _find_champion_cache(experiment)
    preds = pickle.load(open(entry / "predictions.pkl", "rb"))
    log.info("loaded cached champion from %s: %d rows (%s)", entry.name, len(preds), experiment)
    return preds


def _run_warmup_champion(train: pd.DataFrame, historical: pd.DataFrame, side: str) -> pd.DataFrame:
    tr_ts = pd.to_datetime(train["Trading Date"])
    warm = train[(tr_ts >= WARMUP_START) & (tr_ts <= WARMUP_END)].reset_index(drop=True)
    log.info("warm-up slice: %d rows (%s .. %s)", len(warm), WARMUP_START.date(), WARMUP_END.date())
    exog = build_exog_provider(EXOG_SPEC, side=side, lag_days=LAG_DAYS)
    ctx = PredictContext(
        train=train, test=warm, historical=historical, side=side,
        features="v_grid", split="canonical", lag_days=LAG_DAYS, exog_provider=exog,
    )
    strat = get("predict", "rolling_refit")
    t0 = time.time()
    preds = strat(ctx, **CHAMPION_PARAMS)
    log.info("warm-up champion fit: %d rows in %.1fs", len(preds), time.time() - t0)
    return preds


def _build_rows(champ: pd.DataFrame, hist: pd.DataFrame, hist_idx: dict, window: str) -> list[dict]:
    rows: list[dict] = []
    missing = 0
    for _, r in champ.iterrows():
        iv = int(r["Interval Number"])
        dt = pd.Timestamp(r["Trading Date"])
        # true curve: look up in historical (v_grid table has curves + embedding)
        true_idx = hist_idx.get((iv, dt.date()))
        if true_idx is None:
            missing += 1
            continue
        true_row = hist.iloc[true_idx]
        # baseline lag-7 curve via direct date-7 lookup
        lag_date = (dt - pd.Timedelta(days=LAG_DAYS)).date()
        base_idx = hist_idx.get((iv, lag_date))
        if base_idx is None:
            missing += 1
            continue
        base_row = hist.iloc[base_idx]

        true_q = np.asarray(true_row["curve_qty"], dtype=float)
        true_p = np.asarray(true_row["curve_price"], dtype=float)
        base_q = np.asarray(base_row["curve_qty"], dtype=float)
        base_p = np.asarray(base_row["curve_price"], dtype=float)
        champ_q = np.asarray(r["pred_curve_qty"], dtype=float)
        champ_p = np.asarray(r["pred_curve_price"], dtype=float)

        baseline_hc_mae = curve_hc_mae_distance(true_q, true_p, base_q, base_p)
        curve_hc_mae = curve_hc_mae_distance(true_q, true_p, champ_q, champ_p)

        rows.append({
            "Trading Date": dt.date(),
            "Interval Number": iv,
            "window": window,
            "true_curve_qty": true_q,
            "true_curve_price": true_p,
            "true_embedding": np.asarray(true_row["embedding"], dtype=float),
            "baseline_curve_qty": base_q,
            "baseline_curve_price": base_p,
            "baseline_embedding": np.asarray(base_row["embedding"], dtype=float),
            "baseline_retrieved_date": lag_date,
            "champ_curve_qty": champ_q,
            "champ_curve_price": champ_p,
            "pred_embedding": np.asarray(r["pred_embedding"], dtype=float),
            "retrieved_date": r["retrieved_date"],
            "bucket_size": r.get("bucket_size", np.nan),
            "retrieval_dist": r.get("retrieval_dist", np.nan),
            "retrieval_dist_2": r.get("retrieval_dist_2", np.nan),
            "n_neighbors_used": r.get("n_neighbors_used", np.nan),
            "refit_date": r.get("refit_date", pd.NaT),
            "baseline_hc_mae": float(baseline_hc_mae),
            "curve_hc_mae": float(curve_hc_mae),
        })
    if missing:
        log.warning("%s: %d rows dropped for missing true/baseline curve", window, missing)
    return rows


def _add_lag_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag7-vs-lag14 and lag7-vs-lag28 baseline-curve HC-MAE distances.

    The lag-7 baseline for target date d is the realized curve at d-7 (stored in
    the `baseline_curve_*` columns). To compare it against the lag-14 / lag-28
    curves we reuse the baseline curve of the rows whose target date is d-7 / d-21
    (whose baseline is the curve at d-14 / d-28). All strictly older than d-7 ->
    leak-safe. Missing lags fall back to 0.
    """
    d7 = {
        (int(r["Interval Number"]), pd.Timestamp(r["Trading Date"]).date()): i
        for i, r in df.iterrows()
    }
    v14 = np.zeros(len(df))
    v28 = np.zeros(len(df))
    for i, r in df.iterrows():
        iv = int(r["Interval Number"])
        d = pd.Timestamp(r["Trading Date"])
        bq, bp = r["baseline_curve_qty"], r["baseline_curve_price"]
        for off, arr in ((7, v14), (21, v28)):
            j = d7.get((iv, (d - pd.Timedelta(days=off)).date()))
            if j is None:
                continue
            arr[i] = curve_hc_mae_distance(bq, bp, df.at[j, "baseline_curve_qty"],
                                           df.at[j, "baseline_curve_price"])
    df = df.copy()
    df["lag7_vs_lag14"] = v14
    df["lag7_vs_lag28"] = v28
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the adaptive precompute table for one side.")
    parser.add_argument("--side", default="demand", choices=["demand", "supply", "residual"])
    args = parser.parse_args()
    side = args.side
    champion_experiment = f"{side}.7d.hc_mae.vgrid.rolling_ridge_exog_split"
    out_path = EMBEDDINGS_DIR / f"adaptive_precompute_{side}_{LAG_DAYS}d.pkl"
    sane_champ, sane_base = SANITY.get(side, (float("nan"), float("nan")))

    bootstrap()
    train, test = load_features(side, "v_grid", "canonical")
    hist = historical_from(train, test)
    hist_idx = _key_index(hist)

    # WARM-UP: fresh champion run
    warm_champ = _run_warmup_champion(train, hist, side)
    warm_rows = _build_rows(warm_champ, hist, hist_idx, "warmup")

    # TEST: reuse cached champion predictions
    test_champ = _load_cached_champion(champion_experiment)
    test_rows = _build_rows(test_champ, hist, hist_idx, "test")

    df = pd.DataFrame(warm_rows + test_rows)
    df = df.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
    df = _add_lag_drift(df)

    # Sanity: TEST-window HC-MAE means must match the cached champion / baseline.
    tm = df[df["window"] == "test"]
    log.info(
        "TEST window: n=%d champ_mean=%.4f baseline_mean=%.4f (cache: %.2f / %.2f)",
        len(tm), tm["curve_hc_mae"].mean(), tm["baseline_hc_mae"].mean(), sane_champ, sane_base,
    )
    wm = df[df["window"] == "warmup"]
    log.info(
        "WARM-UP window: n=%d champ_mean=%.4f baseline_mean=%.4f",
        len(wm), wm["curve_hc_mae"].mean(), wm["baseline_hc_mae"].mean(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    log.info("wrote %s (%d rows, %d cols)", out_path, len(df), len(df.columns))


if __name__ == "__main__":
    main()
