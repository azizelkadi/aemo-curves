"""Selector predict strategy: choose the champion or the lag baseline per row.

A selector model (trained by scripts/selector/train_selector.py) predicts the
HC-MAE gap between the rolling v_grid champion and the lag-1 baseline for each
test row. When the champion is predicted to win by more than a fixed threshold,
its curve is used; otherwise the lag-1 curve is used.

This is the inference half of the selector pipeline. It needs two artifacts the
training scripts produce: the backfill cache of champion predictions for the
test window, and the trained selector bundle. When either is missing it raises
with the command to build it.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from aemo_curves.config import ARTIFACTS_DIR
from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict._selector_features import build_feature_table


@register("predict", "selector")
def selector(
    ctx: PredictContext,
    variant: str = "main",
    backfill_tag: str = "vgrid_daily_2024",
) -> pd.DataFrame:
    bundle_path = ARTIFACTS_DIR / "selector" / variant / "selector_model.pkl"
    backfill_path = ARTIFACTS_DIR / "backfill" / backfill_tag / "_combined.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"trained selector missing at {bundle_path}; run scripts/selector/train_selector.py"
        )
    if not backfill_path.exists():
        raise FileNotFoundError(
            f"backfill cache missing at {backfill_path}; run scripts/selector/backfill_selector.py"
        )

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    vgrid_preds = pd.read_pickle(backfill_path)
    history = ctx.historical

    df = build_feature_table(vgrid_preds, history)
    X = df[bundle["feature_names"]].copy()
    delta_raw = bundle["selector"].predict(X)
    delta = bundle["isotonic"].predict(delta_raw)
    pick_vgrid = delta < -bundle["tau_locked"]

    rows = []
    for chose_vgrid, (_, r) in zip(pick_vgrid, df.iterrows()):
        qty = r["vg_pred_qty"] if chose_vgrid else r["lag1_pred_qty"]
        price = r["vg_pred_price"] if chose_vgrid else r["lag1_pred_price"]
        rows.append({
            "Trading Date": r["Trading Date"],
            "Interval Number": int(r["Interval Number"]),
            "true_curve_qty": r["true_curve_qty"],
            "true_curve_price": r["true_curve_price"],
            "pred_curve_qty": np.asarray(qty, dtype=float),
            "pred_curve_price": np.asarray(price, dtype=float),
            "selected_vgrid": bool(chose_vgrid),
        })
    return pd.DataFrame(rows)
