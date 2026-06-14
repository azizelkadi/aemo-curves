"""Diagnostic models that bound what shift-based correctors can achieve.

These are NOT real models - they exist to measure the ceiling of correction-
based strategies. `GridSearchOracleShift` is a Wasserstein-specific diagnostic
and stays tied to the curve Wasserstein metric directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor
from aemo_curves.stages.evaluate.wasserstein import curve_wasserstein_distance

from ._features import hist_index, lag_row


class OracleShiftPredictor(CurvePredictor):
    """Applies the median-optimal translation per row (uses true data - diagnostic).

    Upper bound on what any shift-only corrector could achieve, given the
    baseline curve at lag `base_lag`.
    """

    name = "oracle_shift"

    def __init__(self, base_lag: int = 1) -> None:
        self.base_lag = int(base_lag)

    def params(self) -> dict:
        return {"model": self.name, "base_lag": self.base_lag}

    def fit(self, train_embeddings: pd.DataFrame) -> OracleShiftPredictor:
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            lag1 = lag_row(hist, idx, interval, r["Trading Date"], self.base_lag)
            if lag1 is None:
                continue
            t_emb = np.asarray(r["embedding"], dtype=float)
            b_emb = np.asarray(lag1["embedding"], dtype=float)
            dp = float(np.median(t_emb - b_emb))
            qty = np.asarray(lag1["curve_qty"], dtype=float)
            price = np.asarray(lag1["curve_price"], dtype=float) + dp
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": b_emb + dp,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": qty,
                    "pred_curve_price": price,
                    "retrieved_date": lag1["Trading Date"],
                    "correction": np.array([dp]),
                }
            )
        return pd.DataFrame(rows)


class RealizedShiftPredictor(CurvePredictor):
    """Non-parametric shift: δp_today = alpha * median(base_emb - ref_emb).

    Assumes day-over-day (or month-over-month) shifts have some persistence.
    Zero training; only the persistence coefficient `alpha ∈ [0, 1]`.
    `base_lag` sets which past day's curve is the baseline; `lag_ref` is the
    further-back comparison used to estimate the realized recent shift.
    """

    name = "realized_shift"

    def __init__(self, alpha: float = 0.3, base_lag: int = 1, lag_ref: int = 2) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.base_lag = int(base_lag)
        self.lag_ref = int(lag_ref)

    def params(self) -> dict:
        return {
            "model": self.name,
            "alpha": self.alpha,
            "base_lag": self.base_lag,
            "lag_ref": self.lag_ref,
        }

    def fit(self, train_embeddings: pd.DataFrame) -> RealizedShiftPredictor:
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            lag1 = lag_row(hist, idx, interval, r["Trading Date"], self.base_lag)
            lagn = lag_row(hist, idx, interval, r["Trading Date"], self.lag_ref)
            if lag1 is None or lagn is None:
                continue
            e1 = np.asarray(lag1["embedding"], dtype=float)
            en = np.asarray(lagn["embedding"], dtype=float)
            dp = self.alpha * float(np.median(e1 - en))
            qty = np.asarray(lag1["curve_qty"], dtype=float)
            price = np.asarray(lag1["curve_price"], dtype=float) + dp
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": e1 + dp,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": qty,
                    "pred_curve_price": price,
                    "retrieved_date": lag1["Trading Date"],
                    "correction": np.array([dp]),
                }
            )
        return pd.DataFrame(rows)


class GridSearchOracleShift(CurvePredictor):
    """For each test row, grid-search the shift δ that minimizes curve
    Wasserstein; apply it. This is the TRUE upper bound for shift-only
    corrections (uses true curve - diagnostic only).
    """

    name = "grid_oracle_shift"

    def __init__(self, grid: np.ndarray | None = None, base_lag: int = 1) -> None:
        if grid is None:
            grid = np.arange(-30.0, 30.5, 1.0)
        self.grid = np.asarray(grid, dtype=float)
        self.base_lag = int(base_lag)

    def params(self) -> dict:
        return {
            "model": self.name,
            "grid_range": f"{self.grid.min():.1f}..{self.grid.max():.1f}",
            "base_lag": self.base_lag,
        }

    def fit(self, train_embeddings: pd.DataFrame) -> GridSearchOracleShift:
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            lag1 = lag_row(hist, idx, interval, r["Trading Date"], self.base_lag)
            if lag1 is None:
                continue
            base_qty = np.asarray(lag1["curve_qty"], dtype=float)
            base_price = np.asarray(lag1["curve_price"], dtype=float)
            true_qty = np.asarray(r["curve_qty"], dtype=float)
            true_price = np.asarray(r["curve_price"], dtype=float)
            best_w, best_d = float("inf"), 0.0
            for d in self.grid:
                w = curve_wasserstein_distance(true_qty, true_price, base_qty, base_price + d)
                if w < best_w:
                    best_w, best_d = w, float(d)
            pred_price = base_price + best_d
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": np.asarray(lag1["embedding"], dtype=float) + best_d,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": base_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": lag1["Trading Date"],
                    "correction": np.array([best_d]),
                }
            )
        return pd.DataFrame(rows)
