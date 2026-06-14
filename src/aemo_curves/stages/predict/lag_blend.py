"""Lag-blending predictors: convex combinations of lagged embeddings.

  - `BlendedLagPredictor`: learned scalar weight blending lag-1 and lag-7.
  - `LearnedBlendPredictor`: per-bucket simplex weights over N lags.
  - `SameWeekdayAveragePredictor` / `EMABaselinePredictor`: fixed-weight
    same-interval averages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from aemo_curves.config import DEFAULT_QUANTITY_GRID
from aemo_curves.core.predictor import CurvePredictor

from ._features import hist_index, lag_row


class BlendedLagPredictor(CurvePredictor):
    """Blend lag-1 and lag-7 embeddings.

    - mode="fixed": use a constant weight `w_fixed`.
    - mode="per_interval": for each interval, learn the mean optimal
        w minimizing ||(1-w)*lag1 + w*lag7 - true||² on training rows.
    - mode="per_interval_dow": same as above but stratified by day of week.
    """

    name = "blended_lag_predictor"

    def __init__(
        self,
        mode: str = "per_interval_dow",
        w_fixed: float = 0.5,
        lag_base: int = 1,
        lag_alt: int = 7,
    ) -> None:
        if mode not in {"fixed", "per_interval", "per_interval_dow"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.w_fixed = float(np.clip(w_fixed, 0.0, 1.0))
        self.lag_base = int(lag_base)
        self.lag_alt = int(lag_alt)
        self._weights: dict = {}  # interval -> w or (interval, dow) -> w

    def params(self) -> dict:
        return {
            "model": self.name,
            "mode": self.mode,
            "w_fixed": self.w_fixed,
            "lag_base": self.lag_base,
            "lag_alt": self.lag_alt,
        }

    def _optimal_w(self, lag1_emb: np.ndarray, lag7_emb: np.ndarray, true_emb: np.ndarray) -> float:
        """Closed-form: w = <true - lag1, lag7 - lag1> / ||lag7 - lag1||²."""
        diff = lag7_emb - lag1_emb
        denom = float(np.dot(diff, diff))
        if denom < 1e-9:
            return 0.0
        num = float(np.dot(true_emb - lag1_emb, diff))
        return float(np.clip(num / denom, 0.0, 1.0))

    def fit(self, train_embeddings: pd.DataFrame) -> BlendedLagPredictor:
        if self.mode == "fixed":
            return self
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        buckets: dict = {}
        for _, row in train.iterrows():
            interval = int(row["Interval Number"])
            lag1 = lag_row(train, idx, interval, row["Trading Date"], self.lag_base)
            lag7 = lag_row(train, idx, interval, row["Trading Date"], self.lag_alt)
            if lag1 is None or lag7 is None:
                continue
            lag1_emb = np.asarray(lag1["embedding"], dtype=float)
            lag7_emb = np.asarray(lag7["embedding"], dtype=float)
            true_emb = np.asarray(row["embedding"], dtype=float)
            w = self._optimal_w(lag1_emb, lag7_emb, true_emb)
            key = interval if self.mode == "per_interval" else (
                interval,
                pd.Timestamp(row["Trading Date"]).dayofweek,
            )
            buckets.setdefault(key, []).append(w)
        self._weights = {k: float(np.mean(v)) for k, v in buckets.items()}
        return self

    def _weight_for(self, interval: int, trading_date) -> float:
        if self.mode == "fixed":
            return self.w_fixed
        if self.mode == "per_interval":
            return float(self._weights.get(interval, self.w_fixed))
        key = (interval, pd.Timestamp(trading_date).dayofweek)
        return float(self._weights.get(key, self.w_fixed))

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
            lag1 = lag_row(hist, idx, interval, r["Trading Date"], self.lag_base)
            lag7 = lag_row(hist, idx, interval, r["Trading Date"], self.lag_alt)
            if lag1 is None or lag7 is None:
                continue
            w = self._weight_for(interval, r["Trading Date"])
            lag1_emb = np.asarray(lag1["embedding"], dtype=float)
            lag7_emb = np.asarray(lag7["embedding"], dtype=float)
            pred_emb = (1.0 - w) * lag1_emb + w * lag7_emb
            total = (1.0 - w) * float(np.asarray(lag1["curve_qty"])[-1]) + w * float(
                np.asarray(lag7["curve_qty"])[-1]
            )
            pred_qty = DEFAULT_QUANTITY_GRID * total
            pred_price = pred_emb
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": pred_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": lag1["Trading Date"],
                    "blend_w": float(w),
                }
            )
        return pd.DataFrame(rows)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _solve_simplex(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative weights w summing to 1 such that M @ w ≈ y.

    M: (n_samples × dim, n_lags) stacked so (M @ w) matches stacked y. We solve
    a small unconstrained problem over logits and softmax-project.
    """
    n_lags = M.shape[1]
    if n_lags == 1:
        return np.array([1.0])

    def loss(logits):
        w = _softmax(logits)
        resid = M @ w - y
        return float(resid @ resid)

    x0 = np.zeros(n_lags)
    res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 100})
    return _softmax(res.x)


class LearnedBlendPredictor(CurvePredictor):
    """Convex combination of N lagged embeddings with weights learned per bucket.

    - `lags`: tuple of integer lags in days.
    - `stratify_by`: 'interval_dow' (default), 'interval', or 'global'.
    - Output: same pattern as EMABaselinePredictor - residual vs `lags[0]`
      interpolated onto lag[0]'s step positions, so step structure is preserved.
    """

    name = "learned_blend"

    def __init__(
        self,
        lags: tuple[int, ...] = (30, 31, 37, 44),
        stratify_by: str = "interval_dow",
        min_samples: int = 20,
    ) -> None:
        if stratify_by not in {"interval_dow", "interval", "global"}:
            raise ValueError(stratify_by)
        self.lags = tuple(lags)
        self.stratify_by = stratify_by
        self.min_samples = int(min_samples)
        self._weights: dict = {}

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "stratify_by": self.stratify_by,
        }

    def _bucket_key(self, interval: int, trading_date) -> object:
        if self.stratify_by == "global":
            return "global"
        if self.stratify_by == "interval":
            return int(interval)
        return (int(interval), pd.Timestamp(trading_date).dayofweek)

    def fit(self, train_embeddings: pd.DataFrame) -> LearnedBlendPredictor:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        buckets: dict = {}
        for _, row in train.iterrows():
            interval = int(row["Interval Number"])
            lag_rows = [
                lag_row(train, idx, interval, row["Trading Date"], lag) for lag in self.lags
            ]
            if any(r is None for r in lag_rows):
                continue
            lag_embs = np.vstack([np.asarray(r["embedding"], dtype=float) for r in lag_rows])
            y = np.asarray(row["embedding"], dtype=float)
            k = self._bucket_key(interval, row["Trading Date"])
            buckets.setdefault(k, []).append((lag_embs.T, y))  # lag_embs.T shape: (64, n_lags)

        for k, samples in buckets.items():
            if len(samples) < self.min_samples:
                continue
            # Stack M (sum of sample_dim × n_lags) and y (sum of sample_dim).
            M = np.vstack([m for m, _ in samples])        # (S*64, n_lags)
            y = np.concatenate([yy for _, yy in samples])  # (S*64,)
            w = _solve_simplex(M, y)
            self._weights[k] = w
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        rows = []
        # Fallback: if a bucket is missing, return pure lag[0] baseline.
        default_w = np.zeros(len(self.lags))
        default_w[0] = 1.0
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            lag_rows = [
                lag_row(hist, idx, interval, r["Trading Date"], lag) for lag in self.lags
            ]
            if any(rr is None for rr in lag_rows):
                continue
            base = lag_rows[0]
            lag_embs = np.vstack([np.asarray(rr["embedding"], dtype=float) for rr in lag_rows])
            w = self._weights.get(self._bucket_key(interval, r["Trading Date"]), default_w)
            pred_emb = (w.reshape(-1, 1) * lag_embs).sum(axis=0)
            base_qty = np.asarray(base["curve_qty"], dtype=float)
            base_price = np.asarray(base["curve_price"], dtype=float)
            base_emb = np.asarray(base["embedding"], dtype=float)
            total = float(base_qty[-1]) if base_qty[-1] > 0 else 1.0
            norm_positions = np.clip(base_qty / total, 0.0, 1.0)
            residual = pred_emb - base_emb
            residual_at_pos = np.interp(norm_positions, DEFAULT_QUANTITY_GRID, residual)
            pred_price = base_price + residual_at_pos
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": base_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": base["Trading Date"],
                    "blend_weights": w,
                }
            )
        return pd.DataFrame(rows)


class SameWeekdayAveragePredictor(CurvePredictor):
    """Average the embeddings of the same interval on the most recent N same
    weekdays (lags 7, 14, 21, 28 by default).

    Cheap, zero-training, and a natural stabilization of lag-1: it preserves the
    weekly pattern while averaging out single-day noise.
    """

    name = "same_weekday_avg"

    def __init__(self, lags: tuple[int, ...] = (7, 14, 21, 28)) -> None:
        self.lags = tuple(sorted(lags))

    def params(self) -> dict:
        return {"model": self.name, "lags": "-".join(str(x) for x in self.lags)}

    def fit(self, train_embeddings: pd.DataFrame) -> SameWeekdayAveragePredictor:
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
            embs = []
            totals = []
            for lag in self.lags:
                src = lag_row(hist, idx, interval, r["Trading Date"], lag)
                if src is None:
                    continue
                embs.append(np.asarray(src["embedding"], dtype=float))
                totals.append(float(np.asarray(src["curve_qty"])[-1]))
            if len(embs) < 2:  # need at least 2 to average
                continue
            pred_emb = np.mean(np.vstack(embs), axis=0)
            total = float(np.mean(totals))
            pred_qty = DEFAULT_QUANTITY_GRID * total
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": pred_qty,
                    "pred_curve_price": pred_emb,
                    "retrieved_date": None,
                    "n_lags_used": int(len(embs)),
                }
            )
        return pd.DataFrame(rows)


class EMABaselinePredictor(CurvePredictor):
    """Weighted average of lag-1, lag-2, lag-7 embeddings applied ON TOP of
    the baseline's step structure (qty positions are the lag-1 ones).

    Acts as a Kalman-style smoother: lag-1 dominates (most weight), lag-7
    injects a small weekly component, lag-2 adds a smoothing term.
    """

    name = "ema_baseline"

    def __init__(
        self,
        weights: tuple[float, ...] = (0.7, 0.1, 0.2),
        lags: tuple[int, ...] = (1, 2, 7),
    ) -> None:
        if len(weights) != len(lags):
            raise ValueError("weights and lags must have the same length")
        w = np.asarray(weights, dtype=float)
        self.weights = tuple(w / w.sum())
        self.lags = tuple(lags)

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "weights": "-".join(f"{x:.2f}" for x in self.weights),
        }

    def fit(self, train_embeddings: pd.DataFrame) -> EMABaselinePredictor:
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
            src_rows = [lag_row(hist, idx, interval, r["Trading Date"], lag) for lag in self.lags]
            if any(s is None for s in src_rows):
                continue
            lag1 = src_rows[0]
            # Compute weighted-average embedding, then apply to lag-1 step positions.
            embs = np.vstack([np.asarray(s["embedding"], dtype=float) for s in src_rows])
            pred_emb = (np.asarray(self.weights).reshape(-1, 1) * embs).sum(axis=0)
            base_qty = np.asarray(lag1["curve_qty"], dtype=float)
            base_price = np.asarray(lag1["curve_price"], dtype=float)
            base_emb = np.asarray(lag1["embedding"], dtype=float)
            total = float(base_qty[-1]) if base_qty[-1] > 0 else 1.0
            norm_positions = np.clip(base_qty / total, 0.0, 1.0)
            # Interpolate the embedding residual onto baseline step positions.
            residual = pred_emb - base_emb
            residual_at_pos = np.interp(norm_positions, DEFAULT_QUANTITY_GRID, residual)
            pred_price = base_price + residual_at_pos
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": base_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": lag1["Trading Date"],
                }
            )
        return pd.DataFrame(rows)
