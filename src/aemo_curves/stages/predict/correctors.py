"""Baseline-corrector family: start from the lag-1 curve, then predict a correction.

All predictors in this module:
  1. Take the lag-1 (same-interval, previous-day) curve as a starting point.
  2. Predict a correction using features available up to yesterday.
  3. Output a curve derived from the baseline + correction.

Because corrections are centered at zero and regularized, these models can do
no worse than the baseline IN EXPECTATION when `alpha` is large enough. In
practice we want just enough flexibility to capture systematic drifts without
adding noise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aemo_curves.config import DEFAULT_QUANTITY_GRID
from aemo_curves.core.predictor import CurvePredictor

from ._features import (
    build_daily_regime_stats,
    calendar_features,
    curve_stats,
    daily_regime,
    hist_index,
    lag_row,
)

# ---------------------------------------------------------------------------
# Shared feature builder
# ---------------------------------------------------------------------------

LAGS_DEFAULT = (1, 2, 7, 14)


def _row_features(
    interval: int,
    trading_date,
    hist: pd.DataFrame,
    hist_index_: dict,
    daily_stats: dict,
    lags: tuple[int, ...],
    exog_provider=None,
) -> tuple[np.ndarray, pd.Series] | None:
    """Stationary feature row: only deltas and calendar. No absolute levels.

    Using raw price levels leaks the long-term trend into features, which
    causes massive covariate shift at test time. Deltas and ratios are
    (approximately) stationary across the 4-year training window.

    If `exog_provider` is set, its feature vector is concatenated. When the
    provider cannot supply features for this `(interval, date)` the whole
    row is dropped (returns None), so training and prediction stay aligned.

    Returns (features, lag1_row).
    """
    lag_rows: list[pd.Series] = []
    for lag in lags:
        r = lag_row(hist, hist_index_, interval, trading_date, lag)
        if r is None:
            return None
        lag_rows.append(r)
    regimes = []
    for lag in lags:
        reg = daily_regime(daily_stats, trading_date, lag)
        if reg is None:
            return None
        regimes.append(reg)
    stats = [curve_stats(r["curve_qty"], r["curve_price"]) for r in lag_rows]
    features: list[np.ndarray] = [calendar_features(trading_date)]
    # Short-term trend: lag1 - lag2 stats
    if len(stats) >= 2:
        features.append(stats[0] - stats[1])
    # Weekly change: lag1 - lag7 stats
    if len(stats) >= 3:
        features.append(stats[0] - stats[2])
    # Longer change: lag1 - lag14 stats
    if len(stats) >= 4:
        features.append(stats[0] - stats[3])
    # Daily regime deltas: yesterday-wide level change vs earlier days
    if len(regimes) >= 2:
        features.append(regimes[0] - regimes[1])
    if len(regimes) >= 3:
        features.append(regimes[0] - regimes[2])
    # Log-ratio of total qty (scale-invariant level)
    log_ratio = np.array(
        [np.log(max(stats[0][2], 1e-6) / max(s[2], 1e-6)) for s in stats[1:]],
        dtype=float,
    )
    features.append(log_ratio)
    # Exogenous features (weather, demand forecasts, proxy market state...)
    if exog_provider is not None:
        ex = exog_provider(int(interval), trading_date)
        if ex is None:
            return None
        features.append(np.asarray(ex, dtype=float))
    return np.concatenate(features, axis=0), lag_rows[0]


# ---------------------------------------------------------------------------
# Scalar corrector (shift, shift+scale)
# ---------------------------------------------------------------------------


class _PerIntervalScalarCorrector(CurvePredictor):
    """Per-interval Ridge regressor predicting 1-2 scalar corrections from features."""

    def __init__(
        self,
        *,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        alpha: float = 5.0,
        min_samples: int = 30,
        n_targets: int = 1,
        exog_provider=None,
    ) -> None:
        self.lags = tuple(sorted(lags))
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.n_targets = int(n_targets)
        self.exog_provider = exog_provider
        self._models: dict[int, dict] = {}

    # Subclasses override these ---------------------------------------------
    def _target(self, baseline_row: pd.Series, true_row: pd.Series) -> np.ndarray:
        raise NotImplementedError

    def _apply(self, baseline_row: pd.Series, correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def _fit_interval(self, X: np.ndarray, Y: np.ndarray) -> dict:
        x_scaler = StandardScaler()
        Xs = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        Ys = y_scaler.fit_transform(Y)
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(Xs, Ys)
        # Bound the predicted correction to the range seen in training, so a
        # bad regression cannot emit an extreme curve at test time.
        y_lo = np.percentile(Y, 1, axis=0) - 2 * np.maximum(Y.std(axis=0), 1e-6)
        y_hi = np.percentile(Y, 99, axis=0) + 2 * np.maximum(Y.std(axis=0), 1e-6)
        return {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "ridge": ridge,
            "y_clip": (y_lo, y_hi),
        }

    def fit(self, train_embeddings: pd.DataFrame) -> _PerIntervalScalarCorrector:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        daily = build_daily_regime_stats(train)
        for interval, sub in train.groupby("Interval Number"):
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                built = _row_features(
                    int(interval), row["Trading Date"], train, idx, daily, self.lags,
                    exog_provider=self.exog_provider,
                )
                if built is None:
                    continue
                feat, lag1 = built
                y = self._target(lag1, row)
                X_rows.append(feat)
                Y_rows.append(np.atleast_1d(y))
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows)
            Y = np.vstack(Y_rows)
            self._models[int(interval)] = self._fit_interval(X, Y)
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        daily = build_daily_regime_stats(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            model = self._models.get(interval)
            if model is None:
                continue
            built = _row_features(
                interval, r["Trading Date"], hist, idx, daily, self.lags,
                exog_provider=self.exog_provider,
            )
            if built is None:
                continue
            feat, lag1 = built
            Xs = model["x_scaler"].transform(feat.reshape(1, -1))
            Ys_pred = model["ridge"].predict(Xs)
            Ys_pred = np.atleast_2d(Ys_pred)
            correction = model["y_scaler"].inverse_transform(Ys_pred)[0]
            y_lo, y_hi = model["y_clip"]
            correction = np.clip(correction, y_lo, y_hi)
            pred_qty, pred_price = self._apply(lag1, correction)
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": _interp_embedding(pred_qty, pred_price),
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": pred_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": lag1["Trading Date"],
                    "correction": correction,
                }
            )
        return pd.DataFrame(rows)


def _interp_embedding(qty: np.ndarray, price: np.ndarray) -> np.ndarray:
    from aemo_curves.stages.features.interp import curve_to_embedding
    return curve_to_embedding(np.asarray(qty, dtype=float), np.asarray(price, dtype=float))


class ShiftCorrector(_PerIntervalScalarCorrector):
    """Predict a scalar price shift δp; output (lag1.qty, lag1.price + δp).

    Captures same-interval mean-price drift (fuel prices, demand level, etc.)
    while preserving the lag-1 step structure exactly.

    `shrinkage ∈ [0, 1]` attenuates the predicted shift at test time (useful
    when the regressor is systematically too confident).
    """

    name = "shift_corrector"

    def __init__(
        self,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        alpha: float = 5.0,
        shrinkage: float = 1.0,
        exog_provider=None,
    ) -> None:
        super().__init__(lags=lags, alpha=alpha, n_targets=1, exog_provider=exog_provider)
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "alpha": self.alpha,
            "shrinkage": self.shrinkage,
            "exog": self.exog_provider is not None,
        }

    def _target(self, baseline_row: pd.Series, true_row: pd.Series) -> np.ndarray:
        # Wasserstein-optimal 1-D translation aligning two distributions at
        # matched quantile positions is the median of per-quantile price
        # differences. Embeddings are already on a common quantile grid.
        b_emb = np.asarray(baseline_row["embedding"], dtype=float)
        t_emb = np.asarray(true_row["embedding"], dtype=float)
        return np.array([float(np.median(t_emb - b_emb))], dtype=float)

    def _apply(self, baseline_row: pd.Series, correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dp = float(correction[0]) * self.shrinkage
        qty = np.asarray(baseline_row["curve_qty"], dtype=float)
        price = np.asarray(baseline_row["curve_price"], dtype=float) + dp
        return qty, price


class ShiftScaleCorrector(_PerIntervalScalarCorrector):
    """Predict (price_shift, log_quantity_scale). Shift is Wasserstein-optimal
    1-D translation (median of embedding difference); scale is log-ratio of
    total cumulative quantity."""

    name = "shift_scale_corrector"

    def __init__(
        self,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        alpha: float = 5.0,
        exog_provider=None,
    ) -> None:
        super().__init__(lags=lags, alpha=alpha, n_targets=2, exog_provider=exog_provider)

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "alpha": self.alpha,
            "exog": self.exog_provider is not None,
        }

    def _target(self, baseline_row: pd.Series, true_row: pd.Series) -> np.ndarray:
        b_emb = np.asarray(baseline_row["embedding"], dtype=float)
        t_emb = np.asarray(true_row["embedding"], dtype=float)
        dp = float(np.median(t_emb - b_emb))
        b_total = float(np.asarray(baseline_row["curve_qty"])[-1])
        t_total = float(np.asarray(true_row["curve_qty"])[-1])
        log_scale = float(np.log(max(t_total, 1e-6) / max(b_total, 1e-6)))
        return np.array([dp, log_scale], dtype=float)

    def _apply(self, baseline_row: pd.Series, correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dp, log_s = float(correction[0]), float(correction[1])
        s = float(np.clip(np.exp(log_s), 0.25, 4.0))  # clip extreme scales
        qty = np.asarray(baseline_row["curve_qty"], dtype=float) * s
        price = np.asarray(baseline_row["curve_price"], dtype=float) + dp
        return qty, price


# ---------------------------------------------------------------------------
# 64-d residual corrector
# ---------------------------------------------------------------------------


class ResidualEmbeddingCorrector(CurvePredictor):
    """Predict a 64-d price residual at the normalized quantile grid, added to lag-1.

    pred_embedding = lag1_embedding + r_hat
    Output curve = (DEFAULT_QUANTITY_GRID * baseline_total_qty, pred_embedding).

    Trains a per-interval Ridge with PCA compression on both X and the residual Y.
    Strong regularization biases `r_hat → 0`, which means the model only moves
    away from baseline when it has confident signal.
    """

    name = "residual_embedding_ridge"

    def __init__(
        self,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        alpha: float = 20.0,
        n_pca_x: int | None = 20,
        n_pca_y: int | None = 10,
        min_samples: int = 30,
        shrinkage: float = 1.0,
        exog_provider=None,
    ) -> None:
        self.lags = tuple(sorted(lags))
        self.alpha = float(alpha)
        self.n_pca_x = n_pca_x
        self.n_pca_y = n_pca_y
        self.min_samples = int(min_samples)
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.exog_provider = exog_provider
        self._models: dict[int, dict] = {}

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "alpha": self.alpha,
            "n_pca_x": self.n_pca_x if self.n_pca_x is not None else "none",
            "n_pca_y": self.n_pca_y if self.n_pca_y is not None else "none",
            "shrinkage": self.shrinkage,
            "exog": self.exog_provider is not None,
        }

    def _build_features(
        self,
        interval: int,
        row: pd.Series,
        hist: pd.DataFrame,
        idx: dict,
        daily: dict,
    ) -> tuple[np.ndarray, pd.Series, np.ndarray] | None:
        built = _row_features(
            interval, row["Trading Date"], hist, idx, daily, self.lags,
            exog_provider=self.exog_provider,
        )
        if built is None:
            return None
        scalars, lag1 = built
        lag_embs = []
        for lag in self.lags:
            r = lag_row(hist, idx, interval, row["Trading Date"], lag)
            if r is None:
                return None
            lag_embs.append(np.asarray(r["embedding"], dtype=float))
        x = np.concatenate([*lag_embs, scalars], axis=0)
        return x, lag1, lag_embs[0]

    def fit(self, train_embeddings: pd.DataFrame) -> ResidualEmbeddingCorrector:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        daily = build_daily_regime_stats(train)
        for interval, sub in train.groupby("Interval Number"):
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                built = self._build_features(int(interval), row, train, idx, daily)
                if built is None:
                    continue
                x, lag1, lag1_emb = built
                y = np.asarray(row["embedding"], dtype=float) - lag1_emb
                X_rows.append(x)
                Y_rows.append(y)
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows)
            Y = np.vstack(Y_rows)
            steps_x = [("scaler", StandardScaler())]
            if self.n_pca_x is not None:
                steps_x.append(("pca", PCA(n_components=min(self.n_pca_x, X.shape[0] - 1, X.shape[1]))))
            x_pipe = Pipeline(steps_x)
            Xp = x_pipe.fit_transform(X)
            y_scaler = StandardScaler()
            Ys = y_scaler.fit_transform(Y)
            pca_y = None
            Yt = Ys
            if self.n_pca_y is not None:
                comp = min(self.n_pca_y, Y.shape[0] - 1, Y.shape[1])
                if comp > 0:
                    pca_y = PCA(n_components=comp)
                    Yt = pca_y.fit_transform(Ys)
            ridge = Ridge(alpha=self.alpha)
            ridge.fit(Xp, Yt)
            self._models[int(interval)] = {
                "x_pipeline": x_pipe,
                "y_scaler": y_scaler,
                "pca_y": pca_y,
                "ridge": ridge,
            }
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        daily = build_daily_regime_stats(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            model = self._models.get(interval)
            if model is None:
                continue
            built = self._build_features(interval, r, hist, idx, daily)
            if built is None:
                continue
            x, lag1, lag1_emb = built
            Xp = model["x_pipeline"].transform(x.reshape(1, -1))
            Yt_pred = model["ridge"].predict(Xp)
            if model["pca_y"] is not None:
                Ys_pred = model["pca_y"].inverse_transform(Yt_pred)
            else:
                Ys_pred = Yt_pred
            r_hat = model["y_scaler"].inverse_transform(Ys_pred)[0] * self.shrinkage
            pred_emb = lag1_emb + r_hat
            total_qty = float(np.asarray(lag1["curve_qty"], dtype=float)[-1])
            pred_qty = DEFAULT_QUANTITY_GRID * total_qty
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
                    "residual_norm": float(np.linalg.norm(r_hat)),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Residual-on-baseline-structure corrector: interpolate a 64-d residual onto
# the baseline's actual step positions so we preserve the real step structure.
# ---------------------------------------------------------------------------


class PerDayShiftCorrector(CurvePredictor):
    """One shift per DAY, shared across all 48 intervals of that day.

    Pooling across intervals gives each model ~1200 training samples (one per
    trading date) instead of ~25 per interval, which substantially reduces
    variance. The cost is that we lose interval-specific signal, but for a
    scalar correction that's a net win.
    """

    name = "per_day_shift"

    def __init__(
        self,
        alpha: float = 20.0,
        shrinkage: float = 1.0,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        base_lag: int = 1,
        exog_provider=None,
    ) -> None:
        self.alpha = float(alpha)
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.lags = tuple(sorted(lags))
        self.base_lag = int(base_lag)
        self.exog_provider = exog_provider
        self._model: dict | None = None

    def params(self) -> dict:
        return {
            "model": self.name,
            "alpha": self.alpha,
            "shrinkage": self.shrinkage,
            "lags": "-".join(str(x) for x in self.lags),
            "base_lag": self.base_lag,
            "exog": self.exog_provider is not None,
        }

    def _day_features(self, trading_date, daily: dict) -> np.ndarray | None:
        """Day-wide features. Exog (if any) is appended once per date; the
        chosen interval for the lookup is 1 (no per-interval structure here)."""
        base = calendar_features(trading_date)
        regimes = []
        for lag in self.lags:
            reg = daily_regime(daily, trading_date, lag)
            if reg is None:
                return None
            regimes.append(reg)
        features = [base]
        if len(regimes) >= 2:
            features.append(regimes[0] - regimes[1])
        if len(regimes) >= 3:
            features.append(regimes[0] - regimes[2])
        if len(regimes) >= 4:
            features.append(regimes[0] - regimes[3])
        if self.exog_provider is not None:
            ex = self.exog_provider(1, trading_date)
            if ex is None:
                return None
            features.append(np.asarray(ex, dtype=float))
        return np.concatenate(features, axis=0)

    def _day_target(self, trading_date, hist: pd.DataFrame, hist_index_: dict) -> float | None:
        """Day-wide shift target: median of (true_emb - base_emb) across all intervals."""
        diffs: list[float] = []
        for interval in range(1, 49):
            today = hist_index_.get((interval, trading_date))
            if today is None:
                continue
            lag1 = lag_row(hist, hist_index_, interval, trading_date, self.base_lag)
            if lag1 is None:
                continue
            t_emb = np.asarray(hist.iloc[today]["embedding"], dtype=float)
            b_emb = np.asarray(lag1["embedding"], dtype=float)
            diffs.extend(list(t_emb - b_emb))
        if not diffs:
            return None
        return float(np.median(diffs))

    def fit(self, train_embeddings: pd.DataFrame) -> PerDayShiftCorrector:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        daily = build_daily_regime_stats(train)
        X, Y = [], []
        for date in sorted(train["Trading Date"].unique()):
            feat = self._day_features(date, daily)
            if feat is None:
                continue
            y = self._day_target(date, train, idx)
            if y is None:
                continue
            X.append(feat)
            Y.append([y])
        X = np.vstack(X)
        Y = np.vstack(Y)
        x_scaler = StandardScaler()
        Xs = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        Ys = y_scaler.fit_transform(Y)
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(Xs, Ys)
        y_lo = np.percentile(Y, 1, axis=0) - 2 * np.maximum(Y.std(axis=0), 1e-6)
        y_hi = np.percentile(Y, 99, axis=0) + 2 * np.maximum(Y.std(axis=0), 1e-6)
        self._model = {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "ridge": ridge,
            "y_clip": (y_lo, y_hi),
        }
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        daily = build_daily_regime_stats(hist)
        rows = []
        # Predict one shift per date, then apply to every interval of that date.
        date_shift: dict = {}
        for date in sorted(test_embeddings["Trading Date"].unique()):
            feat = self._day_features(date, daily)
            if feat is None:
                continue
            Xs = self._model["x_scaler"].transform(feat.reshape(1, -1))
            Ys_pred = np.atleast_2d(self._model["ridge"].predict(Xs))
            dp = float(self._model["y_scaler"].inverse_transform(Ys_pred)[0, 0])
            lo, hi = self._model["y_clip"]
            dp = float(np.clip(dp, lo[0], hi[0]))
            date_shift[date] = dp * self.shrinkage
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            date = r["Trading Date"]
            interval = int(r["Interval Number"])
            if date not in date_shift:
                continue
            lag1 = lag_row(hist, idx, interval, date, self.base_lag)
            if lag1 is None:
                continue
            dp = date_shift[date]
            qty = np.asarray(lag1["curve_qty"], dtype=float)
            price = np.asarray(lag1["curve_price"], dtype=float) + dp
            pred_emb = np.asarray(lag1["embedding"], dtype=float) + dp
            rows.append(
                {
                    "Trading Date": date,
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": qty,
                    "pred_curve_price": price,
                    "retrieved_date": lag1["Trading Date"],
                    "correction": np.array([dp]),
                }
            )
        return pd.DataFrame(rows)


class ResidualOnBaselineCorrector(ResidualEmbeddingCorrector):
    """Same model as ResidualEmbeddingCorrector, but the correction is
    interpolated onto the baseline's step positions so the output curve keeps
    the lag-1 step structure (one price per real offer) with prices shifted by
    the per-step predicted residual."""

    name = "residual_on_baseline_ridge"

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        idx = hist_index(hist)
        daily = build_daily_regime_stats(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            model = self._models.get(interval)
            if model is None:
                continue
            built = self._build_features(interval, r, hist, idx, daily)
            if built is None:
                continue
            x, lag1, lag1_emb = built
            Xp = model["x_pipeline"].transform(x.reshape(1, -1))
            Yt_pred = model["ridge"].predict(Xp)
            Ys_pred = (
                model["pca_y"].inverse_transform(Yt_pred)
                if model["pca_y"] is not None
                else Yt_pred
            )
            r_hat = model["y_scaler"].inverse_transform(Ys_pred)[0] * self.shrinkage
            base_qty = np.asarray(lag1["curve_qty"], dtype=float)
            base_price = np.asarray(lag1["curve_price"], dtype=float)
            total = float(base_qty[-1]) if base_qty[-1] > 0 else 1.0
            norm_positions = np.clip(base_qty / total, 0.0, 1.0)
            r_at_positions = np.interp(norm_positions, DEFAULT_QUANTITY_GRID, r_hat)
            pred_price = base_price + r_at_positions
            pred_emb = lag1_emb + r_hat
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
                    "residual_norm": float(np.linalg.norm(r_hat)),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gradient-boosted residual corrector (nonlinear)
# ---------------------------------------------------------------------------


class ResidualEmbeddingGBM(ResidualEmbeddingCorrector):
    """Same family as ResidualEmbeddingCorrector, but the regressor is
    HistGradientBoostingRegressor wrapped for multi-output. Target is PCA-reduced
    to keep training tractable."""

    name = "residual_embedding_gbm"

    def __init__(
        self,
        lags: tuple[int, ...] = LAGS_DEFAULT,
        n_pca_x: int | None = 20,
        n_pca_y: int | None = 6,
        max_iter: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        min_samples: int = 50,
    ) -> None:
        super().__init__(
            lags=lags,
            alpha=1.0,
            n_pca_x=n_pca_x,
            n_pca_y=n_pca_y,
            min_samples=min_samples,
        )
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "n_pca_x": self.n_pca_x if self.n_pca_x is not None else "none",
            "n_pca_y": self.n_pca_y if self.n_pca_y is not None else "none",
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "l2_regularization": self.l2_regularization,
        }

    def _make_regressor(self, n_outputs: int) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            random_state=0,
        )
        return MultiOutputRegressor(base) if n_outputs > 1 else base

    def fit(self, train_embeddings: pd.DataFrame) -> ResidualEmbeddingGBM:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        daily = build_daily_regime_stats(train)
        for interval, sub in train.groupby("Interval Number"):
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                built = self._build_features(int(interval), row, train, idx, daily)
                if built is None:
                    continue
                x, lag1, lag1_emb = built
                y = np.asarray(row["embedding"], dtype=float) - lag1_emb
                X_rows.append(x)
                Y_rows.append(y)
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows)
            Y = np.vstack(Y_rows)
            steps_x = [("scaler", StandardScaler())]
            if self.n_pca_x is not None:
                steps_x.append(("pca", PCA(n_components=min(self.n_pca_x, X.shape[0] - 1, X.shape[1]))))
            x_pipe = Pipeline(steps_x)
            Xp = x_pipe.fit_transform(X)
            y_scaler = StandardScaler()
            Ys = y_scaler.fit_transform(Y)
            pca_y = None
            Yt = Ys
            if self.n_pca_y is not None:
                comp = min(self.n_pca_y, Y.shape[0] - 1, Y.shape[1])
                if comp > 0:
                    pca_y = PCA(n_components=comp)
                    Yt = pca_y.fit_transform(Ys)
            reg = self._make_regressor(n_outputs=Yt.shape[1])
            reg.fit(Xp, Yt if Yt.shape[1] > 1 else Yt.ravel())
            self._models[int(interval)] = {
                "x_pipeline": x_pipe,
                "y_scaler": y_scaler,
                "pca_y": pca_y,
                "ridge": reg,  # predict() reads the regressor under this key
            }
        return self
