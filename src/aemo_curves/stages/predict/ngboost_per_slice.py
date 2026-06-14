"""NGBoost (or HistGBM fallback) per τ-slice of the demand curve.

For each of K=64 fixed positions along the cumulative-quantity axis,
train an independent probabilistic regressor predicting today's price
at that slice from lag features. Isotonise the K predicted medians
to enforce monotonicity, then place them on the standard quantity grid
to form a curve.

This is a probabilistic baseline first; the *point* forecast is the
median (or mean) at each slice. CRPS is reported as a secondary metric.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from aemo_curves.config import DEFAULT_QUANTITY_GRID
from aemo_curves.core.predictor import CurvePredictor

from ._features import calendar_features

log = logging.getLogger(__name__)


def _index(df: pd.DataFrame) -> dict[tuple[int, object], int]:
    return {(int(r["Interval Number"]), r["Trading Date"]): i
            for i, (_, r) in enumerate(df.iterrows())}


def _build_features(
    interval: int,
    trading_date,
    hist: pd.DataFrame,
    hist_idx: dict,
    lags: tuple[int, ...],
) -> np.ndarray | None:
    ts = pd.Timestamp(trading_date)
    parts: list[np.ndarray] = []
    for lag in lags:
        key = (interval, (ts - pd.Timedelta(days=lag)).date())
        idx = hist_idx.get(key)
        if idx is None:
            return None
        parts.append(np.asarray(hist.iloc[idx]["embedding"], dtype=float))
    parts.append(calendar_features(trading_date))
    return np.concatenate(parts, axis=0)


class NGBoostPerSlicePredictor(CurvePredictor):
    """Per-slice probabilistic predictor.

    By default uses sklearn HistGradientBoostingRegressor (single output,
    deterministic) - cheaper and strong empirically. Pass `use_ngboost=True`
    to use the Normal-output NGBoost regressor for native CRPS.
    """

    name = "ngboost_per_slice"

    def __init__(
        self,
        lags: tuple[int, ...] = (30, 31, 37),
        n_slices: int = 64,
        max_iter: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        use_ngboost: bool = False,
        ngb_n_estimators: int = 200,
        seed: int = 0,
    ) -> None:
        self.lags = tuple(sorted(lags))
        self.n_slices = int(n_slices)
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.use_ngboost = bool(use_ngboost)
        self.ngb_n_estimators = int(ngb_n_estimators)
        self.seed = int(seed)
        self._models: list = []                  # one per slice
        self._global_qty_total_mean: float = 1.0  # for output qty grid

    def params(self) -> dict:
        return {
            "model": self.name, "n_slices": self.n_slices,
            "max_iter": self.max_iter, "max_depth": self.max_depth,
            "learning_rate": self.learning_rate, "use_ngboost": self.use_ngboost,
            "lags": "-".join(str(x) for x in self.lags),
        }

    def fit(self, train_embeddings: pd.DataFrame) -> NGBoostPerSlicePredictor:
        train = train_embeddings.reset_index(drop=True)
        idx = _index(train)
        # Build feature matrix once across all rows
        X_rows: list[np.ndarray] = []
        Y_rows: list[np.ndarray] = []
        for _, row in train.iterrows():
            feat = _build_features(int(row["Interval Number"]), row["Trading Date"],
                                   train, idx, self.lags)
            if feat is None:
                continue
            X_rows.append(feat)
            # The 64-d embedding's k-th coordinate is the price at the k-th τ.
            # For arbitrary n_slices we can sample but for our 64-d embedding
            # we just use the coordinates directly (best signal).
            Y_rows.append(np.asarray(row["embedding"], dtype=float))
        X = np.vstack(X_rows).astype(np.float32)
        Y = np.vstack(Y_rows).astype(np.float32)
        log.info("ngboost_per_slice fit: X=%s, Y=%s", X.shape, Y.shape)

        # Total quantity needed at output time
        totals = np.array([float(np.asarray(r["curve_qty"])[-1])
                           for _, r in train.iterrows()], dtype=float)
        self._global_qty_total_mean = float(np.mean(totals))

        # Fit one regressor per slice. Parallelise across slices (cheap models).
        def _fit_slice(k: int):
            if self.use_ngboost:
                from ngboost import NGBRegressor
                from ngboost.distns import Normal
                m = NGBRegressor(
                    Dist=Normal, n_estimators=self.ngb_n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=1.0, verbose=False, random_state=self.seed,
                )
                m.fit(X, Y[:, k])
                return m
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(
                max_iter=self.max_iter, max_depth=self.max_depth,
                learning_rate=self.learning_rate, early_stopping=True,
                random_state=self.seed,
            )
            m.fit(X, Y[:, k])
            return m

        # Sequential fit - HistGBM uses internal threading; nesting in a
        # ThreadPoolExecutor causes oversubscription and the runs hang.
        self._models = []
        n_slices = Y.shape[1]
        t0 = time.time()
        for k in range(n_slices):
            self._models.append(_fit_slice(k))
            if (k + 1) % 8 == 0:
                log.info("  fitted %d/%d slices (%.1fs)", k + 1, n_slices, time.time() - t0)
        log.info("ngboost_per_slice fit complete: %d slices", len(self._models))
        return self

    def predict(
        self, test_embeddings: pd.DataFrame, historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_lookup = _index(hist)
        # First pass: build feature matrix for all rows that have valid lags.
        sorted_test = test_embeddings.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
        feats: list[np.ndarray] = []
        meta: list[dict] = []
        for _, r in sorted_test.iterrows():
            interval = int(r["Interval Number"])
            feat = _build_features(interval, r["Trading Date"], hist, hist_lookup, self.lags)
            if feat is None:
                continue
            feats.append(feat)
            meta.append({
                "Trading Date": r["Trading Date"], "Interval Number": interval,
                "true_embedding": r["embedding"],
                "true_curve_qty": r["curve_qty"], "true_curve_price": r["curve_price"],
            })
        X = np.vstack(feats).astype(np.float32)
        n = X.shape[0]
        log.info("per-slice predict: %d rows × %d slices", n, len(self._models))
        # Batched predict: one call per slice over all rows.
        Y_pred = np.empty((n, len(self._models)), dtype=np.float32)
        for k, m in enumerate(self._models):
            Y_pred[:, k] = m.predict(X)
        # Enforce monotone non-increase across slices (demand curves)
        Y_pred = np.minimum.accumulate(Y_pred, axis=1)
        total_q = self._global_qty_total_mean
        qty = DEFAULT_QUANTITY_GRID * total_q
        rows = []
        for i, md in enumerate(meta):
            rows.append({
                **md,
                "pred_embedding": Y_pred[i],
                "pred_curve_qty": qty,
                "pred_curve_price": Y_pred[i],
            })
        return pd.DataFrame(rows)
