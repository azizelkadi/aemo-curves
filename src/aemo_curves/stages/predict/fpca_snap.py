"""FPCA + per-interval regression + closest-historical-curve snap.

Two predictors share the same pipeline shape but differ in the curve
representation they reduce with PCA:
  - `FoFSnapPredictor`: PCA on the 64-d embeddings (function-on-function).
  - `LogDerivFPCAPredictor`: PCA on the log-derivative of the quantile
    function (Ramsay 1998 trick).

Both fit a per-interval Ridge (or neural) regressor from lag PCA scores to
today's scores, then snap the prediction to the nearest training curve in
PCA-score space so the output is always a real staircase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aemo_curves.core.predictor import CurvePredictor

from ._features import calendar_features as _calendar_features
from ._features import hist_index
from ._shared import key_index as _index


def _stack_emb(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_numpy())


def _build_pca_lag_features(
    interval: int,
    trading_date,
    hist: pd.DataFrame,
    hist_idx: dict,
    train_pca_scores: np.ndarray,
    train_idx_map: dict,
    lags: tuple[int, ...],
    use_calendar: bool,
) -> np.ndarray | None:
    """Stack PCA scores at the requested lags + optional calendar features."""
    ts = pd.Timestamp(trading_date)
    parts: list[np.ndarray] = []
    for lag in lags:
        idx = hist_idx.get((interval, (ts - pd.Timedelta(days=lag)).date()))
        if idx is None:
            return None
        # The PCA scores are indexed by position in the training table.
        # If the lag row is in `hist` but not in train, we can't use it directly.
        # We resolve by mapping (interval, date) → train position.
        tidx = train_idx_map.get((interval, (ts - pd.Timedelta(days=lag)).date()))
        if tidx is None:
            return None
        parts.append(train_pca_scores[tidx])
    if use_calendar:
        parts.append(_calendar_features(trading_date))
    return np.concatenate(parts, axis=0)


class FoFSnapPredictor(CurvePredictor):
    """Functional-PCA + per-interval Ridge + global closest-curve snap."""

    name = "fof_snap"

    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 100.0,
        lags: tuple[int, ...] = (30, 31, 37),
        use_calendar: bool = True,
        min_samples: int = 20,
        predictor: str = "ridge",   # "ridge" or "neural"
        nn_hidden_dim: int = 64,
        nn_epochs: int = 60,
        nn_lr: float = 1e-3,
    ) -> None:
        if predictor not in {"ridge", "neural"}:
            raise ValueError(predictor)
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.lags = tuple(sorted(lags))
        self.use_calendar = bool(use_calendar)
        self.min_samples = int(min_samples)
        self.predictor = predictor
        self.nn_hidden_dim = int(nn_hidden_dim)
        self.nn_epochs = int(nn_epochs)
        self.nn_lr = float(nn_lr)
        self._pca: PCA | None = None
        self._train_scores: np.ndarray | None = None
        self._train_emb_df: pd.DataFrame | None = None
        self._models: dict[int, dict] = {}
        self._train_idx_map: dict[tuple[int, object], int] = {}

    def params(self) -> dict:
        return {
            "model": self.name,
            "n_components": self.n_components,
            "alpha": self.alpha,
            "lags": "-".join(str(x) for x in self.lags),
            "use_calendar": self.use_calendar,
        }

    def fit(self, train_embeddings: pd.DataFrame) -> FoFSnapPredictor:
        train = train_embeddings.reset_index(drop=True)
        self._train_emb_df = train
        E = _stack_emb(train)
        self._pca = PCA(n_components=self.n_components, random_state=0)
        self._train_scores = self._pca.fit_transform(E)
        self._train_idx_map = _index(train)
        hist_idx = self._train_idx_map  # train-only for fit
        # Fit per-interval Ridge mapping lag-PCA-scores → today's PCA scores
        for interval, sub in train.groupby("Interval Number"):
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                feat = _build_pca_lag_features(
                    int(interval), row["Trading Date"], train, hist_idx,
                    self._train_scores, self._train_idx_map,
                    self.lags, self.use_calendar,
                )
                if feat is None:
                    continue
                target_idx = self._train_idx_map[(int(row["Interval Number"]), row["Trading Date"])]
                X_rows.append(feat)
                Y_rows.append(self._train_scores[target_idx])
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows)
            Y = np.vstack(Y_rows)
            pipe = Pipeline([("scaler", StandardScaler())])
            Xp = pipe.fit_transform(X)
            y_scaler = StandardScaler()
            Ys = y_scaler.fit_transform(Y)
            if self.predictor == "ridge":
                est = Ridge(alpha=self.alpha)
                est.fit(Xp, Ys)
                predict_fn = est.predict
            else:  # neural
                from ._nn import _train_mlp
                predict_fn = _train_mlp(
                    Xp.astype(np.float32), Ys.astype(np.float32),
                    hidden_dim=self.nn_hidden_dim, epochs=self.nn_epochs,
                    batch_size=256, lr=self.nn_lr, weight_decay=1e-4, seed=0,
                )
            self._models[int(interval)] = {
                "x_pipeline": pipe, "y_scaler": y_scaler, "predict_fn": predict_fn,
            }
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_idx = hist_index(hist)
        # We need PCA scores for every historical row that might be a lag.
        # Compute on-the-fly when needed via the trained PCA.
        # For the snap step we only ever snap to the *training* library.
        rows: list[dict] = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            mdl = self._models.get(interval)
            if mdl is None:
                continue
            # Build lag features. If the lag row is in test (not train),
            # we still need its PCA score - project via the trained PCA.
            feat = self._build_lag_features_with_hist(
                interval, r["Trading Date"], hist, hist_idx,
            )
            if feat is None:
                continue
            Xp = mdl["x_pipeline"].transform(feat.reshape(1, -1))
            Ys = mdl["predict_fn"](Xp.astype(np.float32) if self.predictor == "neural" else Xp)
            pred_scores = mdl["y_scaler"].inverse_transform(np.atleast_2d(Ys))[0]
            # Snap to nearest training curve in PCA-score space
            d = np.linalg.norm(self._train_scores - pred_scores.reshape(1, -1), axis=1)
            best = int(np.argmin(d))
            src = self._train_emb_df.iloc[best]
            rows.append({
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_embedding": r["embedding"],
                "pred_embedding": np.asarray(src["embedding"], dtype=float),
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": src["curve_qty"],
                "pred_curve_price": src["curve_price"],
                "retrieved_date": src["Trading Date"],
                "embedding_retrieval_distance": float(d[best]),
                "pca_score_distance": float(d[best]),
            })
        return pd.DataFrame(rows)

    def _build_lag_features_with_hist(
        self, interval: int, trading_date, hist: pd.DataFrame, hist_idx: dict,
    ) -> np.ndarray | None:
        """Like _build_pca_lag_features but projects new lag rows via the trained PCA."""
        ts = pd.Timestamp(trading_date)
        parts: list[np.ndarray] = []
        for lag in self.lags:
            tidx = self._train_idx_map.get((interval, (ts - pd.Timedelta(days=lag)).date()))
            if tidx is not None:
                parts.append(self._train_scores[tidx])
                continue
            # Fallback: look up in hist and project via trained PCA.
            hidx = hist_idx.get((interval, (ts - pd.Timedelta(days=lag)).date()))
            if hidx is None:
                return None
            emb = np.asarray(hist.iloc[hidx]["embedding"], dtype=float).reshape(1, -1)
            parts.append(self._pca.transform(emb)[0])
        if self.use_calendar:
            parts.append(_calendar_features(trading_date))
        return np.concatenate(parts, axis=0)


GRID_SIZE = 128       # number of u-points on which Q(u) is sampled
EPS_DERIV = 1e-3       # floor for derivative (keeps log finite)


def _curve_to_logderiv_q(qty: np.ndarray, price: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample Q(u) on `grid` via cumulative-quantity, then return log(Q'(u))."""
    qty = np.asarray(qty, dtype=float)
    price = np.asarray(price, dtype=float)
    max_q = float(qty[-1]) if len(qty) else 0.0
    if max_q <= 0:
        return np.zeros_like(grid)
    norm_q = np.clip(qty / max_q, 0.0, 1.0)
    Q = np.interp(grid, norm_q, price, left=price[0], right=price[-1])
    # Numerical derivative w.r.t. u
    du = np.diff(grid)
    dQ = np.diff(Q)
    deriv = dQ / np.maximum(du, 1e-9)
    # log of |derivative| with floor to avoid log(0); centred-difference
    # at the midpoints projected back to the grid by repeating the last value.
    deriv = np.maximum(np.abs(deriv), EPS_DERIV)
    log_d = np.log(deriv)
    # Pad to grid length
    return np.concatenate([log_d, log_d[-1:]])


def _stack_log_deriv(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Return the log-deriv matrix."""
    n = len(df)
    L = np.empty((n, len(grid)), dtype=float)
    for i, (_, r) in enumerate(df.iterrows()):
        qty = np.asarray(r["curve_qty"], dtype=float)
        price = np.asarray(r["curve_price"], dtype=float)
        L[i] = _curve_to_logderiv_q(qty, price, grid)
    return L


class LogDerivFPCAPredictor(CurvePredictor):
    """FPCA-on-log-derivative + per-interval Ridge + closest-curve snap."""

    name = "log_deriv_fpca"

    def __init__(
        self,
        n_components: int = 12,
        alpha: float = 100.0,
        lags: tuple[int, ...] = (30, 31, 37),
        use_calendar: bool = True,
        min_samples: int = 20,
        grid_size: int = GRID_SIZE,
        predictor: str = "ridge",   # "ridge" or "neural"
        nn_hidden_dim: int = 64,
        nn_epochs: int = 60,
        nn_lr: float = 1e-3,
    ) -> None:
        if predictor not in {"ridge", "neural"}:
            raise ValueError(predictor)
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.lags = tuple(sorted(lags))
        self.use_calendar = bool(use_calendar)
        self.min_samples = int(min_samples)
        self.grid_size = int(grid_size)
        self.predictor = predictor
        self.nn_hidden_dim = int(nn_hidden_dim)
        self.nn_epochs = int(nn_epochs)
        self.nn_lr = float(nn_lr)
        self._grid = np.linspace(0.0, 1.0, self.grid_size)
        self._pca: PCA | None = None
        self._train_scores: np.ndarray | None = None
        self._train_emb_df: pd.DataFrame | None = None
        self._models: dict[int, dict] = {}
        self._train_idx_map: dict[tuple[int, object], int] = {}

    def params(self) -> dict:
        return {
            "model": self.name, "n_components": self.n_components,
            "alpha": self.alpha,
            "lags": "-".join(str(x) for x in self.lags),
            "grid_size": self.grid_size,
        }

    def fit(self, train_embeddings: pd.DataFrame) -> LogDerivFPCAPredictor:
        train = train_embeddings.reset_index(drop=True)
        self._train_emb_df = train
        L = _stack_log_deriv(train, self._grid)
        self._pca = PCA(n_components=self.n_components, random_state=0)
        self._train_scores = self._pca.fit_transform(L)
        self._train_idx_map = _index(train)
        for interval, sub in train.groupby("Interval Number"):
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                feat = self._build_features(
                    int(interval), row["Trading Date"], train, self._train_idx_map,
                )
                if feat is None:
                    continue
                tidx = self._train_idx_map[(int(row["Interval Number"]), row["Trading Date"])]
                X_rows.append(feat)
                Y_rows.append(self._train_scores[tidx])
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows); Y = np.vstack(Y_rows)
            pipe = Pipeline([("scaler", StandardScaler())])
            Xp = pipe.fit_transform(X)
            y_scaler = StandardScaler()
            Ys = y_scaler.fit_transform(Y)
            if self.predictor == "ridge":
                est = Ridge(alpha=self.alpha)
                est.fit(Xp, Ys)
                predict_fn = est.predict
            else:
                from ._nn import _train_mlp
                predict_fn = _train_mlp(
                    Xp.astype(np.float32), Ys.astype(np.float32),
                    hidden_dim=self.nn_hidden_dim, epochs=self.nn_epochs,
                    batch_size=256, lr=self.nn_lr, weight_decay=1e-4, seed=0,
                )
            self._models[int(interval)] = {
                "x_pipeline": pipe, "y_scaler": y_scaler, "predict_fn": predict_fn,
            }
        return self

    def _build_features(
        self, interval: int, trading_date, hist: pd.DataFrame, hist_lookup: dict,
    ) -> np.ndarray | None:
        ts = pd.Timestamp(trading_date)
        parts: list[np.ndarray] = []
        for lag in self.lags:
            key = (interval, (ts - pd.Timedelta(days=lag)).date())
            tidx = self._train_idx_map.get(key)
            if tidx is not None:
                parts.append(self._train_scores[tidx])
                continue
            hidx = hist_lookup.get(key)
            if hidx is None:
                return None
            row = hist.iloc[hidx]
            L = _curve_to_logderiv_q(row["curve_qty"], row["curve_price"], self._grid)
            parts.append(self._pca.transform(L.reshape(1, -1))[0])
        if self.use_calendar:
            parts.append(_calendar_features(trading_date))
        return np.concatenate(parts, axis=0)

    def predict(
        self, test_embeddings: pd.DataFrame, historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_lookup = _index(hist)
        rows: list[dict] = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            mdl = self._models.get(interval)
            if mdl is None:
                continue
            feat = self._build_features(interval, r["Trading Date"], hist, hist_lookup)
            if feat is None:
                continue
            Xp = mdl["x_pipeline"].transform(feat.reshape(1, -1))
            Ys = mdl["predict_fn"](Xp.astype(np.float32) if self.predictor == "neural" else Xp)
            pred_scores = mdl["y_scaler"].inverse_transform(np.atleast_2d(Ys))[0]
            # Snap to nearest training curve in score space
            d = np.linalg.norm(self._train_scores - pred_scores.reshape(1, -1), axis=1)
            best = int(np.argmin(d))
            src = self._train_emb_df.iloc[best]
            rows.append({
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_embedding": r["embedding"],
                "pred_embedding": np.asarray(src["embedding"], dtype=float),
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": src["curve_qty"],
                "pred_curve_price": src["curve_price"],
                "retrieved_date": src["Trading Date"],
                "score_distance": float(d[best]),
            })
        return pd.DataFrame(rows)
