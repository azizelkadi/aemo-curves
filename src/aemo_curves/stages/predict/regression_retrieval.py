"""Per-interval regression on lagged embeddings, then L2 retrieval of the
nearest training curve. One base class drives five presets (ridge, gbm, rf,
xgb, neural) that differ only in the estimator, the X-pipeline form, optional
PCA on Y, and the neural PCA seed."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aemo_curves.core.predictor import CurvePredictor

from ._features import feature_from_lags
from ._shared import key_index, l2_retrieve, library_by_interval, retrieval_row


class EmbeddingRegressionRetrievalPredictor(CurvePredictor):
    name = "embedding_regression_retrieval"

    def __init__(
        self,
        estimator_factory: Callable[[int], object],
        *,
        lags: tuple[int, ...] = (1, 2, 7, 14),
        n_pca_x: int | None = 20,
        n_pca_y: int | None = None,
        min_samples: int = 20,
        use_calendar: bool = True,
        use_trend: bool = True,
        use_qty_total: bool = False,
        exog_provider=None,
        pipeline_x: bool = False,
        pca_random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.lags = tuple(sorted(lags))
        self.n_pca_x = n_pca_x
        self.n_pca_y = n_pca_y
        self.min_samples = int(min_samples)
        self.use_calendar = bool(use_calendar)
        self.use_trend = bool(use_trend)
        self.use_qty_total = bool(use_qty_total)
        self.exog_provider = exog_provider
        self.pipeline_x = bool(pipeline_x)
        self.pca_random_state = pca_random_state
        if name is not None:
            self.name = name
        self._models: dict[int, dict] = {}
        self._library_by_interval: dict[int, pd.DataFrame] = {}

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "n_pca_x": self.n_pca_x if self.n_pca_x is not None else "none",
            "n_pca_y": self.n_pca_y if self.n_pca_y is not None else "none",
            "use_calendar": self.use_calendar,
            "use_trend": self.use_trend,
        }

    def _new_x_pipeline(self, X: np.ndarray) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        if self.n_pca_x is not None:
            components = min(self.n_pca_x, X.shape[0] - 1, X.shape[1])
            if components > 0:
                if self.pca_random_state is not None:
                    steps.append(("pca", PCA(n_components=components, random_state=self.pca_random_state)))
                else:
                    steps.append(("pca", PCA(n_components=components)))
        return Pipeline(steps)

    def _feat(self, interval, trading_date, hist, hist_idx):
        return feature_from_lags(
            interval, trading_date, hist, hist_idx,
            self.lags, self.use_calendar, self.use_trend,
            use_qty_total=self.use_qty_total, exog_provider=self.exog_provider,
        )

    def fit(self, train_embeddings: pd.DataFrame) -> EmbeddingRegressionRetrievalPredictor:
        train = train_embeddings.reset_index(drop=True)
        self._library_by_interval = library_by_interval(train)
        idx = key_index(train)
        for interval in sorted(self._library_by_interval):
            sub = self._library_by_interval[interval]
            X_rows, Y_rows = [], []
            for _, row in sub.iterrows():
                feat = self._feat(interval, row["Trading Date"], train, idx)
                if feat is None:
                    continue
                X_rows.append(feat)
                Y_rows.append(np.asarray(row["embedding"], dtype=float))
            if len(X_rows) < self.min_samples:
                continue
            X = np.vstack(X_rows)
            Y = np.vstack(Y_rows)
            if self.pipeline_x:
                x_pipe = self._new_x_pipeline(X)
                Xp = x_pipe.fit_transform(X)
                y_scaler = StandardScaler()
                Ys = y_scaler.fit_transform(Y)
                est = self.estimator_factory(Ys.shape[1])
                est.fit(Xp, Ys)
                self._models[interval] = {
                    "x_pipeline": x_pipe, "y_scaler": y_scaler, "estimator": est,
                }
            else:
                x_scaler = StandardScaler().fit(X)
                Xs = x_scaler.transform(X)
                pca_x = None
                Xt = Xs
                if self.n_pca_x is not None:
                    comp = min(self.n_pca_x, X.shape[0] - 1, X.shape[1])
                    if comp > 0:
                        pca_x = PCA(n_components=comp).fit(Xs)
                        Xt = pca_x.transform(Xs)
                y_scaler = StandardScaler().fit(Y)
                Ys = y_scaler.transform(Y)
                pca_y = None
                Yt = Ys
                if self.n_pca_y is not None:
                    comp = min(self.n_pca_y, Y.shape[0] - 1, Y.shape[1])
                    if comp > 0:
                        pca_y = PCA(n_components=comp).fit(Ys)
                        Yt = pca_y.transform(Ys)
                est = self.estimator_factory(Yt.shape[1])
                est.fit(Xt, Yt)
                self._models[interval] = {
                    "x_scaler": x_scaler, "pca_x": pca_x,
                    "y_scaler": y_scaler, "pca_y": pca_y, "estimator": est,
                }
        return self

    def predict_embedding_for_row(self, row, hist, hist_idx) -> np.ndarray | None:
        interval = int(row["Interval Number"])
        mdl = self._models.get(interval)
        if mdl is None:
            return None
        feat = self._feat(interval, row["Trading Date"], hist, hist_idx)
        if feat is None:
            return None
        if self.pipeline_x:
            Xp = mdl["x_pipeline"].transform(feat.reshape(1, -1))
            Yp = mdl["estimator"].predict(Xp)
            return mdl["y_scaler"].inverse_transform(np.atleast_2d(Yp))[0]
        Xs = mdl["x_scaler"].transform(feat.reshape(1, -1))
        Xt = mdl["pca_x"].transform(Xs) if mdl["pca_x"] is not None else Xs
        Yt = mdl["estimator"].predict(Xt)
        if Yt.ndim == 1:
            Yt = Yt.reshape(1, -1)
        Ys = mdl["pca_y"].inverse_transform(Yt) if mdl["pca_y"] is not None else Yt
        return mdl["y_scaler"].inverse_transform(Ys)[0]

    def predict_embedding(self, row, hist, hist_idx) -> np.ndarray | None:
        return self.predict_embedding_for_row(row, hist, hist_idx)

    def predict(self, test_embeddings, historical_embeddings) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_idx = key_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            mdl = self._models.get(interval)
            library = self._library_by_interval.get(interval)
            if mdl is None or library is None or len(library) == 0:
                continue
            pred_emb = self.predict_embedding_for_row(r, hist, hist_idx)
            if pred_emb is None:
                continue
            best, distances = l2_retrieve(library, pred_emb)
            src = library.iloc[best]
            rows.append(retrieval_row(r, interval, pred_emb, src, distances[best]))
        return pd.DataFrame(rows)


def _ridge_factory(alpha):
    from sklearn.linear_model import Ridge
    return lambda n_out: Ridge(alpha=alpha)


class RidgeRetrievalPredictor(EmbeddingRegressionRetrievalPredictor):
    name = "ridge_retrieval"

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 7, 14),
        alpha: float = 2.0,
        n_pca: int | None = 20,
        min_samples: int = 20,
        use_calendar: bool = True,
        use_trend: bool = True,
        use_qty_total: bool = False,
        exog_provider=None,
    ) -> None:
        self.alpha = float(alpha)
        self.n_pca = n_pca
        super().__init__(
            _ridge_factory(self.alpha),
            lags=lags, n_pca_x=n_pca, n_pca_y=None, min_samples=min_samples,
            use_calendar=use_calendar, use_trend=use_trend, use_qty_total=use_qty_total,
            exog_provider=exog_provider, pipeline_x=True, pca_random_state=None,
            name="ridge_retrieval",
        )

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "alpha": self.alpha,
            "n_pca": self.n_pca if self.n_pca is not None else "none",
            "min_samples": self.min_samples,
            "use_calendar": self.use_calendar,
            "use_trend": self.use_trend,
        }


class GBMRetrievalPredictor(EmbeddingRegressionRetrievalPredictor):
    name = "gbm_retrieval"

    def __init__(
        self,
        lags: tuple[int, ...] = (30,),
        n_pca_x: int | None = 20,
        n_pca_y: int | None = 8,
        max_iter: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        min_samples: int = 50,
        use_calendar: bool = True,
        use_trend: bool = True,
    ) -> None:
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)

        def factory(n_outputs):
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

        super().__init__(
            factory, lags=lags, n_pca_x=n_pca_x, n_pca_y=n_pca_y, min_samples=min_samples,
            use_calendar=use_calendar, use_trend=use_trend,
            use_qty_total=False, exog_provider=None, pipeline_x=False, name="gbm_retrieval",
        )

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
            "use_calendar": self.use_calendar,
            "use_trend": self.use_trend,
        }


class RFRetrievalPredictor(EmbeddingRegressionRetrievalPredictor):
    name = "rf_retrieval"

    def __init__(
        self,
        lags: tuple[int, ...] = (7,),
        n_pca_x: int | None = 20,
        n_pca_y: int | None = None,
        n_estimators: int = 300,
        max_depth: int | None = 16,
        min_samples_leaf: int = 5,
        min_samples: int = 50,
        use_calendar: bool = True,
        use_trend: bool = True,
        use_qty_total: bool = False,
        exog_provider=None,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        def factory(n_outputs):
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        super().__init__(
            factory, lags=lags, n_pca_x=n_pca_x, n_pca_y=n_pca_y, min_samples=min_samples,
            use_calendar=use_calendar, use_trend=use_trend, use_qty_total=use_qty_total,
            exog_provider=exog_provider, pipeline_x=False, name="rf_retrieval",
        )

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "n_pca_x": self.n_pca_x if self.n_pca_x is not None else "none",
            "n_pca_y": self.n_pca_y if self.n_pca_y is not None else "none",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth if self.max_depth is not None else "none",
            "min_samples_leaf": self.min_samples_leaf,
            "use_calendar": self.use_calendar,
            "use_trend": self.use_trend,
        }


class XGBRetrievalPredictor(EmbeddingRegressionRetrievalPredictor):
    name = "xgb_retrieval"

    def __init__(
        self,
        lags: tuple[int, ...] = (7,),
        n_pca_x: int | None = 20,
        n_pca_y: int | None = None,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_samples: int = 50,
        use_calendar: bool = True,
        use_trend: bool = True,
        use_qty_total: bool = False,
        exog_provider=None,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        def factory(n_outputs):
            from xgboost import XGBRegressor
            try:
                return XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_lambda=self.reg_lambda,
                    tree_method="hist",
                    multi_strategy="multi_output_tree",
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbosity=0,
                )
            except TypeError:
                from sklearn.multioutput import MultiOutputRegressor
                base = XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_lambda=self.reg_lambda,
                    tree_method="hist",
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbosity=0,
                )
                return MultiOutputRegressor(base)

        super().__init__(
            factory, lags=lags, n_pca_x=n_pca_x, n_pca_y=n_pca_y, min_samples=min_samples,
            use_calendar=use_calendar, use_trend=use_trend, use_qty_total=use_qty_total,
            exog_provider=exog_provider, pipeline_x=False, name="xgb_retrieval",
        )

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "n_pca_x": self.n_pca_x if self.n_pca_x is not None else "none",
            "n_pca_y": self.n_pca_y if self.n_pca_y is not None else "none",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "use_calendar": self.use_calendar,
            "use_trend": self.use_trend,
        }


class NeuralRetrievalPredictor(EmbeddingRegressionRetrievalPredictor):
    name = "neural_retrieval"

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 7, 14),
        n_pca: int | None = 20,
        min_samples: int = 20,
        use_calendar: bool = True,
        use_trend: bool = True,
        use_qty_total: bool = False,
        hidden_dim: int = 64,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        exog_provider=None,
    ) -> None:
        from ._nn import _MLPEstimator
        self.n_pca = n_pca
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        mlp_kwargs = dict(
            hidden_dim=self.hidden_dim, epochs=self.epochs,
            batch_size=self.batch_size, lr=self.lr,
            weight_decay=self.weight_decay, seed=self.seed,
        )
        super().__init__(
            lambda n_out: _MLPEstimator(**mlp_kwargs),
            lags=lags, n_pca_x=n_pca, n_pca_y=None, min_samples=min_samples,
            use_calendar=use_calendar, use_trend=use_trend, use_qty_total=use_qty_total,
            exog_provider=exog_provider, pipeline_x=True, pca_random_state=self.seed,
            name="neural_retrieval",
        )

    def params(self) -> dict:
        return {
            "model": self.name,
            "lags": "-".join(str(x) for x in self.lags),
            "n_pca": self.n_pca if self.n_pca is not None else "none",
            "hidden_dim": self.hidden_dim, "epochs": self.epochs,
            "lr": self.lr, "weight_decay": self.weight_decay,
            "use_calendar": self.use_calendar, "use_trend": self.use_trend,
            "use_qty_total": self.use_qty_total,
        }
