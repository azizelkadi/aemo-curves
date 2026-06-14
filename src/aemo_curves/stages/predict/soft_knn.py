"""Soft / stratified k-NN retrieval over the 64-d interp embedding library.

Both predictors here aim to **close the gap to the oracle** in
embedding-space retrieval (`KNNRetrievalPredictor(query="oracle")`):

- `SoftKNNRetrievalPredictor` averages the curves of the top-K nearest
  training neighbours instead of picking only the top-1, smoothing out
  embedding-prediction noise.
- `StratifiedKNNRetrievalPredictor` restricts the candidate library to
  rows from the **same interval and same day-of-week** as the target -
  much fewer but much more relevant candidates.

Both can be combined: a stratified soft k-NN with a Ridge-predicted
query embedding is the strongest expected configuration for closing the
oracle gap.

Output curve representation
---------------------------
Each predictor emits the **actual step (qty, price)** of the top-1
retrieved row, preserving the real step structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor

from ._features import hist_index, lag_row


def _index_by_interval(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(i): sub.reset_index(drop=True) for i, sub in df.groupby("Interval Number")}


def _query_embedding(
    query_mode: str,
    row: pd.Series,
    hist: pd.DataFrame,
    hist_idx: dict,
    ridge_predictor=None,
    lag_days: int = 1,
) -> np.ndarray | None:
    """Query embedding for retrieval.

    `query_mode` ∈
      - ``"oracle"``: row's true embedding (diagnostic)
      - ``"lag"``:    embedding at (interval, date - `lag_days`)
      - ``"lag1"``:   alias of `query="lag", lag_days=1`
      - ``"ridge"``:  fitted Ridge predictor's predicted embedding
    """
    if query_mode == "oracle":
        return np.asarray(row["embedding"], dtype=float)
    if query_mode in ("lag", "lag1"):
        interval = int(row["Interval Number"])
        lag = int(lag_days) if query_mode == "lag" else 1
        lag_r = lag_row(hist, hist_idx, interval, row["Trading Date"], lag)
        if lag_r is None:
            return None
        return np.asarray(lag_r["embedding"], dtype=float)
    if query_mode in ("ridge", "gbm", "neural", "xgboost", "rf"):
        if ridge_predictor is None:
            raise ValueError(f"query_mode={query_mode!r} requires a fitted query predictor")
        return ridge_predictor.predict_embedding(row, hist, hist_idx)
    raise ValueError(query_mode)


def _softmax_weights(distances: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        # Hard top-1 (uniform weight on single nearest)
        w = np.zeros_like(distances)
        w[int(np.argmin(distances))] = 1.0
        return w
    z = -distances / float(tau)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


_STRATIFIED_QUERY_MODELS = frozenset({"ridge", "gbm", "neural", "xgboost", "rf"})


def _build_stratified_query_model(self):
    q = self.query
    if q == "ridge":
        from .regression_retrieval import RidgeRetrievalPredictor
        return RidgeRetrievalPredictor(
            lags=self.ridge_lags, alpha=self.ridge_alpha, n_pca=self.ridge_n_pca,
            use_calendar=True, use_trend=True,
            use_qty_total=self.ridge_use_qty_total, exog_provider=self.exog_provider,
        )
    if q == "gbm":
        from .regression_retrieval import GBMRetrievalPredictor
        return GBMRetrievalPredictor(lags=self.ridge_lags, **self.gbm_kwargs)
    if q == "neural":
        from .regression_retrieval import NeuralRetrievalPredictor
        return NeuralRetrievalPredictor(
            lags=self.ridge_lags, n_pca=self.ridge_n_pca, use_calendar=True, use_trend=True,
            use_qty_total=self.ridge_use_qty_total, exog_provider=self.exog_provider,
            **self.gbm_kwargs,
        )
    if q == "xgboost":
        from .regression_retrieval import XGBRetrievalPredictor
        return XGBRetrievalPredictor(
            lags=self.ridge_lags, n_pca_x=self.ridge_n_pca, use_calendar=True, use_trend=True,
            use_qty_total=self.ridge_use_qty_total, exog_provider=self.exog_provider,
            **self.gbm_kwargs,
        )
    if q == "rf":
        from .regression_retrieval import RFRetrievalPredictor
        return RFRetrievalPredictor(
            lags=self.ridge_lags, n_pca_x=self.ridge_n_pca, use_calendar=True, use_trend=True,
            use_qty_total=self.ridge_use_qty_total, exog_provider=self.exog_provider,
            **self.gbm_kwargs,
        )
    raise ValueError(q)


class SoftKNNRetrievalPredictor(CurvePredictor):
    """k-NN retrieval with soft (weighted) averaging over the top-K neighbours.

    Library = all training rows of the same `Interval Number`. Query
    embedding can be the lag-1 embedding (`lag1`), the Ridge-predicted
    embedding (`ridge`), or the **true** test embedding (`oracle`,
    diagnostic).

    Output mode (`output_mode`):

    - ``"top1_curve"`` (default): the predicted curve is the **actual
      step (qty, price)** of the top-1 retrieved row. Preserves the step
      structure exactly - at k=1 this is identical to
      :class:`KNNRetrievalPredictor`. For k>1 the soft weights affect
      the *ranking* via a weighted-distance argmin (top-1 still wins by
      L2, so this mode is effectively the same as KNNRetrievalPredictor
      when we use the same query embedding).
    """

    name = "soft_knn_retrieval"

    def __init__(
        self,
        k: int = 10,
        query: str = "ridge",
        tau: float = 5.0,
        ridge_alpha: float = 2.0,
        ridge_n_pca: int | None = 20,
        ridge_lags: tuple[int, ...] = (1, 2, 7, 14),
        lag_days: int = 1,
        output_mode: str = "top1_curve",
    ) -> None:
        if query not in {"oracle", "lag", "lag1", "ridge"}:
            raise ValueError(query)
        if output_mode not in {"top1_curve"}:
            raise ValueError(output_mode)
        self.k = int(k)
        self.query = query
        self.tau = float(tau)
        self.ridge_alpha = float(ridge_alpha)
        self.ridge_n_pca = ridge_n_pca
        self.ridge_lags = tuple(ridge_lags)
        self.lag_days = int(lag_days)
        self.output_mode = output_mode
        self._library_by_interval: dict[int, pd.DataFrame] = {}
        self._ridge_adapter = None

    def params(self) -> dict:
        return {
            "model": self.name, "k": self.k, "query": self.query,
            "tau": self.tau, "output_mode": self.output_mode,
            "lag_days": self.lag_days,
            "ridge_lags": "-".join(str(x) for x in self.ridge_lags),
        }

    def fit(self, train_embeddings: pd.DataFrame) -> SoftKNNRetrievalPredictor:
        self._library_by_interval = _index_by_interval(train_embeddings)
        if self.query == "ridge":
            from .regression_retrieval import RidgeRetrievalPredictor
            inner = RidgeRetrievalPredictor(
                lags=self.ridge_lags,
                alpha=self.ridge_alpha,
                n_pca=self.ridge_n_pca,
                use_calendar=True, use_trend=True,
            )
            inner.fit(train_embeddings)
            self._ridge_adapter = inner
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_idx = hist_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            library = self._library_by_interval.get(interval)
            if library is None or len(library) == 0:
                continue
            q = _query_embedding(
                self.query, r, hist, hist_idx, self._ridge_adapter,
                lag_days=self.lag_days,
            )
            if q is None:
                continue
            cand = np.vstack(library["embedding"].to_numpy())
            d = np.linalg.norm(cand - q.reshape(1, -1), axis=1)
            k = min(self.k, len(d))
            top = np.argpartition(d, k - 1)[:k] if k > 1 else np.array([int(np.argmin(d))])
            top_d = d[top]
            w = _softmax_weights(top_d, self.tau)
            top_embs = cand[top]
            pred_emb = (w[:, None] * top_embs).sum(axis=0)
            top1 = int(top[int(np.argmin(top_d))])
            src = library.iloc[top1]
            pred_qty = np.asarray(src["curve_qty"], dtype=float)
            pred_price = np.asarray(src["curve_price"], dtype=float)
            rows.append({
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_embedding": r["embedding"],
                "pred_embedding": pred_emb,
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": pred_qty,
                "pred_curve_price": pred_price,
                "retrieved_date": src["Trading Date"],
                "n_neighbors_used": k,
                "weight_top1": float(w[int(np.argmin(top_d))]),
            })
        return pd.DataFrame(rows)


class StratifiedKNNRetrievalPredictor(CurvePredictor):
    """k-NN retrieval with the candidate library restricted to (interval, dow).

    Combines optionally with soft averaging (top-K + softmax weights) and
    a configurable query embedding (lag-1 / Ridge / oracle).
    """

    name = "stratified_knn_retrieval"

    def __init__(
        self,
        k: int = 5,
        query: str = "ridge",
        tau: float = 5.0,
        bucket: str = "interval_dow",   # 'interval_dow' or 'interval_weekend'
        ridge_alpha: float = 2.0,
        ridge_n_pca: int | None = 20,
        ridge_lags: tuple[int, ...] = (1, 2, 7, 14),
        ridge_use_qty_total: bool = False,
        lag_days: int = 1,
        output_mode: str = "top1_curve",
        pool_offsets: tuple[tuple[int, int], ...] = ((0, 0),),
        gbm_kwargs: dict | None = None,
        exog_provider=None,
    ) -> None:
        if query not in {"oracle", "lag", "lag1", "ridge", "gbm", "neural", "xgboost", "rf"}:
            raise ValueError(query)
        if bucket not in {"interval_dow", "interval_weekend"}:
            raise ValueError(bucket)
        if output_mode not in {"top1_curve"}:
            raise ValueError(output_mode)
        self.k = int(k)
        self.query = query
        self.tau = float(tau)
        self.bucket = bucket
        self.ridge_alpha = float(ridge_alpha)
        self.ridge_n_pca = ridge_n_pca
        self.ridge_lags = tuple(ridge_lags)
        self.ridge_use_qty_total = bool(ridge_use_qty_total)
        self.lag_days = int(lag_days)
        self.output_mode = output_mode
        self.pool_offsets = tuple((int(di), int(dd)) for di, dd in pool_offsets)
        self.gbm_kwargs = dict(gbm_kwargs or {})
        self.exog_provider = exog_provider
        self._library: dict = {}
        self._ridge_adapter = None  # adapter object (Ridge or GBM)

    def params(self) -> dict:
        return {
            "model": self.name, "k": self.k, "query": self.query,
            "tau": self.tau, "bucket": self.bucket,
            "output_mode": self.output_mode,
            "lag_days": self.lag_days,
            "ridge_lags": "-".join(str(x) for x in self.ridge_lags),
            "ridge_use_qty_total": self.ridge_use_qty_total,
            "pool_offsets": ";".join(f"{di},{dd}" for di, dd in self.pool_offsets),
        }

    def _key(self, interval: int, trading_date) -> tuple:
        ts = pd.Timestamp(trading_date)
        if self.bucket == "interval_dow":
            return (int(interval), int(ts.dayofweek))
        return (int(interval), int(ts.dayofweek >= 5))

    def _candidate_keys(self, interval: int, trading_date) -> list[tuple]:
        ts = pd.Timestamp(trading_date)
        out: list[tuple] = []
        for di, dd in self.pool_offsets:
            i2 = int(interval) + int(di)
            if i2 < 1 or i2 > 48:
                continue
            if self.bucket == "interval_dow":
                out.append((i2, (int(ts.dayofweek) + int(dd)) % 7))
            else:
                # interval_weekend: dd flips the weekend flag when |dd|>=1
                wknd = int(ts.dayofweek >= 5)
                if dd != 0:
                    wknd = 1 - wknd
                out.append((i2, wknd))
        # dedup, preserve order
        seen = set(); deduped = []
        for k in out:
            if k not in seen:
                seen.add(k); deduped.append(k)
        return deduped

    def fit(self, train_embeddings: pd.DataFrame) -> StratifiedKNNRetrievalPredictor:
        buckets: dict[tuple, list[int]] = {}
        for pos, (_, row) in enumerate(train_embeddings.iterrows()):
            buckets.setdefault(self._key(row["Interval Number"], row["Trading Date"]), []).append(pos)
        self._library = {k: train_embeddings.iloc[idx].reset_index(drop=True) for k, idx in buckets.items()}
        if self.query in _STRATIFIED_QUERY_MODELS:
            inner = _build_stratified_query_model(self)
            inner.fit(train_embeddings)
            self._ridge_adapter = inner
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_idx = hist_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            keys = self._candidate_keys(interval, r["Trading Date"])
            parts = [self._library[k] for k in keys if k in self._library and len(self._library[k]) > 0]
            if not parts:
                continue
            library = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            q = _query_embedding(
                self.query, r, hist, hist_idx, self._ridge_adapter,
                lag_days=self.lag_days,
            )
            if q is None:
                continue
            cand = np.vstack(library["embedding"].to_numpy())
            d = np.linalg.norm(cand - q.reshape(1, -1), axis=1)
            k = min(self.k, len(d))
            top = np.argpartition(d, k - 1)[:k] if k > 1 else np.array([int(np.argmin(d))])
            top_d = d[top]
            w = _softmax_weights(top_d, self.tau)
            top_embs = cand[top]
            pred_emb = (w[:, None] * top_embs).sum(axis=0)
            top1 = int(top[int(np.argmin(top_d))])
            src = library.iloc[top1]
            # Leakage-safe retrieval-quality features (query-to-neighbour distances).
            # These do NOT involve the truth at the test row and can therefore be
            # used downstream by a selector model. `retrieval_dist` is the
            # distance from query to the chosen (top-1) neighbour;
            # `retrieval_dist_2` is the distance from query to the second-nearest
            # neighbour in the full library (NaN if library has fewer than 2 rows).
            retrieval_dist = float(top_d.min())
            if len(d) >= 2:
                retrieval_dist_2 = float(np.partition(d, 1)[1])
            else:
                retrieval_dist_2 = float("nan")
            pred_qty = np.asarray(src["curve_qty"], dtype=float)
            pred_price = np.asarray(src["curve_price"], dtype=float)
            rows.append({
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_embedding": r["embedding"],
                "pred_embedding": pred_emb,
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": pred_qty,
                "pred_curve_price": pred_price,
                "retrieved_date": src["Trading Date"],
                "n_neighbors_used": k,
                "bucket_size": int(len(library)),
                "retrieval_dist": retrieval_dist,
                "retrieval_dist_2": retrieval_dist_2,
            })
        return pd.DataFrame(rows)
