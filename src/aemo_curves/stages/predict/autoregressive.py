"""Autoregressive self-prediction features for the v_grid ridge-query retrieval.

Motivation
----------
The base ``rolling_refit`` model forecasts day ``D`` (interval ``I``) from the
forecast origin ``D - H`` (``H`` = the horizon, 7 days). Its only curve input is
the *real* curve at ``D - H``. The days in between, ``D-1 .. D-(H-1)``, are
closer to the target and more informative, but at the origin they **have not
happened yet**, so their real curves do not exist. This strategy fills that gap
with the model's own predictions for those intermediate days and feeds them as
extra features - i.e. one-step stacking, *not* recursion (stage-1 predictions
never consume stage-1 predictions).

Two stages
----------
1. **Backfill (stage 1).** Run the ordinary base retrieval walk-forward over the
   whole timeline (train + test): for every calendar month, fit on
   ``historical[date <= month_start - H]`` and predict every row of that month.
   This yields, for every ``(Trading Date, Interval Number)``, the base model's
   prediction ``pred(X)`` that respects the horizon (``pred(X)`` is a function of
   data ``<= X - H`` only). At ``k=1`` the ``pred_embedding`` column *is* the
   v_grid embedding of the predicted curve, so no re-embedding is needed. Only
   the prediction columns are kept - the ``true_*`` columns are dropped so the
   truth can never leak into a feature.
2. **Augmented refit (stage 2).** Add a :class:`SelfPredictionSource` to the base
   exogenous provider and re-run the *unchanged* ``rolling_refit`` stage-2 code
   path (via :func:`dataclasses.replace` on the context). For target ``(D, I)``
   the source looks up ``pred(D-1), pred(D-2), pred(D-3)`` from the backfill,
   pools each 64-d embedding into 8 block-means (metric-aligned coarse shape),
   and appends a per-lag fallback flag.

Leakage
-------
``pred(D-k)`` is a function of data ``<= D-k-H`` (``< D-H`` for ``k>=1``), so it is
available at the origin ``D-H`` - strictly leakage-safe, and it injects genuinely
new information (the real curves at ``D-H-1 .. D-H-3``, which the base model does
not otherwise see). When a backfill prediction is missing (earliest history), the
source falls back to the *real* embedding at ``(I, D-k-H)`` (dated ``<= D-H-1``,
also safe) and sets the fallback flag; only if that is missing too does it return
``None`` (dropping the row). See ``autoregressive`` tests for the invariants.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.exogenous.provider import ExogenousProvider, ExogenousSource
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict._orchestration import rolling_refit
from aemo_curves.stages.predict.soft_knn import StratifiedKNNRetrievalPredictor

log = logging.getLogger(__name__)


def _as_date(value) -> object:
    """Normalise any date-like value to a ``datetime.date`` (matches hist_index)."""
    return pd.Timestamp(value).date()


class SelfPredictionSource(ExogenousSource):
    """Exogenous source: the base model's own predictions for the future gap days.

    For target ``(interval, D)`` and each lag ``k`` in ``lags``, look up the
    backfill prediction at ``(interval, D - k)`` and pool its 64-d embedding into
    ``n_blocks`` block-means; append a per-lag fallback flag. If the backfill has
    no prediction at ``(interval, D - k)``, fall back to the *real* embedding at
    ``(interval, D - k - lag_days)`` (flag = 1). Return ``None`` (drop the row)
    only if that real embedding is also missing.
    """

    def __init__(
        self,
        backfill: pd.DataFrame,
        historical: pd.DataFrame,
        *,
        lags: tuple[int, ...] = (1, 2, 3),
        n_blocks: int = 8,
        lag_days: int = 7,
    ) -> None:
        self.lags = tuple(int(k) for k in lags)
        self.n_blocks = int(n_blocks)
        self.lag_days = int(lag_days)

        # (interval, date) -> predicted 64-d embedding (stage-1 backfill)
        self._pred: dict[tuple[int, object], np.ndarray] = {}
        for _, r in backfill.iterrows():
            emb = r.get("pred_embedding")
            if emb is None:
                continue
            self._pred[(int(r["Interval Number"]), _as_date(r["Trading Date"]))] = np.asarray(
                emb, dtype=float
            )

        # (interval, date) -> real 64-d embedding (fallback source)
        self._real: dict[tuple[int, object], np.ndarray] = {}
        for _, r in historical.iterrows():
            emb = r.get("embedding")
            if emb is None:
                continue
            self._real[(int(r["Interval Number"]), _as_date(r["Trading Date"]))] = np.asarray(
                emb, dtype=float
            )

        # Validate the embedding width divides into n_blocks (fail fast).
        sample = next(iter(self._real.values()), None)
        if sample is None:
            raise ValueError("SelfPredictionSource: historical has no embeddings")
        width = int(sample.shape[0])
        if width % self.n_blocks != 0:
            raise ValueError(
                f"embedding width {width} not divisible by n_blocks {self.n_blocks}"
            )
        self._width = width

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for k in self.lags:
            names.extend(f"selfpred_L{k}_b{b}" for b in range(self.n_blocks))
            names.append(f"selfpred_L{k}_fallback")
        return names

    def _pool(self, emb: np.ndarray) -> np.ndarray:
        return np.asarray(emb, dtype=float).reshape(self.n_blocks, -1).mean(axis=1)

    def features(self, interval: int, trading_date) -> np.ndarray | None:
        D = pd.Timestamp(trading_date)
        interval = int(interval)
        out: list[float] = []
        for k in self.lags:
            target = (D - pd.Timedelta(days=k)).date()  # D - k
            emb = self._pred.get((interval, target))
            fallback = 0.0
            if emb is None:
                fb_date = (D - pd.Timedelta(days=k + self.lag_days)).date()  # D - k - H
                emb = self._real.get((interval, fb_date))
                fallback = 1.0
                if emb is None:
                    return None
            out.extend(self._pool(emb).tolist())
            out.append(fallback)
        return np.asarray(out, dtype=float)


def _walkforward_backfill(
    historical: pd.DataFrame,
    *,
    lag_days: int,
    knn_kwargs: dict,
    exog_provider,
) -> pd.DataFrame:
    """Walk-forward monthly base predictions over the whole timeline.

    For every calendar month spanned by ``historical``, fit the base retrieval on
    ``historical[date <= month_start - lag_days]`` and predict every row of that
    month. Returns a frame with only ``Trading Date``, ``Interval Number``,
    ``pred_embedding`` and ``retrieved_date`` (all ``true_*`` columns dropped so
    the truth cannot leak into a downstream feature).
    """
    hist_ts = pd.to_datetime(historical["Trading Date"])
    start = hist_ts.min().normalize().replace(day=1)
    end = hist_ts.max()
    refits = pd.date_range(start=start, end=end, freq="MS")
    window_ends = list(refits[1:]) + [end + pd.Timedelta(days=1)]

    parts: list[pd.DataFrame] = []
    for T_refit, T_next in zip(refits, window_ends):
        cutoff = T_refit - pd.Timedelta(days=lag_days)
        fit_data = historical[hist_ts <= cutoff].reset_index(drop=True)
        if len(fit_data) == 0:
            continue
        win = historical[(hist_ts >= T_refit) & (hist_ts < T_next)].reset_index(drop=True)
        if len(win) == 0:
            continue
        model = StratifiedKNNRetrievalPredictor(
            k=1,
            lag_days=lag_days,
            ridge_lags=(lag_days,),
            output_mode="top1_curve",
            exog_provider=exog_provider,
            **knn_kwargs,
        )
        model.fit(fit_data)
        preds = model.predict(win, historical)
        if len(preds) == 0:
            continue
        parts.append(preds[["Trading Date", "Interval Number", "pred_embedding", "retrieved_date"]])

    if not parts:
        raise RuntimeError("_walkforward_backfill produced no predictions")
    backfill = pd.concat(parts, ignore_index=True)
    log.info(
        "backfill: %d predictions over %s..%s",
        len(backfill),
        _as_date(backfill["Trading Date"].min()),
        _as_date(backfill["Trading Date"].max()),
    )
    return backfill


@register("predict", "rolling_refit_ar")
def rolling_refit_ar(
    ctx: PredictContext,
    query: str = "ridge",
    ridge_alpha: float = 200.0,
    ridge_n_pca: int | None = 20,
    bucket: str = "interval_dow",
    pca_mode: str = "joint",
    gbm_kwargs: dict | None = None,
    self_pred_lags: tuple[int, ...] = (1, 2, 3),
    self_pred_blocks: int = 8,
):
    """Autoregressive variant of ``rolling_refit``.

    Stage 1: walk-forward base backfill over the whole timeline. Stage 2: add a
    :class:`SelfPredictionSource` to the base exogenous provider and reuse the
    unchanged ``rolling_refit`` stage-2 path. All predict.params of the base
    experiment carry over unchanged; ``self_pred_lags`` / ``self_pred_blocks``
    configure the new feature block.
    """
    knn_kwargs = dict(
        query=query,
        bucket=bucket,
        ridge_alpha=ridge_alpha,
        ridge_n_pca=ridge_n_pca,
        pca_mode=pca_mode,
        gbm_kwargs=dict(gbm_kwargs) if gbm_kwargs else None,
    )

    # Stage 1 - base backfill (base exog provider only; no self-prediction).
    backfill = _walkforward_backfill(
        ctx.historical,
        lag_days=ctx.lag_days,
        knn_kwargs=knn_kwargs,
        exog_provider=ctx.exog_provider,
    )

    # Stage 2 - add the self-prediction source and reuse rolling_refit unchanged.
    self_src = SelfPredictionSource(
        backfill,
        ctx.historical,
        lags=tuple(self_pred_lags),
        n_blocks=int(self_pred_blocks),
        lag_days=ctx.lag_days,
    )
    base_sources = list(ctx.exog_provider.sources) if ctx.exog_provider is not None else []
    combined = ExogenousProvider(base_sources + [self_src])
    ctx2 = dataclasses.replace(ctx, exog_provider=combined)

    return rolling_refit(
        ctx2,
        query=query,
        ridge_alpha=ridge_alpha,
        ridge_n_pca=ridge_n_pca,
        bucket=bucket,
        gbm_kwargs=gbm_kwargs,
        pca_mode=pca_mode,
    )
