"""Predict strategies that do more than wrap one estimator.

`static_retrieval` is a fit-once stratified k=1 retrieval at the experiment's lag.
`rolling_refit` re-fits that retrieval at the start of every test month on data
old enough to respect the forecast horizon, then concatenates the per-window
predictions. Both return predictions only; the runner scores them.
"""
from __future__ import annotations

import logging

import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict.soft_knn import StratifiedKNNRetrievalPredictor

log = logging.getLogger(__name__)


def _strat_knn(ctx: PredictContext, query: str, ridge_alpha: float, ridge_n_pca, bucket: str):
    return StratifiedKNNRetrievalPredictor(
        k=1,
        query=query,
        bucket=bucket,
        ridge_alpha=ridge_alpha,
        ridge_n_pca=ridge_n_pca,
        ridge_lags=(ctx.lag_days,),
        lag_days=ctx.lag_days,
        output_mode="top1_curve",
        exog_provider=ctx.exog_provider,
    )


@register("predict", "static_retrieval")
def static_retrieval(
    ctx: PredictContext,
    query: str = "ridge",
    ridge_alpha: float = 200.0,
    ridge_n_pca: int | None = 20,
    bucket: str = "interval_dow",
):
    model = _strat_knn(ctx, query, ridge_alpha, ridge_n_pca, bucket)
    model.fit(ctx.train)
    return model.predict(ctx.test, ctx.historical)


def _monthly_refit_dates(test: pd.DataFrame) -> list[pd.Timestamp]:
    ts = pd.to_datetime(test["Trading Date"])
    start = ts.min().normalize().replace(day=1)
    return list(pd.date_range(start=start, end=ts.max(), freq="MS"))


@register("predict", "rolling_refit")
def rolling_refit(
    ctx: PredictContext,
    query: str = "ridge",
    ridge_alpha: float = 200.0,
    ridge_n_pca: int | None = 20,
    bucket: str = "interval_dow",
    gbm_kwargs: dict | None = None,
):
    historical = ctx.historical
    hist_ts = pd.to_datetime(historical["Trading Date"])
    test_ts = pd.to_datetime(ctx.test["Trading Date"])

    refits = _monthly_refit_dates(ctx.test)
    end_sentinel = test_ts.max() + pd.Timedelta(days=1)
    window_ends = list(refits[1:]) + [end_sentinel]

    parts: list[pd.DataFrame] = []
    for T_refit, T_next in zip(refits, window_ends):
        cutoff = T_refit - pd.Timedelta(days=ctx.lag_days)
        fit_data = historical[hist_ts <= cutoff].reset_index(drop=True)
        win_test = ctx.test[(test_ts >= T_refit) & (test_ts < T_next)].reset_index(drop=True)
        if len(win_test) == 0:
            continue
        model = StratifiedKNNRetrievalPredictor(
            k=1,
            query=query,
            bucket=bucket,
            ridge_alpha=ridge_alpha,
            ridge_n_pca=ridge_n_pca,
            ridge_lags=(ctx.lag_days,),
            lag_days=ctx.lag_days,
            output_mode="top1_curve",
            exog_provider=ctx.exog_provider,
            gbm_kwargs=dict(gbm_kwargs) if gbm_kwargs else None,
        )
        model.fit(fit_data)
        preds = model.predict(win_test, historical)
        preds["refit_date"] = T_refit
        parts.append(preds)

    if not parts:
        raise RuntimeError("rolling_refit produced no predictions; check the test window")
    return pd.concat(parts, ignore_index=True)
