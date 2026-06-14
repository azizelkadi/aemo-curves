"""Predict strategies that wrap one estimator: fit on train, predict on test.

Each registered name maps to an estimator class. The recipe `predict.params`
are the estimator's constructor arguments. Heavy backends are imported inside
their adapter so they load only when the strategy runs.
"""
from __future__ import annotations

import inspect

import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.predict.correctors import (
    PerDayShiftCorrector,
    ResidualEmbeddingCorrector,
    ResidualEmbeddingGBM,
    ResidualOnBaselineCorrector,
    ShiftCorrector,
    ShiftScaleCorrector,
)
from aemo_curves.stages.predict.knn import KNNRetrievalPredictor
from aemo_curves.stages.predict.lag1 import Lag1Predictor
from aemo_curves.stages.predict.lag_blend import (
    BlendedLagPredictor,
    EMABaselinePredictor,
    LearnedBlendPredictor,
    SameWeekdayAveragePredictor,
)
from aemo_curves.stages.predict.regression_retrieval import (
    RFRetrievalPredictor,
    RidgeRetrievalPredictor,
)
from aemo_curves.stages.predict.shift_diagnostics import (
    GridSearchOracleShift,
    OracleShiftPredictor,
    RealizedShiftPredictor,
)
from aemo_curves.stages.predict.soft_knn import (
    SoftKNNRetrievalPredictor,
    StratifiedKNNRetrievalPredictor,
)


def _build(cls, ctx: PredictContext, params: dict):
    """Instantiate `cls`, forwarding the context's exogenous provider when the
    estimator accepts one and the recipe did not set it explicitly."""
    params = dict(params)
    if ctx.exog_provider is not None and "exog_provider" not in params:
        if "exog_provider" in inspect.signature(cls).parameters:
            params["exog_provider"] = ctx.exog_provider
    return cls(**params)


def _fit_predict(model, ctx: PredictContext) -> pd.DataFrame:
    model.fit(ctx.train)
    return model.predict(ctx.test, ctx.historical)


def _simple(name: str, cls):
    @register("predict", name)
    def strategy(ctx: PredictContext, **params):
        return _fit_predict(_build(cls, ctx, params), ctx)

    strategy.__name__ = name
    return strategy


@register("predict", "lag1")
def lag1(ctx: PredictContext, lag_days: int = 1):
    return _fit_predict(Lag1Predictor(lag_days=lag_days), ctx)


_simple("ridge_retrieval", RidgeRetrievalPredictor)
_simple("rf_retrieval", RFRetrievalPredictor)
_simple("knn_retrieval", KNNRetrievalPredictor)
_simple("soft_knn", SoftKNNRetrievalPredictor)
_simple("stratified_knn", StratifiedKNNRetrievalPredictor)
_simple("blended_lag", BlendedLagPredictor)
_simple("ema_baseline", EMABaselinePredictor)
_simple("same_weekday_avg", SameWeekdayAveragePredictor)
_simple("learned_blend", LearnedBlendPredictor)
_simple("residual_embedding", ResidualEmbeddingCorrector)
_simple("residual_embedding_gbm", ResidualEmbeddingGBM)
_simple("residual_on_baseline", ResidualOnBaselineCorrector)
_simple("shift_corrector", ShiftCorrector)
_simple("shift_scale_corrector", ShiftScaleCorrector)
_simple("per_day_shift", PerDayShiftCorrector)
_simple("oracle_shift", OracleShiftPredictor)
_simple("realized_shift", RealizedShiftPredictor)
_simple("grid_search_oracle_shift", GridSearchOracleShift)


@register("predict", "gbm_retrieval")
def gbm_retrieval(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.regression_retrieval import GBMRetrievalPredictor

    return _fit_predict(GBMRetrievalPredictor(**params), ctx)


@register("predict", "xgb_retrieval")
def xgb_retrieval(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.regression_retrieval import XGBRetrievalPredictor

    return _fit_predict(XGBRetrievalPredictor(**params), ctx)


@register("predict", "neural_retrieval")
def neural_retrieval(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.regression_retrieval import NeuralRetrievalPredictor

    return _fit_predict(NeuralRetrievalPredictor(**params), ctx)


@register("predict", "fof_snap")
def fof_snap(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.fpca_snap import FoFSnapPredictor

    return _fit_predict(FoFSnapPredictor(**params), ctx)


@register("predict", "logderiv_fpca")
def logderiv_fpca(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.fpca_snap import LogDerivFPCAPredictor

    return _fit_predict(LogDerivFPCAPredictor(**params), ctx)


@register("predict", "distance_rank")
def distance_rank(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.distance_rank import DistanceRankingRetrieval

    return _fit_predict(DistanceRankingRetrieval(**params), ctx)


@register("predict", "wasserstein_metric")
def wasserstein_metric(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.wasserstein_metric import WassersteinMetricRetrieval

    return _fit_predict(WassersteinMetricRetrieval(**params), ctx)


@register("predict", "ngboost_per_slice")
def ngboost_per_slice(ctx: PredictContext, **params):
    from aemo_curves.stages.predict.ngboost_per_slice import NGBoostPerSlicePredictor

    return _fit_predict(NGBoostPerSlicePredictor(**params), ctx)
