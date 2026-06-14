"""Predict stage: turn feature tables into a predictions frame.

The estimator classes live in this package; the predict strategies that wrap
them (and the orchestration strategies for rolling refit, prototype retrieval,
gating, and rescoring) are registered in `_strategies` and `_orchestration`.

Heavy optional backends (xgboost, torch, ngboost) are imported lazily so the
package imports cheaply when those strategies are not used.
"""
from aemo_curves.core.predictor import CurvePredictor
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


def __getattr__(name):
    if name == "XGBRetrievalPredictor":
        from aemo_curves.stages.predict.regression_retrieval import XGBRetrievalPredictor
        return XGBRetrievalPredictor
    if name == "GBMRetrievalPredictor":
        from aemo_curves.stages.predict.regression_retrieval import GBMRetrievalPredictor
        return GBMRetrievalPredictor
    if name == "NeuralRetrievalPredictor":
        from aemo_curves.stages.predict.regression_retrieval import NeuralRetrievalPredictor
        return NeuralRetrievalPredictor
    if name == "FoFSnapPredictor":
        from aemo_curves.stages.predict.fpca_snap import FoFSnapPredictor
        return FoFSnapPredictor
    if name == "LogDerivFPCAPredictor":
        from aemo_curves.stages.predict.fpca_snap import LogDerivFPCAPredictor
        return LogDerivFPCAPredictor
    if name == "DistanceRankingRetrieval":
        from aemo_curves.stages.predict.distance_rank import DistanceRankingRetrieval
        return DistanceRankingRetrieval
    if name == "WassersteinMetricRetrieval":
        from aemo_curves.stages.predict.wasserstein_metric import WassersteinMetricRetrieval
        return WassersteinMetricRetrieval
    if name == "NGBoostPerSlicePredictor":
        from aemo_curves.stages.predict.ngboost_per_slice import NGBoostPerSlicePredictor
        return NGBoostPerSlicePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from aemo_curves.stages.predict import (
    _orchestration,  # noqa: E402,F401
    _strategies,  # noqa: E402,F401
    dtw_prototype,  # noqa: E402,F401
    from_run,  # noqa: E402,F401
    gate_blend,  # noqa: E402,F401
    selector,  # noqa: E402,F401
)

__all__ = [
    "BlendedLagPredictor",
    "CurvePredictor",
    "EMABaselinePredictor",
    "GridSearchOracleShift",
    "KNNRetrievalPredictor",
    "Lag1Predictor",
    "LearnedBlendPredictor",
    "OracleShiftPredictor",
    "PerDayShiftCorrector",
    "RFRetrievalPredictor",
    "RealizedShiftPredictor",
    "ResidualEmbeddingCorrector",
    "ResidualEmbeddingGBM",
    "ResidualOnBaselineCorrector",
    "RidgeRetrievalPredictor",
    "SameWeekdayAveragePredictor",
    "ShiftCorrector",
    "ShiftScaleCorrector",
    "SoftKNNRetrievalPredictor",
    "StratifiedKNNRetrievalPredictor",
]
