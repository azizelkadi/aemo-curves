"""Evaluate stage: score a predictions frame under a curve-distance metric.

Each metric is registered as an evaluate strategy. An experiment names one of them;
the runner scores the predictions, scores the baseline on the same rows, and
summarises the gap.
"""
from aemo_curves.core.metric import Metric
from aemo_curves.core.registry import register
from aemo_curves.stages.evaluate import metrics as _metrics
from aemo_curves.stages.evaluate._common import attach_baseline, score_predictions, summary_stats
from aemo_curves.stages.evaluate.clearing import attach_clearing, clearing_summary
from aemo_curves.stages.evaluate.dtw import (
    curve_dtw_distance,
    curve_dtw_to_set,
    pairwise_dtw_matrix,
)
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.evaluate.metrics import HC_MAE, WASSERSTEIN
from aemo_curves.stages.evaluate.wasserstein import (
    curve_wasserstein_distance,
    embedding_l2_distance,
)

DTW = _metrics.DTW

register("evaluate", "wasserstein")(WASSERSTEIN)
register("evaluate", "hc_mae")(HC_MAE)
register("evaluate", "dtw")(DTW)

__all__ = [
    "DTW",
    "HC_MAE",
    "Metric",
    "WASSERSTEIN",
    "attach_baseline",
    "attach_clearing",
    "clearing_summary",
    "curve_dtw_distance",
    "curve_dtw_to_set",
    "curve_hc_mae_distance",
    "curve_wasserstein_distance",
    "embedding_l2_distance",
    "pairwise_dtw_matrix",
    "score_predictions",
    "summary_stats",
]
