"""Run one experiment end to end: data -> predict -> evaluate -> result.

The runner reads the experiment, resolves the feature tables, runs the named
predict strategy, validates the predictions against the catalog, scores them
under the named metric against the named baseline, validates the scores, and
returns an `ExperimentResult` (logged to MLflow unless turned off).
"""
from __future__ import annotations

import logging

import pandas as pd

from aemo_curves.catalog import PREDICTIONS, SCORES
from aemo_curves.core.contracts import ContractError
from aemo_curves.core.datasets import historical_from, load_features
from aemo_curves.core.experiment import Experiment
from aemo_curves.core.registry import get
from aemo_curves.core.result import ExperimentResult
from aemo_curves.runner.context import PredictContext
from aemo_curves.tracking import log_experiment_result

log = logging.getLogger(__name__)

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Import the stage packages so their strategies register themselves."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    import aemo_curves.stages.evaluate  # noqa: F401
    import aemo_curves.stages.predict  # noqa: F401

    _BOOTSTRAPPED = True


def _subset(train: pd.DataFrame, test: pd.DataFrame, subset: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim the tables for a fast logic check (not a real evaluation)."""
    test_days = subset.get("test_days")
    if test_days:
        keep = sorted(pd.to_datetime(test["Trading Date"]).dt.normalize().unique())[:test_days]
        keep = set(pd.to_datetime(keep))
        test = test[pd.to_datetime(test["Trading Date"]).dt.normalize().isin(keep)].reset_index(drop=True)
    max_train = subset.get("max_train")
    if max_train and len(train) > max_train:
        train = train.tail(max_train).reset_index(drop=True)
    return train, test


def run_experiment(
    experiment: Experiment,
    *,
    log_to_mlflow: bool = True,
    subset: dict | None = None,
    exog_provider: object | None = None,
) -> ExperimentResult:
    bootstrap()
    data = experiment.data
    side = data.get("side", "supply")
    features = data.get("features", "interp")
    split = data.get("split", "canonical")
    k = data.get("k")
    lag_days = int(data.get("lag_days", experiment.predict_params.get("lag_days", 1)))

    train, test = load_features(side, features, split, k=k)
    if subset:
        train, test = _subset(train, test, subset)
    historical = historical_from(train, test)

    if exog_provider is None and experiment.extra_params.get("exog"):
        from aemo_curves.exogenous import build_exog_provider

        # Shift exogenous features by the forecast horizon so they never peek
        # inside the gap (a 7-day-ahead model may only use data >= 7 days old).
        exog_provider = build_exog_provider(
            experiment.extra_params["exog"], side=side, lag_days=lag_days
        )

    ctx = PredictContext(
        train=train,
        test=test,
        historical=historical,
        side=side,
        features=features,
        split=split,
        lag_days=lag_days,
        exog_provider=exog_provider,
        extra={"k": k} if k is not None else {},
    )

    predict_strategy = get("predict", experiment.predict_strategy)
    predictions = predict_strategy(ctx, **experiment.predict_params)
    PREDICTIONS.validate(predictions)

    metric = get("evaluate", experiment.metric)
    from aemo_curves.stages.evaluate import (
        attach_baseline,
        attach_clearing,
        clearing_summary,
        score_predictions,
        summary_stats,
    )

    scored = score_predictions(predictions, metric)
    baseline_preds = None
    if experiment.baseline is not None:
        baseline_strategy = get("predict", experiment.baseline["strategy"])
        baseline_preds = baseline_strategy(ctx, **experiment.baseline.get("params", {}))
        baseline_scored = score_predictions(baseline_preds, metric)
        scored = attach_baseline(scored, baseline_scored, metric)
    # Market-clearing errors run on every experiment (with or without a baseline),
    # reported alongside the ranking metric and never used to sort.
    scored = attach_clearing(scored, baseline_preds, side=side, split=split)
    metrics = summary_stats(scored, metric)
    metrics.update(clearing_summary(scored))
    SCORES.validate(scored)
    if metric.curve_col not in scored.columns:
        raise ContractError(f"scores: missing metric column {metric.curve_col!r}")

    params = {"side": side, "features": features, "split": split, "strategy": experiment.predict_strategy}
    params.update(experiment.predict_params)
    if experiment.baseline is not None:
        params["baseline"] = experiment.baseline["strategy"]
        params.update({f"baseline_{k}": v for k, v in experiment.baseline.get("params", {}).items()})
    params.update(experiment.extra_params)

    result = ExperimentResult(
        experiment_name=experiment.name,
        model_name=experiment.predict_strategy,
        params=params,
        metrics=metrics,
        predictions=scored,
        notes=experiment.notes,
    )
    if log_to_mlflow:
        result.mlflow_run_id = log_experiment_result(result)
        log.info("Logged MLflow run %s for experiment %s", result.mlflow_run_id, experiment.name)
    return result
