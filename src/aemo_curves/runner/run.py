"""Run one recipe end to end: data -> predict -> evaluate -> result.

The runner reads the recipe, resolves the feature tables, runs the named
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
from aemo_curves.core.recipe import Recipe
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


def run_recipe(
    recipe: Recipe,
    *,
    log_to_mlflow: bool = True,
    subset: dict | None = None,
    exog_provider: object | None = None,
) -> ExperimentResult:
    bootstrap()
    data = recipe.data
    side = data.get("side", "supply")
    features = data.get("features", "interp")
    split = data.get("split", "canonical")
    k = data.get("k")

    train, test = load_features(side, features, split, k=k)
    if subset:
        train, test = _subset(train, test, subset)
    historical = historical_from(train, test)

    if exog_provider is None and recipe.extra_params.get("exog"):
        from aemo_curves.exogenous import build_exog_provider

        exog_provider = build_exog_provider(recipe.extra_params["exog"], side=side)

    lag_days = int(data.get("lag_days", recipe.predict_params.get("lag_days", 1)))
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

    predict_strategy = get("predict", recipe.predict_strategy)
    predictions = predict_strategy(ctx, **recipe.predict_params)
    PREDICTIONS.validate(predictions)

    metric = get("evaluate", recipe.metric)
    from aemo_curves.stages.evaluate import attach_baseline, score_predictions, summary_stats

    scored = score_predictions(predictions, metric)
    if recipe.baseline is not None:
        baseline_strategy = get("predict", recipe.baseline["strategy"])
        baseline_preds = baseline_strategy(ctx, **recipe.baseline.get("params", {}))
        baseline_scored = score_predictions(baseline_preds, metric)
        scored = attach_baseline(scored, baseline_scored, metric)
    metrics = summary_stats(scored, metric)
    SCORES.validate(scored)
    if metric.curve_col not in scored.columns:
        raise ContractError(f"scores: missing metric column {metric.curve_col!r}")

    params = {"side": side, "features": features, "split": split, "strategy": recipe.predict_strategy}
    params.update(recipe.predict_params)
    if recipe.baseline is not None:
        params["baseline"] = recipe.baseline["strategy"]
        params.update({f"baseline_{k}": v for k, v in recipe.baseline.get("params", {}).items()})
    params.update(recipe.extra_params)

    result = ExperimentResult(
        experiment_name=recipe.name,
        model_name=recipe.predict_strategy,
        params=params,
        metrics=metrics,
        predictions=scored,
        notes=recipe.notes,
    )
    if log_to_mlflow:
        result.mlflow_run_id = log_experiment_result(result)
        log.info("Logged MLflow run %s for recipe %s", result.mlflow_run_id, recipe.name)
    return result
