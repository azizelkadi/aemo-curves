"""Predict strategy that reuses the saved predictions of an earlier run.

Used to rescore an existing experiment under a different metric: the strategy
returns the stored true and predicted curves, and the runner scores them with
the recipe's metric and baseline.
"""
from __future__ import annotations

import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext


@register("predict", "from_run")
def from_run(ctx: PredictContext, run: str, latest: bool = True) -> pd.DataFrame:
    from aemo_curves.tracking import load_predictions_by_name

    return load_predictions_by_name(run, latest=latest)
