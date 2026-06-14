"""Shared primitives the whole pipeline is built on: contracts, the strategy
registry, the experiment model, dataset resolution, and the three bases
(CurvePredictor, Metric, ExperimentResult)."""
from aemo_curves.core.contracts import Column, Contract, ContractError, is_array_like
from aemo_curves.core.datasets import historical_from, load_features
from aemo_curves.core.metric import Metric
from aemo_curves.core.predictor import CurvePredictor
from aemo_curves.core.experiment import Experiment
from aemo_curves.core.registry import (
    all_strategies,
    available,
    get,
    has,
    register,
)
from aemo_curves.core.result import ExperimentResult

__all__ = [
    "Column",
    "Contract",
    "ContractError",
    "CurvePredictor",
    "ExperimentResult",
    "Metric",
    "Experiment",
    "all_strategies",
    "available",
    "get",
    "has",
    "historical_from",
    "is_array_like",
    "load_features",
    "register",
]
