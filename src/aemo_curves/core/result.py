"""The single datamodel every experiment returns and we persist."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentResult:
    """Result of running a single recipe on its test window."""

    experiment_name: str
    model_name: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: str = ""
    mlflow_run_id: str | None = None

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions.to_pickle(directory / "predictions.pkl")
        meta = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "params": _json_safe(self.params),
            "metrics": _json_safe(self.metrics),
            "notes": self.notes,
            "mlflow_run_id": self.mlflow_run_id,
        }
        (directory / "result.json").write_text(json.dumps(meta, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: Path) -> ExperimentResult:
        directory = Path(directory)
        meta = json.loads((directory / "result.json").read_text())
        preds = pd.read_pickle(directory / "predictions.pkl")
        return cls(
            experiment_name=meta["experiment_name"],
            model_name=meta["model_name"],
            params=meta.get("params", {}),
            metrics=meta.get("metrics", {}),
            predictions=preds,
            notes=meta.get("notes", ""),
            mlflow_run_id=meta.get("mlflow_run_id"),
        )


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
