"""An experiment is the full description of one experiment.

It names one strategy per stage plus the parameters and the data version. The
runner reads it and runs the stages in order. Experiments are small YAML files under
`experiments/`; this module loads and lightly validates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Experiment:
    name: str
    data: dict
    predict: dict
    evaluate: dict
    description: str = ""
    notes: str = ""
    extra_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> Experiment:
        for key in ("name", "predict", "evaluate"):
            if key not in raw:
                raise ValueError(f"Experiment is missing required key {key!r}")
        if "strategy" not in raw["predict"]:
            raise ValueError(f"Experiment {raw['name']!r}: predict block needs a 'strategy'")
        if "metric" not in raw["evaluate"]:
            raise ValueError(f"Experiment {raw['name']!r}: evaluate block needs a 'metric'")
        return cls(
            name=raw["name"],
            data=raw.get("data", {}),
            predict=raw["predict"],
            evaluate=raw["evaluate"],
            description=raw.get("description", ""),
            notes=raw.get("notes", ""),
            extra_params=raw.get("params", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @property
    def predict_strategy(self) -> str:
        return self.predict["strategy"]

    @property
    def predict_params(self) -> dict:
        return self.predict.get("params", {})

    @property
    def metric(self) -> str:
        return self.evaluate["metric"]

    @property
    def baseline(self) -> dict | None:
        return self.evaluate.get("baseline")
