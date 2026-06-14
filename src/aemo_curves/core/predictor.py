"""Common contract every curve-prediction model must satisfy."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CurvePredictor(ABC):
    """A model that predicts one supply/demand curve per (Trading Date, Interval Number).

    Conventions:
      - `fit(train_embeddings)` receives a curve+embedding table of training auctions.
      - `predict(test_embeddings, historical_embeddings)` returns one row per test
        auction with at least the columns:
            Trading Date, Interval Number,
            pred_curve_qty, pred_curve_price
        and optionally:
            pred_embedding, true_embedding, retrieved_date, embedding_l2
      - `historical_embeddings` is the union of train and test embeddings, used to
        look up lag features that may fall within the test window (e.g., the
        day-before of 2024-06-02 is 2024-06-01, already in test).
    """

    name: str = "CurvePredictor"

    @abstractmethod
    def fit(self, train_embeddings: pd.DataFrame) -> CurvePredictor:
        ...

    @abstractmethod
    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

    def params(self) -> dict:
        """Params worth logging to MLflow. Override to expose hyperparameters."""
        return {"model": self.name}
