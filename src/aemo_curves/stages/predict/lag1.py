"""Lag-1 baseline: same interval, day before."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor


def _key_index(df: pd.DataFrame) -> dict[tuple[int, object], int]:
    return {
        (int(r["Interval Number"]), r["Trading Date"]): i
        for i, (_, r) in enumerate(df.iterrows())
    }


class Lag1Predictor(CurvePredictor):
    """Return the curve from 24h earlier (same interval) as the prediction."""

    name = "lag1_baseline"

    def __init__(self, lag_days: int = 1) -> None:
        self.lag_days = int(lag_days)

    def params(self) -> dict:
        return {"model": self.name, "lag_days": self.lag_days}

    def fit(self, train_embeddings: pd.DataFrame) -> Lag1Predictor:
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        index = _key_index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            date_ts = pd.Timestamp(r["Trading Date"])
            lag_date = (date_ts - pd.Timedelta(days=self.lag_days)).date()
            src_idx = index.get((interval, lag_date))
            if src_idx is None:
                continue
            src = hist.iloc[src_idx]
            row = {
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": src["curve_qty"],
                "pred_curve_price": src["curve_price"],
                "retrieved_date": lag_date,
            }
            if "embedding" in r.index:
                row["true_embedding"] = r["embedding"]
                row["pred_embedding"] = src["embedding"]
                row["embedding_l2"] = float(np.linalg.norm(r["embedding"] - src["embedding"]))
            rows.append(row)
        return pd.DataFrame(rows)
