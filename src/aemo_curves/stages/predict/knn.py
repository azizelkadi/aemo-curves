"""KNN retrieval: use the lag-1 embedding as query; return nearest training curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor


def _index(df: pd.DataFrame) -> dict[tuple[int, object], int]:
    return {
        (int(r["Interval Number"]), r["Trading Date"]): i
        for i, (_, r) in enumerate(df.iterrows())
    }


class KNNRetrievalPredictor(CurvePredictor):
    """Retrieve the nearest training curve from the same interval.

    Query options:
      - query="lag1": use the previous-day, same-interval embedding as the query.
      - query="oracle": use the TRUE test embedding (upper-bound diagnostic).

    Ties are broken by earliest Trading Date.
    """

    name = "knn_retrieval"

    def __init__(self, k: int = 1, query: str = "lag1") -> None:
        if query not in {"lag1", "oracle"}:
            raise ValueError(f"Unknown query: {query!r}")
        self.k = int(k)
        self.query = query
        self._library_by_interval: dict[int, pd.DataFrame] = {}

    def params(self) -> dict:
        return {"model": self.name, "k": self.k, "query": self.query}

    def fit(self, train_embeddings: pd.DataFrame) -> KNNRetrievalPredictor:
        self._library_by_interval = {
            int(i): sub.reset_index(drop=True)
            for i, sub in train_embeddings.groupby("Interval Number")
        }
        return self

    def _query_vector(
        self,
        row: pd.Series,
        hist: pd.DataFrame,
        hist_index: dict[tuple[int, object], int],
    ) -> np.ndarray | None:
        if self.query == "oracle":
            return np.asarray(row["embedding"], dtype=float)
        # lag1
        interval = int(row["Interval Number"])
        lag_date = (pd.Timestamp(row["Trading Date"]) - pd.Timedelta(days=1)).date()
        idx = hist_index.get((interval, lag_date))
        if idx is None:
            return None
        return np.asarray(hist.iloc[idx]["embedding"], dtype=float)

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_index = _index(hist)
        rows = []
        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            library = self._library_by_interval.get(interval)
            if library is None or len(library) == 0:
                continue
            q = self._query_vector(r, hist, hist_index)
            if q is None:
                continue
            candidates = np.vstack(library["embedding"].to_numpy())
            distances = np.linalg.norm(candidates - q.reshape(1, -1), axis=1)
            if self.k == 1:
                best = int(np.argmin(distances))
                src = library.iloc[best]
                pred_qty, pred_price = src["curve_qty"], src["curve_price"]
                pred_emb = np.asarray(src["embedding"], dtype=float)
                retrieved_date = src["Trading Date"]
                dist = float(distances[best])
            else:
                top = np.argsort(distances)[: self.k]
                # Average the k retrieved embeddings and pick the representative curve (k=1 choice).
                pred_emb = np.vstack(library.iloc[top]["embedding"].to_numpy()).mean(axis=0)
                src = library.iloc[int(top[0])]
                pred_qty, pred_price = src["curve_qty"], src["curve_price"]
                retrieved_date = src["Trading Date"]
                dist = float(distances[top[0]])
            rows.append(
                {
                    "Trading Date": r["Trading Date"],
                    "Interval Number": interval,
                    "true_embedding": r["embedding"],
                    "pred_embedding": pred_emb,
                    "true_curve_qty": r["curve_qty"],
                    "true_curve_price": r["curve_price"],
                    "pred_curve_qty": pred_qty,
                    "pred_curve_price": pred_price,
                    "retrieved_date": retrieved_date,
                    "embedding_retrieval_distance": dist,
                    "embedding_l2": float(np.linalg.norm(r["embedding"] - pred_emb)),
                }
            )
        return pd.DataFrame(rows)
