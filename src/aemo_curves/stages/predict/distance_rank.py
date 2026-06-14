"""Distance-ranking retrieval model.

For each test target, retrieve the candidate (from the (interval, dow)
training bucket) whose *predicted Wasserstein distance to the target* is
smallest. The distance predictor is a HistGBM trained on pairs:

    features(target_context, candidate) -> Wasserstein(true_target_curve, candidate_curve)

The target's true curve is the *label* during training. At inference
time the target's curve is unobserved; only the lag-30 context and the
candidate's full features are available.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.evaluate.wasserstein import curve_wasserstein_distance

from ._features import calendar_features, hist_index, lag_row

PAIR_METRICS = {
    "wasserstein": curve_wasserstein_distance,
    "hc_mae": curve_hc_mae_distance,
}

log = logging.getLogger(__name__)


def _pair_features(
    lag30_emb: np.ndarray,
    target_cal: np.ndarray,
    cand_emb: np.ndarray,
    cand_cal: np.ndarray,
    days_diff: float,
) -> np.ndarray:
    """Build a single (target, candidate) feature vector.

    Layout:  lag30(64) | target_cal(5) | cand_emb(64) | cand_cal(5)
            | L2 | L1 | dot | days_diff           (4 cross features)
    """
    lag30 = np.asarray(lag30_emb, dtype=float)
    cand = np.asarray(cand_emb, dtype=float)
    diff = lag30 - cand
    cross = np.array([
        np.linalg.norm(diff),
        float(np.abs(diff).sum()),
        float(np.dot(lag30, cand)),
        float(days_diff),
    ], dtype=float)
    return np.concatenate([lag30, target_cal, cand, cand_cal, cross])


def _build_buckets(df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    out: dict[tuple[int, int], list[int]] = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        dow = pd.Timestamp(row["Trading Date"]).dayofweek
        out.setdefault((int(row["Interval Number"]), int(dow)), []).append(pos)
    return {k: df.iloc[idx].reset_index(drop=True) for k, idx in out.items()}


class DistanceRankingRetrieval(CurvePredictor):
    """Pointwise distance-prediction retriever.

    Training: build N pairs per target row (`n_candidates_per_target`),
    sampled from the same `(interval, dow)` bucket excluding the target
    itself. Pair label is curve-Wasserstein(target, candidate).

    Inference: for each test row, score every candidate in its bucket
    and return the candidate with smallest predicted distance.

    A mandatory degeneracy diagnostic is computed at predict time: the
    Spearman correlation between predicted distance and lag-30→candidate
    L2 across a held-out validation bucket. If ρ > 0.9 the model has
    learned essentially L2-on-embedding and the experiment is a no-op.
    """

    name = "distance_ranking"

    def __init__(
        self,
        n_candidates_per_target: int = 32,
        max_iter: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        bucket: str = "interval_dow",
        lag_days: int = 30,
        seed: int = 0,
        pair_metric: str = "wasserstein",
    ) -> None:
        if bucket != "interval_dow":
            raise ValueError(f"only interval_dow bucket supported, got {bucket!r}")
        if pair_metric not in PAIR_METRICS:
            raise ValueError(f"pair_metric must be one of {set(PAIR_METRICS)}, got {pair_metric!r}")
        self.n_candidates_per_target = int(n_candidates_per_target)
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)
        self.bucket = bucket
        self.lag_days = int(lag_days)
        self.seed = int(seed)
        self.pair_metric = pair_metric
        self._pair_metric_fn = PAIR_METRICS[pair_metric]
        self._gbm = None
        self._library: dict[tuple[int, int], pd.DataFrame] = {}
        self._train_pair_mse: float | None = None  # in-sample fit ceiling
        self._spearman_vs_l2: float | None = None  # degeneracy diagnostic

    def params(self) -> dict:
        return {
            "model": self.name,
            "n_candidates_per_target": self.n_candidates_per_target,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "l2_regularization": self.l2_regularization,
            "bucket": self.bucket,
            "lag_days": self.lag_days,
            "seed": self.seed,
            "pair_metric": self.pair_metric,
        }

    def fit(self, train_embeddings: pd.DataFrame) -> DistanceRankingRetrieval:
        train = train_embeddings.reset_index(drop=True)
        idx = hist_index(train)
        self._library = _build_buckets(train)

        rng = np.random.default_rng(self.seed)
        Xs: list[np.ndarray] = []
        ys: list[float] = []

        log.info("Building training pairs (~%d targets × %d candidates)",
                 len(train), self.n_candidates_per_target)
        t0 = time.time()
        for tpos, (_, target) in enumerate(train.iterrows()):
            interval = int(target["Interval Number"])
            dow = int(pd.Timestamp(target["Trading Date"]).dayofweek)
            bucket = self._library.get((interval, dow))
            if bucket is None or len(bucket) < 2:
                continue
            lag30 = lag_row(train, idx, interval, target["Trading Date"], self.lag_days)
            if lag30 is None:
                continue
            lag30_emb = np.asarray(lag30["embedding"], dtype=float)
            tgt_cal = calendar_features(target["Trading Date"])
            true_qty = target["curve_qty"]
            true_price = target["curve_price"]
            target_ts = pd.Timestamp(target["Trading Date"])

            # Sample candidates from the same bucket, excluding the target.
            avail_mask = np.array([
                pd.Timestamp(d) != target_ts for d in bucket["Trading Date"]
            ])
            avail = bucket[avail_mask]
            n = min(self.n_candidates_per_target, len(avail))
            if n == 0:
                continue
            sample = avail.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
            for _, cand in sample.iterrows():
                cand_emb = np.asarray(cand["embedding"], dtype=float)
                cand_cal = calendar_features(cand["Trading Date"])
                days_diff = (target_ts - pd.Timestamp(cand["Trading Date"])).days
                feat = _pair_features(lag30_emb, tgt_cal, cand_emb, cand_cal, days_diff)
                label = self._pair_metric_fn(
                    true_qty, true_price,
                    cand["curve_qty"], cand["curve_price"],
                )
                Xs.append(feat)
                ys.append(label)
            if (tpos + 1) % 5000 == 0:
                log.info("  pairs so far: %d (after %d/%d targets, %.1fs)",
                         len(Xs), tpos + 1, len(train), time.time() - t0)
        X = np.vstack(Xs).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        log.info("Pair table: X=%s, y=%s, built in %.1fs", X.shape, y.shape, time.time() - t0)

        from sklearn.ensemble import HistGradientBoostingRegressor
        self._gbm = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            random_state=self.seed,
        )
        t0 = time.time()
        self._gbm.fit(X, y)
        log.info("GBM fit on %d pairs in %.1fs", len(X), time.time() - t0)

        # In-sample fit ceiling (modelling capacity diagnostic).
        pred_train = self._gbm.predict(X)
        self._train_pair_mse = float(np.mean((pred_train - y) ** 2))
        log.info("In-sample pair MSE: %.4f", self._train_pair_mse)
        return self

    def predict(
        self,
        test_embeddings: pd.DataFrame,
        historical_embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        hist = historical_embeddings.reset_index(drop=True)
        hist_idx = hist_index(hist)
        rows = []
        # Degeneracy diagnostic accumulators (per-bucket Spearman vs lag30 L2).
        pred_distances_all: list[np.ndarray] = []
        l2_all: list[np.ndarray] = []

        for _, r in test_embeddings.sort_values(["Trading Date", "Interval Number"]).iterrows():
            interval = int(r["Interval Number"])
            dow = int(pd.Timestamp(r["Trading Date"]).dayofweek)
            bucket = self._library.get((interval, dow))
            if bucket is None or len(bucket) == 0:
                continue
            lag30 = lag_row(hist, hist_idx, interval, r["Trading Date"], self.lag_days)
            if lag30 is None:
                continue
            lag30_emb = np.asarray(lag30["embedding"], dtype=float)
            tgt_cal = calendar_features(r["Trading Date"])
            target_ts = pd.Timestamp(r["Trading Date"])

            feats = np.empty((len(bucket), 64 + 5 + 64 + 5 + 4), dtype=np.float32)
            for i, (_, cand) in enumerate(bucket.iterrows()):
                cand_emb = np.asarray(cand["embedding"], dtype=float)
                cand_cal = calendar_features(cand["Trading Date"])
                days_diff = (target_ts - pd.Timestamp(cand["Trading Date"])).days
                feats[i] = _pair_features(lag30_emb, tgt_cal, cand_emb, cand_cal, days_diff)
            preds = self._gbm.predict(feats)
            best = int(np.argmin(preds))
            src = bucket.iloc[best]

            # Lag30→candidate L2 across bucket (degeneracy diagnostic)
            cand_embs = np.vstack(bucket["embedding"].to_numpy())
            l2s = np.linalg.norm(cand_embs - lag30_emb.reshape(1, -1), axis=1)
            pred_distances_all.append(preds)
            l2_all.append(l2s)

            rows.append({
                "Trading Date": r["Trading Date"],
                "Interval Number": interval,
                "true_embedding": r["embedding"],
                "pred_embedding": np.asarray(src["embedding"], dtype=float),
                "true_curve_qty": r["curve_qty"],
                "true_curve_price": r["curve_price"],
                "pred_curve_qty": src["curve_qty"],
                "pred_curve_price": src["curve_price"],
                "retrieved_date": src["Trading Date"],
                "n_neighbors_used": 1,
                "bucket_size": int(len(bucket)),
                "predicted_pair_distance": float(preds[best]),
            })

        # Degeneracy diagnostic - Spearman ρ on a random sample of bucket scorings.
        from scipy.stats import spearmanr
        sample_n = min(500, len(pred_distances_all))
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(len(pred_distances_all), size=sample_n, replace=False)
        rhos = []
        for k in chosen:
            if len(pred_distances_all[k]) < 5:
                continue
            rho, _ = spearmanr(pred_distances_all[k], l2_all[k])
            if np.isfinite(rho):
                rhos.append(rho)
        self._spearman_vs_l2 = float(np.mean(rhos)) if rhos else float("nan")
        log.info("Spearman ρ(predicted_distance, lag30→cand L2): %.3f (over %d buckets)",
                 self._spearman_vs_l2, len(rhos))
        return pd.DataFrame(rows)
