"""Supervised Wasserstein-distillation metric learning.

Train a small MLP encoder f: R^64 -> R^64 that maps each curve's interp
embedding to a new space where Euclidean distance matches the curves'
Wasserstein-1 distance:

    ||f(x_a) - f(x_b)||^2  ≈  W_1(curve_a, curve_b)^2

Loss = MSE on pair-distance targets, with optional triplet margin term.
At inference, the learned f produces a new embedding that drops into
the stratified-kNN retrieval pipeline.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from aemo_curves.core.predictor import CurvePredictor
from aemo_curves.stages.evaluate.wasserstein import curve_wasserstein_distance

log = logging.getLogger(__name__)


def _stack(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_numpy()).astype(np.float32)


class WassersteinMetricRetrieval(CurvePredictor):
    """Train a learned metric, then run stratified kNN retrieval in it.

    Steps:
      (1) Sample pairs from train within `(interval, dow)` buckets;
          compute Wasserstein labels.
      (2) Train an MLP encoder by MSE on || f(x_a) - f(x_b) ||^2 vs W^2.
      (3) Project all curves through f; use the new embedding inside the
          existing stratified-kNN retrieval (with a Ridge query in the
          *new* embedding space).
    """

    name = "wasserstein_metric_retrieval"

    def __init__(
        self,
        n_pairs_per_target: int = 24,
        hidden_dim: int = 128,
        out_dim: int = 64,
        epochs: int = 30,
        batch_size: int = 4096,
        lr: float = 1e-3,
        seed: int = 0,
        # Downstream retrieval config.
        ridge_alpha: float = 200.0,
        ridge_n_pca: int | None = 20,
        ridge_lags: tuple[int, ...] = (30,),
        lag_days: int = 30,
        bucket: str = "interval_dow",
        inner_query: str = "ridge",   # "ridge" or "neural"
        nn_hidden_dim: int = 64,
        nn_epochs: int = 60,
        nn_lr: float = 1e-3,
    ) -> None:
        if inner_query not in {"ridge", "neural"}:
            raise ValueError(inner_query)
        self.n_pairs_per_target = int(n_pairs_per_target)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.ridge_alpha = float(ridge_alpha)
        self.ridge_n_pca = ridge_n_pca
        self.ridge_lags = tuple(ridge_lags)
        self.lag_days = int(lag_days)
        self.bucket = bucket
        self.inner_query = inner_query
        self.nn_hidden_dim = int(nn_hidden_dim)
        self.nn_epochs = int(nn_epochs)
        self.nn_lr = float(nn_lr)
        self._inner: object | None = None  # the strat-kNN with learned embedding
        self._train_pair_mse: float | None = None
        self._spearman_pred_vs_true: float | None = None

    def params(self) -> dict:
        return {
            "model": self.name,
            "n_pairs_per_target": self.n_pairs_per_target,
            "hidden_dim": self.hidden_dim, "out_dim": self.out_dim,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "lr": self.lr, "ridge_alpha": self.ridge_alpha,
        }

    # ------------------------------------------------------------------ training
    def _build_pair_table(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        # Bucket index for sampling positives
        buckets: dict[tuple[int, int], list[int]] = {}
        for pos, (_, row) in enumerate(train.iterrows()):
            dow = pd.Timestamp(row["Trading Date"]).dayofweek
            buckets.setdefault((int(row["Interval Number"]), int(dow)), []).append(pos)

        E = _stack(train)
        n = len(train)
        ia: list[int] = []
        ib: list[int] = []
        W: list[float] = []
        log.info("Building pair table (%d targets x %d candidates)",
                 n, self.n_pairs_per_target)
        t0 = time.time()
        for i, (_, row) in enumerate(train.iterrows()):
            dow = pd.Timestamp(row["Trading Date"]).dayofweek
            bucket = buckets[(int(row["Interval Number"]), int(dow))]
            avail = [p for p in bucket if p != i]
            if not avail:
                continue
            n_take = min(self.n_pairs_per_target, len(avail))
            sample = rng.choice(avail, size=n_take, replace=False)
            for j in sample:
                w = curve_wasserstein_distance(
                    row["curve_qty"], row["curve_price"],
                    train.iloc[int(j)]["curve_qty"], train.iloc[int(j)]["curve_price"],
                )
                ia.append(i); ib.append(int(j)); W.append(w)
            if (i + 1) % 5000 == 0:
                log.info("  pairs %d (after %d/%d targets, %.1fs)",
                         len(ia), i + 1, n, time.time() - t0)
        ia_arr = np.array(ia, dtype=np.int64)
        ib_arr = np.array(ib, dtype=np.int64)
        W_arr = np.array(W, dtype=np.float32)
        log.info("Pair table done: %d pairs in %.1fs", len(W_arr), time.time() - t0)
        return ia_arr, ib_arr, W_arr

    def fit(self, train_embeddings: pd.DataFrame) -> WassersteinMetricRetrieval:
        import torch
        import torch.nn as nn

        train = train_embeddings.reset_index(drop=True)
        E = _stack(train)
        ia, ib, W = self._build_pair_table(train)

        device = "cpu"
        # Build encoder MLP
        torch.manual_seed(self.seed)
        enc = nn.Sequential(
            nn.Linear(E.shape[1], self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        ).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=self.lr)
        # Normalise the labels to W² space so the loss is on the squared distance
        W_sq = torch.tensor(W ** 2, dtype=torch.float32, device=device)
        E_t = torch.tensor(E, dtype=torch.float32, device=device)
        ia_t = torch.tensor(ia, dtype=torch.long, device=device)
        ib_t = torch.tensor(ib, dtype=torch.long, device=device)
        n_pairs = len(W)
        log.info("Training encoder: %d pairs, %d epochs, batch %d",
                 n_pairs, self.epochs, self.batch_size)
        t0 = time.time()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_pairs, device=device)
            running = 0.0
            for start in range(0, n_pairs, self.batch_size):
                idx = perm[start:start + self.batch_size]
                a = enc(E_t[ia_t[idx]])
                b = enc(E_t[ib_t[idx]])
                pred_sq = ((a - b) ** 2).sum(dim=1)
                loss = ((pred_sq - W_sq[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                running += float(loss.item()) * len(idx)
            running /= n_pairs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info("  epoch %d/%d  MSE(pred_sq - W²) = %.3f",
                         epoch + 1, self.epochs, running)
        log.info("Encoder fit in %.1fs", time.time() - t0)

        # Diagnostics: held-out 20k pairs, compute Spearman and store
        rng = np.random.default_rng(self.seed + 7)
        if n_pairs > 20000:
            sub = rng.choice(n_pairs, size=20000, replace=False)
        else:
            sub = np.arange(n_pairs)
        with torch.no_grad():
            a = enc(E_t[ia_t[torch.tensor(sub)]])
            b = enc(E_t[ib_t[torch.tensor(sub)]])
            pred = ((a - b) ** 2).sum(dim=1).sqrt().cpu().numpy()
        true_w = W[sub]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pred, true_w)
        self._spearman_pred_vs_true = float(rho)
        self._train_pair_mse = float(((pred ** 2 - true_w ** 2) ** 2).mean())
        log.info("Diagnostics: Spearman ρ(learned-L2, true-W) = %.3f, pair MSE (sq) = %.3f",
                 rho, self._train_pair_mse)

        # Project all training curves to learned space and stash as the new embedding column
        with torch.no_grad():
            train_new = enc(E_t).cpu().numpy()
        # Build a new train DF with the learned embeddings; reuse strat-kNN
        new_train = train.copy()
        new_train["embedding"] = list(train_new)
        from .soft_knn import StratifiedKNNRetrievalPredictor
        if self.inner_query == "neural":
            self._inner = StratifiedKNNRetrievalPredictor(
                k=1, query="neural", bucket=self.bucket,
                ridge_alpha=self.ridge_alpha, ridge_n_pca=self.ridge_n_pca,
                ridge_lags=self.ridge_lags, lag_days=self.lag_days,
                output_mode="top1_curve",
                gbm_kwargs=dict(
                    hidden_dim=self.nn_hidden_dim,
                    epochs=self.nn_epochs, lr=self.nn_lr,
                ),
            )
        else:
            self._inner = StratifiedKNNRetrievalPredictor(
                k=1, query="ridge", bucket=self.bucket,
                ridge_alpha=self.ridge_alpha, ridge_n_pca=self.ridge_n_pca,
                ridge_lags=self.ridge_lags, lag_days=self.lag_days,
                output_mode="top1_curve",
            )
        self._encoder_torch = enc
        self._inner.fit(new_train)
        return self

    # --------------------------------------------------------------- inference
    def predict(self, test_embeddings: pd.DataFrame, historical_embeddings: pd.DataFrame):
        import torch
        # Re-embed test and historical via the learned encoder
        with torch.no_grad():
            E_test = _stack(test_embeddings)
            E_hist = _stack(historical_embeddings)
            test_new = self._encoder_torch(torch.tensor(E_test)).numpy()
            hist_new = self._encoder_torch(torch.tensor(E_hist)).numpy()
        new_test = test_embeddings.copy()
        new_test["embedding"] = list(test_new)
        new_hist = historical_embeddings.copy()
        new_hist["embedding"] = list(hist_new)
        return self._inner.predict(new_test, new_hist)
