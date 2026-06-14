"""K-medoids prototype embedding for supply / demand curves.

Pipeline
--------
1. Take a training subsample of curves.
2. Compute the pairwise DTW dissimilarity matrix on that subsample.
3. Run **K-medoids (PAM)** to obtain `K` representative *medoid curves*.
4. Embed *any* curve as a `K`-vector of DTW distances to the medoids.
5. The *prototype prediction* of a curve is the medoid with the smallest
   entry in its predicted embedding.

Three evaluation modes are supported by `KMedoidsCurveEmbedding`:

- **baseline** - distance from the lag-1 curve to the true curve.
- **oracle**   - distance from the medoid that is closest to the *true*
  curve (i.e., picked using the true K-d embedding). Best a prototype-
  based method can ever do.
- **predicted** - distance from the medoid picked by some prediction of
  the K-d embedding.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.stages.evaluate.dtw import (
    curve_dtw_to_set,
    pairwise_dtw_matrix,
)

log = logging.getLogger(__name__)


def pam_kmedoids(
    distance_matrix: np.ndarray,
    k: int,
    max_iter: int = 100,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition-Around-Medoids (PAM) on a precomputed (n, n) distance matrix.

    Pure-numpy implementation; no `sklearn-extra` dependency.

    Parameters
    ----------
    distance_matrix:
        Symmetric ``(n, n)`` non-negative distance matrix with zero on the
        diagonal.
    k:
        Number of medoids.
    max_iter:
        Maximum number of swap-improvement passes.
    random_state:
        Seed used to choose the initial medoid set.

    Returns
    -------
    medoid_indices:
        ``(k,)`` int array - the indices of the chosen medoids.
    labels:
        ``(n,)`` int array - for each row, the index *into medoid_indices*
        of its nearest medoid.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("distance_matrix must be square")
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    rng = np.random.default_rng(random_state)
    # BUILD phase: greedy initial medoids minimising total cost.
    chosen = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, k):
        best_idx, best_cost = -1, np.inf
        # Cost reduction = sum_i max(0, current_min[i] - D[i, candidate])
        current_min = D[:, chosen].min(axis=1)
        for cand in range(n):
            if cand in chosen:
                continue
            cost = np.minimum(current_min, D[:, cand]).sum()
            if cost < best_cost:
                best_cost = cost
                best_idx = cand
        if best_idx < 0:
            break
        chosen.append(best_idx)
    medoids = np.array(chosen, dtype=int)

    # SWAP phase: try every (medoid, non-medoid) swap, keep improvements.
    for _ in range(max_iter):
        labels = D[:, medoids].argmin(axis=1)
        cost_now = D[np.arange(n), medoids[labels]].sum()
        improved = False
        for mi in range(k):
            for cand in rng.permutation(n):
                if cand in medoids:
                    continue
                trial = medoids.copy()
                trial[mi] = cand
                new_cost = D[:, trial].min(axis=1).sum()
                if new_cost + 1e-12 < cost_now:
                    medoids = trial
                    cost_now = new_cost
                    improved = True
                    break  # accept first improvement, restart outer loop
            if improved:
                break
        if not improved:
            break

    labels = D[:, medoids].argmin(axis=1)
    return medoids, labels


@dataclass
class KMedoidsCurveEmbedding:
    """K-medoids prototype embedding fitted from a training curve table.

    Parameters
    ----------
    k:
        Number of medoid prototypes.
    train_sample:
        How many training rows to use for the pairwise DTW computation.
        Smaller is faster but loses representativeness.
    random_state:
        Seed for sub-sampling and PAM initialisation.
    qty_weight, multivariate, normalise_qty, window:
        Forwarded to `curve_dtw_distance`.
    """

    k: int = 20
    train_sample: int = 300
    random_state: int = 0
    qty_weight: float = 50.0
    multivariate: bool = True
    normalise_qty: bool = True
    window: int | None = None

    medoid_curves: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, init=False)
    medoid_dates: list[object] = field(default_factory=list, init=False)
    medoid_intervals: list[int] = field(default_factory=list, init=False)
    sample_size: int = field(default=0, init=False)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_curves: pd.DataFrame,
        sample: int | None = None,
        verbose: bool = True,
    ) -> KMedoidsCurveEmbedding:
        """Pick K medoid curves from a sub-sample of `train_curves`.

        Curves are sub-sampled stratified by Interval Number so all 48
        intervals are represented (assuming `sample >= 48`).
        """
        sample_size = int(sample if sample is not None else self.train_sample)
        sample_size = min(sample_size, len(train_curves))
        self.sample_size = sample_size
        rng = np.random.default_rng(self.random_state)
        # Stratified sub-sampling by interval - each interval gets ~equal share
        intervals = sorted(train_curves["Interval Number"].unique())
        per_interval = max(1, sample_size // len(intervals))
        picks: list[int] = []
        for iv in intervals:
            sub_idx = train_curves.index[train_curves["Interval Number"] == iv]
            pick = rng.choice(sub_idx, size=min(per_interval, len(sub_idx)), replace=False)
            picks.extend(pick.tolist())
        # Trim if we overshot
        if len(picks) > sample_size:
            picks = list(rng.choice(picks, size=sample_size, replace=False))
        sample_df = train_curves.loc[picks].reset_index(drop=True)
        if verbose:
            log.info("k-medoids fit: pairwise DTW on %d curves", len(sample_df))
        curves = [
            (np.asarray(r["curve_qty"], dtype=np.float64),
             np.asarray(r["curve_price"], dtype=np.float64))
            for _, r in sample_df.iterrows()
        ]
        D = pairwise_dtw_matrix(
            curves,
            multivariate=self.multivariate,
            qty_weight=self.qty_weight,
            normalise_qty=self.normalise_qty,
            window=self.window,
            show_progress=verbose,
        )
        if verbose:
            log.info("k-medoids fit: PAM on %dx%d DTW matrix", D.shape[0], D.shape[1])
        med_idx_arr, _labels = pam_kmedoids(D, k=self.k, random_state=self.random_state)
        med_idx = list(map(int, med_idx_arr))
        self.medoid_curves = [curves[i] for i in med_idx]
        self.medoid_dates = sample_df.iloc[med_idx]["Trading Date"].tolist()
        self.medoid_intervals = [int(x) for x in sample_df.iloc[med_idx]["Interval Number"]]
        return self

    # ------------------------------------------------------------ transform
    def embed(self, curve_qty: np.ndarray, curve_price: np.ndarray) -> np.ndarray:
        """Return the K-d distance vector for a single curve."""
        if not self.medoid_curves:
            raise RuntimeError("KMedoidsCurveEmbedding not fitted")
        return curve_dtw_to_set(
            curve_qty, curve_price, self.medoid_curves,
            multivariate=self.multivariate,
            qty_weight=self.qty_weight,
            normalise_qty=self.normalise_qty,
            window=self.window,
        )

    def transform(self, curves: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """Embed every row of a curve table. Returns (n, K)."""
        out = np.zeros((len(curves), self.k), dtype=np.float64)
        iterator = curves.iterrows()
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(curves), desc="k-medoids embed")
            except ImportError:
                pass
        for pos, (_, row) in enumerate(iterator):
            out[pos] = self.embed(row["curve_qty"], row["curve_price"])
        return out

    def add_to_table(self, curves: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Return a copy of `curves` with a `kmedoids_embedding` column added."""
        emb = self.transform(curves, verbose=verbose)
        out = curves.copy()
        out["kmedoids_embedding"] = list(emb)
        return out

    # ----------------------------------------------------------- prediction
    def closest_medoid(self, embedding: np.ndarray) -> int:
        """Index of the closest medoid given a K-d distance vector."""
        return int(np.argmin(np.asarray(embedding, dtype=np.float64)))

    def medoid_curve(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the (qty, price) arrays of the `idx`-th medoid."""
        return self.medoid_curves[idx]

    # ---------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> KMedoidsCurveEmbedding:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}")
        return obj


# Conventional artefact paths so scripts can find a fitted embedding.
def default_embedding_path(side: str = "supply", k: int = 20) -> Path:
    return EMBEDDINGS_DIR / f"kmedoids_{side}_k{k}.pkl"
