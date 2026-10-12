"""HC-MAE medoid embedding: a curve described by its HC-MAE to K medoid curves.

The K medoids are real train curves chosen by k-medoids under HC-MAE. The
embedding of a curve is the vector of its official HC-MAE
(`curve_hc_mae_distance`) to each medoid, so the argmin of the embedding is the
medoid nearest to the curve.

Tables under `artifacts/embeddings/` (built by `scripts/build_hcmae_medoids.py`):

- `{split}_hcmae_medoids_{side}_k{K}.pkl`: the curve table of the split plus an
  `embedding` column (K float64 distances, in medoid order).
- `hcmae_medoids_{side}_k{K}.pkl`: one row per medoid, in embedding order.

Medoid search (`build_medoids`)
-------------------------------
The official function is too slow for a 12,000 x 12,000 distance matrix, so the
search uses a grid approximation. Each curve is sampled at G equispaced volumes
on [0, grid max] (the largest volume of the search rows) and held flat outside
its own range, like the official function. The approximate HC-MAE of a pair is
the mean absolute price difference over the grid points inside the union range
of the pair. On 5,000 random train pairs with G=2048 its mean relative error is
0.28% for demand and 0.07% for supply.

1. A sample of `n_sample` rows with the same count in every (month, interval)
   stratum gives the approximate distance matrix.
2. FasterPAM (`kmedoids` package) runs from a BUILD start (K <= 48) and from
   k-medoids++ starts, and the start with the lowest sample objective is kept.
3. Alternate step on all search rows: every row goes to its nearest medoid, and
   each medoid moves to the cluster member with the smallest summed distance to
   the cluster. The candidates are the current medoid, its nearest members and
   some random members, so the objective never goes up. The step repeats until
   no medoid moves or the round limit is reached.

The `kmedoids` package is not a project dependency, so it is imported only when
the search runs.
"""
from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance

log = logging.getLogger(__name__)

FEATURE_PREFIX = "hcmae_medoids_k"

Curve = tuple[np.ndarray, np.ndarray]


def feature_name(k: int) -> str:
    return f"{FEATURE_PREFIX}{int(k)}"


def k_of_feature(features: str) -> int:
    """K of a feature name such as "hcmae_medoids_k48"."""
    tail = features[len(FEATURE_PREFIX):]
    if not features.startswith(FEATURE_PREFIX) or not tail.isdigit():
        raise ValueError(f"{features!r} is not an HC-MAE medoid feature name ({FEATURE_PREFIX}<K>)")
    return int(tail)


def medoid_table_path(side: str, k: int, directory: Path = EMBEDDINGS_DIR) -> Path:
    return Path(directory) / f"hcmae_medoids_{side}_k{int(k)}.pkl"


def embedding_table_path(split: str, side: str, k: int, directory: Path = EMBEDDINGS_DIR) -> Path:
    return Path(directory) / f"{split}_hcmae_medoids_{side}_k{int(k)}.pkl"


def meta_path(side: str, k: int, directory: Path = EMBEDDINGS_DIR) -> Path:
    return Path(directory) / f"hcmae_medoids_meta_{side}_k{int(k)}.json"


def load_medoid_table(side: str, k: int, directory: Path = EMBEDDINGS_DIR) -> pd.DataFrame:
    """One row per medoid: `medoid` (position in the embedding), `Trading Date`,
    `Interval Number`, `curve_qty`, `curve_price`, `train_row` (position in the
    train curve table) and `cluster_size` (train rows nearest to it)."""
    table = pd.read_pickle(medoid_table_path(side, k, directory))
    if list(table["medoid"]) != list(range(int(k))):
        raise ValueError(f"medoid table for {side} K={k} does not hold medoids 0..{k - 1} in order")
    return table


def medoid_curves(table: pd.DataFrame) -> list[Curve]:
    return [
        (np.asarray(q, dtype=np.float64), np.asarray(p, dtype=np.float64))
        for q, p in zip(table["curve_qty"], table["curve_price"], strict=True)
    ]


# ------------------------------------------------------------ exact embedding

def hcmae_to_set(qty, price, medoids: Sequence[Curve]) -> np.ndarray:
    """Official HC-MAE from one curve to each medoid curve."""
    q = np.asarray(qty, dtype=np.float64)
    p = np.asarray(price, dtype=np.float64)
    return np.array([curve_hc_mae_distance(q, p, mq, mp) for mq, mp in medoids], dtype=np.float64)


def _exact_block(curves: Sequence[Curve], medoids: Sequence[Curve]) -> np.ndarray:
    out = np.empty((len(curves), len(medoids)), dtype=np.float64)
    for i, (q, p) in enumerate(curves):
        out[i] = hcmae_to_set(q, p, medoids)
    return out


def exact_embedding(
    curves: Sequence[Curve],
    medoids: Sequence[Curve],
    jobs: int = 1,
    chunk: int = 600,
) -> np.ndarray:
    """(n, K) official HC-MAE from every curve to every medoid, on `jobs` processes."""
    curves = list(curves)
    starts = list(range(0, len(curves), chunk))
    t0 = time.time()
    if jobs <= 1:
        blocks = [_exact_block(curves[s:s + chunk], medoids) for s in starts]
    else:
        from joblib import Parallel, delayed

        blocks = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_exact_block)(curves[s:s + chunk], medoids) for s in starts
        )
    out = np.vstack(blocks) if blocks else np.zeros((0, len(medoids)))
    log.info("exact embedding %s on %d process(es) in %.1fs", out.shape, jobs, time.time() - t0)
    return out


# ---------------------------------------------------------- grid approximation

def _torch():
    import torch

    return torch


class HCMAEGrid:
    """Curves sampled on G equispaced volumes over [0, vmax_grid], flat outside."""

    def __init__(self, vmax_grid: float, G: int) -> None:
        self.vmax_grid = float(vmax_grid)
        self.G = int(G)
        self.g = np.linspace(0.0, self.vmax_grid, self.G)

    def encode(self, curves: Sequence[Curve], rows) -> dict:
        torch = _torch()
        rows = np.asarray(rows)
        n = len(rows)
        X = np.empty((n, self.G), dtype=np.float32)
        nb = np.empty(n, dtype=np.int64)
        na = np.empty(n, dtype=np.int64)
        p0 = np.empty(n, dtype=np.float32)
        pL = np.empty(n, dtype=np.float32)
        g = self.g
        for t, r in enumerate(rows):
            q, p = curves[r]
            idx = np.minimum(np.searchsorted(q, g, side="left"), len(p) - 1)
            X[t] = p[idx]
            # grid points strictly below the first edge and strictly above the last edge
            nb[t] = np.searchsorted(g, q[0], side="left")
            na[t] = self.G - np.searchsorted(g, q[-1], side="right")
            p0[t] = p[0]
            pL[t] = p[-1]
        # float32 counts are exact for integers below 2**24
        return {
            "X": torch.from_numpy(X),
            "nb": torch.from_numpy(nb.astype(np.float32)),
            "na": torch.from_numpy(na.astype(np.float32)),
            "p0": torch.from_numpy(p0),
            "pL": torch.from_numpy(pL),
        }


def _subset(enc: dict, idx) -> dict:
    torch = _torch()
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return {k: v.index_select(0, idx_t) for k, v in enc.items()}


def approx_block(A: dict, B: dict, G: int):
    """Approximate HC-MAE between every row of A and every row of B (float32).

    The full-grid L1 sum minus the grid points outside the union range of the
    pair. Outside the union range both curves are flat at their first (or last)
    price, so that part has a closed form.
    """
    torch = _torch()
    S = torch.cdist(A["X"], B["X"], p=1)
    nb = torch.minimum(A["nb"][:, None], B["nb"][None, :])
    S.sub_((A["p0"][:, None] - B["p0"][None, :]).abs_().mul_(nb))
    na = torch.minimum(A["na"][:, None], B["na"][None, :])
    S.sub_((A["pL"][:, None] - B["pL"][None, :]).abs_().mul_(na))
    cnt = nb.add_(na).neg_().add_(G).clamp_min_(1.0)
    return S.div_(cnt).clamp_min_(0.0)


def approx_to_set(enc_rows: dict, enc_set: dict, G: int, chunk: int = 2048):
    torch = _torch()
    n = enc_rows["X"].shape[0]
    out = torch.empty((n, enc_set["X"].shape[0]), dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sub = {k: v[s:e] for k, v in enc_rows.items()}
        out[s:e] = approx_block(sub, enc_set, G)
    return out


def approx_square(enc: dict, G: int, block: int = 1024) -> np.ndarray:
    """Symmetric (n, n) float32 matrix with zero diagonal, built block by block."""
    n = enc["X"].shape[0]
    D = np.empty((n, n), dtype=np.float32)
    starts = list(range(0, n, block))
    for bi, s in enumerate(starts):
        e = min(n, s + block)
        A = {k: v[s:e] for k, v in enc.items()}
        for t in starts[bi:]:
            u = min(n, t + block)
            B = {k: v[t:u] for k, v in enc.items()}
            blk = approx_block(A, B, G).numpy()
            D[s:e, t:u] = blk
            if t != s:
                D[t:u, s:e] = blk.T
            else:
                D[s:e, t:u] = 0.5 * (blk + blk.T)
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------- medoid search

def _import_kmedoids():
    try:
        import kmedoids
    except ImportError as exc:
        raise ImportError(
            "The medoid search needs the `kmedoids` package (FasterPAM), which is not a "
            "project dependency. Install it with "
            "`uv pip install --python /opt/venv/bin/python kmedoids`, or reuse a medoid "
            "set with --medoids-from."
        ) from exc
    return kmedoids


def stratified_sample(
    months: np.ndarray,
    intervals: np.ndarray,
    rows: np.ndarray,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Equal count per (month, interval) stratum, and the remainder goes to random strata.

    `months` and `intervals` are aligned with all curves, and `rows` are the
    positions to sample from.
    """
    strata: dict[tuple, list] = {}
    for r, mo, iv in zip(rows, months[rows], intervals[rows], strict=True):
        strata.setdefault((mo, int(iv)), []).append(r)
    names = sorted(strata)
    base = n_sample // len(names)
    extra = set(rng.choice(len(names), n_sample - base * len(names), replace=False).tolist())
    picks = []
    for t, name in enumerate(names):
        members = np.asarray(strata[name])
        want = min(len(members), base + (1 if t in extra else 0))
        picks.append(rng.choice(members, want, replace=False))
    return np.sort(np.concatenate(picks))


def kmedoidspp_init(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-medoids++ seeding: each next medoid is drawn with probability proportional
    to its distance to the nearest medoid already chosen."""
    n = D.shape[0]
    chosen = [int(rng.integers(n))]
    dmin = D[chosen[0]].astype(np.float64).copy()
    for _ in range(1, k):
        p = dmin / dmin.sum()
        c = int(rng.choice(n, p=p))
        chosen.append(c)
        np.minimum(dmin, D[c], out=dmin)
    return np.asarray(chosen, dtype=np.uintp)


def run_fasterpam(D: np.ndarray, k: int, seed: int, use_build: bool, n_kmpp: int) -> dict:
    """Best of an optional BUILD start and `n_kmpp` k-medoids++ starts (sample objective).

    BUILD costs O(k n^2) on one thread.
    """
    kmedoids = _import_kmedoids()
    tries = []
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Seed will be ignored")
        if use_build:
            init_build = kmedoids.pam_build(D, k).medoids
            r = kmedoids.fasterpam(D, np.asarray(init_build, dtype=np.uintp), max_iter=200, n_cpu=1,
                                   random_state=seed)
            tries.append(("build", float(r.loss), np.asarray(r.medoids), int(r.n_iter), int(r.n_swap)))
        rng = np.random.default_rng(seed)
        for s in range(n_kmpp):
            init = kmedoidspp_init(D, k, rng)
            r = kmedoids.fasterpam(D, init, max_iter=200, n_cpu=1, random_state=seed + 1 + s)
            tries.append((f"kmpp{s}", float(r.loss), np.asarray(r.medoids), int(r.n_iter), int(r.n_swap)))
    best = min(tries, key=lambda x: x[1])
    n = D.shape[0]
    return {
        "best_init": best[0],
        "sample_objective": best[1] / n,
        "medoids_sample_idx": best[2].astype(np.int64),
        "all_tries": {t[0]: {"objective": t[1] / n, "n_iter": t[3], "n_swap": t[4]} for t in tries},
        "seconds": time.time() - t0,
    }


def _candidate_sums(enc_all: dict, cand_rows: np.ndarray, member_rows: np.ndarray, G: int,
                    chunk: int = 4096):
    torch = _torch()
    C = _subset(enc_all, cand_rows)
    total = torch.zeros(len(cand_rows), dtype=torch.float64)
    for s in range(0, len(member_rows), chunk):
        M = _subset(enc_all, member_rows[s:s + chunk])
        total += approx_block(C, M, G).double().sum(1)
    return total


def alternate_refine(enc_all: dict, medoids: np.ndarray, G: int, n_near: int, n_rand: int,
                     max_rounds: int, rng: np.random.Generator) -> dict:
    """Alternate step on all rows of `enc_all` (medoids are positions in it)."""
    torch = _torch()
    medoids = np.asarray(medoids, dtype=np.int64).copy()
    history = []
    for rnd in range(max_rounds + 1):
        Dm = approx_to_set(enc_all, _subset(enc_all, medoids), G)
        mins, labels = Dm.min(1)
        labels = labels.numpy()
        obj = float(mins.double().mean())
        history.append(obj)
        log.info("    round %d: approx objective %.4f", rnd, obj)
        if rnd == max_rounds:
            break
        if rnd > 0 and history[-2] - obj < 1e-5 * history[-2]:
            break
        moved = 0
        dm_np = Dm.numpy()
        for c in range(len(medoids)):
            members = np.flatnonzero(labels == c)
            if len(members) <= 2:
                continue
            if medoids[c] not in set(members.tolist()):
                members = np.append(members, medoids[c])
            order = np.argsort(dm_np[members, c])
            near = members[order[:n_near]]
            rest = members[order[n_near:]]
            rand = rng.choice(rest, min(n_rand, len(rest)), replace=False) if len(rest) else rest
            cands = np.unique(np.concatenate([[medoids[c]], near, rand]))
            sums = _candidate_sums(enc_all, cands, members, G)
            cur = float(sums[np.flatnonzero(cands == medoids[c])[0]])
            best = int(torch.argmin(sums))
            if float(sums[best]) < cur * (1 - 1e-7):
                medoids[c] = cands[best]
                moved += 1
        log.info("    round %d: moved %d medoids", rnd, moved)
        del Dm, dm_np
        if moved == 0:
            break
    Dm = approx_to_set(enc_all, _subset(enc_all, medoids), G)
    mins, labels = Dm.min(1)
    return {
        "medoids": medoids,
        "labels": labels.numpy(),
        "objective": float(mins.double().mean()),
        "history": history,
    }


def search_settings(k: int) -> dict:
    """Search settings per K: K=48 gets more FasterPAM starts, more candidates
    and more alternate rounds than the other K."""
    heavy = k == 48
    return {
        "use_build": k <= 48,
        "n_kmpp": 3 if heavy else 2,
        "n_near": 200 if heavy else 100,
        "n_rand": 100 if heavy else 50,
        "max_rounds": 6 if heavy else 3,
    }


def build_medoids(
    curves: Sequence[Curve],
    months: np.ndarray,
    intervals: np.ndarray,
    search_rows: np.ndarray,
    ks: Sequence[int] = (48,),
    n_sample: int = 12_000,
    G: int = 2048,
    seed: int = 1007,
    pam_seed: int = 0,
    threads: int | None = None,
) -> dict:
    """Choose medoids among `search_rows` for each K in `ks`, in that order.

    `curves`, `months` ("YYYY-MM") and `intervals` are aligned. `seed` drives
    the stratified sample and the random candidates of the alternate step (one
    random stream shared by all K, in order). FasterPAM for K uses `pam_seed + K`.

    Returns {"meta": ..., "sample_rows": ..., "by_k": {K: {"medoid_rows": ...}}}
    where `medoid_rows` are positions in `curves`.
    """
    torch = _torch()
    if threads is not None:
        torch.set_num_threads(int(threads))
    t_start = time.time()
    search_rows = np.asarray(search_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    vmax = float(max(curves[r][0][-1] for r in search_rows))
    grid = HCMAEGrid(vmax, G)
    sample_rows = stratified_sample(months, intervals, search_rows, n_sample, rng)
    log.info("medoid search: %d search rows, sample %d, grid max %.2f, G=%d",
             len(search_rows), len(sample_rows), vmax, G)

    t0 = time.time()
    enc_s = grid.encode(curves, sample_rows)
    D = approx_square(enc_s, G)
    del enc_s
    t_matrix = time.time() - t0
    log.info("approx matrix %s in %.1fs", D.shape, t_matrix)

    pam = {}
    for k in ks:
        st = search_settings(k)
        pam[k] = run_fasterpam(D, k, seed=pam_seed + k, use_build=st["use_build"], n_kmpp=st["n_kmpp"])
        log.info("K=%d: sample objective %.4f (best start %s, %.1fs)", k,
                 pam[k]["sample_objective"], pam[k]["best_init"], pam[k]["seconds"])
    del D

    t0 = time.time()
    enc_all = grid.encode(curves, search_rows)
    pos_of_row = {int(r): t for t, r in enumerate(search_rows)}
    log.info("encoded %d search rows in %.1fs", len(search_rows), time.time() - t0)
    by_k = {}
    for k in ks:
        st = search_settings(k)
        t0 = time.time()
        init_pos = np.array([pos_of_row[int(sample_rows[i])] for i in pam[k]["medoids_sample_idx"]])
        log.info("K=%d: alternate step on %d rows", k, len(search_rows))
        ref = alternate_refine(enc_all, init_pos, G, n_near=st["n_near"], n_rand=st["n_rand"],
                               max_rounds=st["max_rounds"], rng=rng)
        by_k[k] = {
            "medoid_rows": search_rows[ref["medoids"]],
            "approx_cluster_sizes": np.bincount(ref["labels"], minlength=k),
            "sample_objective": pam[k]["sample_objective"],
            "best_init": pam[k]["best_init"],
            "all_tries": pam[k]["all_tries"],
            "fasterpam_seconds": pam[k]["seconds"],
            "objective_before_alternate": ref["history"][0],
            "objective_after_alternate": ref["objective"],
            "alternate_history": ref["history"],
            "alternate_seconds": time.time() - t0,
            "settings": st,
        }
        log.info("K=%d: approx objective %.4f -> %.4f (%.1fs)", k, ref["history"][0],
                 ref["objective"], time.time() - t0)
    meta = {
        "n_search_rows": int(len(search_rows)),
        "n_sample": int(len(sample_rows)),
        "G": int(G),
        "vmax_grid": vmax,
        "seed": int(seed),
        "pam_seed": int(pam_seed),
        "matrix_seconds": t_matrix,
        "search_seconds": time.time() - t_start,
    }
    return {"meta": meta, "sample_rows": sample_rows, "by_k": by_k}
