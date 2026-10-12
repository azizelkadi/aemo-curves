"""Compute K-d distance-to-medoid embeddings for the cached train and test
curves and persist them.

The output tables have one row per (Trading Date, Interval Number) with a
`kmedoids_embedding` column (a K-vector of DTW distances to each medoid).

Output:
    artifacts/embeddings/{train,test}_kmedoids_<side>_k<K>.pkl

Usage:
    uv run python scripts/build_kmedoids_embeddings.py --side supply --k 20 \
        --window 10 --jobs 8
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.stages.curves import load_curve_table
from aemo_curves.stages.features import KMedoidsCurveEmbedding, default_embedding_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _path(side: str, split: str, k: int) -> Path:
    return EMBEDDINGS_DIR / f"{split}_kmedoids_{side}_k{k}.pkl"


def _embed_row(args):
    qty, price, medoids, kwargs = args
    from aemo_curves.stages.evaluate.dtw import curve_dtw_to_set
    return curve_dtw_to_set(qty, price, medoids, **kwargs)


def embed_table(emb: KMedoidsCurveEmbedding, curves: pd.DataFrame, jobs: int) -> pd.DataFrame:
    kwargs = dict(
        multivariate=emb.multivariate,
        qty_weight=emb.qty_weight,
        normalise_qty=emb.normalise_qty,
        window=emb.window,
    )
    medoids = emb.medoid_curves
    payloads = [
        (np.asarray(r["curve_qty"], dtype=np.float64),
         np.asarray(r["curve_price"], dtype=np.float64),
         medoids, kwargs)
        for _, r in curves.iterrows()
    ]
    log.info("Computing %d embeddings on %d worker(s)...", len(payloads), jobs)
    t0 = time.time()
    if jobs <= 1:
        results = [_embed_row(p) for p in payloads]
    else:
        results = Parallel(n_jobs=jobs, batch_size="auto")(delayed(_embed_row)(p) for p in payloads)
    log.info("Done in %.1fs (%.1f ms/row)", time.time() - t0,
             1000 * (time.time() - t0) / len(payloads))
    out = curves.copy().reset_index(drop=True)
    out["kmedoids_embedding"] = list(results)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["supply", "demand", "residual"], default="supply")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--splits", nargs="*", choices=["train", "test"], default=["train", "test"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--window", type=int, default=None,
                   help="Override the embedding's stored DTW window for embedding speed.")
    args = p.parse_args()

    emb_path = default_embedding_path(side=args.side, k=args.k)
    log.info("Loading k-medoids embedding from %s", emb_path)
    emb = KMedoidsCurveEmbedding.load(emb_path)
    if args.window is not None:
        emb.window = int(args.window)
        log.info("Using DTW window=%d for embedding", args.window)

    for split in args.splits:
        log.info("--- %s ---", split)
        curves = load_curve_table(split, args.side)
        log.info("Loaded %d %s curves", len(curves), split)
        emb_table = embed_table(emb, curves, jobs=args.jobs)
        out_path = _path(args.side, split, args.k)
        emb_table.to_pickle(out_path)
        log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
