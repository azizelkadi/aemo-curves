"""Fit a `KMedoidsCurveEmbedding` on the cached training curves and persist it.

Output:
    artifacts/embeddings/kmedoids_<side>_k<K>.pkl

Usage:
    uv run python scripts/build_kmedoids.py --side supply --k 20 --sample 480
"""
from __future__ import annotations

import argparse
import logging
import time

from aemo_curves.stages.curves import load_curve_table
from aemo_curves.stages.features import KMedoidsCurveEmbedding, default_embedding_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["supply", "demand"], default="supply")
    p.add_argument("--k", type=int, default=20, help="Number of medoid prototypes")
    p.add_argument("--sample", type=int, default=480,
                   help="Training rows used to compute the pairwise DTW matrix")
    p.add_argument("--qty-weight", type=float, default=50.0)
    p.add_argument("--window", type=int, default=None, help="Sakoe-Chiba band width")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    log.info("Loading %s training curves...", args.side)
    train = load_curve_table("train", args.side)
    log.info("Loaded %d curves", len(train))

    emb = KMedoidsCurveEmbedding(
        k=args.k,
        train_sample=args.sample,
        random_state=args.seed,
        qty_weight=args.qty_weight,
        window=args.window,
    )
    t0 = time.time()
    emb.fit(train, sample=args.sample, verbose=True)
    log.info("Fit in %.1fs; %d medoids", time.time() - t0, len(emb.medoid_curves))
    log.info("Medoid intervals: %s", sorted(emb.medoid_intervals))
    log.info("Medoid dates    : %s", emb.medoid_dates)

    out = default_embedding_path(side=args.side, k=args.k)
    emb.save(out)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
