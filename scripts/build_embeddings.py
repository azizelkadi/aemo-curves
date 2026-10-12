"""Attach the fixed-length curve embedding to cached curve tables.

Reads artifacts/curves/{train,test}_{side}.pkl and writes
     artifacts/embeddings/{train,test}_{side}.pkl

Usage:
    uv run python scripts/build_embeddings.py --side supply
"""
from __future__ import annotations

import argparse
import logging

from aemo_curves.stages.curves import load_curve_table
from aemo_curves.stages.features import add_curve_embeddings, save_embedding_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute curve embeddings")
    parser.add_argument("--side", choices=["supply", "demand", "residual"], default="supply")
    parser.add_argument("--splits", nargs="*", choices=["train", "test"], default=["train", "test"])
    args = parser.parse_args()

    for split in args.splits:
        log.info("Loading curves for %s/%s", args.side, split)
        curves = load_curve_table(split, args.side)
        log.info("Curves: %d rows", len(curves))
        emb_table = add_curve_embeddings(curves)
        path = save_embedding_table(emb_table, name=split, side=args.side)
        log.info("Saved %d embeddings -> %s", len(emb_table), path)


if __name__ == "__main__":
    main()
