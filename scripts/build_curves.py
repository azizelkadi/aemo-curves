"""Build and cache train+test supply (and optionally demand) curve tables.

Output: artifacts/curves/{train,test}_{supply,demand}.pkl

Usage:
    uv run python scripts/build_curves.py --side supply
    uv run python scripts/build_curves.py --side both
"""
from __future__ import annotations

import argparse
import logging

from aemo_curves.config import TEST_MONTHS, TRAIN_MONTHS
from aemo_curves.stages.curves import build_curve_table, save_curve_table
from aemo_curves.stages.load import load_auctions_for_months

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def build_and_save(side: str, split: str, months: list[str]) -> None:
    raw_side = "offer" if side == "supply" else "bid"
    log.info("Loading %s auctions for %s (%d months)", raw_side, split, len(months))
    auctions = load_auctions_for_months(months=months, side=raw_side)
    log.info("Loaded %d rows", len(auctions))
    table = build_curve_table(auctions, side=side)
    path = save_curve_table(table, name=split, side=side)
    log.info("Saved %d %s curves -> %s", len(table), side, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test curve tables")
    parser.add_argument("--side", choices=["supply", "demand", "both"], default="supply")
    parser.add_argument("--splits", nargs="*", choices=["train", "test"], default=["train", "test"])
    args = parser.parse_args()

    split_to_months = {"train": TRAIN_MONTHS, "test": TEST_MONTHS}
    sides = ["supply", "demand"] if args.side == "both" else [args.side]

    for split in args.splits:
        for side in sides:
            build_and_save(side=side, split=split, months=split_to_months[split])


if __name__ == "__main__":
    main()
