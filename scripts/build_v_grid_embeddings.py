"""Build train / test v_grid embedding tables.

Supports two split modes:
  --split canonical   (default)  uses load_embedding_table("train"/"test")
                                 directly (the repo's canonical split)
  --split user        uses Trading Date cutoffs (e.g. pre-2024 / 2024)

In `user` mode the script concatenates the canonical train+test tables
and re-splits by date.

V_max_global is *always* computed from the resulting train set, never
from the test set.

Outputs (under `artifacts/embeddings/`):
  train_v_grid_{side}{suffix}.pkl
  test_v_grid_{side}{suffix}.pkl
  v_grid_meta_{side}{suffix}.json
where `suffix` is empty for canonical, ``_user2024`` for the user split.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.stages.features import load_embedding_table
from aemo_curves.stages.features.v_grid import (
    DEFAULT_V_GRID_DIM,
    DEFAULT_V_MAX_QUANTILE,
    curve_to_v_grid_embedding,
    fit_v_max_global,
)

log = logging.getLogger(__name__)


def _vgrid_embed_column(table: pd.DataFrame, v_max_global: float, k: int) -> pd.DataFrame:
    out = table.copy()
    out["embedding"] = out.apply(
        lambda r: curve_to_v_grid_embedding(
            r["curve_qty"], r["curve_price"], v_max_global, k=k,
        ),
        axis=1,
    )
    return out


def _user_split(side: str, train_end: str, test_start: str, test_end: str):
    """Concatenate canonical train+test, re-split by Trading Date."""
    a = load_embedding_table("train", side)
    b = load_embedding_table("test", side)
    full = pd.concat([a, b], ignore_index=True)
    ts = pd.to_datetime(full["Trading Date"])
    train_mask = ts <= pd.Timestamp(train_end)
    test_mask = (ts >= pd.Timestamp(test_start)) & (ts <= pd.Timestamp(test_end))
    train = full[train_mask].reset_index(drop=True)
    test = full[test_mask].reset_index(drop=True)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="demand", choices=("demand", "supply", "residual"))
    parser.add_argument("--k", type=int, default=DEFAULT_V_GRID_DIM)
    parser.add_argument("--quantile", type=float, default=DEFAULT_V_MAX_QUANTILE)
    parser.add_argument("--split", default="canonical", choices=("canonical", "user"))
    parser.add_argument("--train-end", default="2023-12-31",
                        help="Inclusive last date of train (user split only).")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--suffix", default=None,
                        help="File suffix (default empty for canonical, "
                             "'_user2024' for user split).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    suffix = args.suffix
    if suffix is None:
        suffix = "" if args.split == "canonical" else "_user2024"

    if args.split == "user":
        log.info("Loading + applying user split: train ≤ %s, test ∈ [%s, %s]",
                 args.train_end, args.test_start, args.test_end)
        train, test = _user_split(args.side, args.train_end, args.test_start, args.test_end)
    else:
        log.info("Loading canonical train / test (%s)…", args.side)
        train = load_embedding_table("train", args.side)
        test = load_embedding_table("test", args.side)
    log.info("  train rows: %d  (%s → %s)", len(train),
             pd.to_datetime(train["Trading Date"]).min().date(),
             pd.to_datetime(train["Trading Date"]).max().date())
    log.info("  test rows : %d  (%s → %s)", len(test),
             pd.to_datetime(test["Trading Date"]).min().date(),
             pd.to_datetime(test["Trading Date"]).max().date())

    log.info("Fitting V_max_global on %d train curves (quantile=%.3f)…",
             len(train), args.quantile)
    v_max_global = fit_v_max_global(train["curve_qty"], quantile=args.quantile)
    log.info("  V_max_global = %.2f MW", v_max_global)

    # Pre-flight: fraction of TEST totals exceeding the train p99
    test_totals = np.fromiter(
        (float(np.asarray(q, dtype=float)[-1]) for q in test["curve_qty"]),
        dtype=float,
    )
    frac_test_exceeding = float((test_totals > v_max_global).mean())
    log.info("  pre-flight: %.2f%% of test totals exceed train p%g (V_max_global)",
             100 * frac_test_exceeding, args.quantile * 100)

    log.info("Building v_grid embeddings (k=%d)…", args.k)
    t0 = time.time()
    train_vg = _vgrid_embed_column(train, v_max_global, args.k)
    test_vg = _vgrid_embed_column(test, v_max_global, args.k)
    log.info("  done in %.1fs", time.time() - t0)

    out_train = EMBEDDINGS_DIR / f"train_v_grid_{args.side}{suffix}.pkl"
    out_test = EMBEDDINGS_DIR / f"test_v_grid_{args.side}{suffix}.pkl"
    train_vg.to_pickle(out_train)
    test_vg.to_pickle(out_test)
    log.info("  wrote %s (%d rows)", out_train.name, len(train_vg))
    log.info("  wrote %s (%d rows)", out_test.name, len(test_vg))

    meta = {
        "side": args.side,
        "k": args.k,
        "v_max_global": float(v_max_global),
        "quantile": float(args.quantile),
        "split": args.split,
        "train_end": str(args.train_end),
        "test_start": str(args.test_start),
        "test_end": str(args.test_end),
        "n_train": int(len(train_vg)),
        "n_test": int(len(test_vg)),
        "frac_test_totals_exceeding_v_max": frac_test_exceeding,
    }
    meta_path = EMBEDDINGS_DIR / f"v_grid_meta_{args.side}{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("  wrote %s", meta_path.name)
    print()
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
