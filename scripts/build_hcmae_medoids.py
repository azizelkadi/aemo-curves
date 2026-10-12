"""Build the HC-MAE medoid embedding tables of one side.

The medoids are K real train curves (2021-01 to 2024-04) chosen by k-medoids
under HC-MAE. Every train and test curve gets an `embedding` column with its
official HC-MAE to each medoid. Test curves are never used to choose medoids.

Outputs (artifacts/embeddings/):
    {train,test}_hcmae_medoids_<side>_k<K>.pkl   curve table plus `embedding`
    hcmae_medoids_<side>_k<K>.pkl                 one row per medoid
    hcmae_medoids_meta_<side>_k<K>.json           settings, timings and checks

Modes:
    --medoids-from PKL  Reuse a medoid set: a pickled dict whose `medoid_rows`
                        are positions in the train curve table followed by the
                        test curve table, as in
                        pad/fastloop/cache/medoids_<side>_K48_B.pkl.
    default             Search the medoids among all train rows (needs the
                        `kmedoids` package). The method is described in
                        aemo_curves/stages/features/hcmae_medoids.py.

The distances are computed with the official HC-MAE function. With --emb-from,
a saved (rows x K) array in the same row order is loaded instead, after checks.
At the end, --check random distances are read back from the written tables and
compared with the official function.

Usage:
    uv run python scripts/build_hcmae_medoids.py --side demand \
        --medoids-from pad/fastloop/cache/medoids_demand_K48_B.pkl
    uv run python scripts/build_hcmae_medoids.py --side residual --k 48
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.stages.curves import load_curve_table
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.features.hcmae_medoids import (
    build_medoids,
    embedding_table_path,
    exact_embedding,
    feature_name,
    medoid_table_path,
    meta_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("build_hcmae_medoids")

SPLITS = ("train", "test")


def default_seed(side: str) -> int:
    return 1000 if side == "demand" else 1007


def load_side(side: str):
    """Curve tables of both splits, their concatenation (train first) and the curves."""
    tables = {s: load_curve_table(s, side).reset_index(drop=True) for s in SPLITS}
    full = pd.concat([tables[s] for s in SPLITS], ignore_index=True)
    curves = [
        (np.asarray(q, dtype=np.float64), np.asarray(p, dtype=np.float64))
        for q, p in zip(full["curve_qty"], full["curve_price"], strict=True)
    ]
    return tables, full, curves


def medoids_from_file(path: Path, full: pd.DataFrame, curves, n_train: int, k: int) -> tuple[np.ndarray, dict]:
    with open(path, "rb") as f:
        med = pickle.load(f)
    rows = np.asarray(med["medoid_rows"], dtype=np.int64)
    if len(rows) != k or len(np.unique(rows)) != k:
        raise ValueError(f"{path}: expected {k} distinct medoid rows, got {len(rows)}")
    if (rows < 0).any() or (rows >= n_train).any():
        raise ValueError(f"{path}: every medoid must be a train row (0..{n_train - 1})")
    if "medoid_keys" in med:
        keys = med["medoid_keys"]
        dates = pd.to_datetime(keys["Trading Date"]).dt.date.to_numpy()
        ivs = keys["Interval Number"].to_numpy(dtype=np.int64)
        if not (np.array_equal(dates, full["Trading Date"].to_numpy()[rows])
                and np.array_equal(ivs, full["Interval Number"].to_numpy(dtype=np.int64)[rows])):
            raise ValueError(f"{path}: medoid keys do not match the curve table rows")
    if "medoid_qty" in med:
        for j, r in enumerate(rows):
            if not (np.array_equal(np.asarray(med["medoid_qty"][j], dtype=np.float64), curves[r][0])
                    and np.array_equal(np.asarray(med["medoid_price"][j], dtype=np.float64), curves[r][1])):
                raise ValueError(f"{path}: medoid {j} curve differs from curve table row {r}")
    log.info("reused %d medoids from %s", k, path)
    return rows, {"source": str(path), "source_meta": med.get("meta", {})}


def search(full: pd.DataFrame, curves, n_train: int, args) -> tuple[np.ndarray, dict]:
    months = np.array([f"{d.year:04d}-{d.month:02d}" for d in full["Trading Date"]])
    intervals = full["Interval Number"].to_numpy(dtype=np.int64)
    res = build_medoids(
        curves, months, intervals, np.arange(n_train),
        ks=(args.k,), n_sample=args.n_sample, G=args.grid, seed=args.seed,
        pam_seed=args.pam_seed, threads=args.threads,
    )
    out = res["by_k"][args.k]
    info = {
        "search": {key: v for key, v in res["meta"].items()},
        "search_k": {
            key: (v.tolist() if isinstance(v, np.ndarray) else v)
            for key, v in out.items() if key != "medoid_rows"
        },
    }
    return np.asarray(out["medoid_rows"], dtype=np.int64), info


def load_embedding(path: Path, keys_path: Path | None, full: pd.DataFrame, k: int) -> np.ndarray:
    E = np.load(path).astype(np.float64)
    if E.shape != (len(full), k):
        raise ValueError(f"{path}: shape {E.shape}, expected {(len(full), k)}")
    if not np.isfinite(E).all():
        raise ValueError(f"{path}: non-finite distances")
    if keys_path is not None:
        keys = pd.read_pickle(keys_path)
        same = (np.array_equal(pd.to_datetime(keys["Trading Date"]).dt.date.to_numpy(),
                               full["Trading Date"].to_numpy())
                and np.array_equal(keys["Interval Number"].to_numpy(dtype=np.int64),
                                   full["Interval Number"].to_numpy(dtype=np.int64)))
        if not same:
            raise ValueError(f"{keys_path}: row keys differ from the curve tables")
    log.info("loaded distances %s from %s", E.shape, path)
    return E


def spot_check(out_dir: Path, side: str, k: int, n_check: int, seed: int = 123) -> dict:
    """Recompute `n_check` random (row, medoid) distances from the written tables."""
    tables = [pd.read_pickle(embedding_table_path(s, side, k, out_dir)) for s in SPLITS]
    full = pd.concat(tables, ignore_index=True)
    med = pd.read_pickle(medoid_table_path(side, k, out_dir))
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(full), n_check)
    cols = rng.integers(0, k, n_check)
    saved = np.array([full["embedding"].iloc[i][j] for i, j in zip(rows, cols, strict=True)])
    fresh = np.array([
        curve_hc_mae_distance(full["curve_qty"].iloc[i], full["curve_price"].iloc[i],
                              med["curve_qty"].iloc[j], med["curve_price"].iloc[j])
        for i, j in zip(rows, cols, strict=True)
    ])
    diff = np.abs(fresh - saved)
    self_d = np.array([tables[0]["embedding"].iloc[int(r)][j] for j, r in enumerate(med["train_row"])])
    out = {
        "n": int(n_check),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(np.max(diff / np.maximum(np.abs(fresh), 1e-9))),
        "medoid_self_distance_max": float(np.abs(self_d).max()),
    }
    log.info("spot check: %s", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", required=True, choices=["demand", "supply", "residual"])
    p.add_argument("--k", type=int, default=48, help="number of medoids")
    p.add_argument("--medoids-from", type=Path, default=None,
                   help="reuse the medoid set in this pickle instead of searching")
    p.add_argument("--emb-from", type=Path, default=None,
                   help="load the (rows x K) distances from this .npy instead of computing them")
    p.add_argument("--emb-keys", type=Path, default=None,
                   help="row keys of --emb-from (Trading Date, Interval Number) to check the order")
    p.add_argument("--jobs", type=int, default=3, help="worker processes for the official distances")
    p.add_argument("--threads", type=int, default=3, help="torch threads for the medoid search")
    p.add_argument("--n-sample", type=int, default=12_000, help="rows in the search sample matrix")
    p.add_argument("--grid", type=int, default=2048, help="grid points G of the approximate distance")
    p.add_argument("--seed", type=int, default=None,
                   help="seed of the sample and the alternate step (default 1000 for demand, 1007 otherwise)")
    p.add_argument("--pam-seed", type=int, default=0, help="FasterPAM for K uses seed pam_seed + K")
    p.add_argument("--check", type=int, default=300, help="random distances to recompute at the end")
    p.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    args = p.parse_args()
    if args.seed is None:
        args.seed = default_seed(args.side)
    side, k = args.side, int(args.k)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    tables, full, curves = load_side(side)
    n_train = len(tables["train"])
    log.info("%s: %d train rows, %d test rows", side, n_train, len(tables["test"]))

    t0 = time.time()
    if args.medoids_from is not None:
        rows, info = medoids_from_file(args.medoids_from, full, curves, n_train, k)
        info["mode"] = "reuse"
    else:
        rows, info = search(full, curves, n_train, args)
        info["mode"] = "search"
    medoid_seconds = time.time() - t0
    medoids = [curves[r] for r in rows]

    t0 = time.time()
    if args.emb_from is not None:
        E = load_embedding(args.emb_from, args.emb_keys, full, k)
        info["distances"] = f"loaded from {args.emb_from}"
    else:
        E = exact_embedding(curves, medoids, jobs=args.jobs)
        info["distances"] = "official curve_hc_mae_distance"
    embed_seconds = time.time() - t0
    self_d = E[rows, np.arange(k)]
    if np.abs(self_d).max() > 1e-6:
        raise ValueError(f"a medoid row is not at distance 0 from its own medoid (max {self_d.max():.3g})")

    for split in SPLITS:
        lo = 0 if split == "train" else n_train
        table = tables[split].copy()
        table["embedding"] = [E[i].copy() for i in range(lo, lo + len(table))]
        path = embedding_table_path(split, side, k, out_dir)
        table.to_pickle(path)
        log.info("wrote %s (%d rows)", path, len(table))

    nearest_train = E[:n_train].argmin(axis=1)
    med_table = pd.DataFrame({
        "medoid": np.arange(k),
        "Trading Date": full["Trading Date"].to_numpy()[rows],
        "Interval Number": full["Interval Number"].to_numpy(dtype=np.int64)[rows],
        "curve_qty": [c[0] for c in medoids],
        "curve_price": [c[1] for c in medoids],
        "train_row": rows,
        "cluster_size": np.bincount(nearest_train, minlength=k),
    })
    med_path = medoid_table_path(side, k, out_dir)
    med_table.to_pickle(med_path)
    log.info("wrote %s", med_path)

    check = spot_check(out_dir, side, k, args.check) if args.check > 0 else {}
    years = pd.Series([d.year for d in med_table["Trading Date"]]).value_counts().sort_index()
    meta = {
        "side": side,
        "k": k,
        "feature": feature_name(k),
        **info,
        "seed": args.seed if info["mode"] == "search" else None,
        "n_train": int(n_train),
        "n_test": int(len(tables["test"])),
        "medoid_dates": [str(d) for d in med_table["Trading Date"]],
        "medoid_intervals": med_table["Interval Number"].tolist(),
        "medoids_per_year": {int(y): int(c) for y, c in years.items()},
        "nearest_medoid_mean_train": float(E[:n_train].min(axis=1).mean()),
        "nearest_medoid_mean_test": float(E[n_train:].min(axis=1).mean()),
        "spot_check": check,
        "medoid_seconds": medoid_seconds,
        "embedding_seconds": embed_seconds,
        "total_seconds": time.time() - t_start,
    }
    mp = meta_path(side, k, out_dir)
    mp.write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s", mp)
    log.info("nearest-medoid mean HC-MAE: train %.4f, test %.4f; total %.1f min",
             meta["nearest_medoid_mean_train"], meta["nearest_medoid_mean_test"],
             meta["total_seconds"] / 60)


if __name__ == "__main__":
    main()
