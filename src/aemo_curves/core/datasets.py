"""Resolve an experiment `data` block to the train and test feature tables.

The load/curves/features stages are built once by `scripts/build_*.py` and
cached under `artifacts/`. Here we only read those cached tables for a given
side, feature kind, and split, so every experiment that names the same data
block reads byte-identical inputs.
"""
from __future__ import annotations

import pandas as pd

from aemo_curves.config import SPLITS
from aemo_curves.stages.features import load_embedding_table

_FEATURE_TABLES = {
    "interp": ("train", "test"),
    "v_grid": ("train_v_grid", "test_v_grid"),
}

# Tables that exist for the canonical split only, as (train name, test name,
# suffix): the files are `{name}_{side}_{suffix}.pkl`. The HC-MAE medoids were
# chosen among the canonical train rows, so a recut would move rows that chose
# the medoids into the test side.
CANONICAL_ONLY_TABLES = {
    "hcmae_medoids_k48": ("train_hcmae_medoids", "test_hcmae_medoids", "k48"),
}


def _read_kmedoids(name_tr: str, name_te: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from aemo_curves.config import EMBEDDINGS_DIR

    return (
        pd.read_pickle(EMBEDDINGS_DIR / f"{name_tr}.pkl"),
        pd.read_pickle(EMBEDDINGS_DIR / f"{name_te}.pkl"),
    )


def load_features(
    side: str = "supply",
    features: str = "interp",
    split: str = "canonical",
    k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load `(train, test)` feature tables for a side, feature kind, and split.

    `features` is "interp" (64-d), "v_grid", "kmedoids" (the K-vector of DTW
    distances to learned prototypes; `k` selects the prototype count) or
    "hcmae_medoids_k48" (the official HC-MAE to each of 48 medoid curves). `split`
    is "canonical" (the cached split) or a key in `config.SPLITS` (a recut of
    the combined table). The recut reuses the canonical-suffix tables for interp
    and the `f"{side}_<split>"` tables for v_grid.
    """
    if features == "kmedoids":
        if k is None:
            raise ValueError("features='kmedoids' needs a 'k' in the data block")
        if split != "canonical":
            raise ValueError("features='kmedoids' only supports split='canonical'")
        return _read_kmedoids(f"train_kmedoids_{side}_k{k}", f"test_kmedoids_{side}_k{k}")
    if features in CANONICAL_ONLY_TABLES:
        if split != "canonical":
            raise ValueError(f"features={features!r} only supports split='canonical'")
        name_tr, name_te, sfx = CANONICAL_ONLY_TABLES[features]
        return (
            load_embedding_table(name_tr, f"{side}_{sfx}"),
            load_embedding_table(name_te, f"{side}_{sfx}"),
        )
    if features not in _FEATURE_TABLES:
        raise ValueError(
            f"Unknown features {features!r}; expected 'kmedoids' or one of "
            f"{sorted(_FEATURE_TABLES) + sorted(CANONICAL_ONLY_TABLES)}"
        )
    name_tr, name_te = _FEATURE_TABLES[features]
    if split == "canonical":
        return load_embedding_table(name_tr, side), load_embedding_table(name_te, side)
    if split in SPLITS:
        sfx = f"{side}_{split}" if features == "v_grid" else side
        full = pd.concat(
            [load_embedding_table(name_tr, sfx), load_embedding_table(name_te, sfx)],
            ignore_index=True,
        )
        full = full.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
        ts = pd.to_datetime(full["Trading Date"])
        s = SPLITS[split]
        train = full[ts <= s.train_end].reset_index(drop=True)
        test = full[(ts >= s.test_start) & (ts <= s.test_end)].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(
                f"split {split!r} produced an empty side (train={len(train)}, test={len(test)}); "
                "check the recut bounds against the data range"
            )
        return train, test
    raise ValueError(f"Unknown split {split!r}; expected 'canonical' or one of {sorted(SPLITS)}")


def historical_from(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Lag lookups may reach into the test window, so history is train plus test."""
    return pd.concat([train, test], ignore_index=True)
