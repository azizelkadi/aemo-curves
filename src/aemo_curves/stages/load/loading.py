"""Load and clean the raw STEM bids & offers CSVs."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from aemo_curves.config import DATA_DIR

log = logging.getLogger(__name__)

COLUMNS = [
    "Trading Date",
    "Interval Number",
    "Trading Interval",
    "Participant Code",
    "Bid or Offer",
    "Price ($/MWh)",
    "Quantity (MWh)",
]

Side = Literal["offer", "bid", "both"]


def month_to_file(month: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"stem-bids-and-offers-{month}.csv"


def _load_raw(months: Iterable[str], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for month in months:
        f = month_to_file(month, data_dir)
        if not f.exists():
            missing.append(month)
            continue
        frames.append(pd.read_csv(f, usecols=COLUMNS, low_memory=False))
    if missing:
        log.warning("Missing %d monthly files (skipped): %s", len(missing), missing)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and drop malformed rows."""
    out = df.copy()
    out["Trading Date"] = pd.to_datetime(out["Trading Date"], errors="coerce").dt.date
    out["Interval Number"] = pd.to_numeric(out["Interval Number"], errors="coerce")
    out["Price ($/MWh)"] = pd.to_numeric(out["Price ($/MWh)"], errors="coerce")
    out["Quantity (MWh)"] = pd.to_numeric(out["Quantity (MWh)"], errors="coerce")
    out["Bid or Offer"] = out["Bid or Offer"].astype("string").str.lower()
    out = out.dropna(
        subset=[
            "Trading Date",
            "Interval Number",
            "Price ($/MWh)",
            "Quantity (MWh)",
            "Bid or Offer",
        ]
    )
    out["Interval Number"] = out["Interval Number"].astype(int)
    out = out[out["Quantity (MWh)"] > 0]
    return out


def load_auctions_for_months(
    months: Iterable[str],
    side: Side = "offer",
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load a clean frame of auctions for the requested months.

    `side` selects "offer" (supply), "bid" (demand), or "both".
    """
    cleaned = _clean(_load_raw(months=months, data_dir=data_dir))
    if side == "both":
        return cleaned.reset_index(drop=True)
    return cleaned[cleaned["Bid or Offer"] == side].reset_index(drop=True)
