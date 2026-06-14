"""Build step-function supply and demand curves from per-auction bid/offer rows."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from aemo_curves.config import CURVES_DIR, MIN_POINTS_PER_AUCTION

Side = Literal["supply", "demand"]


def _build_step_curve(
    auction: pd.DataFrame,
    ascending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumulative_qty, price) arrays for a step function.

    For supply, ascending=True (accept cheaper MW first).
    For demand, ascending=False (accept most valuable bids first).
    """
    ordered = auction.sort_values("Price ($/MWh)", ascending=ascending).copy()
    ordered["cumulative_quantity"] = ordered["Quantity (MWh)"].cumsum()
    # Turn each price level into a pair of points (min/max cumulative qty) so the step is visible.
    mins = ordered.groupby("Price ($/MWh)", as_index=False)["cumulative_quantity"].min()
    maxs = ordered.groupby("Price ($/MWh)", as_index=False)["cumulative_quantity"].max()
    step = (
        pd.concat([mins, maxs], ignore_index=True)
        .sort_values(["cumulative_quantity", "Price ($/MWh)"], ascending=[True, ascending])
        .drop_duplicates()
    )
    qty = step["cumulative_quantity"].to_numpy(dtype=float)
    price = step["Price ($/MWh)"].to_numpy(dtype=float)
    return qty, price


def build_supply_curve(auction_offers: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Offers sorted ascending by price -> supply curve."""
    return _build_step_curve(auction_offers, ascending=True)


def build_demand_curve(auction_bids: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bids sorted descending by price -> demand curve."""
    return _build_step_curve(auction_bids, ascending=False)


def build_curve_table(
    auctions: pd.DataFrame,
    side: Side = "supply",
    min_points: int = MIN_POINTS_PER_AUCTION,
) -> pd.DataFrame:
    """Collapse a frame of bid/offer rows into one row per (date, interval) with the full curve."""
    rows = []
    build = build_supply_curve if side == "supply" else build_demand_curve
    for (trading_date, interval), auction in auctions.groupby(
        ["Trading Date", "Interval Number"], sort=True
    ):
        if len(auction) < min_points:
            continue
        qty, price = build(auction)
        if len(qty) < 2:
            continue
        rows.append(
            {
                "Trading Date": trading_date,
                "Interval Number": int(interval),
                "curve_qty": qty,
                "curve_price": price,
                "n_offers": int(len(auction)),
                "total_quantity": float(auction["Quantity (MWh)"].sum()),
                "mean_price": float(auction["Price ($/MWh)"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Trading Date", "Interval Number"]).reset_index(drop=True)
    return out


def _curve_table_path(name: str, side: Side, directory: Path = CURVES_DIR) -> Path:
    return directory / f"{name}_{side}.pkl"


def save_curve_table(
    curve_table: pd.DataFrame,
    name: str,
    side: Side,
    directory: Path = CURVES_DIR,
) -> Path:
    """Persist a curve table via pickle (numpy arrays in cells defeat parquet)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _curve_table_path(name, side, directory)
    curve_table.to_pickle(path)
    return path


def load_curve_table(
    name: str,
    side: Side,
    directory: Path = CURVES_DIR,
) -> pd.DataFrame:
    return pd.read_pickle(_curve_table_path(name, side, directory))
