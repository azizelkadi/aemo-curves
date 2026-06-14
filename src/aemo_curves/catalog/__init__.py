"""The catalog: one output contract per stage, in one place.

Every stage writes a frame that matches its contract here. The runner validates
each stage output against these, so a recipe whose strategies disagree on shape
fails before the result is logged.
"""
from aemo_curves.core.contracts import Column, Contract, is_array_like

AUCTIONS = Contract(
    name="auctions",
    columns=(
        Column("Trading Date"),
        Column("Interval Number"),
        Column("Bid or Offer"),
        Column("Price ($/MWh)"),
        Column("Quantity (MWh)"),
    ),
)

CURVES = Contract(
    name="curves",
    columns=(
        Column("Trading Date"),
        Column("Interval Number"),
        Column("curve_qty", check=is_array_like, note="step quantities as an array"),
        Column("curve_price", check=is_array_like, note="step prices as an array"),
    ),
    keys=("Trading Date", "Interval Number"),
)

FEATURES = Contract(
    name="features",
    columns=(
        Column("Trading Date"),
        Column("Interval Number"),
        Column("curve_qty", check=is_array_like),
        Column("curve_price", check=is_array_like),
        Column("embedding", check=is_array_like, note="fixed-length curve embedding"),
    ),
    keys=("Trading Date", "Interval Number"),
)

PREDICTIONS = Contract(
    name="predictions",
    columns=(
        Column("Trading Date"),
        Column("Interval Number"),
        Column("true_curve_qty", check=is_array_like),
        Column("true_curve_price", check=is_array_like),
        Column("pred_curve_qty", check=is_array_like),
        Column("pred_curve_price", check=is_array_like),
        Column("true_embedding", required=False, check=is_array_like),
        Column("pred_embedding", required=False, check=is_array_like),
        Column("retrieved_date", required=False),
    ),
    keys=("Trading Date", "Interval Number"),
)

SCORES = Contract(
    name="scores",
    columns=(
        Column("Trading Date"),
        Column("Interval Number"),
        Column("true_curve_qty", check=is_array_like),
        Column("true_curve_price", check=is_array_like),
        Column("pred_curve_qty", check=is_array_like),
        Column("pred_curve_price", check=is_array_like),
    ),
    keys=("Trading Date", "Interval Number"),
)

__all__ = ["AUCTIONS", "CURVES", "FEATURES", "PREDICTIONS", "SCORES"]
