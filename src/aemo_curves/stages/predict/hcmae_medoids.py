"""HC-MAE medoid oracle.

The feature tables are the HC-MAE medoid tables (`features: hcmae_medoids_k48`).
The `embedding` of a row holds its official HC-MAE to each of the K medoid
curves, and the medoid curves come from `hcmae_medoids_{side}_k{K}.pkl`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.core.registry import register
from aemo_curves.runner.context import PredictContext
from aemo_curves.stages.features.hcmae_medoids import k_of_feature, load_medoid_table, medoid_curves


@register("predict", "hcmae_medoid_oracle")
def hcmae_medoid_oracle(ctx: PredictContext) -> pd.DataFrame:
    """CEILING (uses the answer): each test row gets the medoid curve with the
    smallest HC-MAE to its true curve, so the score is the mean of the smallest
    embedding value. Ties go to the lower medoid index."""
    table = load_medoid_table(ctx.side, k_of_feature(ctx.features))
    medoids = medoid_curves(table)
    test = ctx.test
    E = np.vstack([np.asarray(e, dtype=float) for e in test["embedding"]])
    if E.shape[1] != len(medoids):
        raise ValueError(f"embedding width {E.shape[1]} differs from the {len(medoids)} medoids")
    idx = E.argmin(axis=1)
    return pd.DataFrame({
        "Trading Date": test["Trading Date"].to_numpy(),
        "Interval Number": test["Interval Number"].to_numpy(dtype=int),
        "true_curve_qty": test["curve_qty"].to_numpy(),
        "true_curve_price": test["curve_price"].to_numpy(),
        "pred_curve_qty": [medoids[i][0] for i in idx],
        "pred_curve_price": [medoids[i][1] for i in idx],
        "retrieved_date": table["Trading Date"].to_numpy()[idx],
        "medoid": idx,
        "medoid_hc_mae": E[np.arange(len(E)), idx],
    })
