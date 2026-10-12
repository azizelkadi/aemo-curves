"""Features stage: turn curves into fixed-length embeddings.

Strategies: `interp` (the 64-d default), `v_grid` (aligned to the horizontal-cut
metric), `kmedoids` (DTW distances to learned prototypes) and `hcmae_medoids`
(the official HC-MAE to medoid curves, in `hcmae_medoids.py`). The feature
tables are built once by `scripts/build_*.py` and cached under `artifacts/`.
"""
from aemo_curves.stages.features.interp import (
    add_curve_embeddings,
    curve_to_embedding,
    load_embedding_table,
    save_embedding_table,
)
from aemo_curves.stages.features.kmedoids_distance import (
    KMedoidsCurveEmbedding,
    default_embedding_path,
)
from aemo_curves.stages.features.v_grid import (
    curve_to_v_grid_embedding,
    fit_v_max_global,
)

__all__ = [
    "KMedoidsCurveEmbedding",
    "add_curve_embeddings",
    "curve_to_embedding",
    "curve_to_v_grid_embedding",
    "default_embedding_path",
    "fit_v_max_global",
    "load_embedding_table",
    "save_embedding_table",
]
