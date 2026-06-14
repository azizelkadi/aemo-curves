"""Post-hoc analysis across runs: comparison tables and significance tests."""
from aemo_curves.analysis.analysis import (
    block_wilcoxon,
    build_comparison_table,
    build_table,
    cluster_bootstrap,
    holm_correct,
    monthly_table,
    paired_metric_compare,
    rescore_runs,
    summarise,
)

__all__ = [
    "block_wilcoxon",
    "build_comparison_table",
    "build_table",
    "cluster_bootstrap",
    "holm_correct",
    "monthly_table",
    "paired_metric_compare",
    "rescore_runs",
    "summarise",
]
