# Recipes

One folder per comparable group, one YAML per experiment. Recipes in the same
group share a data block (side, feature kind, split), a metric, and a baseline,
so only same-group results are directly comparable.

Run one by its dotted `<group>.<name>`:

```bash
uv run python scripts/run_experiment.py horizon_1d.baseline_lag1
```

A recipe names one strategy per stage; see
[`docs/architecture.md`](../docs/architecture.md) for the format and
[`CLAUDE.md`](../CLAUDE.md) for the registered strategies and metrics.

## Groups

| Group | Side | Features | Split | Metric | Baseline |
|-------|------|----------|-------|--------|----------|
| `horizon_1d` / `horizon_7d` / `horizon_30d` | supply / demand | interp | canonical | wasserstein | lag 1 / 7 / 30 |
| `horizon_1d_exog` / `horizon_7d_exog` / `horizon_30d_exog` | supply / demand | interp | canonical | wasserstein | lag 1 / 7 / 30 |
| `retrieval_demand_1d` / `_30d` / `_30d_pool` / `_30d_rolling` | demand | interp | canonical | wasserstein | lag 1 / 30 |
| `vgrid_1d_demand` | demand | v_grid | user2024 | wasserstein | lag 1 |
| `vgrid_1d_demand_canon` / `vgrid_7d_demand` / `vgrid_30d_demand` | demand | v_grid | canonical | wasserstein / hc_mae | lag 1 / 7 / 30 |
| `dtw_kmedoids` / `dtw_kmedoids_demand` | supply / demand | kmedoids | canonical | dtw | lag 1 / 7 |
| `gating_7d` | demand | v_grid | canonical | hc_mae | lag 7 |
| `proposals_top5` / `recompute_2024test` | demand | interp | user2024 | wasserstein | lag 30 |

A few experiments are not recipe-shaped and are kept as utilities instead:
rescoring saved runs under HC-MAE (use `from_run` recipes or
`aemo_curves.analysis.rescore_runs`), the selector training pipeline (the
`selector` predict strategy plus its backfill and training scripts), and
composites that blend several already-logged runs.
