# Experiments

One folder per comparable group, one YAML per experiment. Experiments in the
same group share a data block (side, feature kind, split), a metric, and a
baseline, so only same-group results are directly comparable.

Run one by its dotted `<group>.<name>`:

```bash
uv run python scripts/run_experiment.py demand_7d.vgrid_rolling_ridge_exog
```

An experiment names one strategy per stage; see
[`docs/architecture.md`](../docs/architecture.md) for the format and
[`CLAUDE.md`](../CLAUDE.md) for the registered strategies and metrics.

## Groups

Both groups predict the 7-day-ahead curve on the `v_grid` embedding, scored
under HC-MAE against the lag-7 baseline, on the canonical split.

| Group | Side |
|-------|------|
| `supply_7d` | supply |
| `demand_7d` | demand |

## Methods (the same set in each group)

| Experiment | Strategy | Role |
|------------|----------|------|
| `baseline_lag7` | `lag1` (lag 7) | reference |
| `same_weekday_avg` | `same_weekday_avg` | best learning-free method |
| `vgrid_strat_knn` | `static_retrieval` (ridge) | static retrieval |
| `vgrid_rolling_ridge` | `rolling_refit` (ridge) | rolling monthly refit |
| `vgrid_rolling_ridge_exog` | `rolling_refit` (ridge) + proxy/weather | champion |
| `vgrid_oracle` | `static_retrieval` (oracle) | ceiling (not deployable) |

`demand_7d` runs against the cached demand v_grid tables. `supply_7d` needs the
supply v_grid tables first:

```bash
uv run python scripts/build_v_grid_embeddings.py --side supply
```
