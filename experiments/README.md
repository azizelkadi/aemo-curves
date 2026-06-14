# Experiments

Each experiment is one YAML file in a `<side>/<horizon>/<metric>/<family>/<name>.yaml`
tree:

```
experiments/
  demand/7d/hc_mae/
    baseline/lag7.yaml
    weekly/same_weekday_avg.yaml
    vgrid/strat_knn.yaml
    vgrid/rolling_ridge.yaml
    vgrid/rolling_ridge_exog.yaml
    vgrid/oracle.yaml
  supply/7d/hc_mae/        (the same six)
```

A `<side>/<horizon>/<metric>` folder is a **comparable group**: every experiment
under it shares the side, the horizon (so the same lag baseline), and the metric,
so the results are directly comparable. The `<family>` folder just groups methods
of the same kind. An experiment's `evaluate.metric` must match its `<metric>`
folder.

## Run

Run a whole group and get a ranked leaderboard (the main entrypoint):

```bash
uv run python scripts/run_group.py demand.7d.hc_mae
```

Group runs are cached under `artifacts/cache/`, so re-running after adding one
experiment computes only the new one. Run a single experiment by its dotted name
(the dots are the folders):

```bash
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge_exog
```

See [`docs/architecture.md`](../docs/architecture.md) for the format,
[`CLAUDE.md`](../CLAUDE.md) for the registered strategies and metrics, and
[`docs/results.md`](../docs/results.md) for the measured results.

## Current groups

Both predict the 7-day-ahead curve, scored under HC-MAE against the lag-7
baseline on the canonical split. `demand/7d/hc_mae` runs against the cached
demand v_grid tables; `supply/7d/hc_mae` needs the supply v_grid tables first
(`scripts/build_v_grid_embeddings.py --side supply`).

| Family | Experiment | Strategy | Role |
|--------|------------|----------|------|
| baseline | `lag7` | `lag1` (lag 7) | reference |
| weekly | `same_weekday_avg` | `same_weekday_avg` | best learning-free method |
| vgrid | `strat_knn` | `static_retrieval` (ridge) | static retrieval |
| vgrid | `rolling_ridge` | `rolling_refit` (ridge) | rolling monthly refit |
| vgrid | `rolling_ridge_exog` | `rolling_refit` (ridge) + proxy/weather | champion |
| vgrid | `oracle` | `static_retrieval` (oracle) | ceiling (not deployable) |
