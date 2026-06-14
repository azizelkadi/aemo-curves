# CLAUDE.md - agent / contributor guide

Read [`README.md`](README.md) first for the high-level picture and
[`docs/architecture.md`](docs/architecture.md) for the full design. This file
captures the decisions and conventions that are easy to miss when reading code.

## Problem (one line)

Predict the next-day supply (offer) curve - and optionally the demand (bid)
curve - of every 30-minute STEM auction, and beat the lag baseline.

## Data (one line)

Monthly CSVs under `data/` (git-ignored; only
`stem-bids-and-offers-SAMPLE.csv` ships as an example). Each row is one bid or
offer. Each auction is identified by `(Trading Date, Interval Number)`; there
are 48 intervals per day.

## Train / Test (fixed)

- Train: `2021-01` to `2024-04`.
- Test:  `2024-05` to `2025-05` (2025-05 may be missing; the loader warns).

Defined in [`config.py`](src/aemo_curves/config.py). Never touch the test set
during model development. Alternative recuts live in `config.SPLITS` (for
example `user2024`) and are selected per experiment in the experiment's data block.

## The shape of the code

The pipeline is `load -> curves -> features -> predict -> evaluate`. Each stage
has a fixed output contract; each stage has interchangeable strategies; an
experiment is an experiment that picks one strategy per stage. See
[`docs/architecture.md`](docs/architecture.md).

- `core/` - contracts, the strategy registry, the experiment model, dataset
  resolution, and the three bases (`CurvePredictor`, `Metric`,
  `ExperimentResult`).
- `catalog/` - one output contract per stage.
- `stages/load`, `stages/curves`, `stages/features`, `stages/predict`,
  `stages/evaluate` - the strategies. Importing a stage package registers its
  strategies.
- `runner/` - `run_experiment` runs an experiment end to end.
- `experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml` - the experiments.

### Output contracts (the catalog)

[`catalog/`](src/aemo_curves/catalog/__init__.py) defines the columns each stage
must produce. The predictions contract needs `Trading Date`, `Interval Number`,
`true_curve_qty`, `true_curve_price`, `pred_curve_qty`, `pred_curve_price`. The
scores contract adds `curve_<metric>` and `baseline_<metric>`. The runner
validates each stage output against these.

### The metric is a parameter

A experiment's `evaluate.metric` is `wasserstein` (default), `dtw`, or `hc_mae`,
each a `Metric` registered in
[`stages/evaluate`](src/aemo_curves/stages/evaluate/__init__.py). The chosen
metric names the per-row and summary columns. With Wasserstein the columns keep
the names `curve_wasserstein`, `baseline_wasserstein`, `wasserstein_mean`,
`wasserstein_delta_vs_baseline` (model minus baseline; negative is better),
`wasserstein_win_rate`.

Every run is scored against a baseline (the experiment's `evaluate.baseline`,
usually the lag curve at the forecast horizon), computed on the exact same test
rows, so deltas are apples-to-apples.

## Running and adding experiments

Run one experiment by its dotted name:

```bash
uv run python scripts/run_experiment.py demand.7d.hc_mae.baseline.lag7
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge --no-mlflow
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge --subset-days 3
```

`--subset-days N` runs only the first N test days for a fast logic check.

Run and rank a whole comparable group (one `<side>/<horizon>/<metric>` folder).
This is the main entrypoint for producing a comparison:

```bash
uv run python scripts/run_group.py demand.7d.hc_mae          # ranked leaderboard
uv run python scripts/run_group.py demand.7d.hc_mae --no-cache
```

Group runs are cached under `artifacts/cache/` (key = the experiment's
result-affecting definition plus a fingerprint of the tables it reads). So
re-running a group recomputes only what changed: add one experiment and only it
runs. After changing strategy *code*, use `--no-cache` or clear the cache.

- **New experiment**: add `experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml` naming existing
  strategies. No code. Its `evaluate.metric` must match the `<metric>` folder.
- **New method**: add a strategy under `stages/predict/`, register it with
  `@register("predict", "<name>")`, then write an experiment that uses it.
- **New metric**: add a `Metric` under `stages/evaluate/`, register it, then set
  `evaluate.metric` in an experiment.
- **New data range**: add an entry to `config.SPLITS`, then set `data.split`.

## Cached artifacts

Built once (the `artifacts/` directory is git-ignored):

- `scripts/build_curves.py` -> `artifacts/curves/{train,test}_{supply,demand}.pkl`.
- `scripts/build_embeddings.py` -> `artifacts/embeddings/{train,test}_{side}.pkl`
  (the interp embedding).
- `scripts/build_v_grid_embeddings.py` -> the `*_v_grid_*` tables.
- `scripts/build_kmedoids.py` and `scripts/build_kmedoids_embeddings.py` ->
  the prototype embedding and the per-row K-vector tables.

`core.datasets.load_features(side, features, split)` reads these for an experiment.

## MLflow

- Local file store at `mlruns/` (git-ignored; set via `MLFLOW_TRACKING_URI`).
- Experiment name `aemo-curves` by default (set via `MLFLOW_EXPERIMENT_NAME`).
- Each run logs params, metrics, and the `ExperimentResult` artifacts.
- `aemo_curves.tracking.load_run_result(run_id)` rehydrates a result.

UI: `uv run aemo-run ui` (defaults to http://localhost:5000).

## Conventions and gotchas

- Pickle, not parquet, for tables whose cells hold numpy arrays.
- Historical lookups use train union test, so the previous day of test day 2
  (which is test day 1) is legal.
- Retrieval strategies return a real historical curve, never an interpolated
  artefact.
- The lag baseline is strong because it preserves the exact shape of a real
  recent curve; the burden of proof is on any new strategy to beat it.

## Where to start reading

1. [`docs/architecture.md`](docs/architecture.md) - the design.
2. [`config.py`](src/aemo_curves/config.py) - paths, split, split registry.
3. [`catalog/__init__.py`](src/aemo_curves/catalog/__init__.py) - the contracts.
4. [`core/experiment.py`](src/aemo_curves/core/experiment.py) and
   [`runner/run.py`](src/aemo_curves/runner/run.py) - how an experiment runs.
5. [`stages/predict/lag1.py`](src/aemo_curves/stages/predict/lag1.py) - the
   reference strategy.
