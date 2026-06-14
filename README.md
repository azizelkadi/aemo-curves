# AEMO STEM Curve Prediction

Predict the **supply** (offer) curve - and optionally the **demand** (bid)
curve - of every 30-minute interval in the Western Australian **STEM**
wholesale electricity market, one day ahead (and at longer horizons too).

The work is organised as a short pipeline of **stages**, each with a fixed
output shape, several interchangeable **strategies** per stage, and one small
**experiment** (a YAML file) that picks one strategy for each stage. See
[`docs/architecture.md`](docs/architecture.md) for the full design.

---

## Data

The raw monthly CSVs are **not shipped**. They can be large and may have licence
terms, so `data/` is git-ignored. The repo ships one small example file at the
root, `stem-bids-and-offers-SAMPLE.csv`.

To run the full pipeline, place the real monthly files under `data/`:

- `data/stem-bids-and-offers-YYYY-MM.csv` - one CSV per month.
- Columns used: `Trading Date, Interval Number, Trading Interval,
  Participant Code, Bid or Offer, Price ($/MWh), Quantity (MWh)`.
- Each trading day has 48 half-hour intervals.
- Rows with `Bid or Offer = "Offer"` form the supply side; `"Bid"` forms the
  demand side.

For each auction `(Trading Date, Interval Number)` the pipeline builds a step
curve `(cumulative_qty, price)` and a fixed-length embedding. The default
embedding is the 64-d `interp` embedding; some groups use `v_grid` or the
`kmedoids` prototype distances instead.

Optional exogenous inputs (weather, demand forecast) go under `data/external/`;
see [`data/external/README.md`](data/external/README.md).

## Train / Test split

| Split | Range               | Notes |
|-------|---------------------|-------|
| Train | 2021-01 to 2024-04  | Used for fitting and model selection. |
| Test  | 2024-05 to 2025-05  | Never used for training. 2025-05 may be missing; the loader warns. |

Both windows are defined in [`config.py`](src/aemo_curves/config.py). Alternative
recuts (for example `user2024`) live in `config.SPLITS` and are selected per
experiment in the experiment's data block. Experiments in the same group share a data
block, a metric, and a baseline, so only same-group results are comparable.

## Metrics

Three curve-distance metrics are available, each a `Metric` registered in
[`stages/evaluate`](src/aemo_curves/stages/evaluate/__init__.py). An experiment sets
`evaluate.metric`:

| Metric | Name | Typical use |
|--------|------|-------------|
| Wasserstein (default) | `wasserstein` | available |
| Dynamic time warping | `dtw` | available |
| Horizontal-cut MAE | `hc_mae` | the supply/7d and demand/7d groups |

Every run is scored against a baseline (the experiment's `evaluate.baseline`,
usually the lag curve at the forecast horizon) on the exact same test rows, so
deltas are apples-to-apples.

## Results

The full results for the two groups are in [`docs/results.md`](docs/results.md).
On **demand**, the v_grid retrieval champion (`vgrid_rolling_ridge_exog`) reaches
HC-MAE 123.75 against the lag-7 baseline 148.97 (a 25.2 $/MWh gain). On
**supply**, the lag-7 baseline (54.83) is hard to beat: the oracle ceiling is
only 7.77 below it and the retrieval methods do not improve on it.

## Project layout

```
phd/
|- pyproject.toml                  # uv-managed deps; defines the aemo-run CLI
|- README.md                       # you are here
|- CLAUDE.md                       # contributor notes
|- docs/architecture.md            # the full design
|- stem-bids-and-offers-SAMPLE.csv # the only shipped data file
|- data/                           # raw monthly CSVs (git-ignored)
|- artifacts/                      # cached curves & embeddings (git-ignored)
|- mlruns/                         # local MLflow store (git-ignored)
|- src/aemo_curves/
|  |- core/        # contracts, registry, experiment, datasets, the three bases
|  |- catalog/     # one output contract per stage
|  |- stages/      # load, curves, features, predict, evaluate (the strategies)
|  |- runner/      # run an experiment end to end
|  |- analysis/    # comparison and significance tests across runs
|  |- exogenous/   # optional weather / demand-forecast features
|  |- tracking/    # MLflow helpers
|  |- viz/         # comparison and curve plots
|  |- config.py    # paths, the canonical split, the split registry
|  \- cli.py       # aemo-run entry point
|- experiments/    # <side>/<horizon>/<metric>/<family>/<name>.yaml (a side+horizon+metric is a comparable group)
|- scripts/        # build the cached stages, run an experiment, compare runs
\- tests/          # contract, smoke, and experiment-check tests
```

## Quickstart

### 1. Install

```bash
uv venv --python 3.11
uv pip install -e .
```

(Optional) dev extras for jupyter and tests: `uv pip install -e ".[dev]"`.

### 2. Build cached stages (once)

You need the real monthly CSVs under `data/` first. The current groups use the
`v_grid` embedding (and the `interp` embedding for the exogenous proxy
features), so build curves and both embeddings for each side:

```bash
uv run python scripts/build_curves.py --side both
uv run python scripts/build_embeddings.py --side supply
uv run python scripts/build_embeddings.py --side demand
uv run python scripts/build_v_grid_embeddings.py --side supply
uv run python scripts/build_v_grid_embeddings.py --side demand
```

### 3. Run an experiment

Experiments live at `experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml`. Run one by its dotted
`<side>.<horizon>.<metric>.<family>.<name>`:

```bash
uv run python scripts/run_experiment.py demand.7d.hc_mae.baseline.lag7
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge_exog --no-mlflow
uv run python scripts/run_experiment.py supply.7d.hc_mae.vgrid.strat_knn --subset-days 3
```

`--no-mlflow` skips logging; `--subset-days N` runs only the first N test days
for a fast logic check. Each run logs params, metrics, and the predictions to a
local MLflow run under `mlruns/`.

### 4. Run and compare a whole group (main entrypoint)

A group is one `experiments/<side>/<horizon>/<metric>/` folder. Everything under
it is comparable, so the group runner runs them all and prints a leaderboard
ranked by the metric:

```bash
uv run python scripts/run_group.py demand.7d.hc_mae
```

Results are cached under `artifacts/cache/`. Re-running a group recomputes only
what changed, so after adding one experiment the same command computes only the
new one and reuses the rest. Use `--no-cache` to force a full recompute (do this
after changing strategy code), `--subset-days N` for a fast check, and `--save
results.csv` to write the leaderboard. The same is available as `aemo-run group
demand.7d.hc_mae`.

### 5. Browse results

```bash
uv run python scripts/compare_experiments.py list
uv run python scripts/compare_experiments.py compare <run_id_1> <run_id_2>
uv run aemo-run ui          # -> http://localhost:5000
```

## Adding things

- **A new experiment**: add `experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml` naming existing
  strategies. No code.
- **A new method**: add a strategy under `src/aemo_curves/stages/predict/`,
  register it with `@register("predict", "<name>")`, then write an experiment.
- **A new metric**: add a `Metric` under `stages/evaluate/`, register it, then
  set `evaluate.metric`.
- **A new data range**: add an entry to `config.SPLITS`, then set `data.split`.

## Conventions

- Most strategies fit 48 separate per-interval models to exploit the strong
  intra-day seasonality.
- Historical lookups include train and test: the day-before of the second test
  day is legal to use; only the current day's curve is hidden.
- Never tune on test.
- Pickle, not parquet, for tables whose cells hold numpy arrays.
- Retrieval strategies return a real historical curve, never an interpolated
  artefact.
