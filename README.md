# AEMO STEM Curve Prediction

Predict the **supply** (offer) curve - and optionally the **demand** (bid)
curve - of every 30-minute interval in the Western Australian **STEM**
wholesale electricity market, one day ahead (and at longer horizons too).

The work is organised as a short pipeline of **stages**, each with a fixed
output shape, several interchangeable **strategies** per stage, and one small
**recipe** per experiment that picks a strategy for each stage. See
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
experiment in the recipe's data block. Recipes in the same group share a data
block, a metric, and a baseline, so only same-group results are comparable.

## Metrics

Three curve-distance metrics are available, each a `Metric` registered in
[`stages/evaluate`](src/aemo_curves/stages/evaluate/__init__.py). A recipe sets
`evaluate.metric`:

| Metric | Name | Typical use |
|--------|------|-------------|
| Wasserstein (default) | `wasserstein` | most groups |
| Dynamic time warping | `dtw` | the prototype (kmedoids) groups |
| Horizontal-cut MAE | `hc_mae` | the v_grid and demand HC-MAE groups |

Every run is scored against a baseline (the recipe's `evaluate.baseline`,
usually the lag curve at the forecast horizon) on the exact same test rows, so
deltas are apples-to-apples.

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
|  |- core/        # contracts, registry, recipe, datasets, the three bases
|  |- catalog/     # one output contract per stage
|  |- stages/      # load, curves, features, predict, evaluate (the strategies)
|  |- runner/      # run a recipe end to end
|  |- analysis/    # comparison and significance tests across runs
|  |- exogenous/   # optional weather / demand-forecast features
|  |- tracking/    # MLflow helpers
|  |- viz/         # comparison and curve plots
|  |- config.py    # paths, the canonical split, the split registry
|  \- cli.py       # aemo-run entry point
|- recipes/        # one folder per comparable group, one YAML per experiment
|- scripts/        # build the cached stages, run a recipe, compare runs
\- tests/          # contract, smoke, and recipe-check tests
```

## Quickstart

### 1. Install

```bash
uv venv --python 3.11
uv pip install -e .
```

(Optional) dev extras for jupyter and tests: `uv pip install -e ".[dev]"`.

### 2. Build cached stages (once)

You need the real monthly CSVs under `data/` first. Then build the side you
want:

```bash
uv run python scripts/build_curves.py --side supply
uv run python scripts/build_embeddings.py --side supply
```

For demand pass `--side demand`. Some groups need extra tables:

```bash
uv run python scripts/build_v_grid_embeddings.py --side demand
uv run python scripts/build_kmedoids.py --side demand --k 24 --sample 480 --window 10
uv run python scripts/build_kmedoids_embeddings.py --side demand --k 24 --window 10 --jobs 8
```

### 3. Run an experiment

Experiments are recipes at `recipes/<group>/<name>.yaml`. Run one by its dotted
`<group>.<name>`:

```bash
uv run python scripts/run_experiment.py horizon_1d.baseline_lag1
uv run python scripts/run_experiment.py vgrid_7d_demand.rolling_ridge_vgrid --no-mlflow
uv run python scripts/run_experiment.py dtw_kmedoids.oracle_kmedoids --subset-days 3
```

`--no-mlflow` skips logging; `--subset-days N` runs only the first N test days
for a fast logic check. Each run logs params, metrics, and the predictions to a
local MLflow run under `mlruns/`.

### 4. Browse results

```bash
uv run python scripts/compare_experiments.py list
uv run python scripts/compare_experiments.py compare <run_id_1> <run_id_2>
uv run aemo-run ui          # -> http://localhost:5000
```

## Adding things

- **A new experiment**: add `recipes/<group>/<name>.yaml` naming existing
  strategies. No code.
- **A new method**: add a strategy under `src/aemo_curves/stages/predict/`,
  register it with `@register("predict", "<name>")`, then write a recipe.
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
