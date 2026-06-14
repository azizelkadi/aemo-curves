# Architecture: stages, strategies, recipes

This project runs one kind of job: take the bid and offer history of the STEM
market and predict the next supply or demand curve, then score the prediction.
The code is organised so that this job is a short pipeline of **stages**, each
stage has several interchangeable **strategies**, and a single experiment is a
small **recipe** that picks one strategy per stage.

The goal is that adding an experiment means writing a recipe, adding a method
means writing one strategy, and changing the metric or the data range means
editing one line. Results stay comparable because every experiment writes the
same output shape and is scored by the same evaluate stage.

## The pipeline

```
load -> curves -> features -> predict -> evaluate
```

| Stage | Input | Output | Shared or per-experiment |
|-------|-------|--------|--------------------------|
| load | monthly CSVs | auction rows (bids/offers) | shared, built once |
| curves | auction rows | one step curve per (date, interval) | shared, built once |
| features | curves | curve plus a fixed-length embedding | shared, built once |
| predict | feature tables | one predicted curve per test auction | per experiment |
| evaluate | predictions | per-row distance plus a summary | per experiment |

`load`, `curves`, and `features` are built once by `scripts/build_*.py` and
cached under `artifacts/`. Every experiment that names the same data block reads
the same cached tables, so the shared work is never repeated and the inputs are
identical across experiments.

## Output contracts (the catalog)

Each stage writes a frame that must match a contract. The contracts live in one
place, `src/aemo_curves/catalog/`, and the runner validates each stage output
against them, so a strategy that returns the wrong shape fails loudly instead of
flowing on.

- `curves`: `Trading Date`, `Interval Number`, `curve_qty`, `curve_price`.
- `features`: the curves columns plus `embedding`.
- `predictions`: `Trading Date`, `Interval Number`, `true_curve_qty`,
  `true_curve_price`, `pred_curve_qty`, `pred_curve_price`; optional
  `true_embedding`, `pred_embedding`, `retrieved_date`.
- `scores`: the predictions columns plus the metric column `curve_<metric>` and,
  when a baseline is set, `baseline_<metric>`.

## Strategies

Strategies live under `src/aemo_curves/stages/<stage>/`. Importing a stage
package registers its strategies in the registry (`core/registry.py`) under that
stage name. A recipe names a strategy; the runner looks it up.

A **predict** strategy is a callable:

```python
@register("predict", "ridge_retrieval")
def ridge_retrieval(ctx: PredictContext, **params) -> pd.DataFrame:
    model = RidgeRetrievalPredictor(**params)
    model.fit(ctx.train)
    return model.predict(ctx.test, ctx.historical)
```

`ctx` (a `PredictContext`) carries the resolved `train`, `test`, and
`historical` feature tables plus the data choices (`side`, `features`, `split`,
`lag_days`, an optional `exog_provider`). The returned frame must match the
predictions contract.

An **evaluate** strategy is a `Metric` (name plus a curve-distance function).
Three are registered: `wasserstein` (default), `dtw`, `hc_mae`.

## Recipes

A recipe is a small YAML file under `recipes/<group>/<name>.yaml`. It names one
strategy per stage plus the parameters and the data version.

```yaml
name: horizon_1d.ridge_retrieval
description: Per-interval Ridge on lagged embeddings, mapped to the nearest curve.
data:
  side: supply
  features: interp      # interp | v_grid | kmedoids
  split: canonical      # canonical | user2024
  lag_days: 1
predict:
  strategy: ridge_retrieval
  params:
    lags: [1, 2, 7, 14]
    alpha: 2.0
    n_pca: 20
evaluate:
  metric: wasserstein    # wasserstein | dtw | hc_mae
  baseline:
    strategy: lag1
    params: {lag_days: 1}
notes: "..."
params:                  # optional extra fields logged to MLflow
  track: horizon_1d
```

Recipes in the same group share a data block, a metric, and a baseline, so only
same-group results are directly comparable.

## The runner

`runner.run_recipe(recipe)` does the per-experiment stages in order:

1. resolve the feature tables for the data block (`core.datasets.load_features`),
2. run the predict strategy, validate against the predictions contract,
3. score under the metric, score the baseline on the same rows, summarise,
   validate against the scores contract,
4. return an `ExperimentResult` and log it to MLflow.

Run one recipe from the command line:

```bash
uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval
uv run python scripts/run_experiment.py horizon_1d.ridge_retrieval --no-mlflow
```

For a fast logic check on a few test days (not a real evaluation), pass a
subset, for example `--subset-days 3`.

## Data versions

The data block pins the inputs. `side` is `supply` or `demand`. `features` is
`interp` (64-d), `v_grid`, or `kmedoids`. `split` is `canonical` (train
2021-01..2024-04, test 2024-05..2025-05) or a recut listed in
`config.SPLITS` (for example `user2024`). Two recipes with the same data block
read byte-identical tables.

## Directory layout

```
src/aemo_curves/
  core/        contracts, registry, recipe, datasets, predictor base, metric base, result
  catalog/     one contract per stage
  stages/
    load/      months -> auction rows
    curves/    auction rows -> step curves
    features/  curves -> embeddings (interp, v_grid, kmedoids)
    predict/   feature tables -> predictions (the models)
    evaluate/  predictions -> scores (the metrics)
  runner/      run a recipe end to end
  analysis/    post-hoc comparison and statistics across runs
  exogenous/   optional weather and demand-forecast features
  tracking/    MLflow helpers
  viz/         comparison and curve plots
  config.py    paths, the canonical split, the split registry
recipes/        one folder per comparable group, one YAML per experiment
scripts/        build the cached stages, run a recipe, compare runs
tests/          contract, smoke, and recipe-check tests
```

## Adding things

- **A new experiment**: add `recipes/<group>/<name>.yaml` naming existing
  strategies. No code.
- **A new method**: add a strategy under `stages/predict/`, register it, then
  write a recipe that uses it.
- **A new metric**: add a `Metric` under `stages/evaluate/`, register it, then
  set `evaluate.metric` in a recipe.
- **A new data range**: add an entry to `config.SPLITS`, then set
  `data.split` in a recipe.
