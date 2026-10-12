# 02 — Codebase, Design Patterns, and Experiment Design

> **This is document 2 of 3.** It explains *how the code is built and how to work
> with it*: the architecture, the core abstractions, the experiment format, and
> step-by-step recipes for adding experiments, methods, metrics, embeddings, and
> data splits. For the *problem, data, and literature* see
> [`01_problem_and_state_of_the_art.md`](01_problem_and_state_of_the_art.md);
> for *results and status* see
> [`03_current_status_and_results.md`](03_current_status_and_results.md).
>
> All symbols, defaults, and `file.py:line` references below were verified against
> `src/aemo_curves/` (commit `bb21da9`).

---

## 1. Design philosophy in one paragraph

The project runs exactly **one kind of job**: take the STEM bid/offer history and
predict the next supply or demand curve, then score it. The code is organized so
that this job is a short **pipeline of stages**; each stage has several
interchangeable **strategies**; and a single **experiment** is a small YAML file
that picks one strategy per stage plus its parameters and data version. The three
consequences the whole design optimizes for:

- **Adding an experiment = writing a YAML file** (no code).
- **Adding a method = writing one strategy** and registering it.
- **Changing the metric or data range = editing one line.**

Results stay comparable because every strategy writes the **same output shape**
(enforced by contracts) and is scored by the **same evaluate stage** on the
**same rows** against a shared baseline.

---

## 2. The pipeline

```
load  ->  curves  ->  features  ->  predict  ->  evaluate
```

| Stage | Input | Output | Scope |
|-------|-------|--------|-------|
| **load** | monthly CSVs | cleaned auction rows (bids/offers) | shared, built once |
| **curves** | auction rows | one step curve per `(date, interval)` | shared, built once |
| **features** | curves | curve + fixed-length `embedding` | shared, built once |
| **predict** | feature tables | one predicted curve per test auction | **per experiment** |
| **evaluate** | predictions | per-row distance + summary | **per experiment** |

`load`, `curves`, `features` are built once by `scripts/build_*.py` and cached
under `artifacts/`. Every experiment that names the same data block reads the
**byte-identical** cached tables, so shared work is never repeated and inputs are
identical across experiments.

---

## 3. Repository layout

```
phd/
├── pyproject.toml            # uv-managed deps; defines the `aemo-run` CLI
├── README.md                 # quickstart
├── docs/                     # ← these three documents
├── stem-bids-and-offers-SAMPLE.csv   # the only shipped data file
├── data/                     # raw monthly CSVs (git-ignored) + data/external/
├── artifacts/                # cached curves, embeddings, run cache (git-ignored)
├── mlruns/                   # local MLflow store (git-ignored)
├── experiments/              # <side>/<horizon>/<metric>/<family>/<name>.yaml
├── scripts/                  # build caches, run one experiment / a group, compare
├── tests/                    # contract, smoke, metric, cache, family tests
└── src/aemo_curves/
    ├── config.py             # paths, canonical split, SPLITS registry, constants
    ├── cli.py                # `aemo-run` entry point (typer)
    ├── core/                 # contracts, registry, experiment, datasets, bases, result
    ├── catalog/              # one output contract per stage
    ├── runner/               # run one experiment / a group; cache; PredictContext
    ├── stages/
    │   ├── load/             # months -> auction rows
    │   ├── curves/           # auction rows -> step curves
    │   ├── features/         # curves -> embeddings (interp, v_grid, kmedoids)
    │   ├── predict/          # feature tables -> predictions (the models)
    │   └── evaluate/         # predictions -> scores (the metrics)
    ├── exogenous/            # optional weather / proxy / holiday features
    ├── analysis/             # post-hoc comparison + statistics across runs
    ├── tracking/             # MLflow helpers
    └── viz/                  # comparison + curve plots
```

---

## 4. Core abstractions

### 4.1 The strategy registry (plugin pattern) — `core/registry.py`

A single module-global dict `_REGISTRY: dict[stage -> dict[name -> object]]`.

```python
from aemo_curves.core.registry import register, get

@register("predict", "my_method")
def my_method(ctx, **params): ...

strategy = get("predict", "my_method")   # KeyError lists available names if missing
```

- `register(stage, name)` returns a decorator that stores `obj` under
  `_REGISTRY[stage][name]` and returns it unchanged (`registry.py:14-19`).
- **Duplicate names silently overwrite** — there is no collision guard beyond the
  stage dict (`setdefault`). Be careful not to reuse a name.
- Registration is an **import side-effect.** The runner's `bootstrap()`
  (`run.py:28-36`) imports `aemo_curves.stages.evaluate` and
  `aemo_curves.stages.predict`, which import their strategy modules, which run the
  `@register` decorators. Helpers: `has`, `available(stage)`, `all_strategies()`.

### 4.2 Output contracts (fail-loud validation) — `core/contracts.py`, `catalog/`

Each stage output must match a `Contract` (a name + tuple of `Column`s + key
columns). `catalog/__init__.py` defines them; the runner calls `.validate(df)`,
which raises `ContractError` on: not a DataFrame; missing **required** columns;
empty frame; nulls in key columns; duplicate rows over the keys; or an element
check failing (checks run on `df.head(50)` because cells hold numpy arrays).

Required columns (keys `("Trading Date", "Interval Number")` where noted):

| Contract | Required columns |
|----------|------------------|
| `AUCTIONS` | `Trading Date`, `Interval Number`, `Bid or Offer`, `Price ($/MWh)`, `Quantity (MWh)` |
| `CURVES` | + `curve_qty`, `curve_price` (array-like) |
| `FEATURES` | curves + `embedding` (array-like) |
| `PREDICTIONS` | `Trading Date`, `Interval Number`, `true_curve_qty`, `true_curve_price`, `pred_curve_qty`, `pred_curve_price`; *optional* `true_embedding`, `pred_embedding`, `retrieved_date` |
| `SCORES` | the predictions curve columns (metric column checked separately by the runner) |

A strategy that returns the wrong shape **fails loudly at the boundary** instead
of silently flowing downstream.

### 4.3 The Experiment model — `core/experiment.py`

`Experiment` is a plain `@dataclass` (not pydantic). Fields: `name`, `data`,
`predict`, `evaluate`, `description=""`, `notes=""`, `extra_params={}`. The nested
blocks (`data`, `predict`, `evaluate`) are **raw dicts**, not typed sub-models —
their keys are read ad-hoc by the runner. `from_yaml(path)` does
`yaml.safe_load` then `from_dict`, which validates that `name`/`predict`/`evaluate`
exist, `predict` has a `strategy`, and `evaluate` has a `metric`. The top-level
YAML key `params:` maps to `extra_params` (this is where `exog` lives — distinct
from `predict.params`).

Accessor properties: `predict_strategy`, `predict_params`, `metric`, `baseline`.

### 4.4 `PredictContext` — `runner/context.py`

The single argument every predict strategy receives. Fields:
`train`, `test`, `historical` (DataFrames), `side="supply"`, `features="interp"`,
`split="canonical"`, `lag_days=1`, `exog_provider=None`, `extra={}` (holds
`{"k": k}` for the kmedoids track). `historical = concat(train, test)` so lag
lookups can legally reach into the test window (the day-before of test day 2 is
test day 1).

### 4.5 Predict strategies: two layers

There are **two forms** of predict strategy, both stored in the registry:

1. **Estimator classes** subclassing `CurvePredictor` (`core/predictor.py`):
   `fit(train_embeddings) -> self` and
   `predict(test_embeddings, historical_embeddings) -> DataFrame`. Example:
   `StratifiedKNNRetrievalPredictor`, `RidgeRetrievalPredictor`, `Lag1Predictor`.
2. **Registered callables** `f(ctx, **params) -> DataFrame` that the runner calls.
   Most are thin wrappers created by helpers in `stages/predict/_strategies.py`:
   - `_build(cls, ctx, params)` instantiates the class, **auto-forwarding
     `ctx.exog_provider`** iff the constructor accepts an `exog_provider` param and
     the experiment didn't set one.
   - `_fit_predict(model, ctx)` = `model.fit(ctx.train); model.predict(ctx.test, ctx.historical)`.
   - `_simple(name, cls)` registers `name` to `_fit_predict(_build(cls, ctx, params))`
     — so for `_simple`-registered strategies the accepted `params` are exactly the
     wrapped class's constructor arguments.

**Orchestration strategies** (`stages/predict/_orchestration.py`) do more than wrap
one estimator — `static_retrieval` (fit once) and `rolling_refit` (walk-forward
monthly refit) both build a `StratifiedKNNRetrievalPredictor` internally. These
are the strategies the current `vgrid` experiments use.

> **Invariant:** retrieval strategies emit the **actual `(qty, price)` of a real
> retrieved row** (`output_mode="top1_curve"`), never a synthesized/interpolated
> curve. Some non-retrieval strategies (blends, residual-on-grid correctors,
> `ngboost_per_slice`, `same_weekday_avg`) *do* synthesize a curve on the
> `DEFAULT_QUANTITY_GRID`; those are documented per-strategy in §9.

### 4.6 The Metric base — `core/metric.py`

`Metric` is a frozen dataclass: `name`, `pair_fn`, `default_kwargs`. It exposes
`distance(true_qty, true_price, pred_qty, pred_price)` and derives the column
names `curve_col = f"curve_{name}"` and `baseline_col = f"baseline_{name}"`. This
is the **"metric is a parameter"** pattern: the chosen metric object names every
per-row and summary column, so the scoring/summary/leaderboard code is
metric-agnostic.

### 4.7 `ExperimentResult` — `core/result.py`

Dataclass holding `experiment_name`, `model_name`, `params`, `metrics` (the
summary dict), `predictions` (the scored DataFrame), `notes`, `mlflow_run_id`.
`save(dir)` writes `predictions.pkl` + a JSON-safe `result.json`; `load(dir)`
rehydrates. This is the unit the cache and MLflow both persist.

### 4.8 Dataset resolution — `core/datasets.py`

`load_features(side, features, split, k=None) -> (train, test)` maps the data
block to cached pickles under `artifacts/embeddings/`:

- `interp`  → `train_{side}.pkl`, `test_{side}.pkl`
- `v_grid`  → `train_v_grid_{side}.pkl`, `test_v_grid_{side}.pkl`
- `kmedoids`→ `train_kmedoids_{side}_k{k}.pkl`, `test_kmedoids_{side}_k{k}.pkl`
  (requires `k`, canonical only)

For **canonical** the pre-split pickles are read directly. For a **recut**
(`split in SPLITS`, e.g. `user2024`) it concatenates train+test, sorts, and
re-cuts by the `Split` timestamps (v_grid recuts read a `_{split}`-suffixed file;
interp reuses the canonical tables). `historical_from(train, test)` returns
`concat(train, test)`.

---

## 5. The runner — `runner/run.py`

`run_experiment(experiment, *, log_to_mlflow=True, subset=None, exog_provider=None)`
executes, in order (`run.py:52-129`):

1. `bootstrap()` — register strategies (idempotent).
2. Read the data block: `side`/`features`/`split`/`k`; `lag_days` =
   `data.lag_days` else `predict.params.lag_days` else `1`.
3. `train, test = load_features(side, features, split, k=k)`; optional `_subset`
   (fast logic check: first N test days).
4. `historical = historical_from(train, test)`.
5. If no provider passed and `extra_params["exog"]` is set, build one via
   `build_exog_provider(spec, side=side, lag_days=lag_days)` — **shifted by the
   horizon so it never peeks inside the gap** (see the weather caveat in doc 1 §4.5).
6. Build `PredictContext`.
7. `predict_strategy = get("predict", experiment.predict_strategy)`;
   `predictions = predict_strategy(ctx, **experiment.predict_params)`.
8. `PREDICTIONS.validate(predictions)`.
9. `metric = get("evaluate", experiment.metric)`; `scored = score_predictions(predictions, metric)`.
10. If a baseline is set: run the baseline strategy on the **same ctx**, score it,
    and `attach_baseline` merges its distance onto the same rows.
11. `metrics = summary_stats(scored, metric)`; `SCORES.validate(scored)`; assert
    the metric column is present.
12. Assemble a params dict for logging; build `ExperimentResult`; if
    `log_to_mlflow`, log it and stash the run id; return the result.

**Summary keys produced** (for metric name `n`, e.g. `hc_mae`): `n`, `{n}_mean`,
`{n}_median`, `{n}_p75`, `{n}_p95`, `{n}_std`, and (when a baseline is attached)
`baseline_{n}_mean`, `{n}_delta_vs_baseline` (mean of model−baseline; negative is
better), `{n}_win_rate` (fraction of rows where model < baseline). Deltas/win-rate
are computed only over rows where the baseline matched.

---

## 6. The group runner and comparable groups — `runner/group.py`

A **group** is one `experiments/<side>/<horizon>/<metric>/` folder. Everything
under it is comparable: same side, same horizon (hence the same lag baseline), and
same metric. `run_group(group, use_cache=True, log_to_mlflow=False, subset=None,
save_csv=None)`:

1. Resolve the dotted group name to a directory; `rglob("*.yaml")` **recursively**
   (across all family sub-folders).
2. `_check_comparable` asserts every experiment shares `side`, `lag_days`
   (from `horizon.rstrip("d")`), `metric`, and a single baseline; mixed baselines
   raise.
3. Run (or load from cache) each experiment; on error, record `status="FAIL"` and
   continue.
4. Rank ascending by the metric mean (lower is better), NaN/failures last.
5. On a **full run** (`subset is None`), write the leaderboard to
   `results.md` in the group folder ("Auto-generated … do not edit by hand").

This is the **main entrypoint** for producing a comparison — every comparable
group keeps an up-to-date `results.md` next to its experiments.

---

## 7. Caching — `runner/cache.py`

Results are content-addressed under `artifacts/cache/<key>/`. The key is
`sha256(payload)[:16]` where `payload` (JSON, sorted keys) is:

```python
{ "version": CACHE_VERSION,       # currently 1
  "data":     experiment.data,
  "predict":  experiment.predict,
  "evaluate": experiment.evaluate,
  "exog":     experiment.extra_params.get("exog"),
  "subset":   subset,
  "inputs":   _data_fingerprint(experiment) }  # per input file: [name, size, mtime_ns]
```

Consequences:

- Re-running a group recomputes **only what changed**; adding one experiment
  computes just that one.
- Rebuilding a cached embedding table changes its size/mtime → invalidates only
  the experiments that read it.
- Editing a hyperparameter changes `predict` → invalidates only that experiment.
- **Strategy *code* changes are NOT tracked.** After editing a strategy's Python,
  run with `--no-cache`, `rm -rf artifacts/cache`, or bump `CACHE_VERSION` to
  invalidate everything at once.

The input fingerprint uses **file size + mtime (nanoseconds)**, not a content
hash. The exog-using experiments also fingerprint `train_{side}.pkl`,
`test_{side}.pkl`, and the two external example CSVs.

---

## 8. Experiment design — the YAML format

An experiment is a small YAML at
`experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml`. The **path encodes
the comparable group**: `<side>/<horizon>/<metric>` is the group; `<family>` just
buckets methods of a kind; `<name>` is the leaf. The dotted name
(`<side>.<horizon>.<metric>.<family>.<name>`) is that path with dots. The file's
`evaluate.metric` **must** match its `<metric>` folder.

### 8.1 Annotated schema

```yaml
name: demand.7d.hc_mae.vgrid.rolling_ridge        # must equal the dotted path
description: short human summary
data:
  side: demand          # supply | demand
  features: v_grid      # interp | v_grid | kmedoids
  split: canonical      # canonical | user2024
  lag_days: 7           # the forecast horizon (also selects the lag baseline)
predict:
  strategy: rolling_refit          # a registered "predict" strategy
  params: {query: ridge, ridge_alpha: 200.0, ridge_n_pca: 20}
evaluate:
  metric: hc_mae        # wasserstein | dtw | hc_mae  (must match the folder)
  baseline:
    strategy: lag1
    params: {lag_days: 7}
notes: "optional free text logged to MLflow"
params:                 # OPTIONAL top-level block -> Experiment.extra_params
  exog: {sources: [proxy, weather]}   # only when the strategy consumes exog
```

### 8.2 Worked example — the demand champion (verbatim)

`experiments/demand/7d/hc_mae/vgrid/rolling_ridge_exog.yaml`:

```yaml
name: demand.7d.hc_mae.vgrid.rolling_ridge_exog
description: 'Champion: v_grid retrieval, rolling ridge refit, with proxy and weather
  exogenous features.'
data:
  side: demand
  features: v_grid
  split: canonical
  lag_days: 7
predict:
  strategy: rolling_refit
  params:
    query: ridge
    ridge_alpha: 200.0
    ridge_n_pca: 20
evaluate:
  metric: hc_mae
  baseline:
    strategy: lag1
    params:
      lag_days: 7
notes: 'Champion: v_grid retrieval, rolling ridge refit, with proxy and weather exogenous
  features.'
params:
  exog:
    sources:
    - proxy
    - weather
```

Note the `exog` block is under the **top-level `params:`** key (a sibling of
`predict:`/`evaluate:`), which the runner reads from `extra_params` and turns into
an `ExogenousProvider`.

### 8.3 Comparability & conventions

- Two experiments are directly comparable **only within the same
  `<side>/<horizon>/<metric>` folder**. The group runner enforces a single shared
  baseline per group.
- Both current groups' experiments are byte-identical between demand and supply
  except for `name:` and `data.side:` (verified) — **except** that `probabilistic/`
  exists only under demand.
- Each family folder has a `<family>.md` explaining its methods in plain English;
  each YAML opens with a comment.

---

## 9. Predict-strategy catalog

There are **34 registered `predict` strategies**. The current experiments use a
subset; the rest are either helpers exposed as strategies or remnants of pruned
tracks (`dtw_oracle`, `dtw_predicted`, `from_run`, `gate_blend`, `selector`).

### 9.1 The two vgrid orchestrators (current champions)

- **`static_retrieval`** (`_orchestration.py:35`) — params
  `query="ridge"`, `ridge_alpha=200.0`, `ridge_n_pca=20`, `bucket="interval_dow"`.
  Fits **once** a `StratifiedKNNRetrievalPredictor(k=1, output_mode="top1_curve",
  ridge_lags=(lag_days,), lag_days=lag_days, exog_provider=ctx.exog_provider)` on
  the whole train set and predicts. With `query="oracle"` it becomes the retrieval
  **ceiling** (queries with the true embedding).
- **`rolling_refit`** (`_orchestration.py:54`) — same params + optional
  `gbm_kwargs`. **Walk-forward:** for each test month `T`, fit a fresh predictor on
  `historical[date <= T − lag_days]` (so training never enters the 7-day gap),
  predict that month, concatenate. This is the strategy for `rolling_ridge` and the
  champion `rolling_ridge_exog`.

**How the ridge query works.** `StratifiedKNNRetrievalPredictor`
(`soft_knn.py:235`) buckets train rows by `(interval, day-of-week)`. Its inner
`RidgeRetrievalPredictor` is fit per interval on lagged embeddings + calendar +
trend (+ exog) to **predict the target's v_grid embedding**. At predict time that
predicted embedding is the *query*; the model returns the **real** curve of the
k=1 nearest bucket neighbour by L2 (≈ HC-MAE in v_grid space). It also emits
`pred_embedding` and leakage-safe `retrieval_dist`/`bucket_size` diagnostics.

### 9.2 Full list, grouped

| Group | Registered names |
|-------|------------------|
| Baseline / lag | `lag1` |
| vgrid orchestration | `static_retrieval`, `rolling_refit` |
| Retrieval (interp) | `knn_retrieval`, `soft_knn`, `stratified_knn`, `ridge_retrieval`, `rf_retrieval`, `gbm_retrieval`, `xgb_retrieval`, `neural_retrieval` |
| Lag blends | `same_weekday_avg`, `ema_baseline`, `blended_lag`, `learned_blend` |
| Correctors | `shift_corrector`, `shift_scale_corrector`, `residual_on_baseline`, `residual_embedding`, `residual_embedding_gbm`, `per_day_shift` |
| Diagnostics | `oracle_shift`, `realized_shift`, `grid_search_oracle_shift` |
| FDA | `fof_snap`, `logderiv_fpca` |
| Metric / distance learning | `wasserstein_metric`, `distance_rank` |
| Probabilistic | `ngboost_per_slice` |
| DTW prototype (pruned track) | `dtw_oracle`, `dtw_predicted` |
| Meta / plumbing | `from_run`, `gate_blend`, `selector` |

Selected defaults worth knowing (from source): `lag1(lag_days=1)`;
`same_weekday_avg(lags=(7,14,21,28))`; `SoftKNN(k=10, query="ridge", tau=5.0,
ridge_lags=(1,2,7,14))`; `StratifiedKNN(k=5, bucket="interval_dow",
ridge_lags=(1,2,7,14))` (overridden by the orchestrators to k=1, ridge_lags=(h,));
`ShiftCorrector(alpha=5.0)`; `ResidualOnBaselineCorrector(alpha=20, n_pca_x=20,
n_pca_y=10)`. The full per-strategy default reference lives in the source and was
mapped exhaustively during documentation.

> **Exogenous wiring gotcha:** only strategies built through `_simple`/`_build`
> (and the two vgrid orchestrators, which pass it explicitly) auto-receive
> `ctx.exog_provider`. The `**params` lazy strategies (`gbm_retrieval`,
> `xgb_retrieval`, `neural_retrieval`, `fof_snap`, `logderiv_fpca`,
> `distance_rank`, `wasserstein_metric`, `ngboost_per_slice`) bypass `_build`, so
> they do **not** get the provider automatically.

---

## 10. The metrics — `stages/evaluate/`

Registered names: `wasserstein` (default), `hc_mae`, `dtw` (lazy-built to avoid
importing `dtaidistance` unless used). See doc 1 §5 for definitions. The scoring
helpers are metric-parameterized in `stages/evaluate/_common.py`
(`score_predictions`, `attach_baseline`, `summary_stats`); a legacy
Wasserstein-only implementation in `metrics.py` is kept and tested to match.

---

## 11. The embeddings and their build scripts — `stages/features/`

| Embedding | Module | Dim | Build script |
|-----------|--------|-----|--------------|
| `interp` | `interp.py` (`curve_to_embedding`) | 64 (`DEFAULT_QUANTITY_GRID`) | `scripts/build_embeddings.py` |
| `v_grid` | `v_grid.py` (`curve_to_v_grid_embedding`) | 64 (`DEFAULT_V_GRID_DIM`), `V_max_global` = 99th-pct of train totals | `scripts/build_v_grid_embeddings.py` |
| `kmedoids` | `kmedoids_distance.py` (PAM on DTW) | K medoids | `scripts/build_kmedoids.py` + `build_kmedoids_embeddings.py` |

All three write a column named `embedding` (kmedoids uses `kmedoids_embedding`),
so `v_grid` tables are drop-in replacements for `interp` tables. See doc 1 §4.4
for what each encodes and which metric its L2 tracks.

---

## 12. Exogenous features — `exogenous/`

`build_exog_provider(spec, side, lag_days)` (`builder.py`) turns an experiment's
`params.exog` spec into an `ExogenousProvider` that concatenates per-source
feature vectors for `(interval, trading_date)` and returns `None` (dropping the
row) if any source has no data. Sources: `proxy` (`MarketStateProxySource`, real,
shifted by `lag_days`), `weather` (`ExternalCsvSource` over
`data/external/weather_example.csv`, `shift_days=1`), `demand`
(`data/external/demand_forecast_example.csv`, `shift_days=0`).

**No-leakage contract:** proxy features for date `d` are read from the day-state at
`d − lag_days`; daily CSVs default to `shift_days=1`; interval CSVs to
`shift_days=0` (assumed day-ahead forecasts). The provider does not re-check
leakage — it trusts the configured shifts. **Known issue:** the weather source's
`shift_days` is hardcoded to 1 in `builder.py` rather than the horizon, so for the
7-day model the weather (currently synthetic) is shifted by 1, not 7 (doc 1 §4.5,
doc 3 caveats). To use real weather, replace `weather_example.csv` with a
same-schema file and (ideally) fix the shift.

---

## 13. MLflow tracking — `tracking/mlflow_utils.py`

- Local file store at `mlruns/` (`MLFLOW_TRACKING_URI`); experiment name
  `aemo-curves` (`MLFLOW_EXPERIMENT_NAME`); both env-overridable.
- `log_experiment_result(result)` logs every `params` value, every numeric
  `metrics` value, tags (`model_name`, `experiment_name`, `notes`), and the saved
  `ExperimentResult` (`predictions.pkl` + `result.json`) as artifacts under
  `result/`; returns the run id.
- `load_run_result(run_id)` / `load_predictions_by_name(name)` / `find_run_by_name`
  rehydrate a run's result or predictions.
- Browse with `uv run aemo-run ui` (default http://localhost:5000).

---

## 14. CLI reference — `aemo-run` (`cli.py`)

| Command | Options | Does |
|---------|---------|------|
| `aemo-run info` | — | Print paths, tracking URI, train/test month ranges. |
| `aemo-run build-curves` | `--side {supply,demand}` | Shell out to `scripts/build_curves.py`. |
| `aemo-run build-embeddings` | `--side {supply,demand}` | Shell out to `scripts/build_embeddings.py`. |
| `aemo-run run <name>` | `--no-mlflow`, `--subset-days N` | Run one experiment by dotted name. |
| `aemo-run group <name>` | `--no-cache`, `--no-mlflow`, `--subset-days N` | Run + rank a whole group. |
| `aemo-run runs` | — | List MLflow runs. |
| `aemo-run ui` | `--port 5000` | Launch the MLflow UI. |

The equivalent scripts (`scripts/run_experiment.py`, `scripts/run_group.py`) take
the same dotted names and flags.

---

## 15. Scripts reference — `scripts/`

| Script | Purpose | Key flags |
|--------|---------|-----------|
| `build_curves.py` | Raw rows → cached curve tables | `--side {supply,demand,both}`, `--splits` |
| `build_embeddings.py` | Attach 64-d `interp` embedding | `--side`, `--splits` |
| `build_v_grid_embeddings.py` | Build `v_grid` tables (+ `V_max_global` meta) | `--side`, `--k 64`, `--quantile 0.99`, `--split`, `--suffix` |
| `build_kmedoids.py` | Fit K medoids (PAM on DTW) | `--side`, `--k 20`, `--sample 480`, `--qty-weight 50`, `--window` |
| `build_kmedoids_embeddings.py` | Per-row distance-to-medoid tables | `--side`, `--k`, `--splits`, `--jobs 4` |
| `run_experiment.py` | Run one experiment | `<name>`, `--no-mlflow`, `--subset-days N` |
| `run_group.py` | Run + rank a group; write `results.md` | `<group>`, `--no-cache`, `--no-mlflow`, `--subset-days N`, `--save file.csv` |
| `compare_experiments.py` | `list` MLflow runs / `compare <ids>` | subcommands |
| `compare_all.py` | Rank all runs by mean Wasserstein | `--top N` |
| `compare_dtw_kmedoids.py` | DTW-track comparison table + plot | `--side` |
| `compare_exog_1d.py` | 1d-exog ablation heatmap (pruned track) | — |
| `fetch_au_weather.py` | Fetch ERA5/Open-Meteo daily weather → `data/external/weather_au_capitals.csv` | — |

### 15.1 Typical build order (once, needs real CSVs under `data/`)

```bash
uv run python scripts/build_curves.py --side both
uv run python scripts/build_embeddings.py --side supply
uv run python scripts/build_embeddings.py --side demand
uv run python scripts/build_v_grid_embeddings.py --side supply
uv run python scripts/build_v_grid_embeddings.py --side demand
```

Then run a group:

```bash
uv run python scripts/run_group.py demand.7d.hc_mae
uv run python scripts/run_group.py supply.7d.hc_mae
```

---

## 16. Tests — `tests/`

`pytest` suite; several tests **skip** gracefully if cached v_grid tables are
absent. Coverage:

- `test_contracts.py` — catalog contracts accept valid frames, reject missing
  columns / duplicate keys / empty frames.
- `test_curves_and_metrics.py` — supply curves non-decreasing, demand
  non-increasing; interp embedding is 64-d and range-preserving; Wasserstein is 0
  on identical curves.
- `test_metric_layer.py` — metric-parameterized scoring/summary column names for
  Wasserstein and HC-MAE match the legacy path; DTW metric is lazy.
- `test_dtw_kmedoids.py` — DTW properties; PAM picks diverse medoids.
- `test_exogenous.py` — WA holidays; proxy day-state table; shifted lookup respects
  `lag_days`; provider returns None if any source lacks data; missing CSV doesn't
  crash.
- `test_experiment_check.py` — **every** `experiments/**/*.yaml` resolves: its
  predict strategy, metric, and baseline are registered and `data` has
  `side`+`features`. (Run this after adding an experiment.)
- `test_group_cache.py` — cache key is stable, insensitive to name/notes, sensitive
  to hyperparameters; group runner returns ranked results, all cached on re-run.
- `test_smoke_runner.py` — the runner produces a valid scored result on a
  2-test-day subset.
- `test_small_families.py` — merged small families export unchanged `name`s and
  emit the required schema.

---

## 17. Analysis & visualization

- `analysis/analysis.py` — metric-parameterized post-hoc statistics: paired
  **cluster bootstrap** (cluster = Trading Date, 95% CI), **Wilcoxon** signed-rank
  with **Holm** correction, per-month tables, oracle capture-%, and `rescore_runs`
  (recompute all runs under a different metric).
- `viz/comparison.py` — cross-run tables and bar/box/residual plots (currently
  hardcoded to Wasserstein column names).
- `viz/curves.py` — step-plots of true vs predicted (vs baseline) curves, and
  embedding comparisons.
- `notebooks/` — inspection notebooks (HC-MAE sanity, v_grid ridge, conformal
  band, blend, etc.), each with a paired `_build_*.py` source.

---

## 18. Environment & setup

- Python **3.11–3.12**; dependency + venv management via **uv**.
- Core deps (`pyproject.toml`): numpy, pandas, scipy, scikit-learn, matplotlib,
  seaborn, mlflow, pyarrow, tqdm, typer, rich, joblib, pyyaml, **dtaidistance**
  (DTW), **scikit-learn-extra**, **torch** (CPU wheel on Linux), **xgboost**.
- Setup:

```bash
uv venv --python 3.11
uv pip install -e .          # add ".[dev]" for jupyter + pytest + ruff
```

- Lint: `ruff` (line length 110; rules E, F, W, I, UP, B). The `aemo-run` console
  script is defined in `pyproject.toml`.

---

## 19. How to contribute / extend (recipes)

### 19.1 Add a new experiment (no code)

1. Create `experiments/<side>/<horizon>/<metric>/<family>/<name>.yaml` naming
   existing strategies. Set `name:` to the dotted path and `evaluate.metric` to
   match the `<metric>` folder.
2. Verify it resolves: `uv run pytest tests/test_experiment_check.py`.
3. Run its group: `uv run python scripts/run_group.py <side>.<horizon>.<metric>`
   (only the new experiment is computed; the rest come from cache). `results.md`
   refreshes automatically.

### 19.2 Add a new method (one strategy)

1. Add a module under `src/aemo_curves/stages/predict/`. Either:
   - an estimator class subclassing `CurvePredictor` (implement `fit`/`predict`,
     return the `PREDICTIONS` columns; add `exog_provider` to `__init__` if it
     should consume exog), then register it via `_simple("<name>", MyClass)` in
     `_strategies.py`; **or**
   - a callable `@register("predict", "<name>") def f(ctx, **params): ...`.
2. Ensure `stages/predict/__init__.py` (directly or via `_strategies.py`) imports
   your module so the `@register` side-effect runs.
3. Write an experiment YAML that uses it. Run with `--no-cache` the first time
   (code changes aren't cache-tracked).
4. If it's retrieval, honor the invariant: emit a **real** curve's `(qty, price)`.

### 19.3 Add a new metric

1. Implement a `curve_<name>_distance(true_qty, true_price, pred_qty, pred_price)`
   in `stages/evaluate/<name>.py`; wrap it in a `Metric("<name>", fn)` and
   `register("evaluate", "<name>")(THE_METRIC)` in `stages/evaluate/__init__.py`.
2. Create a matching `experiments/<side>/<horizon>/<name>/` group folder and set
   `evaluate.metric: <name>` in its YAMLs. All column naming flows automatically
   from the `Metric` object.

### 19.4 Add a new data range / split

1. Add an entry to `config.SPLITS` (a `Split(train_end, test_start, test_end)`).
2. Set `data.split: <key>` in the experiment. For `v_grid` you must first build a
   `_{key}`-suffixed embedding table (`build_v_grid_embeddings.py --split ... --suffix _<key>`).

### 19.5 Add a new embedding

1. Add `stages/features/<name>.py` with a `curve_to_<name>_embedding(...)` and a
   build script writing `train_<name>_{side}.pkl` / `test_<name>_{side}.pkl` with
   an `embedding` column.
2. Register the filename pattern in `core/datasets.py` (`_FEATURE_TABLES` /
   `load_features`) and in `runner/cache.py` (`_feature_files`) so the cache
   fingerprints it.
3. Reference it via `data.features: <name>`.

---

## 20. Conventions & gotchas (checklist)

- **Never tune on test.** The test window is fixed in `config.py`.
- **Pickle, not parquet**, for any table whose cells hold numpy arrays.
- **Historical lookups span train∪test** — the day-before of test day 2 is legal;
  only the current day's curve is hidden.
- **Retrieval returns a real curve**, never an interpolated artefact.
- **Most strategies fit 48 per-interval models** (intra-day seasonality).
- **Strategy code changes bypass the cache** — use `--no-cache` / clear cache /
  bump `CACHE_VERSION`.
- **`evaluate.metric` must equal the `<metric>` folder** (checked by the group
  runner and `test_experiment_check.py`).
- **The exog block lives under top-level `params:`**, not `predict.params`.
- **Registry names silently overwrite** — pick unique strategy names.
- **`--subset-days N` is a logic check, not an evaluation** (first N test days only).

---

## 21. Design patterns, summarized

| Pattern | Where | Payoff |
|---------|-------|--------|
| **Plugin registry** (decorator + import side-effect) | `core/registry.py` | New methods/metrics without touching the runner. |
| **Pipeline of stages with fixed contracts** | `catalog/`, `contracts.py` | Wrong-shape outputs fail loudly at the boundary. |
| **Experiment-as-data (declarative YAML)** | `core/experiment.py` | New experiments = config, not code. |
| **Path-encoded comparable groups** | `experiments/…`, `runner/group.py` | Only same-group results compared; shared baseline enforced. |
| **Metric as a parameter** | `core/metric.py`, `_common.py` | Scoring/summary/leaderboard code is metric-agnostic. |
| **Content-addressed caching** | `runner/cache.py` | Re-runs recompute only what changed. |
| **Strategy wrapping + capability injection** | `_strategies.py` (`_build`/`_simple`) | Estimators become strategies; exog auto-forwarded when supported. |
| **Two-layer strategies (orchestrator over estimator)** | `_orchestration.py` over `soft_knn.py` | Walk-forward refit / oracle variants reuse one predictor. |
| **Real-curve retrieval invariant** | all retrieval predictors | Predictions keep true step structure. |
