# Closing the oracle-KNN gap on demand 1-day-ahead

Status: this plan was carried out, see [`experiments/retrieval_demand_1d/`](../experiments/retrieval_demand_1d/).

## Context

On demand-side curves at horizon 1d, the **KNN-retrieval oracle**
(`knn_retrieval` with `query="oracle"`) collapses mean Wasserstein from
the lag-1 baseline of 82.67 down to **25.78** ($/MWh) - a 69 % drop, with
85.4 % of test rows beating baseline. The oracle uses the *true* test
embedding as the query into the training library; it answers the
question "if you could pick any one historical training curve to be your
forecast, how good would it be?".

The realised methods we have run today close almost none of that gap:

| Model | Mean W | Δ vs lag-1 | Win % |
|---|---:|---:|---:|
| `knn_retrieval` (oracle query) - ceiling | 25.78 | **-56.89** | 85.4 % |
| `knn_retrieval` (lag-1-embedding query) | 86.60 | +3.93 | 38.3 % |
| `ridge_retrieval` (predicted embedding) | 93.18 | +10.51 | 38.5 % |
| `baseline_lag1` | 82.67 | 0.00 | - |

The gap from 86.60 to 25.78 (≈ 61 $/MWh) is therefore *all* attributable
to **bad embedding prediction**, not to a missing curve in the library.
This document proposes six experiments - five learners and one
diagnostic - designed to close that gap.

All experiments live under a new comparable folder
`experiments/retrieval_demand_1d/` with the same triple as `horizon_1d/`
(lag-1 baseline, 64-d interp embedding, Wasserstein metric). Each
experiment ships a `run_experiment(...)` function, a row in the folder
README's comparison table, and an MLflow run.

The canonical test window remains 2024-05 → 2025-05 (n = 17 232).

---

## Diagnostic 1 - Rank of oracle neighbour in predicted ordering

### Why it's first

The cheapest experiment also tells us *which* of the gap-closing
strategies below is worth pursuing. For each test row:

1. Compute the predicted embedding from `ridge_retrieval`.
2. Sort all 58 368 training rows by L2 distance to the predicted
   embedding.
3. Find the rank of the training row that the oracle would have
   chosen (the row closest to the *true* embedding).

The distribution of ranks tells us:

- if the oracle's neighbour usually sits at rank ≤ 20 in the predicted
  ordering, **soft k-NN over the top-20** closes most of the gap (cheap fix);
- if it sits at rank 50-500, **stratified retrieval** (restricting the
  library to same-interval / same-dow rows) is likely to help;
- if it sits at rank > 1000, the **predictor itself** is the bottleneck -
  GBM / MLP / richer features needed.

### What is measured

For each test row, the rank of the oracle neighbour in the
ridge-predicted ordering. Aggregate stats: median, p50, p90, p99,
fraction at rank 1, fraction in top 10, top 100, top 1000.

### Baseline

None - this is a diagnostic (no comparison vs lag-1).

### Methodology

Single script
`experiments/retrieval_demand_1d/diag_oracle_rank.py`. Re-uses the
already-fitted Ridge models from
`experiments/horizon_1d/ridge_retrieval.py` (or refits per interval).
Per row, computes both the oracle and the predicted argmin and the rank
of the oracle in the predicted ordering. Saves the rank distribution
to `artifacts/diag_oracle_rank_demand.csv` plus a histogram PNG.

### Test data

17 232 demand test auctions (2024-05 → 2025-05). Library = 58 368
training auctions (2021-01 → 2024-04).

### Metrics

- `median_rank`, `p90_rank`, `p99_rank`
- `share_at_rank_1` (= win rate of Ridge being the oracle picker)
- `share_in_top_10`, `share_in_top_100`, `share_in_top_1000`

### Success criterion

There is no pass/fail - the histogram is the output. We act on the
distribution: it tells us which experiment below to invest in.

### Cost / wall-clock

~5 min (one-shot per-row computation; no training loops).

---

## Experiment B1 - Soft k-NN retrieval

### Why this could close the gap

If the oracle's neighbour usually sits in the top 5-20 of the predicted
ordering, replacing argmin with a weighted average of the top-K
candidates produces a curve close to the oracle without ever needing to
predict the embedding perfectly.

### What is measured

Same metrics as `knn_retrieval` and `ridge_retrieval`:
`wasserstein_mean`, `wasserstein_delta_vs_baseline`,
`wasserstein_win_rate`. New aux column: `n_neighbors_used`.

### Baseline

`Lag1Predictor(lag_days=1)` on demand (mean W = 82.67), the same
baseline as the rest of `horizon_1d/`. Plus the existing
`knn_retrieval` (oracle query) as the **ceiling**.

### Methodology

New model `aemo_curves.stages.predict.soft_knn.SoftKNNRetrievalPredictor`:

1. Fit per-interval embedding library (just sub-set training rows by
   interval, no learning).
2. At test time, predict an embedding via Ridge (re-use `RidgeRetrievalPredictor`).
3. Find top-K nearest training rows by L2 to the predicted embedding.
4. Either:
   - weighted **average of curve embeddings** with weights
     `exp(-distance / τ)`, decode via `embedding_to_curve(total_qty)`;
   - **or** weighted average of (qty, price) on a common quantile grid;
   - **or** simple mean of K embeddings with uniform weights.

   Decode the averaged embedding via `embedding_to_curve` using the
   total quantity of the top-1 retrieved curve.

5. Output curve = (DEFAULT_GRID × total_qty, weighted-avg embedding).

### Test data

Same 17 232 rows.

### Metrics

- `wasserstein_mean`, `wasserstein_delta_vs_baseline`, `wasserstein_win_rate`
- Bonus: average of `dtw_predicted` (cross-metric sanity)

### Success criterion

Δ < -5 $/MWh below baseline (i.e., capture ≥ 9 % of the 56.89 oracle gap).

### Hyperparameters to sweep

K ∈ {1, 3, 5, 10, 20, 50, 100}, weights ∈ {uniform, exp(-d/τ) for several τ}.

### File layout

```
src/aemo_curves/models/soft_knn.py            # new SoftKNNRetrievalPredictor
experiments/retrieval_demand_1d/
├── README.md
├── __init__.py
├── baseline_lag1.py            # re-runs lag-1 demand under this group's logging
├── knn_oracle_ceiling.py       # re-runs knn_retrieval(oracle) for context
├── knn_lag1_query.py           # re-runs knn_retrieval(lag1) for context
├── ridge_retrieval.py          # re-runs the existing ridge_retrieval on demand
├── soft_knn_uniform_k5.py
├── soft_knn_uniform_k20.py
├── soft_knn_softmax_tau1_k20.py
└── soft_knn_softmax_tau5_k20.py
```

### Cost / wall-clock

~10 min total for 4-8 K/τ combinations.

---

## Experiment B2 - Stratified retrieval (same interval, same day-of-week)

### Why this could close the gap

The library is currently 58 368 candidates. A demand curve at Saturday
13:00 is very different from a Tuesday 06:00 curve. Restricting the
retrieval to training rows at the same `Interval Number` *and* the same
day-of-week shrinks the bucket to ≈ 170 candidates (40 weeks × ~4 years
of one weekday × one interval) and removes most of the noise that the
predictor would have to filter out.

### What is measured

Same as B1.

### Baseline

Lag-1 baseline + existing `knn_retrieval` (oracle query) on demand.

### Methodology

New model
`aemo_curves.stages.predict.soft_knn.StratifiedKNNRetrievalPredictor`:

1. Indexes the training library by `(interval, dow)` → list of rows.
2. At test time, for `(target_date, interval)` look up the bucket
   `(interval, target_date.dayofweek)`.
3. Use either a Ridge-predicted query embedding or the lag-1 embedding
   directly (configurable). Find top-1 (or top-K) nearest in the
   bucket. Output that curve (or weighted average).

### Test data

Same 17 232 rows.

### Metrics

Same as B1 plus `bucket_size_mean` (sanity check).

### Success criterion

Δ < -5 $/MWh below baseline. Stretch: Δ < -15.

### Hyperparameters

- bucket key: `(interval, dow)` vs `(interval, is_weekend)` vs
  `(interval, dow, holiday)`
- K ∈ {1, 5, 10}
- query: `ridge` vs `lag1` vs `oracle` (oracle = stratified-retrieval ceiling)

### File layout

```
src/aemo_curves/models/stratified_knn.py
experiments/retrieval_demand_1d/
├── strat_knn_oracle.py            # new ceiling under stratification
├── strat_knn_lag1.py              # lag1-query within bucket
├── strat_knn_ridge.py             # ridge-query within bucket
└── strat_knn_holiday_aware.py     # bucket = (interval, dow, holiday)
```

### Cost / wall-clock

~5-10 min total.

---

## Experiment A2 - Per-interval Gradient Boosting embedding predictor

### Why this could close the gap

`ridge_retrieval` is linear; if the demand-curve embedding has
non-linear dependencies on lag features (it almost certainly does for
volatile / spike days), GBM should improve the predicted embedding's
accuracy. The supply-side residual GBM was inconclusive at K=20; this
experiment is different - it *predicts the embedding directly*, then
retrieves.

### What is measured

Same as B1.

### Baseline

`ridge_retrieval` + lag-1 baseline.

### Methodology

New model
`aemo_curves.stages.predict.regression_retrieval.GBMRetrievalPredictor`:

1. Per-interval `HistGradientBoostingRegressor` wrapped in
   `MultiOutputRegressor` for K-d output. To keep training tractable,
   reduce the target to PCA-projected embedding (K-pca = 10) and
   reconstruct.
2. Features: lag-{1, 2, 7, 14} embeddings + calendar.
3. Predict embedding → retrieve nearest training row → output its curve.

### Test data

Same 17 232 rows.

### Metrics

Same as B1.

### Success criterion

Δ < `ridge_retrieval` Δ on demand (i.e., < +10.51). Stretch: Δ < 0.

### File layout

```
src/aemo_curves/models/gbm_retrieval.py
experiments/retrieval_demand_1d/
├── gbm_retrieval_default.py
└── gbm_retrieval_deeper.py        # max_depth=6, more iterations
```

### Cost / wall-clock

~30-60 min for the GBM training (per-interval HistGBM × 64 outputs is
the bottleneck - needs joblib parallelism over intervals to be tractable).

### Risk

GBM training may be too slow at K=64 outputs even with PCA. Falls back
to PCA(8) if needed.

---

## Experiment E1 - K-medoid framework scored under Wasserstein

### Why this could close the gap

We already have a K=24 demand medoid library (built for
`dtw_kmedoids_demand/`). The K-medoid *oracle* under DTW gave Δ=-92.62
(a 38 % win on win-rate 52 %). A natural sister: keep the medoid
library, keep the same Ridge predictor, but score the predicted
medoid's curve under **Wasserstein** instead of DTW. This isolates the
"predict-K-d-distance-vector" signal under the canonical 1d metric.

### What is measured

`wasserstein_mean`, `wasserstein_delta_vs_baseline`,
`wasserstein_win_rate` on the medoid-predicted curve, plus the
DTW-track diagnostic columns.

### Baseline

Lag-1 demand baseline (same as the rest of the folder).

### Methodology

A thin wrapper around `experiments/dtw_kmedoids_demand/predicted_kmedoids_ridge.py`
that *also* scores under Wasserstein at the end. No new models; just an
additional scoring pass and a separate `ExperimentResult`.

### Test data

Same 17 232 rows.

### Metrics

`wasserstein_mean`, `wasserstein_delta_vs_baseline`,
`wasserstein_win_rate`, plus `n_distinct_medoids_chosen` (sanity check).

### Success criterion

Δ < 0 vs baseline. Stretch: Δ matches `same_weekday_avg_30d` margin
scaled to horizon 1d.

### File layout

```
experiments/retrieval_demand_1d/kmedoid_oracle_under_w.py
experiments/retrieval_demand_1d/kmedoid_predicted_ridge_under_w.py
```

### Cost / wall-clock

~5 min.

---

## Experiment B3 - Stratified soft k-NN with Ridge query *(combined)*

### Why this could close the gap

Combines B1 (soft averaging) with B2 (stratified bucket) on top of a
Ridge-predicted query embedding. Each component above closes part of the
gap; combining them should compound the wins.

### What is measured

Same as B1.

### Baseline

Lag-1 demand. Plus `knn_oracle_ceiling`, `soft_knn_*`, and `strat_knn_*`
as comparators (all in the same folder).

### Methodology

Composition: stratify by `(interval, dow)`, sort the bucket by L2 to the
Ridge-predicted embedding, average the top-K with softmax weights.
Output the averaged embedding decoded with the bucket's median total qty.

### Hyperparameters to sweep

- K ∈ {3, 5, 10}
- bucket: `(interval, dow)`
- weight: `softmax(-d / τ)` with τ ∈ {1, 5, 10}

### File layout

`experiments/retrieval_demand_1d/strat_soft_knn_ridge.py`

### Success criterion

Δ < min(Δ from B1, Δ from B2). Stretch: Δ < -15 (capture ≥ 25 % of the
oracle gap).

### Cost / wall-clock

~10 min.

---

## Recommended execution order

| Order | Experiment | Why |
|---|---|---|
| 0 | **D1** Oracle-rank diagnostic | Cheap; tells us which of B1/B2/A2 is the right next bet. |
| 1 | **B2** Stratified retrieval (lag1 query + ridge query) | Largest expected win per minute of compute. |
| 2 | **B1** Soft k-NN over the global library | Likely additive to B2, easy to ablate K and τ. |
| 3 | **B3** Stratified soft k-NN with Ridge query | Combination; runs once B1 and B2 are tuned. |
| 4 | **E1** K-medoid framework under Wasserstein | Re-uses cached artefacts; one-script win. |
| 5 | **A2** GBM embedding predictor | Heaviest; only worth it if D1 says the predictor is the bottleneck. |

A pragmatic minimum to **prove closure of the gap** is D1 + B1 + B2 +
B3. That's < 30 minutes of wall-clock and produces 6-12 MLflow runs.

## Folder & comparability

All proposals share:
- baseline = lag-1 demand
- embedding = 64-d interp
- metric = Wasserstein
- test rows = the same 17 232

So they are mutually comparable inside `experiments/retrieval_demand_1d/`.
The folder README will host the headline comparison table.

## What this can lead to (publication-wise)

If even one of B1/B2/B3 yields Δ ≤ -15 $/MWh on demand 1d, we have:

1. A retrieval-style demand-curve forecaster that beats lag-1 by a
   meaningful margin (paper headline).
2. An *oracle gap analysis* showing 70-80 % of the achievable signal
   remains untapped - a prediction-bottleneck framing that motivates
   deeper-learning follow-ups.
3. A complementary story to the `dtw_kmedoids_demand/` oracle finding:
   same conclusion (demand manifold is highly clusterable), reached
   through two independent metrics.
