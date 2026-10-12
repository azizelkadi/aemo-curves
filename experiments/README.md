# Experiments

Each experiment is one YAML file in a `<side>/<horizon>/<metric>/<family>/<name>.yaml`
tree:

```
experiments/
  demand/7d/hc_mae/
    baseline/lag7.yaml
    weekly/same_weekday_avg.yaml
    vgrid/{strat_knn,rolling_ridge,rolling_ridge_exog,oracle}.yaml
    fda/{fof_snap,logderiv_fpca}.yaml
    metric_learn/wasserstein_metric.yaml
    distance/distance_rank.yaml
    probabilistic/ngboost_per_slice.yaml        (demand only)
    corrector/{shift,shift_scale,residual_on_baseline,residual_gbm,per_day_shift}.yaml
    blend/{ema_baseline,blended_lag,learned_blend}.yaml
    diagnostic/{oracle_shift,realized_shift}.yaml
    retrieval/{gbm_retrieval,soft_knn}.yaml
  supply/7d/hc_mae/        (the same, minus probabilistic/ngboost_per_slice)
```

Every experiment YAML opens with a short comment, and each family folder has a
`<family>.md` (for example `blend/blend.md`) explaining its experiments in plain
English.

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

Each full group run writes an up-to-date `results.md` leaderboard into the group
folder (for example `demand/7d/hc_mae/results.md`). Group runs are cached under
`artifacts/cache/`, so re-running after adding one experiment computes only the
new one. Run a single experiment by its dotted name (the dots are the folders):

```bash
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge_exog
```

See [`docs/architecture.md`](../docs/architecture.md) for the format,
[`CLAUDE.md`](../CLAUDE.md) for the registered strategies and metrics, and
[`docs/results.md`](../docs/results.md) for the measured results.

## Current groups

Both groups predict the 7-day-ahead curve, scored under HC-MAE against the lag-7
baseline on the canonical split (n = 17,232 test auctions). `demand/7d/hc_mae`
runs against the cached demand v_grid tables; `supply/7d/hc_mae` needs the supply
v_grid tables first (`scripts/build_v_grid_embeddings.py --side supply`).

### Metrics

An experiment sets `evaluate.metric`; the metric names the per-row and summary
columns. Both current groups use `hc_mae`.

| Metric | `evaluate.metric` | What it measures | Where used |
|--------|-------------------|------------------|------------|
| Wasserstein-1 (default) | `wasserstein` | Mean optimal-transport distance between predicted and true curve (area between their inverse-CDFs), $/MWh. | default; earlier cross-horizon studies |
| Dynamic time warping | `dtw` | Shape-aligned distance with elastic matching along the curve. | the DTW / k-medoids prototype track |
| Horizontal-cut MAE | `hc_mae` | Mean absolute price gap between the two curves read across the volume axis, $/MWh. | the two active groups below |

### Approaches

One row per experiment (present in both groups unless noted). Each YAML carries a
comment block with the same explanation in more detail.

| Family | Experiment | Strategy | What it does |
|--------|------------|----------|--------------|
| baseline | `lag7` | `lag1` (lag 7) | Curve at the same interval 7 days earlier. The reference everything is measured against. |
| weekly | `same_weekday_avg` | `same_weekday_avg` | Learning-free mean of the last four same-weekday curves (lags 7/14/21/28). |
| vgrid | `strat_knn` | `static_retrieval` | Static per-(interval,dow) ridge-query k=1 retrieval in the v_grid embedding. |
| vgrid | `rolling_ridge` | `rolling_refit` | Same, with the ridge query re-fit each test month. |
| vgrid | `rolling_ridge_exog` | `rolling_refit` + exog | **Demand champion**: adds horizon-shifted proxy + weather features. |
| vgrid | `oracle` | `static_retrieval` (oracle) | **Ceiling** (not deployable): closest train curve to the *true* curve in the bucket. |
| fda | `fof_snap` | `fof_snap` | FDA idea #5. Compress curves with PCA, predict next week's shape with a ridge, then snap to the nearest real curve. |
| fda | `logderiv_fpca` | `logderiv_fpca` | FDA idea #8. Like fof_snap, but it works on the slope of the curve. |
| metric_learn | `wasserstein_metric` | `wasserstein_metric` | Idea #18. Learn a space where plain distance matches the curve distance, then retrieve the nearest curve in it. |
| distance | `distance_rank` | `distance_rank` | A boosted-tree model predicts the distance from a target to each candidate, and the closest one is retrieved. |
| probabilistic | `ngboost_per_slice` | `ngboost_per_slice` | Idea #11. One boosted-tree model per price slice, joined into a monotone curve (**demand only**). |
| corrector | `shift` | `shift_corrector` | Ridge predicts one price shift on the lag-7 curve (keeps its step structure). |
| corrector | `shift_scale` | `shift_scale_corrector` | Predicts a price shift **and** a quantity scale for the lag-7 curve. |
| corrector | `residual_on_baseline` | `residual_on_baseline` | Predicts a 64-d price residual on the lag-7 step positions. |
| corrector | `residual_gbm` | `residual_embedding_gbm` | Nonlinear (HistGBM) version of the residual corrector. |
| corrector | `per_day_shift` | `per_day_shift` | One shift per trading day, pooled across all 48 intervals. |
| blend | `ema_baseline` | `ema_baseline` | Fixed-weight average of lag-7/14/21. |
| blend | `blended_lag` | `blended_lag` | Learned per-(interval,dow) blend of lag-7 and lag-14. |
| blend | `learned_blend` | `learned_blend` | Per-bucket simplex weights over lag-7/14/21/28. |
| diagnostic | `oracle_shift` | `oracle_shift` | **Ceiling**: median-optimal shift of the lag-7 curve (uses the truth). |
| diagnostic | `realized_shift` | `realized_shift` | Deployable: persist a fraction of the recent realised shift. |
| retrieval | `gbm_retrieval` | `gbm_retrieval` | A per-interval boosted-tree model predicts a point, then the nearest real curve is retrieved. |
| retrieval | `soft_knn` | `soft_knn` | Soft top-K kNN (ridge query); emits the real top-1 curve. |

## Results

HC-MAE in $/MWh (lower is better); Δ is vs the lag-7 baseline (negative = better).
Snapshots of the auto-generated `<side>/7d/hc_mae/results.md` (re-run a group to
refresh).

### Demand, 7-day-ahead

| Experiment | HC-MAE | Δ vs lag-7 | Win % |
|------------|-------:|-----------:|------:|
| `vgrid.oracle` *(ceiling)* | 27.96 | -121.01 | 93.8 |
| `vgrid.rolling_ridge_exog` *(champion)* | 123.75 | -25.21 | 60.0 |
| `vgrid.rolling_ridge` | 127.57 | -21.40 | 58.6 |
| `vgrid.strat_knn` | 129.41 | -19.56 | 57.6 |
| `distance.distance_rank` | 137.87 | -11.10 | 52.8 |
| `weekly.same_weekday_avg` | 140.60 | -8.37 | 51.8 |
| `corrector.shift_scale` | 142.03 | -6.94 | 56.8 |
| `blend.blended_lag` | 145.33 | -3.63 | 50.5 |
| `diagnostic.oracle_shift` *(shift ceiling)* | 145.48 | -3.49 | 43.6 |
| `metric_learn.wasserstein_metric` | 145.80 | -3.17 | 49.5 |
| `blend.learned_blend` | 147.36 | -1.61 | 49.6 |
| `blend.ema_baseline` | 147.36 | -1.61 | 51.6 |
| `baseline.lag7` | 148.97 | 0.00 | - |
| `corrector.per_day_shift` | 150.04 | +1.07 | 33.3 |
| `corrector.shift` | 152.68 | +3.71 | 32.4 |
| `diagnostic.realized_shift` | 155.45 | +6.48 | 9.7 |
| `corrector.residual_gbm` | 157.81 | +8.85 | 39.3 |
| `corrector.residual_on_baseline` | 158.60 | +9.63 | 42.5 |
| `probabilistic.ngboost_per_slice` | 177.01 | +28.05 | 38.7 |
| `retrieval.soft_knn` | 190.18 | +41.21 | 36.1 |
| `retrieval.gbm_retrieval` | 190.76 | +41.79 | 36.8 |
| `fda.logderiv_fpca` | 195.08 | +46.11 | 37.2 |
| `fda.fof_snap` | 243.89 | +94.93 | 25.4 |

The v_grid retrieval family leads (champion -25.2). Of the added methods,
`distance_rank` (-11.1) and `shift_scale` (-6.9) are strongest and both beat the
weekly average; metric learning (#18) is modest (-3.2); the FPCA-snap methods
(#5/#8) hurt - snapping to a whole different curve is poison under HC-MAE here.

### Supply, 7-day-ahead

| Experiment | HC-MAE | Δ vs lag-7 | Win % |
|------------|-------:|-----------:|------:|
| `vgrid.oracle` *(ceiling)* | 47.07 | -7.77 | 65.4 |
| `diagnostic.oracle_shift` *(shift ceiling)* | 54.55 | -0.29 | 10.8 |
| `baseline.lag7` | 54.83 | 0.00 | - |
| `diagnostic.realized_shift` | 55.95 | +1.11 | 2.2 |
| `corrector.per_day_shift` | 56.00 | +1.16 | 7.8 |
| `blend.blended_lag` | 56.42 | +1.58 | 46.1 |
| `blend.ema_baseline` | 56.54 | +1.70 | 46.3 |
| `corrector.shift_scale` | 56.94 | +2.10 | 37.6 |
| `corrector.shift` | 57.31 | +2.47 | 8.4 |
| `weekly.same_weekday_avg` | 59.28 | +4.45 | 43.1 |
| `blend.learned_blend` | 59.33 | +4.49 | 42.4 |
| `corrector.residual_on_baseline` | 61.79 | +6.96 | 30.5 |
| `vgrid.rolling_ridge_exog` | 68.34 | +13.50 | 40.7 |
| `vgrid.rolling_ridge` | 70.79 | +15.95 | 37.9 |
| `corrector.residual_gbm` | 72.32 | +17.49 | 20.7 |
| `distance.distance_rank` | 77.86 | +23.03 | 34.2 |
| `metric_learn.wasserstein_metric` | 79.30 | +24.47 | 36.5 |
| `retrieval.soft_knn` | 82.23 | +27.40 | 30.6 |
| `vgrid.strat_knn` | 84.72 | +29.88 | 27.8 |
| `fda.fof_snap` | 86.33 | +31.50 | 26.6 |
| `retrieval.gbm_retrieval` | 89.93 | +35.10 | 27.6 |
| `fda.logderiv_fpca` | 163.81 | +108.98 | 20.8 |

On supply **nothing deployable beats lag-7**. Even the shift ceiling
(`oracle_shift`, -0.29) barely clears it, so there is essentially no headroom for
adjusting the lag curve; the closest challengers are the cheap correctors/blends
(within ~1.2-2.5), while retrieval and FPCA methods lose.
