# HC-MAE results across horizons (1 day, 7 days, 30 days)

## What this document is

A self-contained summary of how well we can predict the next-day, next-week,
and next-month **demand curves** of the Western Australia STEM electricity
auction, measured by the HC-MAE distance between predicted and true curve.

Same data, same test rows, same evaluation everywhere - only the
**how-far-ahead horizon** changes.

## Setup

- **What we predict.** For each 30-minute auction, the full demand curve
  (the buyers' bid stack). Each curve is a step function relating cumulative
  quantity (MW) to price ($/MWh).
- **Data.** Half-hourly WA STEM auctions, 48 intervals/day. Provided as monthly CSVs.
- **Train window.** January 2021 → April 2024 (40 months, ~58 000 auctions).
- **Test window.** May 2024 → April 2025 (12 months, **n = 17 232 auctions**).
- **Horizons compared.**
  - **1 day ahead** - predict tomorrow's curve using data up to today.
  - **7 days ahead** - predict next week's curve using data up to today.
  - **30 days ahead** - predict next month's curve using data up to today.
- **Metric: HC-MAE** (Sinha & Lucheroni 2025) - mean absolute price
  difference between the true and predicted curve, integrated over the
  volume axis. Lower is better. Units: $/MWh.
- **Baseline at each horizon.** The curve at the same interval N days
  earlier (`lag-N`). Every model has to beat this.
- **Oracle ceiling.** The HC-MAE we'd get if we could *perfectly* pick the
  closest historical curve inside the (interval, day-of-week) bucket of the
  target. Not deployable - used only as a "how much headroom is left?" ruler.
- **Champion architecture.** v_grid retrieval - pick the historical curve
  whose v_grid embedding is closest in Euclidean distance to a Ridge-predicted
  target embedding, restricted to the same (interval, day-of-week) bucket.
  Rolling monthly Ridge refit. Adds 7-day-shifted weather and market-state
  proxy features for the 1-day and 7-day champion.

## Cross-horizon headline - 3-row summary

One row per horizon, only the best realised model. All values in $/MWh.

| Horizon | Best model | Baseline HC-MAE | Best HC-MAE | **Δ vs baseline** | Win rate | Oracle ceiling | Available gap (baseline - oracle) | Gap closed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 day** | `vgrid_rolling_ridge_exog` | 122.81 | **111.89** | **-10.93** | 57.2 % | 27.96 | 94.85 | 11.5 % |
| **7 days** | `vgrid_rolling_ridge_exog` | 148.97 | **123.41** | **-25.56** | 60.3 % | 27.96 | 121.01 | 21.1 % |
| **30 days** | `vgrid_rolling_ridge`| 171.54 | **132.62** | **-38.92** | 62.9 % | 27.96 | 143.58 | 27.1 % |

**Reading.** The same retrieval experiment wins at every horizon. The further
out we predict, the **bigger the absolute win** ($/MWh) and the **larger
share of the available headroom** we close - because the baseline gets
much worse with horizon while the oracle stays the same.

## Per-horizon tables

Each table shows: baseline, champion, runner-up, best learning-free
method, best non-retrieval method, a failing alternative, and the oracle.
Sorted by HC-MAE (best first within each group).

### 1-day ahead

Baseline = "yesterday's curve at the same time". A tight reference -
yesterday already carries most of the next-day signal.

| Model | HC-MAE | Δ vs lag-1 | Win % | Notes |
|---|---:|---:|---:|---|
| `vgrid_rolling_ridge_exog_1d` | **111.89** | **-10.93** | 57.2 % | Champion: v_grid retrieval + rolling Ridge + weather/proxy exog |
| `vgrid_rolling_ridge_1d` | 112.63 | -10.19 | 56.8 % | Same without exog - exog adds ~1 HC-MAE |
| `vgrid_strat_knn_1d` | 114.10 | -8.71 | 55.3 % | Static (no rolling refit) - rolling adds ~1.5 |
| `residual_on_baseline_ridge_1d` | 122.87 | +0.05 | 46.5 % | Best "near-baseline" tweaker - essentially a tie |
| `ema_baseline_1d` | 123.26 | +0.44 | 47.7 % | EMA over lags 1, 2, 7 |
| `same_weekday_avg_1d` | 140.60 | +17.78 | 41.4 % | Naive weekly average - loses (yesterday is better than last week) |
| `fof_snap_1d` | 196.26 | +73.44 | 29.0 % | Picks the wrong historical curve under HC-MAE |
| `lag1_baseline` | 122.81 | 0 | - | Reference |
| **Oracle (ceiling)** | **27.96** | **-94.85** | 90.3 % | Best possible bucket-restricted pick |

Lag-1 takeaway: yesterday's curve is hard to beat. The champion saves
~11 $/MWh, closes ~12 % of the available gap. Multi-lag features make
this *worse*, and combining with a weekly baseline makes it *much worse* -
because at lag-1 the most recent past dominates.

### 7-day ahead

Baseline = "the same interval, seven days ago". A weaker reference than
lag-1 - last week is informative but not fresh.

| Model | HC-MAE | Δ vs lag-7 | Win % | Notes |
|---|---:|---:|---:|---|
| `vgrid_rolling_ridge_exog_7d` | **123.41** | **-25.56** | 60.3 % | Champion: v_grid retrieval + rolling Ridge + exog |
| `vgrid_rolling_ridge_7d` | 127.57 | -21.40 | 58.6 % | Same without exog - exog adds ~2 HC-MAE |
| `vgrid_strat_knn_7d` | 129.41 | -19.56 | 57.6 % | Static (no rolling refit) |
| `same_weekday_avg_7d` | 140.60 | -8.37 | 51.8 % | Best learning-free method - useful weekly pattern |
| `residual_on_baseline_ridge_7d` | 148.51 | -0.46 | 53.8 % | Tweaks the lag-7 staircase - negligible gain |
| `ema_baseline_7d` | 147.69 | -1.28 | 54.1 % | EMA over lags 7, 8, 14 |
| `fof_snap_7d` | 242.76 | +93.80 | 26.0 % | Wrong geometry - catastrophic regression |
| `lag7_baseline` | 148.97 | 0 | - | Reference |
| **Oracle (ceiling)** | **27.96** | **-121.01** | 93.8 % | Best possible bucket-restricted pick |

Lag-7 takeaway: the sweet spot for both retrieval and weekly averaging.
The champion saves ~26 $/MWh, closes ~21 % of the gap. Exog gains ~2 HC-MAE
on top of the no-exog retrieval - bigger than at lag-1 because lag-7
features are staler, so external signals carry more new information.

### 30-day ahead

Baseline = "the same interval, thirty days ago". A weak reference -
month-old curves miss any recent regime drift.

| Model | HC-MAE | Δ vs lag-30 | Win % | Notes |
|---|---:|---:|---:|---|
| `vgrid_rolling_ridge_30d` | **132.62** | **-38.92** | 62.9 % | Champion: v_grid retrieval + rolling Ridge (no exog variant tested at this horizon) |
| `vgrid_strat_knn_30d` | 133.78 | -37.76 | 62.6 % | Static (no rolling refit) |
| `same_weekday_avg_30d` | 149.16 | -22.38 | 55.1 % | Best learning-free method - strong weekly seasonality at this horizon |
| `residual_on_baseline_ridge_30d` | 170.77 | -0.77 | 56.7 % | Tiny edit to the lag-30 staircase |
| `ema_baseline_30d` | 170.63 | -0.91 | 56.0 % | EMA over lags 30, 31, 37 |
| `interp_strat_knn_30d` | 174.29 | +2.75 | 48.1 % | Same experiment with the wrong embedding geometry - sign-flips under HC-MAE |
| `lag30_baseline` | 171.54 | 0 | - | Reference |
| `Oracle` | 27.96 | -143.58 | 95.5 % | Best possible bucket-restricted pick |

Lag-30 takeaway: the biggest absolute wins live here. Baseline is so weak
that even the naive weekly average buys ~22 $/MWh. The retrieval champion
saves ~39 $/MWh, closes ~27 % of the gap.
