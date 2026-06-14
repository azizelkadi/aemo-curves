# Results: 7-day-ahead curves under HC-MAE

The two experiment groups in this repo (`supply_7d`, `demand_7d`), scored on the
full test window. Reproduced end to end with the current code.

## Setup

- What we predict: the 7-day-ahead curve of each 30-minute STEM auction
  (supply offer stack or demand bid stack).
- Metric: HC-MAE in $/MWh, lower is better - the mean absolute price gap
  between the true and predicted curve over the volume axis. See
  [`hc_mae.md`](hc_mae.md).
- Baseline: lag-7, the curve at the same interval seven days earlier. Every
  model must beat this.
- Test window: 2024-05 to 2025-05, n = 17,232 auctions, canonical split.
- Embedding: `v_grid` for the retrieval methods; `interp` for `same_weekday_avg`
  and for the exogenous proxy features.
- Oracle: the HC-MAE of the best (interval, day-of-week) bucket-restricted pick.
  A ceiling, not deployable.

## Demand, 7 days ahead

Baseline lag-7 = 148.97.

| Model | HC-MAE | Δ vs lag-7 | Win % | Notes |
|---|---:|---:|---:|---|
| `vgrid_rolling_ridge_exog` | **123.75** | **-25.21** | 60.0 | champion: rolling ridge query + proxy and weather exog |
| `vgrid_rolling_ridge` | 127.57 | -21.40 | 58.6 | same, without exog |
| `vgrid_strat_knn` | 129.41 | -19.56 | 57.6 | static (fit once on train) |
| `same_weekday_avg` | 140.60 | -8.37 | 51.8 | best learning-free method |
| `baseline_lag7` | 148.97 | 0 | - | reference |
| `vgrid_oracle` | 27.96 | -121.01 | 93.8 | ceiling (not deployable) |

Reading: the v_grid retrieval family beats the lag-7 baseline. The champion
saves about 25 $/MWh and closes about 21% of the baseline-to-oracle gap.
Exogenous features (proxy plus weather, shifted by the 7-day horizon so they
never peek inside the forecast gap) add a few $/MWh on top of the no-exog
retrieval.

## Supply, 7 days ahead

Baseline lag-7 = 54.83.

| Model | HC-MAE | Δ vs lag-7 | Win % | Notes |
|---|---:|---:|---:|---|
| `vgrid_oracle` | 47.07 | -7.77 | 65.4 | ceiling (only this beats the baseline) |
| `baseline_lag7` | 54.83 | 0 | - | reference |
| `same_weekday_avg` | 59.28 | +4.45 | 43.1 | loses to the baseline |
| `vgrid_rolling_ridge_exog` | 68.34 | +13.50 | 40.7 | loses |
| `vgrid_rolling_ridge` | 70.79 | +15.95 | 37.9 | loses |
| `vgrid_strat_knn` | 84.72 | +29.88 | 27.8 | loses |

Reading: the supply baseline is far stronger than the demand baseline (54.83 vs
148.97), and the oracle ceiling sits only 7.77 below it, so there is very
little headroom. The retrieval methods do not beat lag-7 on supply - the v_grid
retrieval result is demand-specific. Supply offer stacks change slowly week to
week, so last week's curve is hard to improve on.

## Reproduce

```bash
uv run python scripts/run_experiment.py demand_7d.vgrid_rolling_ridge_exog
```

`supply_7d` needs the supply v_grid tables first:

```bash
uv run python scripts/build_v_grid_embeddings.py --side supply
```
