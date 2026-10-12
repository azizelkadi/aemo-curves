# 03 — Current Status, Best Models, and Results

> **This is document 3 of 3.** It reports *where the project stands empirically*:
> the metric, the two comparable groups, full leaderboards, the champion model,
> what each family contributes, the oracle headroom, honest caveats, and next
> steps. For the *problem and literature* see
> [`01_problem_and_state_of_the_art.md`](01_problem_and_state_of_the_art.md);
> for *how the code and experiments work* see
> [`02_codebase_and_experiment_design.md`](02_codebase_and_experiment_design.md).
>
> All numbers below are the **auto-generated leaderboards** in
> `experiments/demand/7d/hc_mae/results.md` and
> `experiments/supply/7d/hc_mae/results.md` (produced by
> `scripts/run_group.py`, canonical split, n = 17,232 test auctions), verified
> against the code that produces them.

---

## 1. Headline (TL;DR)

- **Two active problems** are being solved right now: predict the **7-day-ahead
  demand curve** and the **7-day-ahead supply curve** of each 30-minute STEM
  auction, scored by **HC-MAE** ($/MWh, lower is better), each measured against
  its own **lag-7 baseline** on the same 17,232 test auctions.
- **Demand: solved in the sense of "beats the baseline."** The champion
  **`vgrid.rolling_ridge_exog`** reaches **HC-MAE 123.75**, i.e. **−25.21 $/MWh**
  vs the lag-7 baseline (148.97), winning on **60.0%** of auctions. It captures
  **~21%** of the baseline→oracle headroom.
- **Supply: an honest negative result.** **No deployable model beats lag-7**
  (54.83). Even the best-possible single translation of the lag curve
  (`oracle_shift`) only reaches 54.55 (−0.29), and the retrieval oracle ceiling is
  just 47.07 (−7.77). Supply offer stacks barely change week-to-week, so last
  week's real curve is extremely hard to improve on.
- **Best learning-free method:** `weekly.same_weekday_avg` (demand −8.37; supply
  +4.45 — loses).
- **Clearest methodological lesson:** the *metric-aligned* `v_grid` retrieval space
  is what makes demand beatable; methods that retrieve/snap in the wrong space
  (`interp`-based `retrieval.*`, `fda.*`) or that swap in a differently-shaped
  curve *lose badly* under HC-MAE.

---

## 2. Experimental setup

| Item | Value |
|------|-------|
| **What is predicted** | The 7-day-ahead step curve of each 30-min STEM auction (supply offer stack **and** demand bid stack), one model per side. |
| **Metric** | **HC-MAE** in $/MWh (lower is better) — mean absolute price gap between true and predicted curve over the volume axis (`n_grid = 10,000`). |
| **Baseline** | **lag-7**: the curve at the same interval seven days earlier (`lag1` with `lag_days=7`). Every model is measured against it on identical rows. |
| **Test window** | 2024-05-01 → 2025-04-24 (2025-05 absent), **n = 17,232** auctions, canonical split. |
| **Train** | 2021-01-01 → 2024-04-30, 58,368 auctions per side. |
| **Retrieval embedding** | `v_grid` (HC-MAE-aligned) for the retrieval methods; `interp` for `same_weekday_avg`, correctors, blends, FDA, and the exogenous proxy features. |
| **Oracle** | `vgrid.oracle` — closest **real training curve to the true curve** within the `(interval, day-of-week)` bucket. A non-deployable ceiling. |
| **Counts** | 23 experiments (demand), 22 (supply). The one extra is demand-only `probabilistic.ngboost_per_slice`. |

**How the delta and win-rate are computed** (verified in
`stages/evaluate/_common.py`): both the model and the baseline are scored on the
same test rows; `Δ = mean(model_hc_mae − baseline_hc_mae)` (negative is better);
`win% = fraction of rows where model < baseline`. Deltas/win-rates use only rows
where the baseline matched.

---

## 3. The metric in use: HC-MAE

HC-MAE (Heterogeneous-Curves MAE; Sinha & Lucheroni, Applied Energy 2025) compares
two step curves that may have **different step structure**: it builds the union
volume support of both curves, lays down 10,000 equispaced volume points, step-
interpolates both prices onto that grid (out-of-range volumes take the curve's
last-step price), and averages `|Δprice|`. The result is in **$/MWh** and is
sensitive to *where on the volume axis* a price step sits — exactly the property a
dispatcher clearing at a given quantity cares about. The repo's `v_grid` embedding
is built so that Euclidean nearest-neighbour ≈ HC-MAE nearest-neighbour, which is
why the winning retrieval methods use it. (Full definition and its contrast with
Wasserstein-1 and DTW: doc 1 §5, and the metric note `pad/prompts/hc_mae.md`.)

---

## 4. Demand, 7-day-ahead — full leaderboard

Baseline `lag7` = **148.97**. Oracle ceiling `vgrid.oracle` = **27.96**
(headroom = 121.01).

| Rank | Experiment | HC-MAE | Δ vs lag-7 | Win % | Note |
|-----:|------------|-------:|-----------:|------:|------|
| — | `vgrid.oracle` | **27.96** | −121.01 | 93.8 | ceiling (not deployable) |
| 1 | **`vgrid.rolling_ridge_exog`** | **123.75** | **−25.21** | 60.0 | **champion** |
| 2 | `vgrid.rolling_ridge` | 127.57 | −21.40 | 58.6 | rolling refit, no exog |
| 3 | `vgrid.strat_knn` | 129.41 | −19.56 | 57.6 | static (fit once) |
| 4 | `distance.distance_rank` | 137.87 | −11.10 | 52.8 | GBM distance ranker |
| 5 | `weekly.same_weekday_avg` | 140.60 | −8.37 | 51.8 | best learning-free |
| 6 | `corrector.shift_scale` | 142.03 | −6.94 | 56.8 | shift + qty scale |
| 7 | `blend.blended_lag` | 145.33 | −3.63 | 50.5 | learned lag-7/14 blend |
| — | `diagnostic.oracle_shift` | 145.48 | −3.49 | 43.6 | shift ceiling (uses truth) |
| 8 | `metric_learn.wasserstein_metric` | 145.80 | −3.17 | 49.5 | W-distillation retrieval |
| 9 | `blend.learned_blend` | 147.36 | −1.61 | 49.6 | simplex over 4 lags |
| 10 | `blend.ema_baseline` | 147.36 | −1.61 | 51.6 | fixed 0.6/0.2/0.2 |
| — | `baseline.lag7` | 148.97 | 0.00 | — | reference |
| 11 | `corrector.per_day_shift` | 150.04 | +1.07 | 33.3 | one shift/day |
| 12 | `corrector.shift` | 152.68 | +3.71 | 32.4 | one price shift |
| 13 | `diagnostic.realized_shift` | 155.45 | +6.48 | 9.7 | persist recent shift |
| 14 | `corrector.residual_gbm` | 157.81 | +8.85 | 39.3 | GBM residual |
| 15 | `corrector.residual_on_baseline` | 158.60 | +9.63 | 42.5 | ridge 64-d residual |
| 16 | `probabilistic.ngboost_per_slice` | 177.01 | +28.05 | 38.7 | per-slice, synthesized |
| 17 | `retrieval.soft_knn` | 190.18 | +41.21 | 36.1 | soft kNN on `interp` |
| 18 | `retrieval.gbm_retrieval` | 190.76 | +41.79 | 36.8 | GBM query on `interp` |
| 19 | `fda.logderiv_fpca` | 195.08 | +46.11 | 37.2 | FPCA log-deriv snap |
| 20 | `fda.fof_snap` | 243.89 | +94.93 | 25.4 | FPCA snap (worst) |

**Reading.** The `v_grid` retrieval family leads decisively (the top 3), and only
methods that either (a) retrieve/correct in a metric-appropriate way or (b) average
same-weekday curves manage to beat the baseline. Everything that swaps in a
differently-shaped curve via the *wrong* space — the `interp`-based `retrieval.*`
and the FPCA-**snap** methods — is heavily penalized (snapping to a whole different
staircase is poison under HC-MAE). Note the champion (−25.21) beats even the
**shift oracle** (−3.49), i.e. no amount of translating the lag-7 curve can match
retrieving a better real curve.

---

## 5. Supply, 7-day-ahead — full leaderboard

Baseline `lag7` = **54.83**. Oracle ceiling `vgrid.oracle` = **47.07**
(headroom = only 7.77).

| Rank | Experiment | HC-MAE | Δ vs lag-7 | Win % | Note |
|-----:|------------|-------:|-----------:|------:|------|
| — | `vgrid.oracle` | **47.07** | −7.77 | 65.4 | ceiling (only this beats lag-7) |
| — | `diagnostic.oracle_shift` | 54.55 | −0.29 | 10.8 | shift ceiling (uses truth) |
| — | `baseline.lag7` | **54.83** | 0.00 | — | reference — **unbeaten by any deployable model** |
| 1 | `diagnostic.realized_shift` | 55.95 | +1.11 | 2.2 | closest deployable |
| 2 | `corrector.per_day_shift` | 56.00 | +1.16 | 7.8 | |
| 3 | `blend.blended_lag` | 56.42 | +1.58 | 46.1 | |
| 4 | `blend.ema_baseline` | 56.54 | +1.70 | 46.3 | |
| 5 | `corrector.shift_scale` | 56.94 | +2.10 | 37.6 | |
| 6 | `corrector.shift` | 57.31 | +2.47 | 8.4 | |
| 7 | `weekly.same_weekday_avg` | 59.28 | +4.45 | 43.1 | |
| 8 | `blend.learned_blend` | 59.33 | +4.49 | 42.4 | |
| 9 | `corrector.residual_on_baseline` | 61.79 | +6.96 | 30.5 | |
| 10 | `vgrid.rolling_ridge_exog` | 68.34 | +13.50 | 40.7 | (demand champion; loses on supply) |
| 11 | `vgrid.rolling_ridge` | 70.79 | +15.95 | 37.9 | |
| 12 | `corrector.residual_gbm` | 72.32 | +17.49 | 20.7 | |
| 13 | `distance.distance_rank` | 77.86 | +23.03 | 34.2 | |
| 14 | `metric_learn.wasserstein_metric` | 79.30 | +24.47 | 36.5 | |
| 15 | `retrieval.soft_knn` | 82.23 | +27.40 | 30.6 | |
| 16 | `vgrid.strat_knn` | 84.72 | +29.88 | 27.8 | static retrieval (worst vgrid) |
| 17 | `fda.fof_snap` | 86.33 | +31.50 | 26.6 | |
| 18 | `retrieval.gbm_retrieval` | 89.93 | +35.10 | 27.6 | |
| 19 | `fda.logderiv_fpca` | 163.81 | +108.98 | 20.8 | worst |

**Reading.** On supply **nothing deployable beats lag-7**. The whole retrieval
family *loses* (retrieving a different real supply curve is worse than keeping last
week's), while the cheap correctors/blends cluster just above the baseline
(+1.1…+2.5). The oracle ceiling itself is only −7.77, so there is almost no
headroom to exploit — a genuinely publishable negative result about the
predictability limits of supply offer stacks. Note that on supply, **rolling refit
helps a lot within the vgrid family** (static `strat_knn` 84.72 → rolling
`rolling_ridge` 70.79 → `rolling_ridge_exog` 68.34), but even the best still loses
to lag-7.

---

## 6. The champion in detail — `vgrid.rolling_ridge_exog` (demand)

**Config** (`experiments/demand/7d/hc_mae/vgrid/rolling_ridge_exog.yaml`):
`data: {side: demand, features: v_grid, split: canonical, lag_days: 7}`;
`predict: {strategy: rolling_refit, params: {query: ridge, ridge_alpha: 200.0,
ridge_n_pca: 20}}`; `evaluate: {metric: hc_mae, baseline: lag1(lag_days=7)}`;
`params: {exog: {sources: [proxy, weather]}}`.

**Algorithm** (verified in `_orchestration.py` + `soft_knn.py`):

1. **Walk-forward monthly refit.** For each calendar month `T` of the test window,
   fit a fresh model on `historical[date ≤ T − 7 days]` — training data is always
   at least 7 days old, so it never enters the forecast gap — and predict that
   month's auctions.
2. **Per-model engine:** `StratifiedKNNRetrievalPredictor(k=1,
   bucket="interval_dow", output_mode="top1_curve", ridge_lags=(7,), lag_days=7,
   exog_provider=<proxy+weather>)`.
3. **Ridge query.** An inner per-interval `RidgeRetrievalPredictor`
   (`alpha=200`, `n_pca=20`, calendar + trend + exog features from the lag-7
   embedding) **predicts the target's v_grid embedding**.
4. **Retrieval.** Within the target's `(interval, day-of-week)` bucket, return the
   **real** curve of the k=1 nearest training row by L2 in v_grid space (≈ HC-MAE).

**Why it wins on demand.** Demand curves have real weekly + regime structure that a
ridge on lagged v_grid embeddings + market-state proxies can point at; retrieving a
*real* curve preserves the true staircase (unlike synthesis), and doing it in the
HC-MAE-aligned space means "nearest in L2" is "nearest in the scored metric."
Rolling refit adapts to drift; the exogenous proxy adds a few more $/MWh.

**Headroom captured.** Baseline 148.97 → champion 123.75 → oracle 27.96. The
champion closes `25.21 / 121.01 ≈ 20.8%` of the baseline-to-oracle gap — real, but
~79% of the ceiling is still on the table (the diagnosis in the design notes: the
*query predictor of the embedding*, not the retrieval library or the metric, is the
remaining bottleneck).

---

## 7. What each family contributes (both sides)

- **vgrid retrieval** — the only family that clearly beats the demand baseline;
  loses on supply. Rolling refit > static; exog > no-exog (demand).
- **weekly average** — strong, free demand method (−8.37); loses on supply.
- **distance ranking** — second-best demand challenger (−11.10) after vgrid.
- **correctors / blends** — small, safe adjustments to the lag curve. On demand,
  `shift_scale` and `blended_lag` modestly beat baseline; the residual correctors
  *hurt*. On supply they are the closest challengers but still lose.
- **metric learning (`wasserstein_metric`)** — modest on demand (−3.17); a
  faithful implementation of proposal #18, but capped near the bucket oracle and,
  as the notes record, high embedding-distance fidelity did not translate into a
  large realized win.
- **FDA snap (`fof_snap`, `logderiv_fpca`)** — worst performers on both sides;
  under HC-MAE, snapping to a different-shaped curve is heavily penalized.
- **probabilistic (`ngboost_per_slice`, demand only)** — loses as a point forecast
  (+28.05); its value is the *probabilistic* capability (CRPS/bands), not point
  HC-MAE. Demand-only because it hardcodes a monotone price decrease.
- **diagnostics** — `oracle_shift` and `realized_shift` bound and probe the
  shift-only correction; both confirm little shift headroom, especially on supply.

---

## 8. Oracle headroom analysis

| Side | Baseline | Champion (deployable) | Retrieval oracle | Shift oracle | Deployable Δ | Oracle Δ | Capture |
|------|---------:|----------------------:|-----------------:|-------------:|-------------:|---------:|--------:|
| Demand | 148.97 | 123.75 | 27.96 | 145.48 | −25.21 | −121.01 | ~21% |
| Supply | 54.83 | *(none beats)* | 47.07 | 54.55 | ~0 | −7.77 | ~0% |

The demand oracle gap is huge (the right curve is usually *in the library*; the
query predictor picks a sub-optimal neighbour) → representation/predictor
improvements have real room. The supply oracle gap is tiny → this is close to a
predictability limit, and the interesting scientific output is the *limit itself*.

---

## 9. Cross-cutting findings

1. **Demand is beatable, supply is (essentially) not** — the sharpest, most
   defensible result.
2. **Metric-aligned representation is decisive.** v_grid (HC-MAE-aligned) retrieval
   wins; interp (Wasserstein-aligned) retrieval loses under HC-MAE.
3. **Preserve the real staircase.** Every method that synthesizes or snaps to a
   differently-shaped curve underperforms the real-curve retrievers.
4. **Rolling refit and exogenous proxies each add a few $/MWh** on demand.
5. **The query predictor is the bottleneck**, not the library or the metric — ~79%
   of the demand oracle gap is unrecovered.

---

## 10. Caveats, limitations, and threats to validity

These are stated plainly so the results are not over-read.

- **The champion's "weather" exogenous is synthetic and mis-shifted.**
  `exogenous/builder.py` points the `weather` source at
  `data/external/weather_example.csv` (a **fabricated** file) and hardcodes
  `shift_days=1` — **not** the 7-day horizon. So the documented claim that weather
  is "shifted by the 7-day horizon so it never peeks inside the gap" is **inaccurate
  for the weather component**; the proxy source *is* correctly shifted by 7. The
  real ~3.8 $/MWh exog gain (123.75 vs 127.57) is therefore attributable to the
  **real market-state proxy**, not weather. A real weather file
  (`weather_au_capitals.csv`) exists but is not wired in. *Action:* wire real
  weather and fix the shift before claiming a weather effect.
- **No dedicated validation set.** Hyperparameters (α=200, n_pca=20, lags, K, …)
  were chosen on train and scored on test; there is no held-out validation slice.
  This is the top methodological gap flagged for publication defensibility (carve a
  ~6-month validation window and re-tune before final claims).
- **Test window is 359 days, not the full configured 13 months** (data ends
  2025-04-24; 2025-05 is missing). n = 17,232.
- **Cache does not track strategy code.** After editing a strategy, results must be
  recomputed with `--no-cache` (or a `CACHE_VERSION` bump) or the leaderboard can
  be stale.
- **Significance testing is not baked into the leaderboards.** The tooling exists
  (`analysis/analysis.py`: cluster bootstrap, Wilcoxon + Holm) but the auto
  `results.md` reports only mean Δ and win% — no CIs or p-values yet.
- **The supply `rolling_ridge_exog` YAML text says "Champion (demand)."** That is
  copied-through wording, not a supply claim; on supply the model loses (+13.50).
- **Some viz is Wasserstein-hardcoded** (`viz/comparison.py`), a minor
  inconsistency with the current HC-MAE groups.
- **Only two metrics of comparison are reported (HC-MAE + baseline).** No
  probabilistic scores (CRPS/coverage/Winkler) are on the leaderboard yet, so the
  probabilistic method is undersold by a point-only ranking.

---

## 11. Reproduction

Prerequisite: the real monthly CSVs under `data/`, then build the cached stages
(doc 2 §15.1). Then:

```bash
# Full ranked leaderboards (writes results.md into each group folder)
uv run python scripts/run_group.py demand.7d.hc_mae
uv run python scripts/run_group.py supply.7d.hc_mae

# The demand champion on its own
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge_exog

# Fast logic check (first 3 test days only — not a real evaluation)
uv run python scripts/run_experiment.py demand.7d.hc_mae.vgrid.rolling_ridge_exog --subset-days 3
```

`supply/7d` needs the supply v_grid tables first
(`scripts/build_v_grid_embeddings.py --side supply`). Results are cached under
`artifacts/cache/`; re-running a group recomputes only what changed. Use
`--no-cache` after changing strategy code.

---

## 12. Status summary and next steps

**Status.**

| Problem | State |
|---------|-------|
| Demand 7d / HC-MAE | **Beaten:** champion −25.2 vs lag-7, 60% win, ~21% oracle capture. Headroom remains (~79% of oracle gap). |
| Supply 7d / HC-MAE | **Negative result:** no deployable model beats lag-7; oracle ceiling only −7.77. |
| Metric | HC-MAE adopted (matches the closest published benchmark); Wasserstein/DTW also available. |
| Infrastructure | Stage/strategy/experiment pipeline, caching, MLflow, 34 predict strategies, group leaderboards — mature. |

**Highest-value next steps** (from the design notes; not yet implemented):

1. **Validation discipline** — carve a validation slice and re-tune; re-report the
   champion under proper discipline.
2. **Fix + wire real exogenous data** — use `weather_au_capitals.csv`, shift by the
   horizon, and quantify the true marginal value of each external input (the
   "regime-aware forecasting" paper, Direction 6).
3. **Close more of the demand oracle gap** — the query predictor is the bottleneck:
   box-relative embedding (Sinha-Lucheroni P1), stronger metric/representation
   learning, or foundation-model + RAFT retrieval.
4. **Add probabilistic outputs and scores** — CRPS/coverage/Winkler bands
   (conformal-wrapped retrieval, DDNN-JSU, top-K retrieval as a sample) as a
   separate deliverable that does not depend on beating the point baseline.
5. **Formal significance in the leaderboard** — surface the existing bootstrap +
   Wilcoxon/Holm machinery in `results.md`.
6. **Frame the supply negative result** as a "fundamental limits" study
   (Direction 1) with the oracle/predictability scaffolding already in place.
