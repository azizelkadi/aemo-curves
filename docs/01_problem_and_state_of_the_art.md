# 01 — Problem, Context, Data, and State of the Art

> **This is document 1 of 3.** It defines *what* the project is about: the market,
> the forecasting problem, the data, the evaluation metrics, the academic state of
> the art, and the space of approaches. For *how the code works and how to run and
> extend experiments*, see [`02_codebase_and_experiment_design.md`](02_codebase_and_experiment_design.md).
> For *where the project stands empirically — best models, numbers, findings*, see
> [`03_current_status_and_results.md`](03_current_status_and_results.md).
>
> Every factual claim here was checked against the code and data in this repository
> (as of commit `bb21da9`, 2026-07). Where a statement reflects an aspiration or a
> pruned/historical experiment rather than something currently implemented, it is
> labelled as such.

---

## 1. Executive summary

This is a PhD research codebase for **forecasting the supply (offer) and demand
(bid) curves of the Western Australian Short-Term Energy Market (STEM)**, one of
the trading mechanisms of the AEMO-operated Wholesale Electricity Market (WEM).

- **Object forecast.** For every 30-minute auction — identified by a
  `(Trading Date, Interval Number)` pair, 48 intervals per day — the market
  aggregates participants' price/quantity bids and offers into two monotone
  **step curves**: a rising supply (offer) stack and a falling demand (bid)
  stack. We predict the whole curve, not a scalar price.
- **Current concrete task.** Predict the **7-day-ahead** curve of each auction,
  for both the **supply** and **demand** sides, and beat the **lag-7 baseline**
  (the same interval's curve seven days earlier). Scored by **HC-MAE**
  (Heterogeneous-Curves Mean Absolute Error, $/MWh, lower is better).
- **Data.** ~4.3 years of raw monthly STEM bid/offer CSVs (2021-01 → 2025-04),
  turned into **58,368 train** and **17,232 test** auctions per side.
- **Headline result (see doc 3).** On **demand**, a v_grid retrieval model
  (`vgrid.rolling_ridge_exog`) beats lag-7 by **−25.2 $/MWh HC-MAE**
  (123.75 vs 148.97). On **supply**, the lag-7 baseline (54.83) is essentially
  unbeaten by any deployable method — the oracle ceiling is only 7.77 below it.
- **Why it is a good research problem.** The lag baseline is *very* strong, so
  the room to improve is small and precisely quantifiable (via oracle ceilings);
  the object is *functional* (a curve, not a number); and the dataset has rich,
  under-exploited structure (participant identity, weekly seasonality, weather
  and rooftop-PV sensitivity). This supports several distinct publishable angles
  — point-forecast improvement, probabilistic forecasting, representation/metric
  learning, generative modelling, decision-relevant evaluation, and a "fundamental
  limits" negative-result study.

---

## 2. The market and the object we forecast

### 2.1 AEMO / WEM / STEM / SWIS (context)

- **AEMO** — the Australian Energy Market Operator.
- **WEM** — the **Wholesale Electricity Market** of Western Australia, operating
  over the **SWIS** (South West Interconnected System), the isolated WA grid
  around Perth. The WEM is a *small, islanded* market compared with the eastern
  Australian **NEM** or European exchanges — fewer participants, high rooftop-PV
  penetration.
- **STEM** — the **Short-Term Energy Market**, the WEM's forward bilateral
  trading auction. Each STEM auction collects participants' price/quantity
  **bids** (willingness to buy) and **offers** (willingness to sell) and forms
  aggregate supply and demand curves whose intersection sets a clearing price and
  quantity.

> **Caveat on market micro-rules.** The project's notes cite AEMO price bounds of
> roughly **P_max ≈ A$17,500/MWh** (cap) and **P_min ≈ −A$1,000/MWh** (floor); these
> come from the literature notes, not from code in this repo, so treat them as
> background rather than verified constants of the pipeline.

### 2.2 The auction unit: `(Trading Date, Interval Number)`

Each trading day is divided into **48 half-hour intervals**. In the raw data,
**Interval Number 1 corresponds to 08:00**, so the WEM trading day runs 08:00
(interval 1) to 07:30 the next calendar day (interval 48). An *auction* — the
atomic unit this project predicts — is one `(Trading Date, Interval Number)`
pair. Most strategies fit **48 separate per-interval models** to exploit the
strong intra-day seasonality (evening peak vs overnight trough have very
different curves).

### 2.3 Bids and offers → supply and demand step curves

Raw rows carry a `Bid or Offer` field. Rows marked **`Offer`** form the
**supply** side; rows marked **`Bid`** form the **demand** side. For one auction:

- **Supply curve** — offers sorted by **ascending** price; cumulative quantity is
  the running sum. Accepting the cheapest MW first yields a **non-decreasing**
  (rising) staircase — the classic *merit order / bid stack*.
- **Demand curve** — bids sorted by **descending** price; the resulting staircase
  of price versus cumulative quantity is **non-increasing** (falling).

### 2.4 What a "curve" is, formally

A curve is a pair of equal-length arrays `(curve_qty, curve_price)` describing a
step function. Step `k` covers the cumulative-quantity interval
`(curve_qty[k-1], curve_qty[k]]` (with `curve_qty[-1] := 0` implicit) at price
`curve_price[k]`. `curve_qty` is monotone increasing (cumulative volume in MWh);
`curve_price` is monotone (rising for supply, falling for demand), in $/MWh. The
last element of `curve_qty` is the auction's **total cleared/offered volume**.

Because cells hold numpy arrays of *different lengths per auction*, the whole
pipeline stores tables as **pickle, not parquet** (a repo-wide convention).

---

## 3. The forecasting problem

### 3.1 Formal statement

Given the full bid/offer history up to some cutoff, predict, for each future
auction `(d, i)` (trading date `d`, interval `i`), the **entire step curve**
`(q̂, p̂)` of the chosen side (supply or demand), at a fixed **horizon** `h`
(days ahead). A prediction is scored against the realized curve `(q, p)` by a
curve-vs-curve distance metric, and compared to a baseline scored on the *same*
rows so deltas are apples-to-apples.

A key modelling stance in this repo: **retrieval strategies return a real
historical curve, never an interpolated artefact.** Preserving the true step
structure of some real recent auction is itself a strong prior (it is exactly why
the lag baseline is hard to beat).

### 3.2 Current concrete scope (what the repo actually runs today)

- **Horizon:** 7 days ahead (`h = 7`, "7d").
- **Sides:** supply *and* demand.
- **Metric:** HC-MAE.
- **Split:** canonical (train 2021-01…2024-04, test 2024-05…2025-05).
- **Baseline:** lag-7.

These are organized as **two comparable groups**:
`experiments/demand/7d/hc_mae/` and `experiments/supply/7d/hc_mae/`.

### 3.3 Historical / pruned scope (context for older numbers)

Earlier phases of the project studied **1-day** and **30-day** horizons under the
**Wasserstein-1** metric, plus a DTW/k-medoids prototype track and several
exogenous-feature and probabilistic tracks. Those experiment folders were
**pruned** (commit `75f5054`, "rename recipes to experiments; prune to supply_7d
and demand_7d (HC-MAE)"). Consequently:

- Wasserstein figures quoted in the design notes — e.g. **lag-1 ≈ 82.67 W** on
  demand 1d, **lag-30 ≈ 112.4 W** on demand 30d, oracle ≈ 25.78 W — belong to
  that **pruned era** and are *not* reproduced by the current two groups.
- Some registered strategies (`dtw_oracle`, `dtw_predicted`, `from_run`,
  `gate_blend`, `selector`) and cached artefacts (kmedoids tables, the
  `user2024` v_grid recut) are **remnants** that support those older tracks and
  are not used by any current experiment YAML.

The design/proposal documents under `pad/prompts/` (mirrored in this `docs/` set)
still describe that broader programme; read them as the research backlog, not the
current experiment set.

### 3.4 Why it is hard

The **lag baseline is extraordinarily strong**. A curve seven days ago at the
same interval preserves the exact real step structure and most of the weekly
seasonality. The burden of proof is entirely on any new method to beat it. The
project quantifies the ceiling with **oracles**:

- **Bucket / retrieval oracle** (`vgrid.oracle`): the closest *real training
  curve to the true curve* within the `(interval, day-of-week)` bucket — the best
  any bucket-restricted k=1 retriever could do. On demand it is HC-MAE 27.96
  (−121 vs baseline); on supply only 47.07 (−7.77 vs baseline).
- **Shift oracle** (`diagnostic.oracle_shift`): the best possible single
  price-translation of the lag-7 curve, using the truth.

The gap between "deployable model" and "oracle" *is the research headroom*. On
supply that gap is tiny — a clean, honest **negative result**.

### 3.5 Why it matters

The clearing price and quantity — and hence generator dispatch, retailer hedging
cost, and battery/DER scheduling — are determined by *where the supply and demand
curves cross*. A good curve forecast is therefore more decision-relevant than a
point price forecast. Academically, the WA STEM is a market that the closest
published precedent (Sinha & Lucheroni 2025, on the Italian IPEX) explicitly named
as *future work*, so results here are novel by market alone.

---

## 4. The data

### 4.1 Raw monthly CSVs

- **Location & shipping.** Raw monthly files live under `data/` and are
  **git-ignored** (large, possible licence terms). The repo ships exactly one
  example file at the root: `stem-bids-and-offers-SAMPLE.csv`.
- **File pattern.** `data/stem-bids-and-offers-YYYY-MM.csv`, one CSV per month.
- **Coverage present in this checkout.** `2021-01` through `2025-04` (52 monthly
  files).
- **Schema (8 columns):**
  `Trading Date`, `Interval Number`, `Trading Interval`, `Participant Code`,
  `Bid or Offer`, `Price ($/MWh)`, `Quantity (MWh)`, `Extracted At`.
  The pipeline reads the first seven; `Extracted At` is ignored.
- **Row = one bid or offer** by one participant at one price/quantity, for one
  auction. `Participant Code` (e.g. `ALCOA`, `ALINTA`) identifies the market
  participant — a **rare, under-exploited asset** (few public electricity
  datasets expose participant identity); several proposed research directions
  hinge on it.

**Cleaning (as implemented).** `Trading Date` → date; numeric coercion of
`Interval Number`/`Price`/`Quantity`; `Bid or Offer` lower-cased; rows with NaNs
in the key fields dropped; **rows with `Quantity ≤ 0` removed**. Curve building
then requires at least `MIN_POINTS_PER_AUCTION = 4` rows per auction and at least
2 resulting steps.

### 4.2 Data volume (verified from the cached tables)

| Split | Date range (as built) | Auctions per side |
|-------|-----------------------|-------------------|
| Train | 2021-01-01 → 2024-04-30 | **58,368** (1,216 days × 48) |
| Test  | 2024-05-01 → **2025-04-24** | **17,232** (359 days × 48) |

Note: the configured test end is `2025-05`, but the shipped data ends
**2025-04-24**, so the realized test window stops there (the loader warns about
the missing month). 17,232 is the exact `n` reported in the results tables.

### 4.3 Train / test split

Defined in `src/aemo_curves/config.py`:

- **Canonical** (the default, used by both current groups): train `2021-01…2024-04`,
  test `2024-05…2025-05`. The cached `train_*`/`test_*` pickles already encode it,
  so no recut is needed. **The test set is fixed and must never be used for
  training or model selection.**
- **`user2024`** (an alternative recut in `config.SPLITS`, used by pruned tracks):
  train ≤ 2023-12-31, test = calendar-year 2024. Selected per experiment via the
  `data.split` field.

> **Methodological gap flagged by the project's own notes:** there is currently
> **no dedicated validation set** — hyperparameters were chosen on train and scored
> on test. The Sinha & Lucheroni comparison notes call this out as a
> venue-defensibility issue; carving a validation slice is a recommended (not yet
> implemented) next step.

### 4.4 Derived representations (built once, cached under `artifacts/`)

For each auction the pipeline builds a step curve and a fixed-length **embedding**.
Three embeddings exist (all default to 64 dimensions for interchangeability):

| Embedding | Dim | What it encodes | Metric its L2 tracks |
|-----------|-----|-----------------|----------------------|
| **`interp`** (default) | 64 | Step price sampled on a **normalized quantile grid** of cumulative quantity (`linspace(0.02, 0.98, 64)`); divides out total volume. | Wasserstein-1 (a discretised quantile function). **Does *not* track HC-MAE** (it drops the absolute-volume axis). |
| **`v_grid`** | 64 | Step price sampled at K equispaced **absolute-volume** positions on `[0, V_max_global]`. | **HC-MAE by construction:** `‖z_a − z_b‖₂ ≈ √K · HC-MAE(a,b)`. `V_max_global` = 99th-percentile of *training* total volumes (1451.7 MW demand, 2586.9 MW supply). Used by the current retrieval champion. |
| **`kmedoids`** | K (default 20/24) | Vector of **DTW distances to K learned medoid curves** (PAM clustering on a DTW dissimilarity matrix). | DTW-space nearest-prototype. **Remnant of the pruned DTW track** — not used by current experiments. |

The choice of embedding is *metric-aligned*: retrieval under HC-MAE uses `v_grid`
so that Euclidean nearest-neighbour ≈ HC-MAE nearest-neighbour.

### 4.5 Exogenous data (optional)

Three families of exogenous features can be joined into predictors (module
`src/aemo_curves/exogenous/`):

- **Market-state proxies** (`MarketStateProxySource`, always available): ~13
  per-day aggregates derived from the cached embeddings themselves (daily
  price mean/median/std, log total quantity, embedding-shape moments, trailing
  7/28-day moving averages and trends, z-scores), plus 2 calendar features
  (sin/cos of day-of-year) and 5 **WA public-holiday flags** — **20 dims** with
  calendar on. Read at `date − lag_days` so they respect the horizon.
- **External CSVs** (`ExternalCsvSource`): daily- or interval-resolution files
  under `data/external/`. Two synthetic examples ship
  (`weather_example.csv`, `demand_forecast_example.csv`); a real multi-city
  weather file (`weather_au_capitals.csv`, fetched from Open-Meteo/ERA5 by
  `scripts/fetch_au_weather.py`) is also present but **not** the file the exog
  builder currently points at.
- **WA holidays** (`holidays.py`): hand-coded WA public holidays 2021–2026 and
  derived flags (is_holiday, day-before, day-after, is_weekend, bridge-day).

> **Important caveat (verified in `exogenous/builder.py`):** the exog *champion*
> `rolling_ridge_exog` requests `sources: [proxy, weather]`. The **proxy** source is
> real and shifted by the 7-day horizon (leakage-safe). The **weather** source
> currently loads the **synthetic** `weather_example.csv` and is hardcoded to
> `shift_days=1` (not the 7-day horizon). So, as configured, the weather
> contribution is placeholder data shifted by one day — see doc 3 §"Caveats".

Recommended real sources (documented in `data/external/README.md`): BoM Perth
climate, AEMO WEM operational demand forecast, gas/coal spot prices, RBA cash
rate.

---

## 5. Evaluation metrics

Three curve-vs-curve metrics are registered (`stages/evaluate/`), selected per
experiment via `evaluate.metric`. Every run is scored against a baseline on the
**exact same test rows**, so the reported delta is paired.

### 5.1 HC-MAE — Heterogeneous-Curves MAE (primary; both current groups)

Proposed by **Sinha & Lucheroni (Applied Energy 2025)** to compare two bid curves
with *different step structure* (different numbers of steps, different volume
break-points), where naive point-wise MAE is undefined.

**Procedure (as implemented in `stages/evaluate/hc_mae.py`):**
1. Union volume support: `v_min = min` of both curves' minimum volumes,
   `v_max = max` of both maxima.
2. `v_common` = `n_grid` equispaced points on `[v_min, v_max]`; **default
   `n_grid = 10,000`** (the paper's recommendation).
3. Step-interpolate both curves' prices onto `v_common` (right-side
   `searchsorted`; out-of-range volumes take the curve's last-step price — the
   "no more generation above this volume" convention).
4. `HC-MAE = mean(|Π_a − Π_b|)`, in **$/MWh**.

**Properties:** non-negative, symmetric, zero iff curves equal (up to grid
resolution). Crucially it **integrates over the volume axis with uniform
weighting**, so it is sensitive to *where on the volume axis* a price step sits —
the exact thing a trader dispatching at a given quantity cares about. It is a
*distance*, not a proper scoring rule.

### 5.2 Wasserstein-1 (`wasserstein`, the historical default)

1-D optimal-transport distance between the two curves' price distributions,
**quantity-weighted** (per-step volume increments), via
`scipy.stats.wasserstein_distance`. Integrates over the **price-quantile** axis.
It is *invariant* to reshuffles of (qty, price) pairs that preserve the price
marginal — a blind spot HC-MAE fixes (two curves with the same price histogram
but different volume placement clear at different prices, yet score W₁ = 0).

### 5.3 DTW (`dtw`)

Dynamic time warping between curves, treating each step as a 2-D point
`(price, normalized_qty × qty_weight)` (default `qty_weight = 50`,
`multivariate=True`), via the `dtaidistance` library. Elastic shape-alignment;
used by the pruned k-medoids prototype track.

### 5.4 Baselines and oracle ceilings

- **Lag-h baseline** (`lag1` with `lag_days=h`): the same interval's real curve
  `h` days earlier. Both current groups use **lag-7**.
- **Oracle ceilings** are diagnostics that use the truth and are *not deployable*:
  the **retrieval oracle** (`vgrid.oracle`, closest real training curve to the
  true curve in the bucket) and the **shift oracle** (`diagnostic.oracle_shift`,
  best single translation of the lag curve). They bound the achievable gain.

### 5.5 Significance / analysis tooling

`analysis/analysis.py` provides paired **cluster bootstrap** (cluster = Trading
Date, 2000 resamples, 95% CI on mean Δ), **Wilcoxon signed-rank** tests
(one-sided Δ < 0) with **Holm** multiple-comparison correction, per-month tables,
and oracle **capture-%**. (Note: Diebold–Mariano is *discussed* in the notes but
the implemented test is Wilcoxon, not DM.)

---

## 6. State of the art (academic literature)

The project sits at the intersection of **electricity bid/offer-curve
forecasting**, **functional data analysis**, and modern **ML time-series /
representation / generative** methods. The literature was surveyed by several
"expert-board" rounds; the full annotated bibliography lives in the project's
research notes at `pad/prompts/papers.md` (git-ignored scratch, present in this
checkout). The clusters:

### 6.1 The direct analogue — Sinha & Lucheroni 2025

*"Demand and supply curve forecasting using a monotonic autoencoder for
short-term day-ahead electricity market bid curves"* (Applied Energy 397:126262,
CC BY 4.0). Forecasts next-day IPEX (Italy, NORD) demand **and** supply bid
curves. Key ideas: **A/B/C curve decomposition** (left anchor, right anchor, and
the "interesting" interior segment) with a **box-relative 200-point grid** that
stretches to fit each day; a **monotonic autoencoder** for the interior; an
**Enhanced Echo-State Network** to forecast anchors and latents; a
**differential-evolution "integration"** step to blend multiple lags; and the
**HC-MAE metric** (which this repo adopts). Their classical benchmarks
(PCA+LASSO, PCA+LSTM, FAR, X-Model) were mostly **worse than naive** — mirroring
this project's "lag baseline is brutal" finding. This is the closest precedent in
the literature; the WA STEM is the market they named as future work.

### 6.2 Functional data analysis (FDA)

Curves as functions. **Li, Alonso & Pascual (IJEPES 2025)** — FPCA + functional
regression on OMIE curves with a "closest historical curve snap" (directly
inspiring this repo's `fof_snap`). **Aneiros, Vilar, Cao & Muñoz San Roque
(IEEE TPWRS 2013)** and **Vilar, Cao & Aneiros (IJEPES 2012)** — functional
AR(1) / kernel-FDA on OMIE residual-demand curves. **Ramsay (1998)** — the
monotone-smoother via integrating `exp(W(t))`, the trick behind guaranteed-
monotone functional forecasts (basis of `logderiv_fpca`). **Peláez et al. (2024)**
— bootstrap prediction regions. **Liebl (2013)** — the functional-time-series
viewpoint on EEX price-demand curves.

### 6.3 Probabilistic / distributional / conformal (the EPF tradition)

**Marcjasz, Narajewski, Weron & Ziel (2023)** — distributional NN with Johnson-SU
output for heavy-tailed prices. **Lipiecki, Uniejewski & Weron (2024)** — isotonic
distributional regression as cheap calibrated post-processing. **Uniejewski &
Weron (2021)** — regularized quantile-regression averaging (QRA). **Duan et al.
(2020)** — NGBoost (basis of `ngboost_per_slice`). **Mastropietro et al. (2024)**,
**Cassina et al. (2025)** — online conformal calibration for day-ahead prices.
**Gouttes et al. (2021)** / **Dabney et al. (2018)** — implicit quantile networks.

### 6.4 Deep time-series & foundation models

**PatchTST** (Nie et al. 2023), **iTransformer** (Liu et al. 2024), **N-HiTS**
(Challu et al. 2023), **TFT** (Lim et al. 2021); foundation models **Chronos**
(Ansari et al. 2024), **TimesFM**, **MOIRAI**, and benchmark studies showing
TSFMs competitive zero-shot on electricity load.

### 6.5 Retrieval-augmented forecasting

**RAFT** (2024) retrieves top-K similar historical windows in a foundation-model
embedding and forecasts with augmented context; **RATD** (NeurIPS 2024) is
retrieval-augmented *diffusion*; **TS-RAG** (2025) uses FAISS for fast retrieval.
This project already does *retrieval* but has not paired it with foundation-model
embeddings or diffusion.

### 6.6 Representation / metric learning

**TS2Vec** (Yue et al. 2022) and **SimCLR** (contrastive); **Khosla et al. (2020)**
(supervised contrastive); **Frogner et al. (2019)** *Learning Embeddings into
Entropic Wasserstein Spaces* — the direct motivation for training an encoder whose
L2 matches curve-Wasserstein (basis of `wasserstein_metric`); **Set Transformer**
/ **Deep Sets** for variable-length step curves; **DSSM** two-tower + cross-encoder
reranking.

### 6.7 Generative / diffusion / OT autoencoders

Conditional diffusion for load profiles (DiffPLF, EnergyDiff) and the general
diffusion-for-time-series lineage (TimeGrad, CSDI); **Wasserstein / Sinkhorn /
Sliced-Wasserstein autoencoders** for OT-respecting latent spaces.

### 6.8 Structural bid-stack & WA-specific

**Coulon & Howison (2009)** and **Carmona & Coulon (2014)** — structural
stochastic models of the bid stack linking fuel prices to electricity prices.
WA-specific: **Bell et al. (2021)** on the rooftop-PV "duck curve" reshaping the
SWIS load, and NEM rooftop-solar studies — motivating PV/weather-conditioned
demand models.

### 6.9 Pointer

The full one-line-per-paper bibliography with URLs is in `pad/prompts/papers.md`.
The 20-proposal and 4-experiment expert-board syntheses (below) are the applied
reading of this literature onto the STEM problem.

---

## 7. Approaches explored / catalogued in this project

### 7.1 Implemented method families (currently in the repo)

Each family is a folder of experiment YAMLs under both
`experiments/{demand,supply}/7d/hc_mae/`, and each maps to a registered predict
strategy. (Exact params, per-experiment numbers, and how to run them are in
docs 2 and 3.)

| Family | Idea | Literature link |
|--------|------|-----------------|
| **baseline** (`lag7`) | Curve at the same interval 7 days ago. The reference. | naive persistence |
| **weekly** (`same_weekday_avg`) | Learning-free mean of the last 4 same-weekday curves. | "smarter naive" |
| **vgrid** (`strat_knn`, `rolling_ridge`, `rolling_ridge_exog`, `oracle`) | Retrieval in the HC-MAE-aligned `v_grid` space: a ridge query predicts where next week lands, return the nearest *real* curve in the `(interval, dow)` bucket. Rolling monthly refit; optional exogenous features. **The demand champion + the retrieval oracle ceiling.** | RAFT-style retrieval; metric-aligned embedding |
| **fda** (`fof_snap`, `logderiv_fpca`) | Compress curves with PCA/FPCA, ridge-predict next week's scores, snap to nearest real curve. | Li-Alonso-Pascual; Ramsay |
| **metric_learn** (`wasserstein_metric`) | Train an encoder so L2 ≈ curve-Wasserstein, then retrieve in it. | Frogner et al.; supervised contrastive |
| **distance** (`distance_rank`) | A boosted tree predicts the target-to-candidate distance; retrieve the closest. | learning-to-rank retrieval |
| **probabilistic** (`ngboost_per_slice`, *demand only*) | One boosted model per volume slice, joined into a monotone curve. | NGBoost; per-τ distributional |
| **corrector** (`shift`, `shift_scale`, `residual_on_baseline`, `residual_gbm`, `per_day_shift`) | Start from the lag-7 curve and predict a small correction (price shift, shift+scale, or per-step residual). | residual/structural correction |
| **blend** (`ema_baseline`, `blended_lag`, `learned_blend`) | Convex combinations of several weekly lags. | forecast combination / QRA |
| **diagnostic** (`oracle_shift`, `realized_shift`) | Ceilings and simple deployable shift persistence. | oracle analysis |
| **retrieval** (`gbm_retrieval`, `soft_knn`) | Other retrieval variants (GBM query; soft top-K kNN) over the `interp` space. | kNN / soft retrieval |

### 7.2 The proposal backlog (aspirational, mostly not implemented)

The design notes (`pad/prompts/`) contain three large synthesis documents, kept in
this `docs/` set for context. They describe a much broader programme than the
current two groups:

- **`expert_board_four_experiments.md`** — an eight-expert board converging on four
  experiments: **F1** FPCA + VAR-X with exogenous; **F2** foundation model +
  RAFT retrieval augmentation; **F3** conditional diffusion with a Wasserstein-
  barycentre point forecast; **F4** a Wasserstein-barycentre ensemble.
- **`research_proposals_20.md`** / **`experiment_proposals.md`** — twenty
  citation-grounded proposals across five families (deep sequence models; FDA;
  probabilistic; structural/exogenous; representation/metric learning), with a
  moderated execution order. The top prior was **#18 supervised Wasserstein-
  distillation metric learning** (implemented as `wasserstein_metric`), then **#8
  FPCA-in-log-derivative-space** (`logderiv_fpca`), **#5 FoF-snap** (`fof_snap`),
  **#11 NGBoost-per-slice** (`ngboost_per_slice`), **#13 temperature-conditioned
  kNN** (not implemented).
- **`area_prediction_proposals.md`** — a four-expert board on shifting from a
  single-curve target to a **probabilistic band / area** (conformal-wrapped
  point forecasts, per-τ quantile bands, top-K retrieval as a sample, DDNN-JSU),
  scored by Winkler / coverage / CRPS. Not implemented.

These reference many experiment folders (`recompute_2024test/`, `horizon_1d_exog/`,
`proposals_top5/`, `dtw_kmedoids/`, `area_pred/`, …) that **no longer exist** in
the repo — they were pruned. Treat those documents as the research roadmap and
rationale, not a description of the current codebase.

---

## 8. Publishable research directions & thesis framing

From `research_directions.md`, ten publishable angles were identified; a PhD
thesis is framed as 3–5 papers. The suggested structure:

1. **Fundamental limits of supply-curve forecasting** — a rigorous oracle /
   predictability-decay study (turns the strong-baseline "negative result" into a
   reference contribution). The oracle-ceiling scaffolding already exists.
2. **Probabilistic curve forecasting with calibration guarantees** — conformal
   prediction for *functional* outputs; report coverage + CRPS, not just point
   distance.
3. **Participant-level generative modelling of bid behaviour** — exploiting the
   rare `Participant Code` asset.
4. **Decision-relevant evaluation + market-power detection** — a trader/dispatch
   simulator re-ranking models by realized P&L, plus strategic-bidding detection.

Plus a **community benchmark** paper (dataset + split + leaderboard + metrics) as
a low-effort companion now that the infrastructure exists. The **chosen first
track** was regime-aware forecasting with exogenous signals (Direction 6),
realized as the exogenous v_grid champion.

---

## 9. Key findings so far (see doc 3 for full numbers)

- **Demand is beatable, supply is (essentially) not.** The v_grid retrieval
  family beats lag-7 on demand by up to −25 $/MWh HC-MAE; on supply nothing
  deployable beats lag-7, and the oracle ceiling is only −7.77 — a clean negative
  result. Supply offer stacks change slowly week-to-week, so last week's curve is
  very hard to improve on.
- **Metric-aligned representation matters.** Retrieval works far better in the
  HC-MAE-aligned `v_grid` space than in the Wasserstein-aligned `interp` space
  (the `interp`-based `retrieval.*` and `fda.*` methods *lose* badly under HC-MAE).
- **Snapping to a whole different curve is dangerous under HC-MAE.** The FPCA-snap
  methods are the worst performers — replacing the real staircase with a
  differently-shaped one is penalized heavily on the volume axis.
- **Rolling refit and exogenous proxies each add a few $/MWh** on demand, on top
  of static retrieval — with the weather-exog caveat noted in §4.5.

---

## 10. Glossary

- **STEM** — Short-Term Energy Market (WA WEM forward auction).
- **Auction** — one `(Trading Date, Interval Number)`; 48 per day, interval 1 = 08:00.
- **Supply / offer curve** — rising step curve of offers (accept cheapest MW first).
- **Demand / bid curve** — falling step curve of bids.
- **HC-MAE** — Heterogeneous-Curves Mean Absolute Error, $/MWh, over the volume axis.
- **Embedding** — fixed-length vector for a curve (`interp` / `v_grid` / `kmedoids`).
- **Lag-h baseline** — the same interval's real curve `h` days earlier.
- **Oracle** — a non-deployable ceiling that uses the truth (retrieval or shift).
- **Bucket** — a `(interval, day-of-week)` or `(interval, weekend-flag)` stratum.
- **Retrieval** — returning a real historical curve (never a synthesized artefact).
- **Rolling refit** — re-fitting the query model at the start of each test month.
- **Exogenous / exog** — extra features (market-state proxy, weather, holidays).

## 11. Provenance

The problem framing, literature survey, and proposals in this document are
distilled from the project's own research notes, which live under `pad/prompts/`
(git-ignored scratch, present in this checkout but **not** part of the published
`docs/` set): `architecture.md`, `hc_mae.md`,
`demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md`,
`research_directions.md`, `research_proposals_20.md`, `experiment_proposals.md`,
`expert_board_four_experiments.md`, `area_prediction_proposals.md`, `papers.md`,
`results.md`. (These were previously tracked under `docs/`; commit `2f7916a`
"move docs" relocated them to `pad/prompts/`.) All code-level claims in this
document were re-verified directly against `src/aemo_curves/` and the cached
artefacts.
