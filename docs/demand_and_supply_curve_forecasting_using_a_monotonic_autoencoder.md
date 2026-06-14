# Demand and supply curve forecasting using a monotonic autoencoder for short-term day-ahead electricity market bid curves

**Authors:** Nabangshu Sinha, Carlo Lucheroni (University of Camerino)
**Venue:** Applied Energy 397 (2025) 126262
**DOI:** 10.1016/j.apenergy.2025.126262
**Published:** Available online 18 June 2025
**License:** CC BY 4.0 (open access)
**Local copy:** [`1-s2.0-S0306261925009924-main (1).pdf`](1-s2.0-S0306261925009924-main%20(1).pdf)

---

## 1. Problem and dataset

Forecasts the next-day **demand** AND **supply** aggregated price/volume bid
curves for the IPEX (Italian Power Exchange) DAM, NORD zone, hour-by-hour
(24 separate hours per day, treated independently).

- Dataset: 730 days, 1 Jan 2018 → 31 Dec 2019.
- Split: 385 train (to 19 Jan 2019) / 165 validation (to 4 Jul 2019) / 180 test
  (to 31 Dec 2019). **Validation set is explicitly used for hyperparameter
  tuning and for fitting the integration-step parameters.**
- Why this window: avoids the August 2020 IPEX rule change (`P_min` shifted
  to -500 €) and the COVID/Ukraine shocks. Authors defer post-2020 data to
  follow-up work.

## 2. The framework

### 2.1 Curve decomposition (the core idea)

Each curve `P(V)` is split into three structurally meaningful pieces:

- **`A_t`** = leftmost interior anchor: `(V_A, P_A)` where
  `V_A = min{V | 0 < P_t(V) < P_max}` (so `A` is where the curve leaves the
  bottom flat zone).
- **`B_t`** = rightmost interior anchor:
  `V_B = max{V | 0 < P_t(V) < P_max}` (right end of the interior).
- **`C_t`** = the "interesting segment" - the curve between `A_t` and `B_t`
  inclusive. This is where almost all of the day-to-day variability lives.

The flat parts at `0` and `P_max` carry little forecasting signal; **the box
`[V_A, V_B] × [P_min, P_max]` slides back and forth in time but the shape
inside is much more stable than the absolute curve**.

### 2.2 Relative-coordinate reparametrisation of C_t

This is the key trick. To decouple "where the box sits" from "what the box
contains":

1. Define a **uniformly spaced grid of 200 points on V *between* `V_A` and
   `V_B`** (same number of points every day, but the *spacing* dilates and
   contracts to fill the day's box).
2. Step-interpolate `P` onto that grid (IPEX uses step interpolation by
   regulation).
3. Discard the V grid; keep just the 200-d price vector `Ĉ_t`.

After this, the `Ĉ_t` vectors share a common index space and have a much
smaller dynamic range than the original `C_t` (their Fig. 8 vs Fig. 4 makes
this dramatic). **This is structurally different from the standard quantile-
grid embedding because the grid stretches to fit the day, rather than being
fixed in V.**

### 2.3 Forecasting the anchors A_t and B_t

A custom **Enhanced Echo State Network (EESN)**:

- Standard ESN reservoir-computing forecaster: random fixed reservoir,
  trains only the linear output layer (very fast).
- Wraps the ESN output in a linear regression that injects exogenous dummy
  variables (weekend/holiday flags) to "peep legally" into tomorrow.
- Forecasts `V_A` and `V_B` separately; the corresponding `P_A`, `P_B` come
  out of the `Ĉ_t` reconstruction step.
- Hyperparameters: number of reservoir neurons + spectral radius, tuned on
  validation.

### 2.4 Forecasting C_t - monotonic autoencoder

1. Pre-train an autoencoder (encoder-latent-decoder) on the 200-d `Ĉ_t`
   vectors of the train+val period, with a **monotonicity-biasing loss term
   added** to push reconstructions towards monotonic curves.
2. Use the EESN to forecast each dimension of the latent code separately
   from yesterday's and last-week's latent.
3. Decode the forecasted latent → 200-d price vector. This is the
   "autoencoder forecast" `Ĉ_t^A`.
4. The decoded curve is **not** guaranteed monotonic, only biased. Cleanup:
   for each forecast, find the **5 nearest monotonic training segments**
   (NN over the latent space) and take a **distance-weighted average**.
   This yields the final `Ĉ_t^f`. Average of monotonic curves is monotonic
   ⇒ guaranteed monotonicity.

### 2.5 Assembling the curve

For demand: prepend `(V=0, P=P_max)`, then `(V_A, P_A)` from the forecast,
the 200-point relative grid filled with `Ĉ_t^f`, then `(V_B, P_B)`. Symmetric
construction for supply.

### 2.6 Integration step (differential evolution)

A,B,C were forecast independently. To couple them and inject `t-7`
information, fit 8 weights `θ_0..θ_7` such that:

```
A_combined = θ_0 + θ_1 · A_t-1^f + θ_2 · A_t-7
B_combined = θ_0 + θ_1 · B_t-1^f + θ_2 · B_t-7   (same θ for both)
C_combined = θ_3 + θ_4 · C_t-1^f + θ_5 · C_t-7
```

Loss = HC-MAE (see below); non-differentiable, so optimised with
**differential evolution** on the validation set. The fitted θ are then
applied at test.

### 2.7 The HC-MAE metric (their proposed curve-vs-curve metric)

Two heterogeneous curves `(V_1, P_1)` and `(V_2, P_2)` cannot be compared
point-wise. Their procedure:

1. `V_u = sort(V_1 ∪ V_2)`.
2. Define `V_common` = `U=10000` equispaced points spanning
   `[min(V_u), max(V_u)]`.
3. Step-interpolate `P_1, P_2` onto `V_common` → `Π_1, Π_2`.
4. `HC-MAE = (1/U) · Σ |Π_1i - Π_2i|`.

Properties: non-negativity, identity, symmetry. Compares the curves over the
**whole V support of the union**, not just the interesting segment. Different
from Wasserstein because it integrates `|ΔP|` over **volume**, not over
**price quantile**.

## 3. Benchmarks and results

### 3.1 Benchmarks

- **Naive**: `X_t = X_{t-1}` (lag-1 curve).
- **Smarter naive**: `X_{t-7}` if today ∈ {Sat, Sun, Mon}, else `X_{t-1}`.
- **PCA + LASSO** (Pelagatti 2012, ref [13]).
- **PCA + LSTM** (Guo et al. 2021, ref [7]).
- **FAR** (Shah & Lisi 2020, ref [11]) - stochastic-functional autoreg.
- **X-Model** (Ziel & Steinert 2016, ref [15]).

### 3.2 Headline numbers

HC-MAE averaged over the 180 test days, per hour:

**Demand (Table 2)** - proposed model wins every hour. Examples:

| Hour | Naive | Smarter naive | PCA+LSTM | FAR | X-Model | Proposed |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 227.34 | 87.15 | 274.22 | 261.74 | 152.63 | **75.91** |
| 12 | 211.81 | 89.55 | 281.61 | 279.66 | 149.35 | **79.52** |
| 18 | 205.38 | 89.12 | 283.47 | 287.29 | 147.44 | **85.53** |
| 24 | 124.18 | 76.08 | 191.38 | 194.49 | 124.07 | **69.76** |

**Supply (Table 3)** - proposed model wins every hour. Examples:

| Hour | Naive | Smarter naive | PCA+LSTM | FAR | X-Model | Proposed |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 80.88 | 69.20 | 231.05 | 203.40 | 306.98 | **65.94** |
| 17 | 71.99 | 63.75 | 261.06 | 208.70 | 310.31 | **57.88** |
| 23 | 53.55 | 66.88 | 279.78 | 205.27 | 326.13 | **50.90** |

Important observations:

- **PCA+LASSO, PCA+LSTM, FAR, X-Model are all *worse than naive* on most
  hours.** This mirrors our finding that lag-1 is a brutal baseline.
- **The smarter-naive baseline is the closest competitor** on demand
  daylight hours, sometimes beating the proposed model's V_A point forecast
  in MAE (Tables 4-5) - but losing on the full-curve HC-MAE.
- Improvement of proposed over smarter-naive is roughly 5-15 €/MWh HC-MAE on
  demand, ~5-10 on supply. Big absolute gains on the worst hours of the
  naive (e.g. demand hour 9: 227 → 76).

### 3.3 What the authors highlight

1. The decomposition + box-relative grid is what does the heavy lifting.
2. Monotonic AE compresses better than PCA on these curves because the
   variability is non-linear.
3. Forecasting `V_A`/`V_B` well is necessary but not sufficient: smarter
   naive sometimes has lower V_A MAE yet loses on the full-curve metric,
   meaning the curve shape inside the box still matters.
4. Mixing `P(V)` and `V(P)` perspectives (forecast V at the anchors,
   forecast P inside the box) gives both sides of the literature.

## 4. Comparison to the related literature (their Section 3)

They taxonomise prior work into two families:

- **`P` as support** (forecast V over uniform price bins): Pelagatti 2012;
  Ziel et al. 2016 (X-Model) + extensions; Mestre et al. 2020 (sigmoid
  basis); Yildirim 2022 (MCMC).
- **`V` as support** (forecast P over uniform volume bins): Shah & Lisi 2020
  (FAR); Ciarreta 2023 (piecewise functional, OMIE-specific); Guo 2021
  (PCA+LSTM); Tang 2024 (PCA/NMF + Transformer).

Most prior work targets supply only. Only Shah & Lisi, Ziel, and Ciarreta
forecast both demand and supply. Only Mestre and Ciarreta enforce
monotonicity.

## 5. Where this paper meets our work

Our project ([../CLAUDE.md](../CLAUDE.md)) is the AEMO STEM (Australia/WA)
analogue of theirs - same problem, larger dataset (~4 years, 17 232 test
auctions), Wasserstein metric, lag-1 / lag-30 baselines, retrieval-based
models. Their paper is the closest direct precedent in the literature.

### 5.1 Methodological gaps the paper exposes in our setup

1. **No validation set.** Currently train (40 mo) + test (13 mo). All
   hyperparameter choices (Ridge α, PCA dim, lag combinations, K) are made
   on train, with the test set used for scoring. No held-out validation set
   = not strictly defensible for *Applied Energy*. **They use a 23 %
   validation slice explicitly for HP tuning + integration parameters.**
2. **No "smarter naive" baseline.** Lag-7-when-weekend is a one-line
   addition and is the strongest classical baseline in the literature for
   demand curves. We currently compare to lag-1 and lag-30 only.
3. **Single metric (Wasserstein).** Reviewers in this venue will expect at
   least one alternative curve-vs-curve metric. HC-MAE is simple to add and
   is the metric used in the most recent published baseline.
4. **No formal monotonicity audit** of our prediction outputs. Retrieval
   methods preserve monotonicity by construction, but ridge-on-quantile-
   embedding methods (`RidgeRetrievalPredictor`'s decoded output, the
   residual correctors) can in principle produce non-monotonic forecasts.
   We have no diagnostic for how often that happens.

### 5.2 Genuinely new ideas the paper offers us

1. **The `[V_A, V_B]`-relative grid for embedding `C_t`.** Our 64-d interp
   embedding ([`embeddings/interp.py`](../src/aemo_curves/embeddings/interp.py))
   uses a fixed quantile grid spanning the full V range. This conflates two
   timescales: where the bid mass sits (fast) and what its shape is when
   stretched to fit (slow). Their decomposition decouples them. **Strong
   candidate for tightening retrieval and closing more of the demand-30d
   oracle-KNN gap (currently -86.80 W oracle, -11.16 W realised, capture
   12.9 %).**
2. **HC-MAE as a complementary metric.** Integrates `|ΔP|` over V; ours
   integrates over the price-quantile axis. Different blind spots.
3. **5-NN weighted average for monotonicity guarantee.** Equivalent in
   spirit to our `SoftKNNRetrievalPredictor` with k=5 - the paper validates
   that this is a publishable approach in its own right.
4. **EESN with exogenous dummies.** ESN is fast, easy to write, and may be
   a useful per-interval `V_A`/`V_B` regressor.
5. **Differential-evolution integration of A/B/C from multiple lags.** A
   "blender" trained on validation under a non-differentiable curve metric.
   This is what our `LearnedBlendPredictor` could become if generalised.

### 5.3 Ideas in their paper that are NOT useful for us

- **EESN is not necessarily better than per-interval Ridge** for our scale
  - Ridge already runs in seconds, and the EESN's main selling point is
  speed vs LSTM. Net change in headline metric is unlikely.
- **Differential-evolution training of θ** is specific to their A/B/C
  decomposition; for retrieval-based forecasts the assembly is trivial.
- **Their PCA+LSTM failure** is on a 730-day dataset; with our 4-year
  history a deep model might fare differently. So their negative result
  doesn't transfer cleanly.

## 6. Concrete proposals for our project

Ranked by expected publishable yield divided by code cost.

### Proposal P1 - A/B/C-decomposition embedding *(highest-yield, most novel)*

Build `BoxRelativeEmbedding`:

- Locate `V_A = min{V | 0 < P_t(V) < P_max}` and
  `V_B = max{V | 0 < P_t(V) < P_max}` for each `(date, interval, side)`.
  Note: AEMO has `P_max ≈ 17 500 AUD/MWh` (cap) and `P_min ≈ -1000 AUD/MWh`
  (floor). For supply use `0 < P < P_max`; for demand mirror it.
- Step-interpolate the supply/demand price onto a 200-point uniform grid
  on `[V_A, V_B]` → 200-d vector `Ĉ_t` (vs our current 64-d on
  fixed-quantile grid).
- Add the 2-d `(V_A, V_B)` as auxiliary features.

Score this embedding under the existing pipeline:

- KNN retrieval (full library) - does the **oracle** drop further, signalling
  a genuinely better embedding?
- Stratified KNN by `(interval, dow)` - best realised gap-capture so far.
- Ridge query → KNN retrieval - our current best on demand 30d.

Outcome interpretation:

- If oracle improves AND realised improves → embedding wins, publishable.
- If oracle worsens but realised improves → the box-relative grid is a
  better *retrieval space* for noisy queries (smaller bucket).
- If oracle worsens AND realised worsens → the absolute V position is more
  informative than the box shape; this is itself an interesting finding.

Cost: ~2-3 hours to add the new embedding builder + a `scripts/build_box_embeddings.py`,
~30 min to wire into the existing experiments. New folder
`experiments/box_relative/` mirrors `retrieval_demand_30d/`.

Stretch: combine with the residual-on-baseline approach by predicting
`(V_A, V_B, Ĉ_t)` separately and re-assembling.

### Proposal P2 - HC-MAE as a complementary metric *(small, free upside)*

Add [`metrics/hc_mae.py`](../src/aemo_curves/metrics/hc_mae.py): the
~30-line implementation from §4.12 of the paper. Score every existing
experiment under HC-MAE and add the column to all comparison tables. If our
ranking under HC-MAE differs from Wasserstein, that itself is publishable
content (different metrics → different model preferences).

Cost: ~30 min implementation + ~20 min reruns (results are cached, only the
metric column needs adding).

### Proposal P3 - Smarter-naive baseline *(small, fairness)*

Add `SmarterNaivePredictor` to `models/`:

```python
def predict(...):
    if is_weekend_or_monday(date):
        return curve_at_(date - 7, interval)
    else:
        return curve_at_(date - 1, interval)
```

Add it to all tables alongside `Lag1Predictor` and `Lag30Predictor`. This is
the *strongest* baseline reported in the most recent literature, so we
*must* beat it (or at least report against it) for the demand experiments.

Cost: ~20 min.

### Proposal P4 - Validation set discipline *(methodological)*

Carve the last 6 months of train (Nov 2023 → Apr 2024) as a validation
slice. Re-tune α, lag selection, K on validation only; only then run on
test. Re-run the headline winner (`strat_knn k=1 q=ridge lags=(30,) α=200
b=int_dow`). If the test number drops noticeably under proper validation
discipline, we need to pick a different headline configuration; if it
holds, we have a defensible result.

Cost: ~1-2 hours of re-tuning + reruns.

### Proposal P5 - Monotonicity audit

Single diagnostic experiment: for every existing model in MLflow, count
fraction of test predictions that are non-monotonic, and the mean magnitude
of the violations. Check whether retrieval methods (KNN, soft-KNN, strat-KNN)
indeed have 0 % violations as expected. The result is one paragraph + one
table; possibly motivates adopting their 5-NN cleanup for any model that
fails the audit.

Cost: ~30 min.

## 7. Recommended execution order

1. **P3** (smarter naive) + **P2** (HC-MAE) - 1 hour total, both must-haves
   for venue defensibility.
2. **P1** (A/B/C embedding) - main scientific contribution attempt.
3. **P5** (monotonicity audit) - quick add-on while P1 runs.
4. **P4** (validation discipline) - once a headline configuration is locked.

If P1 produces an improvement on either the oracle-KNN ceiling or the
realised metric on demand 30d (current `retrieval_demand_30d/` folder), we
have a publishable contribution that directly extends Sinha & Lucheroni
2025 to a new market (AEMO STEM) AND a new methodology (decomposition +
retrieval, vs. their decomposition + autoencoder + EESN).

## 8. Open questions for follow-up

- Does AEMO have well-defined `A_t` and `B_t` analogues? AEMO supply curves
  span quite differently from IPEX; need to inspect a few examples first to
  confirm the "interesting segment" framing applies.
- Their `P_max = 3000 €/MWh` cap is much lower than AEMO's
  `P_max = 17 500 AUD/MWh` cap; the right-flat zone might be much sparser
  in our data → less of `B_t` to find.
- Demand-side `A_t` interpretation: in IPEX it's the kink where
  uninterruptible demand ends. In AEMO's STEM (a small WA-only market),
  whether this kink is as visible needs checking.

## 9. Bibliographic context - papers worth pulling

The references in their Section 3 are the canonical curve-forecasting
literature. Of these, the ones we have NOT already explored:

- **Pelagatti 2012** [13] - first PCA-based supply curve forecasting; the
  ancestor of all PCA approaches.
- **Mestre et al. 2020** [19] - sigmoid-basis monotonic curve fit on IPEX.
- **Yildirim et al. 2022** [10] - MCMC modelling and forecasting.
- **Tang et al. 2024** [14] - PCA/NMF/dictionary + Transformer.
- **Ciarreta 2023** [12] - piecewise functional, both demand and supply,
  inherently monotonic.
- **Li et al.** [21] - distance-metric clustering of supply curves.

Of these, **Ciarreta 2023** and **Tang 2024** are the most likely to have
ideas we haven't borrowed yet (piecewise functional + Transformer
respectively).
