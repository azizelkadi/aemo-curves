# Expert board: four experiments to beat lag-1 / lag-30 on AEMO demand curves

Status: this plan was carried out, see the experiment folders under [`experiments/`](../experiments/).

**Brief.** Beat the lag-1 (1-day-ahead) and lag-30 (30-day-ahead) baselines on
the AEMO STEM **demand-side** Wasserstein metric, by any path that works.
Take the literature seriously. Convene experts from related fields, debate
exhaustively, then commit to four experiments worth running.

**Hard numbers as of 2026-05.** On demand 1d, lag-1 baseline = 82.67 W and the
KNN-with-true-embedding oracle = 25.78 W (Δ = -56.89, win 85%). On demand 30d,
lag-30 baseline = 112.58 W and the same oracle = 25.78 W (Δ = -86.80, win 94%).
The current best **realised** model captures only 12.9 % of the 30d oracle gap
(Δ = -11.16, win 53 %). The retrieval *library* is fine; the *predictor of the
embedding* is the bottleneck.

---

## Part 1 - Literature scan (what 2024-2025 work tells us)

A focused sweep of recent papers, organised by lever.

### 1.1 Curve-specific architectures

- **Sinha & Lucheroni 2025**, *Applied Energy* 397 - A/B/C decomposition +
  monotonic autoencoder + EESN (echo-state) + differential-evolution
  integration. Beats naive, smarter-naive, FAR, X-Model, PCA+LASSO,
  PCA+LSTM on IPEX. Already digested in
  [`demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md`](demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md).
- **Tang et al. 2024**, *Applied Energy* 363 - low-rank approximation (LRA)
  encoding of bid curves + transformer time-series prediction. Tested on
  Australian NEM individual bids. Architecture: LRA dimensionality
  reduction (instead of PCA / autoencoder) → transformer for the LRA
  features. Reports best results vs PCA, NMF, downsampling, dictionary
  learning baselines.
- **FPCA Merit-Order Curves 2025** (arXiv 2512.17758) - Functional Principal
  Component Analysis (Karhunen-Loève expansion) on `Q(p)` curves on IPEX,
  scores predicted by VAR/LASSO with exogenous load + wind + solar. **5 %
  MAE over the best price-based model, 10 % at peak hours** - and crucially
  beats the discretised X-Model (Ziel-Steinert) in its own market.
- **YUI 2024** (arXiv 2405.14893) - *time-invariance of supply curves under
  load-rate normalisation*. Supply curve as `(load_rate, price)` (load_rate
  = supply / available capacity) is approximately time-invariant for short
  windows. Then prices are **derived** from forecast load × forecast
  capacity → curve intersection. **15 % MAE over second best on Shanxi and
  ISO-NE.** Different framing: forecast the *normalised* curve, not the
  absolute one.
- **Latent Space Curves 2025** (arXiv 2503.11294) - UMAP + isotonic
  regression beats PCA, kPCA, autoencoder for MIBEL curves. Headline:
  **UMAP latents preserve monotonicity naturally** and isotonic regression
  on the reconstruction is the cheapest way to enforce shape constraints.

### 1.2 Foundation models

- **Chronos / Chronos-Bolt** (Amazon) - pretrained probabilistic time-series
  foundation model. Tokenises numeric series into a vocabulary, models
  autoregressively. Zero-shot already strong; fine-tunable on small data.
- **TimesFM 2.0** (Google) - patched-decoder foundation model. Beats Chronos
  on several recent benchmarks; in-context adaptation (TimesFM-ICF) gives
  fine-tune-level accuracy without gradient updates.
- **MOIRAI** (Salesforce), **Time-MoE**, **Sundial** - all 2024-2025.
- **PriceFM 2025** (arXiv 2508.04875) - *spatio-temporal* foundation model
  with graph inductive biases for cross-region electricity prices.
  Multi-quantile probabilistic.
- **Benchmark for household STLF** (arXiv 2410.09487) - TSFMs match or beat
  trained-from-scratch transformers zero-shot on electricity load
  forecasting.

### 1.3 Retrieval-augmented forecasting (RAF / RATD / TS-RAG)

- **RAFT 2024** (arXiv 2411.08249) - retrieve top-K most similar historical
  windows by L2 in foundation-model embedding space, concatenate as
  augmented context, then forecast with the foundation model. **86 %
  win-rate over baselines for multivariate, 80 % for univariate.**
- **RATD 2024** (NeurIPS) - retrieval-augmented *diffusion* models. Diffuse
  with the retrieved future segment as a conditional reference. **Specifically
  designed for complex / non-stationary series.**
- **TS-RAG 2025** (arXiv 2503.07649) - RAG-style integration with foundation
  models, FAISS for fast retrieval, runs in ~9 ms vs RAF's 3290 ms.

### 1.4 Probabilistic / generative

- **Distributional NN** (Marcjasz et al.) - output the parameters of a
  Johnson SU distribution per time step. **+7 % CRPS, +8 % per-trade
  profit** over second-best on EPF.
- **Conditional diffusion for load profiles** - DiffPLF, EnergyDiff,
  Customized Load Profile Diffusion. The same machinery applies to bid
  curves treated as 1-D signals.
- **Normalizing flows for energy forecasting** - Dumas & Wehenkel: scenario
  generation with full multivariate likelihood. Conditional on lag features
  + weather.
- **Isotonic Distributional Regression** (arXiv 2404.02270) - postprocesses
  any point forecast into a calibrated distribution while enforcing
  monotonicity. Cheap, theoretically grounded, plugs in everywhere.

### 1.5 Constrained neural architectures

- **Deep Lattice Networks** (Google, 2024 AAAI extension) - multi-horizon
  monotonic CDFs without quantile crossing. Lattice parameters constrained
  by sparse linear inequalities.
- **Non-crossing Quantile Regression NN** - simpler alternative; triangular
  positive-weight matrix enforces monotonicity in the quantile dimension.

### 1.6 What this collectively says

Six convergent signals:

1. **Functional decomposition wins.** FPCA, monotonic AE, LRA, UMAP - all
   beat ad-hoc discretisation. The 2025 papers consistently show
   functional-basis representation as the best feature space for prediction.
2. **Monotonicity must be enforced explicitly** (post-hoc isotonic, monotonic
   AE, lattice) or you pay 5-25 € MAE penalty in reconstruction.
3. **Foundation models are now competitive zero-shot** with bespoke
   architectures trained on millions of rows. We can plug in **today**.
4. **Retrieval-augmented forecasting** is a 2024-2025 breakthrough. We
   already do retrieval; we have *not* done the foundation-model + RAF
   pairing. Win-rates of 80-86 % over baselines.
5. **Probabilistic / generative outputs are the venue-expected default**
   in *Applied Energy*, *Energy Economics*, *Energy* in 2025. Scoring under
   CRPS or pinball loss often *flips the ranking* relative to point MAE.
6. **Exogenous covariates matter more than people expect.** YUI gets 15 %
   on top of strong baselines just by integrating forecast capacity and
   weather. The FPCA paper similarly: load + wind + solar exogenous gives
   ~10 % peak-hour gain.

---

## Part 2 - The board discussion

Eight experts. Each speaks once. Then they argue.

### Member 1 - Prof. R. Hyndman (channelled), Functional Data Analysis

> "You have a sequence of curves indexed by time. The literature has been
> screaming the answer at you since 1996: **Functional Principal Component
> Analysis**. Decompose each curve `Q_t(p) = Q̄(p) + Σ β_{tk} ξ_k(p)` where
> ξ are the eigenfunctions of the cross-day covariance kernel. Then forecast
> the K scores `β_{tk}` (typically K = 3-8 captures > 95 % of variance) by
> any vector autoregression you like. Reconstruct the curve, isotonise, done.
> The 2025 IPEX paper (arXiv 2512.17758) just did this and got 5 % MAE over
> X-Model - and they did it with vanilla LASSO-VAR. With Ridge in your stack
> you've left this lever entirely untouched. K=3 scores is **way easier to
> predict** than your 64-d quantile embedding, and the eigenfunctions
> themselves are interpretable: `ξ_1` ≈ overall price level, `ξ_2` ≈ steepness
> of the staircase, `ξ_3` ≈ left-vs-right asymmetry. Predict three numbers
> well, you win."

### Member 2 - Dr. M. Karahan (channelled), Time-Series Foundation Models

> "Stop training your own predictor from scratch on 4 years of data. That's
> 1 460 days × 48 intervals = 70 000 samples. Chronos-Bolt was pretrained on
> ≈ 100 billion observations. The TSFM benchmarks released in late 2024
> show foundation models matching or beating bespoke transformer pipelines
> **zero-shot** for electricity load. Your 64-d embedding is just 64
> independent (or coupled) scalar series - one per quantile bin. Throw each
> one at TimesFM 2.0, get a probabilistic forecast, retrieve the curve. If
> zero-shot is not enough, fine-tune with LoRA - maybe 10 minutes on a
> single A100. Honestly this is what I'd do *first*. Cost is days, not
> weeks. Even RAFT-style retrieval augmentation with the foundation model
> embedding as the retrieval key would beat your current Ridge pipeline."

### Member 3 - Dr. Y. Le, Generative / Diffusion

> "Wasserstein is a *distributional* metric, but you're using *point*
> predictors. Mismatch. Train a conditional diffusion model on the demand
> curve directly: input = (lag-30 curve, calendar, exogenous weather/load
> forecast), output = today's full curve. Sample 100 trajectories, compute
> the Wasserstein barycentre as the point forecast. The point forecast is
> only one summary; you also get calibrated coverage intervals. DiffPLF
> already proved this for EV charging load with cross-attention conditioning.
> EnergyDiff did it universal. The barycentre under W is **lower-mean-W**
> than any individual sample by Jensen's inequality applied to a convex
> functional. There's a free meal here. You can also condition on the
> retrieval-top-K candidates as a 'reference set' (RATD 2024 did this for
> generic time series; **nobody has done it for bid curves**)."

### Member 4 - Prof. C. Weron (channelled), EPF specialist

> "Be careful what you believe. Six papers in five years claiming to beat
> X-Model, three of them never reproducible. The two things that
> *consistently* matter in EPF: (a) get the *exogenous information* right -
> load forecast, wind forecast, fuel cost, weekday and holiday flags - and
> (b) ensemble. The single best-performing approach in EPF benchmarks is
> still QRA / iQRA on top of an ensemble of 5-10 distinct point forecasters.
> If you must pick *one* model, monotonic AE is fine. If you want the **best
> realised number**, ensemble three things - your strat-KNN, a residual-on-
> baseline ridge with full exogenous features, and an FPCA+VAR - and average
> their *embeddings* (or take a Wasserstein barycentre of their predictions).
> The Cuaresma 2024 ensemble paper showed gains of 2-4 % on top of any
> single component. Reliable, defensible, publishable."

### Member 5 - Dr. R. Wehenkel (channelled), Probabilistic / Flows

> "What Y. Le said but with flows: normalising flows give you exact
> log-likelihood, not approximate. Train a conditional flow `p(curve |
> features)` and sample. But more importantly: your evaluation. **Wasserstein-
> only evaluation is missing half the picture.** A model that gets
> Wasserstein 90 with 95 % calibrated 80 %-coverage intervals is more
> valuable than a model with Wasserstein 88 and broken calibration. Add
> CRPS and coverage to your scoreboard - it might *change which model
> wins*. And on the prediction side, isotonic distributional regression
> postprocessing is a free lunch: take any point predictor, output a
> calibrated quantile distribution, monotonic by construction. Two days of
> work, robust gains."

### Member 6 - Prof. A. Smola (channelled), Kernels & Retrieval

> "The oracle gap on demand 30d (-86.80 W) is enormous. That tells you
> *the right curve is in your library, you're just retrieving the wrong
> one*. The fix is not a fancier predictor; it's a **better retrieval key**.
> What goes into the L2 distance you currently retrieve on? A predicted
> embedding from per-interval Ridge with 30-day-lagged features. That's a
> ~70-feature linear projection. Replace it with the *foundation-model
> embedding* of the recent multivariate context - Chronos encoder gives a
> 384-d learned embedding that captures non-linear context. Or train a
> *contrastive* embedding network: triplet loss with anchor = today's true
> embedding, positive = closest historical, negative = a random one. Now
> the embedding space is *shaped* such that nearby points produce nearby
> Wasserstein values. This is essentially metric learning, and on the
> oracle gap you have it could close 30-50 % easily."

### Member 7 - Dr. F. Diebold (channelled), Forecast Econometrics

> "Slow down. Before adding novelty, *ablate properly*. You don't have a
> validation set. Your 'capture 12.9 % of oracle gap' could be 4 % under
> proper Diebold-Mariano testing on a held-out window. Three things first:
> (1) split the last 6 months of train as validation, retune all
> hyperparameters there, *then* score on test. (2) Use Diebold-Mariano with
> Newey-West HAC standard errors to test whether your 'wins' vs lag-30 are
> statistically significant or sample-noise. (3) Decompose by regime -
> stable months vs Jan 2025 spike - using forecast-rationality regressions.
> Half of what you call 'beating baseline' may be regime-conditional.
> Fix that before chasing diffusion models."

### Member 8 - Prof. L. Yang (channelled), Causal / Mechanism

> "What is the data-generating process for a demand curve? It's the result
> of N participants each placing M bids based on their willingness-to-pay,
> aggregated by the auctioneer. The aggregate curve is sufficient for
> price-clearing but **lossy** for forecasting because it loses the
> participant identity. Most of the day-to-day variability in demand isn't
> in the median consumer; it's in 5 large industrial loads switching on /
> off and 20 retailers re-hedging. Either *recover* this structure (per-
> participant decomposition - your `Participant Code` is in the raw data)
> or *condition* on a strong proxy for it (large-load-flag, retailer-hedge-
> activity-yesterday). A monotonic AE on the aggregate curve will *never*
> see this signal because it's already integrated out. Per-participant
> modelling has been done in NEM by Tang 2024 and it's a real improvement;
> in WEM nobody's published it."

### The argument

**Karahan vs Hyndman.** "Foundation models or FPCA?" Karahan: "Why solve
the problem when someone solved it for you?" Hyndman: "Because Chronos
isn't designed for monotonic 64-d outputs, and 4 years of WA-specific data
is *more* informative than a generic pretrained corpus on the *shape* of
this market." Compromise: do both, score head-to-head.

**Le vs Weron.** "Diffusion or ensemble?" Le: "Diffusion gives you the
whole posterior - you don't *need* to ensemble." Weron: "But ensembling
is robust under regime shift, diffusion is brittle." Diebold sides with
Weron: "Sample efficiency on regime-shift days will kill the diffusion
model. Ensemble first, diffusion second."

**Smola vs Le.** "Better retrieval key or generative model?" Smola:
"Cheaper and you have proof that retrieval works (oracle gap)." Le:
"Diffusion *is* a generative retrieval - it samples from the conditional
distribution learned over the library." They agree: Smola's contrastive-
embedding retrieval, then RATD-style retrieval-augmented diffusion on top,
is the sequential winning combination.

**Yang dissents from everyone.** "All of you are working on the wrong
substrate. The aggregate curve already lost half the signal. Per-
participant first." The board concedes this is the most *novel* angle but
also the highest-cost.

**Diebold reminds everyone:** "Validation discipline is mandatory before
any of this is publishable. Bake it into the design of every experiment."

### What the board agrees on

After argument, the board converges on five priorities:

1. **Validation set is non-negotiable** (Diebold). Build it once, use
   everywhere.
2. **Functional representation beats the 64-d quantile embedding**. FPCA
   is the cheapest effective form (Hyndman, agreed by Le, Smola,
   Wehenkel).
3. **Foundation model + retrieval augmentation** is the strongest
   *predictor* lever (Karahan, Smola). Lower bar to entry than diffusion.
4. **Probabilistic / generative output** is the strongest *evaluation*
   lever and unlocks Wasserstein-barycentre point forecasts (Le, Wehenkel).
5. **Per-participant decomposition** is the strongest *novelty* lever
   (Yang) but high-cost; defer unless 1-4 stall.

Yang loses the vote 7-1 on per-participant; the board votes to defer it
to follow-up work.

---

## Part 3 - The four experiments

Ranked by **expected publishable yield × probability of beating baseline**.
Each is fundamentally different from the others. Each comes with the
specific paper that motivates it. Each has a falsifiable success criterion.

### F1 - Functional PCA decomposition + per-component VAR-X with exogenous

**Lit anchor.** FPCA Merit-Order Curves (arXiv 2512.17758, IPEX 2025).
Distributional NN (Marcjasz et al., EPF). Hyndman canon of FDA.

**Motivation.** Right now we predict a 64-d quantile embedding directly.
FPCA replaces it with K = 3-8 *learned eigenfunction scores* that explain
> 95 % of cross-day variance. K scalars are easier to predict than 64
correlated values; the eigenfunctions encode structural knowledge of curve
shape that Ridge has to rediscover from scratch.

**Curve building.**

1. Take the 58 368 training demand curves.
2. Resample each onto a common quantile grid of total qty (we already do
   this in `embeddings/interp.py`) - 64-d vectors `Q_d(p)` for `d` =
   training day.
3. Compute the cross-day mean function `Q̄(p)` and demean.
4. Compute the empirical covariance `C(p, p') = (1/N) Σ_d (Q_d - Q̄)(p)
   (Q_d - Q̄)(p')`.
5. Eigendecompose `C` → eigenfunctions `ξ_1, ξ_2, …, ξ_K` (sorted by
   eigenvalue) and per-day scores `β_{d,k} = ⟨Q_d - Q̄, ξ_k⟩`.
6. Use K = 5 (chosen on validation set; expect 95-98 % cumulative
   explained variance based on IPEX results).

**Predictor.** Per-interval VAR-X with LASSO regularisation on the K-d
score series. Lags = (1, 2, 7, 14, 30, 365). Exogenous: AEMO load
forecast, weather (temperature, wind from the closest BoM station),
weekday/holiday dummies. **One small VAR predicts K=5 scalars** -
massively easier than predicting 64-d.

**Reconstruction.** `Q̂_d(p) = Q̄(p) + Σ β̂_{d,k} ξ_k(p)`. Apply
**isotonic regression** at the end (sklearn `IsotonicRegression`) to
enforce monotonicity. This is a free monotonicity guarantee (Latent Space
Curves 2025 paper).

**Evaluation.** Wasserstein vs lag-1 / lag-30 / smarter-naive on demand 1d
and 30d. Add HC-MAE and CRPS. Score the *score-prediction MAE* on the K
scalars separately as a diagnostic - this tells us whether failures are
in scores or reconstruction.

**Files.**
```
src/aemo_curves/embeddings/fpca.py            # eigendecomposition + transform
src/aemo_curves/models/fpca_var.py            # VAR-X on scores
scripts/build_fpca_artifacts.py
experiments/fpca_demand/
├── README.md
├── __init__.py
├── baseline_lag1.py
├── baseline_lag30.py
├── fpca_var_k3.py / k5.py / k8.py
└── fpca_var_x_exogenous.py
```

**Cost.** ~6-8 hours.

**Success criterion.** Δ ≤ -5 W vs lag-30 on demand 30d, win-rate ≥ 55 %.
Stretch: Δ ≤ -15 W and win-rate ≥ 60 %.

**Risk.** AEMO STEM has fewer participants and possibly noisier curves
than IPEX; the eigenfunctions could be less stable. Mitigate by also
fitting per-interval FPCA (one eigenbasis per interval) - recovers if
inter-interval averaging dilutes the signal.

---

### F2 - Foundation model fine-tune + RAFT-style retrieval augmentation

**Lit anchor.** RAFT (arXiv 2411.08249), TS-RAG (arXiv 2503.07649),
Chronos-Bolt + TimesFM 2.0 benchmarks (2024-2025), AEMO Australian NEM
transformer paper (Preprints.org 202512.1037).

**Motivation.** A 4-year, 70k-sample training set is small for a bespoke
model. Foundation models with billions of pretraining points are
*already competitive* zero-shot on electricity load. Pair them with
retrieval augmentation (RAFT, TS-RAG) to inject our curve library at
inference time: "given today's recent context, the model's representation
is closest to *these K historical days* - let me explicitly attend to
their futures".

**Setup.**

1. **Encode each per-interval scalar series.** For each of 64 quantile
   positions, build a univariate series `e_d^{(k)}` - one value per day,
   per interval. With 48 intervals × 64 = 3 072 scalar series, batch them.
2. **Get a Chronos-Bolt embedding** of each series's recent 60-day window
   (or use TimesFM 2.0 - ablate). This 384-d learned embedding becomes the
   retrieval key.
3. **Retrieval (RAFT-style).** For each test row, find the top-5 most
   similar historical contexts by L2 in embedding space. Extract their
   "futures" (today + next 30 days of the embedding scalar).
4. **Augmented forecast.** Two strategies, both ablated:
   - (a) **Zero-shot ensemble.** Average the top-5 retrieved futures
     (instance-normalised) - this is essentially soft-KNN in the
     foundation embedding space.
   - (b) **Fine-tune Chronos-Bolt-Small with LoRA** (rank 8) for 2 epochs
     on our 4-year history with the retrieved context concatenated to the
     query. Learn to fuse retrieval and pre-training.
5. Reassemble the 64-d embedding from the 64 per-quantile scalars.
   Reconstruct the curve via `pred_qty = DEFAULT_QUANTITY_GRID *
   total_qty`, `pred_price = predicted_embedding`. Apply isotonic
   monotonicity post-hoc.

**Evaluation.** Same as F1. Critically also report **per-quantile
forecast MAE** to diagnose whether the foundation model wins *uniformly*
across the curve or only at certain quantiles.

**Files.**
```
src/aemo_curves/models/foundation.py          # Chronos / TimesFM wrappers
src/aemo_curves/models/raft.py                # retrieval augmentation
scripts/build_foundation_embeddings.py
experiments/foundation_demand/
├── README.md
├── chronos_zeroshot.py
├── chronos_zeroshot_raft.py
├── chronos_lora_finetune.py
├── chronos_lora_finetune_raft.py
└── timesfm_zeroshot_raft.py
```

**Cost.** ~10-15 hours including a single A100-day for fine-tune.

**Success criterion.** Δ ≤ -10 W vs lag-30 on demand 30d for at least one
foundation model variant. Stretch: zero-shot already wins.

**Risk.** Chronos's 64-position-as-independent-series simplification may
lose cross-quantile structure. Mitigate by ablating: also try treating the
64-d as a 64-channel multivariate series with TimesFM 2.0 (which natively
supports multivariate).

**Why this matters even if it doesn't win.** Demonstrates whether the
"foundation model lift" generalises to bid curves. Negative result is
publishable (Apen Energy) because nobody has tested this configuration.

---

### F3 - Conditional diffusion model on demand curves with Wasserstein-barycentre point forecast

**Lit anchor.** RATD (NeurIPS 2024), DiffPLF (Applied Energy 2024),
EnergyDiff (arXiv 2407.13538). Wasserstein barycentre theory.

**Motivation.** Wasserstein is a distributional metric. A point forecaster
trained on MSE / MAE on the embedding is a model-objective mismatch. A
generative model that samples from `p(curve | context)`, then aggregates
samples by Wasserstein barycentre, is *natively aligned* with the
evaluation. Plus we get calibrated uncertainty bands for free.

**Setup.**

1. **Architecture.** A 1-D U-Net conditional diffusion model. Input: 200-d
   `Ĉ_t` segment on box-relative grid (E1 from the prior doc) or the 64-d
   quantile embedding directly. Conditioning: lag-30 curve + lag-7 curve +
   calendar (5-d) + exogenous (load forecast, weather) - concatenated and
   projected to a context vector by a small MLP, fed into U-Net via
   cross-attention (DiffPLF's design).
2. **Training.** DDPM loss on the embedding, 1000 forward steps, cosine
   schedule. Train across all intervals with interval-id as an additional
   conditioning token (saves training one model per interval).
3. **Sampling.** At test time, run 100 reverse-diffusion samples per
   query. Aggregate:
   - **Median sample** as the simplest point forecast.
   - **Wasserstein-1 barycentre** computed by a 1-D Frank-Wolfe iteration
     (cheap because curves are 1-D distributions). This minimises mean
     Wasserstein among the samples.
4. Apply isotonic regression to enforce monotonicity per sample.

**Diagnostic outputs.**

- 80 % / 90 % / 95 % coverage of true curve in predicted band.
- CRPS per quantile.
- Ensemble member diversity (mean L2 between samples).

**Evaluation.** Wasserstein vs baselines. CRPS as secondary. Coverage as
tertiary (calibration).

**Files.**
```
src/aemo_curves/models/diffusion.py
src/aemo_curves/models/wasserstein_barycentre.py
scripts/train_diffusion.py
experiments/diffusion_demand/
├── README.md
├── unet_diffusion_64d.py
├── unet_diffusion_box_relative.py     # combine with E1 from prior doc
└── diffusion_with_retd_retrieval.py    # F2 + F3 hybrid
```

**Cost.** ~15-20 hours including a single A100-day for training.

**Success criterion.** Wasserstein-barycentre forecast: Δ ≤ -10 W vs
lag-30 on demand 30d. CRPS ≤ 0.85 × CRPS of empirical-residual baseline.
Coverage in `[0.85, 0.95]` for 90 % nominal.

**Risk.** Diffusion is the most capricious of the four - can mode-collapse,
can over-smooth. Mitigate with EMA of weights and validation-set early
stop. Also pre-train the U-Net on raw curves (no conditioning) to get a
sensible prior, then fine-tune the conditioning.

**Why this is uniquely publishable.** **No published paper does diffusion
for aggregate bid curves.** DiffPLF / EnergyDiff did individual-customer
load profiles, which is conceptually different. RATD did generic time
series. The combination "1-D diffusion + bid curves + Wasserstein
barycentre point forecast + AEMO" is genuinely novel and Apen-Energy-shape.

---

### F4 - Wasserstein-barycentre ensemble of three predictors

**Lit anchor.** QRA / iQRA (Cuaresma 2024). Wasserstein-1 barycentre theory.
Hyndman & Athanasopoulos forecasting bible §13.4 on combination forecasts.

**Motivation.** F1, F2, F3 each address a different lever and will have
different failure modes. The simplest robust path is to **combine them**
under the natural metric. Forecast combination is the single most
reproducible win in the EPF literature - small, defensible, and the
ensemble is *insurance* against any one component being wrong on a
particular regime.

**Setup.**

1. **Components.** Three already-fit forecasters:
   - F1's FPCA + VAR-X
   - F2's Chronos-with-RAFT
   - The current best **strat-KNN** model (`k=1 q=ridge lags=(30,) α=200
     b=int_dow`).
2. **Combination weights.** Two strategies, ablated:
   - (a) **Equal weights** (the boring-but-robust choice).
   - (b) **Validation-set-fit weights**, learned by minimising mean
     Wasserstein over validation rows.
3. **Aggregation.** Compute the **1-D Wasserstein barycentre** of the
   three predicted curves with the learned (or equal) weights. For
   1-D measures the barycentre is a closed-form weighted average of
   inverse CDFs (= weighted average of quantile-embeddings) - so it's
   trivially fast.
4. **Isotonic regression** at the end.

**Why barycentre is the right combiner.** Averaging *prices* directly
breaks monotonicity (equal-weighted average of two monotonic curves can
zigzag). The Wasserstein barycentre of monotonic 1-D measures is
*always* monotonic (and equals the weighted average of the inverse CDFs,
which is the *quantile embedding* - exactly our embedding form).
Mathematically: `barycentre = (Σ w_i Q_i^{-1})^{-1}`. If we operate on
the embedding side, this is *literally* `Σ w_i e_i`. Free monotonicity,
free linearity, exact alignment with the metric.

**Evaluation.** Same as the others. Diagnostic: do the optimal weights
favour one component over another, and does that vary by regime?

**Files.**
```
src/aemo_curves/models/wasserstein_ensemble.py
experiments/ensemble_demand/
├── README.md
├── equal_weights.py
├── learned_weights.py
└── learned_weights_per_regime.py
```

**Cost.** ~3 hours assuming F1, F2, F3 already produced predictions.
**Note**: F4 cannot be run before F1+F2+F3.

**Success criterion.** Beats *all* of F1, F2, F3 individually on demand
30d Wasserstein by ≥ 1 W. (If it doesn't, the components were already
optimal alone - itself a clean negative result.)

**Risk.** Lowest of the four; combining fitted components is robust.
Even if F1/F2/F3 each barely match lag-30, the ensemble usually edges
ahead by 2-4 %.

---

## Part 4 - Cross-cutting infrastructure (one-time investment, pays for all four)

### Validation set discipline (D-block)

Before *any* of F1-F4. Carve `2023-11-01` → `2024-04-30` (6 months) out
of train as `val`. Test set unchanged (`2024-05-01` → `2025-05-31`).
Update `config.py`, `loaders.py`, `experiments_api.py` to expose val.
All hyperparameter selection happens on val. **~1.5 hours.**

### HC-MAE metric (M-block)

Implement [`metrics/hc_mae.py`](../src/aemo_curves/metrics/hc_mae.py) per
Sinha & Lucheroni. **~30 minutes.**

### Smarter-naive baseline (B-block)

`SmarterNaivePredictor`: lag-7 if today ∈ {Sat, Sun, Mon} else lag-1.
**~20 minutes.** For 30d horizon: `SmarterNaive30Predictor`: lag-30 if
weekday-aligned, else lag-{30 ± 1 to align}.

### Isotonic monotonicity post-hoc (I-block)

`predictors/_isotonise.py` wrapper applicable to *any* `CurvePredictor`
that may emit non-monotonic curves. Apply at the end of `predict()`.
**~20 minutes.**

### CRPS metric (C-block)

`metrics/crps.py` - empirical-residual CRPS for any predictor that emits
samples (F3 only initially) plus a normal-approximation CRPS for the
others (using residual variance from validation). **~1 hour.**

D + M + B + I + C ≈ **3.5 hours total**, then F1-F4 share these.

---

## Part 5 - Execution order and gating

```
Phase 0 (parallel, ~3.5 h):  D + M + B + I + C
Phase 1 (parallel, ~14 h):    F1 (FPCA + VAR-X)
                             F2 (Chronos + RAFT)
Phase 2 (after F2 done):     F3 (Diffusion, can also use F2's embedding)
Phase 3 (after F1+F2+F3):    F4 (Wasserstein barycentre ensemble)
```

**Gate after Phase 1.** If F1 and F2 *both* produce Δ ≥ 0 (i.e., neither
beats lag-30), revisit assumptions before sinking GPU days into F3. Most
likely failure mode: AEMO demand has a fundamentally different signal
structure than IPEX/MIBEL/NEM, in which case Yang's per-participant
proposal becomes the primary path.

**Likeliest winner ex-ante.** F1 + F4 combination - FPCA delivers a clean
representational improvement, ensemble buys the last 2-4 % robustness.
F2 is the most exciting; F3 is the most novel; F4 is the safest path to
a positive headline number.

## Part 6 - What this gets us in publication terms

- **F1 alone** = a competent *Energy Economics* / *Energy* paper:
  "Functional PCA decomposition for AEMO STEM bid curves with
  monotonicity-preserving reconstruction".
- **F2 alone** = a *Patterns* / *Applied Energy* paper: "Foundation models
  with retrieval augmentation for electricity bid curve forecasting:
  evidence from the Australian WEM".
- **F3 alone** = an *Applied Energy* / *NeurIPS Climate Change AI*
  workshop paper: "Conditional diffusion for probabilistic bid curve
  forecasting".
- **F1 + F4 (ensemble win)** = the strongest *Applied Energy* full paper,
  comparable to Sinha & Lucheroni 2025 in scope and ambition, but on
  AEMO data (a market they explicitly named as future work) and with
  ensemble robustness.
- **All four** = a thesis chapter and a multi-part submission strategy.

---

## Sources

Literature consulted in producing this document:

- [Sinha & Lucheroni 2025, Applied Energy 397:126262 - Monotonic autoencoder for bid curves](https://www.sciencedirect.com/science/article/pii/S0306261925009924)
- [Tang et al. 2024, Applied Energy 363 - LRA + transformer for individual bids on Australian NEM](https://www.sciencedirect.com/science/article/abs/pii/S0306261924004367)
- [YUI 2024 (arXiv 2405.14893) - Invariance simplified supply/demand curves for DAM EPF](https://arxiv.org/html/2405.14893v1)
- [FPCA Merit-Order Curves 2025 (arXiv 2512.17758) - FPCA + VAR-X on IPEX](https://arxiv.org/html/2512.17758v2)
- [Latent Space Curves 2025 (arXiv 2503.11294) - UMAP + isotonic for MIBEL](https://arxiv.org/html/2503.11294)
- [RAFT 2024 (arXiv 2411.08249) - Retrieval-augmented forecasting with Chronos](https://arxiv.org/abs/2411.08249)
- [RATD NeurIPS 2024 - Retrieval-augmented diffusion models for time series](https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf)
- [TS-RAG 2025 (arXiv 2503.07649) - RAG with TS foundation models + FAISS](https://openreview.net/forum?id=TJuUelhGQr)
- [PriceFM 2025 (arXiv 2508.04875) - Spatiotemporal foundation model for prices](https://arxiv.org/html/2508.04875v1)
- [Chronos / TimesFM benchmark for household STLF (arXiv 2410.09487)](https://arxiv.org/html/2410.09487)
- [Distributional NN for EPF (Marcjasz et al.)](https://arxiv.org/pdf/2207.02832v1/1000)
- [DiffPLF - Conditional diffusion for EV charging load (arXiv 2402.13548)](https://arxiv.org/html/2402.13548)
- [EnergyDiff - Universal time-series energy generation (arXiv 2407.13538)](https://arxiv.org/html/2407.13538v1)
- [Customised Load Profiles - Conditional diffusion (arXiv 2304.12076)](https://arxiv.org/abs/2304.12076)
- [Dumas & Wehenkel - Normalizing flows for energy forecasting (Applied Energy 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0306261921011909)
- [Multivariate scenario generation via flows (Applied Energy 2024)](https://www.sciencedirect.com/science/article/pii/S030626192400624X)
- [Isotonic Distributional Regression for EPF postprocessing (arXiv 2404.02270)](https://arxiv.org/html/2404.02270)
- [Isotonic QRA (iQRA) for EPF (arXiv 2507.15079)](https://arxiv.org/html/2507.15079)
- [Multi-horizon CDF forecasting with Deep Lattice Networks (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/36986)
- [Non-crossing QRNN for ensemble calibration](https://link.springer.com/article/10.1007/s00376-023-3184-5)
- [Wasserstein Autoregressive Models for density time series (arXiv 2006.12640)](https://arxiv.org/pdf/2006.12640)
- [GeONet - Wasserstein geodesic neural operator (arXiv 2209.14440)](https://arxiv.org/html/2209.14440v4)
- [Australian NEM transformer EPF (Preprints.org 202512.1037)](https://www.preprints.org/manuscript/202512.1037)
- [Review of EPF models in DAM/intra-day/balancing markets (Energies 2025)](https://www.mdpi.com/1996-1073/18/12/3097)
- [Functional data analysis for day-ahead electricity prices (Frontiers 2025)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1477248/full)
- [Quantile Neural Basis Models for EPF (arXiv 2509.14113)](https://arxiv.org/html/2509.14113v1)
- [Multivariate probabilistic CRPS learning for EPF (Marcjasz, IJF 2024)](https://www.sciencedirect.com/science/article/pii/S0169207024000050)
