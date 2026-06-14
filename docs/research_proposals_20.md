# Twenty research proposals - beating lag-30 on demand curves

Status: this plan was carried out, see the experiment folders under [`experiments/`](../experiments/).

**Date written:** 2026-05-12
**Author:** synthesis of a five-expert board (deep learning · FDA · probabilistic forecasting · domain/structural · representation learning), each instructed to read recent literature on their specialty and produce four well-cited proposals.

## 1. Context for any reader picking up this file

- **Task**: forecast the 30-day-ahead demand step-curve of each 30-min WA STEM auction.
- **Train**: 52 560 demand auctions, 2021-01 → 2023-12.
- **Test**: 17 568 auctions, 2024-01 → 2024-12 (new split - see [`experiments/recompute_2024test/README.md`](../experiments/recompute_2024test/README.md)).
- **Baseline**: `Lag1Predictor(lag_days=30)` - demand curve at the same interval 30 days earlier. Mean Wasserstein on 2024 test = **112.43**.
- **Current best realised model (production-deployable)**: `B1 rolling-refit Ridge`, mean W = 93.48, **Δ_W = -18.95** (16.9 % better than baseline), win-rate 58 %. Capture of bucket-oracle gap = **25.8 %**. Loses by Δ_DTW = +62 under DTW.
- **Static bucket-oracle ceiling**: Δ_W = -73.51 (NOT deployable - knows the future; used only as a reference).

**The 20 proposals below are the synthesis of five expert agents' independent reviews of recent literature**, each producing four citation-grounded ideas. Several overlapping themes were preserved as distinct entries because the *forecaster* or the *training signal* differs even when the *representation* doesn't.

---

## 2. Family A - Deep learning sequence models for embedding forecasting

> Reviewer: deep-learning time-series expert. Focus: transformers, state-space, foundation models. Strongest priors on `iTransformer` and on retrieval-augmented diffusion.

### #1. Monotonic autoencoder + PatchTST on bucket-conditional latents

**Methodology**: Train a monotonic autoencoder (MAE) that maps each demand curve to an 8-16-d latent `z` whose decoder guarantees a non-decreasing reconstruction (via distance-weighted kNN over training curves - Sinha & Lucheroni 2025). Forecast `z_{t+30}` per `(interval, weekday)` bucket with a small sequence model (GRU or PatchTST) conditioned on lag-30 / lag-7 / lag-1 latents and calendar covariates. Decode the forecast `z` through the training-set kNN so the output is a real historical curve combination (preserves the "real curve" property of retrieval baselines).

**Model / implementation**: PyTorch custom MAE (encoder 64→32→16, decoder = monotonic-kNN over training curves with K=5) + `neuralforecast` PatchTST (`patch_len=7`, ≈ 250k params per bucket or 1-2M params with bucket-id token). ~2-4 GPU-h.

**Training data & features**: latent trajectory per bucket, plus optional weather/calendar exogenous covariates. Target = `z_{t+30}`.

**Scoring vs baseline**: primary = mean Wasserstein on demand, Δ_W = W̄(model) - W̄(`Lag1Predictor(lag_days=30)`). Paired bootstrap 95 % CI clustered by Trading Date. Secondary = DTW (probes whether shape gains survive alignment).

**Expected effect / prior**: P(beat baseline) ≈ 0.65; P(capture ≥ 50 % of oracle gap) ≈ 0.30. Risk: latent dynamics may collapse to lag-1-of-latents ≈ lag-30.

**References**:
- Sinha & Lucheroni, *Demand and supply curve forecasting using a monotonic autoencoder...* (Applied Energy 2025). https://doi.org/10.1016/j.apenergy.2025.126262
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)* (ICLR 2023). https://arxiv.org/abs/2211.14730

---

### #2. iTransformer over the 64-d interp embedding as a 64-variate series

**Methodology**: Treat the 64-d interp embedding as a 64-variate time series indexed by trading day, per `(interval, weekday)` bucket. Apply iTransformer - which tokenizes each *variate* as one token, so attention learns cross-quantile dependencies and FFN learns per-quantile dynamics. Predict the full 64-d vector 30 days ahead, then snap to the nearest training curve via L2 retrieval to keep real-curve property.

**Model / implementation**: `neuralforecast` iTransformer, `d_model=128`, 3 encoder layers, ≈ 600k params. One shared model with bucket-id covariate. Lookback L=120 trading days. ~1-2 GPU-h.

**Training data & features**: 64-d embeddings + 30-day-shifted weather/calendar exog. Target = embedding at t+30.

**Scoring vs baseline**: Wasserstein and DTW vs `Lag1Predictor(lag_days=30)`. Paired bootstrap CI.

**Expected effect / prior**: P(beat baseline on W) ≈ 0.55; P(beat under DTW too) ≈ 0.35. Variate attention is exactly the cross-quantile inductive bias missing from per-coordinate Ridge.

**References**:
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* (ICLR 2024). https://arxiv.org/abs/2310.06625
- Hertel et al., *Benchmarking Time Series Foundation Models for Short-Term Household Electricity Load Forecasting* (2024). https://arxiv.org/abs/2410.09487

---

### #3. Retrieval-augmented diffusion (RATD) over curve embeddings

**Methodology**: A conditional diffusion model (CSDI / TimeGrad lineage with Retrieval-Augmented Time-series Diffusion) that generates the t+30 curve embedding conditioned on (a) bucket id, (b) recent lag embeddings, (c) k=5 nearest-neighbour historical embeddings from the existing retrieval index. Diffusion samples N=64 candidate curves; aggregate by medoid (or take the mean for a CRPS-aware probabilistic forecast). Adds probabilistic forecasts as a new capability on top of existing retrieval.

**Model / implementation**: Custom PyTorch with Reference-Modulated Attention on a 1-D U-Net denoiser (~1-3M params), 200 diffusion steps with DDIM acceleration. ~6-10 GPU-h.

**Training data & features**: 64-d embeddings + bucket id + lag-1 / lag-7 / lag-30 references + retrieved neighbours from existing index. Target = embedding at t+30; loss = score-matching.

**Scoring vs baseline**: Wasserstein (medoid point forecast), CRPS (probabilistic), DTW. Δ_W vs `Lag1Predictor(lag_days=30)`. Paired bootstrap CI.

**Expected effect / prior**: P(beat baseline on W) ≈ 0.5; P(meaningfully better CRPS than retrieval) ≈ 0.7. Highest *new-capability* value (probabilistic) but highest engineering cost.

**References**:
- Liu et al., *Retrieval-Augmented Diffusion Models for Time Series Forecasting* (NeurIPS 2024). https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf
- Rasul et al., *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting (TimeGrad)* (ICML 2021). https://arxiv.org/abs/2101.12072
- Tashiro et al., *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* (NeurIPS 2021). https://arxiv.org/abs/2107.03502

---

### #4. Zero-/few-shot foundation models (Chronos-Bolt, TimesFM) per embedding coordinate

**Methodology**: For each of the 64 embedding coordinates, run a pretrained TSFM (Chronos-Bolt 200M, TimesFM 2.0) **zero-shot** per bucket to forecast 30 days ahead; reassemble forecasts into a curve embedding; project to the nearest real training curve. Optionally LoRA-finetune on the train split. Recent ERCOT benchmarks show TSFMs achieving ~47 % MASE reduction vs seasonal-naïve on hourly load - the question is whether that transfers to *curve-coordinate* series.

**Model / implementation**: `chronos-forecasting` (AWS) and `timesfm` (Google) via HuggingFace. Zero-shot inference only: CPU/GPU minutes per bucket. LoRA finetune optional (~30 min on a single GPU with rank-8 adapters).

**Training data & features**: per-bucket per-coordinate univariate series (336 × 64 ≈ 21 500 series of length ~1 200). Target = coordinate value at t+30.

**Scoring vs baseline**: Wasserstein vs `Lag1Predictor(lag_days=30)`. Zero-shot vs LoRA comparison. CRPS via TSFM native sampling.

**Expected effect / prior**: P(zero-shot beats lag-30) ≈ 0.30 (curve coordinates aren't natural load series; monotonicity not preserved). P(LoRA-tuned beats) ≈ 0.55. **Cheapest experiment to run first** - high information per GPU-hour.

**References**:
- Ansari et al., *Chronos: Learning the Language of Time Series* (2024). https://arxiv.org/abs/2403.07815
- Meyer et al., *Time Series Foundation Models for Energy Load Forecasting on Consumer Hardware* (2026). https://arxiv.org/abs/2602.10848

---

## 3. Family B - Functional data analysis on curves

> Reviewer: FDA expert. Focus: replicating the Spanish OMIE / IPEX literature on demand-curve forecasting; FPCA + functional regression + monotonicity-preserving methods.

### #5. Function-on-function regression with FPCA + closest-curve snap (Li-Alonso-Pascual replication)

**Methodology**: Treat the curve as a function `D_t(p)`. Project onto a B-spline / FPCA basis (8-15 components on quantity-as-function-of-price). Regress lag-30/lag-7/lag-1 scores plus calendar / weather scalars on next-step scores with a function-on-function linear model or semi-functional partial linear (Aneiros-Vilar). **Critically, after producing a smooth functional forecast, snap to the nearest historical curve** in functional L² distance - replicating Li-Alonso-Pascual (2025) and mirroring why our lag-1 baseline is strong.

**Model / implementation**: `scikit-fda` (`FPCA`, `FunctionalLinearRegression`) or `fda.usc` (R). Medoid retrieval via brute KNN over FPCA scores. Compute: minutes per fit; monthly rolling refit feasible.

**Training data & features**: lag-30/lag-7/lag-1 functional scores + DoW + public holiday + optional weather. Target = next-day curve, snapped to nearest historical.

**Scoring vs baseline**: Wasserstein primary; Δ_W vs `Lag1Predictor(lag_days=30)`. Capture-of-oracle-gap. Add functional-L² as secondary.

**Expected effect / prior**: ~30 % chance of modest improvement (Δ_W ∈ [-25, -19]). The Li-Alonso-Pascual paper reports gains over naïve/exponential-smoothing baselines on OMIE - closely analogous setting.

**References**:
- Li, Alonso, Pascual, *Predicting electricity supply and demand curves with functional data techniques* (IJEPES 166, 2025). https://doi.org/10.1016/j.ijepes.2025.110561
- Aneiros, Vilar, Raña, *Short-term forecast of daily curves of electricity demand and price* (IJEPES 80, 2016). https://www.sciencedirect.com/science/article/abs/pii/S0142061515004810

---

### #6. Functional autoregression (FAR(1)) on inverse-demand curves

**Methodology**: Model the curve time series per interval as a functional AR(1) `X_{t+30} = Ψ(X_t) + ε`, with Ψ estimated either by regressing FPCA scores or by Kargin-Onatski functional-AR. Fit 48 interval-specific FARs over the full train. Optionally invert demand → inverse demand (price as function of cumulative quantity) so the response is on a fixed `[0, Q_max]` grid; Aneiros et al. found this stabilises functional prediction in spot markets.

**Model / implementation**: Python `scikit-fda` + closed-form Ψ estimation, or `fda.usc::fregre.basis`. ~1 h to fit 48 interval-specific FARs.

**Training data & features**: lag-30 demand curve (functional) + optional scalar FPCA scores of lag-1 and lag-7 (semi-functional partial linear). Calendar dummies in the scalar block.

**Scoring vs baseline**: Wasserstein primary, Δ_W vs `Lag1Predictor(lag_days=30)`. Functional-L² and step-aware DTW as diagnostics. lag-30 is Ψ = identity; any non-trivial Ψ that shrinks toward seasonality should help.

**Expected effect / prior**: ~40 % chance to beat current best. FAR(1) is hard to beat in OMIE residual-demand work; 30-day horizon is where mean-reverting / seasonal AR dynamics dominate.

**References**:
- Aneiros, Vilar, Cao, Muñoz San Roque, *Functional prediction for the residual demand in electricity spot markets* (IEEE TPWRS 28(4), 2013). https://www.researchgate.net/publication/281149602
- Peláez, Aneiros, Vilar, *Bootstrap prediction regions for daily curves of electricity demand and price using functional data* (arXiv 2024). https://arxiv.org/abs/2401.11885

---

### #7. Monotonic autoencoder + Echo-State forecast (Sinha-Lucheroni style)

**Methodology**: Decompose each curve into anchor points `A_t` (low end), `B_t` (high end), and interior monotone segment `C_t`. Train a monotonic autoencoder on `C_t` (positive-weight or cumulative-softmax decoder guarantees non-decreasing reconstruction). Forecast `A_t`, `B_t` with an Echo State Network using calendar+lag features; forecast the latent of `C_t` with a second ESN. Recompose to a guaranteed-monotone forecast. This is the Sinha & Lucheroni (2025) experiment.

**Model / implementation**: Monotonic AE (~2k params, 16-d bottleneck) + `pyESN` / `reservoirpy`. ~2 h training; microseconds inference.

**Training data & features**: 30-day window of past curves (latent codes) + weekday + holiday + interval-of-day + optional load forecast. Target = curve at t+30.

**Scoring vs baseline**: Wasserstein primary; Δ_W vs lag-30. Add monotonicity-violation rate (should be 0). Compare especially on high-volatility tail days where Sinha-Lucheroni report largest gains.

**Expected effect / prior**: ~35 % chance to beat current best. Risk: WA STEM curves are short (10-50 steps), the latent bottleneck may overfit. But guaranteed monotonicity is a structural prior Ridge retrieval can't directly impose.

**References**:
- Sinha & Lucheroni, *Demand and supply curve forecasting using a monotonic autoencoder...* (Applied Energy 2025). https://doi.org/10.1016/j.apenergy.2025.126262
- Shin, Zhou, Ding, *Joint estimation of monotone curves via functional principal component analysis* (CSDA 166, 2022). https://doi.org/10.1016/j.csda.2021.107343

---

### #8. Quantile-curve forecast via FPCA in log-derivative space (Ramsay 1998)

**Methodology**: Re-interpret the cumulative-quantity demand curve as a quantile / inverse-CDF `Q_t(u)`. Take `log Q'_t(u)` (unconstrained), apply FPCA, run any linear forecasting model on the scores, then exponentiate-and-integrate to return a strictly monotone forecast curve. This is the Ramsay (1998) trick. The key insight: **W₁ between curves ≈ L¹ between their inverse CDFs**, so FPCA on `Q` is implicitly a Wasserstein-geometry approximation - and Ridge on the scores is implicitly minimising a proxy of the test metric.

**Model / implementation**: `scikit-fda` FPCA on log-derivatives + scalar Ridge on score vectors. Minutes total; trivially refittable monthly. Naturally compatible with the existing `interp.py` 64-d pipeline as a drop-in replacement embedding.

**Training data & features**: lag-30 + lag-1 + lag-7 score vectors + interval one-hot + holiday flag. Optionally hybridise with closest-curve snap (Proposal #5).

**Scoring vs baseline**: Wasserstein primary; Δ_W vs lag-30. Because of W₁ ≈ L¹-in-Q the loss-metric alignment is **exact** - the strongest theoretical motivation in Family B. Embedding-L2 in score space as diagnostic.

**Expected effect / prior**: ~45 % chance to beat current best (highest in Family B) - because (i) loss-metric alignment is exact, (ii) monotonicity is automatic, (iii) preserves the "real curve" property if hybridised with snap, (iv) FPCA scores have low rank so ridge is well-posed at 30-day horizon. Risk: log-derivative sensitive to flat segments → needs smoothing pre-step.

**References**:
- Ramsay, *Estimating smooth monotone functions* (JRSS B 60, 1998). https://doi.org/10.1111/1467-9868.00130
- Vivó & Alonso, *Prediction of Intraday Electricity Supply Curves* (Applied Sciences 14(22), 2024). https://www.mdpi.com/2076-3417/14/22/10663

---

## 4. Family C - Probabilistic / distributional forecasting

> Reviewer: probabilistic forecasting expert. Focus: CRPS-aware training, conformal calibration, EPF literature (Marcjasz, Weron, Uniejewski). The cleanest direction to also produce calibrated uncertainty intervals.

### #9. Implicit Quantile Network (IQN) over the inverse-CDF

**Methodology**: Each curve *is* a quantile function `Q(τ)` over normalised cumulative quantity. Train an Implicit Quantile Network conditioned on (historical embedding, calendar, lag-30 features, exog) that maps a sampled `τ → predicted price`. Loss = mean pinball over `τ ∼ U(0,1)` - an unbiased CRPS estimator (Dabney-style). Optional non-crossing penalty. At inference, evaluate on the test auction's true `τ` grid to reconstruct a monotone curve via sort. IQN directly minimises CRPS, which is tightly related to W₁ (W₁ = ∫|F-G|, CRPS = E|X-y|).

**Model / implementation**: PyTorch MLP (3-4 layers, 256 units) on `[context_embedding ⊕ cos/sin(πτk) for k=1..64]`. Loss = pinball + optional λ·max(0, ∂price/∂τ < 0)². ~30 min on a single GPU.

**Training data & features**: (τ, price) pairs from each train auction (subsample 64 points). Features = lag-30 interp embedding + DoW + interval + lag-1 / lag-7 prices + exog.

**Scoring vs baseline**: Wasserstein (resample on the true τ grid) + CRPS averaged over τ. Δ_W vs `Lag1Predictor(lag_days=30)`.

**Expected effect / prior**: Moderately positive. IQN beats MQ-RNN by ~5-10 % CRPS on electricity benchmarks. Likely Δ_W in -5 % to -15 %; may not beat retrieval alone but offers calibrated bands as side product.

**References**:
- Gouttes et al., *Probabilistic Time Series Forecasting with Implicit Quantile Networks* (2021). https://arxiv.org/abs/2107.03743
- Smyl et al., *Any-Quantile Probabilistic Forecasting of Short-Term Electricity Demand* (2024). https://arxiv.org/abs/2404.17451

---

### #10. Distributional neural network with Johnson-SU output per τ-slice (Marcjasz et al.)

**Methodology**: Replicate Marcjasz et al.'s **DDNN-JSU**: for each of K∈{32, 64} fixed τ-quantile slices of the curve, the network outputs the 4 parameters of a Johnson's SU (location/scale/skew/kurtosis) - handles the heavy tails Gaussians miss. Reconstruct the curve by taking the median (or chosen quantile) at each `τ_k`. Loss = sum of CRPS_JSU over slices (closed form). Post-isotonise across τ for monotonicity. Essentially Bayesian-style parametric quantile regression directly optimising a CRPS-decomposable objective.

**Model / implementation**: PyTorch DDNN with a "probability layer" producing 4·K outputs; loss = analytic CRPS_JSU or 200-MC-sample numeric CRPS. ~15 min/training, 4-layer MLP, ~200k params. Ensemble of 4 seeds (Marcjasz et al. show ensembling is critical).

**Training data & features**: target = price at K fixed τ grid points. Features = lag-30 embedding + lag-1 + lag-7 + DoW + hour + exog.

**Scoring vs baseline**: Wasserstein + per-slice CRPS. Δ_W vs lag-30. Also Winkler / coverage for 90 % bands.

**Expected effect / prior**: Cautiously optimistic. Marcjasz et al. report 7 % CRPS improvement over best benchmarks on the German market. But per-slice JSU ignores cross-τ correlation (isotonic step recovers ordering but not joint). Δ_W likely in -8 % to -15 %; strong candidate for ensembling with retrieval (Proposal #12).

**References**:
- Marcjasz, Narajewski, Weron, Ziel, *Distributional neural networks for electricity price forecasting* (Energy Economics, 2023). https://arxiv.org/abs/2207.02832
- Lipiecki, Uniejewski, Weron, *Postprocessing of point predictions ... Isotonic Distributional Regression* (Energy Economics, 2024). https://arxiv.org/abs/2404.02270

---

### #11. NGBoost per τ-slice with curve-shape covariates

**Methodology**: For each of K=64 τ-slices independently, train NGBoost (natural-gradient boosted trees) outputting a parametric distribution (Normal or LogNormal). Curves reconstructed by predicting the mean (or median) per slice and isotonising. Pure tabular approach - strong tabular baseline of GEFCom / EPF benchmarks. The any-quantile variant (`τ_k` as a feature in a single joint multi-output regressor) avoids fitting 64 separate models.

**Model / implementation**: `ngboost` library, Normal distribution, 500 estimators, max_depth=4 (one model per slice OR multi-output joint). Alternative: `quantile-forest` (Quantile Regression Forests) trained jointly across all τ. ~1 h CPU total for 64 slices × 52k rows.

**Training data & features**: target = price at `τ_k`. Features = lag-30 embedding (64 dims) + lag-1 / lag-7 prices at the same `τ_k` (very strong autoregressive signal) + calendar + exog + `τ_k` itself (any-quantile trick).

**Scoring vs baseline**: Wasserstein primary, CRPS secondary (NGBoost reports natively). Δ_W vs lag-30.

**Expected effect / prior**: Δ_W likely in -12 % to -20 %; plausibly competitive with current -18.95 %. Tabular boosting is the workhorse winner of GEFCom and EPF benchmarks. Main weakness: per-slice independence (mitigated by lag features encoding shape).

**References**:
- Duan et al., *NGBoost: Natural Gradient Boosting for Probabilistic Prediction* (ICML 2020). https://arxiv.org/abs/1910.03225
- Meinshausen, *Quantile Regression Forests* (JMLR 2006) - and Uniejewski & Weron, *Regularized quantile regression averaging for probabilistic EPF* (Energy Economics 2021). https://www.sciencedirect.com/science/article/abs/pii/S0140988321000268

---

### #12. Conformalised retrieval ensemble (QRA over retrieval + DDNN + NGBoost)

**Methodology**: Highest-prior in Family C because it builds on the already-best model. Take the current B1 retrieval baseline as the *median* forecast; train an auxiliary residual model (light GBM or DDNN) on the empirical residuals on a calibration split; use **Sequential Predictive Conformal Inference (SPCI)** or **EnbPI** for τ-wise prediction intervals with guaranteed marginal coverage. Combine retrieval, DDNN-JSU (#10), and NGBoost (#11) via Uniejewski-Weron **regularised QRA**. Hubicka et al. (2024) show such ensembles beat standalone DDNN-JSU.

**Model / implementation**: Conformal layer via `mapie` or custom SPCI (~200 LoC). Combination via `LassoQuantileRegressor`. Cheap once base models exist.

**Training data & features**: base predictions on a held-out calibration period (last 6 months of train) - residuals at each τ slice. Conformity scores τ-stratified.

**Scoring vs baseline**: Wasserstein of the ensemble median + 50/80/90 % coverage and Winkler. Δ_W = W(ensemble median) - W(`Lag1Predictor(lag_days=30)`).

**Expected effect / prior**: **Highest-prior winner of Family C.** QRA + isotonic combos outperform DDNN-JSU in Hubicka et al. Δ_W ≈ -22 % to -30 % (real improvement over current -18.95 %). Bonus: well-calibrated uncertainty intervals "for free".

**References**:
- Mastropietro et al., *On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices* (Applied Energy 2024). https://arxiv.org/abs/2404.02722
- Lipiecki, Uniejewski, Weron, *Postprocessing of point predictions ... Isotonic Distributional Regression* (Energy Economics 2024). https://arxiv.org/abs/2404.02270

---

## 5. Family D - Structural / exogenous-feature methods

> Reviewer: electricity-market structural expert. Focus: weather, rooftop PV, structural bid-stack models, WA-specific features. The bet: domain signal beyond lag-30 carries the regime-change information the lag features can't.

### #13. Temperature-conditioned k-NN retrieval over lag-30 curves

**Methodology**: WA loads - and hence willingness-to-pay - are dominated by cooling above ~28 °C in Perth and modest heating in winter. Replace naïve lag-30 lookup with weather-conditioned k-NN: for each test auction `(date d, interval i)`, retrieve the `K` historical auctions at the same interval `i` whose ERA5 `(T_max, dewpoint, GHI)` tuple is closest to the d-day forecast, weighted by recency.

**Model / implementation**: New `WeatherKNNPredictor(CurvePredictor)`. Feature `[T_max, T_min, dewpoint, GHI, hour-sin/cos, weekday, is_holiday]`. `BallTree` with Mahalanobis distance; return the medoid curve. Ingest exogenous via join on `Trading Date`, persisted as `artifacts/exog/era5_perth.pkl`.

**Training data & features**: 64-d interp embedding + 7 weather/calendar scalars at the SWIS load centroid (~32 °S, 116 °E). ERA5 single-levels via Copernicus CDS + BoM Perth Metro station 009225.

**Scoring vs baseline**: Wasserstein. Δ_W vs `Lag1Predictor(lag_days=30)`. Per-interval breakdown to confirm heatwave intervals drive the gain.

**Expected effect / prior**: Demand on hot days has a thicker high-price tail (retailers willing to pay $300+). Expect Δ_W ∈ [-5, -20] *on top* of current -18.95; weakest in mild shoulder seasons.

**References**:
- Coulon & Howison, *Stochastic Behaviour of the Electricity Bid Stack* (2009). https://people.maths.ox.ac.uk/howison/papers/CoulonHowison211008.pdf
- *Seasonal forecasting of hourly electricity demand* (Nature Scientific Reports 2025). https://www.nature.com/articles/s41598-025-91878-0
- Data: ERA5 hourly single-levels. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

---

### #14. Rooftop-PV "duck curve" shift: net-demand decomposition

**Methodology**: 36 % of SWIS premises have rooftop PV; behind-the-meter generation shifts the gross-load → net-procurement relationship. Decompose the forecast into (i) slow underlying demand from lag-30 retrieval, and (ii) a fast PV-displacement delta predicted from clear-sky-index and installed-PV capacity, applied as a horizontal price-axis warp on the bid curve (retailers shift bids down at the high-quantity end on sunny midday intervals).

**Model / implementation**: `PVDecomposedPredictor` wrapping the existing best retrieval: `predicted_curve(p, q) = base_curve(p, q + α · PV_forecast_MW_at_interval_i)`. `α` fitted on train. PV-forecast input = `installed_capacity` (AEMO DER register) × `clear-sky-index` (ERA5 SSRD). Calendar-aware monotonic regression maintains monotonicity.

**Training data & features**: curve embedding + per-interval PV generation forecast (MW). AEMO DER register + ERA5 SSRD + AEMO SWIS rooftop-PV time series.

**Scoring vs baseline**: Wasserstein per interval-of-day. Δ_W vs lag-30. Verify intervals 18-32 (09:00-16:00, duck-belly) improve and that the model doesn't hurt evening ramp (intervals 36-44).

**Expected effect / prior**: Strong on sunny weekday middays, neutral on overcast / weekends. Aggregate ≈ 3-8 W; ≥ 15 on relevant intervals.

**References**:
- Bell et al., *The duck curve in a drying pond: The impact of rooftop PV on the Western Australian electricity market transition* (2021). https://www.sciencedirect.com/science/article/abs/pii/S0957178721000667
- *The impact of rooftop solar on wholesale electricity demand in the Australian NEM* (Frontiers Energy Research 2023). https://www.frontiersin.org/articles/10.3389/fenrg.2023.1197504/full
- Data: AEMO WEM data portal. https://aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem

---

### #15. Structural bid-stack model adapted to demand side (Coulon-Howison inverted)

**Methodology**: Coulon-Howison parametrise the *supply* stack as a monotone function of demand, fuel prices, and stochastic noise. Invert this to the *demand* side: each STEM bid curve is parametrised as `D(p; θ_d)` where `θ_d = (intercept a, slope b, kink p*, tail steepness c)`. Forecast the four-dim θ separately as a 30-day-lag plus exogenous regression. A low-dim structural representation that is interpretable and easy to inject exogenous drivers into.

**Model / implementation**: `StructuralBidCurvePredictor`. (i) Fit `θ_d` per train auction by nonlinear least squares on observed (q, p). (ii) Regress each θ dim on lag-30 θ + weather + macro using XGBoost or Ridge. (iii) Reconstruct curve from forecast θ. Monotonicity by construction. Pinball loss on curve quantiles for fitting; Wasserstein at eval.

**Training data & features**: per-auction θ (4 floats). Exogenous: AEMO operational demand forecast, Western Power transmission outages, ABS monthly industrial production for WA mining, Brent/LNG/coal forward prices (retailer hedging cost), Australian public holidays, WA school terms.

**Scoring vs baseline**: Wasserstein + curve-parameter L² on θ (diagnostic). Δ_W vs lag-30. Also `embedding_l2` on retrieved θ.

**Expected effect / prior**: Aggregate Δ_W comparable to lag-30 (~-5 to 0) but with much better tail-risk on event days (WEM 2023 reform 1 Oct 2023 changed dispatch dynamics - structural models generalise better through regime shifts).

**References**:
- Coulon, *Modelling Price Dynamics through Fundamental Relationships in Electricity and Other Energy Markets* (DPhil, Oxford). https://people.maths.ox.ac.uk/howison/papers/
- Carmona & Coulon, *A Survey of Commodity Markets and Structural Models for Electricity Prices* (2014). https://www.researchgate.net/publication/264890880
- Data: AEMO WEM market data. https://aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem

---

### #16. Monotonic autoencoder + TFT on functional bid curves with rich calendar/weather covariates

**Methodology**: Encode each curve into a 16-d latent `z` via a monotonic autoencoder (positive-weight decoder, cumulative-softmax - guaranteed non-decreasing). Forecast `z` 30 days ahead with Temporal Fusion Transformer using past `z` trajectory plus known-future calendar covariates (public holiday, WA school term, daylight savings - note WA does NOT observe DST, a free regime signal) plus forecast weather. Decode back to a curve. Combines the monotonic-AE structural prior with TFT's strong record for electricity load with mixed covariate types.

**Model / implementation**: PyTorch monotonic-MLP decoder (positive-weights via softplus + cumsum). `pytorch-forecasting` TFT mapping `(past_z[t-90:t], future_known[t+30]) → ẑ[t+30]`. Static: interval-of-day one-hot. Past covariates: realised `z`, weather, holiday flags. Known-future: calendar, weather forecast (ERA5 perfect-foresight first, then ACCESS-G operational for realism).

**Training data & features**: latent `z` (target) + 64-d interp embedding (warm start) + ERA5 + ACCESS-G + AEMO operational demand forecast + ABS WA industrial production + AEMO DER register installed PV.

**Scoring vs baseline**: Wasserstein primary; DTW for shape; embedding-L2 in latent space. Δ_W vs lag-30. Ablation: TFT-no-exog vs TFT-full-exog quantifies encoder-vs-covariate contribution.

**Expected effect / prior**: Highest upside, highest cost in Family D. Recent monotonic-AE work on Italian/Spanish day-ahead curves reports 10-30 % RMSE improvement. Expect Δ_W ∈ [-25, -45] **if** exogenous signal attaches to demand (less certain than supply). Risk: 52k rows is modest for a TFT - heavy regularisation, small `d_model` essential.

**References**:
- Sinha & Lucheroni, *Demand and supply curve forecasting using a monotonic autoencoder* (Applied Energy 2025). https://www.sciencedirect.com/science/article/pii/S0306261925009924
- Lim et al., *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting* (IJF 2021). https://www.sciencedirect.com/science/article/pii/S0169207021000637
- Data: BoM ACCESS-G NWP. https://www.bom.gov.au/nwp/doc/access/NWPData.shtml

---

## 6. Family E - Representation learning, metric learning, neural retrieval

> Reviewer: representation-learning expert. Focus: improving the retrieval embedding itself - either self-supervised, supervised by W-distance, neural retrieval pipelines, or generative Wasserstein autoencoders. The reviewer's strongest prior: **the gap between 26 % and 91 % oracle capture is almost entirely a metric-learning problem**.

### #17. Contrastive step-curve encoder (TS2Vec-style)

**Methodology**: Replace the hand-crafted 64-d quantile vector with a learned encoder trained with NT-Xent on augmented views of the same auction's curve. Two views per anchor via curve-preserving augmentations: (i) random quantity-axis step subsampling, (ii) within-slope price jitter, (iii) high-quantity tail truncation, (iv) Mixup between adjacent intervals of the same day. Positive pair = two views of the same `(date, interval, side)`; negatives = all other curves in the batch. Hierarchical contrasting (TS2Vec) at instance and quantity-bin scales.

**Model / implementation**: 1-D dilated CNN backbone (TS2Vec architecture: 10 residual blocks, channel=64, dilations 1..512), input = step curve resampled at 256 anchor quantities; 2-layer MLP projection head. ~0.6M params. PyTorch; ~8 GPU-h on one A10. Embedding dim 64 (drop-in compatible with existing Ridge-KNN pipeline).

**Training data & features**: ~117k training curves; supervision purely self-supervised. Hard-negative variant: sample within `(interval, season)` bucket so the encoder must capture *shape* differences, not temporal context.

**Scoring vs baseline**: Plug learned embeddings into the existing stratified Ridge-KNN retriever. Primary: Δ_W vs `Lag1Predictor(lag_days=30)`. Secondary: bucket-oracle capture %, retrieval recall@1 vs an oracle-W neighbour.

**Expected effect / prior**: Moderate-positive. Realistic Δ_W ≈ -25 to -35 (35-50 % oracle capture). Risk: augmentations that change W distance more than they preserve "same auction" semantics will hurt.

**References**:
- Yue et al., *TS2Vec: Towards Universal Representation of Time Series* (AAAI 2022). https://arxiv.org/abs/2106.10466
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)* (ICML 2020). https://arxiv.org/abs/2002.05709

---

### #18. Supervised metric learning with curve-Wasserstein as supervision (Family E's strongest prior)

**Methodology**: Train an encoder so that **Euclidean distance in embedding space approximates Wasserstein-1 distance between the underlying curves** - i.e. distill the (expensive, non-differentiable-at-retrieval) W metric into a fast L2 lookup. Triplet/margin objective: for anchor `x_a`, mine triplets `(x_a, x_+, x_-)` where `W(x_a, x_+) < W(x_a, x_-) - m` on raw curves, minimise `max(0, ||f(x_a)-f(x_+)||² - ||f(x_a)-f(x_-)||² + α)`. Add a *regression head* predicting `W(x_a, x_b)` from `|f(x_a) - f(x_b)|` (Khosla-style multi-positive contrastive). This is "Learning Embeddings into Entropic Wasserstein Spaces" with explicit W-supervision.

**Model / implementation**: Set Transformer over (quantity, price) step-points + side flag - naturally handles variable-length curves. 4 ISAB blocks, 4 heads, hidden 128, 32 inducing points; PMA → 64-d. ~1.2M params. ~6 GPU-h. Triplets mined offline in batches of 4096 anchors per epoch.

**Training data & features**: Train pairs sampled within the same `(interval, season)` stratum (where the existing retriever leaves 74 % of the oracle ceiling on the table). Supervision = closed-form W₁ computed once and cached.

**Scoring vs baseline**: (i) Wasserstein after KNN with learned L2 metric - Δ_W vs lag-30. (ii) Bucket-restricted-oracle capture %. (iii) Spearman ρ between L2 in embedding space and ground-truth W on a held-out pair set - the *intrinsic* metric quality.

**Expected effect / prior**: **High - cleanest, lowest-risk direction in Family E.** Oracle ceiling is -73.5; you're at 26 %. The gap is *almost entirely a metric-learning problem*: among curves in the same bucket, L2-on-quantiles is a poor proxy for W-on-curves. Distilling W directly should close most of that. Target Δ_W ≈ -45 to -55 (60-75 % of oracle ceiling).

**References**:
- Khosla et al., *Supervised Contrastive Learning* (NeurIPS 2020). https://arxiv.org/abs/2004.11362
- Frogner et al., *Learning Embeddings into Entropic Wasserstein Spaces* (ICLR 2019). https://openreview.net/forum?id=rJg4J3CqFm
- Lee et al., *Set Transformer* (ICML 2019). https://arxiv.org/abs/1810.00825

---

### #19. Two-tower retriever + cross-encoder reranker (DSSM + reranker)

**Methodology**: Two-tower architecture decouples the query (yesterday's curve + exogenous context - hour bucket, DoW, prior-week prices) from the document (candidate historical curve). Both towers output L2-normalised 64-d embeddings; an ANN index recalls top-K=200 by inner product. A cross-encoder reranker jointly encodes (query context, candidate curve) with full cross-attention to score retrieval fitness; trained with listwise distillation where teacher = true Wasserstein ranking. At inference: ANN top-200 → cross-encoder → top-1.

**Model / implementation**: Query tower = 4-layer MLP on `[interp-64-of-lag30 ∥ calendar ∥ scaled-load-forecast]`. Doc tower = 1-D CNN over candidate. Cross-encoder = small Transformer (4 layers, 4 heads, hidden 128) on `(price, quantity, role ∈ {query, candidate})` tokens. ~3M params. ~12 GPU-h. FAISS for ANN.

**Training data & features**: For each train auction `(d, i)`, query = lag-30 curve + exog; gold doc = actual `(d, i)` curve. Hard negatives = top-K from current Ridge-KNN with worse W than gold. In-batch negatives for the rest. Reranker = softmax over W-ranks among top-K (listwise distillation).

**Scoring vs baseline**: Δ_W vs lag-30. ANN recall@10 / @50 vs W-oracle top-1. Reranker gain ablation.

**Expected effect / prior**: High. The bi-encoder + cross-encoder split is exactly how web-scale retrieval gets both speed and precision. Conditioning the *query tower* on exogenous (current pipeline only weakly uses) is the second multiplier. Target Δ_W ≈ -40 to -60. Risk: reranker overfits regimes; mitigate with time-aware splits in training.

**References**:
- Huang et al., *Learning Deep Structured Semantic Models (DSSM)* (CIKM 2013). https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
- Nogueira et al., *Multi-Stage Document Ranking with BERT* (2019). https://arxiv.org/abs/1910.14424
- Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering* (CVPR 2015). https://arxiv.org/abs/1503.03832

---

### #20. Sinkhorn / Wasserstein autoencoder for curves (OT-aware latent space)

**Methodology**: Encoder-decoder where each curve is encoded to `z ∈ ℝ^64` such that (i) decoder reconstructs the curve under a **Sinkhorn divergence reconstruction loss** (Sinkhorn = differentiable approximation of W₁, the *exact* metric we're evaluated on), and (ii) the aggregated posterior `q(z)` is matched to a prior via Sinkhorn divergence in latent space (Sinkhorn-AE / WAE). The latent geometry is *natively* OT-respecting: small `||z_a - z_b||` implies small `W₁(x_a, x_b)` by construction.

**Model / implementation**: Encoder = Set-Transformer or 1-D CNN. Decoder = autoregressive step-point generator (~128 output steps, monotonicity via softplus increments). Loss = Sinkhorn(x, x̂; ε=0.01, 50 iters) + λ·Sinkhorn(q(z), 𝒩(0, I)). ~1.5M params, ~10 GPU-h. `GeomLoss` library for differentiable Sinkhorn.

**Training data & features**: All training curves, no labels. Conditional variant: feed `(interval, DoW, month)` as conditioning so `z` captures residual shape information beyond seasonal mean.

**Scoring vs baseline**: Two modes. (a) **Retrieval**: replace interp embedding with `z`; Δ_W vs lag-30. (b) **Direct generative prediction**: predict `ẑ_{t+1}` with Ridge from `z_t`, decode, score. Compare both - interesting because retrieval and generation are both natively supported.

**Expected effect / prior**: Moderate. Retrieval-mode gain likely smaller than #18 (the latent optimises global reconstruction not local discrimination), but unlocks the **generative escape from retrieval**: producing curves that are *not* in the historical pool - structurally better than any retrieval when the test distribution drifts (which it did post-2024). Target Δ_W ≈ -25 to -40, with bigger upside if the decoder extrapolates.

**References**:
- Tolstikhin et al., *Wasserstein Auto-Encoders* (ICLR 2018). https://arxiv.org/abs/1711.01558
- Patrini et al., *Sinkhorn AutoEncoders* (UAI 2019). https://www.auai.org/uai2019/proceedings/papers/253.pdf
- Kolouri et al., *Sliced-Wasserstein Autoencoder* (ICLR 2019). https://arxiv.org/abs/1804.01947

---

## 7. Cross-family priority ranking (moderator's synthesis)

If you have to pick **3 to run first**, the moderator's synthesis is:

| Priority | Proposal | Family | Reason |
|---|---|---|---|
| 🥇 | **#18. Supervised metric learning with curve-W as supervision** | E | The diagnostic in [`recompute_2024test/`](../experiments/recompute_2024test/README.md) shouts: 74 % of the oracle ceiling is left on the table because L2-on-quantile-embedding ≠ W-on-curves. Distilling W directly into the metric is the most aligned intervention. Highest prior, moderate cost. |
| 🥈 | **#8. Quantile-curve FPCA in log-derivative space (Ramsay 1998)** | B | The *only* proposal where the training loss is *exactly* the test metric (W₁ = L¹-in-Q⁻¹). Cheap, monotonicity-automatic, drop-in replacement for the current embedding. |
| 🥉 | **#13. Temperature-conditioned k-NN retrieval** | D | Cheapest exogenous test. WA demand is dominated by cooling above 28 °C - if temperature alone moves the needle, that's the prior we should have invested in long ago. Composable with #18 and #19. |

If you also have a 4th slot: **#4 (Chronos / TimesFM zero-shot)** - the cheapest experiment in the whole list. Even a negative result is publishable as a TSFM-on-curves benchmark.

If you have GPU budget for a 5th: **#16 (TFT + monotonic AE with full exog stack)** - highest upside, highest cost, single best chance of a paper-worthy single-experiment result.

## 8. What the experts disagreed about

- **Family A's expert** thinks the bottleneck is the *forecaster* (sequence model on the latent series); **Family E's expert** thinks it's the *representation* (metric learning); **Family B and D's experts** think it's the *function-space formulation* and the *exogenous covariates* respectively. The diagnostic in [`recompute_2024test/`](../experiments/recompute_2024test/README.md) §6 (oracle wins by Δ=-73.5 W with the *same library* the realised model has access to) is evidence that representation/metric (Family E) and function-space formulation (Family B) are the more likely levers than the forecaster (Family A).
- **Family C** uniquely proposes *adding probabilistic forecasts* - even if Δ_W only matches current best, calibrated coverage bands are a deliverable on their own.

## 9. Aggregate reference list

### Demand / supply curve forecasting (electricity-specific)
- Aneiros, Vilar, Raña (2016). *Short-term forecast of daily curves of electricity demand and price*. IJEPES 80. https://doi.org/10.1016/j.ijepes.2015.11.069
- Aneiros, Vilar, Cao, Muñoz San Roque (2013). *Functional prediction for the residual demand in electricity spot markets*. IEEE TPWRS 28(4). https://www.researchgate.net/publication/281149602
- Li, Alonso, Pascual (2025). *Predicting electricity supply and demand curves with functional data techniques*. IJEPES 166. https://doi.org/10.1016/j.ijepes.2025.110561
- Sinha & Lucheroni (2025). *Demand and supply curve forecasting using a monotonic autoencoder...* Applied Energy 397. https://doi.org/10.1016/j.apenergy.2025.126262
- Peláez, Aneiros, Vilar (2024). *Bootstrap prediction regions for daily curves of electricity demand and price using functional data*. arXiv. https://arxiv.org/abs/2401.11885
- Vivó & Alonso (2024). *Prediction of Intraday Electricity Supply Curves*. Applied Sciences 14(22), 10663. https://www.mdpi.com/2076-3417/14/22/10663
- Liebl (2013). *Modeling and forecasting electricity spot prices: A functional data perspective*. Ann. Appl. Stat. https://arxiv.org/pdf/1310.1628
- Vilar, Cao, Aneiros (2012). *Forecasting next-day electricity demand and price using nonparametric functional methods*. IJEPES 39. https://www.sciencedirect.com/science/article/pii/S0142061511002201

### Electricity price / probabilistic forecasting
- Marcjasz, Narajewski, Weron, Ziel (2023). *Distributional neural networks for electricity price forecasting*. Energy Economics. https://arxiv.org/abs/2207.02832
- Lipiecki, Uniejewski, Weron (2024). *Postprocessing of point predictions ... Isotonic Distributional Regression*. Energy Economics. https://arxiv.org/abs/2404.02270
- Uniejewski & Weron (2021). *Regularized quantile regression averaging for probabilistic EPF*. Energy Economics. https://www.sciencedirect.com/science/article/abs/pii/S0140988321000268
- Mastropietro et al. (2024). *On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices*. Applied Energy. https://arxiv.org/abs/2404.02722

### Deep learning for time-series forecasting
- Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*. ICLR. https://arxiv.org/abs/2211.14730
- Liu et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR. https://arxiv.org/abs/2310.06625
- Liu et al. (2024). *Retrieval-Augmented Diffusion Models for Time Series Forecasting*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf
- Tashiro et al. (2021). *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*. NeurIPS. https://arxiv.org/abs/2107.03502
- Rasul et al. (2021). *TimeGrad: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting*. ICML. https://arxiv.org/abs/2101.12072
- Challu et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*. AAAI. https://arxiv.org/abs/2201.12886
- Ansari et al. (2024). *Chronos: Learning the Language of Time Series*. https://arxiv.org/abs/2403.07815
- Hertel et al. (2024). *Benchmarking Time Series Foundation Models for Short-Term Household Electricity Load Forecasting*. https://arxiv.org/abs/2410.09487
- Lim et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. IJF. https://www.sciencedirect.com/science/article/pii/S0169207021000637

### Functional data analysis
- Ramsay (1998). *Estimating smooth monotone functions*. JRSS B 60. https://doi.org/10.1111/1467-9868.00130
- Shin, Zhou, Ding (2022). *Joint estimation of monotone curves via functional principal component analysis*. CSDA 166. https://doi.org/10.1016/j.csda.2021.107343

### Probabilistic / quantile forecasting
- Dabney, Ostrovski, Silver, Munos (2018). *Implicit Quantile Networks for Distributional RL*. ICML. https://arxiv.org/abs/1806.06923
- Gouttes et al. (2021). *Probabilistic Time Series Forecasting with Implicit Quantile Networks*. https://arxiv.org/abs/2107.03743
- Smyl et al. (2024). *Any-Quantile Probabilistic Forecasting of Short-Term Electricity Demand*. https://arxiv.org/abs/2404.17451
- Duan et al. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*. ICML. https://arxiv.org/abs/1910.03225

### Representation / metric learning
- Yue et al. (2022). *TS2Vec: Towards Universal Representation of Time Series*. AAAI. https://arxiv.org/abs/2106.10466
- Tonekaboni et al. (2021). *Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding*. ICLR. https://arxiv.org/abs/2106.00750
- Chen et al. (2020). *SimCLR*. ICML. https://arxiv.org/abs/2002.05709
- Khosla et al. (2020). *Supervised Contrastive Learning*. NeurIPS. https://arxiv.org/abs/2004.11362
- Frogner et al. (2019). *Learning Embeddings into Entropic Wasserstein Spaces*. ICLR. https://openreview.net/forum?id=rJg4J3CqFm
- Schroff et al. (2015). *FaceNet (triplet loss)*. CVPR. https://arxiv.org/abs/1503.03832
- Lee et al. (2019). *Set Transformer*. ICML. https://arxiv.org/abs/1810.00825
- Zaheer et al. (2017). *Deep Sets*. NeurIPS. https://arxiv.org/abs/1703.06114
- Huang et al. (2013). *Learning Deep Structured Semantic Models (DSSM)*. CIKM. https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
- Nogueira et al. (2019). *Multi-Stage Document Ranking with BERT*. https://arxiv.org/abs/1910.14424
- Tolstikhin et al. (2018). *Wasserstein Auto-Encoders*. ICLR. https://arxiv.org/abs/1711.01558
- Patrini et al. (2019). *Sinkhorn AutoEncoders*. UAI. https://www.auai.org/uai2019/proceedings/papers/253.pdf
- Kolouri et al. (2019). *Sliced-Wasserstein Autoencoder*. ICLR. https://arxiv.org/abs/1804.01947

### Structural / market / electricity-specific exogenous
- Coulon & Howison (2009). *Stochastic Behaviour of the Electricity Bid Stack*. https://people.maths.ox.ac.uk/howison/papers/CoulonHowison211008.pdf
- Carmona & Coulon (2014). *A Survey of Commodity Markets and Structural Models for Electricity Prices*. https://www.researchgate.net/publication/264890880
- Bell et al. (2021). *The duck curve in a drying pond: The impact of rooftop PV on the Western Australian electricity market transition*. https://www.sciencedirect.com/science/article/abs/pii/S0957178721000667
- *The impact of rooftop solar on wholesale electricity demand in the Australian NEM* (Frontiers Energy Research 2023). https://www.frontiersin.org/articles/10.3389/fenrg.2023.1197504/full
- *Seasonal forecasting of hourly electricity demand* (Nature Scientific Reports 2025). https://www.nature.com/articles/s41598-025-91878-0

### Data sources (for Family D's proposals)
- AEMO WEM market data (STEM curves, DER register, demand forecasts). https://aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem
- AEMO WA Reform Program (WEM 2023 reform details). https://www.aemo.com.au/initiatives/major-programs/wa-reform-program/about-the-wa-reform-program
- BoM Climate Data Online (Perth observations, ACORN-SAT temperature). https://www.bom.gov.au/climate/data/
- BoM ACCESS-G NWP (operational weather forecasts). https://www.bom.gov.au/nwp/doc/access/NWPData.shtml
- ERA5 hourly single-levels (Copernicus reanalysis). https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- WA Education department (school terms). https://www.education.wa.edu.au/future-students/calendars
- WA Commerce department (public holidays). https://www.commerce.wa.gov.au/labour-relations/public-holidays-western-australia
- ABS Industrial Production (WA macro drivers). https://www.abs.gov.au/statistics/industry
