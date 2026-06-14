# Experiment proposals - twenty paths to beat lag-30 on demand curves

Status: this plan was carried out, see the experiment folders under [`experiments/`](../experiments/).

**Last updated**: 2026-05-12. Synthesised from a five-expert literature board; full citations in [`research_proposals_20.md`](research_proposals_20.md) and [`papers.md`](papers.md).

## 0. Where we are

- Task: forecast the 30-day-ahead demand step-curve of each WA STEM 30-min auction.
- Train: 52 560 demand auctions (2021-01 → 2023-12). Test: 17 568 auctions (2024-01 → 2024-12); see [`experiments/recompute_2024test/`](../experiments/recompute_2024test/).
- Baseline: `Lag1Predictor(lag_days=30)`. Mean W = 112.43 on test.
- Current best (deployable): `B1 rolling-refit Ridge`, mean W = 93.48, **Δ_W = -18.95**, win-rate 58.0 %, capture-of-bucket-oracle gap = 25.8 %.
- Static bucket oracle (NOT deployable): mean W = 38.92, Δ_W = -73.51, win-rate 84.1 % - the reference ceiling.

There is ~74 percentage-points of oracle-capture room left on the table. Every realised model so far gets stuck around the L2-on-quantile retrieval signal, which is a poor proxy for the Wasserstein test metric.

---

## 1. The twenty proposals at a glance

| # | Title | Family | Predicted Δ_W | Cost | Risk |
|---|---|---|---:|---|---|
| 1 | Monotonic AE + PatchTST on bucket-conditional latents | DL sequence | -20 to -30 | 2-4 GPU-h | medium |
| 2 | iTransformer over 64-d embedding as 64-variate series | DL sequence | -15 to -25 | 1-2 GPU-h | medium |
| 3 | Retrieval-augmented diffusion (RATD) on curve embeddings | DL sequence | -15 to -25 + CRPS | 6-10 GPU-h | high |
| 4 | Chronos / TimesFM zero-shot per coordinate | DL sequence | -5 to +5 | minutes (zero-shot) | low |
| 5 | FoF regression + closest-curve snap (Li-Alonso-Pascual) | FDA | -19 to -25 | minutes | low |
| 6 | FAR(1) on inverse-demand curves (Aneiros-Vilar-Cao) | FDA | -18 to -25 | ~1 h | low |
| 7 | Monotonic AE + Echo-State (Sinha-Lucheroni) | FDA | -20 to -30 | ~2 h | medium |
| 8 | FPCA on log-derivative Q-space (Ramsay 1998) | FDA | -20 to -35 | minutes | low |
| 9 | IQN over inverse-CDF | Probabilistic | -5 to -15 + CRPS | ~30 min GPU | low |
| 10 | DDNN-JSU per τ-slice (Marcjasz et al.) | Probabilistic | -8 to -15 + CRPS | ~15 min/seed | low |
| 11 | NGBoost / QRF per τ-slice | Probabilistic | -12 to -22 | ~1 h CPU | low |
| 12 | Conformalised QRA over (retrieval, DDNN, NGBoost) | Probabilistic | -22 to -30 + bands | medium (chained) | low |
| 13 | Temperature-conditioned k-NN retrieval | Structural | -5 to -20 on top | ~30 min | low |
| 14 | Rooftop-PV "duck curve" decomposition | Structural | +3 to +8 (intervals) | medium (data pipeline) | medium |
| 15 | Structural bid-stack model (Coulon-Howison inverted) | Structural | -5 to 0 (better tail) | medium | high |
| 16 | Monotonic AE + TFT with rich exog | Structural | -25 to -45 | 10-20 GPU-h | high |
| 17 | Contrastive TS2Vec encoder | Representation | -25 to -35 | ~8 GPU-h | medium |
| 18 | Supervised W-distillation metric learning | Representation | **-45 to -55** | ~6 GPU-h | low |
| 19 | Two-tower retriever + cross-encoder reranker (DSSM) | Representation | -40 to -60 | ~12 GPU-h | medium |
| 20 | Sinkhorn / Wasserstein autoencoder | Representation | -25 to -40 | ~10 GPU-h | medium |

---

## 2. Full proposal texts

### Family A - Deep learning sequence models on the embedding

#### #1. Monotonic AE + PatchTST on bucket-conditional latents
Train a monotonic autoencoder (Sinha-Lucheroni 2025) mapping each demand curve to an 8-16-d latent. Forecast `z_{t+30}` per `(interval, weekday)` bucket with PatchTST conditioned on lag-30 / lag-7 / lag-1 latents. Decode through training-set kNN to keep the "real-curve" property.

- **Implementation**: PyTorch MAE + `neuralforecast` PatchTST. ~250k-2M params. 2-4 GPU-h.
- **Scoring**: Wasserstein and DTW vs `Lag1Predictor(lag_days=30)`. Cluster-bootstrap CI by Trading Date.
- **Refs**: Sinha & Lucheroni (2025); Nie et al., PatchTST (ICLR 2023).

#### #2. iTransformer over the 64-d embedding as a 64-variate series
Treat the 64-d interp embedding as a 64-variate time series indexed by trading day, per bucket. iTransformer tokenises each *variate* so attention learns cross-quantile dependencies. Predict the 64-d vector 30 days ahead, snap to nearest training curve via L2.

- **Implementation**: `neuralforecast` iTransformer, `d_model=128`, ~600k params, single shared model with bucket-id token. 1-2 GPU-h.
- **Scoring**: W + DTW vs lag-30; paired bootstrap CI.
- **Refs**: Liu et al., iTransformer (ICLR 2024); Hertel et al. (2024).

#### #3. Retrieval-augmented diffusion (RATD) over curve embeddings
Conditional diffusion model (CSDI / TimeGrad lineage with RATD) generating t+30 curve embedding conditioned on bucket id, recent lags, and k=5 retrieved historical embeddings. Sample N=64 curves; aggregate by medoid (or take mean for CRPS).

- **Implementation**: Custom PyTorch 1-D U-Net denoiser + Reference-Modulated Attention. ~1-3M params, 200 diffusion steps with DDIM. 6-10 GPU-h.
- **Scoring**: W (medoid), CRPS (probabilistic - new), DTW. Δ_W vs lag-30.
- **Refs**: Liu et al., RATD (NeurIPS 2024); Rasul et al., TimeGrad (ICML 2021); Tashiro et al., CSDI (NeurIPS 2021).

#### #4. Chronos / TimesFM zero-shot per embedding coordinate
For each of 64 embedding coords, run pretrained TSFM (Chronos-Bolt 200M, TimesFM 2.0) zero-shot per bucket. Reassemble into a curve embedding; project to nearest real curve. Optional LoRA-finetune.

- **Implementation**: `chronos-forecasting`, `timesfm` via HuggingFace. CPU minutes per bucket zero-shot; LoRA ~30 min/GPU.
- **Scoring**: Wasserstein vs lag-30. Zero-shot vs LoRA comparison.
- **Refs**: Ansari et al., Chronos (2024); Meyer et al. (2026).

---

### Family B - Functional data analysis

#### #5. Function-on-function regression + closest-curve snap (Li-Alonso-Pascual replication)
Project each curve onto FPCA basis (8-15 components). Regress lag-30/7/1 scores + calendar/weather on next scores. Snap the smooth functional forecast to the nearest historical curve in functional L².

- **Implementation**: `scikit-fda` (`FPCA`, `FunctionalLinearRegression`) or `fda.usc` (R). Medoid retrieval over FPCA scores. Minutes per fit; monthly rolling feasible.
- **Scoring**: Wasserstein primary; capture-of-oracle-gap. Functional L² secondary.
- **Refs**: Li, Alonso, Pascual (IJEPES 2025); Aneiros, Vilar, Raña (IJEPES 2016).

#### #6. Functional AR(1) on inverse-demand curves (Aneiros-Vilar-Cao)
Model the curve time series per interval as a functional AR(1): `X_{t+30} = Ψ(X_t) + ε`. Ψ via FPCA-score regression or Kargin-Onatski. 48 interval-specific FARs. Optionally invert demand → inverse demand so response is on `[0, Q_max]`.

- **Implementation**: Python `scikit-fda` + closed-form Ψ; or `fda.usc::fregre.basis`. ~1 h to fit 48 FARs.
- **Scoring**: Wasserstein primary; functional L² and step-aware DTW as diagnostics.
- **Refs**: Aneiros, Vilar, Cao, Muñoz San Roque (IEEE TPWRS 2013); Peláez et al. (arXiv 2024).

#### #7. Monotonic AE + Echo-State (Sinha-Lucheroni style)
Decompose curve into anchors `A_t`, `B_t`, interior monotone `C_t`. Train monotonic AE on `C_t` (positive-weight / cumulative-softmax decoder). Forecast `A_t`, `B_t` and the latent of `C_t` with two Echo-State Networks. Recompose to guaranteed-monotone forecast.

- **Implementation**: Monotonic AE (~2k params, 16-d bottleneck) + `pyESN`/`reservoirpy`. ~2 h.
- **Scoring**: Wasserstein primary; monotonicity-violation rate (should be 0). High-volatility-day breakdown.
- **Refs**: Sinha & Lucheroni (Applied Energy 2025); Shin, Zhou, Ding (CSDA 2022).

#### #8. FPCA on log-derivative Q-space (Ramsay 1998 trick)
Re-interpret the cumulative-quantity curve as a quantile/inverse-CDF `Q_t(u)`. Take `log Q'_t(u)` (unconstrained), apply FPCA, run Ridge on scores, then exponentiate-and-integrate. **Key insight**: W₁ between curves ≈ L¹ between inverse-CDFs - so this is implicitly Wasserstein-aware.

- **Implementation**: `scikit-fda` FPCA on log-derivatives + scalar Ridge. Minutes total; rolling-refit feasible.
- **Scoring**: Wasserstein primary - **loss-metric alignment is exact**. Embedding-L2 in score space as diagnostic.
- **Refs**: Ramsay (JRSS B 1998); Vivó & Alonso (Applied Sciences 2024).

---

### Family C - Probabilistic / distributional forecasting

#### #9. Implicit Quantile Network on inverse-CDF
Treat the curve as `Q(τ)`. Train IQN conditioned on context to map sampled `τ → predicted price`. Loss = pinball (an unbiased CRPS estimator). Reconstruct curve on the true `τ` grid with non-crossing sort.

- **Implementation**: PyTorch MLP (3-4 layers, 256 units) on `[context ⊕ cos/sin(πτk)]`. Optional λ·max(0, ∂price/∂τ < 0)². ~30 min GPU.
- **Scoring**: Wasserstein + CRPS averaged over τ. Δ_W vs lag-30.
- **Refs**: Gouttes et al. (2021); Smyl et al., Any-Quantile (2024).

#### #10. DDNN with Johnson-SU output per τ-slice (Marcjasz et al.)
For each of K∈{32,64} τ-slices, output the 4 JSU parameters (handles heavy tails). Loss = sum of CRPS_JSU (closed form). Post-isotonise across τ.

- **Implementation**: PyTorch DDNN with 4·K outputs. ~200k params, ~15 min/seed, ensemble of 4 seeds.
- **Scoring**: Wasserstein + per-slice CRPS + 90 % Winkler / coverage.
- **Refs**: Marcjasz et al. (Energy Economics 2023); Lipiecki, Uniejewski, Weron (Energy Economics 2024).

#### #11. NGBoost / QRF per τ-slice
Tabular boosting outputting parametric distribution per slice. Or `quantile-forest` with `τ` as input (any-quantile multi-output). Strong tabular baseline of GEFCom / EPF benchmarks.

- **Implementation**: `ngboost` (Normal, 500 estimators, depth 4) per slice OR multi-output joint. ~1 h CPU total.
- **Scoring**: Wasserstein + CRPS. Δ_W vs lag-30.
- **Refs**: Duan et al., NGBoost (ICML 2020); Uniejewski & Weron, regularised QRA (Energy Economics 2021).

#### #12. Conformalised retrieval + QRA ensemble
Take B1 retrieval as median. Combine retrieval, DDNN (#10), NGBoost (#11) via regularised QRA (Uniejewski-Weron). Add SPCI/EnbPI for τ-wise calibrated bands.

- **Implementation**: `mapie` or custom SPCI (~200 LoC). `LassoQuantileRegressor` for combination. Cheap once base models exist.
- **Scoring**: W on ensemble median + coverage + Winkler. Δ_W vs lag-30.
- **Refs**: Mastropietro et al. (Applied Energy 2024); Lipiecki et al. (Energy Economics 2024).

---

### Family D - Structural / exogenous features

#### #13. Temperature-conditioned k-NN retrieval
Replace naïve lag-30 retrieval with weather-conditioned k-NN: retrieve historical same-interval auctions whose (T_max, dewpoint, GHI) tuple is closest. WA demand is cooling-dominated above 28 °C.

- **Implementation**: `WeatherKNNPredictor(CurvePredictor)`. `BallTree` Mahalanobis. ERA5 + BoM Perth station 009225.
- **Scoring**: Wasserstein + per-interval breakdown (heatwave intervals should drive the gain). Δ_W *on top of* current -18.95.
- **Refs**: Coulon & Howison (2009); *Seasonal forecasting of hourly electricity demand* (Nature Sci. Reports 2025).
- **Data**: ERA5 single-levels (Copernicus CDS); BoM Climate Data Online.

#### #14. Rooftop-PV "duck curve" decomposition
Decompose forecast into (i) lag-30 retrieval base + (ii) PV-displacement delta = `α · (installed_capacity × clear_sky_index)`, applied as horizontal quantity-axis warp. 36 % of SWIS premises have rooftop PV.

- **Implementation**: `PVDecomposedPredictor` wrapping B1. α fitted on train.
- **Scoring**: W per interval-of-day. Verify intervals 18-32 (duck-belly) improve, evening-ramp 36-44 doesn't degrade.
- **Refs**: Bell et al. (2021); *Rooftop solar on wholesale demand in NEM* (Frontiers 2023).
- **Data**: AEMO DER register; AEMO SWIS rooftop-PV time series.

#### #15. Structural bid-stack model (Coulon-Howison inverted to demand)
Parametrise each demand curve as `D(p; θ) = (intercept, slope, kink, tail-steepness)` (4 floats). Fit θ per train auction via NLS; forecast each θ-dim by lag-30 θ + exog (weather, AEMO operational demand forecast, WA industrial production, fuel prices, holidays/school terms). Reconstruct curve from forecast θ.

- **Implementation**: `StructuralBidCurvePredictor`. NLS fit + XGBoost on θ. Monotonicity by construction.
- **Scoring**: W + curve-parameter L² on θ. Particular attention to regime-shift days (WEM 2023 reform).
- **Refs**: Coulon (DPhil thesis); Carmona & Coulon (2014).
- **Data**: AEMO WEM, ABS WA industrial production.

#### #16. Monotonic AE + TFT with rich calendar / weather exog
Encode each curve to 16-d latent via monotonic AE. Forecast `z` 30 days ahead with Temporal Fusion Transformer using past `z` + known-future calendar covariates (holidays, school terms, NO DST in WA) + forecast weather (ERA5 / ACCESS-G).

- **Implementation**: PyTorch monotonic-MLP decoder + `pytorch-forecasting` TFT. Static: interval one-hot. Past: realised `z`, weather, holiday flags. Known-future: calendar + weather forecast.
- **Scoring**: W primary; DTW shape; embedding-L2. Δ_W vs lag-30. Exog ablation.
- **Refs**: Sinha & Lucheroni (Applied Energy 2025); Lim et al., TFT (IJF 2021).
- **Data**: BoM ACCESS-G NWP, ERA5, AEMO operational demand forecast.

---

### Family E - Representation / metric learning

#### #17. Contrastive step-curve encoder (TS2Vec)
Replace hand-crafted 64-d quantile vector with a learned encoder trained with NT-Xent on augmented curve views (step subsampling, in-slope price jitter, tail truncation, intra-day Mixup). Hierarchical contrasting at instance and quantity-bin scales.

- **Implementation**: TS2Vec backbone (10 residual blocks, channel=64) + MLP head. ~0.6M params. ~8 GPU-h on one A10. Output 64-d (drop-in compatible).
- **Scoring**: Plug into existing Ridge-KNN retriever. Δ_W vs lag-30; bucket-oracle capture %; retrieval recall@1.
- **Refs**: Yue et al., TS2Vec (AAAI 2022); Chen et al., SimCLR (ICML 2020).

#### #18. Supervised W-distillation metric learning
Train an encoder so `||f(x_a) - f(x_b)||` ≈ `W₁(x_a, x_b)`. Triplet/margin loss with triplets mined under W; regression head predicts `W` from `|f(x_a) - f(x_b)|`.

- **Implementation**: Set Transformer over (qty, price, side) point set. ~1.2M params. ~6 GPU-h.
- **Scoring**: (i) Δ_W after KNN with learned L2 metric vs lag-30. (ii) Spearman ρ between learned-L2 and true-W on held-out pairs - the intrinsic metric quality.
- **Refs**: Khosla et al., Sup-Contrastive (NeurIPS 2020); Frogner et al., Entropic-W Embeddings (ICLR 2019); Lee et al., Set Transformer (ICML 2019).

#### #19. Two-tower retriever + cross-encoder reranker (DSSM)
Query tower encodes lag-30 curve + exog → 64-d; doc tower encodes candidate curve → 64-d. ANN top-200 → cross-encoder Transformer over joint tokens reranks with listwise W-distillation loss.

- **Implementation**: ~3M params, FAISS for ANN. ~12 GPU-h.
- **Scoring**: Δ_W vs lag-30; ANN recall@10 / @50; reranker-gain ablation.
- **Refs**: Huang et al., DSSM (CIKM 2013); Nogueira et al., Multi-stage ranking (2019); Schroff et al., FaceNet (CVPR 2015).

#### #20. Sinkhorn / Wasserstein autoencoder
Encoder-decoder where reconstruction loss = Sinkhorn divergence (differentiable W₁ approximation) and aggregated posterior matched to a prior via Sinkhorn. Latent geometry is OT-respecting by construction.

- **Implementation**: Set-Transformer encoder + monotone-step decoder. `GeomLoss` for Sinkhorn. ~1.5M params, ~10 GPU-h.
- **Scoring**: Two modes - (a) **retrieval** replace interp emb with `z`; (b) **generative** predict `ẑ_{t+1}` with Ridge from `z_t`, decode. Compare both. Δ_W vs lag-30.
- **Refs**: Tolstikhin et al., WAE (ICLR 2018); Patrini et al., Sinkhorn AE (UAI 2019); Kolouri et al., Sliced-W AE (ICLR 2019).

---

## 3. References

See [`papers.md`](papers.md) for URLs and 1-line summaries of every paper cited. See [`research_proposals_20.md`](research_proposals_20.md) for the longer expert-review write-up with priors.

## 4. Execution order - moderation outcome (2026-05-12)

A second board of five critics - *cost pragmatist · metric-alignment theorist · failure-mode pessimist · replication-first conservative · portfolio strategist* - debated the literature board's prior recommendation (#18 → #8 → #13 → #4 → #16). Each critic was given the 20 proposals plus the repo's track record of past failures and asked to challenge the prior and propose their own top-5.

### Tally (votes across the 5 critics' top-5 lists)

| Proposal | Endorsements |
|---|---:|
| #18 W-distillation metric learning | 4/5 |
| #8 FPCA on log-derivative Q-space | 3/5 |
| #5 FoF + closest-curve snap (Li-Alonso-Pascual replication) | 3/5 |
| #11 NGBoost / QRF per τ-slice | 2/5 |
| #13 Temperature-conditioned k-NN | 2/5 |
| Each of #1, #6, #7, #9, #10, #12, #15, #20 | 1/5 |
| **#4 Chronos / TimesFM zero-shot** | **0/5 (dropped)** |
| **#16 TFT + exog** | **0/5 (dropped)** |

### Major debates the moderator resolved

1. **Should #18 lead or trail?** Pragmatist and portfolio strategist want it early; theorist (Critic 2) wants it last because it is mathematically capped at the bucket oracle (it can't outperform `KNN(oracle_query)`); failure-mode pessimist (Critic 3) recalls that `A_distance_w` - a similar GBM-on-W-labels - *lost* on the 2024-only window (Δ=+4.20). **Verdict**: keep #18 in top 5, place it *after* cheap baselines so we can interpret its result against published benchmarks, not just against lag-1.

2. **Should we drop #4 (Chronos) and #16 (TFT)?** Yes - 0/5 endorsements. #4's foundation models were trained on natural time series, not curve-coordinates-per-quantile. #16's 10-20 GPU-h × WEM-2023-reform-in-training-data risk is too steep.

3. **Should we add a replication?** Yes - Critic 4's argument is strong: Sinha-Lucheroni 2025 and Li-Alonso-Pascual 2025 publish concrete Δ-magnitudes on directly analogous markets. **#5 (Li-Alonso-Pascual replication)** is the cheapest such pick; **#7 (Sinha-Lucheroni MAE+ESN replication)** is the runner-up - keep it as next-tier.

4. **Should we add a probabilistic experiment?** Yes - Family C was absent from the prior 5. Probabilistic forecasts are a *different deliverable type* (calibrated bands, CRPS, Winkler). **#11 (NGBoost/QRF)** is the cheapest Family C entry and unlocks #12 (conformalised QRA) as a follow-up.

5. **Should we add a regime-robustness experiment?** Yes - `recompute_2024test/` explicitly exists *because* of the 2025-01 regime change. **#13 (temperature kNN)** ties the prediction to ERA5 physics; even if its raw Δ is the smallest, it's the only one whose Δ should be robust across regime shifts.

6. **Critic 3's parking-lot picks aren't in the original 20** (B1 + residual hybrid, rolling-refit of A, April-2024 diagnostic). The moderator notes these as the *true* top-prior follow-ups - they directly target the diagnosed bottleneck - but keeps them in a separate parking-lot section since they aren't proposals from the literature board.

### Final recommended execution order (revised, post-debate)

1. **#5 - FoF + closest-curve snap (Li-Alonso-Pascual 2025 replication)** - ~30 min. Endorsed by 3 critics. Published Δ-magnitude on OMIE; lowest-risk first move. Provides a defensible baseline for everything that follows.
2. **#8 - FPCA on log-derivative Q-space (Ramsay 1998)** - ~30 min CPU. Endorsed by 3 critics. Only proposal whose training loss = test metric (W₁ ≈ L¹-in-inverse-CDF). Caveat per failure-mode critic: flat-segment handling needs care.
3. **#11 - NGBoost / QRF per τ-slice** - ~1 h CPU. Endorsed by 2 critics. Opens the probabilistic family (Family C) at minimal cost; provides the base learner for a later conformalised-QRA ensemble (#12); adds CRPS as a new metric.
4. **#13 - Temperature-conditioned k-NN retrieval** - ~30 min. Endorsed by 2 critics. Layered *on top of* B1 (additive). Provides regime-robustness insurance against another 2025-01-style break.
5. **#18 - Supervised W-distillation metric learning** - ~6 GPU-h. Endorsed by 4 critics. The highest-upside novel bet; now defensible because we have 3 cheap reference points (#5, #8, #11) to interpret it against. Acknowledged ceiling: capped at the bucket oracle Δ_W = -73.5.

**Estimated total cost: ~8 GPU-h + a few CPU-h** - under the prior recommendation's budget (which had #16 alone at 10-20 GPU-h).

### Dropped from the prior recommendation
- **#4 Chronos / TimesFM zero-shot** - 0/5 endorsements. Curve-coordinates-per-quantile aren't natural time series; no monotonicity guarantee after coordinate-wise forecast.
- **#16 Monotonic AE + TFT with rich exog** - 0/5 endorsements. 10-20 GPU-h with overfit risk on 52k rows; WEM 2023 reform contaminates training.

### Next-tier (run if first 5 produce signal)
- **#7 Monotonic AE + Echo-State (Sinha-Lucheroni replication)** - second replication. Run if #5 confirms FDA family is viable.
- **#12 Conformalised QRA ensemble** - natural follow-up to #11 if probabilistic looks alive.
- **#1 Monotonic AE + PatchTST** - predictor-diversity probe; run if #18 metric distillation succeeds but the predictor side is the bottleneck.

### Parking lot (Critic 3's diagnosis-driven picks; not in the original 20)
These directly target the diagnosed bottlenecks from [`experiments/recompute_2024test/README.md`](../experiments/recompute_2024test/README.md):

- **Hybrid B1 + residual correction** - use B1 retrieval to nudge the lag-30 staircase rather than replace it. Plausibly wins *both* W and DTW simultaneously (the only candidate that could).
- **Rolling-refit of A (distance-ranking)** - combine the A distance ranker with B1's monthly refit. Tests whether rolling refit fixes the lockbox problem when the scorer is W-aware.
- **April-2024 diagnostic** - B1's one losing month (Δ=+18). Cheap focused investigation; if it's Easter-cluster / participant-cluster, a holiday-flag exog feature might close it.

---

## Appendix - Historical proposals A-F (2025)

The previous round of proposals (A-F) was focused on the supply→demand pivot and is now archived:

- **A** Spanish-framework replication on demand curves - **done**, became `dtw_kmedoids_demand/`.
- **B** K=72 supply replication - done as part of `dtw_kmedoids/`.
- **C** Participant-level forecasting - not run.
- **D** Bid-submission-timing exogenous feature - not run.
- **E** Decision-relevant evaluation (P&L simulator) - not run.
- **F** Probabilistic curve forecasting with conformal coverage - superseded by Proposals #9, #10, #12.
