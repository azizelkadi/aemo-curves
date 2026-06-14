# Area / probabilistic-band forecasting on demand curves

**Date**: 2026-05-13. Synthesised from a four-expert board (probabilistic-band specialist · decision-theorist · generative-model specialist · conformal-prediction specialist) debating the user's question: *"what about predicting an area instead of a curve? a more probabilistic approach"*.

---

## 0. Where the question comes from

The current realised W champion is `Exp5 FoF + closest-curve snap`, Δ_W=-19.23 on 2024-only demand test. The bucket-oracle ceiling is Δ_W=-73.51 - about **74 percentage points are still unrecovered**. The diagnosis from [`experiments/proposals_top5/README.md`](../experiments/proposals_top5/README.md) is that **the query predictor (not the metric, not the representation) is the bottleneck** - confirmed by the W-distillation result where Spearman ρ(learned-L2, true-W) = 0.992 still produced a *losing* model.

The user is asking whether shifting target from "single curve" to "probabilistic area" might:
(a) sidestep the metric mismatch that killed `Exp #11 per-slice HistGBM` (Δ_W=+47.86), and
(b) deliver a *separable publishable contribution* - calibrated bands + coverage + Winkler - that doesn't require beating Exp5 on point-Δ_W.

---

## 1. Moderator's verdict (one paragraph)

**Yes, the area-prediction reframe is worth pursuing - as an *additive* track, not a pivot.** Exp5's point-forecast result (Δ_W=-19.23, 26.2 % oracle capture) stands as a chapter. Adding a calibrated-band chapter is a *new capability*, not an incremental Δ_W win, and it cannot be dismissed as "you pivoted because point was stuck" if it sits on top of Exp5 as a wrapper. There is a clear staged execution: start with **conformal-wrapped Exp5** (minutes of CPU), then **salvage #11 as a per-slice quantile band**, then **DDNN-JSU per τ** (Marcjasz et al. 2023 experiment). Generative architectures (diffusion, Sinkhorn-WAE) are explicitly *deferred* - they are 6-10 GPU-h each and only justified if the cheap wrappers plateau. One critic (decision-theorist) raised an orthogonal point worth flagging: 3 of 4 plausible downstream users actually want a *scalar* probabilistic forecast (e.g. predicted clearing price ± uncertainty at the retailer's load), not a curve band - this is a separable research direction kept here as a parking-lot item.

---

## 2. What "predicting an area" means here

Three operational definitions emerged across the board:

| Definition | Output object | Strengths | Weaknesses |
|---|---|---|---|
| **Marginal per-τ band** | Lower and upper price curves; at each τ-slice the interval [p_lo(τ), p_hi(τ)] | Cheap to produce; per-slice quantile regression; isotonisable to monotone | Does *not* guarantee the true curve lives inside the band jointly - under-covers for joint use |
| **Joint band from N curve samples** | A bag of plausible curves; band = envelope (e.g. 5/95-th-percentile per τ) | Joint coverage; flexible α post-hoc; allows energy-score / Sinkhorn evaluation | Costs a generative model; samples must be valid staircases |
| **W-ball around the point forecast** | The set of curves within Wasserstein distance r of ŷ | Coverage stated *in* the eval metric; theoretically clean | Hard to visualise as a price-axis band; less decision-relevant |

The board consensus is to **start with marginal-per-τ + isotonisation + conformal recalibration to recover joint coverage**, and only escalate to joint-sample bands if marginal bands prove decision-irrelevant for the use case.

---

## 3. The scoring metrics (board consensus)

For every band experiment, report the same four numbers:

| Metric | What it measures | Why |
|---|---|---|
| **Mean Winkler-α** (α ∈ {0.5, 0.8, 0.9}) | Combined sharpness + coverage penalty | Standard proper scoring rule for interval forecasts (Gneiting & Raftery 2007); used as the EPF probabilistic headline (Marcjasz et al. 2023). **Primary headline.** |
| **Empirical coverage @ α** | Fraction of test rows where the true curve lies inside the band | Calibration diagnostic - required alongside Winkler so trivial 100 %-cover bands don't game the headline |
| **Mean / median band width** | Sharpness diagnostic | Pair with coverage; without it Winkler is opaque |
| **Median-curve Δ_W vs lag-30** | The band's *median* curve scored under the existing point metric | Back-compatibility with every other table in [`experiments/README.md`](../experiments/README.md) - lets band methods sit on the same leaderboard as Exp5 |

Optional secondary score:
- **Per-quantile pinball loss + isotonisation** - already what NGBoost / DDNN are trained on.
- **Energy score** (Gneiting & Raftery) on joint samples - only for the sample-based proposals.

---

## 4. The probabilistic baseline

The current `Lag1Predictor(lag_days=30)` is a *single curve*, not a band. For a fair Winkler comparison, the baseline must also be probabilistic. Three options the user listed:

| Option | Recommendation |
|---|---|
| (a) **lag-30 ± k·σ from rolling-window residuals** per (interval, dow, τ-slice) | **✓ Use this.** Simplest, most defensible, faithful to "what would a practitioner do". σ from rolling 60-day window of past residuals at the same (interval, dow). Compute once and treat as the band-equivalent of lag-30. |
| (b) Bootstrap-band from k same-weekday past curves | Valid but introduces nuisance choices (k, weighting); obscures comparisons |
| (c) Marcjasz DDNN-JSU output as baseline | Makes the comparison vacuous since DDNN-JSU *is* one of the proposals |

**Probabilistic baseline = lag-30 ± rolling-σ per (interval, dow), evaluated identically to the model.**

---

## 5. Disagreement worth flagging - the decision-theoretic critique

Critic 2 (decision-theorist) argued that **the curve isn't even the right target for 3 of 4 plausible downstream users**:

| User | Decision | Needs |
|---|---|---|
| Retailer hedging load cost | Buy a cap on cleared price at known load Q | A **band on the scalar cleared price**, not on the curve |
| Aggregator dispatching DER fleet | Schedule charge/discharge by interval | A **band on the scalar cleared price**, optionally with intra-day correlation |
| Market design / regulatory monitoring | Anomaly / market-power detection | Curve + credible region (band-on-curve genuinely useful) |
| Generator bid-stack positioning | "If I bid $X at Q, what is P(clear)?" | **Joint curve sampling** - marginal bands cannot answer the AND-of-events |

This implies a *parallel* research direction worth scoping: a **scalar-cleared-price probabilistic forecast** (NGBoost / DDNN on price alone, conditioned on the same lag features) as a separate deliverable. The board did NOT reach consensus on whether to fork this. The moderator's call: keep it as a clearly-flagged parking-lot item in §8 below, not as one of the recommended proposals - because the existing infrastructure is curve-centric and the marginal cost of also outputting a price-quantile is small.

---

## 6. The proposed experiments - ranked, staged

### Phase 1 - Cheap CPU wrappers (run first)

**P1. Pointwise conformal-wrapped Exp5** - `experiments/area_pred/p1_pointwise_conformal.py`
- **Method**: refit Exp5 on `train[date < 2024-02-01]`; calibration set = `train[2024-02..2024-04]`; conformity score = per-τ residual; band edges = pointwise [α/2, 1-α/2] quantiles of cal residuals, applied as ±q_α to Exp5's predicted curve.
- **Output**: marginal per-τ price band around the Exp5 staircase.
- **Scoring**: Mean Winkler, coverage, width, median-curve Δ_W vs lag-30.
- **Cost**: ~10 min CPU. Reuses Exp5 entirely.
- **Prior**: P(useful calibrated band) ≈ 0.7. Marginal coverage guaranteed by construction; joint coverage will undercover at narrow α.

**P2. W-ball conformal-wrapped Exp5** - `experiments/area_pred/p2_wball_conformal.py`
- **Method**: same calibration split as P1; conformity score = `curve_wasserstein_distance(true, pred)`; q_α = (1-α)-quantile of cal W-residuals. Predicted "area" = the W-ball of radius q_α around Exp5's point forecast.
- **Output**: a single W-radius per test auction; the band is visualised as the convex hull of historical training curves within W ≤ q_α of Exp5's prediction.
- **Scoring**: W-ball coverage (does true curve fall inside?), mean W-radius, median-curve Δ_W vs lag-30.
- **Cost**: ~10 min CPU.
- **Prior**: theoretically cleanest. Coverage is in the eval metric.

**P3. Salvage #11 as a per-slice quantile band**
- **Method**: re-run `proposals_top5/exp11_ngboost_per_slice.py` but emit `{0.1, 0.5, 0.9}` quantile heads (HistGBM with `quantile` loss); isotonise across τ; conformalise residuals on a cal fold.
- **Output**: marginal per-τ band on a uniform quantity grid.
- **Scoring**: Winkler + coverage + width; do *not* report Δ_W vs lag-30 unless the band's median curve is also snapped to a real staircase (the original #11 failure mode was the smooth-uniform-grid output under Wasserstein).
- **Cost**: ~5 min CPU (the trained HistGBM models exist if we re-fit with `quantile` loss; otherwise ~1 h).
- **Prior**: P(Δ_W not catastrophic) ≈ 0.5; P(useful band) ≈ 0.6. **This directly tests "was #11's failure the model or the scoring?"**

### Phase 2 - Modest training cost (run if Phase 1 motivates)

**P4. DDNN-JSU per τ-slice** (Marcjasz et al. 2023 replication; #10 from the original 20)
- **Method**: 4-output JSU head per τ-slice; CRPS_JSU closed-form loss; isotonise across τ; 4-seed ensemble.
- **Output**: marginal per-τ band from JSU quantiles.
- **Scoring**: Winkler + coverage + width + per-slice CRPS + median-curve Δ_W.
- **Cost**: ~6 GPU-h (15 min × 4 seeds × ~6 hyperparameter passes if any).
- **Prior**: P(better calibrated than P1) ≈ 0.6. JSU captures heavy tails that Normal-style #11 missed.

**P5. Exp5 already-probabilistic via top-K retrieval** (Critic 2's #1 proposal)
- **Method**: Exp5 already retrieves nearest historical curve. Instead of `argmin`, return the top-K=20 candidates within bucket. Treat them as a sample from the posterior over tomorrow's curve. No new training.
- **Output**: a sample of K=20 real staircases per test row → marginal-per-τ band via 5/95 percentile envelope.
- **Scoring**: Winkler + coverage + median-curve Δ_W (top-1 is unchanged from Exp5; only the band wrapping is new). Optionally an Energy score on the K curves.
- **Cost**: ~5 min CPU. Cheapest band of all.
- **Prior**: P(reasonable calibration) ≈ 0.5; this is a *free* probe - Exp5's machinery already computes the top-K, we just don't report it.

### Phase 3 - Generative (deferred; run only if Phase 1+2 plateau)

**P6. CVAE-Ramsay band forecaster** (Critic 3's G1)
- **Method**: CVAE with cumulative-softplus decoder (monotonic by construction); encoder takes lag-30 Ramsay log-Q' scores → 8-d latent; decoder samples; reconstruction loss = Sinkhorn on Q(τ).
- **Output**: N=128 monotone samples → 5/95-per-τ envelope.
- **Cost**: ~2-3 GPU-h.
- **Note**: the only band proposal where monotonicity is built-in, not enforced post-hoc.

**P7. Conformalised RATD-diffusion on Ramsay scores** (Critic 3's G2; extends Family A #3)
- **Method**: conditional diffusion on Ramsay log-Q' coordinates conditioned on (bucket, lag-1/7/30 scores, retrieved neighbours). 128 DDIM samples per test row; snap each decoded curve to nearest training staircase; band = 5/95-per-τ; SPCI conformalised on a held-out cal fold.
- **Cost**: ~8-10 GPU-h.
- **Note**: only run if P1-P5 are inadequate. The R&D cost is high enough that the cheap probes must come first.

---

## 7. Recommended next 5 experiments (post-debate)

| Order | Experiment | Family | Cost |
|---:|---|---|---|
| 1 | **P1 Pointwise conformal-wrapped Exp5** | Conformal | ~10 min CPU |
| 2 | **P2 W-ball conformal-wrapped Exp5** | Conformal | ~10 min CPU |
| 3 | **P5 Exp5 top-K retrieval as band** | Retrieval-as-probabilistic | ~5 min CPU |
| 4 | **P3 Salvage #11 as quantile band** | Per-slice probabilistic | ~5-60 min CPU |
| 5 | **P4 DDNN-JSU per τ-slice** | Probabilistic | ~6 GPU-h |

If a band experiment delivers Winkler-90 below the lag-30-±-σ baseline AND coverage within 2 % of nominal, that is a publishable result on its own terms, regardless of whether the median-curve Δ_W is better than Exp5's -19.23.

---

## 8. Parking lot - decision-relevant scalar forecast (Critic 2's reframe)

This is a *separate* research direction worth scoping but not on the band leaderboard:

- **Target**: scalar cleared price at the realised market-clearing quantity (or at the retailer's expected load Q).
- **Method**: same lag features + calendar + (optional) exog; NGBoost / quantile-forest / DDNN-JSU.
- **Scoring**: pinball at α-quantiles; CRPS; **hedge-mis-spec cost** at the retailer's α-band edge (Critic 2's §3a formula).
- **Cost**: ~3 GPU-days end-to-end.

Pros: maps cleanly to a downstream decision (retailer hedging cap); reviewers know the EPF probabilistic-price-forecasting literature already exists (Marcjasz, Weron, Uniejewski).

Cons: sidesteps the curve-forecasting framing the rest of the repo is built around. Adds infrastructure work.

**Moderator's note**: keep the curve as the primary target; offer the scalar-price probabilistic forecast as a side deliverable in the same paper to demonstrate that the curve approach generalises down to scalars but the converse doesn't.

---

## 9. References

All citations from the existing [`docs/papers.md`](papers.md), specifically the probabilistic / conformal cluster:

- Marcjasz, Narajewski, Weron, Ziel (2023). *Distributional neural networks for electricity price forecasting (DDNN-JSU)*. Energy Economics. https://arxiv.org/abs/2207.02832
- Lipiecki, Uniejewski, Weron (2024). *Postprocessing of point predictions ... Isotonic Distributional Regression*. Energy Economics. https://arxiv.org/abs/2404.02270
- Mastropietro et al. (2024). *On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices*. Applied Energy. https://arxiv.org/abs/2404.02722
- Cassina et al. (2025). *Conformal Prediction for Electricity Price Forecasting in the Day-Ahead and Real-Time Balancing Market*. https://arxiv.org/abs/2502.04935
- Duan et al. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*. ICML. https://arxiv.org/abs/1910.03225
- Uniejewski & Weron (2021). *Regularized quantile regression averaging for probabilistic EPF*. Energy Economics. https://www.sciencedirect.com/science/article/abs/pii/S0140988321000268
- Liu et al. (2024). *Retrieval-Augmented Diffusion Models for Time Series Forecasting (RATD)*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf
- Tolstikhin et al. (2018). *Wasserstein Auto-Encoders*. ICLR. https://arxiv.org/abs/1711.01558
- Patrini et al. (2019). *Sinkhorn AutoEncoders*. UAI. https://www.auai.org/uai2019/proceedings/papers/253.pdf
- Sinha & Lucheroni (2025). *Monotonic autoencoder for short-term day-ahead electricity market bid curves*. Applied Energy. https://doi.org/10.1016/j.apenergy.2025.126262
