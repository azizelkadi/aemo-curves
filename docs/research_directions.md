# Publishable research directions for the STEM curve-forecasting project

This document captures 10 concrete research directions for turning the STEM
wholesale-electricity curve-forecasting work into publishable papers and a
coherent PhD thesis.

Generated after a round of experiments in which we established that:

- The **lag-1 baseline is extraordinarily strong** for 1-day-ahead prediction
  (mean Wasserstein ≈ 36.49 $/MWh on the full 2024-05 → 2025-05 test window);
  the oracle shift-only ceiling is only ≈ -0.07 $/MWh below baseline.
- The **lag-30 baseline is substantially weaker** for 30-day-ahead prediction
  (mean Wasserstein ≈ 68.24 $/MWh); the oracle shift ceiling is ≈ -1.33 $/MWh,
  leaving meaningful room for learned corrections.
- The best trained model on 30-day-ahead (an EMA mixture of lags 30, 31, 37)
  captures roughly 17 % of that oracle gap.

That is actually a **publishable finding on its own** - see Direction 1.

---

## Reframe: beating-the-baseline is only one of many publishable angles

A PhD thesis is 3-5 papers. Not every paper must "beat" a baseline in a
point-forecast sense. A strong contribution can be:

- A rigorous **negative result** carefully instrumented (oracle ceilings,
  predictability decay with horizon, information-theoretic bounds).
- A **new methodology** whose main claim is theoretical (coverage guarantees,
  distributional prediction, calibration) rather than Wasserstein improvement.
- A **new problem formulation** (participant-level modelling, strategic-bidding
  detection, decision-relevant evaluation).
- A **generative contribution** (sample realistic curves for stress testing,
  counterfactual analysis).

---

## 10 publishable directions

### 1. "Fundamental limits of electricity supply-curve forecasting"

**Research question.** Given the most predictive representation possible,
what is the theoretical lower bound on W-distance between the predicted and
the true next-day supply curve?

**Novelty.** Nobody has done a rigorous oracle / predictability analysis of
electricity supply curves as *functional* objects. Extend the existing
oracle-shift scaffolding with:
- Information-theoretic bounds (mutual information between lag-k curves and
  today's curve).
- Hurst exponent / long-memory analysis of curve statistics.
- Decomposition: "persistence" vs "shape drift" vs "tail events".
- Horizon sweep: 1d, 7d, 30d, 90d.

**Why publishable.** Fills a gap - every forecasting paper claims its model
"works" without establishing what's achievable. Becomes the reference paper
for the field.

**Venues.** *International Journal of Forecasting*, *Energy Economics*,
*Applied Energy*, *IEEE Trans. Smart Grid*.

**Risk.** Low. Existing code covers most of the scaffolding.

---

### 2. "Probabilistic supply-curve forecasting with calibration guarantees"

**Research question.** Instead of point predictions, output a *predictive
distribution* over curves - with provably-correct coverage.

**Novelty.** Conformal prediction for **functional outputs** (supply curves
are functions, not scalars) is an active ML research topic with very few
applied papers.

**Methodology sketch.**
- For each interval, predict quantile functions `q_τ(x)` such that
  `P(true price ≤ q_τ | features) = τ`.
- Calibrate with split conformal adjustment on a held-out year.
- Enforce monotonicity (no quantile crossing).
- Report **coverage + sharpness** (CRPS / quantile score), not only
  Wasserstein.

**Why publishable.** Novel methodology + strong empirical work + directly
useful to traders and risk managers who need intervals.

**Venues.** *NeurIPS* / *ICML* (methodology paper), *IJF*, *IEEE TPWRS*
(applied).

**Risk.** Medium. Theoretical care needed for validity under distribution
shift.

---

### 3. "Generative modelling of electricity supply curves via diffusion"

**Research question.** Can we learn a generative model `p(curve | context)`
that produces realistic synthetic supply curves conditioned on market state?

**Novelty.** Almost no generative models of market-microstructure objects.
Diffusion / flow-based on curves (in the 64-d embedding space) is clean.

**Methodology sketch.**
- Train a conditional diffusion model
  `p(embedding | day-of-week, recent prices, lagged curves, exogenous)`.
- Sample → decode via monotonic projection to valid step curves.
- Evaluate via distributional quality (MMD / W2), downstream Monte-Carlo
  market-clearing-price prediction, counterfactuals.

**Why publishable.** Generative modelling is hot; curves are a novel
application domain; downstream tasks are economically meaningful.

**Venues.** *NeurIPS* (ML), *Energy Economics* (applied).

**Risk.** Medium-high. Diffusion training is finicky; evaluation design is
critical.

---

### 4. "Participant-level generative models of bid behaviour"

**Research question.** Use `Participant Code` to model each market
participant's bidding strategy; aggregate to the market-level curve.

**Novelty.** Identifiable participants are a **rare and underexploited**
asset in public electricity datasets. Almost no ML papers use participant
IDs. Angles:
- Per-participant embeddings (prod2vec-style) based on bid histories.
- Cluster participants by role (baseload, peaker, flexible, strategic).
- Model each participant's bid function as a conditional distribution.
- Aggregate via stacking to recover the market curve.

**Methodology sketch.**
- Temporal graph of participants.
- Per-participant hierarchical model: global population + participant-
  specific deviation.
- Compare market-level prediction via aggregation vs. monolithic curve
  prediction.

**Why publishable.** Unique data angle. Bridges ML, econometrics, and market
microstructure. Tells a story ("who sets the price") that reviewers care
about.

**Venues.** *Applied Energy*, *Energy Economics*, *Journal of Energy
Markets*, *KDD* applied track.

**Risk.** Medium. Participant data quality, survival bias (participants
enter/exit).

---

### 5. "Strategic bidding and market-power detection"

**Research question.** Identify auctions where participants deviate from
marginal cost (i.e., exercise market power or game the auction).

**Novelty.** Combines econometric theory (Lerner index, residual demand
analysis) with anomaly detection. No large-scale ML study exists for the
Australian STEM.

**Methodology sketch.**
- Estimate counterfactual "marginal cost" supply per participant via rolling-
  window lower-envelope regression.
- Residual = actual bid - estimated marginal cost.
- Detect deviations that coincide with binding demand or low competition.
- Validate against known market events (AEMO regulatory reports).

**Why publishable.** High impact - market power is a major regulatory
concern. Regulators actively read this work.

**Venues.** *Energy Economics*, *The Energy Journal*, *Energy Policy*,
*RAND Journal of Economics*.

**Risk.** Medium. Needs domain collaborator for validation.

---

### 6. "Regime-aware forecasting with external signals"  ★ chosen track ★

**Research question.** Can exogenous data (weather, demand forecasts, fuel
prices, holidays, proxy market-state features) close the gap to the oracle?

**Novelty.** Our results show baseline is near-optimal *without* external
data. Adding them and quantifying the marginal improvement per feature is
a clean ablation study with practical value.

**Methodology sketch.**
- Add features:
  - Bureau of Meteorology temperature / wind / solar irradiance.
  - AEMO day-ahead demand forecast.
  - Australian gas / coal spot prices.
  - Public-holiday calendars.
  - Derived proxies from the dataset itself (daily volatility, HHI,
    participant count).
- Regime-classifier (hot / cold / windy / holiday) → choose between baseline
  and a corrected model.
- Report per-feature Shapley / leave-one-out improvements.

**Why publishable.** Practical, reproducible, quantifies marginal value of
each external input - a contribution practitioners will use.

**Venues.** *Applied Energy*, *IEEE Trans Smart Grid*, *IJF*.

**Risk.** Low. Mostly engineering + standard ML. Main risk is a modest gain
- still publishable if presented honestly with ablations.

**Status in this repo.** See
[`experiments/horizon_1d_exog/`](../experiments/horizon_1d_exog/README.md).

---

### 7. "Decision-relevant evaluation of curve forecasts"

**Research question.** Wasserstein is statistical - but does a better
Wasserstein mean better trading decisions?

**Novelty.** Build a **trader simulator** that uses the forecast curve to
make real dispatch/bid decisions; measure P&L / risk. Re-rank all existing
models under this metric.

**Methodology sketch.**
- Define a stylised decision problem (retailer hedging load, battery
  scheduler).
- Feed each model's prediction into an LP/QP optimiser.
- Compute realised P&L on the test period → new ranking.

**Why publishable.** Nearly nobody does this. Existing ~40 experiments become
data points in a single meta-study. May reveal that a "worse-by-Wasserstein"
model is **better for trading**.

**Venues.** *Energy Economics*, *Journal of Energy Markets*, *EJOR*.

**Risk.** Low-medium. Simulator design needs justification and sensitivity
analysis.

---

### 8. "Rare-event forecasting: price spikes and caps"

**Research question.** Forget the mean - can we predict the *tail* of the
supply curve, which drives price spikes?

**Novelty.** Tail/extreme-event prediction is underserved in electricity
markets. Combine Extreme Value Theory with imbalanced-class classification.

**Methodology sketch.**
- Define "spike" as top-5 % quantile of max daily price.
- Build a spike classifier (AUC, precision@k).
- Conditional curve model for spike days.
- Report precision/recall + expected loss from missed spikes.

**Why publishable.** Spike days are where real trading money is won or lost.
Operators actively want this.

**Venues.** *IJF*, *IEEE Trans Power Systems*, *Energy Economics*.

**Risk.** Medium. Rare events = high variance.

---

### 9. "Neural operator / functional learning for supply curves"

**Research question.** Supply curves are *functions*; predict them with
Fourier / Graph Neural Operators that respect the function-space structure.

**Novelty.** FNO / DeepONet applied to market curves is unexplored. Operator
learning is a hot ML topic.

**Methodology sketch.**
- Represent each curve as a function on `[0, 1]` (normalised quantity).
- FNO maps (recent curves, features) → predicted curve.
- Enforce monotonicity structurally (not as a loss).
- Compare to existing 64-d embedding baselines.

**Why publishable.** Novel application of operator learning; structural
priors (monotonicity) make it principled.

**Venues.** *NeurIPS*, *ICLR*, *JMLR*.

**Risk.** High. Careful implementation; ML reviewers are picky.

---

### 10. "An open benchmark for electricity supply-curve forecasting"

**Research question.** Package the STEM dataset + canonical train/test split
+ a leaderboard of models as a community benchmark.

**Novelty.** There is **no public, well-documented benchmark** for curve
forecasting. You'd be defining the task.

**Contributions.**
- Clean, versioned dataset.
- Three baselines (lag-1, lag-30, EMA).
- Ten diverse models (already built in this repo).
- Standardised metrics (W, CRPS, coverage, economic P&L).
- Leaderboard format.

**Venues.** *NeurIPS Datasets & Benchmarks*, *Scientific Data*, *JOSS*.

**Risk.** Low methodologically; moderate for acceptance (benchmarks must be
genuinely useful).

---

## Suggested 4-paper thesis structure

1. **Chapter 2 / Paper 1** - Fundamental limits (Direction 1). Sets the
   foundation. Negative results turned into a rigorous study.
2. **Chapter 3 / Paper 2** - Probabilistic / conformal curve forecasting
   (Direction 2). The methodology contribution.
3. **Chapter 4 / Paper 3** - Participant-level generative model
   (Direction 4). The unique-data contribution.
4. **Chapter 5 / Paper 4** - Decision-relevant evaluation + market-power
   detection (Directions 5 + 7). The impact contribution.

Plus a benchmark/dataset paper (Direction 10) as a companion, low-effort now
that the infrastructure exists.

---

## Immediate next step taken

We are implementing **Direction 6** first:
- Low risk, re-uses all existing infrastructure.
- The "exogenous data" layer unlocks many later directions (all generative
  models, regime-aware methods, and probabilistic methods benefit from it).
- Expected outcome: modest but measurable improvement over lag-1 baseline,
  plus a clean ablation quantifying the marginal value of each exogenous
  input - publishable in *Applied Energy* or *IJF* on its own.

See [`experiments/horizon_1d_exog/README.md`](../experiments/horizon_1d_exog/README.md).
