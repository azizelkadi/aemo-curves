# Heterogeneous-Curves MAE (HC-MAE)

## What this metric is

**HC-MAE** is a curve-vs-curve similarity metric for two electricity-market
bid curves that may have **different step structure** (different number of
bids, different quantity points where the price jumps). The name stands for
**H**eterogeneous-**C**urves **M**ean **A**bsolute **E**rror - *heterogeneous*
because the two curves don't share a common discretisation, so a naive
point-wise MAE is undefined.

The metric was proposed by Sinha & Lucheroni (Applied Energy 2025) as the
evaluation loss for their monotonic-autoencoder bid-stack forecaster on the
Italian IPEX day-ahead market. Source paper + open-access PDF and project
notes:
[`demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md`](demand_and_supply_curve_forecasting_using_a_monotonic_autoencoder.md)
(DOI 10.1016/j.apenergy.2025.126262, CC BY 4.0).

## Abbreviations and notation used below

- **HC-MAE**: Heterogeneous-Curves Mean Absolute Error.
- **MAE**: Mean Absolute Error, i.e. `mean(|x - y|)`.
- **V**: cumulative quantity along a bid curve, in MW (a.k.a. *volume*).
- **P**: step price along a bid curve, in $/MWh.
- **U**: number of grid points used to discretise the union support
  (`U = 10 000` in the paper).
- **W₁**: 1-D Wasserstein-1 distance - the project's previous default
  curve-vs-curve metric, implemented in
  [`metrics/wasserstein.py`](../src/aemo_curves/metrics/wasserstein.py).
- **Step curve**: a bid stack represented as two equal-length arrays
  `(qty, price)` where step `k` covers cumulative-quantity interval
  `(qty[k-1], qty[k]]` at price `price[k]` (with `qty[-1] := 0` implicit).

## The procedure, step by step

Given two step curves `(V₁, P₁)` and `(V₂, P₂)`:

1. **Build the union support**. Let `V_u = sort(V₁ ∪ V₂)` - concatenate the
   step-boundary V-coordinates of both curves and sort.
2. **Define a common volume grid**. `V_common` = `U` equispaced points
   spanning `[min(V_u), max(V_u)]`. The paper recommends `U = 10 000`.
3. **Step-interpolate each curve onto the grid**. For every grid point V:
   - `Π₁(V) = P₁[k]` where `k` is the smallest index with `V₁[k] ≥ V`.
   - `Π₂(V)` analogously from curve 2.
   - **Edge case** - if V exceeds the maximum V of one curve, that curve's
     last (highest) step price is used. This treats the "past the cleared
     volume" region as the per-curve price cap, which matches how AEMO
     interprets curves above the dispatched quantity.
4. **Mean absolute difference on the grid**:
   `HC-MAE = (1 / U) · Σ_i |Π₁(V_common[i]) - Π₂(V_common[i])|`.

Result is in **price units** ($/MWh in AEMO).

## Properties

| Property | HC-MAE |
|---|---|
| Non-negative | ✓ |
| Identity-of-indiscernibles (zero iff curves equal) | ✓ (modulo grid resolution) |
| Symmetric in `(curve_a, curve_b)` | ✓ |
| Triangle inequality | ✓ in practice on finite-step curves and a fixed grid |
| Sensitive to the **V-position** of price changes | ✓ (this is the differentiator vs Wasserstein) |
| Sensitive to step order along V | ✓ |
| Proper scoring rule on a probabilistic forecast | ✗ - it is a *distance*, not a scoring rule |

## How HC-MAE differs from 1-D Wasserstein

The integration axis is the central difference.

| Aspect | **HC-MAE** | **Wasserstein-1** |
|---|---|---|
| Integration axis | Volume V, uniformly spaced | Price-quantile under the energy measure |
| Weighting | Uniform on V | Δqty (per-step quantity increments) |
| Identity check | Two curves with the same **P(V)** → 0 | Two curves with the same **energy-CDF of P** → 0 |
| Sensitivity to *where on V* a step sits | ✓ - different V-positions of equal-price steps produce non-zero | ✗ - invariant under any reshuffle of (qty, price) pairs that preserves the price marginal |

A concrete failure mode of W₁ that HC-MAE catches:

> **Curve A** has steps (1000 MW @ $40, 200 MW @ $90, 50 MW @ $300).
> **Curve B** has steps (1000 MW @ $40, 50 MW @ $90, 200 MW @ $300).
>
> Same energy-CDF of price → **W₁ = 0**. But under actual demand of 1100 MW
> these clear at $90 (A) vs $300 (B). **HC-MAE captures this asymmetry.**

The opposite failure mode (HC-MAE blind, W₁ catches) is rarer because moving
a step in V always changes the V-integrated MAE, but it can happen when the
two curves have *different total cleared volume* - W₁ is invariant under a
uniform horizontal rescaling that preserves the energy distribution, HC-MAE
is not.

## When to use HC-MAE vs alternatives

- **Use HC-MAE** when:
  - The V-coordinate of a price change matters (e.g. the marginal generator
    in merit-order dispatch).
  - You compare curves with very different step counts.
  - You want a single interpretable number in $/MWh that aligns with the
    most recent published bid-stack benchmark (Sinha & Lucheroni 2025).
- **Use Wasserstein-1** when:
  - Only the price *distribution* matters, not its V-positioning.
  - You're scoring an embedding whose Euclidean distance has a theoretical
    near-isometry with W₁ (e.g. the 64-d interp embedding in this repo -
    see [`embeddings/interp.py`](../src/aemo_curves/embeddings/interp.py)).
- **Use a clearing-price loss** (proposed elsewhere in this project's
  discussion) when:
  - The downstream consumer of the forecast is a trader or dispatcher
    who only cares about the clearing point.

## Metric-aligned embedding: `v_grid`

If you need a *vector* whose Euclidean L2 tracks HC-MAE (e.g. for kNN
retrieval), use [`embeddings/v_grid.py`](../src/aemo_curves/embeddings/v_grid.py).
It samples step prices at K equispaced V positions on `[0, V_max_global]`
so that

    ‖z_a - z_b‖₂ ≈ √K · HC-MAE(a, b)

up to per-pair union-support discretisation. The existing 64-d
[`interp`](../src/aemo_curves/embeddings/interp.py) embedding is **not
aligned with HC-MAE** - its L2 can be uncorrelated or anti-correlated
with HC-MAE because `interp` normalises `qty / max_qty`, discarding the
absolute-volume axis HC-MAE integrates over. See
[`notebooks/inspect_hc_mae.ipynb`](../notebooks/inspect_hc_mae.ipynb) for
the empirical comparison.

| Embedding | Metric its L2 tracks | Why |
|---|---|---|
| `interp` (existing) | W₁ | discretised quantile function - L²([0,1]) ≅ (P₂(ℝ), W₂) |
| **`v_grid`** (new) | **HC-MAE** | prices on a fixed absolute-volume grid |

## Implementation in this repo

Module: [`src/aemo_curves/metrics/hc_mae.py`](../src/aemo_curves/metrics/hc_mae.py).

Public API (mirrors `metrics/wasserstein.py` so HC-MAE drops into any
existing evaluation pipeline):

```python
from aemo_curves.stages.evaluate import curve_hc_mae_distance
d = curve_hc_mae_distance(
    true_qty, true_price, pred_qty, pred_price,
    n_grid=10_000,         # default; the paper's recommendation
)
```

Plus `score_predictions(df)` adds a `curve_hc_mae` column to a predictions
DataFrame, and `summary_stats(scored)` aggregates it the same way
Wasserstein's summary does. Both helpers accept any DataFrame with
`true_curve_qty`, `true_curve_price`, `pred_curve_qty`, `pred_curve_price`
columns - i.e. the schema every MLflow prediction artefact in this repo
already uses.

## Sanity check and head-to-head with Wasserstein

The diagnostic notebook
[`notebooks/inspect_hc_mae.ipynb`](../notebooks/inspect_hc_mae.ipynb) does
three things:

1. **Visual rank check.** Picks one anchor demand curve, ranks 199 other
   random demand curves by HC-MAE, plots the three closest neighbours, the
   three most-different, and a 12-panel grid showing the transition from
   "different" to "similar". The HC-MAE values monotonically grade the
   visual similarity.
2. **Head-to-head EDA with Wasserstein-1.** Computes W₁ on the same 199
   pairs and reports the Spearman and Pearson correlations between HC-MAE
   and W₁, plus a scatter plot of the two metrics.
3. **Disagreement examples.** Identifies pairs where HC-MAE and W₁ rank
   each other most differently (HC-MAE relatively high but W₁ low, and the
   reverse), and plots them side-by-side. These are the cases that reveal
   *which* metric is right for a given kind of difference - open the
   notebook to judge for yourself.

## Caveats

- **Grid resolution.** `n_grid = 10 000` is the paper's choice. The metric
  is numerically very stable in this range; halving to 5 000 typically
  changes the value by < 0.1 %.
- **Out-of-range extrapolation.** When one curve's V-extent is smaller than
  the other's, the missing region is filled with that curve's last step
  price. This is the default convention and corresponds to "no more
  generation available above this V"; if your application needs a different
  rule (e.g. fill with a sentinel `P_max`), wrap `curve_hc_mae_distance`
  with a preprocessing step.
- **Units.** HC-MAE is in `[P]` (e.g. $/MWh). It is **not** dimensionless;
  do not mix HC-MAE values across markets with different price caps without
  normalising first.
