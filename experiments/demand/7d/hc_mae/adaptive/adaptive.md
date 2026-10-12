# Adaptive blending

The best deployable demand model (the champion, `vgrid.rolling_ridge_exog_split`)
beats the lag-7 baseline on average, but not on every auction: on about 40% of
rows the plain lag-7 curve is actually closer to the truth. These methods try to
get the best of both by blending, per auction, the lag-7 baseline curve with the
champion's retrieved curve. The blend weight `w` runs from 1 (use the baseline)
to 0 (use the champion); in between, the two step curves are averaged on the union
of their volume break-points. At the exact endpoints the original curve is
returned untouched, so the blend can never score worse than the metric says the
endpoint does. All methods read a precomputed table of the two candidate curves,
their errors, and the champion's retrieval diagnostics; every feature and model
for a test month uses only data at least 7 days older than that month.

## constant_blend

A control. It uses one blend weight for the whole month, chosen from a grid to
make the recent blended curves as accurate as possible. It ignores which auction
is which, so it tells us how much a single global mix helps.

## stratum_blend

A control. It sets a separate weight for each group of intervals (blocks of four)
split by weekday versus weekend, learned from recent errors and pulled toward the
global weight when a group has little data. It tests whether coarse calendar
structure alone is enough.

## prob_blend

The learned method. A gradient-boosted classifier predicts the chance that the
baseline beats the champion for each auction, using retrieval distances, recent
per-interval errors, baseline drift, and calendar features. That probability is
calibrated (with isotonic regression fitted on past out-of-fold predictions) and
used directly as the blend weight.

## prob_blend_raw

The same classifier as `prob_blend` but without the calibration step. Comparing
the two isolates how much the isotonic calibration is actually worth.

## regret_blend

A learned method that predicts the size of the error gap (champion error minus
baseline error) with a gradient-boosted regressor, then turns it into a smooth
weight with a sigmoid whose temperature is tuned on recent out-of-fold data. It
leans on the baseline where the champion is predicted to do worse.

## oracle_gate

A diagnostic that uses the truth and is **not deployable**. For each auction it
picks whichever endpoint curve is really closer to the realized curve. It is the
ceiling for a perfect hard choice between the two curves.

## oracle_weight

A diagnostic that uses the truth and is **not deployable**. For each auction it
picks the grid weight that gives the smallest realized error, so it can also mix
the two curves. By construction it scores at or below `oracle_gate`, and it is the
ceiling for the whole soft-blend family.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
