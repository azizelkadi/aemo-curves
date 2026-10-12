# Adaptive blending (supply)

The best deployable supply retrieval model (`vgrid.rolling_ridge_exog_split`, ~67.19)
actually *loses* to the lag-7 baseline (54.83) on average — but not on every auction.
These methods blend, per auction, the lag-7 baseline curve with the champion's
retrieved curve. The blend weight `w` runs from 1 (use the baseline) to 0 (use the
champion); in between, the two step curves are averaged on the union of their volume
break-points. At the exact endpoints the original curve is returned untouched, so the
blend can never score worse than the metric says the endpoint does. Because supply
curves are non-decreasing and a convex mix of two rising curves is rising, the blend is
direction-agnostic. All methods read a precomputed table of the two candidate curves,
their errors, and the champion's retrieval diagnostics; every feature and model for a
test month uses only data at least 7 days older than that month.

Only the four variants the family generalizes to extensions are kept here
(constant_blend, stratum_blend, oracle_gate, oracle_weight); the learned per-row
variants were dropped because they lost on demand.

## constant_blend

A control. It uses one blend weight for the whole month, chosen from a grid to make
the recent blended curves as accurate as possible. It ignores which auction is which,
so it tells us how much a single global mix helps. On supply, with the champion behind
the baseline, this is expected to sit near w~1 (almost all baseline).

## stratum_blend

A control. It sets a separate weight for each group of intervals (blocks of four) split
by weekday versus weekend, learned from recent errors and pulled toward the global
weight when a group has little data. It tests whether coarse calendar structure alone
is enough. Per-stratum grid search on a 3,000-row trailing subsample (approximation;
see `_GRID_SELECT_MAX_ROWS`).

## oracle_gate

A diagnostic that uses the truth and is **not deployable**. For each auction it picks
whichever endpoint curve is really closer to the realized curve. It is the ceiling for
a perfect hard choice between the two curves.

## oracle_weight

A diagnostic that uses the truth and is **not deployable**. For each auction it picks
the grid weight that gives the smallest realized error, so it can also mix the two
curves. By construction it scores at or below `oracle_gate`, and it is the ceiling for
the whole soft-blend family.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
