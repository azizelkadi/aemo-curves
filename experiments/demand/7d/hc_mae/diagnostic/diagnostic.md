# Diagnostics

Diagnostics are not real models. They measure limits, to show how much a certain
kind of method could ever gain. Some of them are allowed to look at the true
answer, so they cannot be used in practice.

## oracle_shift

It shifts the lag-7 curve by the best possible amount, using the true curve to
choose that amount. This shows the most that any shift-only correction could ever
gain. If this barely beats the baseline, then shifting the lag curve cannot help
much, and the gains must come from picking a different curve.

## realized_shift

This one does not look at the answer, so it is usable. It measures how much the
curve moved between 14 and 7 days ago and carries 30 percent of that recent move
forward onto the lag-7 curve. It tests a simple idea: that a recent drift tends to
continue a little.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
