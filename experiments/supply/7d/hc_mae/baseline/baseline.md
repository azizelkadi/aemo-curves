# Baseline

The baseline is the simplest forecast in this group, and it is the line that
every other method must beat.

## lag7

It says next week will look exactly like last week. For each auction it takes the
curve from the same 30-minute interval 7 days earlier and uses it without any
change. There is no training and nothing to tune.

This is hard to beat because electricity curves repeat strongly from week to
week. The same weekday at the same time of day usually looks very similar, so
last week's curve is already a strong guess.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
