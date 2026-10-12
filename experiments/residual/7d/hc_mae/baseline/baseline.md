# Baseline

The baseline is the simplest forecast in this group, and it is the line that
every other method must beat. Here the target is the residual
(demand-minus-supply price spread) curve: for each auction, the demand price
minus the supply price along the shared volume axis. It is a falling curve that
goes negative past the market-clearing volume.

## lag7

It says next week's residual curve will look exactly like last week's. For each
auction it takes the residual curve from the same 30-minute interval 7 days
earlier and uses it without any change. There is no training and nothing to tune.

This is hard to beat because electricity curves repeat strongly from week to
week. The same weekday at the same time of day usually looks very similar, so
last week's curve is already a strong guess.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
