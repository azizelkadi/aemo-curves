# Correctors

Correctors start from the lag-7 curve and change it only a little, instead of
building a new curve from scratch.

The idea is that last week is already close. A small, careful adjustment can
remove the part of the error that is easy to predict, without adding noise. For
example, if every price this week is about 5 $/MWh higher than last week, then
moving last week's curve up by 5 fixes most of the error. All methods here use
the curves from 7, 14, 21 and 28 days ago, so the curve they correct is the lag-7
curve and nothing comes from inside the forecast gap.

## shift

A linear model for each interval predicts a single price shift. The prediction is
the lag-7 curve moved up or down by that shift. The staircase shape is kept
exactly. This helps when the whole price level drifts but the shape stays.

## shift_scale

The same as shift, but it also predicts a scale for the volume axis. So the lag-7
curve can move in price and also stretch or shrink in volume. This covers weeks
where both the price level and the total volume change.

## residual_on_baseline

A linear model predicts many small price corrections, one for each point along
the curve, and adds them on top of the lag-7 curve. Strong regularisation keeps
the corrections small, so the result stays close to the baseline unless the data
is clear. This can fix the shape, not just the level.

## residual_gbm

The same as residual_on_baseline, but the corrections come from a boosted-tree
model instead of a linear one, so it can catch patterns that are not a straight
line.

## per_day_shift

It predicts one price shift for the whole day and applies that same shift to all
48 intervals of that day. Sharing one shift across the day gives the model far
more training examples than fitting a separate shift per interval, so the shift is
steadier and less likely to overfit.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
