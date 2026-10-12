# Correctors

Correctors start from a recent curve and change it only a little, instead of
building a new curve from scratch. Most of them start from the lag-7 curve.

The idea is that last week is already close. A small, careful adjustment can
remove the part of the error that is easy to predict, without adding noise. For
example, if every price this week is about 5 $/MWh higher than last week, then
moving last week's curve up by 5 fixes most of the error. Except for
volume_shift_wfc, all methods here use the curves from 7, 14, 21 and 28 days ago,
so the curve they correct is the lag-7 curve. Every method uses only curves dated
7 or more days before the target day, so nothing comes from inside the forecast
gap.

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

## volume_shift_wfc

It starts from the pointwise median of the 14 curves of the same interval 7 to
20 days earlier. This median curve starts at the smallest first edge of the 14
curves, ends at the median of their last edges, and keeps every edge of the 14
curves in between (equal-price steps are not merged). The cliff volume of a
demand curve is the volume of the bids above the -1,000 $/MWh price floor, so it
is the volume where the curve drops to the floor. A median model
(gradient-boosted trees with the absolute error loss) forecasts the cliff volume
of the target auction. The model is refit at the start of every test month on
the rows dated 7 or more days before the refit day. The median curve then moves
along the volume axis by the forecast cliff volume minus its own cliff volume.
Its first step (the highest-priced bids) and its last edge stay where they are,
so the volume range of the curve does not change.

The model reads the cliff volumes and the last edges of the curves 7 to 34 days
earlier, the mean cliff volume of the 48 intervals of the day 7 days earlier, the
calendar of the target day, the market-state proxy of 7 days earlier and
weather_fc. weather_fc is a simulated forecast: the actual Perth weather of the
target day plus seeded random noise of the size of a day-7 forecast error. The
noise is drawn independently for each day, but real forecast errors are
correlated from day to day, so this score is optimistic until it is measured with
real day-7 forecasts.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
