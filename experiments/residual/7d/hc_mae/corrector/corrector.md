# Correctors

Correctors start from the lag-7 curve and change it only a little, instead of
building a new curve from scratch. The target is the residual (demand-minus-supply
price spread) curve.

The idea is that last week is already close. A small, careful adjustment can
remove the part of the error that is easy to predict, without adding noise. All
methods here use the curves from 7, 14, 21 and 28 days ago, so the curve they
correct is the lag-7 curve and nothing comes from inside the forecast gap.

## shift_scale

A linear model for each interval predicts a single price shift and a scale for the
volume axis. So the lag-7 residual curve can move in price and also stretch or
shrink in volume. This covers weeks where both the price level and the total
volume change.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
