# Functional data analysis

Functional data analysis treats each curve as one smooth shape, described by a
few numbers, instead of many separate prices.

Both methods here follow the same three steps. First they compress every curve
into a few shape numbers. Then a small linear model for each interval predicts
next week's shape numbers from past ones. Finally they snap the prediction to the
real training curve whose shape numbers are closest, so the output is always a
real staircase. A few shape numbers are easier to predict well than 64 prices
that move together.

## fof_snap

It finds the shape numbers with PCA, a method that picks out the few main shapes
in the data. It keeps 10 of them and uses the curves from 7, 8 and 14 days ago.

## logderiv_fpca

The same idea, but it first turns each curve into the slope of its price along
the volume axis and works on that. Comparing slopes is closer to the way the
score compares curves, so in theory this should match the metric better. It keeps
12 shape numbers and uses lags 7, 8 and 14.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
