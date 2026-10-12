# Blends

Blends mix several past weeks into one curve. A mix of recent weeks is smoother
than any single week. The target is the residual (demand-minus-supply price
spread) curve.

## ema_baseline

It takes a fixed-weight average of the lag-7, lag-14 and lag-21 residual curves
and lays the result on the lag-7 step shape. The weights are set in advance: 0.6,
0.2 and 0.2, so the most recent week counts the most. There is no training.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
