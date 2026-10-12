# Blends

Blends mix several past weeks into one curve. A mix of recent weeks is smoother
than any single week.

## ema_baseline

It takes a fixed-weight average of the lag-7, lag-14 and lag-21 curves and lays
the result on the lag-7 step shape. The weights are set in advance: 0.6, 0.2 and
0.2, so the most recent week counts the most. There is no training.

## blended_lag

For each interval and weekday it learns the single best weight to mix the lag-7
and lag-14 curves. Some times of day follow last week more closely, while others
follow two weeks ago, and the learned weight fits this difference.

## learned_blend

For each interval and weekday it learns weights over the lag-7, 14, 21 and 28
curves. The weights are positive and add up to one, so the result is a true
average of real weeks. This is a freer mix than blended_lag.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
