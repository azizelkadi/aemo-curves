# Lag combinations

The methods in this group combine the curves of several past days into one
forecast. They use the weekly pattern, but they remove most of the noise of a
single day. There is no training.

## same_weekday_avg

It takes the curves from 7, 14, 21 and 28 days earlier (same interval, same
weekday) and averages them. The average is the forecast.

A single past day can be noisy because of one-off events. Averaging four
same-weekdays keeps the normal weekly shape but removes most of that noise.

## wmedian

It takes the curves of the same interval from 7, 8, 9, 10, 11, 12, 13, 14, 21
and 28 days earlier (10 curves). At each volume, the forecast price is the
weighted median of the 10 prices at that volume. The lag-7 curve counts twice
and each other curve counts once, so the total weight is 11 and the median is
always one of the 10 prices. The forecast's volume range runs from the median of
the 10 first edges to the median of the 10 last edges, and each curve counts
once here.

At each volume the forecast is a price from one of the past curves, and a single
unusual day has little effect on it.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
