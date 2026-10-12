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
always one of the 10 prices. The forecast covers the same volume range as the
lag-7 curve.

At each volume the forecast is a price from one of the past curves, and a single
unusual day has little effect on it.

## wmedian_cap

It is wmedian with one change at the price cap. The cap of a day is the highest
offer price over all supply curves of that day, and the cap known for the
forecast day D is the cap of day D-7. In each of the 10 past curves, every step
priced at or above its own day's cap minus 0.5 $/MWh gets the known cap, and
when the known cap is lower, every price above it is cut to it. Then the
weighted median is taken as in wmedian. So a cap change that is known 7 days
ahead reaches all 10 past curves, not only the lag-7 curve.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
