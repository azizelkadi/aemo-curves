# Per-slice price model

This group has one method, and it runs on the demand side only. It predicts the
curve one piece at a time.

## ngboost_per_slice

The curve is cut into slices along the volume axis. One boosted-tree model per
slice predicts the price at that slice from past data. The slice prices are then
forced to stay in order and joined back into a full curve. It uses the curves
from 7, 8 and 14 days ago.

Predicting one price at a time is a simple and strong table-based approach, and
it can give a probability for each slice, not just one number. It is demand only
because the code forces prices to fall along the curve, which matches demand bids
but not supply offers.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
