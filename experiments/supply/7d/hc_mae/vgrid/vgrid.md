# Retrieval in the v_grid space

These methods do not build a new curve. They pick a real past curve that should
look like next week's.

Each curve is first turned into a short list of numbers, called an embedding,
that describes its shape. Curves with similar shapes have similar embeddings. To
make a forecast, a small model guesses where in this space next week should land,
and the method returns the real training curve that sits closest to that guess.
Returning a real curve keeps the true staircase shape, instead of a smooth, fake
one. All methods here use the v_grid embedding.

## strat_knn

A small linear model (called ridge), fitted for each interval and weekday, guesses
the point from the data known 7 days ahead. The nearest real training curve to
that point is returned. The model is trained once on the training years.

## rolling_ridge

The same as strat_knn, but the ridge model is trained again at the start of every
test month, using only data old enough to respect the 7-day gap. This lets it
follow slow changes in the market over time.

## rolling_ridge_exog

The same as rolling_ridge, plus extra inputs for the query model: a market-state
proxy and weather. These extra inputs are shifted by 7 days, so the model never
sees anything from inside the forecast gap. Demand reacts to weather and to the
state of the market, so these signals help it pick a better curve. This is the
strongest deployable model on demand.

## oracle

This one is not a real model. It is allowed to look at the true curve, and it
picks the training curve in the same interval-and-weekday group that is actually
closest to it. It cannot be used in practice. It only shows the best score this
kind of retrieval could ever reach, which tells us how much room there is to
improve.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
