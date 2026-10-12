# Retrieval in the v_grid space

Most methods here do not build a new curve. They pick a real past curve that should
look like next week's. The two recent_median methods build a new curve from several
real past curves (see their sections).

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

## recent_median

The ridge model of rolling_ridge_exog_proxyonly (market-state proxy, no weather)
guesses the point. The candidates are the curves of the same interval, on any day
of the week, from the 28 days that end 7 days before the target day. These are the
newest curves known when the forecast is made. The 5 candidates closest to the
guess are taken together with the curves of the same interval 7, 8 to 14, 21 and
28 days earlier. At each volume, the forecast price is the weighted median of the
prices of these 15 curves. The curve 7 days earlier has weight 3 and every other
curve has weight 1. The forecast starts at the median start volume of the 15
curves and ends at their median end volume. So this method builds a new curve. At
each volume its price is the price of one of the 15 curves, because the total weight
(17) is odd.

## recent_median_wfc

The same as recent_median, but the ridge model also reads a simulated day-7
weather forecast (as in rolling_ridge_exog_wfc), and the curves 21 and 28 days
earlier have weight 2 (total weight 19).

## oracle

This one is not a real model. It is allowed to look at the true curve, and it
picks the training curve in the same interval-and-weekday group that is actually
closest to it. It cannot be used in practice. It only shows the best score this
kind of retrieval could ever reach, which tells us how much room there is to
improve.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
