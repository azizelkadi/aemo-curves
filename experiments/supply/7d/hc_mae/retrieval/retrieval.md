# Other retrieval methods

Like the v_grid methods, these return a real past curve chosen from a guessed
point in the embedding (a short list of numbers that describes a curve). They try
different ways of making that guess.

## gbm_retrieval

A boosted-tree model for each interval predicts the point from the 7-day-lag data,
then the nearest real training curve is returned. A boosted-tree query can place
the guess better than a straight-line model when the link between past and future
is not a straight line. Training this model on the supply side is slow.

## soft_knn

A ridge model guesses the point. The method finds the 10 nearest training curves
and weights them by how close they are, but it returns the single closest real
curve. Looking at several neighbours makes the choice steadier than trusting only
the first one. It uses a ridge query on lags 7, 14, 21 and 28.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
