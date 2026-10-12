# Metric learning

This group has one method. It first learns a better way to measure how close two
curves are, and then uses that for retrieval.

## wasserstein_metric

A small neural network is trained so that the plain distance between two points
matches the true distance between the two curves they came from. After training,
every curve is passed through the network, and the usual nearest-curve retrieval
runs in this new space.

Retrieval is only as good as its distance. If "close in the new space" really
means "close as curves", then the nearest neighbour should be a better choice. It
uses the 7-day lag. The number of training pairs and rounds is kept small so the
run finishes in reasonable time.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
