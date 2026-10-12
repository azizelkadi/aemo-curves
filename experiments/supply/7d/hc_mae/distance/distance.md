# Distance-ranking retrieval

This group has one method. Like the other retrieval methods it returns a real
past curve, but it chooses by ranking candidates directly.

## distance_rank

A boosted-tree model (a strong table-based model) learns to guess the distance
between a target curve and any candidate curve, using only information known 7
days ahead. For each auction it scores every candidate in the same group and
returns the one with the smallest predicted distance.

Picking the best neighbour is the real goal, so learning to rank neighbours by
their likely distance attacks the problem head-on. The training target uses the
same metric as the final score, which keeps the model honest. The number of
candidates per target is kept small to keep memory use low.

For the score of each experiment against the lag-7 baseline, see `../results.md`.
