# Solver Tournament Design

Goal
- Rank and select solvers per instance/cluster with low overhead and stability.

Method
- Clustering: KMeans on features to obtain instance clusters.
- Ratings: Elo ratings (global + per-cluster) initialized from historical pairwise outcomes.
- Selection: UCB combines exploitation (Elo) and exploration (trial counts).
- Tournament: Swiss-system simulation to produce a short ranking before selection.

Metrics
- Regret/accuracy vs baselines; selection latency.

Notes
- Keep tournament rounds small to bound overhead; rely on cached ratings.
- Use consistent seeds and wall-time caps for reproducibility.

