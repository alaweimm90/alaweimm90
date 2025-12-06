# Abstract

Algorithm selection—choosing the best solver for a given problem instance—can yield order-of-magnitude speedups over any single algorithm. However, state-of-the-art selection methods like SATzilla require 24-254ms selection time, prohibiting real-time applications. We introduce **Librex.Meta**, a tournament-based meta-learning framework that achieves sub-millisecond selection (0.15ms) while maintaining competitive accuracy.

Librex.Meta combines Swiss-system tournaments with Elo rating systems to efficiently rank algorithms during training. We partition the problem space using KMeans clustering and maintain both global and cluster-specific Elo ratings to capture algorithm specialization. At selection time, we use Upper Confidence Bound (UCB) selection with Elo-based priors, achieving O(d) complexity where d is feature dimensionality.

Across 5 diverse ASlib scenarios spanning 4,099 test instances and 42 algorithms, Librex.Meta achieves best average regret (0.0545) compared to SATzilla (0.0603), SMAC (0.0659), AutoFolio (0.0709), and others, with **1664× speedup** over SATzilla. Librex.Meta excels on graph problems (rank 1/7 on GRAPHS-2015) and binary selection tasks (96.5% accuracy on CSP-2010), while remaining hyperparameter robust—only the number of clusters impacts performance.

Our work demonstrates that tournament-based meta-learning enables real-time algorithm selection, identifies problem classes where this approach excels, and provides a methodological lesson: hyperparameters tuned on synthetic data often fail to transfer to real benchmarks.

---

**Word count**: ~210 words
**Status**: Complete draft
**Next**: Full paper revision
