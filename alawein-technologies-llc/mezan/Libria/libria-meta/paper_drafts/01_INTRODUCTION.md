# 1. Introduction

## 1.1 Motivation

Algorithm portfolios have become essential tools for solving hard combinatorial problems across diverse domains including satisfiability (SAT), constraint satisfaction (CSP), graph coloring, and answer set programming (ASP). The key insight is simple but powerful: no single algorithm dominates across all problem instances. Instead, per-instance algorithm selection—choosing the best solver for each specific problem—can yield order-of-magnitude speedups over any single algorithm.

The algorithm selection problem, formalized by Rice (1976), asks: given a portfolio of algorithms and a new problem instance, which algorithm should we run? Modern approaches use machine learning to predict algorithm performance based on instance features, achieving impressive results. SATzilla (Xu et al., 2008), for example, won multiple SAT competitions by learning to select from a portfolio of specialized SAT solvers.

However, existing algorithm selection methods share a critical limitation: **selection overhead**. State-of-the-art approaches like SATzilla require 254ms on average to select an algorithm, while AutoFolio needs 24ms. For problems where individual solves complete in seconds or less, this overhead becomes significant. More critically, for real-time applications—interactive constraint systems, online planning, embedded SAT solving—selection latencies exceeding 1ms are prohibitive.

This raises a fundamental question: **Can we achieve competitive selection accuracy with sub-millisecond selection time?**

## 1.2 Challenges

Achieving fast, accurate algorithm selection faces three core challenges:

**Challenge 1: The speed-accuracy trade-off.** Accurate selection typically requires complex machine learning models. SATzilla employs regression forests with pre-solving for cost-sensitive learning. AutoFolio combines multiple classifiers and regressors through automated configuration. These sophisticated models achieve strong accuracy but at the cost of high computational overhead. Conversely, simple decision rules (e.g., "always pick the algorithm with lowest average runtime") execute instantly but make poor per-instance choices. The challenge is finding a middle ground that preserves accuracy while slashing latency.

**Challenge 2: Problem diversity.** Algorithm portfolios must handle vastly different problem classes. Graph coloring instances differ fundamentally from satisfiability problems, which differ from constraint satisfaction problems. Each domain has distinct feature distributions, algorithm portfolios, and performance landscapes. A selection method must adapt to these varying characteristics while maintaining efficiency. Moreover, within a single domain, instance characteristics can vary dramatically—some SAT instances are easily satisfiable, others require heavy search, still others are unsatisfiable.

**Challenge 3: Meta-learning efficiency.** Algorithm selection is inherently a meta-learning problem: we learn how to select among learning algorithms (or solvers, more generally). The meta-learner must efficiently extract knowledge from limited training data—typically hundreds to thousands of instances paired with algorithm performance measurements. Training must be fast, and the learned model must generalize to unseen instances. Importantly, the selection mechanism must avoid repeating expensive computations at prediction time.

## 1.3 Our Approach

We introduce **Librex.Meta**, a novel tournament-based meta-learning framework for algorithm selection that achieves sub-millisecond selection time while maintaining competitive accuracy. Our key insight is that **tournament dynamics efficiently capture algorithm strengths across problem distributions**.

Librex.Meta employs a Swiss-system tournament structure during training, where algorithms compete head-to-head on problem instances. Winners are determined by which algorithm solves a given instance faster. After each match, we update Elo ratings—a Bayesian skill rating system originating in chess—to track algorithm performance. Crucially, we maintain both global Elo ratings (overall algorithm strength) and cluster-specific ratings (performance within problem subclasses).

The training process works as follows:
1. **Cluster the problem space** using KMeans on instance features, partitioning problems into k coherent subclasses (we find k=3 optimal).
2. **Run Swiss-system tournaments** where algorithms compete on training instances, updating Elo ratings after each match.
3. **Maintain dual ratings**: global Elo (algorithm strength across all problems) and cluster-specific Elo (specialized performance).

At selection time, Librex.Meta operates in three fast steps:
1. **Extract features** from the new instance (sub-millisecond).
2. **Assign to cluster** using the trained KMeans model (O(kd) where k=3, d=feature dimensionality).
3. **Select via UCB** using cluster-specific Elo ratings as priors, balancing exploitation of high-rated algorithms with exploration.

This design achieves **O(d)** selection complexity—linear in feature dimensionality—with no tree traversal, no ensemble inference, and no pre-solving. Empirically, selection completes in 0.15ms on average, **1664x faster than SATzilla** (254ms).

The tournament framework provides several advantages:
- **Efficiency**: Swiss-system tournaments rank n algorithms with O(n log n) comparisons, far fewer than round-robin tournaments.
- **Adaptivity**: Elo ratings adapt to the problem distribution automatically, up-weighting algorithms that win frequently.
- **Specialization**: Cluster-specific ratings capture problem class strengths—e.g., an algorithm may excel on graph problems but struggle on SAT.
- **Simplicity**: The framework requires minimal hyperparameter tuning (we show most parameters have negligible impact).

## 1.4 Contributions

This work makes four primary contributions:

**1. Methodological: A tournament-based algorithm selection framework**

We introduce the first application of Swiss-system tournaments and Elo ratings to meta-learning for algorithm selection. While Elo ratings have been applied to game playing and active learning, their use in algorithm selection is novel. Our framework combines:
- Swiss-system tournament structure for efficient algorithm ranking
- Dual Elo rating system (global + cluster-specific)
- KMeans clustering for problem space partitioning
- UCB selection for exploitation-exploration balance

The result is a conceptually simple yet effective meta-learning approach.

**2. Empirical: Strong performance on diverse benchmarks**

We evaluate Librex.Meta on 5 diverse ASlib scenarios spanning graph coloring, constraint satisfaction, MaxSAT, SAT, and answer set programming—4,099 test instances total with 42 algorithms. Librex.Meta (optimal configuration) achieves:

- **Best average regret**: 0.0545 across all scenarios, outperforming SATzilla (0.0603), SMAC (0.0659), AutoFolio (0.0709), and others
- **Extreme speedup**: 0.15ms selection time vs 254ms for SATzilla (1664x faster), 158x faster than AutoFolio
- **Wins on graph problems**: Rank 1/7 on GRAPHS-2015 with 19% average regret vs 124% for second-best (AutoFolio)
- **Near-perfect binary selection**: 96.5% top-1 accuracy on CSP-2010 (2-way selection)

While statistical significance is limited by scenario count (n=5, Friedman test p=0.85), we observe medium-to-large effect sizes (Cliff's δ = 0.36-0.44 vs several baselines), suggesting practical advantage.

**3. Practical: Identifies problem class strengths and trade-offs**

Through systematic evaluation, we identify where Librex.Meta excels and where it struggles:

- **Excels on**: Graph problems (GRAPHS-2015: rank 1/7), binary/simple selection (CSP-2010: 96.5% accuracy)
- **Competitive on**: MaxSAT (rank 4/7, 2.5% regret)
- **Weak on**: Hard SAT (SAT11-HAND: rank 5/7), ASP (ASP-POTASSCO: rank 5/7)

This analysis reveals Librex.Meta's sweet spot: **real-time graph problem solving** and **fast binary selection tasks**. For applications in these domains, Librex.Meta provides 1600x speedup with minimal accuracy loss (10% regret penalty vs SATzilla).

**4. Methodological insight: Hyperparameter robustness on real data**

Ablation studies reveal a surprising finding: **most hyperparameters have negligible impact on real data**. Specifically:

- **n_clusters**: Only parameter that matters (optimal = 3, provides 9.4% improvement)
- **ucb_c** (exploration constant): All values [0.1, 2.0] yield identical results
- **n_tournament_rounds**: All values [1, 15] yield identical results
- **elo_k** (update rate): Broad range [24, 48] works well; avoid 8 and 16

This robustness contrasts sharply with our earlier experiments on synthetic data, where ucb_c appeared critical (34% improvement). We identify this **mock vs real data gap** as an important methodological lesson: hyperparameters tuned on synthetic benchmarks may not transfer to real problems.

The practical implication is ease of use: Librex.Meta requires minimal tuning, with only n_clusters needing adjustment based on problem diversity.

## 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work in algorithm selection, meta-learning, and tournament-based learning. Section 3 describes the Librex.Meta framework in detail, including tournament structure, Elo rating updates, and UCB selection. Section 4 presents our experimental setup, including ASlib scenarios, baseline methods, and evaluation metrics. Section 5 reports results on overall performance, per-scenario rankings, speed-accuracy trade-offs, and ablation studies. Section 6 discusses why tournament-based selection works, problem class analysis, limitations, and practical implications. Section 7 concludes with future work directions. Full results, statistical tests, and implementation details appear in the appendix.

---

**Word count**: ~1,200 words (~2 pages at 600 words/page)
**Status**: Complete draft
**Next**: Draft Related Work section
