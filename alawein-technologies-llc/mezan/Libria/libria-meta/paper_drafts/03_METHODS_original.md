# 3. Methods

We introduce Librex.Meta, a tournament-based meta-learning framework for fast algorithm selection. This section describes the problem formulation, our approach's key components, and the training and selection mechanisms.

## 3.1 Problem Formulation

Let A = {a₁, a₂, ..., aₘ} be a portfolio of m algorithms, and let X be a problem instance space. For each instance x ∈ X, we denote the runtime of algorithm aᵢ on x as rᵢ(x). The oracle algorithm selection policy selects:

```
a*(x) = argmin_{aᵢ ∈ A} rᵢ(x)
```

However, since runtimes are unknown before execution, we must learn a selection policy π: X → A from training data. Our objective is to minimize **average regret**:

```
Regret(x) = (r_π(x)(x) - r_a*(x)(x)) / r_a*(x)(x)
```

where π(x) is the algorithm selected by our policy for instance x. Regret measures the relative performance loss compared to the oracle.

**Constraint**: Selection must be fast. For real-time applications, we target selection time < 1ms, far faster than state-of-the-art methods (24-254ms).

## 3.2 Framework Overview

Librex.Meta consists of two phases: **training** and **selection**.

**Training phase**:
1. Extract features from training instances
2. Cluster instances into k problem subclasses using KMeans
3. Run Swiss-system tournaments where algorithms compete on instances
4. Update Elo ratings (global + cluster-specific) based on match outcomes

**Selection phase**:
1. Extract features from new instance x
2. Assign x to cluster c using KMeans
3. Select algorithm via UCB using cluster-specific Elo ratings

The key insight is that **Elo ratings computed during training enable O(1) selection at test time**, avoiding expensive model inference.

Figure 1 illustrates the complete framework architecture.

## 3.3 Problem Space Clustering

We partition the problem space into k coherent subclasses using KMeans clustering on instance features. Let φ: X → ℝᵈ be a feature extraction function mapping instances to d-dimensional feature vectors. For training instances {x₁, ..., xₙ}, we compute:

```
{c₁, ..., cₖ} = KMeans({φ(x₁), ..., φ(xₙ)}, k)
```

Each instance is assigned to its nearest cluster centroid. Clustering serves two purposes:

1. **Specialization**: Algorithms may excel on specific problem subclasses. Cluster-specific ratings capture these strengths.
2. **Efficiency**: Small k (we use k=3) ensures fast cluster assignment at selection time: O(kd).

Unlike prior work using fine-grained clustering (k=10-20), we find coarse clustering optimal. Section 6.4 shows k=3 outperforms k=20 by 9.4% regret.

## 3.4 Swiss-System Tournaments

For each cluster, we run Swiss-system tournaments to rank algorithms. Swiss tournaments efficiently determine rankings with O(m log m) comparisons, far fewer than round-robin's O(m²).

**Tournament structure**:
- **Rounds**: We run R rounds (default R=5)
- **Pairing**: In each round, pair algorithms with similar current Elo ratings
- **Matches**: For a pair (aᵢ, aⱼ) matched on instance x in cluster c:
  - Winner: algorithm with lower runtime (rᵢ(x) < rⱼ(x))
  - Update both algorithms' Elo ratings based on outcome

**Why Swiss-system?**
- Efficient: O(m log m) comparisons converge to stable rankings
- Balanced: Each algorithm competes against similarly-rated opponents
- Adaptive: Ratings adjust quickly to problem distribution

## 3.5 Elo Rating System

We maintain two types of Elo ratings for each algorithm:

1. **Global Elo** (Rᵢ): Overall algorithm strength across all instances
2. **Cluster-specific Elo** (Rᵢ,c): Specialized performance in cluster c

All ratings initialize to 1500 (standard Elo convention). After each match between algorithms aᵢ and aⱼ on instance x in cluster c, we update:

**Global Elo update**:
```
Rᵢ ← Rᵢ + K × (Sᵢ - E(Sᵢ))
Rⱼ ← Rⱼ + K × (Sⱼ - E(Sⱼ))
```

**Cluster-specific Elo update**:
```
Rᵢ,c ← Rᵢ,c + K × (Sᵢ - E(Sᵢ))
Rⱼ,c ← Rⱼ,c + K × (Sⱼ - E(Sⱼ))
```

where:
- Sᵢ = 1 if aᵢ wins (rᵢ(x) < rⱼ(x)), else 0
- E(Sᵢ) = 1 / (1 + 10^((Rⱼ - Rᵢ) / 400)) is the expected outcome
- K is the update rate (default K=32)

The expected outcome E(Sᵢ) is the probability that aᵢ wins based on current rating difference. The Elo system updates ratings proportionally to the surprise: if a low-rated algorithm beats a high-rated one, ratings change significantly.

**Dual ratings rationale**: Global Elo captures overall algorithm strength (useful for exploration). Cluster-specific Elo captures specialized performance (used for exploitation during selection). This design balances generalization and specialization.

## 3.6 Fast Selection via UCB

At test time, given a new instance x, we select an algorithm using Upper Confidence Bound (UCB) with Elo-based priors:

**Selection procedure**:
1. Extract features: v = φ(x)
2. Assign to cluster: c = argminⱼ ||v - centroid_j||
3. Compute UCB scores for each algorithm in cluster c:
   ```
   UCB(aᵢ) = normalize(Rᵢ,c) + λ × sqrt(log(N) / nᵢ)
   ```
   where:
   - normalize(Rᵢ,c) = (Rᵢ,c - 1500) / 400 maps Elo to [0,1] range
   - N = total selections made in cluster c
   - nᵢ = selections of algorithm aᵢ in cluster c
   - λ is the exploration constant (default λ=1.0)

4. Select: a* = argmaxᵢ UCB(aᵢ)

**Intuition**: UCB balances **exploitation** (choose high-Elo algorithms) with **exploration** (try algorithms we're uncertain about). The first term exploits cluster-specific performance; the second term encourages exploration of underutilized algorithms.

**Complexity analysis**:
- Feature extraction: O(d) where d = feature dimensionality
- Cluster assignment: O(kd) where k = number of clusters
- UCB computation: O(m) where m = number of algorithms
- **Total: O(d + kd + m) = O(d)** for small k, m

Empirically, with k=3 and m≤15, selection completes in 0.15ms on average—**1664× faster than SATzilla**.

## 3.7 Hyperparameter Configuration

Librex.Meta has four main hyperparameters:

1. **n_clusters** (k): Number of problem subclasses
   - Default: k=5 (general-purpose)
   - Optimal: k=3 (from ablation studies)

2. **ucb_constant** (λ): Exploration weight in UCB
   - Default: λ=1.0 (standard UCB)
   - Finding: No impact on real data (Section 6.4)

3. **n_tournament_rounds** (R): Tournament iterations
   - Default: R=5
   - Finding: No impact on real data (Section 6.4)

4. **elo_k** (K): Rating update rate
   - Default: K=32 (moderate updates)
   - Finding: Broad range works well; avoid K=8, 16

**Robustness discovery**: Ablation studies reveal that **only n_clusters significantly impacts performance**. ucb_constant and n_tournament_rounds have zero effect on real ASlib data, despite appearing critical in synthetic experiments. This hyperparameter robustness is a practical advantage—minimal tuning needed.

We evaluate two configurations:
- **Librex.Meta (default)**: k=5, λ=1.0, R=5, K=32
- **Librex.Meta (optimal)**: k=3, λ=1.0, R=5, K=32

Section 6.4 analyzes hyperparameter sensitivity in detail.

## 3.8 Implementation Details

**Feature extraction**: We use the feature extractors provided by ASlib scenarios, which compute instance-specific characteristics (e.g., clause-to-variable ratio for SAT, graph density for graph problems). Feature extraction is fast (<0.1ms) for all scenarios.

**KMeans clustering**: We use scikit-learn's KMeans with k-means++ initialization. Clustering training instances takes <1 second for all scenarios.

**Tournament execution**: For each cluster, we iterate R rounds. In round r, we pair algorithms with similar Elo ratings (within ±50 points) and randomly sample instances from the cluster for matches. Each algorithm competes in at least one match per round.

**Elo convergence**: Ratings typically converge within 5-7 rounds (see Appendix Figure 5). Additional rounds yield diminishing returns.

**Training time**: Total training (clustering + tournaments) completes in 0.1-0.5 seconds per scenario, making Librex.Meta practical for retraining.

**Code availability**: Our implementation is available at [repository URL to be added], built on Python 3.9 with scikit-learn 1.0 and NumPy 1.21.

---

**Word count**: ~1,100 words (~1.8 pages)
**Status**: Complete draft (slightly over target, can compress to 1.5 pages)
**Next**: Draft Experimental Setup section
