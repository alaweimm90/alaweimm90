# 3. Methods

We introduce Librex.Meta, a tournament-based meta-learning framework for fast algorithm selection. This section describes the problem formulation, key components, and training and selection mechanisms.

## 3.1 Problem Formulation

Let A = {a₁, a₂, ..., aₘ} be a portfolio of m algorithms, and let X be a problem instance space. For each instance x ∈ X, we denote the runtime of algorithm aᵢ on x as rᵢ(x). The oracle algorithm selection policy selects:

```
a*(x) = argmin_{aᵢ ∈ A} rᵢ(x)
```

Since runtimes are unknown before execution, we must learn a selection policy π: X → A from training data. Our objective is to minimize **average regret**:

```
Regret(x) = (r_π(x)(x) - r_a*(x)(x)) / r_a*(x)(x)
```

**Constraint**: Selection must be fast (<1ms), far faster than state-of-the-art methods (24-254ms).

## 3.2 Framework Overview

Librex.Meta consists of two phases: **training** and **selection**.

**Training phase**:
1. Extract features from training instances
2. Cluster instances into k subclasses using KMeans
3. Run Swiss-system tournaments where algorithms compete
4. Update Elo ratings (global + cluster-specific) based on outcomes

**Selection phase**:
1. Extract features from new instance x
2. Assign x to cluster c using KMeans
3. Select algorithm via UCB using cluster-specific Elo ratings

The key insight is that **Elo ratings computed during training enable O(1) selection at test time**, avoiding expensive model inference.

## 3.3 Problem Space Clustering

We partition the problem space into k coherent subclasses using KMeans clustering on instance features. Let φ: X → ℝᵈ be a feature extraction function. For training instances {x₁, ..., xₙ}:

```
{c₁, ..., cₖ} = KMeans({φ(x₁), ..., φ(xₙ)}, k)
```

Clustering serves two purposes:
1. **Specialization**: Cluster-specific ratings capture algorithm strengths on problem subclasses
2. **Efficiency**: Small k ensures fast cluster assignment (O(kd))

Unlike prior work using fine-grained clustering (k=10-20), we find k=3 optimal (Section 6.4: 9.4% regret improvement).

## 3.4 Swiss-System Tournaments

For each cluster, we run Swiss-system tournaments to rank algorithms efficiently with O(m log m) comparisons, far fewer than round-robin's O(m²).

**Tournament structure**:
- **Rounds**: R rounds (default R=5)
- **Pairing**: Pair algorithms with similar Elo ratings
- **Matches**: For pair (aᵢ, aⱼ) on instance x in cluster c, the winner has lower runtime; update both Elo ratings

**Advantages**: Efficient convergence, balanced competition, adaptive to problem distribution.

## 3.5 Elo Rating System

We maintain two types of Elo ratings for each algorithm:
1. **Global Elo** (Rᵢ): Overall strength across all instances
2. **Cluster-specific Elo** (Rᵢ,c): Specialized performance in cluster c

All ratings initialize to 1500. After each match between aᵢ and aⱼ on instance x in cluster c:

```
Rᵢ ← Rᵢ + K × (Sᵢ - E(Sᵢ))
Rᵢ,c ← Rᵢ,c + K × (Sᵢ - E(Sᵢ))
```

where:
- Sᵢ = 1 if aᵢ wins (rᵢ(x) < rⱼ(x)), else 0
- E(Sᵢ) = 1 / (1 + 10^((Rⱼ - Rᵢ) / 400)) is the expected outcome
- K is the update rate (default K=32)

The Elo system updates ratings proportionally to the surprise: unexpected wins cause large rating changes.

**Dual ratings rationale**: Global Elo captures overall strength (exploration). Cluster-specific Elo captures specialized performance (exploitation during selection).

## 3.6 Fast Selection via UCB

At test time, given a new instance x, we use Upper Confidence Bound (UCB) with Elo-based priors:

1. Extract features: v = φ(x)
2. Assign to cluster: c = argminⱼ ||v - centroid_j||
3. Compute UCB scores:
   ```
   UCB(aᵢ) = normalize(Rᵢ,c) + λ × sqrt(log(N) / nᵢ)
   ```
   where normalize(Rᵢ,c) = (Rᵢ,c - 1500) / 400, N = total selections in cluster c, nᵢ = selections of aᵢ, λ = exploration constant

4. Select: a* = argmaxᵢ UCB(aᵢ)

UCB balances **exploitation** (high-Elo algorithms) with **exploration** (uncertain algorithms).

**Complexity**: O(d + kd + m) = O(d) for small k, m. Empirically, with k=3 and m≤15, selection completes in 0.15ms—**1664× faster than SATzilla**.

## 3.7 Hyperparameter Configuration

Librex.Meta has four main hyperparameters:

1. **n_clusters** (k): Number of problem subclasses (default k=5, optimal k=3)
2. **ucb_constant** (λ): Exploration weight (default λ=1.0)
3. **n_tournament_rounds** (R): Tournament iterations (default R=5)
4. **elo_k** (K): Rating update rate (default K=32)

**Robustness discovery**: Ablation studies reveal that **only n_clusters significantly impacts performance**. ucb_constant and n_tournament_rounds have zero effect on real ASlib data, despite appearing critical in synthetic experiments (Section 6.4).

We evaluate:
- **Librex.Meta (default)**: k=5, λ=1.0, R=5, K=32
- **Librex.Meta (optimal)**: k=3, λ=1.0, R=5, K=32

## 3.8 Implementation Details

**Feature extraction**: We use ASlib-provided feature extractors (<0.1ms). **KMeans clustering**: scikit-learn with k-means++ initialization (<1 second). **Tournament execution**: R rounds per cluster, pairing algorithms with similar Elo ratings (±50 points). **Elo convergence**: Ratings converge within 5-7 rounds. **Training time**: 0.1-0.5 seconds per scenario. **Code availability**: [repository URL to be added], Python 3.9 with scikit-learn 1.0 and NumPy 1.21.

---

**Word count**: ~1,000 words (~1.65 pages)
**Status**: Compressed from 1,100 words
**Compression**: -100 words (9% reduction)
