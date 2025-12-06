# Librex.Meta: Tournament-Based Meta-Learning for Fast Algorithm Selection

**Authors**: [To be filled]
**Affiliation**: [To be filled]
**Contact**: [To be filled]

**Target**: AutoML Conference 2025
**Submission Deadline**: March 31, 2025
**Format**: 8 pages + unlimited appendix

---

## Abstract

[See paper_drafts/00_ABSTRACT.md - 210 words - UNCHANGED]

Algorithm selection—choosing the best solver for a given problem instance—can yield order-of-magnitude speedups over any single algorithm. However, state-of-the-art selection methods like SATzilla require 24-254ms selection time, prohibiting real-time applications. We introduce **Librex.Meta**, a tournament-based meta-learning framework that achieves sub-millisecond selection (0.15ms) while maintaining competitive accuracy.

Librex.Meta combines Swiss-system tournaments with Elo rating systems to efficiently rank algorithms during training. We partition the problem space using KMeans clustering and maintain both global and cluster-specific Elo ratings to capture algorithm specialization. At selection time, we use Upper Confidence Bound (UCB) selection with Elo-based priors, achieving O(d) complexity where d is feature dimensionality.

Across 5 diverse ASlib scenarios spanning 4,099 test instances and 42 algorithms, Librex.Meta achieves best average regret (0.0545) compared to SATzilla (0.0603), SMAC (0.0659), AutoFolio (0.0709), and others, with **1664× speedup** over SATzilla. Librex.Meta excels on graph problems (rank 1/7 on GRAPHS-2015) and binary selection tasks (96.5% accuracy on CSP-2010), while remaining hyperparameter robust—only the number of clusters impacts performance.

Our work demonstrates that tournament-based meta-learning enables real-time algorithm selection, identifies problem classes where this approach excels, and provides a methodological lesson: hyperparameters tuned on synthetic data often fail to transfer to real benchmarks.

---

## 1. Introduction

[See paper_drafts/01_INTRODUCTION.md - 1,200 words - UNCHANGED]

Key points:
- Motivation: Algorithm selection critical but slow (SATzilla: 254ms)
- Challenges: Speed-accuracy trade-off, problem diversity, meta-learning efficiency
- Our approach: Tournament-based meta-learning with Swiss system + Elo ratings
- Contributions:
  1. Methodological (tournament framework)
  2. Empirical (best regret, 1664× speedup)
  3. Practical (identifies problem class strengths)
  4. Insight (hyperparameter robustness)

---

## 2. Related Work

[See paper_drafts/02_RELATED_WORK_compressed.md - 600 words - COMPRESSED from 900 words]

Key sections:
- Algorithm Selection: SATzilla, AutoFolio, 3S, ISAC, ASlib
- Meta-Learning: SMAC, Hyperband, BOHB, Auto-sklearn
- Tournament-Based Learning: Elo ratings, TrueSkill
- Clustering: Instance space analysis, feature clustering
- Positioning: Best regret + near-instant selection

---

## 3. Methods

[See paper_drafts/03_METHODS_compressed.md - 1,000 words - COMPRESSED from 1,100 words]

Key sections:
- 3.1 Problem Formulation: Regret minimization, <1ms constraint
- 3.2 Framework Overview: Training (tournaments → Elo) + Selection (cluster → UCB)
- 3.3 Problem Space Clustering: KMeans, k=3 optimal
- 3.4 Swiss-System Tournaments: O(m log m) comparisons
- 3.5 Elo Rating System: Dual ratings (global + cluster-specific)
- 3.6 Fast Selection via UCB: O(d) complexity
- 3.7 Hyperparameter Configuration: Robustness (only n_clusters matters)
- 3.8 Implementation Details: Training time, code availability

---

## 4. Experimental Setup

[See paper_drafts/04_EXPERIMENTS.md - 750 words - UNCHANGED]

Key sections:
- 4.1 Benchmark Suite: 5 diverse ASlib scenarios, 4,099 instances, 42 algorithms
- 4.2 Baseline Methods: 6 established methods (SATzilla, AutoFolio, SMAC, Hyperband, BOHB)
- 4.3 Evaluation Metrics: Average regret (primary), top-1/3 accuracy, selection time
- 4.4 Experimental Protocol: 80/20 splits, hardware, reproducibility

---

## 5. Results

[See paper_drafts/05_RESULTS_compressed.md - 900 words - COMPRESSED from 1,100 words]

Key findings:
- 5.1 Overall Performance: Best average regret (0.0545), 1664× speedup
- 5.2 Per-Scenario Performance: WINS on GRAPHS-2015, competitive on CSP-2010, weak on SAT/ASP
- 5.3 Speed-Accuracy Trade-off: Pareto optimal (fast + competitive)
- 5.4 Hyperparameter Sensitivity: Only n_clusters matters (k=3 optimal)
- 5.5 Statistical Significance: p>0.05 but medium-large effect sizes

**Figures**:
- Figure 2: Per-scenario performance (bar chart) ✅ GENERATED
- Figure 3: Speed-accuracy Pareto frontier (scatter plot) ✅ GENERATED
- Figure 4: Hyperparameter sensitivity (2×2 grid) ✅ GENERATED

---

## 6. Discussion

[See paper_drafts/06_DISCUSSION_compressed.md - 700 words - COMPRESSED from 900 words]

Key sections:
- 6.1 Why Tournament-Based Selection Works: Efficiency, adaptivity, specialization, simplicity
- 6.2 Graph Problem Dominance: Clean feature clustering, algorithm specialization
- 6.3 Limitations and Weaknesses: Statistical significance, problem class specificity, top-1 accuracy, mock data gap
- 6.4 Practical Implications: Real-time systems, embedded solving, graph algorithms, binary selection

---

## 7. Conclusion

[See paper_drafts/07_CONCLUSION_compressed.md - 350 words - COMPRESSED from 450 words]

Summary of contributions:
1. Methodological: Tournament-based meta-learning framework
2. Empirical: Best average regret, 1664× speedup
3. Practical: Identifies sweet spot (graph problems, binary selection)
4. Insight: Hyperparameter robustness (mock data gap)

Future work:
- Scalability (all 45+ ASlib scenarios)
- Deep learning features
- Online learning
- Hybrid approaches
- Theoretical analysis
- Alternative tournament structures

Broader impact: Real-time systems, education, accessibility, industry, research

---

## References

[To be added - BibTeX format]

Key references to include:
- Rice (1976): Algorithm selection problem
- Xu et al. (2008, 2012): SATzilla
- Lindauer et al. (2015): AutoFolio
- Hutter et al. (2011): SMAC
- Li et al. (2017): Hyperband
- Falkner et al. (2018): BOHB
- Bischl et al. (2016): ASlib
- Elo (1978): Elo rating system
- Herbrich et al. (2007): TrueSkill
- Smith-Miles & Bowly (2015): Instance space analysis

---

## Appendix (Unlimited pages)

### A. Complete Results Tables

**Table A1: Full Per-Scenario Breakdown** (all 7 methods × 5 scenarios)

[Data from results/phase2/phase2_results_summary.csv]

### B. Statistical Test Details

**Friedman Test**: χ² = 2.65, p = 0.85 (not significant)

**Wilcoxon Signed-Rank Tests**: Librex.Meta (optimal) vs. each baseline

[Data from results/phase2/statistical_tests.csv]

**Effect Sizes (Cliff's Delta)**:
- vs Hyperband/BOHB: δ = 0.44 (medium-large)
- vs AutoFolio: δ = 0.36 (medium)
- vs SATzilla: δ = 0.20 (small)
- vs SMAC: δ = 0.12 (negligible)

### C. Hyperparameter Ablation Details

**Table C1: n_clusters Ablation** (k ∈ {1, 2, 3, 5, 7, 10, 15, 20})

[Data from results/ablation_real/ablation_n_clusters_real.csv]

**Table C2: ucb_c Ablation** (c ∈ {0.1, 0.2, 0.5, 1.0, 1.5, 2.0})

[Data from results/ablation_real/ablation_ucb_constant_real.csv]

**Table C3: n_tournament_rounds Ablation** (R ∈ {1, 3, 5, 7, 10, 12, 15})

[Data from results/ablation_real/ablation_n_tournament_rounds_real.csv]

**Table C4: elo_k Ablation** (K ∈ {8, 16, 24, 32, 40, 48})

[Data from results/ablation_real/ablation_elo_k_real.csv]

### D. Implementation Details

**Pseudocode: Librex.Meta Training**
```
function TRAIN_Librex.Meta(instances, features, runtimes, k):
    # Cluster problem space
    clusters = KMeans(features, n_clusters=k)

    # Initialize Elo ratings
    for algorithm a in algorithms:
        global_elo[a] = 1500
        for cluster c in clusters:
            cluster_elo[a][c] = 1500

    # Run Swiss-system tournaments
    for round r in 1..n_rounds:
        for cluster c in clusters:
            # Pair algorithms with similar Elo
            pairs = swiss_pairing(cluster_elo[c])

            # Run matches
            for (a1, a2) in pairs:
                instance = sample_from_cluster(c)
                winner = (a1 if runtime[a1][instance] < runtime[a2][instance] else a2)

                # Update Elo ratings
                update_elo(global_elo[a1], global_elo[a2], winner)
                update_elo(cluster_elo[a1][c], cluster_elo[a2][c], winner)

    return clusters, global_elo, cluster_elo
```

**Pseudocode: Librex.Meta Selection**
```
function SELECT_ALGORITHM(instance, clusters, cluster_elo):
    # Extract features
    features = extract_features(instance)  # O(d)

    # Assign to cluster
    cluster = clusters.predict(features)  # O(kd)

    # Compute UCB scores
    for algorithm a in algorithms:
        elo_score = normalize(cluster_elo[a][cluster])
        exploration = ucb_c * sqrt(log(N) / n[a])
        ucb[a] = elo_score + exploration

    # Select best
    return argmax(ucb)  # O(m)
```

**Complexity Analysis**:
- Training: O(R × m × n) where R=rounds, m=algorithms, n=instances
- Selection: O(d + kd + m) = O(d) for small k, m

### E. ASlib Scenario Descriptions

**GRAPHS-2015**:
- Domain: Graph coloring
- Instances: 1,147 test instances
- Algorithms: 9 specialized solvers
- Features: 105 graph-theoretic features
- Performance: Librex.Meta rank 1/7 (best)

**CSP-2010**:
- Domain: Constraint satisfaction problems
- Instances: 486 test instances
- Algorithms: 6 CSP solvers
- Features: 86 constraint network features
- Performance: Librex.Meta rank 2/7, 96.5% top-1 accuracy

**MAXSAT12-PMS**:
- Domain: Partial MaxSAT optimization
- Instances: 876 test instances
- Algorithms: 8 MaxSAT solvers
- Features: 75 CNF formula features
- Performance: Librex.Meta rank 4/7

**SAT11-HAND**:
- Domain: Handcrafted SAT instances (hard industrial)
- Instances: 296 test instances
- Algorithms: 15 SAT solvers
- Features: 105 CNF formula features
- Performance: Librex.Meta rank 5/7 (weak)

**ASP-POTASSCO**:
- Domain: Answer set programming
- Instances: 1,294 test instances
- Algorithms: 4 ASP solvers
- Features: 138 program features
- Performance: Librex.Meta rank 5/7 (weak)

### F. Reproducibility Checklist

- [x] Code repository: [URL to be added]
- [x] Dataset sources: ASlib (http://www.aslib.net/)
- [x] Random seeds: Documented in code
- [x] Cross-validation folds: ASlib standard folds
- [x] Hardware specifications: Intel Xeon E5-2680 v4, 64GB RAM
- [x] Software versions: Python 3.9, scikit-learn 1.0, NumPy 1.21
- [x] Hyperparameters: Table in Appendix C
- [x] Training time: 0.1-0.5 seconds per scenario
- [x] Evaluation protocol: Described in Section 4.4

---

## Document Statistics (COMPRESSED VERSION)

**Total Word Count**: ~5,710 words

| Section | Word Count | Original | Reduction | Status |
|---------|------------|----------|-----------|--------|
| Abstract | 210 | 210 | 0 | ✓ |
| Introduction | 1,200 | 1,200 | 0 | ✓ |
| Related Work | 600 | 900 | **-300** | ✅ COMPRESSED |
| Methods | 1,000 | 1,100 | **-100** | ✅ COMPRESSED |
| Experiments | 750 | 750 | 0 | ✓ |
| Results | 900 | 1,100 | **-200** | ✅ COMPRESSED |
| Discussion | 700 | 900 | **-200** | ✅ COMPRESSED |
| Conclusion | 350 | 450 | **-100** | ✅ COMPRESSED |
| **Total** | **5,710** | **6,610** | **-900** | **~110 words over target** |

**Remaining work**: ~110 words to cut during LaTeX conversion (cross-section redundancy removal)

**Target after final polish**: 5,600 words (8 pages)

**Compression achievements**:
- ✅ Related Work: 33% reduction (900 → 600 words)
- ✅ Methods: 9% reduction (1,100 → 1,000 words)
- ✅ Results: 18% reduction (1,100 → 900 words)
- ✅ Discussion: 22% reduction (900 → 700 words)
- ✅ Conclusion: 22% reduction (450 → 350 words)
- ✅ **Total compression: 13.6% reduction (6,610 → 5,710 words)**

---

**Status**: 🎯 **COMPRESSION COMPLETE - READY FOR LATEX CONVERSION**

**Next Steps (Day 3)**:
1. Download AutoML 2025 LaTeX template
2. Convert all 8 sections to LaTeX
3. Remove final ~110 words of cross-section redundancy
4. Reach 5,600 words (8 pages)
5. Set up Overleaf project

**Progress**: Days 1-2 Complete (Compression) ✅
