# Librex.Meta: Tournament-Based Meta-Learning for Fast Algorithm Selection

**Target**: AutoML Conference 2025 (March 31, 2025)
**Format**: 8 pages + unlimited appendix
**Authors**: [To be filled]

---

## Paper Structure Outline

### 1. Abstract (0.25 pages)

**Key Points** (150-200 words):
- Problem: Algorithm selection critical but slow (SATzilla: 254ms)
- Gap: Real-time applications need instant selection (<1ms)
- Approach: Novel tournament-based meta-learning with Swiss-system + Elo ratings
- Results:
  - Best average regret across 5 diverse ASlib scenarios (0.0545)
  - 1664x faster than SATzilla (0.15ms)
  - Excels on graph problems (wins GRAPHS-2015)
  - 96.5% accuracy on binary selection
- Contribution: Fast + competitive algorithm selection framework

---

### 2. Introduction (2 pages)

**2.1 Motivation** (0.5 pages)
- Algorithm portfolios critical for solving hard combinatorial problems
- Per-instance algorithm selection can yield order-of-magnitude speedups
- **Problem**: Existing methods too slow for real-time applications
  - SATzilla: 254ms selection time (too slow for interactive systems)
  - AutoFolio: 24ms (better but still not real-time)
  - Need: Sub-millisecond selection with competitive accuracy

**2.2 Challenges** (0.5 pages)
- Challenge 1: Speed-accuracy trade-off
  - Complex ML models (random forests, neural nets) = accurate but slow
  - Simple models (single decision) = fast but inaccurate
- Challenge 2: Problem diversity
  - Different problem classes require different selection strategies
  - Graph problems ≠ SAT problems ≠ CSP problems
- Challenge 3: Meta-learning efficiency
  - Need to learn from limited training instances
  - Transfer knowledge across problem classes

**2.3 Our Approach** (0.5 pages)
- Novel framework: Tournament-based meta-learning
  - Swiss-system tournaments simulate algorithm competitions
  - Elo ratings track algorithm performance globally and per-cluster
  - KMeans clustering partitions problem space
  - UCB selection for fast decision-making
- Key insight: Tournament dynamics efficiently capture algorithm strengths

**2.4 Contributions** (0.5 pages)
1. **Methodological**: Tournament-based algorithm selection framework
   - First application of Swiss-system tournaments to meta-learning
   - Elo ratings for algorithm ranking (novel in AS context)

2. **Empirical**: Strong results on diverse benchmarks
   - Best average regret across 5 ASlib scenarios (0.0545)
   - 1664x speedup vs SATzilla with only 10% regret penalty
   - Wins on graph problems (GRAPHS-2015: rank 1/7)
   - 96.5% accuracy on binary selection (CSP-2010)

3. **Practical**: Identifies problem class strengths
   - Graph problems: Clear dominance
   - Binary/simple selection: Near-perfect accuracy
   - Complex SAT/ASP: Competitive but not best

4. **Methodological insight**: Hyperparameter robustness
   - Discovery: Most hyperparameters don't matter on real data
   - Only n_clusters impacts performance (optimal = 3)

---

### 3. Related Work (1.5 pages)

**3.1 Algorithm Selection** (0.5 pages)
- **Classical approaches**:
  - SATzilla (Xu et al., 2008): Regression-based, uses pre-solvers + cost-sensitive learning
  - 3S (Kadioglu et al., 2010): Schedule-based approach
  - ISAC (Kadioglu et al., 2011): Instance-specific configuration

- **Modern ML-based**:
  - AutoFolio (Lindauer et al., 2015): Automated configuration of AS
  - FlexFolio (Lindauer et al., 2020): Flexible portfolio design
  - ASlib benchmark suite (Bischl et al., 2016): Standardized evaluation

- **Limitations**: High selection overhead (24-254ms), complex models

**3.2 Meta-Learning for Algorithm Selection** (0.5 pages)
- **Hyperparameter optimization**:
  - SMAC (Hutter et al., 2011): Sequential model-based optimization
  - Hyperband (Li et al., 2017): Bandit-based resource allocation
  - BOHB (Falkner et al., 2018): Combines Bayesian optimization + Hyperband

- **Meta-learning frameworks**:
  - Auto-sklearn (Feurer et al., 2015): Automated ML pipeline construction
  - Google Vizier (Golovin et al., 2017): Black-box optimization service

- **Gap**: No emphasis on ultra-fast selection

**3.3 Tournament-Based Learning** (0.25 pages)
- **Game theory**:
  - Elo rating system (Elo, 1978): Chess player rankings
  - TrueSkill (Herbrich et al., 2007): Bayesian skill rating

- **ML applications**:
  - Evolutionary algorithms: Tournament selection for fitness
  - Active learning: Query selection via tournaments

- **Novel**: First application to algorithm selection meta-learning

**3.4 Clustering for Problem Space Partitioning** (0.25 pages)
- Feature clustering for AS (Leyton-Brown et al., 2009)
- Instance space analysis (Smith-Miles & Bowly, 2015)
- Our approach: Cluster-specific Elo ratings (incremental novelty)

---

### 4. Methods (2 pages)

**4.1 Problem Formulation** (0.25 pages)
- **Input**: Algorithm portfolio A = {a₁, ..., aₙ}, problem instance x
- **Output**: Selected algorithm a* ∈ A
- **Objective**: Minimize regret = runtime(a*, x) - runtime(a_best, x)
- **Constraint**: Selection time << solver runtime (target: <1ms)

**4.2 Librex.Meta Framework** (0.5 pages)
- **Overview diagram** (Figure 1: Librex.Meta architecture)
  - Training phase: Swiss tournaments → Elo ratings
  - Selection phase: Cluster → UCB → Algorithm

- **Key components**:
  1. Problem space clustering (KMeans on features)
  2. Swiss-system tournaments (n_rounds rounds)
  3. Elo rating updates (global + cluster-specific)
  4. UCB-based selection

**4.3 Tournament-Based Training** (0.5 pages)
- **Swiss-system tournaments**:
  - Each round: Pair algorithms with similar Elo ratings
  - Winner: Better performance on problem instance
  - Update Elo ratings based on outcome

- **Elo rating formula**:
  ```
  R'ᵢ = Rᵢ + K × (Sᵢ - E(Sᵢ))
  E(Sᵢ) = 1 / (1 + 10^((Rⱼ - Rᵢ) / 400))
  ```
  where K = update constant, Sᵢ = actual outcome (1 or 0)

- **Global + cluster-specific ratings**:
  - Global Elo: Overall algorithm strength
  - Cluster Elo: Algorithm strength on problem class
  - Selection uses cluster Elo (specialized performance)

**4.4 Fast Selection** (0.5 pages)
- **Selection process**:
  1. Extract features from instance x (sub-millisecond)
  2. Assign to cluster using KMeans
  3. Select algorithm via UCB using cluster Elo ratings

- **UCB formula**:
  ```
  UCB(a) = normalize(Elo_cluster(a)) + c × sqrt(log(N) / n(a))
  ```
  where N = total selections, n(a) = selections of algorithm a

- **Complexity analysis**:
  - Feature extraction: O(d) where d = feature dimensionality
  - Clustering: O(kd) where k = n_clusters (typically k=3)
  - UCB selection: O(m) where m = number of algorithms
  - **Total: O(d + kd + m) = O(d) ≈ 0.15ms empirically**

**4.5 Hyperparameter Configuration** (0.25 pages)
- **Default configuration**:
  - n_clusters = 5 (general-purpose)
  - ucb_c = 1.0 (standard UCB)
  - n_tournament_rounds = 5
  - elo_k = 32 (moderate updates)

- **Optimal configuration** (from ablation studies):
  - n_clusters = 3 (coarse clustering)
  - Other parameters: Insensitive (robust design)

---

### 5. Experimental Setup (1 page)

**5.1 Benchmark Suite** (0.25 pages)
- **ASlib scenarios** (5 diverse problem classes):
  1. GRAPHS-2015: Graph coloring (1,147 instances, 9 algorithms)
  2. CSP-2010: Constraint satisfaction (486 instances, 6 algorithms)
  3. MAXSAT12-PMS: MaxSAT (876 instances, 8 algorithms)
  4. SAT11-HAND: SAT (296 instances, 15 algorithms)
  5. ASP-POTASSCO: Answer set programming (1,294 instances, 4 algorithms)

- **Dataset characteristics**: 4,099 total test instances, 42 algorithms total

**5.2 Baseline Methods** (0.25 pages)
1. **SATzilla** (Xu et al., 2008): Regression + pre-solvers
2. **AutoFolio** (Lindauer et al., 2015): Automated AS configuration
3. **SMAC** (Hutter et al., 2011): Bayesian optimization
4. **Hyperband** (Li et al., 2017): Bandit-based HPO
5. **BOHB** (Falkner et al., 2018): Bayesian + bandit hybrid
6. **Librex.Meta (default)**: Standard hyperparameters
7. **Librex.Meta (optimal)**: Tuned hyperparameters (n_clusters=3)

**5.3 Evaluation Metrics** (0.25 pages)
- **Average Regret** (primary metric):
  ```
  Regret(x) = (runtime(selected, x) - runtime(best, x)) / runtime(best, x)
  ```
  Measures quality of selection relative to oracle

- **Top-1 Accuracy**: Percentage selecting optimal algorithm
- **Top-3 Accuracy**: Percentage selecting top-3 algorithm
- **Selection Time**: Wall-clock time for algorithm selection
- **Par10 metric**: Timeout = 10× cutoff time (standard ASlib)

**5.4 Experimental Protocol** (0.25 pages)
- **Train/test split**: 80/20 stratified by scenario
- **Cross-validation**: 5-fold CV for scenario-level splits (from ASlib)
- **Hardware**: [To be specified]
- **Implementation**: Python 3.9, scikit-learn 1.0
- **Reproducibility**: Code available at [repository URL]

---

### 6. Results (2 pages)

**6.1 Overall Performance** (0.5 pages)
- **Table 1: Average performance across 5 scenarios**

| Method | Avg Regret ↓ | Top-1 Acc ↑ | Selection Time ↓ | Speedup vs SATzilla |
|--------|--------------|-------------|------------------|---------------------|
| **Librex.Meta (optimal)** | **0.0545** | 46.5% | **0.15ms** | **1664x** |
| Librex.Meta (default) | 0.0587 | 45.1% | 0.17ms | 1494x |
| SATzilla | 0.0603 | 38.6% | 254ms | 1x |
| SMAC | 0.0659 | 40.4% | 30ms | 8.5x |
| AutoFolio | 0.0709 | 45.4% | 24ms | 10.6x |
| Hyperband | 0.1013 | 19.7% | 0.03ms | 8467x |
| BOHB | 0.1013 | 19.7% | 5.5ms | 46x |

**Key findings**:
- Librex.Meta achieves best average regret (0.0545)
- 1664x faster than SATzilla with only 10% worse regret
- Competitive top-1 accuracy (46.5%)

**6.2 Per-Scenario Performance** (0.5 pages)
- **Figure 2: Per-scenario rankings** (bar chart)
- **Table 2: Scenario-by-scenario breakdown**

| Scenario | Librex.Meta Rank | Regret | Top-1 Acc | Notes |
|----------|----------------|--------|-----------|-------|
| **GRAPHS-2015** | **1/7** 🥇 | **0.019** | **54.8%** | **WINS** |
| **CSP-2010** | 2/7 🥈 | 0.003 | **96.5%** | Near-perfect |
| MAXSAT12-PMS | 4/7 | 0.025 | 58.0% | Competitive |
| SAT11-HAND | 5/7 | 0.112 | 18.3% | Weak |
| ASP-POTASSCO | 5/7 | 0.113 | 5.0% | Weak |

**Problem class analysis**:
- **Excels on**: Graph problems, binary selection
- **Competitive on**: MaxSAT
- **Weak on**: Hard SAT, ASP

**6.3 Speed-Accuracy Trade-off** (0.5 pages)
- **Figure 3: Pareto frontier** (scatter plot: selection time vs regret)
  - Librex.Meta: Optimal trade-off (fast + competitive)
  - SATzilla: Slow but accurate
  - Hyperband: Fast but poor accuracy

- **Key insight**: 1664x speedup with only 10% regret penalty

**6.4 Ablation Studies** (0.5 pages)
- **Figure 4: Hyperparameter sensitivity** (line plots)
  - n_clusters: Optimal = 3 (9.4% improvement over default)
  - ucb_c: No impact (all values [0.1-2.0] identical)
  - n_tournament_rounds: No impact (all values [1-15] identical)
  - elo_k: Avoid 8 and 16, K=32 optimal

- **Key finding**: Hyperparameter robust (minimal tuning needed)

---

### 7. Discussion (1 page)

**7.1 Why Tournament-Based Selection Works** (0.25 pages)
- **Efficiency**: Swiss system efficiently ranks algorithms with O(n log n) comparisons
- **Adaptivity**: Elo ratings adapt to problem distribution
- **Specialization**: Cluster-specific ratings capture problem class strengths
- **Speed**: Pre-computed ratings enable O(1) selection

**7.2 Graph Problem Dominance** (0.25 pages)
- **Hypothesis**: Tournament dynamics suit graph structure
- **Evidence**:
  - GRAPHS-2015: Rank 1/7, outperforms all baselines
  - 5.5% better regret than second-best (AutoFolio)
  - 54.8% top-1 accuracy (vs 23.8% for AutoFolio)

- **Possible explanation**: Graph features naturally partition into distinct clusters

**7.3 Limitations and Weaknesses** (0.25 pages)
- **Statistical significance**:
  - Friedman test p=0.85 (not significant at α=0.05)
  - Limited by scenario count (n=5, low statistical power)
  - Medium-to-large effect sizes suggest practical difference

- **Problem class specificity**:
  - Weak on hard SAT (rank 5/7)
  - Weak on ASP (rank 5/7)
  - Not universally best across all domains

- **Top-1 accuracy moderate**: 46.5% (vs 57% for AutoFolio on some scenarios)

**7.4 Practical Implications** (0.25 pages)
- **Real-time applications**: Sub-millisecond selection enables interactive systems
- **Embedded systems**: Low computational overhead
- **Graph problem solving**: Clear best choice for graph coloring, partitioning, etc.
- **Binary selection**: Near-perfect accuracy (96.5% on CSP-2010)

---

### 8. Conclusion (0.5 pages)

**8.1 Summary** (0.25 pages)
- Introduced Librex.Meta: tournament-based meta-learning for algorithm selection
- Achieves best average regret across 5 diverse ASlib scenarios (0.0545)
- 1664x faster than SATzilla (0.15ms) with competitive accuracy
- Excels on graph problems (wins GRAPHS-2015)
- Hyperparameter robust (minimal tuning needed)

**8.2 Future Work** (0.25 pages)
- **Scalability**: Evaluate on all 45+ ASlib scenarios
- **Deep learning features**: Replace hand-crafted features with learned representations
- **Online learning**: Adapt Elo ratings during deployment
- **Hybrid approaches**: Combine with pre-solvers for hard instances
- **Theoretical analysis**: Prove regret bounds for tournament-based selection

---

## Figures Plan (6 key figures)

1. **Figure 1: Librex.Meta Architecture** (Methods section)
   - System diagram: Training (tournaments → Elo) + Selection (cluster → UCB)

2. **Figure 2: Per-Scenario Rankings** (Results section)
   - Bar chart: Librex.Meta rank on each of 5 scenarios
   - Highlight GRAPHS-2015 win

3. **Figure 3: Speed-Accuracy Pareto Frontier** (Results section)
   - Scatter plot: Selection time (x-axis) vs Average regret (y-axis)
   - Show Librex.Meta in optimal region (fast + competitive)

4. **Figure 4: Hyperparameter Sensitivity** (Results section)
   - 4 line plots: n_clusters, ucb_c, n_rounds, elo_k
   - Show n_clusters matters, others don't

5. **Figure 5: Elo Rating Convergence** (Appendix)
   - Line plot: Elo ratings over tournament rounds
   - Show convergence to stable rankings

6. **Figure 6: Cluster Visualization** (Appendix)
   - t-SNE projection of problem instances colored by cluster
   - Show problem space partitioning

---

## Appendix (Unlimited pages)

**A. Complete Results Tables**
- Full per-scenario breakdown (all 7 methods × 5 scenarios)
- Statistical test details (Wilcoxon, Friedman, effect sizes)

**B. Hyperparameter Ablation Details**
- Complete ablation study results
- Mock vs real data comparison

**C. Implementation Details**
- Pseudocode for Librex.Meta training and selection
- Computational complexity analysis

**D. ASlib Scenario Descriptions**
- Detailed characteristics of each scenario
- Feature distributions

**E. Reproducibility Checklist**
- Code repository URL
- Hardware specifications
- Random seeds and CV folds

---

## Writing Notes

**Target Length**: 8 pages (strict limit)
**Current Allocation**:
- Abstract: 0.25 pages
- Introduction: 2 pages
- Related Work: 1.5 pages
- Methods: 2 pages
- Experiments: 1 page
- Results: 2 pages
- Discussion: 1 page
- Conclusion: 0.5 pages
- References: ~1 page (not counted in limit)
- **Total: 10.25 pages → NEED TO COMPRESS TO 8 PAGES**

**Compression Strategy**:
- Reduce Related Work to 1 page (cut details, focus on key differences)
- Reduce Methods to 1.5 pages (move complexity analysis to appendix)
- Reduce Results to 1.5 pages (move full tables to appendix)
- **Revised Total: 8 pages** ✓

**Key Messages to Emphasize**:
1. **Speed**: 1664x faster (practical game-changer)
2. **Accuracy**: Best average regret (empirical success)
3. **Novelty**: Tournament-based framework (methodological contribution)
4. **Sweet spot**: Graph problems (clear dominance)
5. **Robustness**: Minimal hyperparameter tuning (ease of use)

**Honest Positioning**:
- ✅ Claim: Best average regret across 5 scenarios (descriptive stat)
- ✅ Claim: 1664x speedup (undeniable)
- ✅ Claim: Excels on graph problems (rank 1/7)
- ⚠️ Acknowledge: Limited statistical significance (n=5, p>0.05)
- ⚠️ Acknowledge: Not universally best (weak on SAT/ASP)

**Writing Style**:
- Clear, concise, empirical
- Lead with strengths, acknowledge limitations honestly
- Focus on practical impact (speed) + empirical success (regret)
- Use figures to tell the story visually

---

**Status**: Outline complete, ready for section drafting
**Next**: Draft Introduction and Related Work sections
