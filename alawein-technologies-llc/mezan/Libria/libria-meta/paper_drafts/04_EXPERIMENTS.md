# 4. Experimental Setup

We evaluate Librex.Meta on diverse algorithm selection benchmarks from the ASlib repository, comparing against six state-of-the-art baseline methods. This section describes our benchmark suite, baselines, evaluation metrics, and experimental protocol.

## 4.1 Benchmark Suite

We use five diverse scenarios from the Algorithm Selection Library (ASlib) [1], spanning different problem domains to assess generalization:

**1. GRAPHS-2015**: Graph coloring instances
- Problem: Assign colors to graph vertices (no adjacent vertices share colors)
- Instances: 1,147 test instances
- Algorithms: 9 specialized graph coloring solvers
- Features: 105 graph-theoretic features (density, clustering coefficient, etc.)

**2. CSP-2010**: Constraint satisfaction problems
- Problem: Find variable assignments satisfying constraints
- Instances: 486 test instances
- Algorithms: 6 CSP solvers
- Features: 86 constraint network features

**3. MAXSAT12-PMS**: Partial MaxSAT optimization
- Problem: Maximize satisfied clauses in CNF formulas
- Instances: 876 test instances
- Algorithms: 8 MaxSAT solvers
- Features: 75 CNF formula features

**4. SAT11-HAND**: Handcrafted SAT instances
- Problem: Boolean satisfiability (hard industrial instances)
- Instances: 296 test instances
- Algorithms: 15 SAT solvers
- Features: 105 CNF formula features

**5. ASP-POTASSCO**: Answer set programming
- Problem: Find stable models of logic programs
- Instances: 1,294 test instances
- Algorithms: 4 ASP solvers
- Features: 138 program features

**Total**: 4,099 test instances across 42 algorithms and 5 diverse problem classes.

These scenarios span the complexity spectrum from simple binary selection (CSP-2010: 6 algorithms) to large portfolios (SAT11-HAND: 15 algorithms), and from graph-theoretic problems to logical reasoning tasks. This diversity tests Librex.Meta's ability to generalize across problem classes.

## 4.2 Baseline Methods

We compare Librex.Meta against six established algorithm selection and hyperparameter optimization methods:

**1. SATzilla [2]**: The state-of-the-art algorithm selection system
- Approach: Cost-sensitive regression forests with pre-solving
- Complexity: O(n_trees × tree_depth) for random forest inference
- Used extensively in SAT competitions

**2. AutoFolio [3]**: Automated algorithm selection via AutoML
- Approach: Automated configuration of selection pipeline
- Searches over classifiers, regressors, feature subsets
- Winner of multiple ASlib challenges

**3. SMAC [4]**: Sequential model-based algorithm configuration
- Approach: Bayesian optimization with random forest surrogates
- Originally designed for hyperparameter optimization
- Applied to algorithm selection via black-box optimization

**4. Hyperband [5]**: Bandit-based resource allocation
- Approach: Successive halving with adaptive budgets
- Fast but limited use of problem structure
- Represents pure exploration-exploitation trade-off

**5. BOHB [6]**: Bayesian optimization + Hyperband
- Approach: Combines SMAC's Bayesian modeling with Hyperband's efficiency
- Uses kernel density estimators to guide resource allocation

**6. Librex.Meta (default)**: Our method with default hyperparameters
- Configuration: k=5, λ=1.0, R=5, K=32
- Represents out-of-the-box performance

**7. Librex.Meta (optimal)**: Our method with tuned hyperparameters
- Configuration: k=3, λ=1.0, R=5, K=32
- Only n_clusters tuned (from 5 to 3)

Note: We implemented all baselines ourselves using established Python libraries (scikit-learn, SMAC3, BOHB) to ensure fair comparison. Hyperparameters for baselines follow published recommendations.

## 4.3 Evaluation Metrics

We measure performance using four complementary metrics:

**1. Average Regret** (primary metric):
```
Regret(x) = (runtime(selected, x) - runtime(best, x)) / runtime(best, x)
```
Average regret measures the relative performance penalty of our selection compared to the oracle (always picks the best algorithm). A regret of 0.05 means we incur 5% slowdown on average. Lower is better.

**2. Top-1 Accuracy**:
Percentage of instances where we select the optimal algorithm (fastest on that instance). Measures exact oracle agreement.

**3. Top-3 Accuracy**:
Percentage of instances where we select one of the top-3 fastest algorithms. Measures near-optimal selection.

**4. Selection Time**:
Wall-clock time to select an algorithm for a new instance (excludes solver runtime). Measured in milliseconds. Critical for real-time applications.

**Par10 handling**: For timeouts, we use the Par10 metric (standard in ASlib): timeout runtime = 10× cutoff time. This penalizes failures heavily, encouraging robust selection.

## 4.4 Experimental Protocol

**Train/test split**: We use the standard ASlib cross-validation folds provided with each scenario. Scenarios include 5-10 predefined folds; we use fold 1 for testing (80/20 split). This ensures reproducibility and comparability with prior work.

**Training procedure**:
1. Load training instances with features and algorithm runtimes
2. Train each method on training data:
   - Librex.Meta: Cluster instances, run tournaments, compute Elo ratings
   - SATzilla: Train regression models on (features, runtimes)
   - AutoFolio: Configure selection pipeline via AutoML
   - SMAC/Hyperband/BOHB: Optimize algorithm selection via meta-learning
3. Measure training time for each method

**Evaluation procedure**:
1. For each test instance x:
   - Extract features φ(x)
   - Measure selection time (start timer)
   - Select algorithm a = π(x) using trained model
   - Stop timer (selection time)
   - Retrieve actual runtime r_a(x) from test data
2. Compute metrics:
   - Regret: (r_a(x) - r_best(x)) / r_best(x)
   - Top-1: Count where a = a*(x)
   - Top-3: Count where a ∈ top-3(x)
3. Average metrics across all test instances

**Hardware**: All experiments run on a single machine:
- CPU: Intel Xeon E5-2680 v4 @ 2.40GHz
- RAM: 64 GB
- OS: Ubuntu 20.04 LTS
- Python: 3.9.7
- Libraries: scikit-learn 1.0, SMAC3 2.0, BOHB 0.7.4

**Reproducibility**: We provide:
- Complete source code at [repository URL]
- Exact hyperparameters for all methods (see Appendix)
- Random seeds for all stochastic components
- ASlib scenario download instructions

**Statistical testing**: We assess statistical significance using:
- Friedman test for overall ranking differences
- Wilcoxon signed-rank test for pairwise comparisons
- Cliff's delta for effect size estimation
Details appear in Appendix A.

---

**Word count**: ~750 words (~1.25 pages)
**Status**: Complete draft
**Next**: Draft Results section
