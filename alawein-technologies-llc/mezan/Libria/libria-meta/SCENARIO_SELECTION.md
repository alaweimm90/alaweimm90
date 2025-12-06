# ASlib Scenario Selection for Librex.Meta Evaluation

**Total Available**: 46 scenarios
**Target for Evaluation**: 12-15 scenarios (diverse, representative)

---

## Selection Criteria

1. **Domain Diversity**: Cover SAT, CSP, QBF, MaxSAT, ASP, Graphs, MIP
2. **Problem Difficulty**: Mix of easy, medium, hard instances
3. **Dataset Size**: Mix of small (<500 instances) and large (>500 instances)
4. **Number of Algorithms**: Range from 5-50 solvers
5. **Research Impact**: Well-studied scenarios from literature
6. **Data Quality**: Complete feature and performance data

---

## Selected Scenarios (Tier 1 - Primary Evaluation)

### 1. SAT11-HAND
- **Domain**: Boolean Satisfiability (hand-crafted instances)
- **Instances**: ~300
- **Algorithms**: ~20 SAT solvers
- **Features**: ~138 instance features
- **Rationale**: Classic SAT benchmark, widely studied
- **Difficulty**: Medium

### 2. SAT11-INDU
- **Domain**: Boolean Satisfiability (industrial instances)
- **Instances**: ~300
- **Algorithms**: ~20 SAT solvers
- **Features**: ~138 instance features
- **Rationale**: Real-world SAT problems
- **Difficulty**: Hard

### 3. SAT12-ALL
- **Domain**: Boolean Satisfiability (all tracks combined)
- **Instances**: ~1000
- **Algorithms**: ~30+ SAT solvers
- **Features**: ~138 instance features
- **Rationale**: Large-scale comprehensive benchmark
- **Difficulty**: Mixed

### 4. CSP-2010
- **Domain**: Constraint Satisfaction Problems
- **Instances**: ~200
- **Algorithms**: ~8 CSP solvers
- **Features**: ~50 features
- **Rationale**: Classic CSP benchmark
- **Difficulty**: Medium

### 5. CSP-MZN-2013
- **Domain**: MiniZinc CSP instances
- **Instances**: ~150
- **Algorithms**: ~10 solvers
- **Features**: ~100+ features
- **Rationale**: Modeling language benchmark
- **Difficulty**: Medium-Hard

### 6. QBF-2011
- **Domain**: Quantified Boolean Formulas
- **Instances**: ~475
- **Algorithms**: ~10 QBF solvers
- **Features**: ~46 features
- **Rationale**: Classic QBF benchmark
- **Difficulty**: Hard

### 7. QBF-2014
- **Domain**: Quantified Boolean Formulas
- **Instances**: ~500
- **Algorithms**: ~12 QBF solvers
- **Features**: ~50 features
- **Rationale**: Recent QBF benchmark
- **Difficulty**: Hard

### 8. MAXSAT12-PMS
- **Domain**: Maximum Satisfiability (Partial MaxSAT)
- **Instances**: ~200
- **Algorithms**: ~15 MaxSAT solvers
- **Features**: ~60 features
- **Rationale**: MaxSAT optimization
- **Difficulty**: Medium

### 9. ASP-POTASSCO
- **Domain**: Answer Set Programming
- **Instances**: ~1000
- **Algorithms**: ~8 ASP solvers
- **Features**: ~138 features
- **Rationale**: Logic programming benchmark
- **Difficulty**: Medium

### 10. GRAPHS-2015
- **Domain**: Graph problems
- **Instances**: ~200
- **Algorithms**: ~6 graph solvers
- **Features**: ~30 graph features
- **Rationale**: Graph algorithm selection
- **Difficulty**: Easy-Medium

### 11. MIP-2016
- **Domain**: Mixed Integer Programming
- **Instances**: ~200
- **Algorithms**: ~5 MIP solvers
- **Features**: ~100+ features
- **Rationale**: Optimization benchmark
- **Difficulty**: Medium-Hard

### 12. BNSL-2016
- **Domain**: Bayesian Network Structure Learning
- **Instances**: ~100
- **Algorithms**: ~5 solvers
- **Features**: ~60 features
- **Rationale**: Machine learning application
- **Difficulty**: Medium

---

## Selected Scenarios (Tier 2 - Secondary Evaluation)

If time permits, evaluate on:

### 13. SAT15-INDU
- Recent industrial SAT instances
- ~300 instances, ~40 solvers

### 14. MAXSAT15-PMS-INDU
- Industrial MaxSAT instances
- ~200 instances, ~12 solvers

### 15. CPMP-2015
- Complex scheduling problems
- ~100 instances, ~6 solvers

---

## Scenario Properties Summary

| Scenario | Domain | Instances | Algorithms | Features | Difficulty |
|----------|--------|-----------|------------|----------|------------|
| SAT11-HAND | SAT | ~300 | ~20 | 138 | Medium |
| SAT11-INDU | SAT | ~300 | ~20 | 138 | Hard |
| SAT12-ALL | SAT | ~1000 | ~30 | 138 | Mixed |
| CSP-2010 | CSP | ~200 | ~8 | 50 | Medium |
| CSP-MZN-2013 | CSP | ~150 | ~10 | 100+ | Med-Hard |
| QBF-2011 | QBF | ~475 | ~10 | 46 | Hard |
| QBF-2014 | QBF | ~500 | ~12 | 50 | Hard |
| MAXSAT12-PMS | MaxSAT | ~200 | ~15 | 60 | Medium |
| ASP-POTASSCO | ASP | ~1000 | ~8 | 138 | Medium |
| GRAPHS-2015 | Graph | ~200 | ~6 | 30 | Easy-Med |
| MIP-2016 | MIP | ~200 | ~5 | 100+ | Med-Hard |
| BNSL-2016 | ML | ~100 | ~5 | 60 | Medium |

**Total Instances**: ~4,625
**Average Algorithms**: ~13.6 per scenario
**Domain Coverage**: 8 different problem domains

---

## Evaluation Strategy

### Phase 1: Quick Validation (2-3 scenarios)
- SAT11-HAND (medium difficulty, well-studied)
- CSP-2010 (smaller, different domain)
- GRAPHS-2015 (easy baseline)

**Purpose**: Validate evaluation pipeline, debug issues

### Phase 2: Core Evaluation (6-8 scenarios)
- Add: SAT11-INDU, QBF-2011, MAXSAT12-PMS, ASP-POTASSCO, MIP-2016

**Purpose**: Comprehensive cross-domain evaluation

### Phase 3: Large-Scale (12 scenarios)
- Add: SAT12-ALL, QBF-2014, CSP-MZN-2013, BNSL-2016

**Purpose**: Full benchmark for paper

### Phase 4: Extended (if time permits)
- Add Tier 2 scenarios

---

## Expected Computational Cost

### Per Scenario Estimate:
- 6 methods × 10-fold CV × ~200 instances (avg) = ~12,000 evaluations
- Assuming 1 second per evaluation = ~3.3 hours per scenario

### Total Time Estimate:
- **Phase 1** (3 scenarios): ~10 hours
- **Phase 2** (8 scenarios): ~26 hours
- **Phase 3** (12 scenarios): ~40 hours

**Parallelization**: Can run multiple folds in parallel
**Optimization**: Cache feature extraction, use pre-computed data

---

## Success Metrics

For each scenario, we will compute:

1. **Par10 Score**: Penalized average runtime
   - Lower is better
   - Target: 20-30% improvement over SATzilla

2. **Top-k Accuracy**: Fraction where best solver in top-k predictions
   - Top-1 accuracy: Direct selection
   - Top-3 accuracy: Target 70-80%
   - Top-5 accuracy: Should be high (>85%)

3. **Computational Overhead**: Time to select vs. time to solve
   - Target: <5% of total solving time

4. **Online Learning Performance**: Improvement over time
   - Plot learning curves
   - Compare initial vs. final performance

---

## Statistical Significance Testing

For each pair of methods (Librex.Meta vs. Baseline):

1. **Wilcoxon Signed-Rank Test**
   - Paired comparison on each instance
   - p-value < 0.05 for significance

2. **Friedman Test**
   - Rank all methods on each instance
   - Test for differences across all methods

3. **Post-hoc Nemenyi Test**
   - Pairwise comparisons with correction
   - Critical difference diagrams

4. **Effect Size**
   - Cohen's d for practical significance
   - Cliff's delta for non-parametric effect size

---

## Ablation Study Plan (Librex.Meta)

Test impact of key hyperparameters:

### 1. Number of Clusters
- Values: 1, 3, 5, 10, 20
- Expected: 3-5 optimal for most scenarios

### 2. UCB Exploration Constant (c)
- Values: 0.5, 1.0, 1.414 (√2), 2.0
- Expected: 1.414 optimal (theoretical)

### 3. Tournament Rounds
- Values: 1, 3, 5, 10, 20
- Expected: 3-5 sufficient

### 4. Elo K-Factor
- Values: 16, 32, 64, 128
- Expected: 32 optimal (chess standard)

### 5. Clustering Algorithm
- Values: KMeans, DBSCAN, Hierarchical
- Expected: KMeans sufficient

Each ablation on 3-5 representative scenarios.

---

## Deliverables

1. **Benchmark Results Table**
   - All methods × all scenarios
   - Par10, top-k accuracy, overhead

2. **Statistical Analysis**
   - Significance tests
   - Critical difference diagrams
   - Effect sizes

3. **Visualizations**
   - Performance profiles
   - Scatter plots (Librex.Meta vs. each baseline)
   - Learning curves (online methods)
   - Ablation study results

4. **Analysis Report**
   - Strengths and weaknesses
   - Domain-specific insights
   - Recommendations

---

## Timeline

- **Week 3.1** (Current): Scenario selection ✅
- **Week 3.2**: Implement evaluation pipeline
- **Week 3.3**: Run Phase 1-2 evaluation
- **Week 3.4**: Statistical analysis
- **Week 4.1**: Ablation studies
- **Week 4.2**: Visualizations and reporting

**Target Completion**: End of Week 4
