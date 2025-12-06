# Method Taxonomy and Classification

## Overview

This document provides a comprehensive taxonomy of optimization methods in Librex, classifying them by type, applicability, and performance characteristics.

## Classification Hierarchy

```
Optimization Methods
├── Baseline Methods (Production-Ready)
│   ├── Random Sampling
│   ├── Local Search Methods
│   ├── Metaheuristics
│   └── Population-Based Methods
├── Novel Methods (Research-Grade)
│   ├── Spectral Methods (Deprecated)
│   └── Advanced Techniques (Experimental)
└── Hybrid Methods (Planned)
    ├── Multi-Start Strategies
    └── Adaptive Combinations
```

## Baseline Methods (Production-Ready)

### 1. Random Search

**Classification:**
- **Category:** Stochastic Sampling
- **Type:** Baseline / Benchmark
- **Search Strategy:** Global, random
- **Memory:** None (memoryless)

**Characteristics:**
- **Time Complexity:** O(n · iterations)
- **Space Complexity:** O(n)
- **Convergence:** Not guaranteed
- **Optimality:** Probabilistic

**Best Applications:**
- Quick baseline establishment
- Very small problems (n < 10)
- When no domain knowledge available
- Benchmark comparison

**Hyperparameters:**
- `iterations`: Number of random samples (default: 1000)
- `seed`: Random seed for reproducibility

**Implementation:** `Librex.methods.baselines.random_search`

**Status:** ✅ Production-ready

---

### 2. Simulated Annealing

**Classification:**
- **Category:** Metaheuristic
- **Type:** Trajectory-based
- **Search Strategy:** Probabilistic hill climbing
- **Memory:** None (Markovian)
- **Inspiration:** Statistical mechanics (annealing process)

**Characteristics:**
- **Time Complexity:** O(n² · iterations)
- **Space Complexity:** O(n)
- **Convergence:** Asymptotic (with proper cooling)
- **Optimality:** Global optimum with infinite iterations

**Best Applications:**
- General-purpose optimization
- Medium-sized problems (10 ≤ n < 50)
- Problems with many local optima
- When good (not optimal) solutions needed quickly

**Hyperparameters:**
- `iterations`: Number of iterations (default: 10000)
- `initial_temp`: Starting temperature (default: 100.0)
- `cooling_rate`: Temperature decay (default: 0.95)
- `seed`: Random seed

**Mathematical Foundation:**
- Acceptance probability: P(accept) = exp(-ΔE / T)
- Cooling schedule: T(k) = T₀ · α^k
- Metropolis criterion for worse solutions

**Implementation:** `Librex.methods.baselines.simulated_annealing`

**Status:** ✅ Production-ready

---

### 3. Local Search (Hill Climbing)

**Classification:**
- **Category:** Local Search
- **Type:** Deterministic / Multi-start
- **Search Strategy:** Greedy improvement
- **Memory:** None (stateless between restarts)

**Characteristics:**
- **Time Complexity:** O(n³ · restarts) for 2-opt
- **Space Complexity:** O(n)
- **Convergence:** To local optimum (guaranteed)
- **Optimality:** Local only

**Best Applications:**
- Finding local optima quickly
- Solution refinement after global search
- Problems with good initial solutions
- Real-time optimization constraints

**Hyperparameters:**
- `max_iterations`: Iteration limit per restart (default: 5000)
- `restarts`: Number of random restarts (default: 10)
- `seed`: Random seed

**Neighborhood Structure:**
- 2-opt: Swap two elements
- Explores O(n²) neighbors per iteration

**Implementation:** `Librex.methods.baselines.local_search`

**Status:** ✅ Production-ready

---

### 4. Genetic Algorithm

**Classification:**
- **Category:** Evolutionary Algorithm
- **Type:** Population-based
- **Search Strategy:** Selection + Recombination + Mutation
- **Memory:** Population history (implicit)
- **Inspiration:** Biological evolution

**Characteristics:**
- **Time Complexity:** O(n² · population · generations)
- **Space Complexity:** O(n · population)
- **Convergence:** Not guaranteed
- **Optimality:** Good diversity exploration

**Best Applications:**
- Large search spaces (n > 20)
- Multi-modal landscapes
- When population diversity beneficial
- Parallel implementation possible

**Hyperparameters:**
- `population_size`: Number of individuals (default: 100)
- `generations`: Number of generations (default: 100)
- `mutation_rate`: Probability of mutation (default: 0.1)
- `crossover_rate`: Probability of crossover (default: 0.8)
- `seed`: Random seed

**Genetic Operators:**
- **Selection:** Tournament selection (size 2)
- **Crossover:** Order crossover (OX) for permutations
- **Mutation:** Swap mutation

**Implementation:** `Librex.methods.baselines.genetic_algorithm`

**Status:** ✅ Production-ready

---

### 5. Tabu Search

**Classification:**
- **Category:** Metaheuristic
- **Type:** Trajectory-based with memory
- **Search Strategy:** Forbidden move memory
- **Memory:** Tabu list of recent moves
- **Inspiration:** Intelligent search with prohibition

**Characteristics:**
- **Time Complexity:** O(n² · iterations)
- **Space Complexity:** O(tabu_tenure)
- **Convergence:** Not guaranteed
- **Optimality:** Escapes local optima via memory

**Best Applications:**
- Avoiding cycling in local optima
- Medium to large problems
- When diversification from local search needed
- Problems with plateau regions

**Hyperparameters:**
- `iterations`: Number of iterations (default: 5000)
- `tabu_tenure`: Length of tabu list (default: 20)
- `seed`: Random seed

**Tabu Mechanism:**
- Maintains set of forbidden (swap_i, swap_j) moves
- Aspiration criterion: Accept tabu move if best solution
- Aging: Removes oldest tabu entries when list full

**Implementation:** `Librex.methods.baselines.tabu_search`

**Status:** ✅ Production-ready

---

## Novel Methods (Research-Grade)

### 1. FFT-Laplace Preconditioning

**Classification:**
- **Category:** Spectral Method
- **Type:** Continuous relaxation
- **Search Strategy:** Gradient-based with preconditioning
- **Memory:** None

**Status:** ⚠️ **DEPRECATED** - Under mathematical review

**Issues Identified:**
- Spectral Laplacian designed for continuous domains
- FFT transformation lacks justification for discrete optimization
- Preconditioner formula mathematically unsound for combinatorial problems

**Recommendation:** DO NOT USE - Use baseline methods instead

**Implementation:** `Librex.methods.novel.fft_laplace` (raises NotImplementedError)

---

### 2. Reverse-Time Saddle Escape

**Classification:**
- **Category:** Advanced Metaheuristic
- **Type:** Saddle point detection and escape
- **Search Strategy:** Negative curvature exploitation
- **Memory:** Gradient/Hessian history

**Status:** 🧪 Experimental (not included in this release)

**Theoretical Foundation:**
- Time-reversed gradient dynamics
- Eigenvalue-based saddle detection
- Reverse-time continuation for escape

**Potential Applications:**
- Non-convex optimization
- Neural network training
- Problems with many saddle points

---

## Method Comparison Matrix

| Method | Time | Space | Global Opt | Memory | Best For |
|--------|------|-------|------------|--------|----------|
| **Random Search** | O(n·it) | O(n) | ❌ | None | Baseline, n<10 |
| **Simulated Annealing** | O(n²·it) | O(n) | ✓* | None | General, n<50 |
| **Local Search** | O(n³·r) | O(n) | ❌ | None | Fast local opt |
| **Genetic Algorithm** | O(n²·p·g) | O(n·p) | ❌ | Pop | Large n, diverse |
| **Tabu Search** | O(n²·it) | O(t) | ❌ | List | Escape local opt |

*With proper cooling schedule and infinite iterations

**Legend:**
- it = iterations, r = restarts, p = population, g = generations, t = tabu tenure
- ✓* = asymptotically guaranteed

## Selection Guidelines

### By Problem Size

**Tiny (n < 10):**
1. Random Search (fast baseline)
2. Local Search (if good starting point)

**Small (10 ≤ n < 30):**
1. Simulated Annealing (best general-purpose)
2. Tabu Search (if local optima problematic)
3. Local Search (for refinement)

**Medium (30 ≤ n < 100):**
1. Genetic Algorithm (population diversity helps)
2. Simulated Annealing (longer runs)
3. Tabu Search (memory-based exploration)

**Large (n ≥ 100):**
1. Genetic Algorithm (with larger population)
2. Hybrid methods (planned)
3. Parallel implementations (future work)

### By Problem Characteristics

**Many Local Optima:**
- Simulated Annealing (escapes via temperature)
- Tabu Search (escapes via memory)
- Genetic Algorithm (population diversity)

**Smooth Landscape:**
- Local Search (fast convergence)
- Simulated Annealing (low temperature)

**Plateau Regions:**
- Tabu Search (explores plateaus systematically)
- Genetic Algorithm (mutation for exploration)

**Time Constrained:**
- Random Search (quick results)
- Local Search (fast to local optimum)

**Quality Critical:**
- Simulated Annealing (long runs)
- Genetic Algorithm (diverse search)

## Domain-Specific Recommendations

### Quadratic Assignment Problem (QAP)

**Recommended Methods:**
1. **Simulated Annealing** (default) - Best general performance
2. **Tabu Search** - Good for large instances
3. **Genetic Algorithm** - For very large instances (n > 50)

**Typical Hyperparameters:**
```python
# Small QAP (n < 20)
simulated_annealing(iterations=10000, initial_temp=100, cooling_rate=0.95)

# Medium QAP (20 ≤ n < 50)
tabu_search(iterations=20000, tabu_tenure=30)

# Large QAP (n ≥ 50)
genetic_algorithm(population_size=200, generations=500, mutation_rate=0.15)
```

### Traveling Salesman Problem (TSP)

**Recommended Methods:**
1. **Local Search with 2-opt** - Excellent for TSP
2. **Genetic Algorithm with order crossover** - Respects tour structure
3. **Simulated Annealing** - General-purpose fallback

**Typical Hyperparameters:**
```python
# Small TSP (n < 30)
local_search(max_iterations=10000, restarts=20)

# Medium TSP (30 ≤ n < 100)
genetic_algorithm(population_size=150, generations=300, crossover_rate=0.9)

# Large TSP (n ≥ 100)
genetic_algorithm(population_size=300, generations=1000)
```

## Hybrid Strategy Recommendations (Future)

### Multi-Phase Approach
```
Phase 1: Genetic Algorithm (global exploration)
  └─> Diverse population

Phase 2: Simulated Annealing (refinement)
  └─> Best individuals from GA

Phase 3: Local Search (final polish)
  └─> Best solution from SA
```

### Adaptive Method Selection
```
IF problem_size < 30:
    USE simulated_annealing
ELSE IF time_budget_low:
    USE genetic_algorithm (parallel)
ELSE:
    USE multi_phase_hybrid
```

## Performance Benchmarks

### QAPLIB Instances (Target Performance)

**Small (n=12-20):**
- Time: < 1 second
- Gap: Optimal or < 1%
- Method: Simulated Annealing or Local Search

**Medium (n=30-50):**
- Time: < 30 seconds
- Gap: < 5%
- Method: Tabu Search or Genetic Algorithm

**Large (n > 50):**
- Time: < 5 minutes
- Gap: < 15%
- Method: Genetic Algorithm with large population

## Extension Points

### Adding New Methods

To add a new optimization method:

1. **Implement method function:**
```python
def new_method_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    # Implementation
    return {
        'solution': best_solution,
        'objective': best_objective,
        'is_valid': True,
        'iterations': iterations,
        'convergence': {...},
        'metadata': {'method': 'new_method', ...}
    }
```

2. **Add to optimize.py:**
```python
elif method == 'new_method':
    from Librex.methods.category.new_method import new_method_optimize
    result = new_method_optimize(standardized_problem, config)
```

3. **Update documentation:** Add to this taxonomy

4. **Add tests:** Unit + integration tests

## Conclusion

Librex provides a well-rounded set of 5 baseline methods covering major optimization paradigms:
- Stochastic sampling (Random Search)
- Trajectory-based metaheuristics (SA, Tabu)
- Local search (Hill Climbing)
- Population-based evolutionary (GA)

**Production Status:** All 5 baseline methods validated and production-ready

**Research Pipeline:** Novel methods under development with rigorous mathematical validation

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Maintainer:** Librex Development Team
