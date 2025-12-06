# Librex.Evo - Evolutionary Multi-Objective Optimization

**NSGA-II for finding Pareto-optimal trade-offs in ORCHEX workflows**

## Overview

Librex.Evo solves multi-objective optimization problems using NSGA-II (Non-dominated Sorting Genetic Algorithm II), finding Pareto-optimal solutions that balance multiple conflicting objectives.

### Problem Formulation

Given:
- **M objective functions**: f₁(x), f₂(x), ..., f_M(x)
- **Variable bounds**: x ∈ [lower, upper]
- **Conflicting objectives**: Improving one worsens another

Find Pareto frontier:
```
minimize: [f₁(x), f₂(x), ..., f_M(x)]
```

**Output**: Set of non-dominated solutions representing optimal trade-offs

## Installation

```bash
cd MEZAN/Libria/libria-evo
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Multi-Objective Optimization

```python
from libria_evo import Librex.EvoSolver
from MEZAN.core import OptimizationProblem, ProblemType
import numpy as np

# Define objective functions
def speed_objective(config):
    """Minimize execution time"""
    parallelism = config[0]
    redundancy = config[1]
    return 100 / parallelism + 10 * redundancy

def quality_objective(config):
    """Minimize error rate"""
    parallelism = config[0]
    redundancy = config[1]
    return 0.5 / (redundancy + 1) + 0.1 * parallelism

def cost_objective(config):
    """Minimize computational cost"""
    parallelism = config[0]
    redundancy = config[1]
    return 50 * parallelism + 30 * redundancy

# Define problem
problem = OptimizationProblem(
    problem_type=ProblemType.EVO,
    data={
        "objective_functions": [speed_objective, quality_objective, cost_objective],
        "num_variables": 2,  # [parallelism, redundancy]
        "variable_bounds": (0.0, 10.0),  # Both variables in [0, 10]
    }
)

# Solve
solver = Librex.EvoSolver(config={
    "population_size": 100,
    "num_generations": 200,
    "mutation_rate": 0.1,
    "crossover_rate": 0.9
})

solver.initialize()
result = solver.solve(problem)

# Examine Pareto front
pareto_front = result.solution["pareto_front"]
pareto_solutions = result.solution["pareto_solutions"]

print(f"Found {len(pareto_front)} Pareto-optimal solutions\n")

# Show best solution for each objective
for i, obj_name in enumerate(["Speed", "Quality", "Cost"]):
    best_idx = np.argmin([sol[i] for sol in pareto_front])
    best_solution = pareto_solutions[best_idx]
    best_objectives = pareto_front[best_idx]

    print(f"{obj_name}-optimal configuration:")
    print(f"  Config: parallelism={best_solution[0]:.2f}, redundancy={best_solution[1]:.2f}")
    print(f"  Speed={best_objectives[0]:.2f}, Quality={best_objectives[1]:.2f}, Cost={best_objectives[2]:.2f}\n")
```

### ORCHEX Integration

```python
from MEZAN.orchex.atlas_core.src.orchex.optimization_integration import ATLASOptimizationManager

manager = ATLASOptimizationManager()

# Define research workflow objectives
objectives = {
    "speed": lambda cfg: estimate_completion_time(cfg),
    "quality": lambda cfg: estimate_output_quality(cfg),
    "cost": lambda cfg: estimate_credit_usage(cfg)
}

# Configuration space
config_space = {
    "agent_parallelism": (1, 8),      # 1-8 parallel agents
    "redundancy_factor": (1.0, 3.0),  # 1-3x redundancy
    "quality_threshold": (0.7, 0.95)  # Quality acceptance threshold
}

result = manager.optimize_multi_objective_workflow(objectives, config_space)

print(f"Pareto frontier contains {len(result['pareto_solutions'])} configurations\n")

# Interactive selection
print("Select preferred trade-off:")
for i, (solution, objectives) in enumerate(zip(result['pareto_solutions'][:5], result['pareto_front'][:5])):
    print(f"{i+1}. Parallelism={solution[0]:.1f}, Redundancy={solution[1]:.2f}x, Threshold={solution[2]:.2f}")
    print(f"   → Speed={objectives[0]:.1f}min, Quality={objectives[1]:.1%}, Cost=${objectives[2]:.2f}\n")
```

## Algorithm

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

NSGA-II is the state-of-the-art multi-objective evolutionary algorithm:

**Key Features:**
1. **Fast non-dominated sorting**: O(MN²) complexity
2. **Crowding distance**: Maintains diversity
3. **Elitism**: Best solutions always survive
4. **Tournament selection**: Based on rank and crowding

### Algorithm Steps

```
Initialize: Random population P of size N

For each generation:
    1. Non-dominated Sorting:
       - Rank solutions into Pareto fronts F₁, F₂, ...
       - F₁ = non-dominated solutions
       - F₂ = non-dominated after removing F₁
       - etc.

    2. Crowding Distance:
       - For each front, compute crowding distance
       - Preference: boundary solutions > crowded solutions

    3. Selection:
       - Tournament selection using:
         * Rank (lower is better)
         * Crowding distance (higher is better)

    4. Crossover:
       - Simulated Binary Crossover (SBX)
       - Creates offspring near parents

    5. Mutation:
       - Polynomial mutation
       - Small perturbations

    6. Combination:
       - Combine parents and offspring
       - Select best N using rank + crowding
```

**Time Complexity**: O(M N² G) for M objectives, N population, G generations

## Features

### Adaptive Parameters

```python
solver = Librex.EvoSolver(config={
    "adaptive_mutation": True,  # Decrease mutation over time
    "adaptive_crossover": True  # Adjust crossover based on diversity
})
```

### Constraint Handling

```python
# Constrained multi-objective optimization
problem = OptimizationProblem(
    problem_type=ProblemType.EVO,
    data={
        "objective_functions": objectives,
        "constraint_functions": [
            lambda x: x[0] + x[1] - 10,  # x₀ + x₁ ≤ 10
            lambda x: x[0] - 2*x[1]      # x₀ ≥ 2x₁
        ],
        "num_variables": 2,
        "variable_bounds": (0.0, 10.0)
    }
)
```

### Reference Point Optimization

```python
# Optimize towards specific reference point
solver = Librex.EvoSolver(config={
    "reference_point": [5.0, 0.1, 100.0],  # Desired [speed, quality, cost]
    "use_reference_point": True
})
```

## Testing

```bash
# Run all tests
pytest tests/test_solver.py

# Test NSGA-II components
pytest tests/test_solver.py::test_nsga2_sorting -v

# Test Pareto optimality
pytest tests/test_solver.py::test_pareto_front -v

# Test crowding distance
pytest tests/test_solver.py::test_crowding_distance -v
```

## API Reference

### Librex.EvoSolver

```python
class Librex.EvoSolver(OptimizerInterface):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None
    )
```

**Configuration:**
- `population_size`: Population size (default: 100)
- `num_generations`: Number of generations (default: 100)
- `mutation_rate`: Mutation probability (default: 0.1)
- `crossover_rate`: Crossover probability (default: 0.9)
- `adaptive_mutation`: Use adaptive mutation (default: False)
- `reference_point`: Target point for optimization (default: None)

**Methods:**
- `solve(problem: OptimizationProblem) -> OptimizationResult`
- `get_problem_types() -> List[ProblemType]`
- `estimate_complexity(problem: OptimizationProblem) -> str`

## Use Cases

### 1. Workflow Configuration
Balance speed, quality, and cost in research workflows

### 2. Resource Allocation
Multi-objective resource distribution

### 3. Hyperparameter Tuning
Find trade-offs in ML model performance

### 4. Portfolio Optimization
Balance risk, return, and liquidity

## Performance

| Objectives | Variables | Population | Generations | Time |
|------------|-----------|------------|-------------|------|
| 2 | 5 | 50 | 100 | 2s |
| 3 | 10 | 100 | 200 | 15s |
| 5 | 20 | 200 | 300 | 120s |

**Pareto Front Quality**: 85-95% hypervolume vs theoretical optimum (for known benchmarks).

## Mathematical Background

### Pareto Dominance

Solution x dominates y (x ≺ y) if:
- fᵢ(x) ≤ fᵢ(y) for all objectives i
- fⱼ(x) < fⱼ(y) for at least one objective j

### Pareto Front

Set of non-dominated solutions:
```
P* = {x | ∄y such that y ≺ x}
```

### Hypervolume Indicator

Quality metric for Pareto front:
```
HV = volume of space dominated by front
```

Higher hypervolume = better front.

### Convergence and Diversity

Good Pareto approximation requires:
1. **Convergence**: Close to true Pareto front
2. **Diversity**: Uniform spread across objectives

NSGA-II balances both via ranking + crowding distance.

## Production Deployment

```python
from MEZAN.core import OptimizerFactory

factory = OptimizerFactory(config={
    "feature_flags": {
        "enable_evo_libria": True
    }
})

optimizer = factory.create_optimizer(problem, timeout=60.0)
result = optimizer.solve(problem)
```

## Monitoring

Prometheus metrics:
- `libria_evo_generation`: Current generation
- `libria_evo_pareto_size`: Size of Pareto front
- `libria_evo_hypervolume`: Hypervolume indicator
- `libria_evo_diversity`: Crowding distance metric

## Roadmap

- [ ] NSGA-III for many objectives (M > 5)
- [ ] MOEA/D (decomposition-based)
- [ ] SMS-EMOA (S-metric selection)
- [ ] Parallel evaluation (distributed fitness)
- [ ] Interactive optimization (user preferences)

## References

1. Deb et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
2. Zitzler & Thiele (1999). "Multiobjective evolutionary algorithms: a comparative case study"
3. Coello Coello (2006). "Evolutionary Multi-Objective Optimization: A Historical View"
4. Miettinen, K. (1999). "Nonlinear Multiobjective Optimization"

## License

Apache 2.0

## Authors

Meshal Alawein (meshal@berkeley.edu)
MEZAN Development Team
