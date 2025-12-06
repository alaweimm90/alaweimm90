# Advanced Optimization Methods

## Overview

The Librex framework includes 5 state-of-the-art metaheuristic algorithms for solving complex combinatorial optimization problems. These advanced methods complement the baseline algorithms by providing sophisticated search strategies suitable for large-scale and challenging optimization problems.

## Available Methods

### 1. Ant Colony Optimization (ACO)

**Method name:** `ant_colony`

ACO is a probabilistic technique inspired by the foraging behavior of ants. It uses pheromone trails to guide the search towards promising regions of the solution space.

**Key Features:**
- Pheromone-based search guidance
- Balance between exploration and exploitation
- Particularly effective for graph-based problems (TSP, QAP)
- Parallel construction of solutions

**Configuration Parameters:**
```python
config = {
    'n_ants': 20,              # Number of ants
    'n_iterations': 100,       # Number of iterations
    'alpha': 1.0,              # Pheromone importance
    'beta': 2.0,               # Heuristic importance
    'rho': 0.1,                # Evaporation rate
    'q0': 0.9,                 # Exploration probability
    'tau_min': 0.01,           # Min pheromone level
    'tau_max': 10.0,           # Max pheromone level
    'seed': None               # Random seed
}
```

**References:**
- Dorigo, M., & Stützle, T. (2004). Ant colony optimization. MIT Press.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 26(1), 29-41.

### 2. Particle Swarm Optimization (PSO)

**Method name:** `particle_swarm`

PSO is inspired by the social behavior of bird flocking or fish schooling. Particles move through the search space influenced by their own best position and the global best position.

**Key Features:**
- Swarm intelligence approach
- Adapted for discrete permutation problems
- Balance between individual and collective knowledge
- Simple implementation with few parameters

**Configuration Parameters:**
```python
config = {
    'n_particles': 30,         # Number of particles
    'n_iterations': 100,       # Number of iterations
    'w': 0.7,                  # Inertia weight
    'c1': 2.0,                 # Cognitive coefficient
    'c2': 2.0,                 # Social coefficient
    'v_max': 4.0,              # Maximum velocity
    'seed': None               # Random seed
}
```

**References:**
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of IEEE International Conference on Neural Networks, 4, 1942-1948.
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE International Conference on Evolutionary Computation, 69-73.

### 3. Variable Neighborhood Search (VNS)

**Method name:** `variable_neighborhood`

VNS systematically changes neighborhoods during the search to escape local optima. It combines shaking (perturbation) with local search in different neighborhood structures.

**Key Features:**
- Multiple neighborhood structures
- Systematic neighborhood change
- Combines diversification and intensification
- Flexible and parameter-light

**Configuration Parameters:**
```python
config = {
    'max_iterations': 1000,           # Maximum iterations
    'k_max': 5,                      # Maximum neighborhood size
    'local_search_iterations': 100,  # Local search iterations
    'seed': None                     # Random seed
}
```

**References:**
- Hansen, P., & Mladenović, N. (2001). Variable neighborhood search: Principles and applications. European Journal of Operational Research, 130(3), 449-467.
- Hansen, P., Mladenović, N., & Moreno Pérez, J. A. (2010). Variable neighbourhood search: methods and applications. Annals of Operations Research, 175(1), 367-407.

### 4. Iterated Local Search (ILS)

**Method name:** `iterated_local_search`

ILS applies local search iteratively with perturbations to escape local optima. It uses acceptance criteria to decide whether to accept new solutions.

**Key Features:**
- Simple yet effective metaheuristic
- Adaptive perturbation strength
- Multiple acceptance criteria options
- Efficient for many combinatorial problems

**Configuration Parameters:**
```python
config = {
    'max_iterations': 100,            # Maximum iterations
    'local_search_iterations': 100,   # Local search iterations
    'perturbation_strength': 3,       # Perturbation strength (1-5)
    'acceptance': 'better',           # 'better', 'threshold', or 'simulated_annealing'
    'threshold': 0.01,                # For threshold acceptance
    'temperature': 100.0,             # For SA acceptance
    'cooling_rate': 0.95,             # For SA acceptance
    'seed': None                      # Random seed
}
```

**References:**
- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated local search. In Handbook of metaheuristics (pp. 320-353). Springer.
- Stützle, T. (2006). Iterated local search for the quadratic assignment problem. European Journal of Operational Research, 174(3), 1519-1539.

### 5. GRASP (Greedy Randomized Adaptive Search Procedure)

**Method name:** `grasp`

GRASP combines greedy construction with randomization and local search. It builds solutions using a restricted candidate list and improves them through local search.

**Key Features:**
- Two-phase approach: construction + local search
- Greedy randomized construction
- Optional path relinking for intensification
- Effective for various combinatorial problems

**Configuration Parameters:**
```python
config = {
    'max_iterations': 100,            # Maximum GRASP iterations
    'alpha': 0.2,                     # Greediness factor [0,1]
    'local_search_iterations': 100,   # Local search iterations
    'rcl_size': None,                 # Restricted Candidate List size
    'path_relinking': False,          # Enable path relinking
    'elite_size': 10,                 # Elite set size for path relinking
    'seed': None                      # Random seed
}
```

**References:**
- Feo, T. A., & Resende, M. G. (1995). Greedy randomized adaptive search procedures. Journal of Global Optimization, 6(2), 109-133.
- Resende, M. G., & Ribeiro, C. C. (2016). Optimization by GRASP. Springer New York.

## Usage Examples

### Basic Usage

```python
from Librex import optimize
from Librex.adapters.qap import QAPAdapter

# Define problem
problem = {
    'flow_matrix': flow_matrix,
    'distance_matrix': distance_matrix
}

adapter = QAPAdapter()

# Using Ant Colony Optimization
result = optimize(
    problem,
    adapter,
    method='ant_colony',
    config={'n_ants': 30, 'n_iterations': 200}
)

print(f"Best solution: {result['solution']}")
print(f"Objective value: {result['objective']}")
```

### Comparing Multiple Methods

```python
methods = [
    ('ant_colony', {'n_ants': 20, 'n_iterations': 100}),
    ('particle_swarm', {'n_particles': 30, 'n_iterations': 100}),
    ('variable_neighborhood', {'max_iterations': 500, 'k_max': 5}),
    ('iterated_local_search', {'max_iterations': 100}),
    ('grasp', {'max_iterations': 100, 'alpha': 0.3})
]

results = {}
for method_name, config in methods:
    result = optimize(problem, adapter, method=method_name, config=config)
    results[method_name] = {
        'objective': result['objective'],
        'iterations': result['iterations'],
        'converged': result['convergence']['converged']
    }

# Compare results
for method, res in results.items():
    print(f"{method}: obj={res['objective']:.2f}, iters={res['iterations']}")
```

## Performance Characteristics

### Time Complexity

| Method | Time Complexity per Iteration | Space Complexity |
|--------|-------------------------------|------------------|
| ACO | O(m * n²) | O(n²) |
| PSO | O(m * n²) | O(m * n) |
| VNS | O(k * n²) | O(n) |
| ILS | O(n²) | O(n) |
| GRASP | O(n² + n³) | O(n) |

Where:
- n = problem size
- m = population size (ants/particles)
- k = neighborhood size

### Algorithm Selection Guide

Choose your algorithm based on:

1. **Problem characteristics:**
   - Graph-based problems → ACO
   - Continuous optimization adapted to discrete → PSO
   - Strong local optima → VNS or ILS
   - Need for diverse solutions → GRASP

2. **Available computational resources:**
   - Limited time → ILS (simple and fast)
   - Parallel processing available → ACO or PSO
   - Limited memory → VNS or ILS

3. **Solution quality requirements:**
   - High quality needed → VNS with large k_max
   - Good solutions quickly → GRASP with low alpha
   - Consistent performance → ILS with SA acceptance

## Integration with Librex

All advanced methods are fully integrated with:
- StandardizedProblem interface
- Universal optimization API
- Domain adapters (QAP, TSP, etc.)
- Convergence tracking
- Performance benchmarking

## Benchmarking Results

Typical performance on QAP instances (relative to best known solutions):

| Method | Small (n<20) | Medium (n=20-50) | Large (n>50) |
|--------|-------------|------------------|--------------|
| ACO | 2-5% gap | 5-10% gap | 10-15% gap |
| PSO | 3-6% gap | 6-12% gap | 12-18% gap |
| VNS | 1-3% gap | 3-7% gap | 7-12% gap |
| ILS | 2-4% gap | 4-8% gap | 8-14% gap |
| GRASP | 2-5% gap | 5-10% gap | 10-16% gap |

*Note: Results vary significantly based on parameter tuning and problem structure.*

## Best Practices

1. **Parameter Tuning:**
   - Start with default parameters
   - Tune based on preliminary results
   - Use different random seeds for robustness

2. **Hybrid Approaches:**
   - Combine methods (e.g., GRASP construction + VNS improvement)
   - Use ILS as post-processing for other methods

3. **Convergence Monitoring:**
   - Track objective value history
   - Implement early stopping criteria
   - Monitor diversity metrics

4. **Scalability:**
   - Adjust population sizes with problem size
   - Use adaptive parameters when possible
   - Consider time limits for large instances