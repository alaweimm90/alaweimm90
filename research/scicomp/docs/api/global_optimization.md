# global_optimization
**Module:** `Python/Optimization/global_optimization.py`
## Overview
Global Optimization Algorithms
=============================
This module implements various global optimization algorithms including
metaheuristic methods, stochastic optimization, and population-based
approaches for finding global optima in complex landscapes.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `demo()`
Demonstrate global optimization algorithms.
**Source:** [Line 597](Python/Optimization/global_optimization.py#L597)
## Classes
### `GlobalOptimizer`
Abstract base class for global optimization algorithms.
This class provides a common interface for all global
optimization methods in the Berkeley SciComp framework.
#### Methods
##### `__init__(self, max_iterations, tolerance, track_path, verbose, random_seed)`
Initialize global optimizer.
Args:
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
track_path: Whether to track optimization path
verbose: Whether to print progress information
random_seed: Random seed for reproducibility
**Source:** [Line 34](Python/Optimization/global_optimization.py#L34)
##### `minimize(self, objective, bounds, x0)`
Minimize the objective function globally.
Args:
objective: Objective function to minimize
bounds: List of (min, max) bounds for each variable
x0: Initial guess (optional)
**kwargs: Additional algorithm-specific parameters
Returns:
OptimizationResult object
**Source:** [Line 60](Python/Optimization/global_optimization.py#L60)
##### `_evaluate_function(self, func, x)`
Evaluate function with counter increment.
**Source:** [Line 76](Python/Optimization/global_optimization.py#L76)
##### `_random_point(self, bounds)`
Generate random point within bounds.
**Source:** [Line 81](Python/Optimization/global_optimization.py#L81)
##### `_clip_to_bounds(self, x, bounds)`
Clip point to bounds.
**Source:** [Line 85](Python/Optimization/global_optimization.py#L85)
**Class Source:** [Line 26](Python/Optimization/global_optimization.py#L26)
### `SimulatedAnnealing`
Simulated Annealing global optimization algorithm.
Mimics the annealing process in metallurgy to find global optima
by accepting worse solutions with decreasing probability.
#### Methods
##### `__init__(self, initial_temp, cooling_rate, min_temp, step_size, schedule)`
Initialize Simulated Annealing optimizer.
Args:
initial_temp: Initial temperature
cooling_rate: Temperature reduction factor
min_temp: Minimum temperature (stopping criterion)
step_size: Step size for random moves
schedule: Cooling schedule ('exponential', 'linear', 'logarithmic')
**Source:** [Line 100](Python/Optimization/global_optimization.py#L100)
##### `minimize(self, objective, bounds, x0)`
Minimize function using Simulated Annealing.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial point (optional)
Returns:
OptimizationResult
**Source:** [Line 120](Python/Optimization/global_optimization.py#L120)
##### `_generate_neighbor(self, x, bounds)`
Generate neighbor solution.
**Source:** [Line 193](Python/Optimization/global_optimization.py#L193)
##### `_update_temperature(self, temp, iteration)`
Update temperature according to cooling schedule.
**Source:** [Line 198](Python/Optimization/global_optimization.py#L198)
**Class Source:** [Line 92](Python/Optimization/global_optimization.py#L92)
### `ParticleSwarmOptimization`
Particle Swarm Optimization (PSO) algorithm.
Simulates social behavior of bird flocking or fish schooling
to find global optima through collective intelligence.
#### Methods
##### `__init__(self, swarm_size, inertia, cognitive_param, social_param, inertia_decay, velocity_clamp)`
Initialize PSO optimizer.
Args:
swarm_size: Number of particles in swarm
inertia: Inertia weight for velocity update
cognitive_param: Cognitive acceleration parameter
social_param: Social acceleration parameter
inertia_decay: Inertia decay rate
velocity_clamp: Maximum velocity magnitude
**Source:** [Line 217](Python/Optimization/global_optimization.py#L217)
##### `minimize(self, objective, bounds, x0)`
Minimize function using PSO.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial best guess (optional)
Returns:
OptimizationResult
**Source:** [Line 240](Python/Optimization/global_optimization.py#L240)
**Class Source:** [Line 209](Python/Optimization/global_optimization.py#L209)
### `DifferentialEvolution`
Differential Evolution (DE) algorithm.
Evolutionary algorithm that uses vector differences for mutation
and is particularly effective for continuous optimization problems.
#### Methods
##### `__init__(self, population_size, mutation_factor, crossover_prob, strategy)`
Initialize DE optimizer.
Args:
population_size: Population size (default: 15 * dim)
mutation_factor: Differential weight F
crossover_prob: Crossover probability
strategy: Mutation strategy
**Source:** [Line 344](Python/Optimization/global_optimization.py#L344)
##### `minimize(self, objective, bounds, x0)`
Minimize function using Differential Evolution.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial best guess (optional)
Returns:
OptimizationResult
**Source:** [Line 362](Python/Optimization/global_optimization.py#L362)
##### `_mutate(self, population, target_idx, bounds)`
Perform mutation operation.
**Source:** [Line 446](Python/Optimization/global_optimization.py#L446)
##### `_crossover(self, target, mutant)`
Perform crossover operation.
**Source:** [Line 469](Python/Optimization/global_optimization.py#L469)
**Class Source:** [Line 336](Python/Optimization/global_optimization.py#L336)
### `BasinHopping`
Basin Hopping global optimization algorithm.
Combines local optimization with random perturbations to escape
local minima and find global optima.
#### Methods
##### `__init__(self, step_size, local_optimizer, accept_test)`
Initialize Basin Hopping optimizer.
Args:
step_size: Step size for random displacement
local_optimizer: Local optimization method
accept_test: Custom acceptance test function
**Source:** [Line 489](Python/Optimization/global_optimization.py#L489)
##### `minimize(self, objective, bounds, x0)`
Minimize function using Basin Hopping.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial point (optional)
Returns:
OptimizationResult
**Source:** [Line 504](Python/Optimization/global_optimization.py#L504)
##### `_local_minimize(self, objective, x0, bounds)`
Perform local minimization.
**Source:** [Line 571](Python/Optimization/global_optimization.py#L571)
##### `_perturb(self, x, bounds)`
Apply random perturbation.
**Source:** [Line 583](Python/Optimization/global_optimization.py#L583)
##### `_accept_step(self, f_current, f_trial)`
Determine whether to accept step.
**Source:** [Line 589](Python/Optimization/global_optimization.py#L589)
**Class Source:** [Line 481](Python/Optimization/global_optimization.py#L481)