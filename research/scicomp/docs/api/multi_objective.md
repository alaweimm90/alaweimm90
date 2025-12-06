# multi_objective
**Module:** `Python/Optimization/multi_objective.py`
## Overview
Multi-Objective Optimization Algorithms
======================================
This module implements algorithms for multi-objective optimization including
Pareto optimization, weighted sum methods, epsilon-constraint methods, and
evolutionary multi-objective algorithms like NSGA-II.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `dominates(obj1, obj2)`
Check if obj1 dominates obj2 (for minimization).
obj1 dominates obj2 if:
- obj1 is at least as good as obj2 in all objectives
- obj1 is strictly better than obj2 in at least one objective
**Source:** [Line 115](Python/Optimization/multi_objective.py#L115)
### `non_dominated_sort(objectives)`
Perform non-dominated sorting of population.
Args:
objectives: Array of objective values (n_individuals x n_objectives)
Returns:
List of fronts, where each front is a list of indices
**Source:** [Line 127](Python/Optimization/multi_objective.py#L127)
### `crowding_distance(objectives, indices)`
Calculate crowding distance for solutions in a front.
Args:
objectives: All objective values
indices: Indices of solutions in the front
Returns:
Crowding distances for each solution in the front
**Source:** [Line 173](Python/Optimization/multi_objective.py#L173)
### `demo()`
Demonstrate multi-objective optimization algorithms.
**Source:** [Line 743](Python/Optimization/multi_objective.py#L743)
## Classes
### `MultiObjectiveResult`
Results from multi-objective optimization.
Attributes:
pareto_set: Set of Pareto optimal solutions
pareto_front: Corresponding objective values
hypervolume: Hypervolume indicator (if computed)
spacing: Spacing metric for diversity
success: Whether optimization was successful
message: Description of termination
nit: Number of iterations/generations
nfev: Number of function evaluations
execution_time: Time taken
metadata: Additional algorithm-specific data
#### Methods
##### `__post_init__(self)`
*No documentation available.*
**Source:** [Line 54](Python/Optimization/multi_objective.py#L54)
**Class Source:** [Line 27](Python/Optimization/multi_objective.py#L27)
### `MultiObjectiveProblem`
Represents a multi-objective optimization problem.
Encapsulates multiple objective functions and constraints
for multi-objective optimization algorithms.
#### Methods
##### `__init__(self, objectives, n_variables, bounds, constraints, objective_names)`
Initialize multi-objective problem.
Args:
objectives: List of objective functions
n_variables: Number of decision variables
bounds: Variable bounds
constraints: List of constraints (optional)
objective_names: Names of objectives (optional)
**Source:** [Line 66](Python/Optimization/multi_objective.py#L66)
##### `evaluate(self, x)`
Evaluate all objectives at point x.
**Source:** [Line 89](Python/Optimization/multi_objective.py#L89)
##### `evaluate_population(self, population)`
Evaluate objectives for entire population.
**Source:** [Line 94](Python/Optimization/multi_objective.py#L94)
##### `is_feasible(self, x)`
Check if solution is feasible.
**Source:** [Line 98](Python/Optimization/multi_objective.py#L98)
**Class Source:** [Line 58](Python/Optimization/multi_objective.py#L58)
### `NSGA2`
Non-dominated Sorting Genetic Algorithm II (NSGA-II).
One of the most popular evolutionary algorithms for multi-objective
optimization, using non-dominated sorting and crowding distance.
#### Methods
##### `__init__(self, population_size, generations, crossover_rate, mutation_rate, crossover_eta, mutation_eta)`
Initialize NSGA-II.
Args:
population_size: Size of population
generations: Number of generations
crossover_rate: Crossover probability
mutation_rate: Mutation probability (default: 1/n_variables)
crossover_eta: Distribution index for crossover
mutation_eta: Distribution index for mutation
**Source:** [Line 224](Python/Optimization/multi_objective.py#L224)
##### `solve(self, problem)`
Solve multi-objective problem using NSGA-II.
Args:
problem: Multi-objective optimization problem
Returns:
MultiObjectiveResult
**Source:** [Line 245](Python/Optimization/multi_objective.py#L245)
##### `_initialize_population(self, problem)`
Initialize random population within bounds.
**Source:** [Line 343](Python/Optimization/multi_objective.py#L343)
##### `_create_offspring(self, population, problem)`
Create offspring population through crossover and mutation.
**Source:** [Line 354](Python/Optimization/multi_objective.py#L354)
##### `_tournament_selection(self, population, tournament_size)`
Binary tournament selection.
**Source:** [Line 380](Python/Optimization/multi_objective.py#L380)
##### `_sbx_crossover(self, parent1, parent2, bounds)`
Simulated Binary Crossover (SBX).
**Source:** [Line 385](Python/Optimization/multi_objective.py#L385)
##### `_polynomial_mutation(self, individual, bounds)`
Polynomial mutation.
**Source:** [Line 413](Python/Optimization/multi_objective.py#L413)
##### `_calculate_hypervolume(self, pareto_front)`
Calculate hypervolume indicator (simplified 2D version).
**Source:** [Line 436](Python/Optimization/multi_objective.py#L436)
##### `_calculate_spacing(self, pareto_front)`
Calculate spacing metric for diversity.
**Source:** [Line 466](Python/Optimization/multi_objective.py#L466)
**Class Source:** [Line 216](Python/Optimization/multi_objective.py#L216)
### `ParetoOptimization`
Basic Pareto optimization using various scalarization methods.
Provides methods to find Pareto optimal solutions through
different scalarization techniques.
#### Methods
##### `__init__(self, base_optimizer)`
Initialize Pareto optimization.
Args:
base_optimizer: Single-objective optimizer to use
**Source:** [Line 501](Python/Optimization/multi_objective.py#L501)
##### `solve_weighted_sum(self, problem, weights, n_solutions)`
Solve using weighted sum method.
Args:
problem: Multi-objective problem
weights: List of weight vectors (if None, generates uniform weights)
n_solutions: Number of solutions to generate (if weights not provided)
Returns:
MultiObjectiveResult
**Source:** [Line 514](Python/Optimization/multi_objective.py#L514)
##### `_generate_uniform_weights(self, n_objectives, n_solutions)`
Generate uniformly distributed weight vectors.
**Source:** [Line 570](Python/Optimization/multi_objective.py#L570)
##### `_filter_dominated(self, solutions, objectives)`
Remove dominated solutions.
**Source:** [Line 589](Python/Optimization/multi_objective.py#L589)
**Class Source:** [Line 493](Python/Optimization/multi_objective.py#L493)
### `WeightedSum`
Weighted sum method for multi-objective optimization.
**Class Source:** [Line 604](Python/Optimization/multi_objective.py#L604)
### `EpsilonConstraint`
Epsilon-constraint method for multi-objective optimization.
Optimizes one objective while constraining others to be
below specified epsilon values.
#### Methods
##### `__init__(self, base_optimizer)`
Initialize epsilon-constraint method.
Args:
base_optimizer: Constrained optimizer to use
**Source:** [Line 616](Python/Optimization/multi_objective.py#L616)
##### `solve(self, problem, primary_objective, epsilon_values, n_solutions)`
Solve using epsilon-constraint method.
Args:
problem: Multi-objective problem
primary_objective: Index of objective to optimize
epsilon_values: List of epsilon constraint values
n_solutions: Number of solutions to generate
Returns:
MultiObjectiveResult
**Source:** [Line 629](Python/Optimization/multi_objective.py#L629)
##### `_generate_epsilon_values(self, problem, primary_objective, n_solutions)`
Generate epsilon constraint values.
**Source:** [Line 706](Python/Optimization/multi_objective.py#L706)
##### `_filter_dominated(self, solutions, objectives)`
Remove dominated solutions.
**Source:** [Line 728](Python/Optimization/multi_objective.py#L728)
**Class Source:** [Line 608](Python/Optimization/multi_objective.py#L608)