# optimization
**Module:** `Python/Monte_Carlo/optimization.py`
## Overview
Monte Carlo Optimization Algorithms.
This module implements various Monte Carlo-based optimization algorithms
including simulated annealing, genetic algorithms, and particle swarm optimization.
Classes:
SimulatedAnnealing: Simulated annealing optimizer
GeneticAlgorithm: Genetic algorithm optimizer
ParticleSwarmOptimization: Particle swarm optimizer
CrossEntropyMethod: Cross-entropy method optimizer
Functions:
simulated_annealing: Convenience function for SA
genetic_algorithm: Convenience function for GA
particle_swarm: Convenience function for PSO
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `simulated_annealing(objective, bounds, initial_guess, initial_temperature, max_iterations, random_state)`
Convenience function for simulated annealing.
**Source:** [Line 717](Python/Monte_Carlo/optimization.py#L717)
### `genetic_algorithm(objective, bounds, population_size, n_generations, random_state)`
Convenience function for genetic algorithm.
**Source:** [Line 734](Python/Monte_Carlo/optimization.py#L734)
### `particle_swarm(objective, bounds, n_particles, max_iterations, random_state)`
Convenience function for particle swarm optimization.
**Source:** [Line 750](Python/Monte_Carlo/optimization.py#L750)
### `cross_entropy_method(objective, bounds, population_size, max_iterations, random_state)`
Convenience function for cross-entropy method.
**Source:** [Line 766](Python/Monte_Carlo/optimization.py#L766)
## Classes
### `OptimizationResult`
Result of Monte Carlo optimization.
Attributes:
x_optimal: Optimal solution found
f_optimal: Optimal function value
n_evaluations: Number of function evaluations
convergence_history: History of best values
success: Whether optimization succeeded
message: Status message
metadata: Additional optimization information
**Class Source:** [Line 31](Python/Monte_Carlo/optimization.py#L31)
### `MonteCarloOptimizer`
Abstract base class for Monte Carlo optimizers.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 55](Python/Monte_Carlo/optimization.py#L55)
##### `optimize(self, objective, bounds)`
Optimize objective function.
**Source:** [Line 61](Python/Monte_Carlo/optimization.py#L61)
**Class Source:** [Line 52](Python/Monte_Carlo/optimization.py#L52)
### `SimulatedAnnealing`
Simulated Annealing optimizer.
Implements simulated annealing with various cooling schedules
and neighborhood generation strategies.
Parameters:
initial_temperature: Starting temperature
cooling_schedule: Cooling schedule ('linear', 'exponential', 'logarithmic')
max_iterations: Maximum number of iterations
min_temperature: Minimum temperature
step_size: Initial step size for neighborhood generation
#### Methods
##### `__init__(self, initial_temperature, cooling_schedule, max_iterations, min_temperature, step_size, random_state, verbose)`
*No documentation available.*
**Source:** [Line 81](Python/Monte_Carlo/optimization.py#L81)
##### `optimize(self, objective, bounds, initial_guess)`
Optimize using simulated annealing.
Args:
objective: Objective function to minimize
bounds: Variable bounds [(min, max), ...]
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
OptimizationResult: Optimization results
**Source:** [Line 96](Python/Monte_Carlo/optimization.py#L96)
##### `_generate_neighbor(self, x, bounds, temperature)`
Generate neighbor solution.
**Source:** [Line 190](Python/Monte_Carlo/optimization.py#L190)
##### `_clip_to_bounds(self, x, bounds)`
Clip solution to bounds.
**Source:** [Line 202](Python/Monte_Carlo/optimization.py#L202)
##### `_accept_candidate(self, current_f, candidate_f, temperature)`
Decide whether to accept candidate solution.
**Source:** [Line 206](Python/Monte_Carlo/optimization.py#L206)
##### `_update_temperature(self, iteration)`
Update temperature according to cooling schedule.
**Source:** [Line 217](Python/Monte_Carlo/optimization.py#L217)
**Class Source:** [Line 67](Python/Monte_Carlo/optimization.py#L67)
### `GeneticAlgorithm`
Genetic Algorithm optimizer.
Implements genetic algorithm with various selection, crossover,
and mutation strategies.
#### Methods
##### `__init__(self, population_size, n_generations, crossover_rate, mutation_rate, selection_method, crossover_method, mutation_method, elitism_rate, random_state, verbose)`
*No documentation available.*
**Source:** [Line 238](Python/Monte_Carlo/optimization.py#L238)
##### `optimize(self, objective, bounds)`
Optimize using genetic algorithm.
**Source:** [Line 259](Python/Monte_Carlo/optimization.py#L259)
##### `_initialize_population(self, bounds)`
Initialize random population.
**Source:** [Line 333](Python/Monte_Carlo/optimization.py#L333)
##### `_evaluate_population(self, objective, population)`
Evaluate fitness of entire population.
**Source:** [Line 341](Python/Monte_Carlo/optimization.py#L341)
##### `_selection(self, population, fitness)`
Select individuals for reproduction.
**Source:** [Line 346](Python/Monte_Carlo/optimization.py#L346)
##### `_tournament_selection(self, population, fitness, tournament_size)`
Tournament selection.
**Source:** [Line 355](Python/Monte_Carlo/optimization.py#L355)
##### `_roulette_selection(self, population, fitness)`
Roulette wheel selection.
**Source:** [Line 370](Python/Monte_Carlo/optimization.py#L370)
##### `_crossover(self, population, bounds)`
Apply crossover to population.
**Source:** [Line 384](Python/Monte_Carlo/optimization.py#L384)
##### `_uniform_crossover(self, parent1, parent2)`
Uniform crossover.
**Source:** [Line 413](Python/Monte_Carlo/optimization.py#L413)
##### `_single_point_crossover(self, parent1, parent2)`
Single-point crossover.
**Source:** [Line 421](Python/Monte_Carlo/optimization.py#L421)
##### `_arithmetic_crossover(self, parent1, parent2)`
Arithmetic crossover.
**Source:** [Line 429](Python/Monte_Carlo/optimization.py#L429)
##### `_mutation(self, population, bounds)`
Apply mutation to population.
**Source:** [Line 437](Python/Monte_Carlo/optimization.py#L437)
##### `_gaussian_mutation(self, individual, bounds)`
Gaussian mutation.
**Source:** [Line 452](Python/Monte_Carlo/optimization.py#L452)
##### `_uniform_mutation(self, individual, bounds)`
Uniform mutation.
**Source:** [Line 460](Python/Monte_Carlo/optimization.py#L460)
**Class Source:** [Line 231](Python/Monte_Carlo/optimization.py#L231)
### `ParticleSwarmOptimization`
Particle Swarm Optimization.
Implements PSO with inertia weight and constriction factor variants.
#### Methods
##### `__init__(self, n_particles, max_iterations, w, c1, c2, w_min, w_max, random_state, verbose)`
*No documentation available.*
**Source:** [Line 479](Python/Monte_Carlo/optimization.py#L479)
##### `optimize(self, objective, bounds)`
Optimize using particle swarm optimization.
**Source:** [Line 498](Python/Monte_Carlo/optimization.py#L498)
##### `_initialize_positions(self, bounds)`
Initialize particle positions.
**Source:** [Line 591](Python/Monte_Carlo/optimization.py#L591)
##### `_initialize_velocities(self, bounds)`
Initialize particle velocities.
**Source:** [Line 599](Python/Monte_Carlo/optimization.py#L599)
**Class Source:** [Line 473](Python/Monte_Carlo/optimization.py#L473)
### `CrossEntropyMethod`
Cross-Entropy Method optimizer.
Implements the cross-entropy method for continuous optimization.
#### Methods
##### `__init__(self, population_size, elite_fraction, max_iterations, smoothing_factor, random_state, verbose)`
*No documentation available.*
**Source:** [Line 615](Python/Monte_Carlo/optimization.py#L615)
##### `optimize(self, objective, bounds)`
Optimize using cross-entropy method.
**Source:** [Line 629](Python/Monte_Carlo/optimization.py#L629)
##### `_sample_population(self, mean, std, bounds)`
Sample population from current distribution.
**Source:** [Line 705](Python/Monte_Carlo/optimization.py#L705)
**Class Source:** [Line 609](Python/Monte_Carlo/optimization.py#L609)