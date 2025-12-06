# genetic_algorithms
**Module:** `Python/Optimization/genetic_algorithms.py`
## Overview
Genetic Algorithms and Evolutionary Computation
==============================================
This module implements genetic algorithms, evolution strategies, and
other evolutionary computation methods for optimization and machine learning.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `demo()`
Demonstrate genetic algorithms and evolutionary computation.
**Source:** [Line 645](Python/Optimization/genetic_algorithms.py#L645)
## Classes
### `Individual`
Represents an individual in the population.
Attributes:
genes: Genetic representation
fitness: Fitness value
age: Age of individual
metadata: Additional information
#### Methods
##### `__post_init__(self)`
*No documentation available.*
**Source:** [Line 41](Python/Optimization/genetic_algorithms.py#L41)
**Class Source:** [Line 26](Python/Optimization/genetic_algorithms.py#L26)
### `EvolutionaryAlgorithm`
Abstract base class for evolutionary algorithms.
Provides common framework for genetic algorithms, evolution strategies,
and other population-based optimization methods.
#### Methods
##### `__init__(self, population_size, generations, mutation_rate, crossover_rate, selection_method, tournament_size, elitism, elite_size, track_diversity, verbose, random_seed)`
Initialize evolutionary algorithm.
Args:
population_size: Size of population
generations: Number of generations
mutation_rate: Probability of mutation
crossover_rate: Probability of crossover
selection_method: Selection method ('tournament', 'roulette', 'rank')
tournament_size: Size of tournament for tournament selection
elitism: Whether to preserve best individuals
elite_size: Number of elite individuals to preserve
track_diversity: Whether to track population diversity
verbose: Whether to print progress
random_seed: Random seed for reproducibility
**Source:** [Line 53](Python/Optimization/genetic_algorithms.py#L53)
##### `initialize_population(self, bounds)`
Initialize population.
**Source:** [Line 96](Python/Optimization/genetic_algorithms.py#L96)
##### `mutate(self, individual, bounds)`
Mutate individual.
**Source:** [Line 101](Python/Optimization/genetic_algorithms.py#L101)
##### `crossover(self, parent1, parent2, bounds)`
Crossover two parents.
**Source:** [Line 106](Python/Optimization/genetic_algorithms.py#L106)
##### `evaluate_fitness(self, objective, individual)`
Evaluate fitness of individual.
**Source:** [Line 111](Python/Optimization/genetic_algorithms.py#L111)
##### `select_parents(self, population)`
Select parents for reproduction.
**Source:** [Line 118](Python/Optimization/genetic_algorithms.py#L118)
##### `_tournament_selection(self, population)`
Tournament selection.
**Source:** [Line 129](Python/Optimization/genetic_algorithms.py#L129)
##### `_roulette_selection(self, population)`
Roulette wheel selection.
**Source:** [Line 140](Python/Optimization/genetic_algorithms.py#L140)
##### `_rank_selection(self, population)`
Rank-based selection.
**Source:** [Line 160](Python/Optimization/genetic_algorithms.py#L160)
##### `calculate_diversity(self, population)`
Calculate population diversity.
**Source:** [Line 176](Python/Optimization/genetic_algorithms.py#L176)
##### `get_statistics(self, population)`
Get population statistics.
**Source:** [Line 192](Python/Optimization/genetic_algorithms.py#L192)
**Class Source:** [Line 45](Python/Optimization/genetic_algorithms.py#L45)
### `GeneticAlgorithm`
Standard Genetic Algorithm implementation.
Uses real-valued encoding with Gaussian mutation and
blend crossover for continuous optimization problems.
#### Methods
##### `__init__(self, mutation_strength, crossover_alpha)`
Initialize Genetic Algorithm.
Args:
mutation_strength: Standard deviation for Gaussian mutation
crossover_alpha: Alpha parameter for blend crossover
**Source:** [Line 216](Python/Optimization/genetic_algorithms.py#L216)
##### `minimize(self, objective, bounds, x0)`
Minimize objective function using genetic algorithm.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial best guess (optional)
Returns:
OptimizationResult
**Source:** [Line 229](Python/Optimization/genetic_algorithms.py#L229)
##### `initialize_population(self, bounds)`
Initialize population with random individuals.
**Source:** [Line 326](Python/Optimization/genetic_algorithms.py#L326)
##### `mutate(self, individual, bounds)`
Apply Gaussian mutation.
**Source:** [Line 337](Python/Optimization/genetic_algorithms.py#L337)
##### `crossover(self, parent1, parent2, bounds)`
Apply blend crossover (BLX-α).
**Source:** [Line 353](Python/Optimization/genetic_algorithms.py#L353)
**Class Source:** [Line 208](Python/Optimization/genetic_algorithms.py#L208)
### `EvolutionStrategy`
Evolution Strategy (ES) algorithm.
Uses self-adaptive mutation with strategy parameters that
evolve along with the solution parameters.
#### Methods
##### `__init__(self, strategy, mu, lambda_, initial_sigma, tau, tau_prime)`
Initialize Evolution Strategy.
Args:
strategy: ES strategy ('(μ+λ)' or '(μ,λ)')
mu: Number of parents
lambda_: Number of offspring
initial_sigma: Initial mutation strength
tau: Global learning rate
tau_prime: Individual learning rate
**Source:** [Line 391](Python/Optimization/genetic_algorithms.py#L391)
##### `minimize(self, objective, bounds, x0)`
Minimize objective function using evolution strategy.
Args:
objective: Function to minimize
bounds: Variable bounds
x0: Initial best guess (optional)
Returns:
OptimizationResult
**Source:** [Line 422](Python/Optimization/genetic_algorithms.py#L422)
##### `initialize_population(self, bounds)`
Initialize population with strategy parameters.
**Source:** [Line 504](Python/Optimization/genetic_algorithms.py#L504)
##### `mutate(self, individual, bounds)`
Apply self-adaptive mutation.
**Source:** [Line 521](Python/Optimization/genetic_algorithms.py#L521)
##### `crossover(self, parent1, parent2, bounds)`
Crossover not typically used in ES.
**Source:** [Line 545](Python/Optimization/genetic_algorithms.py#L545)
**Class Source:** [Line 383](Python/Optimization/genetic_algorithms.py#L383)
### `GeneticProgramming`
Genetic Programming for evolving programs/expressions.
Evolves tree-structured programs using genetic operators
adapted for symbolic expressions.
#### Methods
##### `__init__(self, function_set, terminal_set, max_depth, init_method)`
Initialize Genetic Programming.
Args:
function_set: Available functions
terminal_set: Available terminals
max_depth: Maximum tree depth
init_method: Initialization method
**Source:** [Line 558](Python/Optimization/genetic_algorithms.py#L558)
##### `minimize(self, objective, bounds, x0)`
Evolve programs using genetic programming.
Note: This is a simplified implementation for demonstration.
Real GP would require proper tree representation and evaluation.
**Source:** [Line 584](Python/Optimization/genetic_algorithms.py#L584)
##### `initialize_population(self, bounds)`
Initialize population of programs.
**Source:** [Line 603](Python/Optimization/genetic_algorithms.py#L603)
##### `mutate(self, individual, bounds)`
Mutate program tree.
**Source:** [Line 615](Python/Optimization/genetic_algorithms.py#L615)
##### `crossover(self, parent1, parent2, bounds)`
Crossover program trees.
**Source:** [Line 627](Python/Optimization/genetic_algorithms.py#L627)
**Class Source:** [Line 550](Python/Optimization/genetic_algorithms.py#L550)