"""
Genetic Algorithms and Evolutionary Computation
==============================================
This module implements genetic algorithms, evolution strategies, and
other evolutionary computation methods for optimization and machine learning.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from .unconstrained import OptimizationResult
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class Individual:
    """
    Represents an individual in the population.
    Attributes:
        genes: Genetic representation
        fitness: Fitness value
        age: Age of individual
        metadata: Additional information
    """
    genes: np.ndarray
    fitness: float = float('inf')
    age: int = 0
    metadata: Dict[str, Any] = None
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
class EvolutionaryAlgorithm(ABC):
    """
    Abstract base class for evolutionary algorithms.
    Provides common framework for genetic algorithms, evolution strategies,
    and other population-based optimization methods.
    """
    def __init__(self, population_size: int = 100, generations: int = 500,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 selection_method: str = 'tournament', tournament_size: int = 3,
                 elitism: bool = True, elite_size: int = 1,
                 track_diversity: bool = False, verbose: bool = False,
                 random_seed: Optional[int] = None):
        """
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
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.elite_size = elite_size
        self.track_diversity = track_diversity
        self.verbose = verbose
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        # Statistics
        self.fitness_history = []
        self.diversity_history = []
        self.nfev = 0
    @abstractmethod
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> List[Individual]:
        """Initialize population."""
        pass
    @abstractmethod
    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        """Mutate individual."""
        pass
    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual,
                 bounds: List[Tuple[float, float]]) -> Tuple[Individual, Individual]:
        """Crossover two parents."""
        pass
    def evaluate_fitness(self, objective: Callable, individual: Individual) -> float:
        """Evaluate fitness of individual."""
        self.nfev += 1
        fitness = objective(individual.genes)
        individual.fitness = fitness
        return fitness
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """Select parents for reproduction."""
        if self.selection_method == 'tournament':
            return self._tournament_selection(population)
        elif self.selection_method == 'roulette':
            return self._roulette_selection(population)
        elif self.selection_method == 'rank':
            return self._rank_selection(population)
        else:
            return self._tournament_selection(population)
    def _tournament_selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            winner = min(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected
    def _roulette_selection(self, population: List[Individual]) -> List[Individual]:
        """Roulette wheel selection."""
        # Convert fitness to selection probabilities (assuming minimization)
        fitness_values = np.array([ind.fitness for ind in population])
        # Handle negative fitness values
        if np.min(fitness_values) < 0:
            fitness_values = fitness_values - np.min(fitness_values) + 1e-10
        # Invert for minimization (lower fitness = higher probability)
        inv_fitness = 1.0 / (fitness_values + 1e-10)
        probabilities = inv_fitness / np.sum(inv_fitness)
        selected = []
        for _ in range(self.population_size):
            idx = np.random.choice(len(population), p=probabilities)
            selected.append(population[idx])
        return selected
    def _rank_selection(self, population: List[Individual]) -> List[Individual]:
        """Rank-based selection."""
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        # Assign selection probabilities based on rank
        ranks = np.arange(1, len(population) + 1)
        probabilities = ranks / np.sum(ranks)
        selected = []
        for _ in range(self.population_size):
            idx = np.random.choice(len(sorted_pop), p=probabilities)
            selected.append(sorted_pop[idx])
        return selected
    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        genes_matrix = np.array([ind.genes for ind in population])
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(genes_matrix[i] - genes_matrix[j])
                distances.append(distance)
        return np.mean(distances) if distances else 0.0
    def get_statistics(self, population: List[Individual]) -> Dict[str, float]:
        """Get population statistics."""
        fitness_values = [ind.fitness for ind in population]
        stats = {
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values)
        }
        if self.track_diversity:
            stats['diversity'] = self.calculate_diversity(population)
        return stats
class GeneticAlgorithm(EvolutionaryAlgorithm):
    """
    Standard Genetic Algorithm implementation.
    Uses real-valued encoding with Gaussian mutation and
    blend crossover for continuous optimization problems.
    """
    def __init__(self, mutation_strength: float = 0.1, crossover_alpha: float = 0.5,
                 **kwargs):
        """
        Initialize Genetic Algorithm.
        Args:
            mutation_strength: Standard deviation for Gaussian mutation
            crossover_alpha: Alpha parameter for blend crossover
        """
        super().__init__(**kwargs)
        self.mutation_strength = mutation_strength
        self.crossover_alpha = crossover_alpha
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize objective function using genetic algorithm.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial best guess (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize population
        population = self.initialize_population(bounds)
        # Include initial guess if provided
        if x0 is not None:
            population[0].genes = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        # Evaluate initial population
        for individual in population:
            self.evaluate_fitness(objective, individual)
        self.nfev = len(population)
        self.fitness_history = []
        self.diversity_history = []
        # Evolution loop
        for generation in range(self.generations):
            # Statistics
            stats = self.get_statistics(population)
            self.fitness_history.append(stats['best_fitness'])
            if self.track_diversity:
                self.diversity_history.append(stats['diversity'])
            if self.verbose and generation % 50 == 0:
                print(f"Gen {generation}: Best = {stats['best_fitness']:.6e}, "
                      f"Mean = {stats['mean_fitness']:.6e}")
            # Selection
            parents = self.select_parents(population)
            # Create new population
            new_population = []
            # Elitism
            if self.elitism:
                elite = sorted(population, key=lambda x: x.fitness)[:self.elite_size]
                new_population.extend(elite)
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2, bounds)
                else:
                    child1 = Individual(parent1.genes.copy())
                    child2 = Individual(parent2.genes.copy())
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1, bounds)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2, bounds)
                # Evaluate offspring
                self.evaluate_fitness(objective, child1)
                self.evaluate_fitness(objective, child2)
                new_population.extend([child1, child2])
            # Trim to population size
            population = new_population[:self.population_size]
        # Find best solution
        best_individual = min(population, key=lambda x: x.fitness)
        execution_time = time.time() - start_time
        success = abs(best_individual.fitness) < 1e-6  # Simple convergence check
        return OptimizationResult(
            x=best_individual.genes,
            fun=best_individual.fitness,
            nit=self.generations,
            nfev=self.nfev,
            success=success,
            message="Evolution completed",
            execution_time=execution_time
        )
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> List[Individual]:
        """Initialize population with random individuals."""
        population = []
        for _ in range(self.population_size):
            genes = np.array([np.random.uniform(low, high) for low, high in bounds])
            individual = Individual(genes)
            population.append(individual)
        return population
    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        """Apply Gaussian mutation."""
        mutated_genes = individual.genes.copy()
        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, self.mutation_strength)
                mutated_genes[i] += mutation
                # Clip to bounds
                low, high = bounds[i]
                mutated_genes[i] = np.clip(mutated_genes[i], low, high)
        return Individual(mutated_genes)
    def crossover(self, parent1: Individual, parent2: Individual,
                 bounds: List[Tuple[float, float]]) -> Tuple[Individual, Individual]:
        """Apply blend crossover (BLX-α)."""
        p1_genes = parent1.genes
        p2_genes = parent2.genes
        child1_genes = np.zeros_like(p1_genes)
        child2_genes = np.zeros_like(p2_genes)
        for i in range(len(p1_genes)):
            # Blend crossover
            min_val = min(p1_genes[i], p2_genes[i])
            max_val = max(p1_genes[i], p2_genes[i])
            range_val = max_val - min_val
            # Extend range by alpha
            extended_min = min_val - self.crossover_alpha * range_val
            extended_max = max_val + self.crossover_alpha * range_val
            # Clip to bounds
            low, high = bounds[i]
            extended_min = max(extended_min, low)
            extended_max = min(extended_max, high)
            # Generate children
            child1_genes[i] = np.random.uniform(extended_min, extended_max)
            child2_genes[i] = np.random.uniform(extended_min, extended_max)
        return Individual(child1_genes), Individual(child2_genes)
class EvolutionStrategy(EvolutionaryAlgorithm):
    """
    Evolution Strategy (ES) algorithm.
    Uses self-adaptive mutation with strategy parameters that
    evolve along with the solution parameters.
    """
    def __init__(self, strategy: str = '(μ+λ)', mu: int = 15, lambda_: int = 100,
                 initial_sigma: float = 1.0, tau: Optional[float] = None,
                 tau_prime: Optional[float] = None, **kwargs):
        """
        Initialize Evolution Strategy.
        Args:
            strategy: ES strategy ('(μ+λ)' or '(μ,λ)')
            mu: Number of parents
            lambda_: Number of offspring
            initial_sigma: Initial mutation strength
            tau: Global learning rate
            tau_prime: Individual learning rate
        """
        super().__init__(population_size=lambda_, **kwargs)
        self.strategy = strategy
        self.mu = mu
        self.lambda_ = lambda_
        self.initial_sigma = initial_sigma
        # Set default learning rates
        if tau is None:
            self.tau = 1.0 / np.sqrt(2 * len(kwargs.get('bounds', [1])))
        else:
            self.tau = tau
        if tau_prime is None:
            self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(len(kwargs.get('bounds', [1]))))
        else:
            self.tau_prime = tau_prime
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize objective function using evolution strategy.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial best guess (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize population
        population = self.initialize_population(bounds)
        # Include initial guess if provided
        if x0 is not None:
            population[0].genes = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        # Evaluate initial population
        for individual in population:
            self.evaluate_fitness(objective, individual)
        self.nfev = len(population)
        self.fitness_history = []
        # Evolution loop
        for generation in range(self.generations):
            # Sort population
            population.sort(key=lambda x: x.fitness)
            # Statistics
            stats = self.get_statistics(population)
            self.fitness_history.append(stats['best_fitness'])
            if self.verbose and generation % 50 == 0:
                print(f"Gen {generation}: Best = {stats['best_fitness']:.6e}")
            # Select parents (best μ individuals)
            parents = population[:self.mu]
            # Generate offspring
            offspring = []
            for _ in range(self.lambda_):
                # Select random parent
                parent = random.choice(parents)
                # Create offspring through mutation
                child = self.mutate(parent, bounds)
                self.evaluate_fitness(objective, child)
                offspring.append(child)
            # Environmental selection
            if self.strategy == '(μ+λ)':
                # Combine parents and offspring
                combined = parents + offspring
                combined.sort(key=lambda x: x.fitness)
                population = combined[:self.mu]
            else:  # (μ,λ)
                # Select best offspring only
                offspring.sort(key=lambda x: x.fitness)
                population = offspring[:self.mu]
        # Find best solution
        best_individual = min(population, key=lambda x: x.fitness)
        execution_time = time.time() - start_time
        success = abs(best_individual.fitness) < 1e-6
        return OptimizationResult(
            x=best_individual.genes,
            fun=best_individual.fitness,
            nit=self.generations,
            nfev=self.nfev,
            success=success,
            message="Evolution completed",
            execution_time=execution_time
        )
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> List[Individual]:
        """Initialize population with strategy parameters."""
        population = []
        dim = len(bounds)
        for _ in range(self.mu):
            genes = np.array([np.random.uniform(low, high) for low, high in bounds])
            # Initialize strategy parameters (mutation strengths)
            sigma = np.full(dim, self.initial_sigma)
            individual = Individual(genes)
            individual.metadata['sigma'] = sigma
            population.append(individual)
        return population
    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        """Apply self-adaptive mutation."""
        genes = individual.genes.copy()
        sigma = individual.metadata['sigma'].copy()
        # Update strategy parameters
        global_factor = np.exp(self.tau * np.random.normal())
        for i in range(len(sigma)):
            individual_factor = np.exp(self.tau_prime * np.random.normal())
            sigma[i] *= global_factor * individual_factor
            # Mutate gene
            genes[i] += sigma[i] * np.random.normal()
            # Clip to bounds
            low, high = bounds[i]
            genes[i] = np.clip(genes[i], low, high)
        child = Individual(genes)
        child.metadata['sigma'] = sigma
        return child
    def crossover(self, parent1: Individual, parent2: Individual,
                 bounds: List[Tuple[float, float]]) -> Tuple[Individual, Individual]:
        """Crossover not typically used in ES."""
        return parent1, parent2
class GeneticProgramming(EvolutionaryAlgorithm):
    """
    Genetic Programming for evolving programs/expressions.
    Evolves tree-structured programs using genetic operators
    adapted for symbolic expressions.
    """
    def __init__(self, function_set: List[str] = None, terminal_set: List[str] = None,
                 max_depth: int = 6, init_method: str = 'half_and_half', **kwargs):
        """
        Initialize Genetic Programming.
        Args:
            function_set: Available functions
            terminal_set: Available terminals
            max_depth: Maximum tree depth
            init_method: Initialization method
        """
        super().__init__(**kwargs)
        if function_set is None:
            self.function_set = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
        else:
            self.function_set = function_set
        if terminal_set is None:
            self.terminal_set = ['x', 'y'] + [str(i) for i in range(-5, 6)]
        else:
            self.terminal_set = terminal_set
        self.max_depth = max_depth
        self.init_method = init_method
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Evolve programs using genetic programming.
        Note: This is a simplified implementation for demonstration.
        Real GP would require proper tree representation and evaluation.
        """
        # For now, defer to standard GA for numerical optimization
        ga = GeneticAlgorithm(
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            verbose=self.verbose
        )
        return ga.minimize(objective, bounds, x0, **kwargs)
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> List[Individual]:
        """Initialize population of programs."""
        # Simplified: use numerical representation
        population = []
        for _ in range(self.population_size):
            genes = np.array([np.random.uniform(low, high) for low, high in bounds])
            individual = Individual(genes)
            population.append(individual)
        return population
    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        """Mutate program tree."""
        # Simplified: numerical mutation
        mutated_genes = individual.genes.copy()
        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                low, high = bounds[i]
                mutated_genes[i] = np.random.uniform(low, high)
        return Individual(mutated_genes)
    def crossover(self, parent1: Individual, parent2: Individual,
                 bounds: List[Tuple[float, float]]) -> Tuple[Individual, Individual]:
        """Crossover program trees."""
        # Simplified: single-point crossover
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        if len(genes1) > 1:
            crossover_point = random.randint(1, len(genes1) - 1)
            child1_genes = np.concatenate([genes1[:crossover_point], genes2[crossover_point:]])
            child2_genes = np.concatenate([genes2[:crossover_point], genes1[crossover_point:]])
        else:
            child1_genes = genes1
            child2_genes = genes2
        return Individual(child1_genes), Individual(child2_genes)
def demo():
    """Demonstrate genetic algorithms and evolutionary computation."""
    print("Genetic Algorithms Demo")
    print("======================")
    print()
    # Test functions
    def sphere(x):
        """Sphere function."""
        return sum(xi**2 for xi in x)
    def rosenbrock(x):
        """Rosenbrock function."""
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    def rastrigin(x):
        """Rastrigin function."""
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    bounds = [(-5, 5)] * 3
    # Test algorithms
    algorithms = {
        'Genetic Algorithm': GeneticAlgorithm(population_size=50, generations=100, verbose=False),
        'Evolution Strategy': EvolutionStrategy(mu=15, lambda_=60, generations=100, verbose=False),
        'Genetic Programming': GeneticProgramming(population_size=50, generations=100, verbose=False)
    }
    test_functions = [('Sphere', sphere), ('Rosenbrock', rosenbrock), ('Rastrigin', rastrigin)]
    for func_name, func in test_functions:
        print(f"Testing on {func_name} function:")
        print("-" * (20 + len(func_name)))
        for alg_name, optimizer in algorithms.items():
            try:
                result = optimizer.minimize(func, bounds)
                print(f"{alg_name:20s}: f = {result.fun:.6e}, "
                      f"x = [{', '.join([f'{xi:.4f}' for xi in result.x])}], "
                      f"gen = {result.nit}")
            except Exception as e:
                print(f"{alg_name:20s}: Error - {str(e)}")
        print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()