"""Monte Carlo Optimization Algorithms.
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
"""
import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
from .utils import set_seed, compute_statistics
@dataclass
class OptimizationResult:
    """Result of Monte Carlo optimization.
    Attributes:
        x_optimal: Optimal solution found
        f_optimal: Optimal function value
        n_evaluations: Number of function evaluations
        convergence_history: History of best values
        success: Whether optimization succeeded
        message: Status message
        metadata: Additional optimization information
    """
    x_optimal: np.ndarray
    f_optimal: float
    n_evaluations: int
    convergence_history: List[float]
    success: bool
    message: str
    metadata: Dict[str, Any]
class MonteCarloOptimizer(ABC):
    """Abstract base class for Monte Carlo optimizers."""
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    @abstractmethod
    def optimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                **kwargs) -> OptimizationResult:
        """Optimize objective function."""
        pass
class SimulatedAnnealing(MonteCarloOptimizer):
    """Simulated Annealing optimizer.
    Implements simulated annealing with various cooling schedules
    and neighborhood generation strategies.
    Parameters:
        initial_temperature: Starting temperature
        cooling_schedule: Cooling schedule ('linear', 'exponential', 'logarithmic')
        max_iterations: Maximum number of iterations
        min_temperature: Minimum temperature
        step_size: Initial step size for neighborhood generation
    """
    def __init__(self,
                 initial_temperature: float = 100.0,
                 cooling_schedule: str = 'exponential',
                 max_iterations: int = 10000,
                 min_temperature: float = 1e-8,
                 step_size: float = 1.0,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(random_state, verbose)
        self.initial_temperature = initial_temperature
        self.cooling_schedule = cooling_schedule
        self.max_iterations = max_iterations
        self.min_temperature = min_temperature
        self.step_size = step_size
    def optimize(self,
                 objective: Callable,
                 bounds: List[Tuple[float, float]],
                 initial_guess: Optional[np.ndarray] = None,
                 **kwargs) -> OptimizationResult:
        """Optimize using simulated annealing.
        Args:
            objective: Objective function to minimize
            bounds: Variable bounds [(min, max), ...]
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            OptimizationResult: Optimization results
        """
        bounds = np.array(bounds)
        n_vars = len(bounds)
        # Initialize solution
        if initial_guess is None:
            current_x = np.array([self.rng.uniform(low, high)
                                 for low, high in bounds])
        else:
            current_x = np.array(initial_guess)
            current_x = self._clip_to_bounds(current_x, bounds)
        current_f = objective(current_x)
        best_x = current_x.copy()
        best_f = current_f
        # Storage
        convergence_history = [best_f]
        n_evaluations = 1
        n_accepted = 0
        # Cooling parameters
        temperature = self.initial_temperature
        for iteration in range(self.max_iterations):
            # Generate candidate solution
            candidate_x = self._generate_neighbor(current_x, bounds, temperature)
            candidate_f = objective(candidate_x)
            n_evaluations += 1
            # Acceptance decision
            if self._accept_candidate(current_f, candidate_f, temperature):
                current_x = candidate_x
                current_f = candidate_f
                n_accepted += 1
                # Update best solution
                if candidate_f < best_f:
                    best_x = candidate_x.copy()
                    best_f = candidate_f
            # Update temperature
            temperature = self._update_temperature(iteration)
            # Store convergence history
            convergence_history.append(best_f)
            # Check termination criteria
            if temperature < self.min_temperature:
                break
            # Progress reporting
            if self.verbose and (iteration + 1) % 1000 == 0:
                acceptance_rate = n_accepted / (iteration + 1)
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best: {best_f:.6f}, Temperature: {temperature:.6f}, "
                      f"Acceptance: {acceptance_rate:.3f}")
        # Final results
        success = n_evaluations < self.max_iterations * 2  # Heuristic success criterion
        message = f"Optimization completed in {iteration + 1} iterations"
        metadata = {
            'final_temperature': temperature,
            'acceptance_rate': n_accepted / (iteration + 1),
            'cooling_schedule': self.cooling_schedule,
            'n_accepted': n_accepted
        }
        return OptimizationResult(
            x_optimal=best_x,
            f_optimal=best_f,
            n_evaluations=n_evaluations,
            convergence_history=convergence_history,
            success=success,
            message=message,
            metadata=metadata
        )
    def _generate_neighbor(self, x: np.ndarray, bounds: np.ndarray,
                          temperature: float) -> np.ndarray:
        """Generate neighbor solution."""
        # Adaptive step size based on temperature
        adaptive_step = self.step_size * np.sqrt(temperature / self.initial_temperature)
        # Generate perturbation
        perturbation = self.rng.normal(0, adaptive_step, len(x))
        candidate = x + perturbation
        return self._clip_to_bounds(candidate, bounds)
    def _clip_to_bounds(self, x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Clip solution to bounds."""
        return np.clip(x, bounds[:, 0], bounds[:, 1])
    def _accept_candidate(self, current_f: float, candidate_f: float,
                         temperature: float) -> bool:
        """Decide whether to accept candidate solution."""
        if candidate_f < current_f:
            return True
        else:
            # Metropolis criterion
            delta = candidate_f - current_f
            probability = np.exp(-delta / temperature)
            return self.rng.random() < probability
    def _update_temperature(self, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        progress = iteration / self.max_iterations
        if self.cooling_schedule == 'linear':
            return self.initial_temperature * (1 - progress)
        elif self.cooling_schedule == 'exponential':
            return self.initial_temperature * np.exp(-5 * progress)
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temperature / (1 + np.log(1 + iteration))
        else:
            raise ValueError(f"Unknown cooling schedule: {self.cooling_schedule}")
class GeneticAlgorithm(MonteCarloOptimizer):
    """Genetic Algorithm optimizer.
    Implements genetic algorithm with various selection, crossover,
    and mutation strategies.
    """
    def __init__(self,
                 population_size: int = 50,
                 n_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'uniform',
                 mutation_method: str = 'gaussian',
                 elitism_rate: float = 0.1,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(random_state, verbose)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.elitism_rate = elitism_rate
    def optimize(self,
                 objective: Callable,
                 bounds: List[Tuple[float, float]],
                 **kwargs) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        bounds = np.array(bounds)
        n_vars = len(bounds)
        # Initialize population
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(objective, population)
        # Storage
        convergence_history = []
        n_evaluations = len(population)
        for generation in range(self.n_generations):
            # Selection
            selected_population = self._selection(population, fitness)
            # Crossover
            offspring = self._crossover(selected_population, bounds)
            # Mutation
            offspring = self._mutation(offspring, bounds)
            # Elitism
            elite_indices = np.argsort(fitness)[:int(self.elitism_rate * self.population_size)]
            elite_population = population[elite_indices]
            # Form new population
            n_elite = len(elite_population)
            n_offspring = self.population_size - n_elite
            population = np.vstack([elite_population, offspring[:n_offspring]])
            # Evaluate new population
            fitness = self._evaluate_population(objective, population)
            n_evaluations += len(population)
            # Track convergence
            best_fitness = np.min(fitness)
            convergence_history.append(best_fitness)
            # Progress reporting
            if self.verbose and (generation + 1) % 10 == 0:
                avg_fitness = np.mean(fitness)
                print(f"Generation {generation + 1}/{self.n_generations}, "
                      f"Best: {best_fitness:.6f}, Average: {avg_fitness:.6f}")
        # Final results
        best_idx = np.argmin(fitness)
        best_x = population[best_idx]
        best_f = fitness[best_idx]
        success = True  # GA typically converges
        message = f"Genetic algorithm completed {self.n_generations} generations"
        metadata = {
            'final_population_size': len(population),
            'selection_method': self.selection_method,
            'crossover_method': self.crossover_method,
            'mutation_method': self.mutation_method
        }
        return OptimizationResult(
            x_optimal=best_x,
            f_optimal=best_f,
            n_evaluations=n_evaluations,
            convergence_history=convergence_history,
            success=success,
            message=message,
            metadata=metadata
        )
    def _initialize_population(self, bounds: np.ndarray) -> np.ndarray:
        """Initialize random population."""
        population = np.zeros((self.population_size, len(bounds)))
        for i in range(len(bounds)):
            low, high = bounds[i]
            population[:, i] = self.rng.uniform(low, high, self.population_size)
        return population
    def _evaluate_population(self, objective: Callable,
                           population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of entire population."""
        return np.array([objective(individual) for individual in population])
    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select individuals for reproduction."""
        if self.selection_method == 'tournament':
            return self._tournament_selection(population, fitness)
        elif self.selection_method == 'roulette':
            return self._roulette_selection(population, fitness)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    def _tournament_selection(self, population: np.ndarray,
                            fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            # Select tournament participants
            tournament_indices = self.rng.choice(len(population), tournament_size)
            tournament_fitness = fitness[tournament_indices]
            # Select winner
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])
        return np.array(selected)
    def _roulette_selection(self, population: np.ndarray,
                          fitness: np.ndarray) -> np.ndarray:
        """Roulette wheel selection."""
        # Convert to selection probabilities (for minimization)
        max_fitness = np.max(fitness)
        selection_fitness = max_fitness - fitness + 1e-8
        probabilities = selection_fitness / np.sum(selection_fitness)
        # Select individuals
        selected_indices = self.rng.choice(len(population),
                                         size=self.population_size,
                                         p=probabilities)
        return population[selected_indices]
    def _crossover(self, population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Apply crossover to population."""
        offspring = []
        n_pairs = len(population) // 2
        for i in range(n_pairs):
            parent1 = population[2*i]
            parent2 = population[2*i + 1]
            if self.rng.random() < self.crossover_rate:
                if self.crossover_method == 'uniform':
                    child1, child2 = self._uniform_crossover(parent1, parent2)
                elif self.crossover_method == 'single_point':
                    child1, child2 = self._single_point_crossover(parent1, parent2)
                elif self.crossover_method == 'arithmetic':
                    child1, child2 = self._arithmetic_crossover(parent1, parent2)
                else:
                    raise ValueError(f"Unknown crossover method: {self.crossover_method}")
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Ensure bounds
            child1 = np.clip(child1, bounds[:, 0], bounds[:, 1])
            child2 = np.clip(child2, bounds[:, 0], bounds[:, 1])
            offspring.extend([child1, child2])
        return np.array(offspring)
    def _uniform_crossover(self, parent1: np.ndarray,
                          parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = self.rng.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    def _single_point_crossover(self, parent1: np.ndarray,
                               parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = self.rng.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    def _arithmetic_crossover(self, parent1: np.ndarray,
                             parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Arithmetic crossover."""
        alpha = self.rng.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    def _mutation(self, population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Apply mutation to population."""
        mutated = population.copy()
        for i in range(len(population)):
            if self.rng.random() < self.mutation_rate:
                if self.mutation_method == 'gaussian':
                    mutated[i] = self._gaussian_mutation(population[i], bounds)
                elif self.mutation_method == 'uniform':
                    mutated[i] = self._uniform_mutation(population[i], bounds)
                else:
                    raise ValueError(f"Unknown mutation method: {self.mutation_method}")
        return mutated
    def _gaussian_mutation(self, individual: np.ndarray,
                          bounds: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutation_strength = 0.1 * (bounds[:, 1] - bounds[:, 0])
        mutation = self.rng.normal(0, mutation_strength)
        mutated = individual + mutation
        return np.clip(mutated, bounds[:, 0], bounds[:, 1])
    def _uniform_mutation(self, individual: np.ndarray,
                         bounds: np.ndarray) -> np.ndarray:
        """Uniform mutation."""
        mutated = individual.copy()
        mutation_mask = self.rng.random(len(individual)) < 0.1  # 10% of genes
        for i in np.where(mutation_mask)[0]:
            low, high = bounds[i]
            mutated[i] = self.rng.uniform(low, high)
        return mutated
class ParticleSwarmOptimization(MonteCarloOptimizer):
    """Particle Swarm Optimization.
    Implements PSO with inertia weight and constriction factor variants.
    """
    def __init__(self,
                 n_particles: int = 30,
                 max_iterations: int = 1000,
                 w: float = 0.9,  # Inertia weight
                 c1: float = 2.0,  # Cognitive parameter
                 c2: float = 2.0,  # Social parameter
                 w_min: float = 0.4,
                 w_max: float = 0.9,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(random_state, verbose)
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
    def optimize(self,
                 objective: Callable,
                 bounds: List[Tuple[float, float]],
                 **kwargs) -> OptimizationResult:
        """Optimize using particle swarm optimization."""
        bounds = np.array(bounds)
        n_vars = len(bounds)
        # Initialize particles
        positions = self._initialize_positions(bounds)
        velocities = self._initialize_velocities(bounds)
        # Evaluate initial positions
        fitness = np.array([objective(pos) for pos in positions])
        n_evaluations = len(positions)
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        # Storage
        convergence_history = [global_best_fitness]
        for iteration in range(self.max_iterations):
            # Update inertia weight
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
            # Update velocities and positions
            for i in range(self.n_particles):
                # Velocity update
                r1, r2 = self.rng.random(2)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = (w * velocities[i] +
                               cognitive_velocity +
                               social_velocity)
                # Position update
                positions[i] += velocities[i]
                # Enforce bounds
                positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
                # Evaluate new position
                fitness[i] = objective(positions[i])
                n_evaluations += 1
                # Update personal best
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_fitness[i] = fitness[i]
                    # Update global best
                    if fitness[i] < global_best_fitness:
                        global_best_position = positions[i].copy()
                        global_best_fitness = fitness[i]
            # Store convergence history
            convergence_history.append(global_best_fitness)
            # Progress reporting
            if self.verbose and (iteration + 1) % 100 == 0:
                avg_fitness = np.mean(fitness)
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best: {global_best_fitness:.6f}, Average: {avg_fitness:.6f}")
        # Final results
        success = True
        message = f"PSO completed {self.max_iterations} iterations"
        metadata = {
            'final_inertia_weight': w,
            'n_particles': self.n_particles,
            'cognitive_parameter': self.c1,
            'social_parameter': self.c2
        }
        return OptimizationResult(
            x_optimal=global_best_position,
            f_optimal=global_best_fitness,
            n_evaluations=n_evaluations,
            convergence_history=convergence_history,
            success=success,
            message=message,
            metadata=metadata
        )
    def _initialize_positions(self, bounds: np.ndarray) -> np.ndarray:
        """Initialize particle positions."""
        positions = np.zeros((self.n_particles, len(bounds)))
        for i in range(len(bounds)):
            low, high = bounds[i]
            positions[:, i] = self.rng.uniform(low, high, self.n_particles)
        return positions
    def _initialize_velocities(self, bounds: np.ndarray) -> np.ndarray:
        """Initialize particle velocities."""
        velocities = np.zeros((self.n_particles, len(bounds)))
        for i in range(len(bounds)):
            low, high = bounds[i]
            v_max = (high - low) * 0.1  # 10% of range
            velocities[:, i] = self.rng.uniform(-v_max, v_max, self.n_particles)
        return velocities
class CrossEntropyMethod(MonteCarloOptimizer):
    """Cross-Entropy Method optimizer.
    Implements the cross-entropy method for continuous optimization.
    """
    def __init__(self,
                 population_size: int = 100,
                 elite_fraction: float = 0.1,
                 max_iterations: int = 100,
                 smoothing_factor: float = 0.7,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(random_state, verbose)
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.max_iterations = max_iterations
        self.smoothing_factor = smoothing_factor
        self.n_elite = int(elite_fraction * population_size)
    def optimize(self,
                 objective: Callable,
                 bounds: List[Tuple[float, float]],
                 **kwargs) -> OptimizationResult:
        """Optimize using cross-entropy method."""
        bounds = np.array(bounds)
        n_vars = len(bounds)
        # Initialize distribution parameters
        mean = np.mean(bounds, axis=1)
        std = (bounds[:, 1] - bounds[:, 0]) / 6  # 3-sigma rule
        # Storage
        convergence_history = []
        n_evaluations = 0
        for iteration in range(self.max_iterations):
            # Sample population
            population = self._sample_population(mean, std, bounds)
            # Evaluate population
            fitness = np.array([objective(x) for x in population])
            n_evaluations += len(population)
            # Select elite samples
            elite_indices = np.argsort(fitness)[:self.n_elite]
            elite_samples = population[elite_indices]
            elite_fitness = fitness[elite_indices]
            # Update distribution parameters
            new_mean = np.mean(elite_samples, axis=0)
            new_std = np.std(elite_samples, axis=0, ddof=1)
            # Smoothing
            mean = self.smoothing_factor * mean + (1 - self.smoothing_factor) * new_mean
            std = self.smoothing_factor * std + (1 - self.smoothing_factor) * new_std
            # Store convergence history
            best_fitness = np.min(fitness)
            convergence_history.append(best_fitness)
            # Progress reporting
            if self.verbose and (iteration + 1) % 10 == 0:
                avg_fitness = np.mean(fitness)
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best: {best_fitness:.6f}, Average: {avg_fitness:.6f}")
        # Final results
        final_population = self._sample_population(mean, std, bounds)
        final_fitness = np.array([objective(x) for x in final_population])
        n_evaluations += len(final_population)
        best_idx = np.argmin(final_fitness)
        best_x = final_population[best_idx]
        best_f = final_fitness[best_idx]
        success = True
        message = f"Cross-entropy method completed {self.max_iterations} iterations"
        metadata = {
            'final_mean': mean,
            'final_std': std,
            'elite_fraction': self.elite_fraction,
            'smoothing_factor': self.smoothing_factor
        }
        return OptimizationResult(
            x_optimal=best_x,
            f_optimal=best_f,
            n_evaluations=n_evaluations,
            convergence_history=convergence_history,
            success=success,
            message=message,
            metadata=metadata
        )
    def _sample_population(self, mean: np.ndarray, std: np.ndarray,
                          bounds: np.ndarray) -> np.ndarray:
        """Sample population from current distribution."""
        population = self.rng.normal(mean, std, (self.population_size, len(mean)))
        # Clip to bounds
        population = np.clip(population, bounds[:, 0], bounds[:, 1])
        return population
# Convenience functions
def simulated_annealing(objective: Callable,
                       bounds: List[Tuple[float, float]],
                       initial_guess: Optional[np.ndarray] = None,
                       initial_temperature: float = 100.0,
                       max_iterations: int = 10000,
                       random_state: Optional[int] = None,
                       **kwargs) -> OptimizationResult:
    """Convenience function for simulated annealing."""
    optimizer = SimulatedAnnealing(
        initial_temperature=initial_temperature,
        max_iterations=max_iterations,
        random_state=random_state,
        **kwargs
    )
    return optimizer.optimize(objective, bounds, initial_guess=initial_guess)
def genetic_algorithm(objective: Callable,
                     bounds: List[Tuple[float, float]],
                     population_size: int = 50,
                     n_generations: int = 100,
                     random_state: Optional[int] = None,
                     **kwargs) -> OptimizationResult:
    """Convenience function for genetic algorithm."""
    optimizer = GeneticAlgorithm(
        population_size=population_size,
        n_generations=n_generations,
        random_state=random_state,
        **kwargs
    )
    return optimizer.optimize(objective, bounds)
def particle_swarm(objective: Callable,
                  bounds: List[Tuple[float, float]],
                  n_particles: int = 30,
                  max_iterations: int = 1000,
                  random_state: Optional[int] = None,
                  **kwargs) -> OptimizationResult:
    """Convenience function for particle swarm optimization."""
    optimizer = ParticleSwarmOptimization(
        n_particles=n_particles,
        max_iterations=max_iterations,
        random_state=random_state,
        **kwargs
    )
    return optimizer.optimize(objective, bounds)
def cross_entropy_method(objective: Callable,
                        bounds: List[Tuple[float, float]],
                        population_size: int = 100,
                        max_iterations: int = 100,
                        random_state: Optional[int] = None,
                        **kwargs) -> OptimizationResult:
    """Convenience function for cross-entropy method."""
    optimizer = CrossEntropyMethod(
        population_size=population_size,
        max_iterations=max_iterations,
        random_state=random_state,
        **kwargs
    )
    return optimizer.optimize(objective, bounds)