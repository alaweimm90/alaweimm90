"""
Global Optimization Algorithms
=============================
This module implements various global optimization algorithms including
metaheuristic methods, stochastic optimization, and population-based
approaches for finding global optima in complex landscapes.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
import random
from .unconstrained import OptimizationResult
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
class GlobalOptimizer(ABC):
    """
    Abstract base class for global optimization algorithms.
    This class provides a common interface for all global
    optimization methods in the Berkeley SciComp framework.
    """
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 track_path: bool = False, verbose: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize global optimizer.
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            track_path: Whether to track optimization path
            verbose: Whether to print progress information
            random_seed: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.track_path = track_path
        self.verbose = verbose
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        # Counters
        self.nfev = 0
    @abstractmethod
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize the objective function globally.
        Args:
            objective: Objective function to minimize
            bounds: List of (min, max) bounds for each variable
            x0: Initial guess (optional)
            **kwargs: Additional algorithm-specific parameters
        Returns:
            OptimizationResult object
        """
        pass
    def _evaluate_function(self, func: Callable, x: np.ndarray) -> float:
        """Evaluate function with counter increment."""
        self.nfev += 1
        return func(x)
    def _random_point(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate random point within bounds."""
        return np.array([np.random.uniform(low, high) for low, high in bounds])
    def _clip_to_bounds(self, x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Clip point to bounds."""
        clipped = x.copy()
        for i, (low, high) in enumerate(bounds):
            clipped[i] = np.clip(clipped[i], low, high)
        return clipped
class SimulatedAnnealing(GlobalOptimizer):
    """
    Simulated Annealing global optimization algorithm.
    Mimics the annealing process in metallurgy to find global optima
    by accepting worse solutions with decreasing probability.
    """
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.95,
                 min_temp: float = 0.01, step_size: float = 1.0,
                 schedule: str = 'exponential', **kwargs):
        """
        Initialize Simulated Annealing optimizer.
        Args:
            initial_temp: Initial temperature
            cooling_rate: Temperature reduction factor
            min_temp: Minimum temperature (stopping criterion)
            step_size: Step size for random moves
            schedule: Cooling schedule ('exponential', 'linear', 'logarithmic')
        """
        super().__init__(**kwargs)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.step_size = step_size
        self.schedule = schedule
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize function using Simulated Annealing.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial point (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        if x0 is not None:
            x_current = self._clip_to_bounds(x0.copy(), bounds)
        else:
            x_current = self._random_point(bounds)
        f_current = self._evaluate_function(objective, x_current)
        # Best solution found
        x_best = x_current.copy()
        f_best = f_current
        # Temperature
        temp = self.initial_temp
        path = [x_current.copy()] if self.track_path else None
        self.nfev = 0
        iteration = 0
        while temp > self.min_temp and iteration < self.max_iterations:
            # Generate neighbor
            x_new = self._generate_neighbor(x_current, bounds)
            f_new = self._evaluate_function(objective, x_new)
            # Accept or reject move
            delta_f = f_new - f_current
            if delta_f < 0 or np.random.random() < np.exp(-delta_f / temp):
                x_current = x_new
                f_current = f_new
                # Update best solution
                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new
            # Cool down
            temp = self._update_temperature(temp, iteration)
            if self.track_path:
                path.append(x_current.copy())
            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: f_best = {f_best:.6e}, f_current = {f_current:.6e}, T = {temp:.6e}")
            iteration += 1
        success = temp <= self.min_temp or abs(f_best) < self.tolerance
        message = "Temperature below minimum" if temp <= self.min_temp else f"Maximum iterations reached"
        execution_time = time.time() - start_time
        return OptimizationResult(
            x=x_best, fun=f_best, nit=iteration, nfev=self.nfev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _generate_neighbor(self, x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate neighbor solution."""
        neighbor = x + np.random.normal(0, self.step_size, size=len(x))
        return self._clip_to_bounds(neighbor, bounds)
    def _update_temperature(self, temp: float, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        if self.schedule == 'exponential':
            return temp * self.cooling_rate
        elif self.schedule == 'linear':
            return self.initial_temp * (1 - iteration / self.max_iterations)
        elif self.schedule == 'logarithmic':
            return self.initial_temp / np.log(iteration + 2)
        else:
            return temp * self.cooling_rate
class ParticleSwarmOptimization(GlobalOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm.
    Simulates social behavior of bird flocking or fish schooling
    to find global optima through collective intelligence.
    """
    def __init__(self, swarm_size: int = 30, inertia: float = 0.9,
                 cognitive_param: float = 2.0, social_param: float = 2.0,
                 inertia_decay: float = 0.99, velocity_clamp: float = None,
                 **kwargs):
        """
        Initialize PSO optimizer.
        Args:
            swarm_size: Number of particles in swarm
            inertia: Inertia weight for velocity update
            cognitive_param: Cognitive acceleration parameter
            social_param: Social acceleration parameter
            inertia_decay: Inertia decay rate
            velocity_clamp: Maximum velocity magnitude
        """
        super().__init__(**kwargs)
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.inertia_decay = inertia_decay
        self.velocity_clamp = velocity_clamp
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize function using PSO.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial best guess (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        dim = len(bounds)
        # Initialize swarm
        positions = np.array([self._random_point(bounds) for _ in range(self.swarm_size)])
        velocities = np.zeros((self.swarm_size, dim))
        # Evaluate initial positions
        fitness = np.array([self._evaluate_function(objective, pos) for pos in positions])
        # Personal best positions and fitness
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        # Global best
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        path = [global_best_position.copy()] if self.track_path else None
        self.nfev = self.swarm_size  # Already evaluated all particles
        current_inertia = self.inertia
        for iteration in range(self.max_iterations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.random(dim), np.random.random(dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])
                velocities[i] = (current_inertia * velocities[i] +
                               cognitive_velocity + social_velocity)
                # Clamp velocity if specified
                if self.velocity_clamp is not None:
                    velocity_magnitude = np.linalg.norm(velocities[i])
                    if velocity_magnitude > self.velocity_clamp:
                        velocities[i] = (self.velocity_clamp / velocity_magnitude) * velocities[i]
                # Update position
                positions[i] = self._clip_to_bounds(positions[i] + velocities[i], bounds)
                # Evaluate fitness
                fitness[i] = self._evaluate_function(objective, positions[i])
                # Update personal best
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_fitness[i] = fitness[i]
                    # Update global best
                    if fitness[i] < global_best_fitness:
                        global_best_position = positions[i].copy()
                        global_best_fitness = fitness[i]
            # Update inertia
            current_inertia *= self.inertia_decay
            if self.track_path:
                path.append(global_best_position.copy())
            if self.verbose and iteration % 50 == 0:
                print(f"Iter {iteration}: f_best = {global_best_fitness:.6e}, inertia = {current_inertia:.6e}")
            # Check convergence
            if abs(global_best_fitness) < self.tolerance:
                success = True
                message = f"Converged to tolerance: {global_best_fitness:.2e}"
                break
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        return OptimizationResult(
            x=global_best_position, fun=global_best_fitness, nit=iteration+1,
            nfev=self.nfev, success=success, message=message,
            execution_time=execution_time, path=path
        )
class DifferentialEvolution(GlobalOptimizer):
    """
    Differential Evolution (DE) algorithm.
    Evolutionary algorithm that uses vector differences for mutation
    and is particularly effective for continuous optimization problems.
    """
    def __init__(self, population_size: int = None, mutation_factor: float = 0.8,
                 crossover_prob: float = 0.7, strategy: str = 'rand/1/bin',
                 **kwargs):
        """
        Initialize DE optimizer.
        Args:
            population_size: Population size (default: 15 * dim)
            mutation_factor: Differential weight F
            crossover_prob: Crossover probability
            strategy: Mutation strategy
        """
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize function using Differential Evolution.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial best guess (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        dim = len(bounds)
        # Set default population size
        if self.population_size is None:
            pop_size = max(15 * dim, 50)
        else:
            pop_size = self.population_size
        # Initialize population
        population = np.array([self._random_point(bounds) for _ in range(pop_size)])
        # Include initial guess if provided
        if x0 is not None:
            population[0] = self._clip_to_bounds(x0.copy(), bounds)
        # Evaluate population
        fitness = np.array([self._evaluate_function(objective, ind) for ind in population])
        # Track best solution
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        path = [best_individual.copy()] if self.track_path else None
        self.nfev = pop_size
        for generation in range(self.max_iterations):
            for i in range(pop_size):
                # Mutation
                mutant = self._mutate(population, i, bounds)
                # Crossover
                trial = self._crossover(population[i], mutant)
                trial = self._clip_to_bounds(trial, bounds)
                # Selection
                trial_fitness = self._evaluate_function(objective, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    # Update best
                    if trial_fitness < best_fitness:
                        best_individual = trial.copy()
                        best_fitness = trial_fitness
            if self.track_path:
                path.append(best_individual.copy())
            if self.verbose and generation % 50 == 0:
                print(f"Gen {generation}: f_best = {best_fitness:.6e}")
            # Check convergence
            if abs(best_fitness) < self.tolerance:
                success = True
                message = f"Converged to tolerance: {best_fitness:.2e}"
                break
        else:
            success = False
            message = f"Maximum generations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        return OptimizationResult(
            x=best_individual, fun=best_fitness, nit=generation+1,
            nfev=self.nfev, success=success, message=message,
            execution_time=execution_time, path=path
        )
    def _mutate(self, population: np.ndarray, target_idx: int,
               bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Perform mutation operation."""
        pop_size, dim = population.shape
        # Select random individuals (different from target)
        candidates = [i for i in range(pop_size) if i != target_idx]
        if len(candidates) < 3:
            return population[target_idx].copy()
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        if self.strategy == 'rand/1/bin':
            mutant = population[r1] + self.mutation_factor * (population[r2] - population[r3])
        elif self.strategy == 'best/1/bin':
            best_idx = np.argmin([self._evaluate_function(lambda x: 0, ind) for ind in population])
            mutant = population[best_idx] + self.mutation_factor * (population[r1] - population[r2])
        else:
            # Default to rand/1/bin
            mutant = population[r1] + self.mutation_factor * (population[r2] - population[r3])
        return self._clip_to_bounds(mutant, bounds)
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform crossover operation."""
        dim = len(target)
        trial = target.copy()
        # Ensure at least one parameter is from mutant
        cross_points = np.random.random(dim) < self.crossover_prob
        cross_points[np.random.randint(0, dim)] = True
        trial[cross_points] = mutant[cross_points]
        return trial
class BasinHopping(GlobalOptimizer):
    """
    Basin Hopping global optimization algorithm.
    Combines local optimization with random perturbations to escape
    local minima and find global optima.
    """
    def __init__(self, step_size: float = 0.5, local_optimizer: str = 'L-BFGS-B',
                 accept_test: Optional[Callable] = None, **kwargs):
        """
        Initialize Basin Hopping optimizer.
        Args:
            step_size: Step size for random displacement
            local_optimizer: Local optimization method
            accept_test: Custom acceptance test function
        """
        super().__init__(**kwargs)
        self.step_size = step_size
        self.local_optimizer = local_optimizer
        self.accept_test = accept_test
    def minimize(self, objective: Callable, bounds: List[Tuple[float, float]],
                x0: Optional[np.ndarray] = None, **kwargs) -> OptimizationResult:
        """
        Minimize function using Basin Hopping.
        Args:
            objective: Function to minimize
            bounds: Variable bounds
            x0: Initial point (optional)
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        if x0 is not None:
            x_current = self._clip_to_bounds(x0.copy(), bounds)
        else:
            x_current = self._random_point(bounds)
        # Perform initial local optimization
        f_current = self._local_minimize(objective, x_current, bounds)
        # Best solution
        x_best = x_current.copy()
        f_best = f_current
        path = [x_current.copy()] if self.track_path else None
        self.nfev = 0
        for iteration in range(self.max_iterations):
            # Random perturbation
            x_trial = self._perturb(x_current, bounds)
            # Local optimization
            f_trial = self._local_minimize(objective, x_trial, bounds)
            # Acceptance test
            accept = self._accept_step(f_current, f_trial)
            if accept:
                x_current = x_trial
                f_current = f_trial
                # Update best
                if f_trial < f_best:
                    x_best = x_trial.copy()
                    f_best = f_trial
            if self.track_path:
                path.append(x_current.copy())
            if self.verbose and iteration % 50 == 0:
                print(f"Iter {iteration}: f_best = {f_best:.6e}, f_current = {f_current:.6e}")
        success = abs(f_best) < self.tolerance
        message = "Converged" if success else f"Maximum iterations reached"
        execution_time = time.time() - start_time
        return OptimizationResult(
            x=x_best, fun=f_best, nit=self.max_iterations,
            nfev=self.nfev, success=success, message=message,
            execution_time=execution_time, path=path
        )
    def _local_minimize(self, objective: Callable, x0: np.ndarray,
                       bounds: List[Tuple[float, float]]) -> float:
        """Perform local minimization."""
        # Simple gradient descent for local optimization
        from .unconstrained import GradientDescent
        gd = GradientDescent(learning_rate=0.01, max_iterations=100)
        result = gd.minimize(objective, x0)
        self.nfev += result.nfev
        return result.fun
    def _perturb(self, x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Apply random perturbation."""
        perturbation = np.random.normal(0, self.step_size, size=len(x))
        x_new = x + perturbation
        return self._clip_to_bounds(x_new, bounds)
    def _accept_step(self, f_current: float, f_trial: float) -> bool:
        """Determine whether to accept step."""
        if self.accept_test is not None:
            return self.accept_test(f_current, f_trial)
        else:
            # Simple acceptance: always accept improvements
            return f_trial <= f_current
def demo():
    """Demonstrate global optimization algorithms."""
    print("Global Optimization Demo")
    print("=======================")
    print()
    # Test functions
    def rastrigin(x):
        """Rastrigin function (highly multimodal)."""
        A = 10
        n = len(x)
        return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    def ackley(x):
        """Ackley function (multimodal with many local minima)."""
        n = len(x)
        sum1 = sum([xi**2 for xi in x])
        sum2 = sum([np.cos(2 * np.pi * xi) for xi in x])
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    bounds = [(-5, 5), (-5, 5)]
    # Test algorithms
    algorithms = {
        'Simulated Annealing': SimulatedAnnealing(initial_temp=100, max_iterations=1000),
        'Particle Swarm': ParticleSwarmOptimization(swarm_size=30, max_iterations=200),
        'Differential Evolution': DifferentialEvolution(population_size=60, max_iterations=200),
        'Basin Hopping': BasinHopping(step_size=0.5, max_iterations=100)
    }
    test_functions = [('Rastrigin', rastrigin), ('Ackley', ackley)]
    for func_name, func in test_functions:
        print(f"Testing on {func_name} function:")
        print("-" * (20 + len(func_name)))
        for alg_name, optimizer in algorithms.items():
            result = optimizer.minimize(func, bounds)
            print(f"{alg_name:20s}: f = {result.fun:.6e}, x = [{result.x[0]:.4f}, {result.x[1]:.4f}], iter = {result.nit}")
        print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()