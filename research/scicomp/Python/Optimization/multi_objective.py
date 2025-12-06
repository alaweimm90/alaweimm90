"""
Multi-Objective Optimization Algorithms
======================================
This module implements algorithms for multi-objective optimization including
Pareto optimization, weighted sum methods, epsilon-constraint methods, and
evolutionary multi-objective algorithms like NSGA-II.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from .unconstrained import OptimizationResult
from .genetic_algorithms import Individual, EvolutionaryAlgorithm
import warnings
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class MultiObjectiveResult:
    """
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
    """
    pareto_set: np.ndarray
    pareto_front: np.ndarray
    hypervolume: Optional[float] = None
    spacing: Optional[float] = None
    success: bool = False
    message: str = ""
    nit: int = 0
    nfev: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
class MultiObjectiveProblem:
    """
    Represents a multi-objective optimization problem.
    Encapsulates multiple objective functions and constraints
    for multi-objective optimization algorithms.
    """
    def __init__(self, objectives: List[Callable], n_variables: int,
                 bounds: List[Tuple[float, float]], constraints: Optional[List] = None,
                 objective_names: Optional[List[str]] = None):
        """
        Initialize multi-objective problem.
        Args:
            objectives: List of objective functions
            n_variables: Number of decision variables
            bounds: Variable bounds
            constraints: List of constraints (optional)
            objective_names: Names of objectives (optional)
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.n_variables = n_variables
        self.bounds = bounds
        self.constraints = constraints or []
        self.objective_names = objective_names or [f"f{i}" for i in range(self.n_objectives)]
        # Counters
        self.nfev = 0
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all objectives at point x."""
        self.nfev += 1
        return np.array([obj(x) for obj in self.objectives])
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate objectives for entire population."""
        return np.array([self.evaluate(ind) for ind in population])
    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if solution is feasible."""
        # Check bounds
        for i, (low, high) in enumerate(self.bounds):
            if x[i] < low or x[i] > high:
                return False
        # Check constraints
        for constraint in self.constraints:
            value = constraint.fun(x)
            if constraint.type == 'eq' and abs(value) > 1e-6:
                return False
            elif constraint.type == 'ineq' and value > 1e-6:
                return False
        return True
def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 dominates obj2 (for minimization).
    obj1 dominates obj2 if:
    - obj1 is at least as good as obj2 in all objectives
    - obj1 is strictly better than obj2 in at least one objective
    """
    at_least_as_good = np.all(obj1 <= obj2)
    strictly_better = np.any(obj1 < obj2)
    return at_least_as_good and strictly_better
def non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Perform non-dominated sorting of population.
    Args:
        objectives: Array of objective values (n_individuals x n_objectives)
    Returns:
        List of fronts, where each front is a list of indices
    """
    n = len(objectives)
    # For each individual, track:
    # - domination_count: number of individuals that dominate it
    # - dominated_set: set of individuals it dominates
    domination_count = np.zeros(n, dtype=int)
    dominated_sets = [[] for _ in range(n)]
    # First pass: determine domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_sets[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_sets[j].append(i)
                domination_count[i] += 1
    # Create fronts
    fronts = []
    current_front = [i for i in range(n) if domination_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
    return fronts
def crowding_distance(objectives: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Calculate crowding distance for solutions in a front.
    Args:
        objectives: All objective values
        indices: Indices of solutions in the front
    Returns:
        Crowding distances for each solution in the front
    """
    if len(indices) <= 2:
        return np.full(len(indices), float('inf'))
    front_objectives = objectives[indices]
    n_solutions = len(indices)
    n_objectives = objectives.shape[1]
    # Initialize distances
    distances = np.zeros(n_solutions)
    # For each objective
    for m in range(n_objectives):
        # Sort by this objective
        sorted_indices = np.argsort(front_objectives[:, m])
        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        # Calculate range of objective
        obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
        if obj_range > 0:
            # Calculate distances for intermediate solutions
            for i in range(1, n_solutions - 1):
                distances[sorted_indices[i]] += (
                    front_objectives[sorted_indices[i + 1], m] -
                    front_objectives[sorted_indices[i - 1], m]
                ) / obj_range
    return distances
class NSGA2(EvolutionaryAlgorithm):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    One of the most popular evolutionary algorithms for multi-objective
    optimization, using non-dominated sorting and crowding distance.
    """
    def __init__(self, population_size: int = 100, generations: int = 500,
                 crossover_rate: float = 0.9, mutation_rate: float = None,
                 crossover_eta: float = 20.0, mutation_eta: float = 20.0,
                 **kwargs):
        """
        Initialize NSGA-II.
        Args:
            population_size: Size of population
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability (default: 1/n_variables)
            crossover_eta: Distribution index for crossover
            mutation_eta: Distribution index for mutation
        """
        super().__init__(population_size=population_size, generations=generations,
                        crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                        **kwargs)
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
    def solve(self, problem: MultiObjectiveProblem) -> MultiObjectiveResult:
        """
        Solve multi-objective problem using NSGA-II.
        Args:
            problem: Multi-objective optimization problem
        Returns:
            MultiObjectiveResult
        """
        start_time = time.time()
        # Set default mutation rate
        if self.mutation_rate is None:
            self.mutation_rate = 1.0 / problem.n_variables
        # Initialize population
        population = self._initialize_population(problem)
        objectives = problem.evaluate_population(population)
        # Evolution loop
        for generation in range(self.generations):
            # Create offspring
            offspring = self._create_offspring(population, problem)
            offspring_objectives = problem.evaluate_population(offspring)
            # Combine parent and offspring populations
            combined_population = np.vstack([population, offspring])
            combined_objectives = np.vstack([objectives, offspring_objectives])
            # Non-dominated sorting
            fronts = non_dominated_sort(combined_objectives)
            # Select next generation
            next_population = []
            next_objectives = []
            next_indices = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    # Add entire front
                    for idx in front:
                        next_population.append(combined_population[idx])
                        next_objectives.append(combined_objectives[idx])
                        next_indices.append(idx)
                else:
                    # Need to select from this front using crowding distance
                    remaining = self.population_size - len(next_population)
                    # Calculate crowding distances
                    distances = crowding_distance(combined_objectives, front)
                    # Sort by crowding distance (descending)
                    sorted_indices = np.argsort(distances)[::-1]
                    # Select individuals with highest crowding distance
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        next_population.append(combined_population[idx])
                        next_objectives.append(combined_objectives[idx])
                        next_indices.append(idx)
                    break
            population = np.array(next_population)
            objectives = np.array(next_objectives)
            if self.verbose and generation % 50 == 0:
                # Get Pareto front (first front)
                pareto_indices = fronts[0]
                n_pareto = len(pareto_indices)
                print(f"Generation {generation}: Pareto front size = {n_pareto}")
        # Extract final Pareto front
        final_fronts = non_dominated_sort(objectives)
        pareto_indices = final_fronts[0]
        pareto_set = population[pareto_indices]
        pareto_front = objectives[pareto_indices]
        # Calculate metrics
        hypervolume = self._calculate_hypervolume(pareto_front)
        spacing = self._calculate_spacing(pareto_front)
        execution_time = time.time() - start_time
        return MultiObjectiveResult(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            hypervolume=hypervolume,
            spacing=spacing,
            success=True,
            message="NSGA-II completed",
            nit=self.generations,
            nfev=problem.nfev,
            execution_time=execution_time,
            metadata={'final_population': population, 'final_objectives': objectives}
        )
    def _initialize_population(self, problem: MultiObjectiveProblem) -> np.ndarray:
        """Initialize random population within bounds."""
        population = np.zeros((self.population_size, problem.n_variables))
        for i in range(self.population_size):
            for j in range(problem.n_variables):
                low, high = problem.bounds[j]
                population[i, j] = np.random.uniform(low, high)
        return population
    def _create_offspring(self, population: np.ndarray, problem: MultiObjectiveProblem) -> np.ndarray:
        """Create offspring population through crossover and mutation."""
        n_offspring = self.population_size
        offspring = []
        while len(offspring) < n_offspring:
            # Tournament selection
            parent1 = self._tournament_selection(population, 2)
            parent2 = self._tournament_selection(population, 2)
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._sbx_crossover(parent1, parent2, problem.bounds)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._polynomial_mutation(child1, problem.bounds)
            if np.random.random() < self.mutation_rate:
                child2 = self._polynomial_mutation(child2, problem.bounds)
            offspring.extend([child1, child2])
        return np.array(offspring[:n_offspring])
    def _tournament_selection(self, population: np.ndarray, tournament_size: int) -> np.ndarray:
        """Binary tournament selection."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        return population[indices[0]]  # Simplified: random selection
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                      bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            if np.random.random() < 0.5:  # 50% chance for each variable
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    # Calculate beta
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.crossover_eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.crossover_eta + 1))
                    # Create children
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                    # Apply bounds
                    low, high = bounds[i]
                    child1[i] = np.clip(child1[i], low, high)
                    child2[i] = np.clip(child2[i], low, high)
        return child1, child2
    def _polynomial_mutation(self, individual: np.ndarray,
                           bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Polynomial mutation."""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                low, high = bounds[i]
                delta = high - low
                if delta > 0:
                    u = np.random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (self.mutation_eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (self.mutation_eta + 1))
                    mutated[i] = individual[i] + delta_q * delta
                    mutated[i] = np.clip(mutated[i], low, high)
        return mutated
    def _calculate_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified 2D version)."""
        if pareto_front.shape[1] != 2:
            return None  # Only implemented for 2D
        # Reference point (worst possible values)
        ref_point = np.max(pareto_front, axis=0) * 1.1
        # Sort by first objective
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        # Calculate hypervolume
        hypervolume = 0.0
        prev_x = 0.0
        for point in sorted_front:
            width = point[0] - prev_x
            height = ref_point[1] - point[1]
            hypervolume += width * height
            prev_x = point[0]
        # Add final rectangle
        if len(sorted_front) > 0:
            width = ref_point[0] - sorted_front[-1, 0]
            height = ref_point[1] - sorted_front[-1, 1]
            hypervolume += width * height
        return hypervolume
    def _calculate_spacing(self, pareto_front: np.ndarray) -> float:
        """Calculate spacing metric for diversity."""
        if len(pareto_front) < 2:
            return 0.0
        # Calculate minimum distance to other solutions for each solution
        min_distances = []
        for i in range(len(pareto_front)):
            distances = []
            for j in range(len(pareto_front)):
                if i != j:
                    distance = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    distances.append(distance)
            if distances:
                min_distances.append(min(distances))
        if not min_distances:
            return 0.0
        # Spacing metric: standard deviation of minimum distances
        mean_distance = np.mean(min_distances)
        spacing = np.sqrt(np.mean((np.array(min_distances) - mean_distance) ** 2))
        return spacing
class ParetoOptimization:
    """
    Basic Pareto optimization using various scalarization methods.
    Provides methods to find Pareto optimal solutions through
    different scalarization techniques.
    """
    def __init__(self, base_optimizer=None):
        """
        Initialize Pareto optimization.
        Args:
            base_optimizer: Single-objective optimizer to use
        """
        if base_optimizer is None:
            from .unconstrained import BFGS
            self.base_optimizer = BFGS()
        else:
            self.base_optimizer = base_optimizer
    def solve_weighted_sum(self, problem: MultiObjectiveProblem,
                          weights: Optional[List[np.ndarray]] = None,
                          n_solutions: int = 50) -> MultiObjectiveResult:
        """
        Solve using weighted sum method.
        Args:
            problem: Multi-objective problem
            weights: List of weight vectors (if None, generates uniform weights)
            n_solutions: Number of solutions to generate (if weights not provided)
        Returns:
            MultiObjectiveResult
        """
        start_time = time.time()
        # Generate weights if not provided
        if weights is None:
            weights = self._generate_uniform_weights(problem.n_objectives, n_solutions)
        pareto_set = []
        pareto_front = []
        for w in weights:
            # Create weighted objective
            def weighted_objective(x):
                obj_vals = problem.evaluate(x)
                return np.dot(w, obj_vals)
            # Random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
            # Optimize
            result = self.base_optimizer.minimize(weighted_objective, x0)
            if result.success:
                pareto_set.append(result.x)
                pareto_front.append(problem.evaluate(result.x))
        pareto_set = np.array(pareto_set)
        pareto_front = np.array(pareto_front)
        # Remove dominated solutions
        pareto_set, pareto_front = self._filter_dominated(pareto_set, pareto_front)
        execution_time = time.time() - start_time
        return MultiObjectiveResult(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            success=True,
            message="Weighted sum method completed",
            nfev=problem.nfev,
            execution_time=execution_time
        )
    def _generate_uniform_weights(self, n_objectives: int, n_solutions: int) -> List[np.ndarray]:
        """Generate uniformly distributed weight vectors."""
        if n_objectives == 2:
            # For 2 objectives, evenly space weights
            weights = []
            for i in range(n_solutions):
                w1 = i / (n_solutions - 1)
                w2 = 1 - w1
                weights.append(np.array([w1, w2]))
            return weights
        else:
            # For more objectives, use random weights
            weights = []
            for _ in range(n_solutions):
                w = np.random.random(n_objectives)
                w = w / np.sum(w)  # Normalize
                weights.append(w)
            return weights
    def _filter_dominated(self, solutions: np.ndarray, objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove dominated solutions."""
        n = len(solutions)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[j]:
                    if dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break
        non_dominated_indices = ~is_dominated
        return solutions[non_dominated_indices], objectives[non_dominated_indices]
class WeightedSum(ParetoOptimization):
    """Weighted sum method for multi-objective optimization."""
    pass
class EpsilonConstraint:
    """
    Epsilon-constraint method for multi-objective optimization.
    Optimizes one objective while constraining others to be
    below specified epsilon values.
    """
    def __init__(self, base_optimizer=None):
        """
        Initialize epsilon-constraint method.
        Args:
            base_optimizer: Constrained optimizer to use
        """
        if base_optimizer is None:
            from .constrained import AugmentedLagrangian
            self.base_optimizer = AugmentedLagrangian()
        else:
            self.base_optimizer = base_optimizer
    def solve(self, problem: MultiObjectiveProblem, primary_objective: int = 0,
              epsilon_values: Optional[List[np.ndarray]] = None,
              n_solutions: int = 50) -> MultiObjectiveResult:
        """
        Solve using epsilon-constraint method.
        Args:
            problem: Multi-objective problem
            primary_objective: Index of objective to optimize
            epsilon_values: List of epsilon constraint values
            n_solutions: Number of solutions to generate
        Returns:
            MultiObjectiveResult
        """
        start_time = time.time()
        # Generate epsilon values if not provided
        if epsilon_values is None:
            epsilon_values = self._generate_epsilon_values(problem, primary_objective, n_solutions)
        pareto_set = []
        pareto_front = []
        for epsilons in epsilon_values:
            # Create constrained problem
            def objective(x):
                return problem.objectives[primary_objective](x)
            # Add epsilon constraints
            from .constrained import Constraint
            constraints = list(problem.constraints)  # Copy existing constraints
            for i in range(problem.n_objectives):
                if i != primary_objective:
                    def make_constraint(idx, eps):
                        return lambda x: problem.objectives[idx](x) - eps
                    constraints.append(Constraint(
                        fun=make_constraint(i, epsilons[i]),
                        type='ineq'
                    ))
            # Random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
            # Optimize
            try:
                result = self.base_optimizer.minimize(objective, x0, constraints=constraints)
                if result.success:
                    pareto_set.append(result.x)
                    pareto_front.append(problem.evaluate(result.x))
            except:
                continue
        if pareto_set:
            pareto_set = np.array(pareto_set)
            pareto_front = np.array(pareto_front)
            # Remove dominated solutions
            pareto_set, pareto_front = self._filter_dominated(pareto_set, pareto_front)
        else:
            pareto_set = np.array([])
            pareto_front = np.array([])
        execution_time = time.time() - start_time
        return MultiObjectiveResult(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            success=len(pareto_set) > 0,
            message="Epsilon-constraint method completed",
            nfev=problem.nfev,
            execution_time=execution_time
        )
    def _generate_epsilon_values(self, problem: MultiObjectiveProblem,
                                primary_objective: int, n_solutions: int) -> List[np.ndarray]:
        """Generate epsilon constraint values."""
        # First, estimate the range of each objective
        # This is a simplified approach - in practice, you'd want to solve
        # individual objectives to find their ranges
        epsilon_values = []
        for i in range(n_solutions):
            epsilons = np.zeros(problem.n_objectives)
            # For non-primary objectives, set epsilon constraints
            for j in range(problem.n_objectives):
                if j != primary_objective:
                    # Simple linear spacing - in practice, you'd want better coverage
                    epsilons[j] = 100 * (1 - i / (n_solutions - 1))  # Placeholder values
            epsilon_values.append(epsilons)
        return epsilon_values
    def _filter_dominated(self, solutions: np.ndarray, objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove dominated solutions."""
        n = len(solutions)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[j]:
                    if dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break
        non_dominated_indices = ~is_dominated
        return solutions[non_dominated_indices], objectives[non_dominated_indices]
def demo():
    """Demonstrate multi-objective optimization algorithms."""
    print("Multi-Objective Optimization Demo")
    print("================================")
    print()
    # Define test problem: ZDT1
    def zdt1_f1(x):
        return x[0]
    def zdt1_f2(x):
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(x[0] / g))
    # Create problem
    n_vars = 30
    bounds = [(0, 1)] * n_vars
    problem = MultiObjectiveProblem(
        objectives=[zdt1_f1, zdt1_f2],
        n_variables=n_vars,
        bounds=bounds,
        objective_names=['f1', 'f2']
    )
    print(f"Test Problem: ZDT1")
    print(f"Variables: {n_vars}")
    print(f"Objectives: 2 (minimize both)")
    print(f"True Pareto front: f2 = 1 - sqrt(f1), f1 ∈ [0,1]")
    print("-" * 50)
    # Test NSGA-II
    print("\n1. NSGA-II")
    print("----------")
    nsga2 = NSGA2(population_size=100, generations=200, verbose=True)
    result_nsga2 = nsga2.solve(problem)
    print(f"Pareto front size: {len(result_nsga2.pareto_front)}")
    print(f"Hypervolume: {result_nsga2.hypervolume:.4f}" if result_nsga2.hypervolume else "Hypervolume: N/A")
    print(f"Spacing: {result_nsga2.spacing:.4f}" if result_nsga2.spacing else "Spacing: N/A")
    print(f"Function evaluations: {result_nsga2.nfev}")
    print()
    # Test Weighted Sum
    print("2. Weighted Sum Method")
    print("---------------------")
    weighted_sum = WeightedSum()
    result_ws = weighted_sum.solve_weighted_sum(problem, n_solutions=20)
    print(f"Pareto front size: {len(result_ws.pareto_front)}")
    print(f"Function evaluations: {result_ws.nfev}")
    print()
    # Simple 2-objective problem for visualization
    print("3. Simple 2D Problem")
    print("-------------------")
    def simple_f1(x):
        return x[0]**2 + x[1]**2
    def simple_f2(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2
    simple_problem = MultiObjectiveProblem(
        objectives=[simple_f1, simple_f2],
        n_variables=2,
        bounds=[(-2, 2), (-2, 2)],
        objective_names=['Distance from origin', 'Distance from (1,1)']
    )
    # Solve with NSGA-II
    nsga2_simple = NSGA2(population_size=50, generations=50)
    result_simple = nsga2_simple.solve(simple_problem)
    print(f"Simple problem Pareto front size: {len(result_simple.pareto_front)}")
    # Print some solutions
    if len(result_simple.pareto_front) > 0:
        print("\nSample Pareto optimal solutions:")
        n_show = min(5, len(result_simple.pareto_front))
        for i in range(n_show):
            x = result_simple.pareto_set[i]
            f = result_simple.pareto_front[i]
            print(f"  x = [{x[0]:.3f}, {x[1]:.3f}], f1 = {f[0]:.3f}, f2 = {f[1]:.3f}")
    print("\nDemo completed!")
if __name__ == "__main__":
    demo()