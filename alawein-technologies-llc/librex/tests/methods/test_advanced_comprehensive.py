"""
Comprehensive tests for Advanced Optimization Methods.
Tests ACO, PSO, VNS, ILS, and GRASP algorithms.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import time
from typing import List, Tuple, Dict, Any

from Librex.methods.advanced import (
    AntColonyOptimizer,
    ParticleSwarmOptimizer,
    VariableNeighborhoodSearch,
    IteratedLocalSearch,
    GRASPOptimizer
)
from Librex.core.interfaces import OptimizationProblem, Solution


class TestProblem(OptimizationProblem):
    """Test problem for optimization methods."""

    def __init__(self, size: int = 10):
        self.size = size
        self.flow = np.random.rand(size, size)
        self.distance = np.random.rand(size, size)
        self.evaluation_count = 0

    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate a solution."""
        self.evaluation_count += 1
        cost = 0
        for i in range(self.size):
            for j in range(self.size):
                cost += self.flow[i, j] * self.distance[solution[i], solution[j]]
        return cost

    def get_neighbors(self, solution: np.ndarray) -> List[np.ndarray]:
        """Get neighboring solutions."""
        neighbors = []
        for i in range(self.size):
            for j in range(i + 1, self.size):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors


class TestAntColonyOptimizer:
    """Test Ant Colony Optimization algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create an ACO optimizer."""
        return AntColonyOptimizer(
            n_ants=10,
            n_iterations=50,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.1,
            q=100
        )

    @pytest.fixture
    def problem(self):
        """Create a test problem."""
        return TestProblem(size=10)

    def test_initialization(self, optimizer):
        """Test ACO initialization."""
        assert optimizer.n_ants == 10
        assert optimizer.n_iterations == 50
        assert optimizer.alpha == 1.0
        assert optimizer.beta == 2.0
        assert optimizer.evaporation_rate == 0.1
        assert optimizer.q == 100

    def test_pheromone_initialization(self, optimizer, problem):
        """Test pheromone matrix initialization."""
        optimizer.initialize_pheromones(problem)
        assert optimizer.pheromones.shape == (problem.size, problem.size)
        assert np.all(optimizer.pheromones > 0)
        assert np.allclose(optimizer.pheromones, optimizer.pheromones[0, 0])

    def test_ant_solution_construction(self, optimizer, problem):
        """Test single ant solution construction."""
        optimizer.initialize_pheromones(problem)
        solution = optimizer.construct_ant_solution(problem)

        assert len(solution) == problem.size
        assert set(solution) == set(range(problem.size))

    def test_pheromone_update(self, optimizer, problem):
        """Test pheromone update mechanism."""
        optimizer.initialize_pheromones(problem)
        initial_pheromones = optimizer.pheromones.copy()

        # Create mock ant solutions
        solutions = [np.random.permutation(problem.size) for _ in range(optimizer.n_ants)]
        costs = [problem.evaluate(sol) for sol in solutions]

        optimizer.update_pheromones(solutions, costs)

        # Check that pheromones changed
        assert not np.allclose(optimizer.pheromones, initial_pheromones)

        # Check evaporation
        assert np.all(optimizer.pheromones <= initial_pheromones.max() * (1 + optimizer.q))

    def test_optimization_process(self, optimizer, problem):
        """Test full ACO optimization process."""
        best_solution, best_cost = optimizer.optimize(problem)

        assert len(best_solution) == problem.size
        assert set(best_solution) == set(range(problem.size))
        assert best_cost > 0
        assert problem.evaluation_count > 0

    def test_convergence(self, optimizer, problem):
        """Test ACO convergence behavior."""
        # Run optimization with history tracking
        optimizer.track_history = True
        best_solution, best_cost = optimizer.optimize(problem)

        history = optimizer.get_history()
        assert len(history) == optimizer.n_iterations

        # Check for improvement over iterations
        costs = [h['best_cost'] for h in history]
        assert costs[-1] <= costs[0]  # Should improve or stay same

    def test_parameter_sensitivity(self, problem):
        """Test sensitivity to ACO parameters."""
        results = {}

        for alpha in [0.5, 1.0, 2.0]:
            opt = AntColonyOptimizer(n_ants=5, n_iterations=20, alpha=alpha)
            _, cost = opt.optimize(problem)
            results[f'alpha_{alpha}'] = cost

        for beta in [1.0, 2.0, 3.0]:
            opt = AntColonyOptimizer(n_ants=5, n_iterations=20, beta=beta)
            _, cost = opt.optimize(problem)
            results[f'beta_{beta}'] = cost

        # Different parameters should give different results
        assert len(set(results.values())) > 1

    def test_elite_ant_strategy(self, optimizer, problem):
        """Test elite ant strategy."""
        optimizer.use_elite_ants = True
        optimizer.n_elite = 2

        best_solution, best_cost = optimizer.optimize(problem)

        # Elite strategy should be applied
        assert best_solution is not None
        assert best_cost > 0


class TestParticleSwarmOptimizer:
    """Test Particle Swarm Optimization algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create a PSO optimizer."""
        return ParticleSwarmOptimizer(
            n_particles=20,
            n_iterations=100,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

    @pytest.fixture
    def continuous_problem(self):
        """Create a continuous optimization problem."""
        def sphere_function(x):
            return np.sum(x ** 2)

        return {
            'objective': sphere_function,
            'bounds': [(-5, 5)] * 10,
            'dimension': 10
        }

    def test_initialization(self, optimizer):
        """Test PSO initialization."""
        assert optimizer.n_particles == 20
        assert optimizer.n_iterations == 100
        assert optimizer.w == 0.7
        assert optimizer.c1 == 1.5
        assert optimizer.c2 == 1.5

    def test_particle_initialization(self, optimizer, continuous_problem):
        """Test particle swarm initialization."""
        optimizer.initialize_swarm(continuous_problem)

        assert optimizer.positions.shape == (20, 10)
        assert optimizer.velocities.shape == (20, 10)
        assert optimizer.personal_best_positions.shape == (20, 10)
        assert len(optimizer.personal_best_scores) == 20

    def test_velocity_update(self, optimizer, continuous_problem):
        """Test particle velocity update."""
        optimizer.initialize_swarm(continuous_problem)
        initial_velocities = optimizer.velocities.copy()

        optimizer.update_velocities()

        # Velocities should change
        assert not np.allclose(optimizer.velocities, initial_velocities)

        # Check velocity limits
        v_max = 0.2 * (continuous_problem['bounds'][0][1] - continuous_problem['bounds'][0][0])
        assert np.all(np.abs(optimizer.velocities) <= v_max)

    def test_position_update(self, optimizer, continuous_problem):
        """Test particle position update."""
        optimizer.initialize_swarm(continuous_problem)
        initial_positions = optimizer.positions.copy()

        optimizer.update_positions()

        # Positions should change
        assert not np.allclose(optimizer.positions, initial_positions)

        # Check bounds
        for i, (low, high) in enumerate(continuous_problem['bounds']):
            assert np.all(optimizer.positions[:, i] >= low)
            assert np.all(optimizer.positions[:, i] <= high)

    def test_optimization_continuous(self, optimizer, continuous_problem):
        """Test PSO on continuous optimization."""
        best_position, best_score = optimizer.optimize(continuous_problem)

        assert len(best_position) == 10
        assert best_score >= 0
        # For sphere function, optimum is at origin
        assert best_score < 10  # Should find reasonable solution

    def test_discrete_pso(self):
        """Test PSO for discrete optimization."""
        optimizer = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=50,
            discrete=True
        )

        problem = TestProblem(size=8)
        best_solution, best_cost = optimizer.optimize(problem)

        assert len(best_solution) == 8
        assert set(best_solution) == set(range(8))

    def test_adaptive_parameters(self, continuous_problem):
        """Test PSO with adaptive parameters."""
        optimizer = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=50,
            adaptive=True
        )

        best_position, best_score = optimizer.optimize(continuous_problem)

        # Check that parameters adapted
        assert optimizer.w != 0.7  # Should have changed
        assert best_score < 10

    def test_topology_variations(self, continuous_problem):
        """Test different swarm topologies."""
        results = {}

        for topology in ['global', 'ring', 'random']:
            optimizer = ParticleSwarmOptimizer(
                n_particles=10,
                n_iterations=30,
                topology=topology
            )
            _, score = optimizer.optimize(continuous_problem)
            results[topology] = score

        # Different topologies may give different results
        assert len(results) == 3


class TestVariableNeighborhoodSearch:
    """Test Variable Neighborhood Search algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create a VNS optimizer."""
        return VariableNeighborhoodSearch(
            k_max=5,
            n_iterations=100,
            local_search_iterations=10
        )

    @pytest.fixture
    def problem(self):
        """Create a test problem."""
        return TestProblem(size=10)

    def test_initialization(self, optimizer):
        """Test VNS initialization."""
        assert optimizer.k_max == 5
        assert optimizer.n_iterations == 100
        assert optimizer.local_search_iterations == 10

    def test_neighborhood_structures(self, optimizer, problem):
        """Test neighborhood structure generation."""
        solution = np.arange(problem.size)
        neighborhoods = optimizer.get_neighborhood_structures(problem)

        assert len(neighborhoods) >= optimizer.k_max

        for k, neighborhood_func in enumerate(neighborhoods):
            neighbor = neighborhood_func(solution)
            assert len(neighbor) == problem.size
            assert set(neighbor) == set(range(problem.size))

    def test_shaking_procedure(self, optimizer, problem):
        """Test shaking procedure."""
        solution = np.arange(problem.size)

        for k in range(1, optimizer.k_max + 1):
            shaken = optimizer.shake(solution, k)
            assert len(shaken) == problem.size
            assert set(shaken) == set(range(problem.size))

            # Higher k should lead to more changes
            changes = np.sum(shaken != solution)
            assert changes >= k

    def test_local_search(self, optimizer, problem):
        """Test local search procedure."""
        initial_solution = np.random.permutation(problem.size)
        initial_cost = problem.evaluate(initial_solution)

        improved_solution, improved_cost = optimizer.local_search(
            initial_solution,
            problem
        )

        assert len(improved_solution) == problem.size
        assert improved_cost <= initial_cost  # Should not get worse

    def test_optimization_process(self, optimizer, problem):
        """Test full VNS optimization."""
        best_solution, best_cost = optimizer.optimize(problem)

        assert len(best_solution) == problem.size
        assert set(best_solution) == set(range(problem.size))
        assert best_cost > 0

    def test_vns_variants(self, problem):
        """Test VNS variants."""
        results = {}

        # Basic VNS
        basic_vns = VariableNeighborhoodSearch(k_max=3, n_iterations=50)
        _, cost = basic_vns.optimize(problem)
        results['basic'] = cost

        # Reduced VNS
        reduced_vns = VariableNeighborhoodSearch(
            k_max=3,
            n_iterations=50,
            variant='reduced'
        )
        _, cost = reduced_vns.optimize(problem)
        results['reduced'] = cost

        # Variable Neighborhood Descent
        vnd = VariableNeighborhoodSearch(
            k_max=3,
            n_iterations=50,
            variant='vnd'
        )
        _, cost = vnd.optimize(problem)
        results['vnd'] = cost

        assert len(results) == 3

    def test_adaptive_neighborhood_selection(self, optimizer, problem):
        """Test adaptive neighborhood selection."""
        optimizer.adaptive = True
        optimizer.neighborhood_weights = np.ones(optimizer.k_max)

        best_solution, best_cost = optimizer.optimize(problem)

        # Weights should have been updated
        assert not np.allclose(optimizer.neighborhood_weights, 1.0)
        assert best_solution is not None


class TestIteratedLocalSearch:
    """Test Iterated Local Search algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create an ILS optimizer."""
        return IteratedLocalSearch(
            n_iterations=100,
            perturbation_strength=3,
            local_search_iterations=20
        )

    @pytest.fixture
    def problem(self):
        """Create a test problem."""
        return TestProblem(size=10)

    def test_initialization(self, optimizer):
        """Test ILS initialization."""
        assert optimizer.n_iterations == 100
        assert optimizer.perturbation_strength == 3
        assert optimizer.local_search_iterations == 20

    def test_perturbation(self, optimizer, problem):
        """Test perturbation mechanism."""
        solution = np.arange(problem.size)

        perturbed = optimizer.perturb(solution)
        assert len(perturbed) == problem.size
        assert set(perturbed) == set(range(problem.size))

        # Should make specified number of changes
        changes = np.sum(perturbed != solution)
        assert changes >= optimizer.perturbation_strength * 2  # Each swap changes 2 positions

    def test_acceptance_criterion(self, optimizer):
        """Test acceptance criterion."""
        # Better solution should always be accepted
        assert optimizer.accept(100, 90, temperature=1.0) is True

        # Worse solution acceptance depends on temperature
        prob = optimizer.accept(90, 100, temperature=1.0)
        assert isinstance(prob, bool)

        # At zero temperature, only improvements accepted
        assert optimizer.accept(90, 100, temperature=0.0) is False

    def test_optimization_with_history(self, optimizer, problem):
        """Test ILS optimization with history tracking."""
        optimizer.track_history = True
        best_solution, best_cost = optimizer.optimize(problem)

        history = optimizer.get_history()
        assert len(history) <= optimizer.n_iterations

        # Check for exploration
        unique_costs = set(h['cost'] for h in history)
        assert len(unique_costs) > 1  # Should explore different solutions

    def test_adaptive_perturbation(self, problem):
        """Test ILS with adaptive perturbation strength."""
        optimizer = IteratedLocalSearch(
            n_iterations=50,
            adaptive_perturbation=True
        )

        best_solution, best_cost = optimizer.optimize(problem)

        # Perturbation strength should have adapted
        assert optimizer.perturbation_strength != 3
        assert best_solution is not None

    def test_restart_strategy(self, optimizer, problem):
        """Test ILS with restart strategy."""
        optimizer.use_restarts = True
        optimizer.restart_threshold = 20

        best_solution, best_cost = optimizer.optimize(problem)

        # Should have performed restarts
        assert optimizer.restart_count > 0
        assert best_solution is not None


class TestGRASPOptimizer:
    """Test GRASP (Greedy Randomized Adaptive Search Procedure) algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create a GRASP optimizer."""
        return GRASPOptimizer(
            n_iterations=50,
            alpha=0.3,
            local_search_iterations=10
        )

    @pytest.fixture
    def problem(self):
        """Create a test problem."""
        return TestProblem(size=10)

    def test_initialization(self, optimizer):
        """Test GRASP initialization."""
        assert optimizer.n_iterations == 50
        assert optimizer.alpha == 0.3
        assert optimizer.local_search_iterations == 10

    def test_greedy_randomized_construction(self, optimizer, problem):
        """Test greedy randomized construction phase."""
        solution = optimizer.construct_solution(problem)

        assert len(solution) == problem.size
        assert set(solution) == set(range(problem.size))

    def test_restricted_candidate_list(self, optimizer, problem):
        """Test RCL (Restricted Candidate List) construction."""
        partial_solution = [0, 1, 2]
        candidates = list(range(3, problem.size))

        rcl = optimizer.build_rcl(candidates, partial_solution, problem)

        assert len(rcl) > 0
        assert len(rcl) <= len(candidates)
        assert all(c in candidates for c in rcl)

    def test_construction_randomness(self, optimizer, problem):
        """Test randomness in construction phase."""
        solutions = [optimizer.construct_solution(problem) for _ in range(10)]

        # Should generate different solutions
        unique_solutions = set(tuple(sol) for sol in solutions)
        assert len(unique_solutions) > 1

    def test_optimization_process(self, optimizer, problem):
        """Test full GRASP optimization."""
        best_solution, best_cost = optimizer.optimize(problem)

        assert len(best_solution) == problem.size
        assert set(best_solution) == set(range(problem.size))
        assert best_cost > 0

    def test_alpha_sensitivity(self, problem):
        """Test sensitivity to alpha parameter."""
        results = {}

        for alpha in [0.0, 0.3, 0.7, 1.0]:
            optimizer = GRASPOptimizer(
                n_iterations=30,
                alpha=alpha
            )
            _, cost = optimizer.optimize(problem)
            results[alpha] = cost

        # alpha=0 is pure greedy, alpha=1 is pure random
        assert results[0.0] != results[1.0]

    def test_path_relinking(self, optimizer, problem):
        """Test GRASP with path relinking."""
        optimizer.use_path_relinking = True
        optimizer.elite_size = 5

        best_solution, best_cost = optimizer.optimize(problem)

        # Should maintain elite set
        assert len(optimizer.elite_set) <= optimizer.elite_size
        assert best_solution is not None

    def test_reactive_grasp(self, problem):
        """Test Reactive GRASP (adaptive alpha)."""
        optimizer = GRASPOptimizer(
            n_iterations=50,
            reactive=True
        )

        best_solution, best_cost = optimizer.optimize(problem)

        # Alpha values should have been adapted
        assert hasattr(optimizer, 'alpha_probs')
        assert len(optimizer.alpha_probs) > 1
        assert best_solution is not None


class TestMethodComparison:
    """Compare performance of different advanced methods."""

    @pytest.fixture
    def problems(self):
        """Create a set of test problems."""
        return [
            TestProblem(size=10),
            TestProblem(size=15),
            TestProblem(size=20)
        ]

    def test_comparative_performance(self, problems):
        """Compare all advanced methods."""
        methods = {
            'ACO': AntColonyOptimizer(n_ants=5, n_iterations=20),
            'PSO': ParticleSwarmOptimizer(n_particles=10, n_iterations=20, discrete=True),
            'VNS': VariableNeighborhoodSearch(k_max=3, n_iterations=20),
            'ILS': IteratedLocalSearch(n_iterations=20),
            'GRASP': GRASPOptimizer(n_iterations=20)
        }

        results = {name: [] for name in methods}

        for problem in problems:
            for name, optimizer in methods.items():
                _, cost = optimizer.optimize(problem)
                results[name].append(cost)

        # All methods should produce valid results
        for name, costs in results.items():
            assert len(costs) == len(problems)
            assert all(c > 0 for c in costs)

    def test_scalability(self):
        """Test scalability of advanced methods."""
        sizes = [5, 10, 15, 20]
        times = {}

        optimizer = AntColonyOptimizer(n_ants=5, n_iterations=10)

        for size in sizes:
            problem = TestProblem(size=size)
            start_time = time.time()
            optimizer.optimize(problem)
            elapsed = time.time() - start_time
            times[size] = elapsed

        # Time should increase with problem size
        assert times[20] > times[5]

    def test_solution_quality_metrics(self, problems):
        """Test solution quality metrics."""
        optimizer = GRASPOptimizer(n_iterations=30)

        for problem in problems:
            # Run multiple times to get statistics
            costs = []
            for _ in range(5):
                _, cost = optimizer.optimize(problem)
                costs.append(cost)

            # Calculate metrics
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            best_cost = np.min(costs)
            worst_cost = np.max(costs)

            assert mean_cost > 0
            assert std_cost >= 0
            assert best_cost <= mean_cost <= worst_cost