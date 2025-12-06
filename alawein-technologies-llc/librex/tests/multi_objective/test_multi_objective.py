"""
Comprehensive tests for multi-objective optimization module.

Tests cover:
- Core functionality (Pareto dominance, non-dominated sorting)
- NSGA-II algorithm
- NSGA-III algorithm
- MOEA/D algorithm
- Quality indicators
"""

import numpy as np
import pytest

from Librex.multi_objective import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    ParetoFront,
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    hypervolume,
    NSGA2Optimizer,
    NSGA3Optimizer,
    MOEADOptimizer,
)
from Librex.multi_objective.moead import DecompositionMethod
from Librex.multi_objective.indicators import (
    calculate_igd,
    calculate_gd,
    calculate_spread,
    calculate_epsilon_indicator,
    calculate_spacing,
    PerformanceMetrics,
)


class TestCore:
    """Test core multi-objective functionality."""

    def test_pareto_dominance(self):
        """Test Pareto dominance checking."""
        # Test basic dominance
        obj1 = np.array([1, 2])
        obj2 = np.array([2, 3])
        assert dominates(obj1, obj2)
        assert not dominates(obj2, obj1)

        # Test non-dominance
        obj3 = np.array([1, 3])
        obj4 = np.array([2, 2])
        assert not dominates(obj3, obj4)
        assert not dominates(obj4, obj3)

        # Test equality (no dominance)
        obj5 = np.array([1, 1])
        obj6 = np.array([1, 1])
        assert not dominates(obj5, obj6)

    def test_multi_objective_problem(self):
        """Test MultiObjectiveProblem class."""
        # Define test objectives
        def f1(x):
            return x[0] ** 2

        def f2(x):
            return (x[0] - 2) ** 2

        problem = MultiObjectiveProblem(
            objectives=[f1, f2],
            n_objectives=2,
            n_variables=1,
            bounds=(np.array([0]), np.array([5]))
        )

        # Test evaluation
        x = np.array([1.0])
        obj_values = problem.evaluate(x)
        assert len(obj_values) == 2
        assert obj_values[0] == 1.0  # f1(1) = 1
        assert obj_values[1] == 1.0  # f2(1) = (1-2)^2 = 1

    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        # Create test population
        solutions = [
            MultiObjectiveSolution(np.array([1]), np.array([1, 5])),
            MultiObjectiveSolution(np.array([2]), np.array([2, 4])),
            MultiObjectiveSolution(np.array([3]), np.array([3, 3])),
            MultiObjectiveSolution(np.array([4]), np.array([4, 2])),
            MultiObjectiveSolution(np.array([5]), np.array([5, 1])),
        ]

        fronts = fast_non_dominated_sort(solutions)

        # All solutions should be non-dominated (form Pareto front)
        assert len(fronts) == 1
        assert len(fronts[0]) == 5

        # Add dominated solution
        solutions.append(
            MultiObjectiveSolution(np.array([3.5]), np.array([3.5, 3.5]))
        )
        fronts = fast_non_dominated_sort(solutions)

        # Should have two fronts now
        assert len(fronts) == 2
        assert len(fronts[0]) == 5  # Original Pareto front
        assert len(fronts[1]) == 1  # Dominated solution

    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        solutions = [
            MultiObjectiveSolution(np.array([1]), np.array([1, 9])),
            MultiObjectiveSolution(np.array([2]), np.array([3, 7])),
            MultiObjectiveSolution(np.array([3]), np.array([5, 5])),
            MultiObjectiveSolution(np.array([4]), np.array([7, 3])),
            MultiObjectiveSolution(np.array([5]), np.array([9, 1])),
        ]

        indices = list(range(len(solutions)))
        crowding_distance(solutions, indices)

        # Boundary solutions should have infinite distance
        assert solutions[0].crowding_distance == float('inf')
        assert solutions[4].crowding_distance == float('inf')

        # Middle solutions should have finite positive distance
        for i in [1, 2, 3]:
            assert 0 < solutions[i].crowding_distance < float('inf')

    def test_pareto_front(self):
        """Test ParetoFront container."""
        front = ParetoFront()

        # Add non-dominated solutions
        sol1 = MultiObjectiveSolution(np.array([1]), np.array([1, 5]))
        sol2 = MultiObjectiveSolution(np.array([2]), np.array([5, 1]))
        sol3 = MultiObjectiveSolution(np.array([3]), np.array([3, 3]))

        assert front.add(sol1)
        assert front.add(sol2)
        assert front.add(sol3)
        assert front.size() == 3

        # Try to add dominated solution
        sol4 = MultiObjectiveSolution(np.array([4]), np.array([4, 4]))
        assert not front.add(sol4)
        assert front.size() == 3

        # Add solution that dominates existing one
        sol5 = MultiObjectiveSolution(np.array([5]), np.array([2, 2]))
        assert front.add(sol5)
        assert front.size() == 3  # sol3 should be removed

    def test_hypervolume_2d(self):
        """Test 2D hypervolume calculation."""
        points = np.array([
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1]
        ])
        ref_point = np.array([6, 6])

        hv = hypervolume(points, ref_point)
        assert hv > 0

        # Single point hypervolume
        single_point = np.array([[2, 3]])
        hv_single = hypervolume(single_point, ref_point)
        assert hv_single == (6 - 2) * (6 - 3)  # 4 * 3 = 12

    def test_hypervolume_3d(self):
        """Test 3D hypervolume calculation."""
        points = np.array([
            [1, 1, 1],
            [2, 2, 2]
        ])
        ref_point = np.array([3, 3, 3])

        hv = hypervolume(points, ref_point)
        assert hv > 0

        # Single point 3D hypervolume
        single_point = np.array([[1, 1, 1]])
        hv_single = hypervolume(single_point, ref_point)
        assert hv_single == 2 * 2 * 2  # 8


class TestNSGA2:
    """Test NSGA-II algorithm."""

    def create_test_problem(self):
        """Create ZDT1 test problem."""
        def f1(x):
            return x[0]

        def f2(x):
            g = 1 + 9 * np.mean(x[1:])
            return g * (1 - np.sqrt(x[0] / g))

        return MultiObjectiveProblem(
            objectives=[f1, f2],
            n_objectives=2,
            n_variables=5,
            bounds=(np.zeros(5), np.ones(5))
        )

    def test_nsga2_initialization(self):
        """Test NSGA-II initialization."""
        problem = self.create_test_problem()
        optimizer = NSGA2Optimizer(
            problem,
            population_size=20,
            n_generations=10
        )

        assert optimizer.population_size == 20
        assert optimizer.n_generations == 10

    def test_nsga2_optimization(self):
        """Test NSGA-II optimization process."""
        problem = self.create_test_problem()
        optimizer = NSGA2Optimizer(
            problem,
            population_size=20,
            n_generations=10,
            seed=42
        )

        pareto_front = optimizer.optimize()

        # Should find some non-dominated solutions
        assert pareto_front.size() > 0

        # Check that solutions are valid
        solutions = pareto_front.solutions
        for sol in solutions:
            assert sol.objectives.shape == (2,)
            assert sol.variables.shape == (5,)
            assert np.all(sol.variables >= 0)
            assert np.all(sol.variables <= 1)

    def test_nsga2_convergence(self):
        """Test NSGA-II convergence on simple problem."""
        # Simple bi-objective problem
        def f1(x):
            return x[0]

        def f2(x):
            return 1 - x[0]

        problem = MultiObjectiveProblem(
            objectives=[f1, f2],
            n_objectives=2,
            n_variables=1,
            bounds=(np.array([0]), np.array([1]))
        )

        optimizer = NSGA2Optimizer(
            problem,
            population_size=20,
            n_generations=20,
            seed=42
        )

        pareto_front = optimizer.optimize()

        # Should cover the Pareto front [0,1] x [0,1]
        objectives = pareto_front.get_objectives()
        assert np.min(objectives[:, 0]) < 0.2
        assert np.max(objectives[:, 0]) > 0.8


class TestNSGA3:
    """Test NSGA-III algorithm."""

    def create_dtlz2_problem(self, n_obj=3):
        """Create DTLZ2 test problem."""
        def dtlz2_obj(x, obj_idx):
            k = 5  # Number of position variables
            g = np.sum((x[n_obj-1:] - 0.5) ** 2)

            f = 1 + g
            for i in range(n_obj - obj_idx - 1):
                f *= np.cos(x[i] * np.pi / 2)
            if obj_idx > 0:
                f *= np.sin(x[n_obj - obj_idx - 1] * np.pi / 2)

            return f

        objectives = []
        for i in range(n_obj):
            objectives.append(lambda x, idx=i: dtlz2_obj(x, idx))

        return MultiObjectiveProblem(
            objectives=objectives,
            n_objectives=n_obj,
            n_variables=n_obj + 4,  # n_obj - 1 + k
            bounds=(np.zeros(n_obj + 4), np.ones(n_obj + 4))
        )

    def test_nsga3_initialization(self):
        """Test NSGA-III initialization."""
        problem = self.create_dtlz2_problem(3)
        optimizer = NSGA3Optimizer(
            problem,
            n_generations=10,
            n_partitions=4
        )

        # Check reference points generation
        assert len(optimizer.reference_points) > 0
        assert optimizer.reference_points.shape[1] == 3

    def test_nsga3_optimization(self):
        """Test NSGA-III optimization."""
        problem = self.create_dtlz2_problem(3)
        optimizer = NSGA3Optimizer(
            problem,
            population_size=30,
            n_generations=20,
            n_partitions=4,
            seed=42
        )

        pareto_front = optimizer.optimize()

        # Should find diverse solutions
        assert pareto_front.size() > 0

        # Check solution validity
        for sol in pareto_front.solutions:
            assert sol.objectives.shape == (3,)
            assert np.all(sol.variables >= 0)
            assert np.all(sol.variables <= 1)

    def test_nsga3_many_objective(self):
        """Test NSGA-III on many-objective problem."""
        problem = self.create_dtlz2_problem(5)
        optimizer = NSGA3Optimizer(
            problem,
            n_generations=10,
            n_partitions=3,
            seed=42
        )

        pareto_front = optimizer.optimize()

        # Should handle 5-objective problem
        assert pareto_front.size() > 0
        objectives = pareto_front.get_objectives()
        assert objectives.shape[1] == 5


class TestMOEAD:
    """Test MOEA/D algorithm."""

    def create_test_problem(self):
        """Create simple bi-objective test problem."""
        def f1(x):
            return x[0]

        def f2(x):
            return (1 + x[1]) * (1 - x[0] / (1 + x[1]))

        return MultiObjectiveProblem(
            objectives=[f1, f2],
            n_objectives=2,
            n_variables=2,
            bounds=(np.zeros(2), np.ones(2))
        )

    def test_moead_initialization(self):
        """Test MOEA/D initialization."""
        problem = self.create_test_problem()
        optimizer = MOEADOptimizer(
            problem,
            population_size=20,
            n_generations=10
        )

        # Check weight vectors
        assert optimizer.weight_vectors.shape == (20, 2)
        assert np.allclose(np.sum(optimizer.weight_vectors, axis=1), 1.0)

        # Check neighborhoods
        assert len(optimizer.neighborhoods) == 20

    def test_moead_decomposition_methods(self):
        """Test different decomposition methods."""
        problem = self.create_test_problem()

        for method in [
            DecompositionMethod.WEIGHTED_SUM,
            DecompositionMethod.TCHEBYCHEFF,
            DecompositionMethod.PBI
        ]:
            optimizer = MOEADOptimizer(
                problem,
                population_size=10,
                n_generations=5,
                decomposition=method,
                seed=42
            )

            pareto_front = optimizer.optimize()
            assert pareto_front.size() > 0

    def test_moead_optimization(self):
        """Test MOEA/D optimization."""
        problem = self.create_test_problem()
        optimizer = MOEADOptimizer(
            problem,
            population_size=20,
            n_generations=20,
            n_neighbors=5,
            seed=42
        )

        pareto_front = optimizer.optimize()

        # Should find diverse solutions
        assert pareto_front.size() > 0

        # Check solution validity
        for sol in pareto_front.solutions:
            assert sol.objectives.shape == (2,)
            assert np.all(sol.variables >= 0)
            assert np.all(sol.variables <= 1)


class TestIndicators:
    """Test quality indicators."""

    def create_test_fronts(self):
        """Create test Pareto fronts."""
        # True Pareto front (convex)
        true_front = []
        for i in range(11):
            x = i / 10.0
            true_front.append([x, 1 - x ** 2])
        true_front = np.array(true_front)

        # Approximate front (slightly worse)
        approx_front = []
        for i in range(9):
            x = i / 8.0
            approx_front.append([x + 0.05, 1 - x ** 2 + 0.05])
        approx_front = np.array(approx_front)

        return true_front, approx_front

    def test_igd(self):
        """Test IGD calculation."""
        true_front, approx_front = self.create_test_fronts()

        igd_value = calculate_igd(approx_front, true_front)
        assert igd_value > 0

        # Perfect match should have IGD close to 0
        perfect_igd = calculate_igd(true_front, true_front)
        assert perfect_igd < 1e-10

    def test_gd(self):
        """Test GD calculation."""
        true_front, approx_front = self.create_test_fronts()

        gd_value = calculate_gd(approx_front, true_front)
        assert gd_value > 0

        # Perfect match should have GD close to 0
        perfect_gd = calculate_gd(true_front, true_front)
        assert perfect_gd < 1e-10

    def test_spread(self):
        """Test Spread indicator."""
        true_front, approx_front = self.create_test_fronts()

        spread_value = calculate_spread(approx_front, true_front)
        assert spread_value >= 0

        # Perfectly distributed front should have low spread
        uniform_front = np.array([[i/10, 1-i/10] for i in range(11)])
        uniform_spread = calculate_spread(uniform_front)
        assert uniform_spread < 1.0

    def test_epsilon_indicator(self):
        """Test Epsilon indicator."""
        true_front, approx_front = self.create_test_fronts()

        eps_value = calculate_epsilon_indicator(approx_front, true_front)
        assert eps_value >= 1.0  # Multiplicative epsilon

        # Perfect match should have epsilon = 1
        perfect_eps = calculate_epsilon_indicator(true_front, true_front)
        assert abs(perfect_eps - 1.0) < 1e-10

    def test_spacing(self):
        """Test Spacing indicator."""
        front = np.array([[i/10, 1-i/10] for i in range(11)])

        spacing_value = calculate_spacing(front)
        assert spacing_value >= 0

        # Uniform distribution should have low spacing
        assert spacing_value < 0.1

    def test_performance_metrics(self):
        """Test comprehensive performance metrics."""
        true_front, approx_front = self.create_test_fronts()

        metrics_calc = PerformanceMetrics(
            reference_set=true_front,
            reference_point=np.array([2, 2])
        )

        metrics = metrics_calc.evaluate(approx_front)

        assert 'n_solutions' in metrics
        assert 'igd' in metrics
        assert 'gd' in metrics
        assert 'hypervolume' in metrics
        assert 'spacing' in metrics

        # All metrics should be valid
        for key, value in metrics.items():
            if key != 'n_solutions':
                assert value >= 0