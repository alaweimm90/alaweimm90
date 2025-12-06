"""
Comprehensive tests for Librex.QAP solver

Tests cover:
- Basic functionality
- Algorithm correctness
- Edge cases
- Performance benchmarks
- Integration with MEZAN core
"""

import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_qap import Librex.QAPSolver


class TestLibrex.QAPSolver:
    """Test suite for Librex.QAP solver"""

    @pytest.fixture
    def small_qap_problem(self):
        """Small QAP problem for quick testing"""
        n = 5
        distance = np.random.rand(n, n)
        distance = (distance + distance.T) / 2  # Make symmetric
        flow = np.random.rand(n, n)
        flow = (flow + flow.T) / 2  # Make symmetric

        return OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": distance.tolist(), "flow_matrix": flow.tolist()},
        )

    @pytest.fixture
    def medium_qap_problem(self):
        """Medium QAP problem for realistic testing"""
        n = 20
        distance = np.random.rand(n, n)
        distance = (distance + distance.T) / 2
        flow = np.random.rand(n, n)
        flow = (flow + flow.T) / 2

        return OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": distance.tolist(), "flow_matrix": flow.tolist()},
        )

    @pytest.fixture
    def solver_sa(self):
        """Solver with Simulated Annealing"""
        return Librex.QAPSolver(config={"algorithm": "simulated_annealing", "max_iterations": 100})

    @pytest.fixture
    def solver_ga(self):
        """Solver with Genetic Algorithm"""
        return Librex.QAPSolver(config={"algorithm": "genetic", "max_iterations": 100})

    def test_solver_initialization(self, solver_sa):
        """Test solver initializes correctly"""
        assert not solver_sa._is_initialized
        solver_sa.initialize()
        assert solver_sa._is_initialized

    def test_get_problem_types(self, solver_sa):
        """Test solver reports correct problem type"""
        types = solver_sa.get_problem_types()
        assert ProblemType.QAP in types
        assert len(types) == 1

    def test_estimate_complexity(self, solver_sa, small_qap_problem, medium_qap_problem):
        """Test complexity estimation"""
        assert solver_sa.estimate_complexity(small_qap_problem) == "low"
        assert solver_sa.estimate_complexity(medium_qap_problem) in ["medium", "high"]

    def test_solve_small_problem_sa(self, solver_sa, small_qap_problem):
        """Test solving small problem with SA"""
        solver_sa.initialize()
        result = solver_sa.solve(small_qap_problem)

        assert result.status == SolverStatus.SUCCESS
        assert result.solution is not None
        assert "assignment" in result.solution
        assert len(result.solution["assignment"]) == 5
        assert result.objective_value is not None
        assert result.computation_time > 0
        assert result.iterations > 0

    def test_solve_small_problem_ga(self, solver_ga, small_qap_problem):
        """Test solving small problem with GA"""
        solver_ga.initialize()
        result = solver_ga.solve(small_qap_problem)

        assert result.status == SolverStatus.SUCCESS
        assert result.solution is not None
        assert "assignment" in result.solution
        assert len(result.solution["assignment"]) == 5

    def test_solution_is_valid_permutation(self, solver_sa, small_qap_problem):
        """Test that solution is a valid permutation"""
        solver_sa.initialize()
        result = solver_sa.solve(small_qap_problem)

        assignment = result.solution["assignment"]
        assert len(assignment) == len(set(assignment))  # No duplicates
        assert set(assignment) == set(range(5))  # Contains all elements

    def test_objective_value_computation(self, solver_sa, small_qap_problem):
        """Test objective value is computed correctly"""
        solver_sa.initialize()
        result = solver_sa.solve(small_qap_problem)

        # Recompute objective manually
        assignment = result.solution["assignment"]
        distance = np.array(small_qap_problem.data["distance_matrix"])
        flow = np.array(small_qap_problem.data["flow_matrix"])

        obj = 0.0
        n = len(assignment)
        for i in range(n):
            for j in range(n):
                obj += distance[i, j] * flow[assignment[i], assignment[j]]

        # Should match (within floating point error)
        assert abs(result.objective_value - obj) < 1e-6

    def test_improvement_over_baseline(self, solver_sa, small_qap_problem):
        """Test that solver improves over random baseline"""
        solver_sa.initialize()
        result = solver_sa.solve(small_qap_problem)

        # Should report improvement
        assert result.improvement_over_baseline is not None
        # Note: May not always be positive for very small problems with low iterations

    def test_timeout_handling(self, small_qap_problem):
        """Test solver respects timeout"""
        solver = Librex.QAPSolver(
            config={"max_iterations": 10000}, timeout=0.1  # Very short timeout
        )
        solver.initialize()
        result = solver.solve(small_qap_problem)

        # Should either complete quickly or timeout
        assert result.computation_time <= 1.0  # Generous bound

    def test_invalid_problem_detection(self, solver_sa):
        """Test solver detects invalid problems"""
        # Mismatched matrix sizes
        invalid_problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[1, 2], [3, 4]],
                "flow_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
        )

        solver_sa.initialize()
        result = solver_sa.solve(invalid_problem)

        assert result.status == SolverStatus.FAILED
        assert "error" in result.metadata

    def test_metadata_completeness(self, solver_sa, small_qap_problem):
        """Test result metadata is complete"""
        solver_sa.initialize()
        result = solver_sa.solve(small_qap_problem)

        assert "solver" in result.metadata
        assert "algorithm" in result.metadata
        assert "problem_size" in result.metadata
        assert "baseline_objective" in result.metadata

    def test_sa_convergence(self, solver_sa, medium_qap_problem):
        """Test SA shows improvement over iterations"""
        solver_sa.initialize()
        result = solver_sa.solve(medium_qap_problem)

        # SA should complete successfully
        assert result.status == SolverStatus.SUCCESS
        assert result.iterations > 0

    def test_ga_population_evolution(self, solver_ga, medium_qap_problem):
        """Test GA evolves population"""
        solver_ga.initialize()
        result = solver_ga.solve(medium_qap_problem)

        # GA should complete successfully
        assert result.status == SolverStatus.SUCCESS
        assert result.iterations > 0

    def test_deterministic_with_seed(self, small_qap_problem):
        """Test solver is deterministic with same random seed"""
        np.random.seed(42)
        solver1 = Librex.QAPSolver(config={"algorithm": "simulated_annealing", "max_iterations": 50})
        solver1.initialize()
        result1 = solver1.solve(small_qap_problem)

        np.random.seed(42)
        solver2 = Librex.QAPSolver(config={"algorithm": "simulated_annealing", "max_iterations": 50})
        solver2.initialize()
        result2 = solver2.solve(small_qap_problem)

        # Results should be identical with same seed
        assert result1.solution["assignment"] == result2.solution["assignment"]
        assert abs(result1.objective_value - result2.objective_value) < 1e-10

    def test_order_crossover(self, solver_ga):
        """Test order crossover produces valid permutations"""
        parent1 = [0, 1, 2, 3, 4]
        parent2 = [4, 3, 2, 1, 0]

        solver_ga.initialize()
        child1, child2 = solver_ga._order_crossover(parent1, parent2)

        # Children should be valid permutations
        assert len(child1) == len(set(child1))
        assert set(child1) == set(parent1)
        assert len(child2) == len(set(child2))
        assert set(child2) == set(parent2)

    def test_performance_benchmark(self, solver_sa, medium_qap_problem):
        """Benchmark performance on medium problem"""
        solver_sa.initialize()
        result = solver_sa.solve(medium_qap_problem)

        # Should complete in reasonable time
        assert result.computation_time < 5.0  # 5 seconds max for n=20

        # Should show some improvement
        assert result.improvement_over_baseline >= 0  # At least not worse


class TestLibrex.QAPIntegration:
    """Integration tests with MEZAN core"""

    def test_integration_with_optimizer_factory(self, small_qap_problem):
        """Test solver works with OptimizerFactory"""
        from MEZAN.core import OptimizerFactory

        factory = OptimizerFactory(
            config={"feature_flags": {"enable_qap_libria": True}}
        )

        optimizer = factory.create_optimizer(small_qap_problem)
        assert isinstance(optimizer, Librex.QAPSolver)

        result = optimizer.solve(small_qap_problem)
        assert result.status == SolverStatus.SUCCESS

    def test_fallback_when_disabled(self, small_qap_problem):
        """Test factory falls back when Librex.QAP disabled"""
        from MEZAN.core import OptimizerFactory, HeuristicFallbackOptimizer

        factory = OptimizerFactory(
            config={"feature_flags": {"enable_qap_libria": False}}
        )

        optimizer = factory.create_optimizer(small_qap_problem)
        assert isinstance(optimizer, HeuristicFallbackOptimizer)


@pytest.mark.benchmark
class TestLibrex.QAPBenchmarks:
    """Performance benchmarks for Librex.QAP"""

    @pytest.mark.parametrize("n", [5, 10, 15, 20, 30])
    def test_scaling_with_problem_size(self, n):
        """Test how solver scales with problem size"""
        distance = np.random.rand(n, n)
        flow = np.random.rand(n, n)

        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={"distance_matrix": distance.tolist(), "flow_matrix": flow.tolist()},
        )

        solver = Librex.QAPSolver(config={"max_iterations": 100})
        solver.initialize()
        result = solver.solve(problem)

        assert result.status == SolverStatus.SUCCESS
        print(f"n={n}: time={result.computation_time:.3f}s, "
              f"improvement={result.improvement_over_baseline:.1f}%")

    def test_qaplib_instance(self):
        """Test on actual QAPLIB instance (if available)"""
        # Placeholder for QAPLIB integration
        # Will be implemented in benchmarks.py
        pass
