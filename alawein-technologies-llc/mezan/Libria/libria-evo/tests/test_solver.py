"""Tests for Librex.Evo solver"""
import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_evo import Librex.EvoSolver

class TestLibrex.EvoSolver:
    @pytest.fixture
    def multi_objective_problem(self):
        def obj1(x):
            return np.sum(x ** 2)

        def obj2(x):
            return np.sum((x - 1) ** 2)

        return OptimizationProblem(
            problem_type=ProblemType.EVO,
            data={
                "objective_functions": [obj1, obj2],
                "num_variables": 5,
                "variable_bounds": (0.0, 2.0),
            },
        )

    def test_solve(self, multi_objective_problem):
        solver = Librex.EvoSolver(config={"population_size": 20, "num_generations": 30})
        solver.initialize()
        result = solver.solve(multi_objective_problem)

        assert result.status == SolverStatus.SUCCESS
        assert "pareto_front" in result.solution
        assert "pareto_solutions" in result.solution
        assert len(result.solution["pareto_front"]) > 0

    def test_pareto_front_quality(self, multi_objective_problem):
        solver = Librex.EvoSolver(config={"population_size": 30, "num_generations": 50})
        solver.initialize()
        result = solver.solve(multi_objective_problem)

        pareto_front = result.solution["pareto_front"]
        # Pareto front should have multiple trade-off solutions
        assert len(pareto_front) >= 3

    def test_nsga2_components(self):
        solver = Librex.EvoSolver()
        solver.initialize()

        # Test non-dominated sorting
        objectives = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [0.5, 0.5]])
        fronts = solver._fast_non_dominated_sort(objectives)

        assert len(fronts) > 0
        assert 3 in fronts[0]  # [0.5, 0.5] dominates all others
