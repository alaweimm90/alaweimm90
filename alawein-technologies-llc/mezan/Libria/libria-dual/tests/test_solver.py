"""Tests for Librex.Dual solver"""
import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_dual import Librex.DualSolver

class TestLibrex.DualSolver:
    @pytest.fixture
    def robust_optimization_problem(self):
        def objective_fn(x, y):
            return np.sum((x - y) ** 2)

        return OptimizationProblem(
            problem_type=ProblemType.DUAL,
            data={
                "objective_function": objective_fn,
                "initial_config": [1.0, 2.0, 3.0],
                "constraints": {},
                "problem_dimension": 3,
            },
        )

    def test_solve(self, robust_optimization_problem):
        solver = Librex.DualSolver(config={"max_iterations": 20})
        solver.initialize()
        result = solver.solve(robust_optimization_problem)

        assert result.status == SolverStatus.SUCCESS
        assert "robust_configuration" in result.solution
        assert "worst_case_value" in result.solution

    def test_alternating_optimization(self, robust_optimization_problem):
        solver = Librex.DualSolver(config={"max_iterations": 30})
        solver.initialize()
        result = solver.solve(robust_optimization_problem)

        assert result.iterations <= 30
        assert result.iterations > 0
