"""Tests for Librex.Alloc solver"""
import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_alloc import Librex.AllocSolver

class TestLibrex.AllocSolver:
    @pytest.fixture
    def simple_allocation_problem(self):
        return OptimizationProblem(
            problem_type=ProblemType.ALLOC,
            data={
                "resource_demands": [("agent1", 10.0), ("agent2", 20.0), ("agent3", 15.0)],
                "budget_constraint": 100.0,
            },
        )

    def test_solve(self, simple_allocation_problem):
        solver = Librex.AllocSolver()
        solver.initialize()
        result = solver.solve(simple_allocation_problem)

        assert result.status == SolverStatus.SUCCESS
        assert "allocation" in result.solution
        assert len(result.solution["allocation"]) == 3
        assert sum(result.solution["allocation"]) <= 100.0  # Budget constraint

    def test_thompson_sampling_exploration(self, simple_allocation_problem):
        solver = Librex.AllocSolver(config={"thompson_sampling_horizon": 50})
        solver.initialize()
        result = solver.solve(simple_allocation_problem)

        assert result.iterations > 0
        assert result.improvement_over_baseline is not None
