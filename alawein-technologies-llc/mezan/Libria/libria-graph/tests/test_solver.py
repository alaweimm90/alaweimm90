"""Tests for Librex.Graph solver"""
import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_graph import Librex.GraphSolver

class TestLibrex.GraphSolver:
    @pytest.fixture
    def network_problem(self):
        return OptimizationProblem(
            problem_type=ProblemType.GRAPH,
            data={
                "nodes": ["a1", "a2", "a3", "a4"],
                "communication_matrix": [
                    [0, 10, 5, 2],
                    [10, 0, 8, 3],
                    [5, 8, 0, 6],
                    [2, 3, 6, 0],
                ],
            },
        )

    def test_solve(self, network_problem):
        solver = Librex.GraphSolver()
        solver.initialize()
        result = solver.solve(network_problem)

        assert result.status == SolverStatus.SUCCESS
        assert "adjacency_matrix" in result.solution
        assert "entropy" in result.solution

    def test_entropy_minimization(self, network_problem):
        solver = Librex.GraphSolver()
        solver.initialize()
        result = solver.solve(network_problem)

        # Optimized entropy should be less than baseline
        assert result.improvement_over_baseline >= 0
