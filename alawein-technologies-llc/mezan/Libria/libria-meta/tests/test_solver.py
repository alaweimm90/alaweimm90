"""Tests for Librex.Meta solver"""
import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_meta import Librex.MetaSolver

class TestLibrex.MetaSolver:
    @pytest.fixture
    def qap_problem(self):
        return OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                "flow_matrix": [[0, 5, 3], [5, 0, 2], [3, 2, 0]],
            },
        )

    @pytest.fixture
    def flow_problem(self):
        return OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": ["a", "b", "c"],
                    "edges": [("a", "b"), ("b", "c")],
                    "start_node": "a",
                    "goal_node": "c",
                },
                "confidence_scores": {"a": 1.0, "b": 0.9, "c": 1.0},
            },
        )

    def test_solver_selection_qap(self, qap_problem):
        solver = Librex.MetaSolver()
        solver.initialize()
        result = solver.solve(qap_problem)

        assert result.status == SolverStatus.SUCCESS
        assert result.solution["selected_solver"] == "Librex.QAP"

    def test_solver_selection_flow(self, flow_problem):
        solver = Librex.MetaSolver()
        solver.initialize()
        result = solver.solve(flow_problem)

        assert result.status == SolverStatus.SUCCESS
        assert result.solution["selected_solver"] == "Librex.Flow"

    def test_ucb_exploration(self, qap_problem):
        solver = Librex.MetaSolver()
        solver.initialize()

        # Solve multiple times to test UCB
        for _ in range(10):
            solver.solve(qap_problem)

        # Total selections should increase
        assert solver.total_selections > 0

    def test_feature_extraction(self, qap_problem):
        solver = Librex.MetaSolver()
        solver.initialize()

        features = solver._extract_features(qap_problem)

        assert "problem_type" in features
        assert features["problem_type"] == "quadratic_assignment"
        assert "problem_size" in features

    def test_statistics_update(self):
        solver = Librex.MetaSolver()
        solver.initialize()

        solver._update_statistics("Librex.QAP", 0.8)
        assert solver.solver_stats["Librex.QAP"]["count"] == 1
        assert solver.solver_stats["Librex.QAP"]["avg_reward"] == 0.8

        solver._update_statistics("Librex.QAP", 0.6)
        assert solver.solver_stats["Librex.QAP"]["count"] == 2
        assert abs(solver.solver_stats["Librex.QAP"]["avg_reward"] - 0.7) < 1e-6

    def test_solver_rankings(self):
        solver = Librex.MetaSolver()
        solver.initialize()

        solver._update_statistics("Librex.QAP", 0.9)
        solver._update_statistics("Librex.Flow", 0.7)

        rankings = solver.get_solver_rankings()

        assert len(rankings) > 0
        assert rankings[0][0] == "Librex.QAP"  # Highest ranked

    def test_reset_statistics(self):
        solver = Librex.MetaSolver()
        solver.initialize()

        solver._update_statistics("Librex.QAP", 0.8)
        solver.reset_statistics()

        assert solver.solver_stats["Librex.QAP"]["count"] == 0
        assert solver.total_selections == 0
