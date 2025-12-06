"""
Comprehensive tests for Librex.Flow solver
"""

import pytest
import numpy as np
from MEZAN.core import OptimizationProblem, ProblemType, SolverStatus
from libria_flow import Librex.FlowSolver


class TestLibrex.FlowSolver:
    """Test suite for Librex.Flow solver"""

    @pytest.fixture
    def simple_workflow(self):
        """Simple linear workflow for testing"""
        return OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": ["start", "agent1", "agent2", "goal"],
                    "edges": [
                        ("start", "agent1"),
                        ("agent1", "agent2"),
                        ("agent2", "goal"),
                    ],
                    "start_node": "start",
                    "goal_node": "goal",
                },
                "confidence_scores": {
                    "start": 1.0,
                    "agent1": 0.9,
                    "agent2": 0.8,
                    "goal": 1.0,
                },
            },
        )

    @pytest.fixture
    def branching_workflow(self):
        """Branching workflow with multiple paths"""
        return OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": ["start", "a1", "a2", "a3", "goal"],
                    "edges": [
                        ("start", "a1"),
                        ("start", "a2"),
                        ("a1", "a3"),
                        ("a2", "a3"),
                        ("a3", "goal"),
                    ],
                    "start_node": "start",
                    "goal_node": "goal",
                },
                "confidence_scores": {
                    "start": 1.0,
                    "a1": 0.95,  # High confidence path
                    "a2": 0.7,   # Low confidence path
                    "a3": 0.9,
                    "goal": 1.0,
                },
            },
        )

    @pytest.fixture
    def solver(self):
        """Default solver instance"""
        return Librex.FlowSolver()

    def test_solver_initialization(self, solver):
        """Test solver initializes correctly"""
        assert not solver._is_initialized
        solver.initialize()
        assert solver._is_initialized

    def test_solve_simple_workflow(self, solver, simple_workflow):
        """Test solving simple linear workflow"""
        solver.initialize()
        result = solver.solve(simple_workflow)

        assert result.status == SolverStatus.SUCCESS
        assert result.solution is not None
        assert "path" in result.solution
        assert "confidence" in result.solution

        # Path should go through all nodes
        path = result.solution["path"]
        assert path[0] == "start"
        assert path[-1] == "goal"

    def test_confidence_maximization(self, solver, branching_workflow):
        """Test solver chooses high-confidence path"""
        solver.initialize()
        result = solver.solve(branching_workflow)

        path = result.solution["path"]
        # Should prefer a1 (0.95) over a2 (0.7)
        assert "a1" in path

    def test_confidence_computation(self, solver, simple_workflow):
        """Test confidence is computed as product"""
        solver.initialize()
        result = solver.solve(simple_workflow)

        confidence = result.solution["confidence"]
        # Should be product of all node confidences
        expected = 1.0 * 0.9 * 0.8 * 1.0  # start * agent1 * agent2 * goal
        assert abs(confidence - expected) < 0.01

    def test_improvement_over_baseline(self, solver, branching_workflow):
        """Test improvement over greedy baseline"""
        solver.initialize()
        result = solver.solve(branching_workflow)

        assert result.improvement_over_baseline is not None

    def test_invalid_problem_no_path(self, solver):
        """Test handling when no path exists"""
        no_path_problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": ["start", "isolated", "goal"],
                    "edges": [],  # No edges!
                    "start_node": "start",
                    "goal_node": "goal",
                },
                "confidence_scores": {"start": 1.0, "isolated": 0.5, "goal": 1.0},
            },
        )

        solver.initialize()
        result = solver.solve(no_path_problem)

        assert result.status == SolverStatus.FAILED

    def test_metadata_completeness(self, solver, simple_workflow):
        """Test result metadata"""
        solver.initialize()
        result = solver.solve(simple_workflow)

        assert "solver" in result.metadata
        assert "num_nodes" in result.metadata
        assert "path_length" in result.metadata


class TestLibrex.FlowAlgorithms:
    """Test specific algorithms"""

    def test_confidence_dijkstra_correctness(self):
        """Test Dijkstra variant finds optimal path"""
        solver = Librex.FlowSolver()
        solver.initialize()

        # Create problem where optimal path is non-obvious
        problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": ["s", "a", "b", "c", "g"],
                    "edges": [
                        ("s", "a"),
                        ("s", "b"),
                        ("a", "c"),
                        ("b", "c"),
                        ("c", "g"),
                    ],
                    "start_node": "s",
                    "goal_node": "g",
                },
                "confidence_scores": {
                    "s": 1.0,
                    "a": 0.9,   # path s->a->c->g: 1.0 * 0.9 * 0.95 * 1.0 = 0.855
                    "b": 0.85,  # path s->b->c->g: 1.0 * 0.85 * 0.95 * 1.0 = 0.8075
                    "c": 0.95,
                    "g": 1.0,
                },
            },
        )

        result = solver.solve(problem)
        path = result.solution["path"]

        # Should choose path through 'a' (higher confidence)
        assert "a" in path


@pytest.mark.benchmark
class TestLibrex.FlowBenchmarks:
    """Performance benchmarks"""

    @pytest.mark.parametrize("num_nodes", [10, 20, 50, 100])
    def test_scaling(self, num_nodes):
        """Test scaling with graph size"""
        # Create random graph
        nodes = [f"n{i}" for i in range(num_nodes)]
        edges = [(f"n{i}", f"n{i+1}") for i in range(num_nodes - 1)]
        confidence = {node: 0.8 + 0.2 * np.random.rand() for node in nodes}

        problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    "nodes": nodes,
                    "edges": edges,
                    "start_node": "n0",
                    "goal_node": f"n{num_nodes-1}",
                },
                "confidence_scores": confidence,
            },
        )

        solver = Librex.FlowSolver()
        solver.initialize()
        result = solver.solve(problem)

        assert result.status == SolverStatus.SUCCESS
        print(f"num_nodes={num_nodes}: time={result.computation_time:.3f}s")
