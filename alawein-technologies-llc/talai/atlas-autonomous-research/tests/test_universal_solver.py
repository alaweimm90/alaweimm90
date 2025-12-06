"""
UARO Test Suite - Universal Solver

Tests for UniversalSolver, ProblemClassifier, and meta-algorithm.
Validates problem solving, convergence, and meta-learning.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import (
    UniversalSolver,
    ProblemClassifier,
    PrimitiveRegistry,
    Problem,
    solve_with_uaro,
)


class MazeProblem(Problem):
    """Test maze problem"""

    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)

    def initial_state(self):
        return (0, 0)

    def goal_test(self, state):
        return state == self.goal

    def actions(self, state):
        x, y = state
        actions = []
        if x > 0:
            actions.append(("left", (x - 1, y)))
        if x < self.size - 1:
            actions.append(("right", (x + 1, y)))
        if y > 0:
            actions.append(("up", (x, y - 1)))
        if y < self.size - 1:
            actions.append(("down", (x, y + 1)))
        return actions

    def result(self, state, action):
        return action[1]

    def cost(self, state, action):
        return 1


class TrivialProblem(Problem):
    """Problem that is already solved"""

    def initial_state(self):
        return "solved"

    def goal_test(self, state):
        return state == "solved"

    def actions(self, state):
        return []

    def result(self, state, action):
        return state

    def cost(self, state, action):
        return 0


class ImpossibleProblem(Problem):
    """Problem with no solution"""

    def initial_state(self):
        return 0

    def goal_test(self, state):
        return state == -1  # Can never reach

    def actions(self, state):
        return [("increment", state + 1)]  # Can only increment

    def result(self, state, action):
        return action[1]

    def cost(self, state, action):
        return 1


def test_solver_initialization():
    """Test UniversalSolver initialization"""
    solver = UniversalSolver()

    assert solver.registry is not None
    assert solver.max_iterations == 1000
    assert solver.confidence_threshold == 0.95
    assert len(solver.primitive_scores) == 0


def test_solver_on_maze():
    """Test solver on simple maze problem"""
    solver = UniversalSolver(max_iterations=100)
    problem = MazeProblem(size=3)

    result = solver.solve(problem)

    assert result is not None
    assert result.iterations > 0
    assert result.iterations <= 100
    assert result.solution is not None
    assert result.duration_seconds > 0


def test_solver_on_trivial_problem():
    """Test solver on already-solved problem"""
    solver = UniversalSolver()
    problem = TrivialProblem()

    result = solver.solve(problem)

    assert result.success
    assert result.iterations >= 1
    assert result.solution == "solved"
    assert result.confidence >= 0.5


def test_solver_on_impossible_problem():
    """Test solver on unsolvable problem"""
    solver = UniversalSolver(max_iterations=10)
    problem = ImpossibleProblem()

    result = solver.solve(problem)

    assert result.iterations == 10  # Should hit max
    assert result.solution is not None  # Returns best attempt
    assert len(result.reasoning_trace) > 0


def test_solver_meta_learning():
    """Test solver meta-learning updates"""
    solver = UniversalSolver()
    problem = MazeProblem(size=3)

    result = solver.solve(problem)

    assert len(solver.primitive_scores) > 0
    used_primitives = set(step.primitive_name for step in result.reasoning_trace)

    for prim_name in used_primitives:
        assert prim_name in solver.primitive_scores
        assert 0 <= solver.primitive_scores[prim_name] <= 1


def test_solver_convergence():
    """Test solver convergence detection"""
    solver = UniversalSolver(confidence_threshold=0.9, max_iterations=100)
    problem = TrivialProblem()

    result = solver.solve(problem)

    assert result.iterations < 100  # Should converge early
    assert result.success


def test_solver_reasoning_trace():
    """Test solver reasoning trace completeness"""
    solver = UniversalSolver()
    problem = MazeProblem(size=3)

    result = solver.solve(problem)

    assert len(result.reasoning_trace) > 0

    for step in result.reasoning_trace:
        assert step.iteration > 0
        assert step.primitive_name is not None
        assert step.confidence >= 0
        assert step.reasoning is not None


def test_problem_classifier():
    """Test ProblemClassifier on different problem types"""
    classifier = ProblemClassifier()

    maze = MazeProblem()
    types = classifier.classify(maze)

    assert "search_problem" in types

    class DecomposableProblem:
        def split(self):
            return self, self

    decomp = DecomposableProblem()
    types = classifier.classify(decomp)
    assert "decomposable" in types


def test_problem_classifier_recommendations():
    """Test ProblemClassifier primitive recommendations"""
    classifier = ProblemClassifier()

    maze = MazeProblem()
    recommendations = classifier.get_recommended_primitives(maze)

    assert len(recommendations) > 0
    assert any(
        "search" in rec for rec in recommendations
    )  # Should recommend search primitives


def test_convenience_function():
    """Test solve_with_uaro convenience function"""
    problem = MazeProblem(size=3)

    result = solve_with_uaro(problem, max_iterations=50)

    assert result is not None
    assert result.iterations > 0
    assert result.iterations <= 50


def test_solver_with_custom_registry():
    """Test solver with custom primitive registry"""
    registry = PrimitiveRegistry()

    assert len(registry.primitives) > 0  # Has defaults

    solver = UniversalSolver(primitive_registry=registry)
    problem = MazeProblem(size=3)

    result = solver.solve(problem)
    assert result is not None


def test_solver_primitive_selection():
    """Test solver selects appropriate primitives"""
    solver = UniversalSolver()
    problem = MazeProblem(size=3)

    result = solver.solve(problem)

    used_primitives = set(step.primitive_name for step in result.reasoning_trace)

    search_primitives = {
        "breadth_first_search",
        "depth_first_search",
        "best_first_search",
        "beam_search",
    }

    assert any(prim in search_primitives for prim in used_primitives)


def test_solver_statistics():
    """Test solver statistics tracking"""
    solver = UniversalSolver()
    problem = MazeProblem(size=3)

    solver.solve(problem)

    stats = solver.get_primitive_stats()

    assert len(stats) > 0

    for prim_name, prim_stats in stats.items():
        assert "category" in prim_stats
        assert "usage_count" in prim_stats
        assert "success_rate" in prim_stats
        assert 0 <= prim_stats["success_rate"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
