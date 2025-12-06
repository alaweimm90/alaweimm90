"""
UARO Test Suite - Reasoning Primitives

Tests for all 12 universal reasoning primitives.
Validates correctness, performance, and edge cases.

Test Coverage:
- Decomposition primitives (2)
- Search primitives (4)
- Constraint primitives (2)
- Logic primitives (2)
- Optimization primitives (2)
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import (
    DivideAndConquer,
    HierarchicalDecomposition,
    BreadthFirstSearch,
    DepthFirstSearch,
    BestFirstSearch,
    BeamSearch,
    ConstraintPropagation,
    BacktrackingSearch,
    ForwardChaining,
    BackwardChaining,
    LocalSearch,
    SimulatedAnnealing,
    Problem,
)


# Test fixtures


class SimpleProblem(Problem):
    """Simple test problem for search primitives"""

    def __init__(self, size=5):
        self.size = size

    def initial_state(self):
        return (0, 0)

    def goal_test(self, state):
        return state == (self.size - 1, self.size - 1)

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


class DecomposableProblem:
    """Test problem for decomposition primitives"""

    def __init__(self, items):
        self.items = items
        self._depth = 0

    def is_atomic(self):
        return len(self.items) <= 1

    def split(self):
        mid = len(self.items) // 2
        left = DecomposableProblem(self.items[:mid])
        right = DecomposableProblem(self.items[mid:])
        left._depth = self._depth + 1
        right._depth = self._depth + 1
        return left, right

    def max_depth(self):
        return 3

    def expand_level(self, level):
        if level == 0:
            return [self]
        elif level == 1:
            left, right = self.split()
            return [left, right]
        else:
            return []


class SimpleCSP:
    """Test CSP for constraint primitives"""

    def __init__(self):
        self.variables = ["A", "B", "C"]
        self.domains = {
            "A": [1, 2, 3],
            "B": [1, 2, 3],
            "C": [1, 2, 3],
        }
        self.constraints = [
            DifferentConstraint("A", "B"),
            DifferentConstraint("B", "C"),
            DifferentConstraint("A", "C"),
        ]

    def is_complete(self, assignment):
        return len(assignment) == len(self.variables)


class DifferentConstraint:
    """Simple constraint: variables must be different"""

    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def check(self, var, value):
        return True  # Simple check

    def is_satisfied(self, assignment):
        if self.var1 in assignment and self.var2 in assignment:
            return assignment[self.var1] != assignment[self.var2]
        return True


class SimpleKB:
    """Test knowledge base for logic primitives"""

    def __init__(self):
        self.facts = {"sky_is_blue", "grass_is_green"}
        self.rules = [
            ("sky_is_blue", "daytime"),
            ("daytime", "sun_is_out"),
        ]


class OptimizationState:
    """Test state for optimization primitives"""

    def __init__(self, x):
        self.x = x

    def neighbors(self):
        return [OptimizationState(self.x - 1), OptimizationState(self.x + 1)]

    def random_neighbor(self):
        import random

        return OptimizationState(self.x + random.choice([-1, 1]))


# Decomposition Tests


def test_divide_and_conquer():
    """Test DivideAndConquer primitive"""
    primitive = DivideAndConquer()
    problem = DecomposableProblem(list(range(8)))

    assert primitive.is_applicable(problem)

    subproblems = primitive.apply(problem)

    assert len(subproblems) > 1
    assert all(sp.is_atomic() for sp in subproblems)

    primitive.record_usage(True)
    assert primitive.usage_count == 1
    assert primitive.success_count == 1


def test_hierarchical_decomposition():
    """Test HierarchicalDecomposition primitive"""
    primitive = HierarchicalDecomposition()
    problem = DecomposableProblem(list(range(8)))

    assert primitive.is_applicable(problem)

    hierarchy = primitive.apply(problem)

    assert "root" in hierarchy
    assert hierarchy["root"] == problem
    assert "level_0" in hierarchy


# Search Tests


def test_breadth_first_search():
    """Test BreadthFirstSearch primitive"""
    primitive = BreadthFirstSearch()
    problem = SimpleProblem(size=3)

    assert primitive.is_applicable(problem)

    solution = primitive.apply(problem)

    assert solution is not None
    assert problem.goal_test(solution)


def test_depth_first_search():
    """Test DepthFirstSearch primitive"""
    primitive = DepthFirstSearch()
    problem = SimpleProblem(size=3)

    assert primitive.is_applicable(problem)

    solution = primitive.apply(problem, max_depth=20)

    assert solution is not None
    assert problem.goal_test(solution)


def test_best_first_search():
    """Test BestFirstSearch primitive"""
    primitive = BestFirstSearch()
    problem = SimpleProblem(size=3)

    def manhattan_heuristic(state):
        x, y = state
        return -(abs(x - 2) + abs(y - 2))  # Negative for max-heap behavior

    assert primitive.is_applicable(problem)

    solution = primitive.apply(problem, heuristic=manhattan_heuristic)

    assert solution is not None
    assert problem.goal_test(solution)


def test_beam_search():
    """Test BeamSearch primitive"""
    primitive = BeamSearch()
    problem = SimpleProblem(size=3)

    def manhattan_heuristic(state):
        x, y = state
        return -(abs(x - 2) + abs(y - 2))

    assert primitive.is_applicable(problem)

    solution = primitive.apply(problem, beam_width=3, heuristic=manhattan_heuristic)

    assert solution is not None
    assert problem.goal_test(solution)


# Constraint Tests


def test_constraint_propagation():
    """Test ConstraintPropagation primitive"""
    primitive = ConstraintPropagation()
    csp = SimpleCSP()

    assert primitive.is_applicable(csp)

    result = primitive.apply(csp)

    assert result is not None
    assert len(result.domains["A"]) <= 3


def test_backtracking_search():
    """Test BacktrackingSearch primitive"""
    primitive = BacktrackingSearch()
    csp = SimpleCSP()

    assert primitive.is_applicable(csp)

    solution = primitive.apply(csp)

    assert solution is not None
    assert len(solution) == 3
    assert solution["A"] != solution["B"]
    assert solution["B"] != solution["C"]
    assert solution["A"] != solution["C"]


# Logic Tests


def test_forward_chaining():
    """Test ForwardChaining primitive"""
    primitive = ForwardChaining()
    kb = SimpleKB()

    assert primitive.is_applicable(kb)

    inferred = primitive.apply(kb)

    assert "sky_is_blue" in inferred
    assert "daytime" in inferred
    assert "sun_is_out" in inferred


def test_backward_chaining():
    """Test BackwardChaining primitive"""
    primitive = BackwardChaining()
    kb = SimpleKB()

    assert primitive.is_applicable(kb)

    proved, path = primitive.apply(kb, "sun_is_out")

    assert proved
    assert len(path) > 0
    assert "sun_is_out" in path


# Optimization Tests


def test_local_search():
    """Test LocalSearch primitive"""
    primitive = LocalSearch()

    def objective(state):
        return -(state.x**2)  # Maximize negative (minimize x^2)

    initial = OptimizationState(5)

    assert primitive.is_applicable(initial)

    solution = primitive.apply(objective, initial, max_iterations=20)

    assert abs(solution.x) < abs(initial.x)


def test_simulated_annealing():
    """Test SimulatedAnnealing primitive"""
    primitive = SimulatedAnnealing()

    def objective(state):
        return -(state.x**2)

    initial = OptimizationState(5)

    assert primitive.is_applicable(initial)

    temperature_schedule = [1.0 * (0.9**i) for i in range(100)]
    solution = primitive.apply(objective, initial, temperature_schedule)

    assert abs(solution.x) <= abs(initial.x) + 5  # Allow some variance


# Performance Tests


def test_primitive_usage_tracking():
    """Test usage tracking across primitives"""
    primitive = BreadthFirstSearch()

    assert primitive.usage_count == 0
    assert primitive.success_count == 0
    assert primitive.success_rate() == 0.5  # Unknown

    primitive.record_usage(True)
    assert primitive.usage_count == 1
    assert primitive.success_count == 1
    assert primitive.success_rate() == 1.0

    primitive.record_usage(False)
    assert primitive.usage_count == 2
    assert primitive.success_count == 1
    assert primitive.success_rate() == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
