"""
Comprehensive tests for constraint handling module.

Tests cover:
- Constraint types (equality, inequality, box, linear, nonlinear)
- Constraint handlers (penalty methods, repair operators)
- Feasibility preservation techniques
"""

import numpy as np
import pytest

from Librex.constraints import (
    Constraint,
    EqualityConstraint,
    InequalityConstraint,
    BoxConstraint,
    LinearConstraint,
    NonlinearConstraint,
    ConstraintSet,
    StaticPenalty,
    DynamicPenalty,
    AdaptivePenalty,
    DeathPenalty,
    ProjectionRepair,
    FeasibilityRestoration,
    ConstraintDomination,
    EpsilonConstraintMethod,
    StochasticRanking,
    FeasibilityPreservingCrossover,
    FeasibilityPreservingMutation,
    feasibility_check,
    project_to_feasible_region,
    random_feasible_solution,
)


class TestConstraintTypes:
    """Test various constraint type implementations."""

    def test_equality_constraint(self):
        """Test equality constraint."""
        # h(x) = x[0] + x[1] - 1 = 0
        def h(x):
            return x[0] + x[1] - 1

        constraint = EqualityConstraint(h)

        # Test feasible point
        x_feasible = np.array([0.3, 0.7])
        assert constraint.is_feasible(x_feasible)
        assert constraint.evaluate(x_feasible) < 1e-6

        # Test infeasible point
        x_infeasible = np.array([0.5, 0.6])
        assert not constraint.is_feasible(x_infeasible)
        assert constraint.evaluate(x_infeasible) > 0

    def test_inequality_constraint(self):
        """Test inequality constraint."""
        # g(x) = x[0]^2 + x[1]^2 - 1 <= 0 (unit circle)
        def g(x):
            return x[0] ** 2 + x[1] ** 2 - 1

        constraint = InequalityConstraint(g)

        # Test feasible point (inside circle)
        x_feasible = np.array([0.5, 0.5])
        assert constraint.is_feasible(x_feasible)
        assert constraint.evaluate(x_feasible) == 0

        # Test infeasible point (outside circle)
        x_infeasible = np.array([1.0, 1.0])
        assert not constraint.is_feasible(x_infeasible)
        assert constraint.evaluate(x_infeasible) > 0

    def test_box_constraint(self):
        """Test box constraint."""
        lower_bounds = np.array([0, -1, -2])
        upper_bounds = np.array([1, 2, 3])

        constraint = BoxConstraint(lower_bounds, upper_bounds)

        # Test feasible point
        x_feasible = np.array([0.5, 0, 1])
        assert constraint.is_feasible(x_feasible)
        assert constraint.evaluate(x_feasible) == 0

        # Test infeasible points
        x_below = np.array([-0.1, 0, 0])
        assert not constraint.is_feasible(x_below)
        assert constraint.evaluate(x_below) > 0

        x_above = np.array([0.5, 3, 0])
        assert not constraint.is_feasible(x_above)
        assert constraint.evaluate(x_above) > 0

        # Test projection
        x_projected = constraint.project(np.array([-1, 3, 4]))
        np.testing.assert_array_equal(x_projected, [0, 2, 3])

    def test_linear_inequality_constraint(self):
        """Test linear inequality constraint."""
        # Ax <= b: x[0] + 2*x[1] <= 3
        A = np.array([[1, 2]])
        b = np.array([3])

        constraint = LinearConstraint(A, b, constraint_type="inequality")

        # Test feasible point
        x_feasible = np.array([1, 0.5])
        assert constraint.is_feasible(x_feasible)

        # Test infeasible point
        x_infeasible = np.array([2, 1])
        assert not constraint.is_feasible(x_infeasible)

    def test_linear_equality_constraint(self):
        """Test linear equality constraint."""
        # Ax = b: x[0] - x[1] = 1
        A = np.array([[1, -1]])
        b = np.array([1])

        constraint = LinearConstraint(A, b, constraint_type="equality")

        # Test feasible point
        x_feasible = np.array([2, 1])
        assert constraint.is_feasible(x_feasible)

        # Test infeasible point
        x_infeasible = np.array([1, 1])
        assert not constraint.is_feasible(x_infeasible)

    def test_nonlinear_constraint(self):
        """Test nonlinear constraint."""
        # g(x) = x[0]^3 - x[1] <= 0
        def g(x):
            return x[0] ** 3 - x[1]

        def g_grad(x):
            return np.array([3 * x[0] ** 2, -1])

        constraint = NonlinearConstraint(
            g,
            gradient_function=g_grad,
            constraint_type="inequality"
        )

        # Test feasible point
        x_feasible = np.array([0.5, 0.5])
        assert constraint.is_feasible(x_feasible)

        # Test infeasible point
        x_infeasible = np.array([2, 1])
        assert not constraint.is_feasible(x_infeasible)

        # Test gradient
        grad = constraint.gradient(np.array([1, 1]))
        np.testing.assert_array_almost_equal(grad, [3, -1])

    def test_constraint_set(self):
        """Test ConstraintSet container."""
        constraints = ConstraintSet([
            BoxConstraint(np.array([0, 0]), np.array([1, 1])),
            InequalityConstraint(lambda x: x[0] + x[1] - 1.5)
        ])

        # Test feasible point
        x_feasible = np.array([0.5, 0.5])
        assert constraints.is_feasible(x_feasible)

        # Test infeasible points
        x_outside_box = np.array([1.5, 0.5])
        assert not constraints.is_feasible(x_outside_box)

        x_violates_sum = np.array([0.8, 0.8])
        assert not constraints.is_feasible(x_violates_sum)

        # Test evaluation
        total_violation, violations = constraints.evaluate(x_violates_sum)
        assert total_violation > 0
        assert len(violations) == 2


class TestPenaltyMethods:
    """Test penalty-based constraint handlers."""

    def create_test_constraints(self):
        """Create test constraint set."""
        return ConstraintSet([
            InequalityConstraint(lambda x: x[0] ** 2 + x[1] ** 2 - 1),  # Circle
            InequalityConstraint(lambda x: x[0] + x[1] - 1.5)  # Linear
        ])

    def test_static_penalty(self):
        """Test static penalty method."""
        constraints = self.create_test_constraints()
        penalty_handler = StaticPenalty(penalty_factor=1000.0)

        # Feasible solution
        x_feasible = np.array([0.5, 0.5])
        objective = 1.0
        penalized_obj, _ = penalty_handler.handle(
            x_feasible, objective, constraints
        )
        assert penalized_obj == objective  # No penalty

        # Infeasible solution
        x_infeasible = np.array([1.0, 1.0])
        objective = 1.0
        penalized_obj, _ = penalty_handler.handle(
            x_infeasible, objective, constraints
        )
        assert penalized_obj > objective  # Penalty applied

    def test_dynamic_penalty(self):
        """Test dynamic penalty method."""
        constraints = self.create_test_constraints()
        penalty_handler = DynamicPenalty(base_factor=0.5, growth_rate=2.0)

        x_infeasible = np.array([1.0, 1.0])
        objective = 1.0

        # Penalty should increase with generation
        penalties = []
        for gen in [1, 10, 100]:
            penalized_obj, _ = penalty_handler.handle(
                x_infeasible, objective, constraints, generation=gen
            )
            penalties.append(penalized_obj - objective)

        assert penalties[0] < penalties[1] < penalties[2]

    def test_adaptive_penalty(self):
        """Test adaptive penalty method."""
        constraints = self.create_test_constraints()
        penalty_handler = AdaptivePenalty(
            initial_factor=100.0,
            target_feasibility_ratio=0.5
        )

        # Simulate population feasibility
        # Too many feasible - should increase penalty
        penalty_handler.update_feasibility_ratio([True] * 8 + [False] * 2)
        factor1 = penalty_handler.current_factor

        # Too few feasible - should decrease penalty
        penalty_handler.update_feasibility_ratio([True] * 2 + [False] * 8)
        factor2 = penalty_handler.current_factor

        assert factor1 > penalty_handler.penalty_factor  # Increased
        assert factor2 < factor1  # Decreased

    def test_death_penalty(self):
        """Test death penalty method."""
        constraints = self.create_test_constraints()
        penalty_handler = DeathPenalty(death_value=1e10)

        # Feasible solution
        x_feasible = np.array([0.5, 0.5])
        objective = 1.0
        penalized_obj, _ = penalty_handler.handle(
            x_feasible, objective, constraints
        )
        assert penalized_obj == objective

        # Infeasible solution
        x_infeasible = np.array([1.0, 1.0])
        penalized_obj, _ = penalty_handler.handle(
            x_infeasible, objective, constraints
        )
        assert penalized_obj == 1e10


class TestRepairOperators:
    """Test repair-based constraint handlers."""

    def test_projection_repair(self):
        """Test projection repair operator."""
        # Box constraints for simple projection
        constraints = ConstraintSet([
            BoxConstraint(np.array([0, 0]), np.array([1, 1]))
        ])

        repair_op = ProjectionRepair()

        # Infeasible solution
        x_infeasible = np.array([1.5, -0.5])
        x_repaired = repair_op.repair(x_infeasible, constraints)

        # Should be projected to box
        assert constraints.is_feasible(x_repaired)
        np.testing.assert_array_almost_equal(x_repaired, [1.0, 0.0])

    def test_feasibility_restoration(self):
        """Test gradient-based feasibility restoration."""
        # Simple linear constraint: x[0] + x[1] <= 1
        constraints = ConstraintSet([
            InequalityConstraint(lambda x: x[0] + x[1] - 1)
        ])

        restoration = FeasibilityRestoration(step_size=0.1)

        # Infeasible solution
        x_infeasible = np.array([0.8, 0.8])
        x_repaired = restoration.repair(x_infeasible, constraints)

        # Should move toward feasible region
        violation_before = constraints.evaluate(x_infeasible)[0]
        violation_after = constraints.evaluate(x_repaired)[0]
        assert violation_after < violation_before


class TestConstraintDomination:
    """Test constraint domination for multi-objective optimization."""

    def test_constraint_domination(self):
        """Test constraint domination comparison."""
        domination = ConstraintDomination()

        # Feasible dominates infeasible
        x1 = np.array([1, 1])
        obj1 = np.array([1, 2])
        viol1 = 0.0

        x2 = np.array([0.5, 0.5])
        obj2 = np.array([0.5, 1.5])
        viol2 = 0.1

        assert domination.dominates(x1, obj1, viol1, x2, obj2, viol2)
        assert not domination.dominates(x2, obj2, viol2, x1, obj1, viol1)

        # Both infeasible - less violation dominates
        x3 = np.array([2, 2])
        obj3 = np.array([2, 3])
        viol3 = 0.2

        x4 = np.array([3, 3])
        obj4 = np.array([1, 1])
        viol4 = 0.5

        assert domination.dominates(x3, obj3, viol3, x4, obj4, viol4)

        # Both feasible - Pareto dominance
        x5 = np.array([1, 1])
        obj5 = np.array([1, 1])
        viol5 = 0.0

        x6 = np.array([2, 2])
        obj6 = np.array([2, 2])
        viol6 = 0.0

        assert domination.dominates(x5, obj5, viol5, x6, obj6, viol6)


class TestEpsilonConstraint:
    """Test epsilon-constraint method."""

    def test_epsilon_constraint(self):
        """Test epsilon-constraint comparison."""
        eps_method = EpsilonConstraintMethod(
            initial_epsilon=0.1,
            reduction_rate=0.9
        )

        # Both epsilon-feasible - compare objectives
        result = eps_method.compare(
            obj1=1.0, violation1=0.05,
            obj2=2.0, violation2=0.08
        )
        assert result == 1  # First is better (lower objective)

        # One epsilon-feasible
        result = eps_method.compare(
            obj1=3.0, violation1=0.05,
            obj2=1.0, violation2=0.15
        )
        assert result == 1  # First is epsilon-feasible

        # Update epsilon
        eps_method.update(generation=10)
        assert eps_method.current_epsilon < 0.1


class TestStochasticRanking:
    """Test stochastic ranking."""

    def test_stochastic_ranking(self):
        """Test stochastic ranking of solutions."""
        ranker = StochasticRanking(probability_feasibility=0.45)

        # Test solutions with objectives and violations
        objectives = np.array([3.0, 1.0, 2.0, 4.0])
        violations = np.array([0.0, 0.1, 0.0, 0.2])

        indices = ranker.rank(objectives, violations)

        # First should be feasible with best objective (index 2)
        # or infeasible with least violation (index 1)
        assert indices[0] in [1, 2]


class TestFeasibilityPreservation:
    """Test feasibility preservation techniques."""

    def test_feasibility_preserving_crossover(self):
        """Test feasibility-preserving crossover."""
        constraints = ConstraintSet([
            BoxConstraint(np.array([0, 0]), np.array([1, 1]))
        ])

        crossover = FeasibilityPreservingCrossover(constraints)

        parent1 = np.array([0.3, 0.3])
        parent2 = np.array([0.7, 0.7])

        # Blend crossover
        offspring1, offspring2 = crossover.blend_crossover(
            parent1, parent2, alpha=0.5
        )

        # At least one offspring should be feasible
        feasible = (
            constraints.is_feasible(offspring1) or
            constraints.is_feasible(offspring2)
        )
        assert feasible

        # Arithmetic crossover (convex combination)
        offspring1, offspring2 = crossover.arithmetic_crossover(
            parent1, parent2
        )

        # Convex combination of feasible points should be feasible
        assert constraints.is_feasible(offspring1)
        assert constraints.is_feasible(offspring2)

    def test_feasibility_preserving_mutation(self):
        """Test feasibility-preserving mutation."""
        constraints = ConstraintSet([
            BoxConstraint(np.array([0, 0]), np.array([1, 1]))
        ])

        mutation = FeasibilityPreservingMutation(constraints)

        x = np.array([0.5, 0.5])
        bounds = (np.array([0, 0]), np.array([1, 1]))

        # Boundary mutation
        mutated = mutation.boundary_mutation(x, bounds, mutation_prob=1.0)
        assert constraints.is_feasible(mutated)

        # Gaussian mutation
        mutated = mutation.gaussian_mutation(x, sigma=0.1, bounds=bounds)
        assert constraints.is_feasible(mutated)

        # Non-uniform mutation
        mutated = mutation.non_uniform_mutation(
            x, generation=10, max_generation=100, bounds=bounds
        )
        assert constraints.is_feasible(mutated)

    def test_feasibility_utilities(self):
        """Test feasibility utility functions."""
        constraints = ConstraintSet([
            BoxConstraint(np.array([0, 0]), np.array([1, 1])),
            InequalityConstraint(lambda x: x[0] + x[1] - 1.5)
        ])

        # Test feasibility check
        x = np.array([0.5, 0.5])
        is_feasible, violation, violated = feasibility_check(x, constraints)
        assert is_feasible
        assert violation == 0
        assert len(violated) == 0

        # Test projection
        x_infeasible = np.array([1.5, 1.5])
        x_projected = project_to_feasible_region(
            x_infeasible, constraints, method="sequential"
        )
        is_feasible, _, _ = feasibility_check(x_projected, constraints)
        assert is_feasible or np.linalg.norm(x_projected - x_infeasible) < np.linalg.norm(x_infeasible)

        # Test random feasible solution generation
        bounds = (np.array([0, 0]), np.array([1, 1]))
        x_random = random_feasible_solution(
            constraints, bounds, max_attempts=1000
        )
        if x_random is not None:
            assert constraints.is_feasible(x_random)