"""
Constraint handling methods for optimization.

This module provides various strategies for handling constraints:
- Penalty methods (static, dynamic, adaptive)
- Repair operators (projection, feasibility restoration)
- Constraint domination
- ε-constraint method
- Stochastic ranking
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np

from .constraints import Constraint, ConstraintSet

logger = logging.getLogger(__name__)


class ConstraintHandler(ABC):
    """Abstract base class for constraint handling methods."""

    @abstractmethod
    def handle(
        self,
        x: np.ndarray,
        objective: float,
        constraints: ConstraintSet,
        generation: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Handle constraints for a solution.

        Args:
            x: Decision variables
            objective: Original objective value
            constraints: Set of constraints
            generation: Optional generation/iteration number

        Returns:
            Tuple of (modified_objective, modified_solution)
        """
        pass


class PenaltyMethod(ConstraintHandler):
    """Base class for penalty-based constraint handling."""

    def __init__(self, penalty_factor: float = 1.0):
        """
        Initialize penalty method.

        Args:
            penalty_factor: Base penalty factor
        """
        self.penalty_factor = penalty_factor

    @abstractmethod
    def calculate_penalty(
        self,
        violations: List[float],
        generation: Optional[int] = None
    ) -> float:
        """Calculate penalty from constraint violations."""
        pass

    def handle(
        self,
        x: np.ndarray,
        objective: float,
        constraints: ConstraintSet,
        generation: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """Apply penalty to objective function."""
        total_violation, violations = constraints.evaluate(x)

        if total_violation > 0:
            penalty = self.calculate_penalty(violations, generation)
            penalized_objective = objective + penalty
        else:
            penalized_objective = objective

        return penalized_objective, x


class StaticPenalty(PenaltyMethod):
    """
    Static penalty method with fixed penalty coefficients.

    Mathematical formulation:
    f'(x) = f(x) + r * Σ max(0, g_i(x))^p
    where r is the penalty factor and p is the penalty exponent.
    """

    def __init__(
        self,
        penalty_factor: float = 1000.0,
        penalty_exponent: float = 2.0
    ):
        """
        Initialize static penalty.

        Args:
            penalty_factor: Fixed penalty coefficient
            penalty_exponent: Power for constraint violations
        """
        super().__init__(penalty_factor)
        self.penalty_exponent = penalty_exponent

    def calculate_penalty(
        self,
        violations: List[float],
        generation: Optional[int] = None
    ) -> float:
        """Calculate static penalty."""
        penalty = 0.0
        for violation in violations:
            if violation > 0:
                penalty += self.penalty_factor * (violation ** self.penalty_exponent)
        return penalty


class DynamicPenalty(PenaltyMethod):
    """
    Dynamic penalty method with generation-dependent coefficients.

    Mathematical formulation:
    f'(x) = f(x) + (C * t)^α * Σ max(0, g_i(x))^β
    where t is the generation number.
    """

    def __init__(
        self,
        base_factor: float = 0.5,
        growth_rate: float = 2.0,
        violation_exponent: float = 2.0
    ):
        """
        Initialize dynamic penalty.

        Args:
            base_factor: Base penalty coefficient (C)
            growth_rate: Growth exponent (α)
            violation_exponent: Violation exponent (β)
        """
        super().__init__(base_factor)
        self.growth_rate = growth_rate
        self.violation_exponent = violation_exponent

    def calculate_penalty(
        self,
        violations: List[float],
        generation: Optional[int] = None
    ) -> float:
        """Calculate dynamic penalty based on generation."""
        if generation is None:
            generation = 1

        # Dynamic penalty factor increases with generation
        dynamic_factor = (self.penalty_factor * generation) ** self.growth_rate

        penalty = 0.0
        for violation in violations:
            if violation > 0:
                penalty += dynamic_factor * (violation ** self.violation_exponent)

        return penalty


class AdaptivePenalty(PenaltyMethod):
    """
    Adaptive penalty method that adjusts based on feasibility ratio.

    The penalty factor is increased if too few feasible solutions exist,
    and decreased if too many feasible solutions exist.
    """

    def __init__(
        self,
        initial_factor: float = 100.0,
        target_feasibility_ratio: float = 0.45,
        adaptation_rate: float = 1.5,
        violation_exponent: float = 2.0
    ):
        """
        Initialize adaptive penalty.

        Args:
            initial_factor: Initial penalty factor
            target_feasibility_ratio: Target ratio of feasible solutions
            adaptation_rate: Rate of penalty adjustment
            violation_exponent: Power for violations
        """
        super().__init__(initial_factor)
        self.target_feasibility_ratio = target_feasibility_ratio
        self.adaptation_rate = adaptation_rate
        self.violation_exponent = violation_exponent
        self.feasibility_history = []
        self.current_factor = initial_factor

    def update_feasibility_ratio(self, is_feasible: List[bool]):
        """Update feasibility statistics and adapt penalty."""
        feasibility_ratio = sum(is_feasible) / len(is_feasible)
        self.feasibility_history.append(feasibility_ratio)

        # Adapt penalty factor based on feasibility ratio
        if feasibility_ratio < self.target_feasibility_ratio - 0.1:
            # Too few feasible solutions - decrease penalty
            self.current_factor /= self.adaptation_rate
        elif feasibility_ratio > self.target_feasibility_ratio + 0.1:
            # Too many feasible solutions - increase penalty
            self.current_factor *= self.adaptation_rate

        # Keep factor within reasonable bounds
        self.current_factor = np.clip(self.current_factor, 1e-6, 1e6)

    def calculate_penalty(
        self,
        violations: List[float],
        generation: Optional[int] = None
    ) -> float:
        """Calculate adaptive penalty."""
        penalty = 0.0
        for violation in violations:
            if violation > 0:
                penalty += self.current_factor * (violation ** self.violation_exponent)
        return penalty


class DeathPenalty(ConstraintHandler):
    """
    Death penalty: assigns infinite penalty to infeasible solutions.

    Simple but can be ineffective if feasible region is small.
    """

    def __init__(self, death_value: float = 1e10):
        """
        Initialize death penalty.

        Args:
            death_value: Penalty value for infeasible solutions
        """
        self.death_value = death_value

    def handle(
        self,
        x: np.ndarray,
        objective: float,
        constraints: ConstraintSet,
        generation: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """Apply death penalty if infeasible."""
        if not constraints.is_feasible(x):
            return self.death_value, x
        return objective, x


class RepairOperator(ConstraintHandler):
    """Base class for repair-based constraint handling."""

    @abstractmethod
    def repair(
        self,
        x: np.ndarray,
        constraints: ConstraintSet
    ) -> np.ndarray:
        """Repair infeasible solution."""
        pass

    def handle(
        self,
        x: np.ndarray,
        objective: float,
        constraints: ConstraintSet,
        generation: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """Repair solution if infeasible."""
        if not constraints.is_feasible(x):
            x = self.repair(x, constraints)
            # Recalculate objective for repaired solution
            # (This would need the objective function passed in)
        return objective, x


class ProjectionRepair(RepairOperator):
    """
    Repair by projecting to feasible region.

    Uses constraint-specific projection methods when available.
    """

    def __init__(self, max_iterations: int = 10):
        """
        Initialize projection repair.

        Args:
            max_iterations: Maximum projection iterations
        """
        self.max_iterations = max_iterations

    def repair(
        self,
        x: np.ndarray,
        constraints: ConstraintSet
    ) -> np.ndarray:
        """Project solution to feasible region."""
        return constraints.project(x, self.max_iterations)


class FeasibilityRestoration(RepairOperator):
    """
    Restore feasibility using gradient-based methods.

    Moves solution towards feasible region using constraint gradients.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        max_steps: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize feasibility restoration.

        Args:
            step_size: Step size for gradient descent
            max_steps: Maximum restoration steps
            tolerance: Feasibility tolerance
        """
        self.step_size = step_size
        self.max_steps = max_steps
        self.tolerance = tolerance

    def repair(
        self,
        x: np.ndarray,
        constraints: ConstraintSet
    ) -> np.ndarray:
        """Restore feasibility using gradients."""
        repaired = x.copy()

        for step in range(self.max_steps):
            total_violation, _ = constraints.evaluate(repaired)

            if total_violation <= self.tolerance:
                break

            # Move in direction opposite to constraint gradient
            gradient = constraints.gradient(repaired)
            if np.linalg.norm(gradient) > 1e-10:
                repaired -= self.step_size * gradient / np.linalg.norm(gradient)

        return repaired


class ConstraintDomination:
    """
    Constraint-domination for multi-objective optimization.

    Solutions are compared based on:
    1. Feasibility (feasible dominates infeasible)
    2. Constraint violation (less violation dominates)
    3. Objective values (Pareto dominance)
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize constraint domination.

        Args:
            tolerance: Feasibility tolerance
        """
        self.tolerance = tolerance

    def dominates(
        self,
        x1: np.ndarray,
        obj1: np.ndarray,
        violation1: float,
        x2: np.ndarray,
        obj2: np.ndarray,
        violation2: float
    ) -> bool:
        """
        Check if solution 1 constraint-dominates solution 2.

        Args:
            x1, x2: Decision variables
            obj1, obj2: Objective values
            violation1, violation2: Constraint violations

        Returns:
            True if x1 dominates x2
        """
        # Both feasible - use Pareto dominance
        if violation1 <= self.tolerance and violation2 <= self.tolerance:
            return self._pareto_dominates(obj1, obj2)

        # x1 feasible, x2 infeasible - x1 dominates
        if violation1 <= self.tolerance and violation2 > self.tolerance:
            return True

        # x1 infeasible, x2 feasible - x2 dominates
        if violation1 > self.tolerance and violation2 <= self.tolerance:
            return False

        # Both infeasible - less violation dominates
        return violation1 < violation2

    def _pareto_dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check Pareto dominance (minimization)."""
        not_worse = np.all(obj1 <= obj2)
        strictly_better = np.any(obj1 < obj2)
        return not_worse and strictly_better


class EpsilonConstraintMethod:
    """
    ε-constraint method for handling constraints.

    Allows controlled constraint violation up to threshold ε,
    which is gradually reduced over time.
    """

    def __init__(
        self,
        initial_epsilon: float = 0.1,
        reduction_rate: float = 0.95,
        min_epsilon: float = 1e-6
    ):
        """
        Initialize ε-constraint method.

        Args:
            initial_epsilon: Initial constraint violation threshold
            reduction_rate: Rate of epsilon reduction per generation
            min_epsilon: Minimum epsilon value
        """
        self.initial_epsilon = initial_epsilon
        self.reduction_rate = reduction_rate
        self.min_epsilon = min_epsilon
        self.current_epsilon = initial_epsilon

    def update(self, generation: int):
        """Update epsilon value based on generation."""
        self.current_epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.reduction_rate ** generation)
        )

    def is_feasible(self, violation: float) -> bool:
        """Check if solution is feasible under current epsilon."""
        return violation <= self.current_epsilon

    def compare(
        self,
        obj1: float,
        violation1: float,
        obj2: float,
        violation2: float
    ) -> int:
        """
        Compare two solutions under ε-constraint.

        Returns:
            1 if solution 1 is better
            -1 if solution 2 is better
            0 if equivalent
        """
        feasible1 = self.is_feasible(violation1)
        feasible2 = self.is_feasible(violation2)

        # Both ε-feasible - compare objectives
        if feasible1 and feasible2:
            if obj1 < obj2:
                return 1
            elif obj2 < obj1:
                return -1
            else:
                return 0

        # Only one ε-feasible
        if feasible1 and not feasible2:
            return 1
        if feasible2 and not feasible1:
            return -1

        # Both ε-infeasible - compare violations
        if violation1 < violation2:
            return 1
        elif violation2 < violation1:
            return -1
        else:
            return 0


class StochasticRanking:
    """
    Stochastic ranking for constraint handling.

    Balances objective and constraint violations using
    probabilistic comparison.
    """

    def __init__(self, probability_feasibility: float = 0.45):
        """
        Initialize stochastic ranking.

        Args:
            probability_feasibility: Probability of using feasibility
                                    for comparison (Pf)
        """
        self.pf = probability_feasibility

    def rank(
        self,
        objectives: np.ndarray,
        violations: np.ndarray
    ) -> np.ndarray:
        """
        Rank solutions using stochastic ranking.

        Args:
            objectives: Array of objective values
            violations: Array of constraint violations

        Returns:
            Indices sorted by stochastic rank
        """
        n = len(objectives)
        indices = np.arange(n)

        # Bubble sort with stochastic comparison
        for i in range(n):
            swapped = False
            for j in range(n - 1):
                idx1, idx2 = indices[j], indices[j + 1]

                # Determine if swap needed
                if self._should_swap(
                    objectives[idx1], violations[idx1],
                    objectives[idx2], violations[idx2]
                ):
                    indices[j], indices[j + 1] = idx2, idx1
                    swapped = True

            if not swapped:
                break

        return indices

    def _should_swap(
        self,
        obj1: float,
        viol1: float,
        obj2: float,
        viol2: float
    ) -> bool:
        """
        Determine if solution 1 should be swapped with solution 2.

        Uses stochastic comparison based on Pf.
        """
        # Both feasible - compare objectives
        if viol1 == 0 and viol2 == 0:
            return obj1 > obj2

        # Use probability for comparison
        if np.random.random() < self.pf:
            # Compare by constraint violation
            return viol1 > viol2
        else:
            # Compare by objective
            return obj1 > obj2