"""
Advanced constraint handling for optimization problems.

This module provides comprehensive constraint handling capabilities including:
- Various constraint types (equality, inequality, box, linear, nonlinear)
- Multiple constraint handling methods (penalty, repair, domination)
- Feasibility preservation techniques
- Adaptive penalty strategies
"""

from .constraints import (
    Constraint,
    EqualityConstraint,
    InequalityConstraint,
    BoxConstraint,
    LinearConstraint,
    NonlinearConstraint,
    ConstraintSet,
)
from .handlers import (
    PenaltyMethod,
    StaticPenalty,
    DynamicPenalty,
    AdaptivePenalty,
    DeathPenalty,
    RepairOperator,
    ProjectionRepair,
    FeasibilityRestoration,
    ConstraintDomination,
    EpsilonConstraintMethod,
    StochasticRanking,
)
from .feasibility import (
    FeasibilityPreservingCrossover,
    FeasibilityPreservingMutation,
    feasibility_check,
    project_to_feasible_region,
    random_feasible_solution,
)

__all__ = [
    # Constraint types
    "Constraint",
    "EqualityConstraint",
    "InequalityConstraint",
    "BoxConstraint",
    "LinearConstraint",
    "NonlinearConstraint",
    "ConstraintSet",
    # Handlers
    "PenaltyMethod",
    "StaticPenalty",
    "DynamicPenalty",
    "AdaptivePenalty",
    "DeathPenalty",
    "RepairOperator",
    "ProjectionRepair",
    "FeasibilityRestoration",
    "ConstraintDomination",
    "EpsilonConstraintMethod",
    "StochasticRanking",
    # Feasibility
    "FeasibilityPreservingCrossover",
    "FeasibilityPreservingMutation",
    "feasibility_check",
    "project_to_feasible_region",
    "random_feasible_solution",
]