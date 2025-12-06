"""
Refutation Strategies

Five orthogonal strategies for hypothesis falsification:
1. Logical Contradiction - Find internal contradictions
2. Empirical Counter-Examples - Search for counter-examples in data
3. Analogical Falsification - Find similar claims that failed
4. Boundary Violations - Test at extreme parameter values
5. Mechanism Implausibility - Check causal mechanism plausibility
"""

from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.strategies.logical_contradiction import LogicalContradictionStrategy
from self_refutation.strategies.empirical_counter_example import EmpiricalCounterExampleStrategy
from self_refutation.strategies.analogical_falsification import AnalogicalFalsificationStrategy
from self_refutation.strategies.boundary_violation import BoundaryViolationStrategy
from self_refutation.strategies.mechanism_implausibility import MechanismImplausibilityStrategy

__all__ = [
    "BaseRefutationStrategy",
    "StrategyError",
    "LogicalContradictionStrategy",
    "EmpiricalCounterExampleStrategy",
    "AnalogicalFalsificationStrategy",
    "BoundaryViolationStrategy",
    "MechanismImplausibilityStrategy",
]
