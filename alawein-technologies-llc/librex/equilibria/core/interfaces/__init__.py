"""Universal optimization interfaces"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class StandardizedProblem:
    """Standardized optimization problem representation"""

    dimension: int
    objective_matrix: Optional[np.ndarray] = None
    objective_function: Optional[Callable] = None
    constraint_matrix: Optional[np.ndarray] = None
    problem_metadata: Optional[Dict[str, Any]] = None


@dataclass
class StandardizedSolution:
    """Standardized optimization solution representation"""

    vector: np.ndarray
    objective_value: float
    is_valid: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Validation result for optimization solution"""

    is_valid: bool
    constraint_violations: List[str]
    violation_magnitudes: List[float]


class UniversalOptimizationInterface:
    """Base interface for domain adapters"""

    def __init__(self, domain_metadata: Dict[str, Any]):
        self.domain_metadata = domain_metadata

    def encode_problem(self, instance: Any) -> StandardizedProblem:
        """Convert domain-specific problem to standardized format"""
        raise NotImplementedError

    def decode_solution(self, solution: StandardizedSolution) -> Any:
        """Convert standardized solution to domain-specific format"""
        raise NotImplementedError

    def validate_solution(self, solution: Any) -> ValidationResult:
        """Validate a domain-specific solution"""
        raise NotImplementedError

    def compute_objective(self, solution: Any) -> float:
        """Compute objective value for a solution"""
        raise NotImplementedError


__all__ = [
    "StandardizedProblem",
    "StandardizedSolution",
    "ValidationResult",
    "UniversalOptimizationInterface",
]
