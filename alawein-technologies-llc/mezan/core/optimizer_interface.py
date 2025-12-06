"""
Optimizer Interface - Abstract Base Class for MEZAN Optimization Solvers

This module defines the interface that all MEZAN optimization solvers
(Libria Suite) must implement. It provides a clean abstraction layer
between V4.0.0 infrastructure and optimization algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ProblemType(Enum):
    """Types of optimization problems supported by MEZAN"""

    QAP = "quadratic_assignment"  # Agent-task assignment with synergies
    FLOW = "workflow_routing"  # Confidence-aware workflow routing
    ALLOC = "resource_allocation"  # Budget-constrained resource allocation
    GRAPH = "network_topology"  # Agent communication network optimization
    DUAL = "adversarial_robust"  # Min-max robust optimization
    EVO = "multi_objective"  # Multi-objective evolutionary optimization
    META = "algorithm_selection"  # Meta-learning for solver selection


class SolverStatus(Enum):
    """Status of optimization solver execution"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    FAILED = "failed"
    FALLBACK = "fallback_used"


@dataclass
class OptimizationProblem:
    """
    Represents an optimization problem instance

    This is the input to any MEZAN optimizer.
    """

    problem_type: ProblemType
    data: Dict[str, Any]  # Problem-specific data (matrices, constraints, etc.)
    constraints: Optional[Dict[str, Any]] = None
    objectives: Optional[List[str]] = None  # For multi-objective problems
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate problem definition

        Returns:
            (is_valid, error_message)
        """
        if not self.data:
            return False, "Problem data cannot be empty"

        # Problem-type-specific validation
        if self.problem_type == ProblemType.QAP:
            required_keys = ["distance_matrix", "flow_matrix"]
            missing = [k for k in required_keys if k not in self.data]
            if missing:
                return False, f"QAP problem missing keys: {missing}"

            # Validate matrix dimensions
            dist = self.data.get("distance_matrix")
            flow = self.data.get("flow_matrix")
            if dist is not None and flow is not None:
                if len(dist) != len(flow):
                    return False, "Distance and flow matrices must have same dimension"

        elif self.problem_type == ProblemType.FLOW:
            required_keys = ["workflow_graph", "confidence_scores"]
            missing = [k for k in required_keys if k not in self.data]
            if missing:
                return False, f"FLOW problem missing keys: {missing}"

        elif self.problem_type == ProblemType.ALLOC:
            required_keys = ["resource_demands", "budget_constraint"]
            missing = [k for k in required_keys if k not in self.data]
            if missing:
                return False, f"ALLOC problem missing keys: {missing}"

        return True, None


@dataclass
class OptimizationResult:
    """
    Result from an optimization solver

    This is the output from any MEZAN optimizer.
    """

    status: SolverStatus
    solution: Optional[Any]  # Problem-specific solution format
    objective_value: Optional[float]  # Objective function value
    metadata: Dict[str, Any]  # Solver-specific metadata

    # Performance metrics
    computation_time: float  # Seconds
    iterations: Optional[int] = None
    convergence_info: Optional[Dict[str, Any]] = None

    # Quality metrics
    optimality_gap: Optional[float] = None  # Gap to known optimal (if available)
    improvement_over_baseline: Optional[float] = None  # % improvement

    def is_valid(self) -> bool:
        """Check if result is valid and usable"""
        return self.status in (SolverStatus.SUCCESS, SolverStatus.FALLBACK) and self.solution is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "solution": self.solution,
            "objective_value": self.objective_value,
            "computation_time": self.computation_time,
            "iterations": self.iterations,
            "optimality_gap": self.optimality_gap,
            "improvement_over_baseline": self.improvement_over_baseline,
            "metadata": self.metadata,
        }


class OptimizerInterface(ABC):
    """
    Abstract base class for all MEZAN optimization solvers

    All Libria Suite solvers must implement this interface.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Initialize optimizer

        Args:
            config: Solver-specific configuration
            enable_gpu: Whether to use GPU acceleration (if available)
            timeout: Maximum computation time in seconds (None = no limit)
        """
        self.config = config or {}
        self.enable_gpu = enable_gpu
        self.timeout = timeout
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize solver (load models, allocate resources, etc.)

        This is called once before first solve() call.
        """
        pass

    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve an optimization problem

        Args:
            problem: The optimization problem instance

        Returns:
            OptimizationResult with solution and metadata
        """
        pass

    @abstractmethod
    def get_problem_types(self) -> List[ProblemType]:
        """
        Get the problem types this solver can handle

        Returns:
            List of supported ProblemType enums
        """
        pass

    @abstractmethod
    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        """
        Estimate computational complexity for a problem

        Args:
            problem: The optimization problem instance

        Returns:
            Complexity estimate as string ("low", "medium", "high", "very_high")
        """
        pass

    def validate_problem(self, problem: OptimizationProblem) -> Tuple[bool, Optional[str]]:
        """
        Validate that this solver can handle the given problem

        Args:
            problem: The optimization problem instance

        Returns:
            (can_solve, error_message)
        """
        # Check problem type compatibility
        if problem.problem_type not in self.get_problem_types():
            return (
                False,
                f"Solver does not support {problem.problem_type.value} problems",
            )

        # Validate problem definition
        is_valid, error = problem.validate()
        if not is_valid:
            return False, error

        return True, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gpu={self.enable_gpu}, timeout={self.timeout})"


class HeuristicFallbackOptimizer(OptimizerInterface):
    """
    Simple heuristic optimizer used as fallback when Libria solvers fail or timeout

    This ensures MEZAN always has a working solution, even if suboptimal.
    """

    def initialize(self) -> None:
        """Initialize heuristic solver (no-op for simple heuristics)"""
        self._is_initialized = True

    def get_problem_types(self) -> List[ProblemType]:
        """Support all problem types with basic heuristics"""
        return list(ProblemType)

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        """Heuristics are always low complexity"""
        return "low"

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve using simple heuristics

        For QAP: Random assignment
        For FLOW: Shortest path
        For ALLOC: Equal distribution
        etc.
        """
        import time

        start_time = time.time()

        # Validate problem before attempting to solve
        is_valid, error = problem.validate()
        if not is_valid:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"solver": "HeuristicFallback", "error": error},
                computation_time=time.time() - start_time,
            )

        # Problem-type-specific heuristics
        try:
            if problem.problem_type == ProblemType.QAP:
                solution = self._solve_qap_heuristic(problem)
            elif problem.problem_type == ProblemType.FLOW:
                solution = self._solve_flow_heuristic(problem)
            elif problem.problem_type == ProblemType.ALLOC:
                solution = self._solve_alloc_heuristic(problem)
            else:
                # Default: random solution
                solution = {"type": "random", "value": None}
        except Exception as e:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"solver": "HeuristicFallback", "error": str(e)},
                computation_time=time.time() - start_time,
            )

        computation_time = time.time() - start_time

        return OptimizationResult(
            status=SolverStatus.FALLBACK,
            solution=solution,
            objective_value=None,  # Heuristics don't optimize
            metadata={"solver": "HeuristicFallback", "note": "Suboptimal solution"},
            computation_time=computation_time,
        )

    def _solve_qap_heuristic(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Simple QAP heuristic: random assignment"""
        n = len(problem.data["distance_matrix"])
        assignment = list(range(n))
        np.random.shuffle(assignment)
        return {"assignment": assignment, "method": "random"}

    def _solve_flow_heuristic(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Simple FLOW heuristic: highest confidence path"""
        return {"path": "greedy_highest_confidence", "method": "greedy"}

    def _solve_alloc_heuristic(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Simple ALLOC heuristic: equal distribution"""
        budget = problem.data.get("budget_constraint", 1.0)
        demands = problem.data.get("resource_demands", [])
        n = len(demands)
        allocation = [budget / n] * n if n > 0 else []
        return {"allocation": allocation, "method": "equal_distribution"}
