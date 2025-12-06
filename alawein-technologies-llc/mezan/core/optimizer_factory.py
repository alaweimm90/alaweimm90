"""
Optimizer Factory - Dynamic Solver Selection and Instantiation

This module provides the factory pattern for creating MEZAN optimizers
based on problem type, feature flags, and configuration.
"""

import logging
from typing import Dict, Any, Optional, Type
from enum import Enum

from .optimizer_interface import (
    OptimizerInterface,
    HeuristicFallbackOptimizer,
    ProblemType,
    OptimizationProblem,
)

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """Feature flags for gradual Libria solver rollout"""

    # Individual solver flags
    ENABLE_QAP_LIBRIA = "enable_qap_libria"
    ENABLE_FLOW_LIBRIA = "enable_flow_libria"
    ENABLE_ALLOC_LIBRIA = "enable_alloc_libria"
    ENABLE_GRAPH_LIBRIA = "enable_graph_libria"
    ENABLE_DUAL_LIBRIA = "enable_dual_libria"
    ENABLE_EVO_LIBRIA = "enable_evo_libria"
    ENABLE_META_LIBRIA = "enable_meta_libria"

    # Master flags
    ENABLE_ALL_LIBRIA = "enable_all_libria"  # Enable all Libria solvers
    FORCE_HEURISTIC = "force_heuristic"  # Force heuristic fallback (for testing)

    # GPU acceleration
    ENABLE_GPU = "enable_gpu"


class OptimizerFactory:
    """
    Factory for creating MEZAN optimizers

    Handles:
    - Feature flag checking
    - Solver registration and instantiation
    - Fallback to heuristics
    - GPU enablement
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize factory

        Args:
            config: Configuration dictionary with feature flags and solver configs
        """
        self.config = config or {}
        self.feature_flags = self.config.get("feature_flags", {})
        self.solver_registry: Dict[ProblemType, Type[OptimizerInterface]] = {}
        self.solver_configs: Dict[ProblemType, Dict[str, Any]] = {}

        # Register default heuristic fallback
        self._register_heuristic_fallback()

        # Attempt to register Libria solvers (graceful degradation if not available)
        self._register_libria_solvers()

    def _register_heuristic_fallback(self) -> None:
        """Register heuristic fallback for all problem types"""
        for problem_type in ProblemType:
            if problem_type not in self.solver_registry:
                self.solver_registry[problem_type] = HeuristicFallbackOptimizer
                logger.info(
                    f"Registered HeuristicFallbackOptimizer for {problem_type.value}"
                )

    def _register_libria_solvers(self) -> None:
        """
        Attempt to register Libria solvers if available and enabled

        This uses dynamic imports to gracefully degrade if solvers not installed.
        """
        # Librex.QAP
        if self._is_enabled(FeatureFlag.ENABLE_QAP_LIBRIA):
            try:
                from libria_qap import Librex.QAPSolver

                self.solver_registry[ProblemType.QAP] = Librex.QAPSolver
                logger.info("Registered Librex.QAPSolver for QAP problems")
            except ImportError:
                logger.warning(
                    "Librex.QAP enabled but not installed, using heuristic fallback"
                )

        # Librex.Flow
        if self._is_enabled(FeatureFlag.ENABLE_FLOW_LIBRIA):
            try:
                from libria_flow import Librex.FlowSolver

                self.solver_registry[ProblemType.FLOW] = Librex.FlowSolver
                logger.info("Registered Librex.FlowSolver for FLOW problems")
            except ImportError:
                logger.warning(
                    "Librex.Flow enabled but not installed, using heuristic fallback"
                )

        # Librex.Alloc
        if self._is_enabled(FeatureFlag.ENABLE_ALLOC_LIBRIA):
            try:
                from libria_alloc import Librex.AllocSolver

                self.solver_registry[ProblemType.ALLOC] = Librex.AllocSolver
                logger.info("Registered Librex.AllocSolver for ALLOC problems")
            except ImportError:
                logger.warning(
                    "Librex.Alloc enabled but not installed, using heuristic fallback"
                )

        # Librex.Graph
        if self._is_enabled(FeatureFlag.ENABLE_GRAPH_LIBRIA):
            try:
                from libria_graph import Librex.GraphSolver

                self.solver_registry[ProblemType.GRAPH] = Librex.GraphSolver
                logger.info("Registered Librex.GraphSolver for GRAPH problems")
            except ImportError:
                logger.warning(
                    "Librex.Graph enabled but not installed, using heuristic fallback"
                )

        # Librex.Dual
        if self._is_enabled(FeatureFlag.ENABLE_DUAL_LIBRIA):
            try:
                from libria_dual import Librex.DualSolver

                self.solver_registry[ProblemType.DUAL] = Librex.DualSolver
                logger.info("Registered Librex.DualSolver for DUAL problems")
            except ImportError:
                logger.warning(
                    "Librex.Dual enabled but not installed, using heuristic fallback"
                )

        # Librex.Evo
        if self._is_enabled(FeatureFlag.ENABLE_EVO_LIBRIA):
            try:
                from libria_evo import Librex.EvoSolver

                self.solver_registry[ProblemType.EVO] = Librex.EvoSolver
                logger.info("Registered Librex.EvoSolver for EVO problems")
            except ImportError:
                logger.warning(
                    "Librex.Evo enabled but not installed, using heuristic fallback"
                )

        # Librex.Meta
        if self._is_enabled(FeatureFlag.ENABLE_META_LIBRIA):
            try:
                from libria_meta import Librex.MetaSolver

                self.solver_registry[ProblemType.META] = Librex.MetaSolver
                logger.info("Registered Librex.MetaSolver for META problems")
            except ImportError:
                logger.warning(
                    "Librex.Meta enabled but not installed, using heuristic fallback"
                )

    def _is_enabled(self, flag: FeatureFlag) -> bool:
        """
        Check if a feature flag is enabled

        Args:
            flag: FeatureFlag enum value

        Returns:
            True if enabled, False otherwise
        """
        # Check for force heuristic (overrides all)
        if self.feature_flags.get(FeatureFlag.FORCE_HEURISTIC.value, False):
            return False

        # Check for enable all
        if self.feature_flags.get(FeatureFlag.ENABLE_ALL_LIBRIA.value, False):
            return True

        # Check specific flag
        return self.feature_flags.get(flag.value, False)

    def create_optimizer(
        self,
        problem: OptimizationProblem,
        enable_gpu: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> OptimizerInterface:
        """
        Create an optimizer for the given problem

        Args:
            problem: The optimization problem instance
            enable_gpu: Override GPU setting (None = use config default)
            timeout: Override timeout setting (None = use config default)

        Returns:
            OptimizerInterface instance ready to solve the problem
        """
        problem_type = problem.problem_type

        # Get solver class from registry
        solver_class = self.solver_registry.get(problem_type, HeuristicFallbackOptimizer)

        # Determine GPU setting
        if enable_gpu is None:
            enable_gpu = self._is_enabled(FeatureFlag.ENABLE_GPU)

        # Get solver-specific config
        solver_config = self.solver_configs.get(problem_type, {})

        # Get default timeout from config
        if timeout is None:
            timeout = self.config.get("default_timeout", 60.0)  # 60 seconds default

        # Instantiate solver
        try:
            optimizer = solver_class(
                config=solver_config, enable_gpu=enable_gpu, timeout=timeout
            )
            logger.info(
                f"Created {solver_class.__name__} for {problem_type.value} "
                f"(GPU={enable_gpu}, timeout={timeout}s)"
            )
            return optimizer
        except Exception as e:
            logger.error(
                f"Failed to create {solver_class.__name__}: {e}, "
                "falling back to heuristic"
            )
            return HeuristicFallbackOptimizer(
                config=solver_config, enable_gpu=False, timeout=timeout
            )

    def register_solver(
        self,
        problem_type: ProblemType,
        solver_class: Type[OptimizerInterface],
        solver_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a custom solver for a problem type

        Args:
            problem_type: The problem type this solver handles
            solver_class: The OptimizerInterface subclass
            solver_config: Solver-specific configuration
        """
        self.solver_registry[problem_type] = solver_class
        if solver_config:
            self.solver_configs[problem_type] = solver_config
        logger.info(f"Registered {solver_class.__name__} for {problem_type.value}")

    def get_registered_solvers(self) -> Dict[ProblemType, str]:
        """
        Get all registered solvers

        Returns:
            Dictionary mapping ProblemType to solver class name
        """
        return {pt: solver.__name__ for pt, solver in self.solver_registry.items()}

    def update_feature_flags(self, flags: Dict[str, bool]) -> None:
        """
        Update feature flags dynamically

        Args:
            flags: Dictionary of flag names to boolean values
        """
        self.feature_flags.update(flags)
        logger.info(f"Updated feature flags: {flags}")

        # Re-register solvers with new flags
        self._register_libria_solvers()


# Singleton instance for global access
_global_factory: Optional[OptimizerFactory] = None


def get_optimizer_factory(
    config: Optional[Dict[str, Any]] = None,
) -> OptimizerFactory:
    """
    Get the global optimizer factory instance

    Args:
        config: Configuration (only used if factory not yet created)

    Returns:
        OptimizerFactory singleton instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = OptimizerFactory(config=config)
    return _global_factory


def reset_optimizer_factory() -> None:
    """Reset the global factory instance (useful for testing)"""
    global _global_factory
    _global_factory = None
