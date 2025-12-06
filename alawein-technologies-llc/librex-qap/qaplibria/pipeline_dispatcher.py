"""
Pipeline dispatcher - Refactored method orchestration.

Eliminates repetitive if-statements by using a registry-based dispatch pattern.
This makes adding new methods trivial and improves maintainability.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from Librex.QAP.logging_config import get_logger
from Librex.QAP.methods import baselines, novel


@dataclass
class MethodRegistry:
    """Registry of available methods with metadata."""

    name: str
    func: Callable
    requires_problem_matrices: bool = False  # True if needs (flow, distance)
    cleanup_after: bool = False  # True if needs Sinkhorn projection after


class PipelineDispatcher:
    """
    Registry-based method dispatcher for QAP pipeline.

    Replaces repetitive if-statements with clean, extensible pattern.
    """

    def __init__(self):
        """Initialize method registry."""
        self.logger = get_logger()
        self._registry: Dict[str, MethodRegistry] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Build method registry from available methods."""
        # Novel methods
        novel_methods = [
            MethodRegistry(
                "spectral_preconditioning",
                novel.apply_spectral_preconditioning,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "multi_scale_gradient_flow",
                novel.apply_multi_scale_gradient_flow,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "stochastic_gradient_variant",
                novel.apply_stochastic_gradient_variant,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "memory_efficient",
                novel.apply_memory_efficient_computation,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "reverse_time_saddle_escape",
                novel.apply_reverse_time_saddle_escape,
                requires_problem_matrices=True,
                cleanup_after=True,
            ),
            MethodRegistry(
                "constrained_step",
                novel.apply_constrained_step,
                requires_problem_matrices=False,
                cleanup_after=False,  # Returns SinkhornResult
            ),
            MethodRegistry(
                "adaptive_momentum",
                novel.apply_adaptive_momentum,
                requires_problem_matrices=False,
                cleanup_after=True,
            ),
            MethodRegistry(
                "hybrid_continuous_discrete",
                novel.apply_hybrid_continuous_discrete,
                requires_problem_matrices=False,
                cleanup_after=True,
            ),
            MethodRegistry(
                "Librex_tracking",
                novel.track_Librex_metrics,
                requires_problem_matrices=False,
            ),
        ]

        # Baseline methods
        baseline_methods = [
            MethodRegistry(
                "sinkhorn_projection",
                baselines.apply_sinkhorn_projection,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "hungarian_rounding",
                baselines.apply_hungarian_rounding,
                requires_problem_matrices=False,
            ),
            MethodRegistry(
                "two_opt_local_search",
                baselines.apply_two_opt_local_search,
                requires_problem_matrices=True,
            ),
        ]

        for method in novel_methods + baseline_methods:
            self._registry[method.name] = method

    def register_method(self, registry: MethodRegistry) -> None:
        """
        Register a new method.

        Parameters
        ----------
        registry : MethodRegistry
            Method registry entry
        """
        self._registry[registry.name] = registry
        self.logger.info(f"Registered method: {registry.name}")

    def is_available(self, method_name: str) -> bool:
        """Check if method is registered."""
        return method_name in self._registry

    def get_enabled_methods(self, config: Dict[str, bool]) -> List[str]:
        """
        Get list of enabled methods from config.

        Parameters
        ----------
        config : dict
            Configuration flags like {'method_name': True/False}

        Returns
        -------
        enabled : list of str
            Names of enabled methods
        """
        return [name for name, enabled in config.items() if enabled and self.is_available(name)]

    def apply_method(
        self,
        method_name: str,
        X: Any,
        flow: Any = None,
        distance: Any = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Apply a registered method.

        Parameters
        ----------
        method_name : str
            Method to apply
        X : array-like
            Current solution state
        flow : array-like, optional
            Problem flow matrix (needed for some methods)
        distance : array-like, optional
            Problem distance matrix (needed for some methods)
        **kwargs
            Additional arguments for the method

        Returns
        -------
        result : array-like or tuple
            Method result
        log : MethodLog
            Execution log
        """
        if method_name not in self._registry:
            raise ValueError(f"Method not registered: {method_name}")

        registry = self._registry[method_name]

        if registry.requires_problem_matrices:
            if flow is None or distance is None:
                raise ValueError(f"Method '{method_name}' requires flow and distance matrices")
            result, log = registry.func(flow, distance, X, **kwargs)
        else:
            result, log = registry.func(X, **kwargs)

        self.logger.debug(f"Applied method: {method_name}")

        return result, log

    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """
        Get information about a method.

        Parameters
        ----------
        method_name : str
            Method name

        Returns
        -------
        info : dict
            Method metadata
        """
        if method_name not in self._registry:
            raise ValueError(f"Method not registered: {method_name}")

        registry = self._registry[method_name]

        return {
            "name": registry.name,
            "requires_problem_matrices": registry.requires_problem_matrices,
            "cleanup_after": registry.cleanup_after,
        }

    def list_methods(self, category: str = "all") -> List[str]:
        """
        List available methods.

        Parameters
        ----------
        category : str, default='all'
            Filter by category: 'novel', 'baseline', or 'all'

        Returns
        -------
        methods : list of str
            Method names
        """
        all_methods = list(self._registry.keys())

        if category == "all":
            return sorted(all_methods)
        if category == "novel":
            # Novel methods names contain these keywords
            novel_keywords = [
                "spectral_preconditioning",
                "multi_scale",
                "stochastic_gradient",
                "memory_efficient",
                "reverse_time",
                "constrained_step",
                "adaptive_momentum",
                "hybrid",
                "Librex",
            ]
            return sorted([m for m in all_methods if any(kw in m for kw in novel_keywords)])
        if category == "baseline":
            baseline_methods = ["sinkhorn", "hungarian", "two_opt"]
            return sorted([m for m in all_methods if any(b in m for b in baseline_methods)])
        raise ValueError(f"Unknown category: {category}")

    def print_registry(self) -> None:
        """Print registry of available methods."""
        print("\n" + "=" * 70)
        print("METHOD REGISTRY")
        print("=" * 70)

        novel_methods = self.list_methods("novel")
        baseline_methods = self.list_methods("baseline")

        print(f"\nNOVEL METHODS ({len(novel_methods)}):")
        for name in novel_methods:
            info = self.get_method_info(name)
            needs_matrices = "✓" if info["requires_problem_matrices"] else "✗"
            needs_cleanup = "✓" if info["cleanup_after"] else "✗"
            print(f"  • {name:<35} matrices:{needs_matrices} cleanup:{needs_cleanup}")

        print(f"\nBASELINE METHODS ({len(baseline_methods)}):")
        for name in baseline_methods:
            info = self.get_method_info(name)
            needs_matrices = "✓" if info["requires_problem_matrices"] else "✗"
            needs_cleanup = "✓" if info["cleanup_after"] else "✗"
            print(f"  • {name:<35} matrices:{needs_matrices} cleanup:{needs_cleanup}")

        print("=" * 70 + "\n")


# Global dispatcher instance
_global_dispatcher: PipelineDispatcher = None


def get_dispatcher() -> PipelineDispatcher:
    """Get global pipeline dispatcher."""
    global _global_dispatcher

    if _global_dispatcher is None:
        _global_dispatcher = PipelineDispatcher()

    return _global_dispatcher


__all__ = [
    "MethodRegistry",
    "PipelineDispatcher",
    "get_dispatcher",
]
