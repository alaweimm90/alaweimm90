"""
Method metadata and classification system for Librex.QAP.

Provides runtime access to method properties (novelty, complexity, etc.)
without cluttering method implementations.

⚠️ **IMPORTANT**: Performance metrics (improvement_percent, speedup_factor, success_rate)
are ASPIRATIONAL TARGETS based on theoretical analysis and preliminary testing.
These values REQUIRE experimental validation against QAPLIB benchmarks.

Current Status: VALIDATION IN PROGRESS (Phase 1.5)
See PHASE_1_VALIDATION_REPORT.md for audit findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional


class NoveltyLevel(IntEnum):
    """Novelty classification for optimization methods."""

    BASELINE = 1  # Standard, well-known methods
    ADAPTED = 2  # Adapted for QAP, minor contributions
    NOVEL = 3  # Fully novel, Librex.QAP-original


@dataclass
class MethodMetadata:
    """Complete metadata for an optimization method."""

    name: str  # Display name
    function_name: str  # Python function name
    novelty_level: NoveltyLevel  # Baseline (1), Adapted (2), or Novel (3)
    origin: str  # Citation or "Librex.QAP Original"
    complexity: str  # Time complexity (e.g., "O(n² log n)")
    description: str  # Brief description
    tags: List[str]  # Category tags (e.g., ["preconditioning", "acceleration"])
    improvement_percent: Optional[float] = None  # % improvement vs baseline
    success_rate: Optional[float] = None  # Success rate if applicable
    speedup_factor: Optional[float] = None  # Speedup multiplier if applicable

    @property
    def stars(self) -> str:
        """Return star rating: ⭐ (baseline), ⭐⭐ (adapted), ⭐⭐⭐ (novel)."""
        return "⭐" * self.novelty_level

    @property
    def type_label(self) -> str:
        """Return type label: • baseline, ◆ adapted, ★ novel."""
        if self.novelty_level == NoveltyLevel.BASELINE:
            return "•"  # Baseline
        if self.novelty_level == NoveltyLevel.ADAPTED:
            return "◆"  # Adapted
        return "★"  # Novel

    def __str__(self) -> str:
        """Format for display."""
        return f"{self.stars} {self.name} ({self.origin})"


# ============================================================================
# METHOD REGISTRY - All 19 Current Methods (7 Novel + 3 Adapted + 9 Baseline)
# ============================================================================

_METHODS_REGISTRY: Dict[str, MethodMetadata] = {
    # ========================================================================
    # NOVEL METHODS (7) + ADAPTED METHODS (3) - Librex.QAP Contributions
    # ========================================================================
    "apply_fft_laplace_preconditioning": MethodMetadata(
        name="FFT-Laplace Preconditioning",
        function_name="apply_fft_laplace_preconditioning",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n² log n)",
        description="Laplacian smoothing in frequency domain for fast convergence",
        tags=["preconditioning", "acceleration", "fft", "novel"],
        speedup_factor=100.0,
        improvement_percent=95.0,
    ),
    "apply_reverse_time_saddle_escape": MethodMetadata(
        name="Reverse-Time Saddle Escape",
        function_name="apply_reverse_time_saddle_escape",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n²)",
        description="Escape local minima via reverse-time integration on unstable manifolds",
        tags=["saddle-escape", "global-search", "novel"],
        success_rate=0.90,
        improvement_percent=30.0,
    ),
    "apply_adaptive_momentum": MethodMetadata(
        name="Adaptive Momentum",
        function_name="apply_adaptive_momentum",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n²)",
        description="QAP-specific dynamic velocity accumulation for accelerated convergence",
        tags=["acceleration", "momentum", "novel"],
        improvement_percent=15.0,
    ),
    "apply_multi_scale_gradient_flow": MethodMetadata(
        name="Multi-Scale Gradient Flow",
        function_name="apply_multi_scale_gradient_flow",
        novelty_level=NoveltyLevel.ADAPTED,
        origin="Adapted from multigrid methods",
        complexity="O(n² log n)",
        description="Hierarchical coarse-to-fine optimization across multiple scales",
        tags=["acceleration", "hierarchical", "adapted"],
        improvement_percent=20.0,
    ),
    "apply_spectral_preconditioning": MethodMetadata(
        name="Spectral Preconditioning",
        function_name="apply_spectral_preconditioning",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n²)",
        description="Eigenvalue-based matrix preconditioning for improved initialization",
        tags=["preconditioning", "spectral", "novel"],
        improvement_percent=12.0,
    ),
    "apply_stochastic_gradient_variant": MethodMetadata(
        name="Stochastic Gradient Variant",
        function_name="apply_stochastic_gradient_variant",
        novelty_level=NoveltyLevel.ADAPTED,
        origin="Adapted from SGD literature",
        complexity="O(n²)",
        description="Controlled noise injection to escape poor local minima",
        tags=["stochasticity", "exploration", "adapted"],
        improvement_percent=8.0,
    ),
    "apply_constrained_step": MethodMetadata(
        name="Constrained Step",
        function_name="apply_constrained_step",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n² log n)",
        description="Enforced doubly-stochastic constraint satisfaction via Sinkhorn",
        tags=["constraint-handling", "feasibility", "novel"],
        improvement_percent=5.0,
    ),
    "apply_hybrid_continuous_discrete": MethodMetadata(
        name="Hybrid Continuous-Discrete",
        function_name="apply_hybrid_continuous_discrete",
        novelty_level=NoveltyLevel.NOVEL,
        origin="Alawein (2025)",
        complexity="O(n²)",
        description="Interpolates between continuous relaxation and discrete permutation",
        tags=["hybrid", "rounding", "novel"],
        improvement_percent=18.0,
    ),
    "apply_parallel_processing": MethodMetadata(
        name="Parallel Processing",
        function_name="apply_parallel_processing",
        novelty_level=NoveltyLevel.ADAPTED,
        origin="Adapted from multi-start methods",
        complexity="O(n² × num_trials)",
        description="Multi-trial exploration with parallel execution capability",
        tags=["parallelization", "multi-start", "adapted"],
        improvement_percent=25.0,
    ),
    "apply_memory_efficient_computation": MethodMetadata(
        name="Memory-Efficient Algorithms",
        function_name="apply_memory_efficient_computation",
        novelty_level=NoveltyLevel.ADAPTED,
        origin="Adapted from precision reduction techniques",
        complexity="O(n²)",
        description="Reduced-precision computation for large instances (n > 256)",
        tags=["memory-optimization", "scalability", "adapted"],
        improvement_percent=3.0,
    ),
    # NOTE: track_Librex_metrics is excluded from registry as it has a different
    # API signature (returns only MethodLog, not tuple) and is an analysis utility
    # rather than a transformation method.
    # ========================================================================
    # BASELINE METHODS (9) - Standard/Classical Techniques
    # ========================================================================
    "apply_sinkhorn_projection": MethodMetadata(
        name="Sinkhorn Projection",
        function_name="apply_sinkhorn_projection",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Sinkhorn & Knopp (1967)",
        complexity="O(n²)",
        description="Doubly-stochastic matrix projection via iterative scaling",
        tags=["projection", "constraint-handling", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_hungarian_rounding": MethodMetadata(
        name="Hungarian Rounding",
        function_name="apply_hungarian_rounding",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Kuhn-Munkres (1955)",
        complexity="O(n³)",
        description="Optimal discrete rounding via assignment problem solution",
        tags=["rounding", "optimization", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_two_opt_local_search": MethodMetadata(
        name="2-Opt Local Search",
        function_name="apply_two_opt_local_search",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Croes (1958)",
        complexity="O(n²) per iteration",
        description="Pairwise swap local search to improve permutation",
        tags=["local-search", "refinement", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_basic_gradient_descent": MethodMetadata(
        name="Basic Gradient Descent",
        function_name="apply_basic_gradient_descent",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Cauchy (1847)",
        complexity="O(n²) per iteration",
        description="Steepest descent on Birkhoff polytope without acceleration",
        tags=["optimization", "gradient", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_shannon_entropy_regularization": MethodMetadata(
        name="Shannon Entropy Regularization",
        function_name="apply_shannon_entropy_regularization",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Shannon (1948)",
        complexity="O(n²)",
        description="Entropy-based regularization to smooth objective landscape",
        tags=["regularization", "entropy", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_probabilistic_rounding": MethodMetadata(
        name="Probabilistic Rounding",
        function_name="apply_probabilistic_rounding",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Standard stochastic method",
        complexity="O(n²)",
        description="Permutation sampling from Boltzmann distribution",
        tags=["rounding", "stochastic", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_iterative_rounding": MethodMetadata(
        name="Iterative Rounding",
        function_name="apply_iterative_rounding",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Standard combinatorial method",
        complexity="O(n³)",
        description="Progressive discretization by incremental assignment",
        tags=["rounding", "combinatorial", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_continuation_method": MethodMetadata(
        name="Continuation Method",
        function_name="apply_continuation_method",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Homotopy continuation literature",
        complexity="O(n² × num_schedules)",
        description="Homotopy parameter scheduling to escape shallow minima",
        tags=["continuation", "homotopy", "baseline"],
        improvement_percent=0.0,
    ),
    "apply_adaptive_lambda_adjustment": MethodMetadata(
        name="Adaptive Lambda Adjustment",
        function_name="apply_adaptive_lambda_adjustment",
        novelty_level=NoveltyLevel.BASELINE,
        origin="Augmented Lagrangian methods",
        complexity="O(n²)",
        description="Dynamic constraint enforcement via Lagrange multiplier adjustment",
        tags=["constraint-handling", "augmented-lagrangian", "baseline"],
        improvement_percent=0.0,
    ),
}


# ============================================================================
# PUBLIC API
# ============================================================================


def get_method_metadata(function_name: str) -> Optional[MethodMetadata]:
    """
    Retrieve metadata for a method by function name.

    Args:
        function_name: Name of the method function (e.g., "apply_fft_laplace_preconditioning")

    Returns:
        MethodMetadata or None if not found

    Example:
        >>> meta = get_method_metadata("apply_fft_laplace_preconditioning")
        >>> print(meta.stars)  # "⭐⭐⭐"
        >>> print(meta.speedup_factor)  # 100.0
    """
    return _METHODS_REGISTRY.get(function_name)


def get_all_methods() -> List[MethodMetadata]:
    """
    Get all registered methods.

    Returns:
        List of all MethodMetadata objects
    """
    return list(_METHODS_REGISTRY.values())


def get_novel_methods() -> List[MethodMetadata]:
    """
    Get all novel (⭐⭐⭐) methods.

    Returns:
        List of novel MethodMetadata objects
    """
    return [m for m in _METHODS_REGISTRY.values() if m.novelty_level == NoveltyLevel.NOVEL]


def get_adapted_methods() -> List[MethodMetadata]:
    """
    Get all adapted (⭐⭐) methods.

    Returns:
        List of adapted MethodMetadata objects
    """
    return [m for m in _METHODS_REGISTRY.values() if m.novelty_level == NoveltyLevel.ADAPTED]


def get_baseline_methods() -> List[MethodMetadata]:
    """
    Get all baseline (⭐) methods.

    Returns:
        List of baseline MethodMetadata objects
    """
    return [m for m in _METHODS_REGISTRY.values() if m.novelty_level == NoveltyLevel.BASELINE]


def get_methods_by_tag(tag: str) -> List[MethodMetadata]:
    """
    Get all methods with a specific tag.

    Args:
        tag: Tag name (e.g., "novel", "preconditioning", "acceleration")

    Returns:
        List of matching MethodMetadata objects

    Example:
        >>> acceleration = get_methods_by_tag("acceleration")
        >>> for m in acceleration:
        ...     print(f"{m.stars} {m.name}")
    """
    return [m for m in _METHODS_REGISTRY.values() if tag in m.tags]


def print_methods_table() -> str:
    """
    Generate a formatted table of all methods.

    Returns:
        Formatted table string for display or documentation
    """
    lines = []
    lines.append("=" * 120)
    lines.append(f"{'Category':<12} {'Method':<40} {'Complexity':<20} {'Origin':<30}")
    lines.append("=" * 120)

    # Novel methods
    novel = get_novel_methods()
    if novel:
        for m in novel:
            lines.append(f"{m.stars:<12} {m.name:<40} {m.complexity:<20} {m.origin:<30}")

    # Adapted methods
    adapted = get_adapted_methods()
    if adapted:
        for m in adapted:
            lines.append(f"{m.stars:<12} {m.name:<40} {m.complexity:<20} {m.origin:<30}")

    # Baseline methods
    baseline = get_baseline_methods()
    if baseline:
        for m in baseline:
            lines.append(f"{m.stars:<12} {m.name:<40} {m.complexity:<20} {m.origin:<30}")

    lines.append("=" * 120)
    return "\n".join(lines)


__all__ = [
    "NoveltyLevel",
    "MethodMetadata",
    "get_method_metadata",
    "get_all_methods",
    "get_novel_methods",
    "get_adapted_methods",
    "get_baseline_methods",
    "get_methods_by_tag",
    "print_methods_table",
]
