"""
Librex.QAP Notebook Library - Modular QAP Optimization Framework
===============================================================

A comprehensive framework for solving Quadratic Assignment Problems (QAP)
through continuous optimization on the Birkhoff polytope.

**Architecture**: Modular design with clear separation of concerns
- core: Pipeline orchestration and main execution
- methods: Novel (11) and baseline (11) optimization methods
- analysis: Visualization (15 functions) and tables (6 functions)
- utils: Core algorithms and utilities

**Author**: Meshal Alawein
**Version**: 2.0 (Refactored with CODEX minimal philosophy)
**Date**: 2025-10-16

**Quick Start**:
    from notebook_lib import QAPBenchmarkPipeline
    pipeline = QAPBenchmarkPipeline(data_dir, best_known)
    result = pipeline.run('had12')

**Visualization**:
    from notebook_lib import plot_fft_speedup, generate_performance_table
    plot_fft_speedup(vanilla_time=10, fft_time=1)
    generate_performance_table(results)
"""

import os
from pathlib import Path

# Read version from VERSION file
_version_file = Path(__file__).parent.parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text().strip()
else:
    __version__ = "0.1.0"  # Fallback version

__author__ = "Meshal Alawein"

# ============================================================================
# CORE PIPELINE (Orchestration)
# ============================================================================

# ============================================================================
# CHAMPIONSHIP (Benchmarking & Visualization)
# ============================================================================
from .best_known import (
    QAPLIB_BEST_KNOWN,
    get_best_known,
)
from .benchmarking_suite import (
    BenchmarkEntry,
    MethodStats,
    QAPSquadChampionship,
)
from .breakthrough_pursuit import (
    BreakthroughMilestone,
    BreakthroughPursuitSystem,
    BreakthroughTier,
)
from .championship_visualizer import (
    ChampionshipVisualizer,
)
from .core import QAPBenchmarkPipeline
from .logging_config import (
    PerformanceTracker,
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)

# ============================================================================
# METHODS (Optimization Methods)
# ============================================================================
from .methods import baselines, metadata, novel
from .methods.baselines import (
    apply_adaptive_lambda_adjustment,
    apply_basic_gradient_descent,
    apply_continuation_method,
    apply_hungarian_rounding,
    apply_iterative_rounding,
    apply_probabilistic_rounding,
    apply_shannon_entropy_regularization,
    apply_sinkhorn_projection,
    apply_two_opt_local_search,
)
from .methods.metadata import (
    MethodMetadata,
    NoveltyLevel,
    get_adapted_methods,
    get_all_methods,
    get_baseline_methods,
    get_method_metadata,
    get_methods_by_tag,
    get_novel_methods,
    print_methods_table,
)
from .methods.novel import (
    MethodLog,
    apply_adaptive_momentum,
    apply_constrained_step,
    apply_fft_laplace_preconditioning,
    apply_hybrid_continuous_discrete,
    apply_memory_efficient_computation,
    apply_multi_scale_gradient_flow,
    apply_parallel_processing,
    apply_reverse_time_saddle_escape,
    apply_spectral_preconditioning,
    apply_stochastic_gradient_variant,
    summarize_method_usage,
    track_Librex_metrics,
)
from .pipeline_dispatcher import (
    MethodRegistry,
    PipelineDispatcher,
    get_dispatcher,
)

# ============================================================================
# ANALYSIS (Visualization & Tables)
# ============================================================================
from .plots import (
    create_convergence_animation,
    create_publication_figure,
    plot_basin_clustering,
    # Original 6 plots
    plot_convergence_3panel,
    plot_ds_residual_tracking,
    plot_eigenvalue_analysis,
    plot_energy_histogram,
    plot_Librex_search,
    plot_evolution_sequence,
    plot_fft_speedup,
    plot_gap_heatmap,
    plot_method_effectiveness,
    # NEW 9 plots
    plot_objective_annotated,
    plot_permutation_comparison,
    plot_saddle_timeline,
)
from .plots_base import (
    add_best_indicator,
    add_gate_line,
    configure_single_axis,
    create_figure_with_subplots,
    save_and_show,
    setup_publication_style,
)
from .tables import (
    create_method_comparison,
    create_parameter_sweep_table,
    create_ranking_table,
    # Original 4 tables
    create_summary_table,
    generate_ablation_table,
    # NEW 2 tables
    generate_performance_table,
)

# ============================================================================
# UTILITIES (Core Algorithms)
# ============================================================================
from .utils import (
    BenchmarkResult,
    fft_laplace_precondition,
    hungarian_round,
    load_qaplib_instance,
    permutation_matrix,
    qap_cost,
    relaxed_qap_gradient,
    sinkhorn_knopp,
    spectral_initialization,
)

# ============================================================================
# INFRASTRUCTURE (Validation, Logging, Dispatcher)
# ============================================================================
from .validation import (
    validate_doubly_stochastic,
    validate_history_dict,
    validate_matrix,
    validate_permutation,
    validate_permutation_vector,
    validate_qap_problem,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Data helpers
    "QAPLIB_BEST_KNOWN",
    "get_best_known",
    # Pipeline
    "QAPBenchmarkPipeline",
    # Methods - Novel (11)
    "apply_fft_laplace_preconditioning",
    "apply_reverse_time_saddle_escape",
    "apply_adaptive_momentum",
    "apply_multi_scale_gradient_flow",
    "apply_spectral_preconditioning",
    "apply_stochastic_gradient_variant",
    "apply_constrained_step",
    "apply_hybrid_continuous_discrete",
    "apply_parallel_processing",
    "apply_memory_efficient_computation",
    "track_Librex_metrics",
    "summarize_method_usage",
    "MethodLog",
    # Methods - Baselines (9)
    "apply_sinkhorn_projection",
    "apply_hungarian_rounding",
    "apply_two_opt_local_search",
    "apply_basic_gradient_descent",
    "apply_shannon_entropy_regularization",
    "apply_probabilistic_rounding",
    "apply_iterative_rounding",
    "apply_continuation_method",
    "apply_adaptive_lambda_adjustment",
    # Method Metadata (Classification & Discovery)
    "NoveltyLevel",
    "MethodMetadata",
    "get_method_metadata",
    "get_all_methods",
    "get_novel_methods",
    "get_adapted_methods",
    "get_baseline_methods",
    "get_methods_by_tag",
    "print_methods_table",
    # Visualizations - Plotting (15)
    "plot_convergence_3panel",
    "plot_gap_heatmap",
    "plot_permutation_comparison",
    "plot_eigenvalue_analysis",
    "plot_Librex_search",
    "plot_ds_residual_tracking",
    "plot_objective_annotated",
    "plot_evolution_sequence",
    "plot_energy_histogram",
    "plot_basin_clustering",
    "plot_fft_speedup",
    "plot_saddle_timeline",
    "plot_method_effectiveness",
    "create_publication_figure",
    "create_convergence_animation",
    # Tables (6)
    "create_summary_table",
    "create_method_comparison",
    "create_parameter_sweep_table",
    "create_ranking_table",
    "generate_performance_table",
    "generate_ablation_table",
    # Utilities
    "BenchmarkResult",
    "sinkhorn_knopp",
    "hungarian_round",
    "spectral_initialization",
    "fft_laplace_precondition",
    "relaxed_qap_gradient",
    "qap_cost",
    "permutation_matrix",
    "load_qaplib_instance",
    # Infrastructure - Validation
    "validate_matrix",
    "validate_doubly_stochastic",
    "validate_permutation",
    "validate_qap_problem",
    "validate_permutation_vector",
    "validate_history_dict",
    # Infrastructure - Logging
    "setup_logging",
    "get_logger",
    "PerformanceTracker",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    # Infrastructure - Pipeline Dispatcher
    "MethodRegistry",
    "PipelineDispatcher",
    "get_dispatcher",
    # Infrastructure - Plotting Base
    "setup_publication_style",
    "configure_single_axis",
    "save_and_show",
    "create_figure_with_subplots",
    "add_gate_line",
    "add_best_indicator",
    # Championship (Benchmarking & Visualization)
    "QAPSquadChampionship",
    "BenchmarkEntry",
    "MethodStats",
    "ChampionshipVisualizer",
    # Breakthrough Pursuit (Continuous Optimization)
    "BreakthroughPursuitSystem",
    "BreakthroughTier",
    "BreakthroughMilestone",
]
