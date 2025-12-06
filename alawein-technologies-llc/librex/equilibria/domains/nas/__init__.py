"""Neural Architecture Search (NAS) domain for Librex.

This module provides comprehensive Neural Architecture Search capabilities,
enabling optimization of deep learning architectures using various search
strategies and evaluation methods.

Key Components:
    - Architecture representations (cell-based and macro search)
    - NAS problem definitions with multi-objective support
    - Domain adapter for Librex integration
    - Efficient search methods (evolutionary, differentiable, Bayesian)
    - Zero-cost proxies for fast evaluation
    - Hardware-aware optimization

Example:
    >>> from Librex.domains.nas import NASProblem, NASAdapter
    >>> from Librex import optimize
    >>>
    >>> # Define NAS problem
    >>> problem = NASProblem(
    ...     dataset='cifar10',
    ...     search_space='cell',
    ...     objectives=['accuracy', 'params']
    ... )
    >>>
    >>> # Run optimization
    >>> result = optimize(problem, NASAdapter(), method='evolutionary')
    >>> best_arch = result['solution']
"""

from .architecture import NASCell, MacroArchitecture, Layer, Operation
from .nas_problem import NASProblem, NASObjective, NASConstraint
from .nas_adapter import NASAdapter
from .zero_cost_proxies import (
    grad_norm_proxy,
    jacob_cov_proxy,
    ntk_proxy,
    synflow_proxy,
    zen_score
)
from .hardware_aware import HardwareAwareNAS, LatencyPredictor, EnergyEstimator
from .methods import (
    evolutionary_nas,
    differentiable_nas,
    bayesian_optimization_nas,
    random_search_nas,
    regularized_evolution,
    gdas,
    enas
)

__all__ = [
    # Architecture representations
    'NASCell',
    'MacroArchitecture',
    'Layer',
    'Operation',

    # Problem definition
    'NASProblem',
    'NASObjective',
    'NASConstraint',

    # Adapter
    'NASAdapter',

    # Zero-cost proxies
    'grad_norm_proxy',
    'jacob_cov_proxy',
    'ntk_proxy',
    'synflow_proxy',
    'zen_score',

    # Hardware-aware
    'HardwareAwareNAS',
    'LatencyPredictor',
    'EnergyEstimator',

    # Search methods
    'evolutionary_nas',
    'differentiable_nas',
    'bayesian_optimization_nas',
    'random_search_nas',
    'regularized_evolution',
    'gdas',
    'enas'
]

__version__ = '1.0.0'