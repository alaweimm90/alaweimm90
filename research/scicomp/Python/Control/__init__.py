"""
Control Systems Module
Professional control systems implementations for scientific computing.
Includes PID controllers, state-space methods, optimal control, and modern control techniques.
Modules:
    core: Core control algorithms and utilities
    examples: Educational examples at multiple difficulty levels
    tests: Comprehensive test suites
    benchmarks: Performance analysis tools
"""
from .core.pid_controller import PIDController
from .core.state_space import StateSpaceSystem, LinearQuadraticRegulator
from .core.optimal_control import OptimalController
from .core.robust_control import RobustController
__all__ = [
    'PIDController',
    'StateSpaceSystem',
    'LinearQuadraticRegulator',
    'OptimalController',
    'RobustController'
]
__version__ = "1.0.0"
__author__ = "SciComp Development Team"