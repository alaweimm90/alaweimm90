"""ODE and PDE Solvers Package.
This package provides comprehensive tools for solving ordinary differential equations (ODEs)
and partial differential equations (PDEs) using various numerical methods including
finite difference, finite element, and spectral methods.
Modules:
    ode_solvers: Classical ODE solving methods
    pde_solvers: PDE solving with various discretizations
    boundary_conditions: Boundary condition handling
    adaptive_methods: Adaptive time stepping and mesh refinement
    spectral_methods: Fourier and Chebyshev methods
    finite_element: Finite element method implementation
    stability_analysis: Stability and convergence analysis
    nonlinear_solvers: Newton and continuation methods
    visualization: Berkeley-themed plotting for solutions
    utils: Utility functions and mesh generation
Classes:
    ODESolver: Base class for ODE solvers
    PDESolver: Base class for PDE solvers
    BoundaryConditions: Boundary condition management
    AdaptiveSolver: Adaptive time stepping
    SpectralSolver: Spectral method solver
    FiniteElementSolver: FEM solver
    NonlinearSolver: Nonlinear equation solver
Functions:
    solve_ode: General ODE solving interface
    solve_pde: General PDE solving interface
    analyze_stability: Stability analysis
    generate_mesh: Mesh generation utilities
Author: Berkeley SciComp Team
Date: 2024
"""
from .ode_solvers import (
    ODESolver, ExplicitEuler, ImplicitEuler, RungeKutta4,
    RungeKuttaFehlberg, AdamsBashforth, AdamsMoulton,
    BDF, solve_ode
)
from .pde_solvers import (
    PDESolver, FiniteDifferencePDE, HeatEquationSolver,
    WaveEquationSolver, PoissonSolver, AdvectionDiffusionSolver,
    NavierStokesSolver, solve_pde
)
from .boundary_conditions import (
    BoundaryCondition, DirichletBC, NeumannBC, RobinBC,
    PeriodicBC, BoundaryConditions, apply_boundary_conditions
)
from .adaptive_methods import (
    AdaptiveTimeStepper, AdaptiveMeshRefiner, ErrorEstimator,
    adaptive_rk_step, estimate_local_error, compute_optimal_timestep
)
from .spectral_methods import (
    SpectralSolver, FourierSpectral, ChebyshevSpectral,
    SpectralDifferentiation, pseudospectral_solve, fourier_derivative,
    chebyshev_derivative
)
from .finite_element import (
    FiniteElement, LinearElement, QuadraticElement, FEMSolver,
    FEMAssembler, create_1d_mesh, solve_fem_poisson
)
from .stability_analysis import (
    StabilityAnalyzer, LinearStabilityAnalyzer, VonNeumannAnalyzer,
    analyze_rk_stability, von_neumann_analysis, compute_stability_region
)
from .nonlinear_solvers import (
    NonlinearSolver, NewtonSolver, DampedNewtonSolver, FixedPointSolver,
    ContinuationSolver, newton_raphson, solve_nonlinear_ode, solve_nonlinear_pde
)
from .visualization import (
    ODEPDEVisualizer, plot_ode_solution, plot_pde_solution,
    animate_pde_evolution, create_phase_portrait
)
from .utils import (
    MeshGenerator, generate_uniform_mesh, generate_adaptive_mesh,
    compute_derivatives, interpolate_solution, check_convergence
)
# Package metadata
__version__ = "1.0.0"
__author__ = "Berkeley SciComp Team"
__email__ = "scicomp@berkeley.edu"
# Package-level configuration
import numpy as np
import matplotlib.pyplot as plt
# Set default numerical precision
DEFAULT_TOLERANCE = 1e-8
DEFAULT_MAX_ITERATIONS = 1000
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
# Configure matplotlib for Berkeley styling
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [10, 6],
    'axes.prop_cycle': plt.cycler('color', [BERKELEY_BLUE, CALIFORNIA_GOLD,
                                          '#3B7EA1', '#C4820E', '#00B0DA']),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10
})
# Validation functions
def validate_initial_conditions(y0, t0=0.0):
    """Validate initial conditions for ODE/PDE."""
    y0 = np.asarray(y0)
    if not np.isfinite(y0).all():
        raise ValueError("Initial conditions must be finite")
    return y0, float(t0)
def validate_time_span(t_span):
    """Validate time span for time-dependent problems."""
    t_span = np.asarray(t_span)
    if len(t_span) != 2:
        raise ValueError("Time span must be [t0, tf]")
    if t_span[1] <= t_span[0]:
        raise ValueError("Final time must be greater than initial time")
    return t_span
def validate_mesh(mesh):
    """Validate mesh structure."""
    if 'nodes' not in mesh:
        raise ValueError("Mesh must contain 'nodes'")
    if not isinstance(mesh['nodes'], np.ndarray):
        raise ValueError("Mesh nodes must be numpy array")
    return mesh
# Package-level constants
class Constants:
    """Mathematical and physical constants for ODE/PDE problems."""
    # Mathematical constants
    PI = np.pi
    E = np.e
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    # Numerical constants
    MACHINE_EPSILON = np.finfo(float).eps
    SQRT_EPS = np.sqrt(MACHINE_EPSILON)
    # Default tolerances for different methods
    ODE_TOLERANCE = 1e-8
    PDE_TOLERANCE = 1e-6
    NEWTON_TOLERANCE = 1e-10
    # CFL limits for stability
    CFL_EXPLICIT = 0.5
    CFL_IMPLICIT = 2.0
# Quick access functions
def quick_ode_solve(func, y0, t_span, method='rk45', **kwargs):
    """Quick ODE solve with sensible defaults."""
    return solve_ode(func, y0, t_span, method=method, **kwargs)
def quick_pde_solve(pde_type, initial_condition, boundary_conditions,
                   domain, time_span=None, **kwargs):
    """Quick PDE solve with sensible defaults."""
    return solve_pde(pde_type, initial_condition, boundary_conditions,
                    domain, time_span=time_span, **kwargs)
# Package info
def get_package_info():
    """Get package information."""
    return {
        'name': 'ODE_PDE',
        'version': __version__,
        'author': __author__,
        'description': 'Berkeley SciComp ODE and PDE Solvers',
        'methods': {
            'ode_methods': ['euler', 'rk4', 'rk45', 'adams', 'bdf'],
            'pde_methods': ['finite_difference', 'finite_element', 'spectral'],
            'boundary_types': ['dirichlet', 'neumann', 'robin', 'periodic'],
            'adaptive_methods': ['time_stepping', 'mesh_refinement', 'error_control']
        }
    }
# Convenience imports for common use cases
__all__ = [
    # Core solvers
    'ODESolver', 'PDESolver', 'solve_ode', 'solve_pde',
    # ODE methods
    'ExplicitEuler', 'ImplicitEuler', 'RungeKutta4', 'RungeKuttaFehlberg',
    'AdamsBashforth', 'AdamsMoulton', 'BDF',
    # PDE methods
    'FiniteDifferencePDE', 'HeatEquationSolver', 'WaveEquationSolver',
    'PoissonSolver', 'AdvectionDiffusionSolver', 'NavierStokesSolver',
    # Boundary conditions
    'BoundaryCondition', 'DirichletBC', 'NeumannBC', 'RobinBC', 'PeriodicBC',
    'BoundaryConditions', 'apply_boundary_conditions',
    # Advanced methods
    'AdaptiveTimeStepper', 'AdaptiveMeshRefiner', 'SpectralSolver',
    'FEMSolver', 'NonlinearSolver', 'NewtonSolver', 'ContinuationSolver',
    # Analysis tools
    'StabilityAnalyzer', 'analyze_stability', 'ErrorEstimator',
    # Visualization
    'ODEPDEVisualizer', 'plot_ode_solution', 'plot_pde_solution',
    'animate_pde_evolution', 'create_phase_portrait',
    # Utilities
    'MeshGenerator', 'generate_uniform_mesh', 'validate_initial_conditions',
    'validate_time_span', 'Constants',
    # Quick access
    'quick_ode_solve', 'quick_pde_solve', 'get_package_info'
]