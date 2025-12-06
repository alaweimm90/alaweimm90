"""Stability Analysis for ODE and PDE Methods.
This module provides comprehensive stability analysis tools for numerical
methods including linear stability analysis, von Neumann analysis,
and eigenvalue-based stability assessment.
Classes:
    StabilityAnalyzer: Main stability analysis class
    LinearStabilityAnalyzer: Linear stability analysis
    VonNeumannAnalyzer: Fourier stability analysis
Functions:
    analyze_rk_stability: Runge-Kutta stability analysis
    von_neumann_analysis: Von Neumann stability analysis
    compute_stability_region: Stability region computation
    cfl_analysis: CFL condition analysis
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, norm
from scipy.optimize import fsolve
import cmath
@dataclass
class StabilityResult:
    """Result of stability analysis."""
    stable: bool
    stability_region: Optional[np.ndarray]
    eigenvalues: Optional[np.ndarray]
    spectral_radius: float
    cfl_limit: Optional[float]
    method_order: int
    analysis_type: str
    message: str
@dataclass
class AmplificationMatrix:
    """Amplification matrix for stability analysis."""
    matrix: np.ndarray
    eigenvalues: np.ndarray
    spectral_radius: float
    max_eigenvalue: complex
    stable: bool
class StabilityAnalyzer:
    """General stability analyzer for numerical methods."""
    def __init__(self, method_name: str, order: int = 1):
        """Initialize stability analyzer.
        Args:
            method_name: Name of numerical method
            order: Order of the method
        """
        self.method_name = method_name
        self.order = order
    def analyze_method_stability(self, lambda_values: np.ndarray) -> StabilityResult:
        """Analyze stability for given eigenvalues.
        Args:
            lambda_values: Eigenvalues of the spatial discretization
        Returns:
            StabilityResult
        """
        # Get stability function for method
        stability_func = self._get_stability_function()
        # Evaluate stability function
        z_values = lambda_values  # Assume already scaled by dt
        R_values = stability_func(z_values)
        # Check stability condition |R(z)| <= 1
        max_amplification = np.max(np.abs(R_values))
        stable = max_amplification <= 1.0 + 1e-12  # Small tolerance for numerical errors
        return StabilityResult(
            stable=stable,
            stability_region=None,
            eigenvalues=lambda_values,
            spectral_radius=max_amplification,
            cfl_limit=None,
            method_order=self.order,
            analysis_type="linear_stability",
            message=f"Method {'stable' if stable else 'unstable'}: max |R(z)| = {max_amplification:.6f}"
        )
    def _get_stability_function(self) -> Callable:
        """Get stability function R(z) for the method."""
        method = self.method_name.lower()
        if method == "forward_euler" or method == "euler":
            return lambda z: 1 + z
        elif method == "backward_euler" or method == "implicit_euler":
            return lambda z: 1 / (1 - z)
        elif method == "crank_nicolson":
            return lambda z: (1 + z/2) / (1 - z/2)
        elif method == "rk4":
            return lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24
        elif method == "rk2" or method == "midpoint":
            return lambda z: 1 + z + z**2/2
        elif method == "rk3":
            return lambda z: 1 + z + z**2/2 + z**3/6
        elif method == "adams_bashforth_2":
            # Multi-step method - more complex stability analysis needed
            return lambda z: 1 + 3*z/2 - z**2/2  # Simplified
        elif method == "adams_bashforth_3":
            return lambda z: 1 + 23*z/12 - 16*z**2/12 + 5*z**3/12  # Simplified
        else:
            warnings.warn(f"Unknown method {method}, using forward Euler")
            return lambda z: 1 + z
class LinearStabilityAnalyzer:
    """Linear stability analysis for ODEs and PDEs."""
    def __init__(self):
        """Initialize linear stability analyzer."""
        pass
    def analyze_ode_system(self, jacobian: np.ndarray, dt: float,
                          method: str = "forward_euler") -> StabilityResult:
        """Analyze stability of ODE system du/dt = f(u).
        Args:
            jacobian: Jacobian matrix df/du
            dt: Time step
            method: Time integration method
        Returns:
            StabilityResult
        """
        # Eigenvalues of Jacobian
        eigenvalues = eigvals(jacobian)
        # Scale by time step
        z_values = dt * eigenvalues
        # Analyze stability
        analyzer = StabilityAnalyzer(method)
        result = analyzer.analyze_method_stability(z_values)
        # Additional ODE-specific checks
        # For stability, we need Re(λ) < 0 for all eigenvalues
        real_parts = np.real(eigenvalues)
        ode_stable = np.all(real_parts <= 0)
        if not ode_stable:
            result.message += f" (ODE system unstable: max Re(λ) = {np.max(real_parts):.6f})"
            result.stable = False
        return result
    def analyze_pde_discretization(self, spatial_operator: np.ndarray,
                                  dx: float, dt: float,
                                  method: str = "forward_euler") -> StabilityResult:
        """Analyze stability of PDE spatial discretization.
        Args:
            spatial_operator: Spatial discretization matrix
            dx: Spatial step size
            dt: Time step
            method: Time integration method
        Returns:
            StabilityResult
        """
        # Eigenvalues of spatial operator
        eigenvalues = eigvals(spatial_operator)
        # Scale by time step
        z_values = dt * eigenvalues
        # Analyze stability
        analyzer = StabilityAnalyzer(method)
        result = analyzer.analyze_method_stability(z_values)
        # Compute CFL-like condition
        max_eigenvalue_magnitude = np.max(np.abs(eigenvalues))
        if max_eigenvalue_magnitude > 0:
            cfl_limit = 1.0 / (max_eigenvalue_magnitude * dt)
            result.cfl_limit = cfl_limit
        return result
class VonNeumannAnalyzer:
    """Von Neumann (Fourier) stability analysis for PDEs."""
    def __init__(self):
        """Initialize von Neumann analyzer."""
        pass
    def analyze_heat_equation(self, alpha: float, dx: float, dt: float,
                             method: str = "forward_euler") -> StabilityResult:
        """Von Neumann analysis for heat equation ∂u/∂t = α∂²u/∂x².
        Args:
            alpha: Thermal diffusivity
            dx: Spatial step size
            dt: Time step
            method: Time integration method
        Returns:
            StabilityResult
        """
        # Fourier modes (wavenumber)
        k_max = np.pi / dx  # Maximum wavenumber
        k_values = np.linspace(-k_max, k_max, 1000)
        amplification_factors = []
        for k in k_values:
            # Spatial discretization eigenvalue
            if method in ["forward_euler", "backward_euler", "crank_nicolson"]:
                # Central differences for second derivative
                lambda_k = -4 * alpha * np.sin(k * dx / 2)**2 / dx**2
            else:
                lambda_k = -4 * alpha * np.sin(k * dx / 2)**2 / dx**2
            # Time discretization
            z = dt * lambda_k
            if method == "forward_euler":
                G = 1 + z
            elif method == "backward_euler":
                G = 1 / (1 - z)
            elif method == "crank_nicolson":
                G = (1 + z/2) / (1 - z/2)
            else:
                analyzer = StabilityAnalyzer(method)
                stability_func = analyzer._get_stability_function()
                G = stability_func(z)
            amplification_factors.append(abs(G))
        max_amplification = np.max(amplification_factors)
        stable = max_amplification <= 1.0 + 1e-12
        # CFL condition for heat equation
        if method == "forward_euler":
            cfl_limit = dx**2 / (2 * alpha * dt)
        else:
            cfl_limit = None
        return StabilityResult(
            stable=stable,
            stability_region=np.array(amplification_factors),
            eigenvalues=k_values,
            spectral_radius=max_amplification,
            cfl_limit=cfl_limit,
            method_order=1,
            analysis_type="von_neumann",
            message=f"Von Neumann analysis: max |G| = {max_amplification:.6f}"
        )
    def analyze_wave_equation(self, c: float, dx: float, dt: float) -> StabilityResult:
        """Von Neumann analysis for wave equation ∂²u/∂t² = c²∂²u/∂x².
        Args:
            c: Wave speed
            dx: Spatial step size
            dt: Time step
        Returns:
            StabilityResult
        """
        # Fourier modes
        k_max = np.pi / dx
        k_values = np.linspace(-k_max, k_max, 1000)
        amplification_factors = []
        for k in k_values:
            # CFL number
            nu = c * dt / dx
            # Amplification factor for central differences
            # G = 1 - 4*nu²*sin²(kh/2)
            G = 1 - 4 * nu**2 * np.sin(k * dx / 2)**2
            amplification_factors.append(abs(G))
        max_amplification = np.max(amplification_factors)
        stable = max_amplification <= 1.0 + 1e-12
        # CFL condition: c*dt/dx <= 1
        cfl_number = c * dt / dx
        cfl_stable = cfl_number <= 1.0
        stable = stable and cfl_stable
        return StabilityResult(
            stable=stable,
            stability_region=np.array(amplification_factors),
            eigenvalues=k_values,
            spectral_radius=max_amplification,
            cfl_limit=dx / c,  # Maximum stable dt
            method_order=2,
            analysis_type="von_neumann_wave",
            message=f"Wave equation CFL = {cfl_number:.6f}, max |G| = {max_amplification:.6f}"
        )
    def analyze_advection_equation(self, v: float, dx: float, dt: float,
                                  scheme: str = "upwind") -> StabilityResult:
        """Von Neumann analysis for advection equation ∂u/∂t + v∂u/∂x = 0.
        Args:
            v: Advection velocity
            dx: Spatial step size
            dt: Time step
            scheme: Spatial discretization scheme
        Returns:
            StabilityResult
        """
        # Fourier modes
        k_max = np.pi / dx
        k_values = np.linspace(-k_max, k_max, 1000)
        amplification_factors = []
        cfl = v * dt / dx
        for k in k_values:
            kh = k * dx
            if scheme == "upwind":
                if v >= 0:
                    # Forward upwind
                    G = 1 - cfl * (1 - np.exp(-1j * kh))
                else:
                    # Backward upwind
                    G = 1 - cfl * (np.exp(1j * kh) - 1)
            elif scheme == "central":
                # Central differences
                G = 1 - 1j * cfl * np.sin(kh)
            elif scheme == "lax_friedrichs":
                # Lax-Friedrichs
                G = np.cos(kh) - 1j * cfl * np.sin(kh)
            elif scheme == "lax_wendroff":
                # Lax-Wendroff
                G = 1 - 1j * cfl * np.sin(kh) - cfl**2 * (np.cos(kh) - 1)
            else:
                warnings.warn(f"Unknown scheme {scheme}, using upwind")
                G = 1 - cfl * (1 - np.exp(-1j * kh))
            amplification_factors.append(abs(G))
        max_amplification = np.max(amplification_factors)
        stable = max_amplification <= 1.0 + 1e-12
        # CFL condition for advection
        cfl_stable = abs(cfl) <= 1.0
        stable = stable and cfl_stable
        return StabilityResult(
            stable=stable,
            stability_region=np.array(amplification_factors),
            eigenvalues=k_values,
            spectral_radius=max_amplification,
            cfl_limit=dx / abs(v),
            method_order=1,
            analysis_type="von_neumann_advection",
            message=f"Advection CFL = {cfl:.6f}, max |G| = {max_amplification:.6f}"
        )
# Utility functions
def analyze_rk_stability(order: int, z_values: np.ndarray) -> StabilityResult:
    """Analyze Runge-Kutta stability.
    Args:
        order: Order of RK method
        z_values: Scaled eigenvalues
    Returns:
        StabilityResult
    """
    method_name = f"rk{order}"
    analyzer = StabilityAnalyzer(method_name, order)
    return analyzer.analyze_method_stability(z_values)
def von_neumann_analysis(pde_type: str, parameters: Dict[str, float],
                        dx: float, dt: float, method: str = "forward_euler") -> StabilityResult:
    """General von Neumann stability analysis.
    Args:
        pde_type: Type of PDE ('heat', 'wave', 'advection')
        parameters: PDE parameters
        dx: Spatial step size
        dt: Time step
        method: Time integration method
    Returns:
        StabilityResult
    """
    analyzer = VonNeumannAnalyzer()
    if pde_type == "heat":
        alpha = parameters.get("alpha", 1.0)
        return analyzer.analyze_heat_equation(alpha, dx, dt, method)
    elif pde_type == "wave":
        c = parameters.get("c", 1.0)
        return analyzer.analyze_wave_equation(c, dx, dt)
    elif pde_type == "advection":
        v = parameters.get("v", 1.0)
        scheme = parameters.get("scheme", "upwind")
        return analyzer.analyze_advection_equation(v, dx, dt, scheme)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
def compute_stability_region(method: str, real_range: Tuple[float, float] = (-4, 2),
                           imag_range: Tuple[float, float] = (-3, 3),
                           resolution: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stability region for numerical method.
    Args:
        method: Numerical method name
        real_range: Range for real axis
        imag_range: Range for imaginary axis
        resolution: Grid resolution
    Returns:
        Tuple of (real_grid, imag_grid, stability_matrix)
    """
    # Create complex grid
    real = np.linspace(real_range[0], real_range[1], resolution)
    imag = np.linspace(imag_range[0], imag_range[1], resolution)
    R, I = np.meshgrid(real, imag)
    Z = R + 1j * I
    # Get stability function
    analyzer = StabilityAnalyzer(method)
    stability_func = analyzer._get_stability_function()
    # Evaluate stability function
    stability_values = stability_func(Z)
    stability_matrix = np.abs(stability_values)
    return R, I, stability_matrix
def cfl_analysis(pde_type: str, parameters: Dict[str, float],
                dx: float, target_cfl: float = 0.5) -> Dict[str, float]:
    """Analyze CFL conditions and recommend time step.
    Args:
        pde_type: Type of PDE
        parameters: PDE parameters
        dx: Spatial step size
        target_cfl: Target CFL number
    Returns:
        Dictionary with CFL analysis results
    """
    if pde_type == "heat":
        alpha = parameters.get("alpha", 1.0)
        # For explicit methods: dt <= dx²/(2α)
        dt_max_explicit = dx**2 / (2 * alpha)
        dt_recommended = target_cfl * dt_max_explicit
        return {
            "dt_max_explicit": dt_max_explicit,
            "dt_recommended": dt_recommended,
            "cfl_explicit": target_cfl,
            "stability_limit": "dt <= dx²/(2α)"
        }
    elif pde_type == "wave":
        c = parameters.get("c", 1.0)
        # CFL condition: c*dt/dx <= 1
        dt_max = dx / c
        dt_recommended = target_cfl * dt_max
        return {
            "dt_max": dt_max,
            "dt_recommended": dt_recommended,
            "cfl_number": target_cfl,
            "stability_limit": "c*dt/dx <= 1"
        }
    elif pde_type == "advection":
        v = parameters.get("v", 1.0)
        # CFL condition: |v|*dt/dx <= 1
        dt_max = dx / abs(v)
        dt_recommended = target_cfl * dt_max
        return {
            "dt_max": dt_max,
            "dt_recommended": dt_recommended,
            "cfl_number": target_cfl,
            "velocity": v,
            "stability_limit": "|v|*dt/dx <= 1"
        }
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
def plot_stability_region(method: str, save_path: Optional[str] = None,
                         title: Optional[str] = None) -> plt.Figure:
    """Plot stability region for numerical method.
    Args:
        method: Numerical method name
        save_path: Path to save figure
        title: Plot title
    Returns:
        Figure object
    """
    # Compute stability region
    R, I, stability = compute_stability_region(method)
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot stability region
    levels = [0.5, 1.0, 1.5, 2.0]
    cs = ax.contour(R, I, stability, levels=levels, colors=['blue'], alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=10)
    # Fill stable region
    stable_region = stability <= 1.0
    ax.contourf(R, I, stable_region, levels=[0.5, 1.5], colors=['gold'], alpha=0.3)
    # Highlight stability boundary
    ax.contour(R, I, stability, levels=[1.0], colors=['blue'], linewidths=3)
    # Formatting
    ax.set_xlabel('Real(z)')
    ax.set_ylabel('Imag(z)')
    if title is None:
        title = f'Stability Region: {method.title().replace("_", " ")}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', alpha=0.3, label='Stable Region (|R(z)| ≤ 1)'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Stability Boundary')
    ]
    ax.legend(handles=legend_elements)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
def stability_comparison(methods: List[str], z_test: complex = -1.0) -> Dict[str, Dict[str, Any]]:
    """Compare stability properties of different methods.
    Args:
        methods: List of method names
        z_test: Test point in complex plane
    Returns:
        Dictionary with comparison results
    """
    results = {}
    for method in methods:
        analyzer = StabilityAnalyzer(method)
        stability_func = analyzer._get_stability_function()
        # Evaluate at test point
        R_value = stability_func(z_test)
        stable_at_point = abs(R_value) <= 1.0
        # Find stability boundary (approximate)
        def stability_boundary_equation(z_real):
            z = z_real + 0j
            return abs(stability_func(z)) - 1.0
        try:
            # Find real axis intercept
            z_boundary = fsolve(stability_boundary_equation, -1.0)[0]
        except:
            z_boundary = None
        results[method] = {
            "R_at_test_point": R_value,
            "stable_at_test_point": stable_at_point,
            "real_axis_boundary": z_boundary,
            "method_type": "explicit" if "euler" in method.lower() and "backward" not in method.lower() else "implicit"
        }
    return results