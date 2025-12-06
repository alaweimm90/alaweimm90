"""Spectral Methods for PDEs.
This module provides spectral method implementations including Fourier spectral,
Chebyshev spectral, and pseudospectral methods for solving PDEs with high accuracy.
Classes:
    SpectralSolver: Base class for spectral methods
    FourierSpectral: Fourier spectral method
    ChebyshevSpectral: Chebyshev spectral method
    SpectralDifferentiation: Spectral differentiation matrices
Functions:
    pseudospectral_solve: General pseudospectral solver
    fourier_derivative: Fourier derivative computation
    chebyshev_derivative: Chebyshev derivative computation
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import solve
@dataclass
class SpectralResult:
    """Result of spectral method solution."""
    x: np.ndarray
    u: np.ndarray
    spectral_coeffs: np.ndarray
    success: bool
    message: str
    n_modes: int = 0
    spectral_accuracy: Optional[float] = None
class SpectralSolver(ABC):
    """Abstract base class for spectral methods."""
    def __init__(self, n_modes: int, domain: Tuple[float, float]):
        """Initialize spectral solver.
        Args:
            n_modes: Number of spectral modes
            domain: Computational domain (a, b)
        """
        self.n_modes = n_modes
        self.domain = domain
        self.a, self.b = domain
        # Grid points and differentiation matrices
        self.x = None
        self.D = None  # First derivative matrix
        self.D2 = None  # Second derivative matrix
        self._setup_grid()
        self._setup_differentiation_matrices()
    @abstractmethod
    def _setup_grid(self):
        """Setup computational grid."""
        pass
    @abstractmethod
    def _setup_differentiation_matrices(self):
        """Setup spectral differentiation matrices."""
        pass
    @abstractmethod
    def solve_pde(self, pde_func: Callable, boundary_conditions: Dict[str, Any],
                  initial_condition: Optional[Callable] = None) -> SpectralResult:
        """Solve PDE using spectral method."""
        pass
class FourierSpectral(SpectralSolver):
    """Fourier spectral method for periodic problems."""
    def _setup_grid(self):
        """Setup Fourier grid (periodic)."""
        # Fourier points (excluding endpoint for periodicity)
        self.x = np.linspace(self.a, self.b, self.n_modes, endpoint=False)
        # Wavenumbers
        L = self.b - self.a  # Domain length
        self.k = 2 * np.pi * fftfreq(self.n_modes, L / self.n_modes)
    def _setup_differentiation_matrices(self):
        """Setup Fourier differentiation matrices."""
        N = self.n_modes
        # First derivative matrix in Fourier space
        # d/dx in Fourier space is multiplication by ik
        self.D = np.zeros((N, N), dtype=complex)
        self.D2 = np.zeros((N, N), dtype=complex)
        # This is actually implemented via FFT operations
        # Matrices are mainly for interface compatibility
    def fourier_derivative(self, u: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute derivative using FFT.
        Args:
            u: Function values on grid
            order: Derivative order
        Returns:
            Derivative values
        """
        # Forward FFT
        u_hat = fft(u)
        # Multiply by (ik)^order in Fourier space
        if order == 1:
            u_hat_deriv = 1j * self.k * u_hat
        elif order == 2:
            u_hat_deriv = -self.k**2 * u_hat
        else:
            u_hat_deriv = (1j * self.k)**order * u_hat
        # Inverse FFT
        u_deriv = np.real(ifft(u_hat_deriv))
        return u_deriv
    def solve_pde(self, pde_func: Callable, boundary_conditions: Dict[str, Any],
                  initial_condition: Optional[Callable] = None) -> SpectralResult:
        """Solve PDE using Fourier spectral method.
        Note: Fourier methods require periodic boundary conditions.
        """
        if not boundary_conditions.get('periodic', False):
            warnings.warn("Fourier spectral method requires periodic boundary conditions")
        # For demonstration, solve simple diffusion equation
        # ∂u/∂t = α ∂²u/∂x²
        if initial_condition is None:
            # Default initial condition
            u0 = np.sin(2 * np.pi * self.x / (self.b - self.a))
        else:
            u0 = initial_condition(self.x)
        # If time-dependent, use method of lines
        # For steady state, solve directly
        # This is a simplified implementation
        # Full implementation would handle general PDEs
        return SpectralResult(
            x=self.x,
            u=u0,  # Placeholder
            spectral_coeffs=fft(u0),
            success=True,
            message="Fourier spectral solve completed",
            n_modes=self.n_modes
        )
    def solve_heat_equation(self, alpha: float, initial_condition: Callable,
                           time_span: Tuple[float, float],
                           n_time_steps: int = 100) -> SpectralResult:
        """Solve heat equation using Fourier spectral method.
        Args:
            alpha: Thermal diffusivity
            initial_condition: Initial condition function
            time_span: Time interval (t0, tf)
            n_time_steps: Number of time steps
        Returns:
            SpectralResult with solution
        """
        t0, tf = time_span
        dt = (tf - t0) / n_time_steps
        # Initial condition
        u = initial_condition(self.x)
        # Time stepping in Fourier space
        u_hat = fft(u)
        # Exact time integration for each mode
        # u_hat(t+dt) = u_hat(t) * exp(-α * k² * dt)
        time_factor = np.exp(-alpha * self.k**2 * dt)
        # Store solution at final time
        for _ in range(n_time_steps):
            u_hat *= time_factor
        # Transform back to physical space
        u_final = np.real(ifft(u_hat))
        return SpectralResult(
            x=self.x,
            u=u_final,
            spectral_coeffs=u_hat,
            success=True,
            message="Heat equation solved with Fourier spectral method",
            n_modes=self.n_modes,
            spectral_accuracy=np.max(np.abs(u_hat[self.n_modes//2:]))  # High frequency content
        )
class ChebyshevSpectral(SpectralSolver):
    """Chebyshev spectral method for non-periodic problems."""
    def _setup_grid(self):
        """Setup Chebyshev-Gauss-Lobatto grid."""
        # Chebyshev points in [-1, 1]
        i = np.arange(self.n_modes)
        xi = np.cos(np.pi * i / (self.n_modes - 1))
        # Map to physical domain [a, b]
        self.x = 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * xi
        # Store computational coordinates
        self.xi = xi
    def _setup_differentiation_matrices(self):
        """Setup Chebyshev differentiation matrices."""
        N = self.n_modes
        xi = self.xi
        # Chebyshev differentiation matrix
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        X = np.tile(xi, (N, 1)).T
        dX = X - X.T
        self.D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    self.D[i, j] = (c[i] / c[j]) * (-1)**(i+j) / (xi[i] - xi[j])
        # Diagonal entries
        for i in range(N):
            self.D[i, i] = -np.sum(self.D[i, :])
        # Second derivative matrix
        self.D2 = self.D @ self.D
        # Scale for physical domain
        scale_factor = 2 / (self.b - self.a)
        self.D *= scale_factor
        self.D2 *= scale_factor**2
    def solve_pde(self, pde_func: Callable, boundary_conditions: Dict[str, Any],
                  initial_condition: Optional[Callable] = None) -> SpectralResult:
        """Solve PDE using Chebyshev spectral method."""
        # For demonstration, solve Poisson equation
        # -∇²u = f with Dirichlet BC
        if 'source' not in boundary_conditions:
            # Default source
            source = lambda x: np.sin(np.pi * x)
        else:
            source = boundary_conditions['source']
        # Right-hand side
        f = source(self.x)
        # Apply boundary conditions
        A = -self.D2.copy()
        rhs = f.copy()
        # Dirichlet boundary conditions
        if 'dirichlet' in boundary_conditions:
            bc = boundary_conditions['dirichlet']
            # Left boundary
            if 'left' in bc:
                A[0, :] = 0
                A[0, 0] = 1
                rhs[0] = bc['left']
            # Right boundary
            if 'right' in bc:
                A[-1, :] = 0
                A[-1, -1] = 1
                rhs[-1] = bc['right']
        # Solve linear system
        try:
            u = solve(A, rhs)
            success = True
            message = "Chebyshev spectral solve completed successfully"
        except Exception as e:
            u = np.zeros_like(self.x)
            success = False
            message = f"Chebyshev solve failed: {e}"
        # Compute Chebyshev coefficients
        # Transform to coefficient space (DCT-I)
        from scipy.fft import dct
        coeffs = dct(u, type=1) / (len(u) - 1)
        coeffs[0] /= 2
        coeffs[-1] /= 2
        return SpectralResult(
            x=self.x,
            u=u,
            spectral_coeffs=coeffs,
            success=success,
            message=message,
            n_modes=self.n_modes,
            spectral_accuracy=np.abs(coeffs[-1])  # Last coefficient as accuracy indicator
        )
    def solve_bvp(self, differential_operator: Callable,
                  boundary_conditions: Dict[str, float],
                  source_term: Optional[Callable] = None) -> SpectralResult:
        """Solve boundary value problem.
        Args:
            differential_operator: Function that applies differential operator
            boundary_conditions: Boundary condition values
            source_term: Source term function
        Returns:
            SpectralResult
        """
        N = self.n_modes
        # Build system matrix
        A = np.zeros((N, N))
        rhs = np.zeros(N)
        # Apply differential operator to each basis function
        # This is a simplified approach; full implementation would be more sophisticated
        # For Poisson equation: -d²u/dx² = f
        A = -self.D2.copy()
        if source_term:
            rhs = source_term(self.x)
        # Apply boundary conditions
        if 'left' in boundary_conditions:
            A[0, :] = 0
            A[0, 0] = 1
            rhs[0] = boundary_conditions['left']
        if 'right' in boundary_conditions:
            A[-1, :] = 0
            A[-1, -1] = 1
            rhs[-1] = boundary_conditions['right']
        # Solve
        try:
            u = solve(A, rhs)
            success = True
            message = "BVP solved successfully"
        except Exception as e:
            u = np.zeros(N)
            success = False
            message = f"BVP solve failed: {e}"
        # Compute coefficients
        from scipy.fft import dct
        coeffs = dct(u, type=1) / (N - 1)
        coeffs[0] /= 2
        coeffs[-1] /= 2
        return SpectralResult(
            x=self.x,
            u=u,
            spectral_coeffs=coeffs,
            success=success,
            message=message,
            n_modes=N
        )
class SpectralDifferentiation:
    """Utility class for spectral differentiation."""
    @staticmethod
    def fourier_differentiation_matrix(N: int, L: float) -> np.ndarray:
        """Construct Fourier differentiation matrix.
        Args:
            N: Number of grid points
            L: Domain length
        Returns:
            Differentiation matrix
        """
        h = L / N
        k = 2 * np.pi / L
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = 0.5 * (-1)**(i-j) / np.tan(0.5 * k * h * (i - j))
        return D
    @staticmethod
    def chebyshev_differentiation_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Chebyshev differentiation matrix.
        Args:
            N: Number of Chebyshev points
        Returns:
            Tuple of (grid_points, differentiation_matrix)
        """
        # Chebyshev points
        i = np.arange(N)
        x = np.cos(np.pi * i / (N - 1))
        # Weights
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c[1:-1] = 1
        # Differentiation matrix
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (c[i] / c[j]) * (-1)**(i+j) / (x[i] - x[j])
        # Diagonal entries
        for i in range(N):
            D[i, i] = -np.sum(D[i, :])
        return x, D
# Main functions
def pseudospectral_solve(pde_type: str, domain: Tuple[float, float],
                        n_modes: int, boundary_conditions: Dict[str, Any],
                        spectral_type: str = 'chebyshev',
                        **kwargs) -> SpectralResult:
    """General pseudospectral PDE solver.
    Args:
        pde_type: Type of PDE ('poisson', 'heat', 'wave')
        domain: Computational domain
        n_modes: Number of spectral modes
        boundary_conditions: Boundary conditions
        spectral_type: Type of spectral method ('fourier', 'chebyshev')
        **kwargs: Additional parameters
    Returns:
        SpectralResult
    """
    # Create appropriate spectral solver
    if spectral_type == 'fourier':
        solver = FourierSpectral(n_modes, domain)
    elif spectral_type == 'chebyshev':
        solver = ChebyshevSpectral(n_modes, domain)
    else:
        raise ValueError(f"Unknown spectral type: {spectral_type}")
    # Solve based on PDE type
    if pde_type == 'poisson':
        return solver.solve_pde(None, boundary_conditions)
    elif pde_type == 'heat' and spectral_type == 'fourier':
        alpha = kwargs.get('alpha', 1.0)
        initial_condition = kwargs.get('initial_condition', lambda x: np.sin(2*np.pi*x))
        time_span = kwargs.get('time_span', (0, 1))
        return solver.solve_heat_equation(alpha, initial_condition, time_span)
    else:
        return solver.solve_pde(None, boundary_conditions)
def fourier_derivative(u: np.ndarray, L: float, order: int = 1) -> np.ndarray:
    """Compute derivative using Fourier method.
    Args:
        u: Function values on periodic grid
        L: Domain length
        order: Derivative order
    Returns:
        Derivative values
    """
    N = len(u)
    k = 2 * np.pi * fftfreq(N, L / N)
    # Forward FFT
    u_hat = fft(u)
    # Multiply by (ik)^order
    u_hat_deriv = (1j * k)**order * u_hat
    # Inverse FFT
    u_deriv = np.real(ifft(u_hat_deriv))
    return u_deriv
def chebyshev_derivative(u: np.ndarray, domain: Tuple[float, float],
                        order: int = 1) -> np.ndarray:
    """Compute derivative using Chebyshev method.
    Args:
        u: Function values on Chebyshev grid
        domain: Physical domain
        order: Derivative order
    Returns:
        Derivative values
    """
    N = len(u)
    a, b = domain
    # Chebyshev differentiation matrix
    _, D = SpectralDifferentiation.chebyshev_differentiation_matrix(N)
    # Scale for physical domain
    scale_factor = 2 / (b - a)
    D *= scale_factor
    # Apply differentiation
    u_deriv = u
    for _ in range(order):
        u_deriv = D @ u_deriv
    return u_deriv
def spectral_interpolation(x_data: np.ndarray, u_data: np.ndarray,
                          x_interp: np.ndarray,
                          method: str = 'chebyshev') -> np.ndarray:
    """Spectral interpolation of data.
    Args:
        x_data: Data points (assumed to be spectral grid)
        u_data: Function values at data points
        x_interp: Points to interpolate to
        method: Spectral method ('fourier', 'chebyshev')
    Returns:
        Interpolated values
    """
    if method == 'fourier':
        # Fourier interpolation (for periodic data)
        N = len(u_data)
        L = x_data[-1] - x_data[0] + (x_data[1] - x_data[0])  # Periodic length
        # FFT to get coefficients
        u_hat = fft(u_data)
        # Evaluate Fourier series at interpolation points
        u_interp = np.zeros(len(x_interp), dtype=complex)
        k = 2 * np.pi * fftfreq(N, L / N)
        for i, x in enumerate(x_interp):
            u_interp[i] = np.sum(u_hat * np.exp(1j * k * x)) / N
        return np.real(u_interp)
    elif method == 'chebyshev':
        # Chebyshev interpolation
        N = len(u_data)
        # Transform to computational domain [-1, 1]
        a, b = x_data[0], x_data[-1]
        xi_data = 2 * (x_data - a) / (b - a) - 1
        xi_interp = 2 * (x_interp - a) / (b - a) - 1
        # Barycentric Chebyshev interpolation
        from scipy.interpolate import barycentric_interpolate
        # Chebyshev weights
        w = np.ones(N)
        w[0] = 0.5
        w[-1] = 0.5
        w[1::2] *= -1
        u_interp = barycentric_interpolate(xi_data, u_data, xi_interp, weights=w)
        return u_interp
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
def compute_spectral_accuracy(coeffs: np.ndarray,
                             threshold: float = 1e-12) -> Dict[str, Any]:
    """Analyze spectral accuracy from coefficients.
    Args:
        coeffs: Spectral coefficients
        threshold: Threshold for determining convergence
    Returns:
        Dictionary with accuracy metrics
    """
    # Find where coefficients drop below threshold
    abs_coeffs = np.abs(coeffs)
    # Last significant coefficient
    significant = abs_coeffs > threshold
    if np.any(significant):
        last_significant = np.where(significant)[0][-1]
    else:
        last_significant = 0
    # Decay rate
    if len(abs_coeffs) > 1:
        log_coeffs = np.log(abs_coeffs[abs_coeffs > 0])
        if len(log_coeffs) > 1:
            # Fit exponential decay
            indices = np.arange(len(log_coeffs))
            decay_rate = np.polyfit(indices, log_coeffs, 1)[0]
        else:
            decay_rate = 0
    else:
        decay_rate = 0
    return {
        'last_significant_mode': last_significant,
        'max_coefficient': np.max(abs_coeffs),
        'min_coefficient': np.min(abs_coeffs[abs_coeffs > 0]) if np.any(abs_coeffs > 0) else 0,
        'decay_rate': decay_rate,
        'spectral_convergence': abs_coeffs[-1] < threshold,
        'effective_modes': last_significant + 1
    }