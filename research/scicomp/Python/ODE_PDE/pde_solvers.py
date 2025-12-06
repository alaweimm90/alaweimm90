"""Partial Differential Equation Solvers.
This module provides comprehensive PDE solving capabilities including
finite difference, finite element, and spectral methods for various
types of PDEs including heat, wave, Poisson, and Navier-Stokes equations.
Classes:
    PDESolver: Base class for PDE solvers
    FiniteDifferencePDE: Finite difference method base
    HeatEquationSolver: Heat/diffusion equation solver
    WaveEquationSolver: Wave equation solver
    PoissonSolver: Poisson equation solver
    AdvectionDiffusionSolver: Advection-diffusion equation
    NavierStokesSolver: Navier-Stokes equations
Functions:
    solve_pde: General PDE solving interface
    solve_heat_equation: Heat equation solver
    solve_wave_equation: Wave equation solver
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.linalg import solve
@dataclass
class PDEResult:
    """Result of PDE solution."""
    x: np.ndarray
    t: Optional[np.ndarray]
    u: np.ndarray
    success: bool
    message: str
    iterations: int = 0
    residual: float = 0.0
    solve_time: float = 0.0
    method: str = ""
@dataclass
class PDEOptions:
    """Options for PDE solvers."""
    method: str = 'finite_difference'
    time_scheme: str = 'implicit'  # explicit, implicit, crank_nicolson
    max_iterations: int = 1000
    tolerance: float = 1e-8
    cfl_number: float = 0.5
    theta: float = 0.5  # For theta methods (0=explicit, 0.5=CN, 1=implicit)
    boundary_method: str = 'direct'  # direct, penalty, lagrange
    stabilization: Optional[str] = None  # upwind, supg, etc.
class PDESolver(ABC):
    """Abstract base class for PDE solvers."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 options: PDEOptions):
        """Initialize PDE solver.
        Args:
            domain: Spatial domain specification
            boundary_conditions: Boundary conditions
            options: Solver options
        """
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        self.options = options
        # Extract domain information
        self.x = domain.get('x', np.linspace(0, 1, 101))
        self.y = domain.get('y', None)
        self.z = domain.get('z', None)
        # Determine problem dimension
        self.ndim = 1
        if self.y is not None:
            self.ndim = 2
        if self.z is not None:
            self.ndim = 3
        # Grid spacing
        self.dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        self.dy = self.y[1] - self.y[0] if self.y is not None and len(self.y) > 1 else 1.0
        self.dz = self.z[1] - self.z[0] if self.z is not None and len(self.z) > 1 else 1.0
        # Problem size
        self.nx = len(self.x)
        self.ny = len(self.y) if self.y is not None else 1
        self.nz = len(self.z) if self.z is not None else 1
    @abstractmethod
    def solve_steady(self, source_term: Optional[Callable] = None) -> PDEResult:
        """Solve steady-state PDE."""
        pass
    @abstractmethod
    def solve_transient(self, initial_condition: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Solve time-dependent PDE."""
        pass
    def _apply_boundary_conditions(self, matrix: sparse.spmatrix,
                                  rhs: np.ndarray) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply boundary conditions to system."""
        # This is a simplified implementation
        # In practice, each BC type requires specific handling
        if 'dirichlet' in self.boundary_conditions:
            dirichlet_bc = self.boundary_conditions['dirichlet']
            for node, value in dirichlet_bc.items():
                if isinstance(node, int) and 0 <= node < matrix.shape[0]:
                    # Set row to identity
                    matrix.data[matrix.indptr[node]:matrix.indptr[node+1]] = 0
                    matrix[node, node] = 1.0
                    rhs[node] = value
        return matrix, rhs
class FiniteDifferencePDE(PDESolver):
    """Finite difference method for PDEs."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 options: PDEOptions):
        super().__init__(domain, boundary_conditions, options)
    def build_laplacian_1d(self) -> sparse.spmatrix:
        """Build 1D Laplacian matrix."""
        n = self.nx
        data = np.ones((3, n))
        data[0, :] = 1  # Upper diagonal
        data[1, :] = -2  # Main diagonal
        data[2, :] = 1  # Lower diagonal
        # Handle boundaries
        data[1, 0] = -2  # Left boundary
        data[1, -1] = -2  # Right boundary
        offsets = [1, 0, -1]
        L = sparse.diags(data, offsets, shape=(n, n), format='csr')
        L = L / (self.dx**2)
        return L
    def build_laplacian_2d(self) -> sparse.spmatrix:
        """Build 2D Laplacian matrix using finite differences."""
        nx, ny = self.nx, self.ny
        n = nx * ny
        # Build 2D Laplacian as Kronecker sum
        # ∇² = ∂²/∂x² + ∂²/∂y²
        # 1D second derivative operators
        e = np.ones(nx)
        Lx = sparse.diags([e[1:], -2*e, e[:-1]], [1, 0, -1], shape=(nx, nx))
        Lx = Lx / (self.dx**2)
        e = np.ones(ny)
        Ly = sparse.diags([e[1:], -2*e, e[:-1]], [1, 0, -1], shape=(ny, ny))
        Ly = Ly / (self.dy**2)
        # 2D Laplacian via Kronecker sum
        Ix = sparse.eye(nx)
        Iy = sparse.eye(ny)
        L = sparse.kron(Iy, Lx) + sparse.kron(Ly, Ix)
        return L.tocsr()
    def build_gradient_1d(self) -> sparse.spmatrix:
        """Build 1D gradient matrix (central differences)."""
        n = self.nx
        data = np.ones((2, n))
        data[0, :] = 1  # Upper diagonal
        data[1, :] = -1  # Lower diagonal
        offsets = [1, -1]
        G = sparse.diags(data, offsets, shape=(n, n), format='csr')
        G = G / (2 * self.dx)
        return G
class HeatEquationSolver(FiniteDifferencePDE):
    """Solver for heat/diffusion equation: ∂u/∂t = α∇²u + f."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 thermal_diffusivity: float = 1.0,
                 options: PDEOptions = PDEOptions()):
        super().__init__(domain, boundary_conditions, options)
        self.alpha = thermal_diffusivity
    def solve_steady(self, source_term: Optional[Callable] = None) -> PDEResult:
        """Solve steady-state heat equation: -α∇²u = f."""
        import time
        start_time = time.time()
        if self.ndim == 1:
            # Build Laplacian
            L = self.build_laplacian_1d()
            # Right-hand side
            rhs = np.zeros(self.nx)
            if source_term:
                rhs = source_term(self.x)
            # Apply boundary conditions
            A = -self.alpha * L
            A, rhs = self._apply_boundary_conditions(A, rhs)
            # Solve
            u = spla.spsolve(A, rhs)
        elif self.ndim == 2:
            # Build 2D Laplacian
            L = self.build_laplacian_2d()
            # Right-hand side
            rhs = np.zeros(self.nx * self.ny)
            if source_term:
                X, Y = np.meshgrid(self.x, self.y, indexing='ij')
                f_vals = source_term(X, Y)
                rhs = f_vals.ravel()
            # Apply boundary conditions
            A = -self.alpha * L
            A, rhs = self._apply_boundary_conditions(A, rhs)
            # Solve
            u = spla.spsolve(A, rhs)
            u = u.reshape((self.nx, self.ny))
        else:
            raise NotImplementedError("3D heat equation not implemented")
        solve_time = time.time() - start_time
        return PDEResult(
            x=self.x,
            t=None,
            u=u,
            success=True,
            message="Steady-state heat equation solved successfully",
            solve_time=solve_time,
            method="finite_difference"
        )
    def solve_transient(self, initial_condition: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Solve transient heat equation."""
        import time
        start_time = time.time()
        t0, tf = time_span
        nt = int((tf - t0) / dt) + 1
        t = np.linspace(t0, tf, nt)
        if self.ndim == 1:
            # Initialize solution
            u = np.zeros((nt, self.nx))
            u[0] = initial_condition(self.x)
            # Build spatial operator
            L = self.build_laplacian_1d()
            M = sparse.eye(self.nx)  # Mass matrix
            # Time stepping
            theta = self.options.theta
            for n in range(nt - 1):
                # Source term
                f_current = np.zeros(self.nx)
                f_next = np.zeros(self.nx)
                if source_term:
                    f_current = source_term(self.x, t[n])
                    f_next = source_term(self.x, t[n+1])
                # Theta method: (M - θ*dt*α*L)*u^{n+1} = (M + (1-θ)*dt*α*L)*u^n + dt*θ*f^{n+1} + dt*(1-θ)*f^n
                A = M - theta * dt * self.alpha * L
                b = (M + (1 - theta) * dt * self.alpha * L) @ u[n] + dt * (theta * f_next + (1 - theta) * f_current)
                # Apply boundary conditions
                A, b = self._apply_boundary_conditions(A, b)
                # Solve
                u[n+1] = spla.spsolve(A, b)
        elif self.ndim == 2:
            # Initialize solution
            u = np.zeros((nt, self.nx, self.ny))
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            u[0] = initial_condition(X, Y)
            # Build spatial operator
            L = self.build_laplacian_2d()
            M = sparse.eye(self.nx * self.ny)
            # Time stepping
            theta = self.options.theta
            for n in range(nt - 1):
                u_flat = u[n].ravel()
                # Source term
                f_current = np.zeros(self.nx * self.ny)
                f_next = np.zeros(self.nx * self.ny)
                if source_term:
                    f_current = source_term(X, Y, t[n]).ravel()
                    f_next = source_term(X, Y, t[n+1]).ravel()
                # Theta method
                A = M - theta * dt * self.alpha * L
                b = (M + (1 - theta) * dt * self.alpha * L) @ u_flat + dt * (theta * f_next + (1 - theta) * f_current)
                # Apply boundary conditions
                A, b = self._apply_boundary_conditions(A, b)
                # Solve
                u_next = spla.spsolve(A, b)
                u[n+1] = u_next.reshape((self.nx, self.ny))
        else:
            raise NotImplementedError("3D transient heat equation not implemented")
        solve_time = time.time() - start_time
        return PDEResult(
            x=self.x,
            t=t,
            u=u,
            success=True,
            message="Transient heat equation solved successfully",
            solve_time=solve_time,
            method="finite_difference"
        )
class WaveEquationSolver(FiniteDifferencePDE):
    """Solver for wave equation: ∂²u/∂t² = c²∇²u + f."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 wave_speed: float = 1.0,
                 options: PDEOptions = PDEOptions()):
        super().__init__(domain, boundary_conditions, options)
        self.c = wave_speed
    def solve_steady(self, source_term: Optional[Callable] = None) -> PDEResult:
        """Wave equation has no meaningful steady state."""
        raise ValueError("Wave equation has no steady-state solution")
    def solve_transient(self, initial_condition: Callable,
                       initial_velocity: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Solve transient wave equation using central differences."""
        import time
        start_time = time.time()
        t0, tf = time_span
        nt = int((tf - t0) / dt) + 1
        t = np.linspace(t0, tf, nt)
        if self.ndim == 1:
            # Check CFL condition
            cfl = self.c * dt / self.dx
            if cfl > 1.0:
                warnings.warn(f"CFL number {cfl:.3f} > 1.0, solution may be unstable")
            # Initialize solution
            u = np.zeros((nt, self.nx))
            u[0] = initial_condition(self.x)
            # First time step using initial velocity
            u_dot_0 = initial_velocity(self.x)
            # Build Laplacian
            L = self.build_laplacian_1d()
            # Second-order accurate first step
            u[1] = u[0] + dt * u_dot_0 + 0.5 * dt**2 * (self.c**2 * L @ u[0])
            # Time stepping with central differences
            for n in range(1, nt - 1):
                # Source term
                f = np.zeros(self.nx)
                if source_term:
                    f = source_term(self.x, t[n])
                # Central difference in time: u^{n+1} = 2u^n - u^{n-1} + dt²(c²∇²u^n + f^n)
                u[n+1] = 2 * u[n] - u[n-1] + dt**2 * (self.c**2 * L @ u[n] + f)
                # Apply boundary conditions (simplified)
                if 'dirichlet' in self.boundary_conditions:
                    dirichlet_bc = self.boundary_conditions['dirichlet']
                    for node, value in dirichlet_bc.items():
                        if isinstance(node, int) and 0 <= node < self.nx:
                            u[n+1, node] = value
        elif self.ndim == 2:
            # Check CFL condition
            cfl = self.c * dt * np.sqrt(1/self.dx**2 + 1/self.dy**2)
            if cfl > 1.0:
                warnings.warn(f"CFL number {cfl:.3f} > 1.0, solution may be unstable")
            # Initialize solution
            u = np.zeros((nt, self.nx, self.ny))
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            u[0] = initial_condition(X, Y)
            # First time step
            u_dot_0 = initial_velocity(X, Y)
            L = self.build_laplacian_2d()
            u_flat = u[0].ravel()
            u_dot_flat = u_dot_0.ravel()
            u_1_flat = u_flat + dt * u_dot_flat + 0.5 * dt**2 * (self.c**2 * L @ u_flat)
            u[1] = u_1_flat.reshape((self.nx, self.ny))
            # Time stepping
            for n in range(1, nt - 1):
                u_n_flat = u[n].ravel()
                u_nm1_flat = u[n-1].ravel()
                # Source term
                f = np.zeros(self.nx * self.ny)
                if source_term:
                    f = source_term(X, Y, t[n]).ravel()
                # Central difference
                u_np1_flat = 2 * u_n_flat - u_nm1_flat + dt**2 * (self.c**2 * L @ u_n_flat + f)
                u[n+1] = u_np1_flat.reshape((self.nx, self.ny))
        else:
            raise NotImplementedError("3D wave equation not implemented")
        solve_time = time.time() - start_time
        return PDEResult(
            x=self.x,
            t=t,
            u=u,
            success=True,
            message="Wave equation solved successfully",
            solve_time=solve_time,
            method="finite_difference"
        )
class PoissonSolver(FiniteDifferencePDE):
    """Solver for Poisson equation: -∇²u = f."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 options: PDEOptions = PDEOptions()):
        super().__init__(domain, boundary_conditions, options)
    def solve_steady(self, source_term: Callable) -> PDEResult:
        """Solve Poisson equation."""
        import time
        start_time = time.time()
        if self.ndim == 1:
            # Build Laplacian
            L = self.build_laplacian_1d()
            # Right-hand side
            rhs = source_term(self.x)
            # Apply boundary conditions
            A = -L  # -∇²u = f
            A, rhs = self._apply_boundary_conditions(A, rhs)
            # Solve
            u = spla.spsolve(A, rhs)
        elif self.ndim == 2:
            # Build 2D Laplacian
            L = self.build_laplacian_2d()
            # Right-hand side
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            f_vals = source_term(X, Y)
            rhs = f_vals.ravel()
            # Apply boundary conditions
            A = -L
            A, rhs = self._apply_boundary_conditions(A, rhs)
            # Solve
            u = spla.spsolve(A, rhs)
            u = u.reshape((self.nx, self.ny))
        else:
            raise NotImplementedError("3D Poisson equation not implemented")
        solve_time = time.time() - start_time
        return PDEResult(
            x=self.x,
            t=None,
            u=u,
            success=True,
            message="Poisson equation solved successfully",
            solve_time=solve_time,
            method="finite_difference"
        )
    def solve_transient(self, initial_condition: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Poisson equation is elliptic, no time dependence."""
        raise ValueError("Poisson equation is not time-dependent")
class AdvectionDiffusionSolver(FiniteDifferencePDE):
    """Solver for advection-diffusion equation: ∂u/∂t + v·∇u = D∇²u + f."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 velocity: Union[float, np.ndarray, Callable],
                 diffusivity: float = 1.0,
                 options: PDEOptions = PDEOptions()):
        super().__init__(domain, boundary_conditions, options)
        self.velocity = velocity
        self.D = diffusivity
    def solve_steady(self, source_term: Optional[Callable] = None) -> PDEResult:
        """Solve steady advection-diffusion: v·∇u = D∇²u + f."""
        import time
        start_time = time.time()
        if self.ndim == 1:
            # Build operators
            L = self.build_laplacian_1d()  # Diffusion
            G = self.build_gradient_1d()   # Advection
            # Velocity field
            if isinstance(self.velocity, (int, float)):
                v = np.full(self.nx, self.velocity)
            elif callable(self.velocity):
                v = self.velocity(self.x)
            else:
                v = np.asarray(self.velocity)
            # Build advection matrix
            V = sparse.diags(v, format='csr')
            A_adv = V @ G
            # Combined operator: -D∇² + v·∇
            A = -self.D * L + A_adv
            # Right-hand side
            rhs = np.zeros(self.nx)
            if source_term:
                rhs = source_term(self.x)
            # Apply boundary conditions
            A, rhs = self._apply_boundary_conditions(A, rhs)
            # Solve
            u = spla.spsolve(A, rhs)
        else:
            raise NotImplementedError("2D/3D advection-diffusion not implemented")
        solve_time = time.time() - start_time
        return PDEResult(
            x=self.x,
            t=None,
            u=u,
            success=True,
            message="Steady advection-diffusion solved successfully",
            solve_time=solve_time,
            method="finite_difference"
        )
    def solve_transient(self, initial_condition: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Solve transient advection-diffusion equation."""
        # Implementation similar to heat equation but with advection term
        # For brevity, implementing simplified version
        t0, tf = time_span
        nt = int((tf - t0) / dt) + 1
        t = np.linspace(t0, tf, nt)
        if self.ndim == 1:
            # Initialize
            u = np.zeros((nt, self.nx))
            u[0] = initial_condition(self.x)
            # Build operators
            L = self.build_laplacian_1d()
            G = self.build_gradient_1d()
            M = sparse.eye(self.nx)
            # Velocity
            if isinstance(self.velocity, (int, float)):
                v = np.full(self.nx, self.velocity)
            elif callable(self.velocity):
                v = self.velocity(self.x)
            else:
                v = np.asarray(self.velocity)
            V = sparse.diags(v, format='csr')
            A_adv = V @ G
            # Time stepping (implicit)
            for n in range(nt - 1):
                # Source term
                f = np.zeros(self.nx)
                if source_term:
                    f = source_term(self.x, t[n+1])
                # Implicit scheme: (M + dt*(A_adv - D*L))*u^{n+1} = M*u^n + dt*f
                A = M + dt * (A_adv - self.D * L)
                b = M @ u[n] + dt * f
                # Apply boundary conditions
                A, b = self._apply_boundary_conditions(A, b)
                # Solve
                u[n+1] = spla.spsolve(A, b)
        else:
            raise NotImplementedError("2D/3D transient advection-diffusion not implemented")
        return PDEResult(
            x=self.x,
            t=t,
            u=u,
            success=True,
            message="Transient advection-diffusion solved successfully",
            method="finite_difference"
        )
class NavierStokesSolver(FiniteDifferencePDE):
    """Simplified Navier-Stokes solver for incompressible flow."""
    def __init__(self, domain: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, Any],
                 viscosity: float = 1.0,
                 density: float = 1.0,
                 options: PDEOptions = PDEOptions()):
        super().__init__(domain, boundary_conditions, options)
        self.nu = viscosity
        self.rho = density
    def solve_steady(self, source_term: Optional[Callable] = None) -> PDEResult:
        """Solve steady Navier-Stokes (simplified)."""
        # This is a placeholder for the complex Navier-Stokes implementation
        # Full implementation would require pressure-velocity coupling (SIMPLE, etc.)
        raise NotImplementedError("Steady Navier-Stokes solver not fully implemented")
    def solve_transient(self, initial_condition: Callable,
                       time_span: Tuple[float, float],
                       dt: float,
                       source_term: Optional[Callable] = None) -> PDEResult:
        """Solve transient Navier-Stokes (simplified)."""
        raise NotImplementedError("Transient Navier-Stokes solver not fully implemented")
# Main PDE solving interface
def solve_pde(pde_type: str,
              domain: Dict[str, np.ndarray],
              boundary_conditions: Dict[str, Any],
              initial_condition: Optional[Callable] = None,
              time_span: Optional[Tuple[float, float]] = None,
              dt: Optional[float] = None,
              source_term: Optional[Callable] = None,
              parameters: Optional[Dict[str, Any]] = None,
              options: PDEOptions = PDEOptions()) -> PDEResult:
    """General PDE solving interface.
    Args:
        pde_type: Type of PDE ('heat', 'wave', 'poisson', 'advection_diffusion')
        domain: Spatial domain specification
        boundary_conditions: Boundary conditions
        initial_condition: Initial condition (for time-dependent)
        time_span: Time interval (for time-dependent)
        dt: Time step (for time-dependent)
        source_term: Source term function
        parameters: PDE-specific parameters
        options: Solver options
    Returns:
        PDEResult containing solution
    """
    if parameters is None:
        parameters = {}
    # Create appropriate solver
    if pde_type == 'heat':
        alpha = parameters.get('thermal_diffusivity', 1.0)
        solver = HeatEquationSolver(domain, boundary_conditions, alpha, options)
    elif pde_type == 'wave':
        c = parameters.get('wave_speed', 1.0)
        solver = WaveEquationSolver(domain, boundary_conditions, c, options)
    elif pde_type == 'poisson':
        solver = PoissonSolver(domain, boundary_conditions, options)
    elif pde_type == 'advection_diffusion':
        velocity = parameters.get('velocity', 1.0)
        diffusivity = parameters.get('diffusivity', 1.0)
        solver = AdvectionDiffusionSolver(domain, boundary_conditions,
                                        velocity, diffusivity, options)
    elif pde_type == 'navier_stokes':
        viscosity = parameters.get('viscosity', 1.0)
        density = parameters.get('density', 1.0)
        solver = NavierStokesSolver(domain, boundary_conditions,
                                  viscosity, density, options)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    # Solve
    if time_span is None or initial_condition is None:
        # Steady-state
        return solver.solve_steady(source_term)
    else:
        # Time-dependent
        if dt is None:
            dt = (time_span[1] - time_span[0]) / 100  # Default 100 time steps
        if pde_type == 'wave':
            # Wave equation needs initial velocity
            initial_velocity = parameters.get('initial_velocity',
                                            lambda *args: np.zeros_like(initial_condition(*args)))
            return solver.solve_transient(initial_condition, initial_velocity,
                                        time_span, dt, source_term)
        else:
            return solver.solve_transient(initial_condition, time_span, dt, source_term)
# Convenience functions
def solve_heat_equation(domain: Dict[str, np.ndarray],
                       boundary_conditions: Dict[str, Any],
                       initial_condition: Optional[Callable] = None,
                       time_span: Optional[Tuple[float, float]] = None,
                       thermal_diffusivity: float = 1.0,
                       source_term: Optional[Callable] = None) -> PDEResult:
    """Solve heat equation with default settings."""
    parameters = {'thermal_diffusivity': thermal_diffusivity}
    return solve_pde('heat', domain, boundary_conditions, initial_condition,
                    time_span, source_term=source_term, parameters=parameters)
def solve_wave_equation(domain: Dict[str, np.ndarray],
                       boundary_conditions: Dict[str, Any],
                       initial_condition: Callable,
                       initial_velocity: Callable,
                       time_span: Tuple[float, float],
                       wave_speed: float = 1.0,
                       source_term: Optional[Callable] = None) -> PDEResult:
    """Solve wave equation with default settings."""
    parameters = {'wave_speed': wave_speed, 'initial_velocity': initial_velocity}
    return solve_pde('wave', domain, boundary_conditions, initial_condition,
                    time_span, source_term=source_term, parameters=parameters)