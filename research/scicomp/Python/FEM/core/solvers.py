"""
Advanced Solvers for Finite Element Analysis
Comprehensive solution algorithms including static, dynamic, nonlinear,
and eigenvalue solvers for finite element systems.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve, eigsh, cg, gmres
import warnings
from .assembly import GlobalAssembly
class StaticSolver:
    """
    Static equilibrium solver for linear problems.
    Features:
    - Direct and iterative solvers
    - Multiple boundary condition methods
    - Solution verification
    """
    def __init__(self, assembly: GlobalAssembly):
        """
        Initialize static solver.
        Parameters:
            assembly: Global assembly system
        """
        self.assembly = assembly
        self.solution: Optional[np.ndarray] = None
        self.convergence_history: List[float] = []
    def solve_direct(self, method: str = 'lu') -> np.ndarray:
        """
        Solve using direct method.
        Parameters:
            method: Solution method ('lu', 'cholesky')
        Returns:
            Displacement vector
        """
        print(f"Solving using direct method: {method}")
        K = self.assembly.global_stiffness
        f = self.assembly.load_vector
        if K is None or f is None:
            raise ValueError("System matrices not assembled")
        # Apply boundary conditions
        K_mod, f_mod = self._apply_boundary_conditions(K, f)
        if method.lower() == 'lu':
            self.solution = spsolve(K_mod, f_mod)
        elif method.lower() == 'cholesky':
            # Check if matrix is positive definite
            try:
                factor = sparse.linalg.spsolve_triangular
                self.solution = spsolve(K_mod, f_mod)
            except Exception as e:
                warnings.warn(f"Cholesky failed, falling back to LU: {e}")
                self.solution = spsolve(K_mod, f_mod)
        else:
            raise ValueError(f"Unknown direct method: {method}")
        print(f"Direct solution completed. Max displacement: {np.max(np.abs(self.solution)):.6e}")
        return self.solution
    def solve_iterative(self, method: str = 'cg', tolerance: float = 1e-8,
                       max_iterations: int = None) -> np.ndarray:
        """
        Solve using iterative method.
        Parameters:
            method: Iterative method ('cg', 'gmres', 'bicgstab')
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        Returns:
            Displacement vector
        """
        print(f"Solving using iterative method: {method}")
        K = self.assembly.global_stiffness
        f = self.assembly.load_vector
        if K is None or f is None:
            raise ValueError("System matrices not assembled")
        # Apply boundary conditions
        K_mod, f_mod = self._apply_boundary_conditions(K, f)
        if max_iterations is None:
            max_iterations = K_mod.shape[0]
        # Callback for convergence monitoring
        self.convergence_history = []
        def callback(x):
            residual = np.linalg.norm(K_mod @ x - f_mod)
            self.convergence_history.append(residual)
        try:
            if method.lower() == 'cg':
                self.solution, info = cg(K_mod, f_mod, tol=tolerance,
                                       maxiter=max_iterations, callback=callback)
            elif method.lower() == 'gmres':
                self.solution, info = gmres(K_mod, f_mod, tol=tolerance,
                                          maxiter=max_iterations, callback=callback)
            elif method.lower() == 'bicgstab':
                from scipy.sparse.linalg import bicgstab
                self.solution, info = bicgstab(K_mod, f_mod, tol=tolerance,
                                             maxiter=max_iterations, callback=callback)
            else:
                raise ValueError(f"Unknown iterative method: {method}")
            if info > 0:
                warnings.warn(f"Iterative solver did not converge in {info} iterations")
            elif info < 0:
                raise RuntimeError(f"Iterative solver failed with error code {info}")
        except Exception as e:
            warnings.warn(f"Iterative solver failed: {e}, falling back to direct solver")
            return self.solve_direct()
        print(f"Iterative solution completed in {len(self.convergence_history)} iterations")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return self.solution
    def _apply_boundary_conditions(self, K: sparse.csr_matrix, f: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Apply boundary conditions using penalty method."""
        K_mod = K.copy()
        f_mod = f.copy()
        # Penalty method
        penalty = 1e12 * np.max(np.abs(K.data))
        for global_dof, prescribed_value in zip(self.assembly.prescribed_dofs,
                                              self.assembly.prescribed_values):
            K_mod[global_dof, global_dof] += penalty
            f_mod[global_dof] += penalty * prescribed_value
        return K_mod, f_mod
class DynamicSolver:
    """
    Dynamic analysis solver for time-dependent problems.
    Features:
    - Modal analysis
    - Time integration (Newmark, central difference)
    - Frequency response analysis
    """
    def __init__(self, assembly: GlobalAssembly):
        """
        Initialize dynamic solver.
        Parameters:
            assembly: Global assembly system
        """
        self.assembly = assembly
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.time_history: Optional[Dict[str, np.ndarray]] = None
    def modal_analysis(self, num_modes: int = 10, which: str = 'SM') -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform modal analysis to find natural frequencies and mode shapes.
        Parameters:
            num_modes: Number of modes to extract
            which: Which eigenvalues to find ('SM'=smallest magnitude, 'LM'=largest)
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        print(f"Performing modal analysis for {num_modes} modes...")
        K = self.assembly.global_stiffness
        M = self.assembly.global_mass
        if K is None or M is None:
            raise ValueError("Stiffness and mass matrices not assembled")
        # Apply boundary conditions by removing constrained DOFs
        free_dofs = self._get_free_dofs()
        K_free = K[np.ix_(free_dofs, free_dofs)]
        M_free = M[np.ix_(free_dofs, free_dofs)]
        try:
            # Solve generalized eigenvalue problem: K φ = λ M φ
            eigenvals, eigenvecs = eigsh(K_free, M=M_free, k=num_modes, which=which)
            # Expand eigenvectors to full size
            full_eigenvectors = np.zeros((self.assembly.num_dofs, num_modes))
            full_eigenvectors[free_dofs, :] = eigenvecs
            # Sort by frequency
            idx = np.argsort(eigenvals)
            self.eigenvalues = eigenvals[idx]
            self.eigenvectors = full_eigenvectors[:, idx]
            # Convert to frequencies (Hz)
            frequencies = np.sqrt(np.abs(self.eigenvalues)) / (2 * np.pi)
            print(f"Modal analysis completed:")
            for i, freq in enumerate(frequencies):
                print(f"  Mode {i+1}: {freq:.2f} Hz")
            return self.eigenvalues, self.eigenvectors
        except Exception as e:
            raise RuntimeError(f"Modal analysis failed: {e}")
    def newmark_integration(self, time_span: Tuple[float, float], time_step: float,
                           initial_conditions: Dict[str, np.ndarray],
                           force_function: Callable[[float], np.ndarray],
                           beta: float = 0.25, gamma: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Perform time integration using Newmark method.
        Parameters:
            time_span: (start_time, end_time)
            time_step: Time step size
            initial_conditions: {'displacement': u0, 'velocity': v0, 'acceleration': a0}
            force_function: Function returning force vector at given time
            beta, gamma: Newmark parameters
        Returns:
            Time history dictionary
        """
        print(f"Starting Newmark time integration...")
        print(f"Time span: {time_span[0]:.3f} to {time_span[1]:.3f} s")
        print(f"Time step: {time_step:.6f} s")
        t_start, t_end = time_span
        times = np.arange(t_start, t_end + time_step, time_step)
        num_steps = len(times)
        K = self.assembly.global_stiffness
        M = self.assembly.global_mass
        if K is None or M is None:
            raise ValueError("Stiffness and mass matrices not assembled")
        # Initialize arrays
        n_dof = self.assembly.num_dofs
        displacement = np.zeros((n_dof, num_steps))
        velocity = np.zeros((n_dof, num_steps))
        acceleration = np.zeros((n_dof, num_steps))
        # Set initial conditions
        displacement[:, 0] = initial_conditions.get('displacement', np.zeros(n_dof))
        velocity[:, 0] = initial_conditions.get('velocity', np.zeros(n_dof))
        acceleration[:, 0] = initial_conditions.get('acceleration', np.zeros(n_dof))
        # Newmark constants
        c1 = 1.0 / (beta * time_step**2)
        c2 = gamma / (beta * time_step)
        c3 = 1.0 / (beta * time_step)
        c4 = 1.0 / (2.0 * beta) - 1.0
        c5 = gamma / beta - 1.0
        c6 = time_step * 0.5 * (gamma / beta - 2.0)
        # Effective stiffness matrix
        K_eff = K + c1 * M
        # Apply boundary conditions to effective stiffness
        K_eff_mod, _ = self._apply_boundary_conditions_dynamic(K_eff)
        # Time integration loop
        for i in range(1, num_steps):
            t = times[i]
            # External force at current time
            f_ext = force_function(t)
            # Effective force
            f_eff = (f_ext + M @ (c1 * displacement[:, i-1] +
                                 c3 * velocity[:, i-1] +
                                 c4 * acceleration[:, i-1]))
            # Apply boundary conditions to force
            f_eff = self._apply_boundary_conditions_force(f_eff)
            # Solve for displacement
            displacement[:, i] = spsolve(K_eff_mod, f_eff)
            # Apply displacement boundary conditions
            for global_dof, prescribed_value in zip(self.assembly.prescribed_dofs,
                                                  self.assembly.prescribed_values):
                displacement[global_dof, i] = prescribed_value
            # Update velocity and acceleration
            acceleration[:, i] = (c1 * (displacement[:, i] - displacement[:, i-1]) -
                                c3 * velocity[:, i-1] - c4 * acceleration[:, i-1])
            velocity[:, i] = (velocity[:, i-1] +
                            time_step * ((1.0 - gamma) * acceleration[:, i-1] +
                                       gamma * acceleration[:, i]))
        self.time_history = {
            'time': times,
            'displacement': displacement,
            'velocity': velocity,
            'acceleration': acceleration
        }
        print(f"Time integration completed: {num_steps} steps")
        return self.time_history
    def frequency_response(self, frequency_range: Tuple[float, float],
                          num_frequencies: int, damping_ratio: float = 0.02,
                          excitation_dof: int = 0, response_dof: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute frequency response function.
        Parameters:
            frequency_range: (min_freq, max_freq) in Hz
            num_frequencies: Number of frequency points
            damping_ratio: Modal damping ratio
            excitation_dof: DOF where force is applied
            response_dof: DOF where response is measured
        Returns:
            Frequency response data
        """
        print(f"Computing frequency response...")
        K = self.assembly.global_stiffness
        M = self.assembly.global_mass
        if K is None or M is None:
            raise ValueError("Stiffness and mass matrices not assembled")
        # Frequency vector
        f_min, f_max = frequency_range
        frequencies = np.linspace(f_min, f_max, num_frequencies)
        omega = 2 * np.pi * frequencies
        # Initialize response
        magnitude = np.zeros(num_frequencies)
        phase = np.zeros(num_frequencies)
        # Apply boundary conditions
        free_dofs = self._get_free_dofs()
        K_free = K[np.ix_(free_dofs, free_dofs)]
        M_free = M[np.ix_(free_dofs, free_dofs)]
        # Map global DOFs to free DOFs
        if excitation_dof in free_dofs and response_dof in free_dofs:
            exc_idx = np.where(free_dofs == excitation_dof)[0][0]
            resp_idx = np.where(free_dofs == response_dof)[0][0]
        else:
            raise ValueError("Excitation or response DOF is constrained")
        # Frequency sweep
        for i, w in enumerate(omega):
            # Proportional damping matrix
            C_free = 2 * damping_ratio * np.sqrt(np.diag(K_free.diagonal()) *
                                               np.diag(M_free.diagonal()))
            C_free = sparse.diags(C_free)
            # Dynamic stiffness matrix
            Z = K_free - w**2 * M_free + 1j * w * C_free
            # Unit force vector
            f_unit = np.zeros(len(free_dofs), dtype=complex)
            f_unit[exc_idx] = 1.0
            # Solve for response
            try:
                response = spsolve(Z, f_unit)
                transfer_function = response[resp_idx]
                magnitude[i] = np.abs(transfer_function)
                phase[i] = np.angle(transfer_function, deg=True)
            except Exception as e:
                warnings.warn(f"Failed to solve at frequency {frequencies[i]:.2f} Hz: {e}")
                magnitude[i] = 0.0
                phase[i] = 0.0
        return {
            'frequency': frequencies,
            'magnitude': magnitude,
            'phase': phase
        }
    def _get_free_dofs(self) -> np.ndarray:
        """Get unconstrained degrees of freedom."""
        all_dofs = np.arange(self.assembly.num_dofs)
        constrained_dofs = np.array(self.assembly.prescribed_dofs)
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
        return free_dofs
    def _apply_boundary_conditions_dynamic(self, K: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, None]:
        """Apply boundary conditions for dynamic analysis."""
        K_mod = K.copy()
        penalty = 1e12 * np.max(np.abs(K.data))
        for global_dof in self.assembly.prescribed_dofs:
            K_mod[global_dof, global_dof] += penalty
        return K_mod, None
    def _apply_boundary_conditions_force(self, f: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to force vector."""
        f_mod = f.copy()
        penalty = 1e12
        for global_dof, prescribed_value in zip(self.assembly.prescribed_dofs,
                                              self.assembly.prescribed_values):
            f_mod[global_dof] += penalty * prescribed_value
        return f_mod
class NonlinearSolver:
    """
    Nonlinear solver using Newton-Raphson iteration.
    Features:
    - Material and geometric nonlinearity
    - Arc-length methods
    - Line search algorithms
    """
    def __init__(self, assembly: GlobalAssembly):
        """
        Initialize nonlinear solver.
        Parameters:
            assembly: Global assembly system
        """
        self.assembly = assembly
        self.convergence_history: List[Dict[str, float]] = []
    def newton_raphson(self, initial_guess: Optional[np.ndarray] = None,
                      max_iterations: int = 20, tolerance: float = 1e-6,
                      line_search: bool = True) -> np.ndarray:
        """
        Solve nonlinear system using Newton-Raphson method.
        Parameters:
            initial_guess: Initial displacement guess
            max_iterations: Maximum Newton iterations
            tolerance: Convergence tolerance
            line_search: Enable line search
        Returns:
            Converged displacement vector
        """
        print(f"Starting Newton-Raphson iteration...")
        if initial_guess is None:
            u = np.zeros(self.assembly.num_dofs)
        else:
            u = initial_guess.copy()
        self.convergence_history = []
        for iteration in range(max_iterations):
            # Compute residual and tangent stiffness
            residual = self._compute_residual(u)
            tangent_stiffness = self._compute_tangent_stiffness(u)
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            displacement_norm = np.linalg.norm(u)
            # Store convergence data
            conv_data = {
                'iteration': iteration,
                'residual_norm': residual_norm,
                'displacement_norm': displacement_norm,
                'relative_residual': residual_norm / (np.linalg.norm(self.assembly.load_vector) + 1e-12)
            }
            self.convergence_history.append(conv_data)
            print(f"  Iteration {iteration}: Residual = {residual_norm:.2e}")
            if residual_norm < tolerance:
                print(f"Newton-Raphson converged in {iteration} iterations")
                return u
            # Solve for displacement increment
            try:
                delta_u = self._solve_tangent_system(tangent_stiffness, -residual)
            except Exception as e:
                raise RuntimeError(f"Failed to solve tangent system at iteration {iteration}: {e}")
            # Line search
            if line_search:
                alpha = self._line_search(u, delta_u, residual)
            else:
                alpha = 1.0
            # Update displacement
            u += alpha * delta_u
            print(f"    Step size: {alpha:.3f}")
        raise RuntimeError(f"Newton-Raphson failed to converge in {max_iterations} iterations")
    def _compute_residual(self, u: np.ndarray) -> np.ndarray:
        """Compute residual vector."""
        # For linear case, residual = K*u - f
        internal_force = self.assembly.global_stiffness @ u
        external_force = self.assembly.load_vector
        residual = internal_force - external_force
        # Apply boundary conditions
        for global_dof, prescribed_value in zip(self.assembly.prescribed_dofs,
                                              self.assembly.prescribed_values):
            residual[global_dof] = u[global_dof] - prescribed_value
        return residual
    def _compute_tangent_stiffness(self, u: np.ndarray) -> sparse.csr_matrix:
        """Compute tangent stiffness matrix."""
        # For linear case, tangent stiffness = K
        K_tangent = self.assembly.global_stiffness.copy()
        # Apply boundary conditions
        penalty = 1e12 * np.max(np.abs(K_tangent.data))
        for global_dof in self.assembly.prescribed_dofs:
            K_tangent[global_dof, global_dof] = penalty
        return K_tangent
    def _solve_tangent_system(self, K_tangent: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve tangent system."""
        return spsolve(K_tangent, rhs)
    def _line_search(self, u: np.ndarray, delta_u: np.ndarray, residual: np.ndarray,
                    max_trials: int = 10, reduction_factor: float = 0.5) -> float:
        """
        Perform line search to find optimal step size.
        Parameters:
            u: Current displacement
            delta_u: Displacement increment
            residual: Current residual
            max_trials: Maximum line search trials
            reduction_factor: Step size reduction factor
        Returns:
            Optimal step size
        """
        alpha = 1.0
        current_residual_norm = np.linalg.norm(residual)
        for trial in range(max_trials):
            # Try step
            u_trial = u + alpha * delta_u
            residual_trial = self._compute_residual(u_trial)
            trial_residual_norm = np.linalg.norm(residual_trial)
            # Check if step reduces residual
            if trial_residual_norm < current_residual_norm:
                return alpha
            # Reduce step size
            alpha *= reduction_factor
        # If line search fails, return small step
        return alpha