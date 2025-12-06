"""
State-Space Control Systems
Modern control theory implementations using state-space representation.
Includes system analysis, controllability/observability tests, and optimal control design.
"""
import numpy as np
from scipy import linalg
from scipy.integrate import odeint
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings
@dataclass
class StateSpaceMatrices:
    """State-space system matrices container."""
    A: np.ndarray  # System matrix (n x n)
    B: np.ndarray  # Input matrix (n x m)
    C: np.ndarray  # Output matrix (p x n)
    D: np.ndarray  # Feedthrough matrix (p x m)
    def __post_init__(self):
        """Validate matrix dimensions."""
        self.A = np.atleast_2d(self.A)
        self.B = np.atleast_2d(self.B)
        self.C = np.atleast_2d(self.C)
        self.D = np.atleast_2d(self.D)
        n, n_check = self.A.shape
        if n != n_check:
            raise ValueError("A matrix must be square")
        if self.B.shape[0] != n:
            raise ValueError("B matrix must have same number of rows as A")
        if self.C.shape[1] != n:
            raise ValueError("C matrix must have same number of columns as A")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D matrix dimensions must be (p x m)")
class StateSpaceSystem:
    """
    Linear time-invariant state-space system representation.
    Represents systems of the form:
        dx/dt = Ax + Bu
        y = Cx + Du
    Features:
    - System analysis (poles, zeros, stability)
    - Controllability and observability analysis
    - Time and frequency response simulation
    - System transformations
    Examples:
        >>> A = np.array([[-1, 1], [0, -2]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> sys = StateSpaceSystem(A, B, C, D)
        >>> print(f"System is stable: {sys.is_stable()}")
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        """
        Initialize state-space system.
        Parameters:
            A: System matrix (n x n)
            B: Input matrix (n x m)
            C: Output matrix (p x n)
            D: Feedthrough matrix (p x m)
        """
        self.matrices = StateSpaceMatrices(A, B, C, D)
        self._validate_system()
    def _validate_system(self):
        """Validate system matrices and properties."""
        # Check for numerical issues
        matrices = [self.A, self.B, self.C, self.D]
        for matrix in matrices:
            if np.any(np.isnan(matrix)):
                raise ValueError("System matrices contain NaN values")
            if np.any(np.isinf(matrix)):
                raise ValueError("System matrices contain infinite values")
    @property
    def A(self) -> np.ndarray:
        """System matrix."""
        return self.matrices.A
    @property
    def B(self) -> np.ndarray:
        """Input matrix."""
        return self.matrices.B
    @property
    def C(self) -> np.ndarray:
        """Output matrix."""
        return self.matrices.C
    @property
    def D(self) -> np.ndarray:
        """Feedthrough matrix."""
        return self.matrices.D
    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.A.shape[0]
    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return self.B.shape[1]
    @property
    def n_outputs(self) -> int:
        """Number of outputs."""
        return self.C.shape[0]
    def poles(self) -> np.ndarray:
        """Compute system poles (eigenvalues of A)."""
        return linalg.eigvals(self.A)
    def is_stable(self) -> bool:
        """Check if system is asymptotically stable."""
        poles = self.poles()
        return np.all(np.real(poles) < 0)
    def controllability_matrix(self) -> np.ndarray:
        """Compute controllability matrix."""
        n = self.n_states
        Wc = self.B.copy()
        A_power = np.eye(n)
        for i in range(1, n):
            A_power = A_power @ self.A
            Wc = np.hstack([Wc, A_power @ self.B])
        return Wc
    def is_controllable(self) -> bool:
        """Check if system is completely controllable."""
        Wc = self.controllability_matrix()
        rank = np.linalg.matrix_rank(Wc)
        return rank == self.n_states
    def observability_matrix(self) -> np.ndarray:
        """Compute observability matrix."""
        n = self.n_states
        Wo = self.C.copy()
        A_power = np.eye(n)
        for i in range(1, n):
            A_power = A_power @ self.A
            Wo = np.vstack([Wo, self.C @ A_power])
        return Wo
    def is_observable(self) -> bool:
        """Check if system is completely observable."""
        Wo = self.observability_matrix()
        rank = np.linalg.matrix_rank(Wo)
        return rank == self.n_states
    def simulate(self,
                t: np.ndarray,
                u: Union[np.ndarray, Callable],
                x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate system response.
        Parameters:
            t: Time vector
            u: Input signal (array or function)
            x0: Initial state (default: zeros)
        Returns:
            Tuple of (state_response, output_response)
        """
        if x0 is None:
            x0 = np.zeros(self.n_states)
        def system_dynamics(x, t_val):
            if callable(u):
                u_val = u(t_val)
                if np.isscalar(u_val):
                    u_val = np.array([u_val])
            else:
                # Handle input interpolation
                if u.ndim == 1:
                    # Single input case
                    u_val = np.array([np.interp(t_val, t, u)])
                else:
                    # Multiple inputs case
                    u_val = np.zeros(self.n_inputs)
                    for i in range(self.n_inputs):
                        u_val[i] = np.interp(t_val, t, u[:, i])
            return self.A @ x + self.B @ u_val
        # Solve ODE
        x_response = odeint(system_dynamics, x0, t)
        # Compute output
        y_response = np.zeros((len(t), self.n_outputs))
        for i, (t_val, x_val) in enumerate(zip(t, x_response)):
            if callable(u):
                u_val = u(t_val)
                if np.isscalar(u_val):
                    u_val = np.array([u_val])
            else:
                # Handle input interpolation
                if u.ndim == 1:
                    u_val = np.array([np.interp(t_val, t, u)])
                else:
                    u_val = np.zeros(self.n_inputs)
                    for i_input in range(self.n_inputs):
                        u_val[i_input] = np.interp(t_val, t, u[:, i_input])
            y_response[i] = self.C @ x_val + self.D @ u_val
        return x_response, y_response
    def step_response(self, t: np.ndarray, input_channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute step response.
        Parameters:
            t: Time vector
            input_channel: Which input to apply step to
        Returns:
            Tuple of (state_response, output_response)
        """
        u = np.zeros((len(t), self.n_inputs))
        u[:, input_channel] = 1.0
        return self.simulate(t, u)
    def impulse_response(self, t: np.ndarray, input_channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute impulse response.
        Parameters:
            t: Time vector
            input_channel: Which input to apply impulse to
        Returns:
            Tuple of (state_response, output_response)
        """
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        u = np.zeros((len(t), self.n_inputs))
        u[0, input_channel] = 1.0 / dt  # Approximation of delta function
        return self.simulate(t, u)
class LinearQuadraticRegulator:
    """
    Linear Quadratic Regulator (LQR) optimal controller.
    Solves the optimal control problem:
        min ∫(x'Qx + u'Ru + 2x'Nu) dt
    Subject to: dx/dt = Ax + Bu
    Examples:
        >>> sys = StateSpaceSystem(A, B, C, D)
        >>> Q = np.eye(sys.n_states)
        >>> R = np.eye(sys.n_inputs)
        >>> lqr = LinearQuadraticRegulator(sys, Q, R)
        >>> K = lqr.gain_matrix()
    """
    def __init__(self,
                 system: StateSpaceSystem,
                 Q: np.ndarray,
                 R: np.ndarray,
                 N: Optional[np.ndarray] = None):
        """
        Initialize LQR controller.
        Parameters:
            system: State-space system
            Q: State weighting matrix (n x n, positive semi-definite)
            R: Input weighting matrix (m x m, positive definite)
            N: Cross-term matrix (n x m, optional)
        """
        self.system = system
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        if N is None:
            self.N = np.zeros((system.n_states, system.n_inputs))
        else:
            self.N = np.atleast_2d(N)
        self._validate_weights()
        self._solve_riccati()
    def _validate_weights(self):
        """Validate weighting matrices."""
        # Check dimensions
        if self.Q.shape != (self.system.n_states, self.system.n_states):
            raise ValueError("Q matrix must be n x n")
        if self.R.shape != (self.system.n_inputs, self.system.n_inputs):
            raise ValueError("R matrix must be m x m")
        if self.N.shape != (self.system.n_states, self.system.n_inputs):
            raise ValueError("N matrix must be n x m")
        # Check positive definiteness
        if not np.allclose(self.Q, self.Q.T):
            warnings.warn("Q matrix is not symmetric")
        if not np.allclose(self.R, self.R.T):
            warnings.warn("R matrix is not symmetric")
        try:
            linalg.cholesky(self.R)
        except linalg.LinAlgError:
            raise ValueError("R matrix must be positive definite")
    def _solve_riccati(self):
        """Solve algebraic Riccati equation."""
        try:
            # Check if N is zero (most common case)
            if np.allclose(self.N, 0):
                self.P = linalg.solve_continuous_are(
                    self.system.A, self.system.B, self.Q, self.R
                )
            else:
                self.P = linalg.solve_continuous_are(
                    self.system.A, self.system.B, self.Q, self.R, s=self.N
                )
        except Exception as e:
            raise RuntimeError(f"Failed to solve Riccati equation: {e}")
    def gain_matrix(self) -> np.ndarray:
        """Compute optimal feedback gain matrix K."""
        R_inv = linalg.inv(self.R)
        K = R_inv @ (self.system.B.T @ self.P + self.N.T)
        return K
    def closed_loop_system(self) -> StateSpaceSystem:
        """Compute closed-loop system with LQR feedback."""
        K = self.gain_matrix()
        A_cl = self.system.A - self.system.B @ K
        B_cl = self.system.B  # For reference input
        C_cl = self.system.C
        D_cl = self.system.D
        return StateSpaceSystem(A_cl, B_cl, C_cl, D_cl)
    def cost_function_value(self, x0: np.ndarray) -> float:
        """
        Compute optimal cost for given initial condition.
        Parameters:
            x0: Initial state
        Returns:
            Optimal cost J* = x0' P x0
        """
        return x0.T @ self.P @ x0