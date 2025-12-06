"""
Optimal Control Algorithms
Advanced optimal control methods including model predictive control,
dynamic programming, and trajectory optimization.
"""
import numpy as np
from scipy import optimize, linalg
from scipy.integrate import solve_ivp
from typing import Optional, Tuple, Callable, Dict, List
from dataclasses import dataclass
import warnings
@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""
    prediction_horizon: int = 10  # Prediction horizon steps
    control_horizon: int = None   # Control horizon (None = same as prediction)
    # Constraints
    u_min: Optional[np.ndarray] = None  # Input lower bounds
    u_max: Optional[np.ndarray] = None  # Input upper bounds
    du_min: Optional[np.ndarray] = None # Input rate lower bounds
    du_max: Optional[np.ndarray] = None # Input rate upper bounds
    # Weighting matrices
    Q: Optional[np.ndarray] = None      # State penalty
    R: Optional[np.ndarray] = None      # Input penalty
    Qf: Optional[np.ndarray] = None     # Terminal state penalty
    # Solver options
    solver: str = 'quadprog'  # Optimization solver
    max_iter: int = 100       # Maximum iterations
    tolerance: float = 1e-6   # Convergence tolerance
class ModelPredictiveController:
    """
    Model Predictive Control (MPC) implementation.
    Solves the finite-horizon optimal control problem at each time step:
        min Σ(x'Qx + u'Ru) + xf'Qf*xf
    Subject to:
        x(k+1) = Ax(k) + Bu(k)
        u_min ≤ u(k) ≤ u_max
        du_min ≤ u(k) - u(k-1) ≤ du_max
    Examples:
        >>> config = MPCConfig(prediction_horizon=20, Q=Q, R=R)
        >>> mpc = ModelPredictiveController(A, B, config)
        >>> u_opt = mpc.solve(x_current, reference_trajectory)
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, config: MPCConfig):
        """
        Initialize MPC controller.
        Parameters:
            A: System matrix (n x n)
            B: Input matrix (n x m)
            config: MPC configuration
        """
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.config = config
        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        if config.control_horizon is None:
            self.config.control_horizon = config.prediction_horizon
        self._setup_weights()
        self._build_prediction_matrices()
    def _setup_weights(self):
        """Setup default weighting matrices if not provided."""
        if self.config.Q is None:
            self.config.Q = np.eye(self.n_states)
        if self.config.R is None:
            self.config.R = np.eye(self.n_inputs)
        if self.config.Qf is None:
            self.config.Qf = self.config.Q
    def _build_prediction_matrices(self):
        """Build prediction matrices for the MPC formulation."""
        N = self.config.prediction_horizon
        Nc = self.config.control_horizon
        # State prediction matrix
        self.Phi = np.zeros((N * self.n_states, self.n_states))
        A_power = np.eye(self.n_states)
        for i in range(N):
            self.Phi[i*self.n_states:(i+1)*self.n_states, :] = A_power
            A_power = A_power @ self.A
        # Input-to-state matrix
        self.Gamma = np.zeros((N * self.n_states, Nc * self.n_inputs))
        for i in range(N):
            A_power = np.eye(self.n_states)
            for j in range(min(i+1, Nc)):
                if i-j >= 0:
                    row_start = i * self.n_states
                    row_end = (i+1) * self.n_states
                    col_start = j * self.n_inputs
                    col_end = (j+1) * self.n_inputs
                    self.Gamma[row_start:row_end, col_start:col_end] = A_power @ self.B
                A_power = A_power @ self.A
        # Build cost matrices
        Q_bar = linalg.block_diag(*[self.config.Q for _ in range(N-1)], self.config.Qf)
        R_bar = linalg.block_diag(*[self.config.R for _ in range(Nc)])
        self.H = self.Gamma.T @ Q_bar @ self.Gamma + R_bar
        # Ensure H is positive definite
        eigvals = linalg.eigvals(self.H)
        if np.min(eigvals) <= 0:
            self.H += 1e-8 * np.eye(self.H.shape[0])
    def solve(self,
              x_current: np.ndarray,
              reference: Optional[np.ndarray] = None,
              u_previous: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve MPC optimization problem.
        Parameters:
            x_current: Current state
            reference: Reference trajectory (N x n_states)
            u_previous: Previous control input (for rate constraints)
        Returns:
            Optimal control sequence (Nc x n_inputs)
        """
        N = self.config.prediction_horizon
        Nc = self.config.control_horizon
        # Setup reference trajectory
        if reference is None:
            reference = np.zeros((N, self.n_states))
        elif reference.ndim == 1:
            reference = np.tile(reference, (N, 1))
        ref_vec = reference.flatten()
        # Compute prediction without control
        x_pred = self.Phi @ x_current
        # Setup quadratic program: min 0.5 u'Hu + f'u
        f = self.Gamma.T @ self.config.__dict__.get('Q_bar', np.eye(N * self.n_states)) @ (x_pred - ref_vec)
        # Setup constraints
        bounds = []
        A_ineq = []
        b_ineq = []
        for i in range(Nc):
            # Input bounds
            if self.config.u_min is not None:
                bounds.extend([(u_min, None) for u_min in self.config.u_min])
            else:
                bounds.extend([(None, None) for _ in range(self.n_inputs)])
            if self.config.u_max is not None:
                for j, (low, _) in enumerate(bounds[-self.n_inputs:]):
                    bounds[-self.n_inputs + j] = (low, self.config.u_max[j])
        # Input rate constraints
        if (self.config.du_min is not None or self.config.du_max is not None) and u_previous is not None:
            for i in range(Nc):
                row = np.zeros(Nc * self.n_inputs)
                row[i*self.n_inputs:(i+1)*self.n_inputs] = 1.0
                if i == 0:
                    # First control move relative to previous
                    if self.config.du_min is not None:
                        A_ineq.append(-row)
                        b_ineq.append(-self.config.du_min - u_previous)
                    if self.config.du_max is not None:
                        A_ineq.append(row)
                        b_ineq.append(self.config.du_max + u_previous)
                else:
                    # Subsequent moves relative to previous in sequence
                    prev_row = np.zeros(Nc * self.n_inputs)
                    prev_row[(i-1)*self.n_inputs:i*self.n_inputs] = 1.0
                    diff_row = row - prev_row
                    if self.config.du_min is not None:
                        A_ineq.append(-diff_row)
                        b_ineq.append(-self.config.du_min)
                    if self.config.du_max is not None:
                        A_ineq.append(diff_row)
                        b_ineq.append(self.config.du_max)
        # Solve optimization problem
        try:
            if A_ineq:
                constraints = {'type': 'ineq', 'fun': lambda u: np.array(b_ineq) - np.array(A_ineq) @ u}
            else:
                constraints = None
            result = optimize.minimize(
                fun=lambda u: 0.5 * u.T @ self.H @ u + f.T @ u,
                x0=np.zeros(Nc * self.n_inputs),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iter, 'ftol': self.config.tolerance}
            )
            if not result.success:
                warnings.warn(f"MPC optimization failed: {result.message}")
            u_opt = result.x.reshape((Nc, self.n_inputs))
        except Exception as e:
            warnings.warn(f"MPC solver error: {e}")
            u_opt = np.zeros((Nc, self.n_inputs))
        return u_opt
    def predict_trajectory(self,
                          x_current: np.ndarray,
                          u_sequence: np.ndarray) -> np.ndarray:
        """
        Predict state trajectory given control sequence.
        Parameters:
            x_current: Current state
            u_sequence: Control sequence (Nc x n_inputs)
        Returns:
            Predicted state trajectory (N x n_states)
        """
        N = self.config.prediction_horizon
        Nc = self.config.control_horizon
        # Extend control sequence if needed
        if len(u_sequence) < N:
            u_extended = np.zeros((N, self.n_inputs))
            u_extended[:Nc] = u_sequence
            # Hold last control value
            if Nc < N:
                u_extended[Nc:] = u_sequence[-1]
        else:
            u_extended = u_sequence[:N]
        # Predict trajectory
        x_pred = np.zeros((N+1, self.n_states))
        x_pred[0] = x_current
        for k in range(N):
            x_pred[k+1] = self.A @ x_pred[k] + self.B @ u_extended[k]
        return x_pred[1:]  # Return N predicted states
def solve_lq_tracking(A: np.ndarray,
                     B: np.ndarray,
                     Q: np.ndarray,
                     R: np.ndarray,
                     reference: np.ndarray,
                     horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve finite-horizon LQ tracking problem.
    Parameters:
        A: System matrix
        B: Input matrix
        Q: State penalty matrix
        R: Input penalty matrix
        reference: Reference trajectory (horizon x n_states)
        horizon: Time horizon
    Returns:
        Tuple of (optimal_control, optimal_trajectory)
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    # Initialize arrays
    P = np.zeros((horizon+1, n_states, n_states))
    v = np.zeros((horizon+1, n_states))
    K = np.zeros((horizon, n_inputs, n_states))
    k = np.zeros((horizon, n_inputs))
    # Terminal conditions
    P[horizon] = Q
    v[horizon] = -Q @ reference[horizon-1]
    # Backward pass
    for t in range(horizon-1, -1, -1):
        # Compute feedback gains
        temp = linalg.inv(R + B.T @ P[t+1] @ B)
        K[t] = temp @ B.T @ P[t+1] @ A
        k[t] = temp @ B.T @ v[t+1]
        # Update cost-to-go
        P[t] = Q + A.T @ P[t+1] @ A - A.T @ P[t+1] @ B @ K[t]
        v[t] = -Q @ reference[t] + (A - B @ K[t]).T @ v[t+1]
    return K, k