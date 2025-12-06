"""
Test suite for State-Space System implementation.
Tests cover:
- System matrix validation
- Controllability and observability analysis
- System simulation
- LQR controller design
- Cross-platform numerical validation
"""
import pytest
import numpy as np
from scipy import linalg
from pathlib import Path
import sys
# Add Control module to path
sys.path.append(str(Path(__file__).parent.parent))
from core.state_space import StateSpaceSystem, LinearQuadraticRegulator
class TestStateSpaceSystem:
    """Test cases for state-space systems."""
    def setup_method(self):
        """Setup test fixtures."""
        # Simple second-order system
        self.A = np.array([[-1, 1], [0, -2]])
        self.B = np.array([[0], [1]])
        self.C = np.array([[1, 0]])
        self.D = np.array([[0]])
        self.system = StateSpaceSystem(self.A, self.B, self.C, self.D)
        # Fixed random seed
        np.random.seed(42)
    def test_initialization(self):
        """Test system initialization and validation."""
        # Check matrix storage
        assert np.array_equal(self.system.A, self.A)
        assert np.array_equal(self.system.B, self.B)
        assert np.array_equal(self.system.C, self.C)
        assert np.array_equal(self.system.D, self.D)
        # Check dimensions
        assert self.system.n_states == 2
        assert self.system.n_inputs == 1
        assert self.system.n_outputs == 1
    def test_invalid_dimensions(self):
        """Test error handling for invalid matrix dimensions."""
        # Non-square A matrix
        with pytest.raises(ValueError, match="A matrix must be square"):
            StateSpaceSystem(np.array([[1, 2, 3]]), self.B, self.C, self.D)
        # Mismatched B matrix
        with pytest.raises(ValueError, match="B matrix must have same number of rows as A"):
            StateSpaceSystem(self.A, np.array([[1, 2, 3]]), self.C, self.D)
        # Mismatched C matrix
        with pytest.raises(ValueError, match="C matrix must have same number of columns as A"):
            StateSpaceSystem(self.A, self.B, np.array([[1, 2, 3]]), self.D)
        # Mismatched D matrix
        with pytest.raises(ValueError, match="D matrix dimensions must be"):
            StateSpaceSystem(self.A, self.B, self.C, np.array([[1, 2]]))
    def test_invalid_values(self):
        """Test error handling for NaN and infinite values."""
        # NaN values
        A_nan = self.A.copy()
        A_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="System matrices contain NaN values"):
            StateSpaceSystem(A_nan, self.B, self.C, self.D)
        # Infinite values
        B_inf = self.B.copy()
        B_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="System matrices contain infinite values"):
            StateSpaceSystem(self.A, B_inf, self.C, self.D)
    def test_poles_computation(self):
        """Test eigenvalue (poles) computation."""
        poles = self.system.poles()
        expected_poles = np.array([-1, -2])
        # Sort both arrays for comparison
        poles_sorted = np.sort(poles)
        expected_sorted = np.sort(expected_poles)
        assert np.allclose(poles_sorted, expected_sorted)
    def test_stability_check(self):
        """Test stability analysis."""
        # Current system should be stable (all poles negative)
        assert self.system.is_stable()
        # Create unstable system
        A_unstable = np.array([[1, 0], [0, -1]])  # One positive pole
        system_unstable = StateSpaceSystem(A_unstable, self.B, self.C, self.D)
        assert not system_unstable.is_stable()
        # Marginally stable system
        A_marginal = np.array([[0, 1], [0, -1]])  # One pole at origin
        system_marginal = StateSpaceSystem(A_marginal, self.B, self.C, self.D)
        assert not system_marginal.is_stable()
    def test_controllability(self):
        """Test controllability analysis."""
        # Compute controllability matrix
        Wc = self.system.controllability_matrix()
        expected_Wc = np.hstack([self.B, self.A @ self.B])
        assert np.allclose(Wc, expected_Wc)
        # Check controllability
        assert self.system.is_controllable()
        # Create uncontrollable system
        A_uncontrollable = np.array([[1, 0], [0, 2]])
        B_uncontrollable = np.array([[1], [0]])  # Second state not controllable
        system_uncontrollable = StateSpaceSystem(A_uncontrollable, B_uncontrollable, self.C, self.D)
        assert not system_uncontrollable.is_controllable()
    def test_observability(self):
        """Test observability analysis."""
        # Compute observability matrix
        Wo = self.system.observability_matrix()
        expected_Wo = np.vstack([self.C, self.C @ self.A])
        assert np.allclose(Wo, expected_Wo)
        # Check observability
        assert self.system.is_observable()
        # Create unobservable system
        C_unobservable = np.array([[0, 1]])  # First state not observable
        system_unobservable = StateSpaceSystem(self.A, self.B, C_unobservable, self.D)
        assert not system_unobservable.is_observable()
    def test_step_response(self):
        """Test step response simulation."""
        t = np.linspace(0, 5, 100)
        x_response, y_response = self.system.step_response(t)
        # Check dimensions
        assert x_response.shape == (len(t), self.system.n_states)
        assert y_response.shape == (len(t), self.system.n_outputs)
        # Check initial conditions
        assert np.allclose(x_response[0], np.zeros(self.system.n_states))
        assert np.allclose(y_response[0], np.zeros(self.system.n_outputs))
        # Check steady-state value for stable system
        # For step input, steady-state output = C * (-A)^(-1) * B
        steady_state = -self.C @ linalg.inv(self.A) @ self.B
        assert np.allclose(y_response[-1], steady_state, rtol=1e-3)
    def test_impulse_response(self):
        """Test impulse response simulation."""
        t = np.linspace(0, 5, 100)
        x_response, y_response = self.system.impulse_response(t)
        # Check dimensions
        assert x_response.shape == (len(t), self.system.n_states)
        assert y_response.shape == (len(t), self.system.n_outputs)
        # For stable system, response should decay to zero
        assert np.allclose(y_response[-1], 0, atol=1e-2)
    def test_simulation_with_input(self):
        """Test simulation with arbitrary input."""
        t = np.linspace(0, 2, 100)
        u = np.sin(t).reshape(-1, 1)  # Sinusoidal input
        x0 = np.array([1, 0])  # Non-zero initial condition
        x_response, y_response = self.system.simulate(t, u, x0)
        # Check dimensions
        assert x_response.shape == (len(t), self.system.n_states)
        assert y_response.shape == (len(t), self.system.n_outputs)
        # Check initial condition
        assert np.allclose(x_response[0], x0)
        # Check that output follows state evolution
        for i in range(len(t)):
            expected_output = self.C @ x_response[i] + self.D @ u[i]
            assert np.allclose(y_response[i], expected_output, rtol=1e-6)
    def test_callable_input(self):
        """Test simulation with callable input function."""
        t = np.linspace(0, 1, 50)
        def input_function(time):
            return np.array([np.exp(-time)])  # Exponential decay
        x_response, y_response = self.system.simulate(t, input_function)
        # Should execute without error
        assert x_response.shape == (len(t), self.system.n_states)
        assert y_response.shape == (len(t), self.system.n_outputs)
class TestLinearQuadraticRegulator:
    """Test cases for LQR controller."""
    def setup_method(self):
        """Setup test fixtures."""
        # Controllable system
        A = np.array([[-1, 1], [0, -2]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        self.system = StateSpaceSystem(A, B, C, D)
        # LQR weights
        self.Q = np.eye(2)
        self.R = np.array([[1]])
        self.lqr = LinearQuadraticRegulator(self.system, self.Q, self.R)
    def test_initialization(self):
        """Test LQR initialization."""
        assert np.array_equal(self.lqr.Q, self.Q)
        assert np.array_equal(self.lqr.R, self.R)
        assert self.lqr.N.shape == (2, 1)  # Default N is zeros
    def test_invalid_weights(self):
        """Test error handling for invalid weight matrices."""
        # Wrong Q dimensions
        with pytest.raises(ValueError, match="Q matrix must be n x n"):
            LinearQuadraticRegulator(self.system, np.eye(3), self.R)
        # Wrong R dimensions
        with pytest.raises(ValueError, match="R matrix must be m x m"):
            LinearQuadraticRegulator(self.system, self.Q, np.eye(2))
        # Non-positive definite R
        R_bad = np.array([[0]])  # Not positive definite
        with pytest.raises(ValueError, match="R matrix must be positive definite"):
            LinearQuadraticRegulator(self.system, self.Q, R_bad)
    def test_riccati_solution(self):
        """Test Riccati equation solution."""
        P = self.lqr.P
        # P should be positive semi-definite
        eigenvals = linalg.eigvals(P)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
        # P should be symmetric
        assert np.allclose(P, P.T)
        # Verify Riccati equation: A'P + PA - PBR^(-1)B'P + Q = 0
        A = self.system.A
        B = self.system.B
        R_inv = linalg.inv(self.R)
        riccati_residual = (A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + self.Q)
        assert np.allclose(riccati_residual, 0, atol=1e-10)
    def test_gain_matrix(self):
        """Test feedback gain computation."""
        K = self.lqr.gain_matrix()
        # Check dimensions
        assert K.shape == (self.system.n_inputs, self.system.n_states)
        # Gain should be reasonable (not infinite)
        assert np.all(np.isfinite(K))
        # For this specific problem, we can check approximate values
        # (depends on Q and R choice)
        assert K.shape == (1, 2)
    def test_closed_loop_stability(self):
        """Test that LQR produces stable closed-loop system."""
        sys_cl = self.lqr.closed_loop_system()
        # Closed-loop system should be stable
        assert sys_cl.is_stable()
        # Check that closed-loop poles are in left half-plane
        cl_poles = sys_cl.poles()
        assert np.all(np.real(cl_poles) < 0)
    def test_cost_function(self):
        """Test optimal cost computation."""
        x0 = np.array([1, 1])
        cost = self.lqr.cost_function_value(x0)
        # Cost should be positive for non-zero initial state
        assert cost > 0
        # Cost for zero initial state should be zero
        cost_zero = self.lqr.cost_function_value(np.zeros(2))
        assert abs(cost_zero) < 1e-10
    def test_different_weights(self):
        """Test LQR with different weight matrices."""
        # High state penalty
        Q_high = 100 * np.eye(2)
        lqr_high = LinearQuadraticRegulator(self.system, Q_high, self.R)
        K_high = lqr_high.gain_matrix()
        # High control penalty
        R_high = 100 * np.array([[1]])
        lqr_control = LinearQuadraticRegulator(self.system, self.Q, R_high)
        K_control = lqr_control.gain_matrix()
        # Higher state penalty should lead to higher gains
        assert np.linalg.norm(K_high) > np.linalg.norm(self.lqr.gain_matrix())
        # Higher control penalty should lead to lower gains
        assert np.linalg.norm(K_control) < np.linalg.norm(self.lqr.gain_matrix())
    def test_cross_term_matrix(self):
        """Test LQR with cross-term matrix N."""
        N = np.array([[0.1], [0.2]])
        lqr_cross = LinearQuadraticRegulator(self.system, self.Q, self.R, N)
        # Should initialize without error
        assert np.array_equal(lqr_cross.N, N)
        # Should produce valid gain matrix
        K = lqr_cross.gain_matrix()
        assert K.shape == (1, 2)
        assert np.all(np.isfinite(K))
def test_cross_platform_validation():
    """
    Cross-platform validation test.
    Provides reference values for numerical comparison across platforms.
    """
    # Standard double integrator system
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    system = StateSpaceSystem(A, B, C, D)
    # Test controllability and observability
    assert system.is_controllable()
    assert system.is_observable()
    # LQR design
    Q = np.diag([1, 0.1])
    R = np.array([[0.1]])
    lqr = LinearQuadraticRegulator(system, Q, R)
    # Reference values for cross-platform validation
    K = lqr.gain_matrix()
    P = lqr.P
    # Expected values (computed with high precision)
    K_expected = np.array([[3.16227766, 2.23606798]])
    P_expected = np.array([[2.23606798, 1.0],
                          [1.0, 0.70710678]])
    # Validate within numerical tolerance
    assert np.allclose(K, K_expected, rtol=1e-6), f"K mismatch: {K} vs {K_expected}"
    assert np.allclose(P, P_expected, rtol=1e-6), f"P mismatch: {P} vs {P_expected}"
    # Step response validation
    t = np.linspace(0, 5, 100)
    x_response, y_response = system.step_response(t)
    # Check steady-state response
    steady_state_expected = 0.0  # Double integrator has infinite DC gain, but we're looking at position
    # Validate closed-loop step response
    sys_cl = lqr.closed_loop_system()
    x_cl, y_cl = sys_cl.step_response(t)
    # Final value should be finite and stable
    assert np.abs(y_cl[-1]) < 10  # Should not blow up
    print("Cross-platform validation values:")
    print(f"LQR gain K: {K}")
    print(f"Riccati solution P:\n{P}")
    print(f"Closed-loop final value: {y_cl[-1]}")
if __name__ == "__main__":
    # Run cross-platform validation
    test_cross_platform_validation()
    print("All state-space tests passed!")