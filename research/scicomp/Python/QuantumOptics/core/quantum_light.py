"""
Quantum light states and operations.
This module provides tools for quantum optics including:
- Fock states
- Coherent states
- Squeezed states
- Photon statistics
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy.special import factorial, hermite
from scipy.linalg import expm
import scipy.sparse as sp
class FockStates:
    """Fock (number) states and operations."""
    @staticmethod
    def fock_state(n: int, dim: int) -> np.ndarray:
        """
        Create Fock state |n⟩.
        Args:
            n: Photon number
            dim: Hilbert space dimension
        Returns:
            Fock state vector
        """
        state = np.zeros(dim, dtype=complex)
        if n < dim:
            state[n] = 1
        return state
    @staticmethod
    def creation_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.spmatrix]:
        """
        Creation operator a†.
        Args:
            dim: Hilbert space dimension
            sparse: Return sparse matrix
        Returns:
            Creation operator
        """
        if sparse:
            a_dag = sp.lil_matrix((dim, dim), dtype=complex)
            for n in range(dim - 1):
                a_dag[n + 1, n] = np.sqrt(n + 1)
            return a_dag.tocsr()
        else:
            a_dag = np.zeros((dim, dim), dtype=complex)
            for n in range(dim - 1):
                a_dag[n + 1, n] = np.sqrt(n + 1)
            return a_dag
    @staticmethod
    def annihilation_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.spmatrix]:
        """
        Annihilation operator a.
        Args:
            dim: Hilbert space dimension
            sparse: Return sparse matrix
        Returns:
            Annihilation operator
        """
        if sparse:
            a = sp.lil_matrix((dim, dim), dtype=complex)
            for n in range(1, dim):
                a[n - 1, n] = np.sqrt(n)
            return a.tocsr()
        else:
            a = np.zeros((dim, dim), dtype=complex)
            for n in range(1, dim):
                a[n - 1, n] = np.sqrt(n)
            return a
    @staticmethod
    def number_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.spmatrix]:
        """
        Number operator n = a†a.
        Args:
            dim: Hilbert space dimension
            sparse: Return sparse matrix
        Returns:
            Number operator
        """
        if sparse:
            return sp.diag(np.arange(dim), dtype=complex)
        else:
            return np.diag(np.arange(dim)).astype(complex)
    @staticmethod
    def displacement_operator(alpha: complex, dim: int) -> np.ndarray:
        """
        Displacement operator D(α) = exp(αa† - α*a).
        Args:
            alpha: Displacement parameter
            dim: Hilbert space dimension
        Returns:
            Displacement operator
        """
        a = FockStates.annihilation_operator(dim)
        a_dag = FockStates.creation_operator(dim)
        return expm(alpha * a_dag - np.conj(alpha) * a)
class CoherentStates:
    """Coherent states and operations."""
    @staticmethod
    def coherent_state(alpha: complex, dim: int, method: str = 'displacement') -> np.ndarray:
        """
        Create coherent state |α⟩.
        Args:
            alpha: Coherent state parameter
            dim: Hilbert space dimension
            method: 'displacement' or 'fock_expansion'
        Returns:
            Coherent state vector
        """
        if method == 'displacement':
            # |α⟩ = D(α)|0⟩
            D = FockStates.displacement_operator(alpha, dim)
            vacuum = FockStates.fock_state(0, dim)
            return D @ vacuum
        elif method == 'fock_expansion':
            # |α⟩ = e^(-|α|²/2) Σ_n α^n/√(n!) |n⟩
            state = np.zeros(dim, dtype=complex)
            for n in range(dim):
                state[n] = alpha**n / np.sqrt(factorial(n))
            state *= np.exp(-np.abs(alpha)**2 / 2)
            return state
        else:
            raise ValueError(f"Unknown method: {method}")
    @staticmethod
    def coherent_state_overlap(alpha: complex, beta: complex) -> complex:
        """
        Calculate overlap ⟨α|β⟩.
        Args:
            alpha: First coherent state parameter
            beta: Second coherent state parameter
        Returns:
            Overlap
        """
        return np.exp(-0.5 * (np.abs(alpha)**2 + np.abs(beta)**2 - 2 * np.conj(alpha) * beta))
    @staticmethod
    def husimi_q_function(state: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
        """
        Calculate Husimi Q-function.
        Args:
            state: Quantum state vector or density matrix
            alpha_grid: Grid of coherent state parameters
        Returns:
            Q-function values
        """
        dim = len(state) if state.ndim == 1 else state.shape[0]
        Q = np.zeros(alpha_grid.shape, dtype=float)
        for idx in np.ndindex(alpha_grid.shape):
            alpha = alpha_grid[idx]
            coherent = CoherentStates.coherent_state(alpha, dim)
            if state.ndim == 1:
                # Pure state
                Q[idx] = np.abs(np.vdot(coherent, state))**2 / np.pi
            else:
                # Mixed state (density matrix)
                Q[idx] = np.real(np.vdot(coherent, state @ coherent)) / np.pi
        return Q
    @staticmethod
    def glauber_sudarshan_p_function(state: np.ndarray, alpha: complex,
                                    regularization: float = 0.01) -> complex:
        """
        Calculate regularized P-function.
        Args:
            state: Density matrix
            alpha: Coherent state parameter
            regularization: Regularization parameter
        Returns:
            P-function value
        """
        dim = state.shape[0]
        # Regularized P-function using convolution
        coherent = CoherentStates.coherent_state(alpha, dim)
        coherent_dm = np.outer(coherent, coherent.conj())
        # Add regularization (thermal noise)
        n_th = regularization
        thermal = np.diag([(n_th / (n_th + 1))**(n+1) for n in range(dim)])
        regularized_state = state + regularization * thermal
        regularized_state /= np.trace(regularized_state)
        return np.trace(regularized_state @ coherent_dm) / np.pi
class SqueezedStates:
    """Squeezed states and operations."""
    @staticmethod
    def squeeze_operator(xi: complex, dim: int) -> np.ndarray:
        """
        Squeeze operator S(ξ) = exp((ξ*a² - ξa†²)/2).
        Args:
            xi: Squeezing parameter (r*e^(iθ))
            dim: Hilbert space dimension
        Returns:
            Squeeze operator
        """
        a = FockStates.annihilation_operator(dim)
        a_dag = FockStates.creation_operator(dim)
        a_squared = a @ a
        a_dag_squared = a_dag @ a_dag
        return expm(0.5 * (np.conj(xi) * a_squared - xi * a_dag_squared))
    @staticmethod
    def squeezed_state(xi: complex, alpha: complex, dim: int) -> np.ndarray:
        """
        Create squeezed coherent state |α, ξ⟩.
        Args:
            xi: Squeezing parameter
            alpha: Displacement parameter
            dim: Hilbert space dimension
        Returns:
            Squeezed coherent state
        """
        # |α, ξ⟩ = D(α)S(ξ)|0⟩
        S = SqueezedStates.squeeze_operator(xi, dim)
        D = FockStates.displacement_operator(alpha, dim)
        vacuum = FockStates.fock_state(0, dim)
        return D @ S @ vacuum
    @staticmethod
    def two_mode_squeeze_operator(xi: complex, dim: int) -> np.ndarray:
        """
        Two-mode squeeze operator.
        Args:
            xi: Squeezing parameter
            dim: Dimension per mode
        Returns:
            Two-mode squeeze operator
        """
        a1 = np.kron(FockStates.annihilation_operator(dim), np.eye(dim))
        a2 = np.kron(np.eye(dim), FockStates.annihilation_operator(dim))
        a1_dag = np.kron(FockStates.creation_operator(dim), np.eye(dim))
        a2_dag = np.kron(np.eye(dim), FockStates.creation_operator(dim))
        return expm(xi * a1_dag @ a2_dag - np.conj(xi) * a1 @ a2)
    @staticmethod
    def quadrature_variances(state: np.ndarray) -> Tuple[float, float]:
        """
        Calculate quadrature variances.
        Args:
            state: Quantum state vector
        Returns:
            Variances of X and P quadratures
        """
        dim = len(state)
        a = FockStates.annihilation_operator(dim)
        a_dag = FockStates.creation_operator(dim)
        # Quadrature operators
        X = (a + a_dag) / np.sqrt(2)
        P = 1j * (a_dag - a) / np.sqrt(2)
        # Calculate variances
        X_avg = np.real(np.vdot(state, X @ state))
        X2_avg = np.real(np.vdot(state, X @ X @ state))
        var_X = X2_avg - X_avg**2
        P_avg = np.real(np.vdot(state, P @ state))
        P2_avg = np.real(np.vdot(state, P @ P @ state))
        var_P = P2_avg - P_avg**2
        return var_X, var_P
class PhotonStatistics:
    """Photon counting statistics and correlations."""
    @staticmethod
    def photon_distribution(state: Union[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Calculate photon number distribution.
        Args:
            state: State vector or density matrix
        Returns:
            Probability distribution P(n)
        """
        if state.ndim == 1:
            # Pure state
            return np.abs(state)**2
        else:
            # Density matrix
            return np.real(np.diag(state))
    @staticmethod
    def mean_photon_number(state: Union[np.ndarray, np.ndarray]) -> float:
        """
        Calculate mean photon number.
        Args:
            state: State vector or density matrix
        Returns:
            Mean photon number
        """
        dim = len(state) if state.ndim == 1 else state.shape[0]
        n_op = FockStates.number_operator(dim)
        if state.ndim == 1:
            return np.real(np.vdot(state, n_op @ state))
        else:
            return np.real(np.trace(n_op @ state))
    @staticmethod
    def mandel_q_parameter(state: Union[np.ndarray, np.ndarray]) -> float:
        """
        Calculate Mandel Q parameter.
        Args:
            state: State vector or density matrix
        Returns:
            Mandel Q parameter
        """
        dim = len(state) if state.ndim == 1 else state.shape[0]
        n_op = FockStates.number_operator(dim)
        if state.ndim == 1:
            n_mean = np.real(np.vdot(state, n_op @ state))
            n2_mean = np.real(np.vdot(state, n_op @ n_op @ state))
        else:
            n_mean = np.real(np.trace(n_op @ state))
            n2_mean = np.real(np.trace(n_op @ n_op @ state))
        variance = n2_mean - n_mean**2
        if n_mean > 0:
            return (variance - n_mean) / n_mean
        else:
            return 0
    @staticmethod
    def g2_correlation(state: Union[np.ndarray, np.ndarray], tau: float = 0) -> float:
        """
        Calculate second-order correlation g²(τ).
        Args:
            state: State vector or density matrix
            tau: Time delay (0 for equal-time)
        Returns:
            g²(τ) correlation
        """
        dim = len(state) if state.ndim == 1 else state.shape[0]
        a = FockStates.annihilation_operator(dim)
        a_dag = FockStates.creation_operator(dim)
        n_op = FockStates.number_operator(dim)
        if tau == 0:
            # Equal-time correlation
            if state.ndim == 1:
                n_mean = np.real(np.vdot(state, n_op @ state))
                a2_a2dag = np.real(np.vdot(state, a_dag @ a_dag @ a @ a @ state))
            else:
                n_mean = np.real(np.trace(n_op @ state))
                a2_a2dag = np.real(np.trace(a_dag @ a_dag @ a @ a @ state))
            if n_mean > 0:
                return a2_a2dag / n_mean**2
            else:
                return 0
        else:
            # Time-delayed correlation (requires dynamics)
            raise NotImplementedError("Time-delayed correlations require system dynamics")
    @staticmethod
    def antibunching_parameter(state: Union[np.ndarray, np.ndarray]) -> float:
        """
        Calculate antibunching parameter.
        Args:
            state: State vector or density matrix
        Returns:
            Antibunching parameter (negative for antibunching)
        """
        g2 = PhotonStatistics.g2_correlation(state)
        return g2 - 1
class WignerFunction:
    """Wigner quasi-probability distribution."""
    @staticmethod
    def wigner_function(state: Union[np.ndarray, np.ndarray],
                       xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
        """
        Calculate Wigner function.
        Args:
            state: State vector or density matrix
            xvec: Position grid
            pvec: Momentum grid
        Returns:
            Wigner function W(x,p)
        """
        if state.ndim == 1:
            # Convert to density matrix
            rho = np.outer(state, state.conj())
        else:
            rho = state
        dim = rho.shape[0]
        X, P = np.meshgrid(xvec, pvec)
        W = np.zeros_like(X)
        # Calculate Wigner function using characteristic function
        for i, x in enumerate(xvec):
            for j, p in enumerate(pvec):
                W[j, i] = WignerFunction._wigner_point(rho, x, p, dim)
        return W
    @staticmethod
    def _wigner_point(rho: np.ndarray, x: float, p: float, dim: int) -> float:
        """Calculate Wigner function at single point."""
        alpha = (x + 1j * p) / np.sqrt(2)
        # Use displaced parity operator method
        D = FockStates.displacement_operator(alpha, dim)
        parity = np.diag([(-1)**n for n in range(dim)])
        displaced_parity = D @ parity @ D.conj().T
        return np.real(np.trace(rho @ displaced_parity)) * 2 / np.pi
    @staticmethod
    def wigner_from_characteristic(chi: np.ndarray, beta_grid: np.ndarray,
                                 xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
        """
        Calculate Wigner function from characteristic function.
        Args:
            chi: Characteristic function values
            beta_grid: Grid for characteristic function
            xvec: Position grid
            pvec: Momentum grid
        Returns:
            Wigner function
        """
        X, P = np.meshgrid(xvec, pvec)
        W = np.zeros_like(X)
        dbeta = beta_grid[0, 1] - beta_grid[0, 0]
        for i, x in enumerate(xvec):
            for j, p in enumerate(pvec):
                alpha = (x + 1j * p) / np.sqrt(2)
                # Fourier transform of characteristic function
                integrand = chi * np.exp(-2j * np.real(np.conj(alpha) * beta_grid))
                W[j, i] = np.real(np.sum(integrand)) * dbeta**2 / (np.pi**2)
        return W