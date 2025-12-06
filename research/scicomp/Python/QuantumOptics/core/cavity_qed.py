"""
Cavity Quantum Electrodynamics (QED) simulations.
This module provides tools for cavity QED including:
- Jaynes-Cummings model
- Rabi oscillations
- Cavity-atom interactions
- Dissipative dynamics
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import scipy.sparse as sp
class JaynesCummings:
    """Jaynes-Cummings model for cavity-atom interaction."""
    def __init__(self, omega_c: float, omega_a: float, g: float, n_max: int = 20):
        """
        Initialize Jaynes-Cummings system.
        Args:
            omega_c: Cavity frequency
            omega_a: Atomic transition frequency
            g: Coupling strength
            n_max: Maximum photon number
        """
        self.omega_c = omega_c
        self.omega_a = omega_a
        self.g = g
        self.n_max = n_max
        self.detuning = omega_a - omega_c
        # Build Hamiltonian
        self._build_hamiltonian()
    def _build_hamiltonian(self):
        """Construct Jaynes-Cummings Hamiltonian."""
        dim = 2 * (self.n_max + 1)  # Two-level atom × Fock states
        self.H = np.zeros((dim, dim), dtype=complex)
        # Cavity and atom energies
        for n in range(self.n_max + 1):
            # |g, n⟩ state
            idx_g = 2 * n
            self.H[idx_g, idx_g] = n * self.omega_c
            # |e, n⟩ state
            idx_e = 2 * n + 1
            self.H[idx_e, idx_e] = n * self.omega_c + self.omega_a
            # Interaction terms
            if n > 0:
                # |g, n⟩ ↔ |e, n-1⟩
                idx_e_prev = 2 * (n - 1) + 1
                coupling = self.g * np.sqrt(n)
                self.H[idx_g, idx_e_prev] = coupling
                self.H[idx_e_prev, idx_g] = coupling
    def time_evolution(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Calculate time evolution of state.
        Args:
            psi0: Initial state vector
            times: Time points
        Returns:
            States at each time point
        """
        states = np.zeros((len(times), len(psi0)), dtype=complex)
        for i, t in enumerate(times):
            U = expm(-1j * self.H * t)
            states[i] = U @ psi0
        return states
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues of Jaynes-Cummings Hamiltonian."""
        return np.linalg.eigvals(self.H)
    def rabi_oscillations(self, n_photons: int, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Rabi oscillations for initial Fock state.
        Args:
            n_photons: Initial photon number
            times: Time points
        Returns:
            Dictionary with atomic and photon dynamics
        """
        # Initial state |g, n⟩
        psi0 = np.zeros(2 * (self.n_max + 1), dtype=complex)
        psi0[2 * n_photons] = 1
        # Time evolution
        states = self.time_evolution(psi0, times)
        # Calculate observables
        results = {
            'atomic_excitation': np.zeros(len(times)),
            'photon_number': np.zeros(len(times)),
            'g_population': np.zeros(len(times)),
            'e_population': np.zeros(len(times))
        }
        for i, state in enumerate(states):
            # Atomic excitation probability
            e_prob = 0
            g_prob = 0
            avg_n = 0
            for n in range(self.n_max + 1):
                idx_g = 2 * n
                idx_e = 2 * n + 1
                g_prob += np.abs(state[idx_g])**2
                e_prob += np.abs(state[idx_e])**2
                avg_n += n * (np.abs(state[idx_g])**2 + np.abs(state[idx_e])**2)
            results['atomic_excitation'][i] = e_prob
            results['g_population'][i] = g_prob
            results['e_population'][i] = e_prob
            results['photon_number'][i] = avg_n
        return results
    def vacuum_rabi_splitting(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate vacuum Rabi splitting.
        Returns:
            Eigenvalues and eigenvectors showing splitting
        """
        # Focus on single excitation subspace
        H_single = np.array([
            [self.omega_c, self.g],
            [self.g, self.omega_a]
        ])
        eigenvalues, eigenvectors = np.linalg.eigh(H_single)
        return eigenvalues, eigenvectors
    def dressed_states(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dressed states for n-photon manifold.
        Args:
            n: Photon number
        Returns:
            Dressed state energies and states
        """
        if n == 0:
            # Only ground state
            return np.array([0]), np.array([[1]])
        # n-photon manifold: |g,n⟩ and |e,n-1⟩
        H_n = np.array([
            [n * self.omega_c, self.g * np.sqrt(n)],
            [self.g * np.sqrt(n), (n-1) * self.omega_c + self.omega_a]
        ])
        eigenvalues, eigenvectors = np.linalg.eigh(H_n)
        return eigenvalues, eigenvectors
class DissipativeJaynesCummings:
    """Jaynes-Cummings model with dissipation."""
    def __init__(self, omega_c: float, omega_a: float, g: float,
                 kappa: float = 0, gamma: float = 0, n_max: int = 20):
        """
        Initialize dissipative system.
        Args:
            omega_c: Cavity frequency
            omega_a: Atomic frequency
            g: Coupling strength
            kappa: Cavity decay rate
            gamma: Atomic decay rate
            n_max: Maximum photon number
        """
        self.jc = JaynesCummings(omega_c, omega_a, g, n_max)
        self.kappa = kappa
        self.gamma = gamma
        self.n_max = n_max
        # Build Lindblad operators
        self._build_lindblad_operators()
    def _build_lindblad_operators(self):
        """Construct Lindblad jump operators."""
        dim = 2 * (self.n_max + 1)
        # Cavity decay operator
        self.a = np.zeros((dim, dim), dtype=complex)
        for n in range(1, self.n_max + 1):
            # |g,n⟩ → |g,n-1⟩
            self.a[2*(n-1), 2*n] = np.sqrt(n)
            # |e,n⟩ → |e,n-1⟩
            self.a[2*(n-1)+1, 2*n+1] = np.sqrt(n)
        # Atomic decay operator
        self.sigma_minus = np.zeros((dim, dim), dtype=complex)
        for n in range(self.n_max + 1):
            # |e,n⟩ → |g,n⟩
            self.sigma_minus[2*n, 2*n+1] = 1
    def lindblad_evolution(self, rho0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Solve master equation with Lindblad terms.
        Args:
            rho0: Initial density matrix
            times: Time points
        Returns:
            Density matrices at each time
        """
        dim = rho0.shape[0]
        def lindblad_rhs(t, rho_vec):
            rho = rho_vec.reshape((dim, dim))
            # Hamiltonian evolution
            drho = -1j * (self.jc.H @ rho - rho @ self.jc.H)
            # Cavity decay
            if self.kappa > 0:
                drho += self.kappa * (
                    self.a @ rho @ self.a.conj().T -
                    0.5 * (self.a.conj().T @ self.a @ rho + rho @ self.a.conj().T @ self.a)
                )
            # Atomic decay
            if self.gamma > 0:
                drho += self.gamma * (
                    self.sigma_minus @ rho @ self.sigma_minus.conj().T -
                    0.5 * (self.sigma_minus.conj().T @ self.sigma_minus @ rho +
                          rho @ self.sigma_minus.conj().T @ self.sigma_minus)
                )
            return drho.flatten()
        # Solve ODE
        rho0_vec = rho0.flatten()
        sol = solve_ivp(lindblad_rhs, [times[0], times[-1]], rho0_vec,
                       t_eval=times, method='RK45')
        # Reshape results
        rho_t = np.zeros((len(times), dim, dim), dtype=complex)
        for i in range(len(times)):
            rho_t[i] = sol.y[:, i].reshape((dim, dim))
        return rho_t
    def steady_state(self) -> np.ndarray:
        """
        Calculate steady state using eigenvalue method.
        Returns:
            Steady state density matrix
        """
        dim = 2 * (self.n_max + 1)
        # Build Liouvillian superoperator
        L = self._build_liouvillian()
        # Find zero eigenvalue
        eigenvalues, eigenvectors = np.linalg.eig(L)
        idx = np.argmin(np.abs(eigenvalues))
        # Reshape eigenvector to density matrix
        rho_ss = eigenvectors[:, idx].reshape((dim, dim))
        # Normalize
        rho_ss /= np.trace(rho_ss)
        return rho_ss
    def _build_liouvillian(self) -> np.ndarray:
        """Build Liouvillian superoperator."""
        dim = 2 * (self.n_max + 1)
        L = np.zeros((dim**2, dim**2), dtype=complex)
        # Hamiltonian part
        I = np.eye(dim)
        L += -1j * (np.kron(self.jc.H, I) - np.kron(I, self.jc.H.T))
        # Cavity decay
        if self.kappa > 0:
            L += self.kappa * (
                np.kron(self.a, self.a.conj()) -
                0.5 * (np.kron(self.a.conj().T @ self.a, I) +
                      np.kron(I, (self.a.conj().T @ self.a).T))
            )
        # Atomic decay
        if self.gamma > 0:
            L += self.gamma * (
                np.kron(self.sigma_minus, self.sigma_minus.conj()) -
                0.5 * (np.kron(self.sigma_minus.conj().T @ self.sigma_minus, I) +
                      np.kron(I, (self.sigma_minus.conj().T @ self.sigma_minus).T))
            )
        return L
class CavityModes:
    """Optical cavity modes and field distributions."""
    @staticmethod
    def hermite_gaussian(n: int, x: np.ndarray, w0: float = 1.0) -> np.ndarray:
        """
        Hermite-Gaussian mode profile.
        Args:
            n: Mode number
            x: Position array
            w0: Beam waist
        Returns:
            Mode profile
        """
        from scipy.special import hermite
        # Normalized coordinates
        xi = x / w0
        # Hermite polynomial
        H_n = hermite(n)
        # Mode profile
        psi = (1 / (2**n * np.math.factorial(n) * np.sqrt(np.pi * w0)))**0.5 * \
              np.exp(-xi**2 / 2) * H_n(xi)
        return psi
    @staticmethod
    def laguerre_gaussian(p: int, l: int, r: np.ndarray, phi: np.ndarray,
                          w0: float = 1.0) -> np.ndarray:
        """
        Laguerre-Gaussian mode profile.
        Args:
            p: Radial mode number
            l: Azimuthal mode number
            r: Radial coordinate
            phi: Angular coordinate
            w0: Beam waist
        Returns:
            Mode profile
        """
        from scipy.special import genlaguerre
        # Normalized coordinates
        rho = 2 * r**2 / w0**2
        # Generalized Laguerre polynomial
        L = genlaguerre(p, abs(l))
        # Mode profile
        psi = np.sqrt(2 * np.math.factorial(p) /
                     (np.pi * np.math.factorial(p + abs(l)))) * \
              (r * np.sqrt(2) / w0)**abs(l) * \
              np.exp(-r**2 / w0**2) * \
              L(rho) * \
              np.exp(1j * l * phi)
        return psi
    @staticmethod
    def cavity_spectrum(length: float, n_modes: int = 100,
                       fsr: Optional[float] = None) -> np.ndarray:
        """
        Calculate cavity mode frequencies.
        Args:
            length: Cavity length
            n_modes: Number of modes
            fsr: Free spectral range (calculated if None)
        Returns:
            Mode frequencies
        """
        c = 3e8  # Speed of light
        if fsr is None:
            fsr = c / (2 * length)
        # Mode frequencies
        frequencies = np.array([n * fsr for n in range(1, n_modes + 1)])
        return frequencies
    @staticmethod
    def finesse(r1: float, r2: float, loss: float = 0) -> float:
        """
        Calculate cavity finesse.
        Args:
            r1: Mirror 1 reflectivity
            r2: Mirror 2 reflectivity
            loss: Round-trip loss
        Returns:
            Cavity finesse
        """
        r_eff = np.sqrt(r1 * r2) * np.exp(-loss)
        finesse = np.pi * np.sqrt(r_eff) / (1 - r_eff)
        return finesse
    @staticmethod
    def mode_volume(w0: float, length: float) -> float:
        """
        Calculate cavity mode volume.
        Args:
            w0: Beam waist
            length: Cavity length
        Returns:
            Mode volume
        """
        return np.pi * w0**2 * length / 4
class PulseShaping:
    """Quantum pulse shaping and control."""
    @staticmethod
    def gaussian_pulse(t: np.ndarray, t0: float, sigma: float,
                      amplitude: float = 1.0) -> np.ndarray:
        """
        Gaussian pulse shape.
        Args:
            t: Time array
            t0: Pulse center
            sigma: Pulse width
            amplitude: Peak amplitude
        Returns:
            Pulse envelope
        """
        return amplitude * np.exp(-(t - t0)**2 / (2 * sigma**2))
    @staticmethod
    def sech_pulse(t: np.ndarray, t0: float, width: float,
                  amplitude: float = 1.0) -> np.ndarray:
        """
        Hyperbolic secant pulse.
        Args:
            t: Time array
            t0: Pulse center
            width: Pulse width
            amplitude: Peak amplitude
        Returns:
            Pulse envelope
        """
        return amplitude / np.cosh((t - t0) / width)
    @staticmethod
    def chirped_pulse(t: np.ndarray, omega0: float, chirp: float,
                     envelope: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Chirped pulse with frequency sweep.
        Args:
            t: Time array
            omega0: Central frequency
            chirp: Chirp rate
            envelope: Pulse envelope (constant if None)
        Returns:
            Chirped pulse
        """
        if envelope is None:
            envelope = np.ones_like(t)
        phase = omega0 * t + 0.5 * chirp * t**2
        return envelope * np.exp(1j * phase)
    @staticmethod
    def optimal_control_pulse(H0: np.ndarray, H_control: List[np.ndarray],
                             initial: np.ndarray, target: np.ndarray,
                             T: float, n_steps: int = 100,
                             max_iter: int = 100) -> np.ndarray:
        """
        Calculate optimal control pulse using GRAPE algorithm.
        Args:
            H0: Drift Hamiltonian
            H_control: List of control Hamiltonians
            initial: Initial state
            target: Target state
            T: Total time
            n_steps: Number of time steps
            max_iter: Maximum iterations
        Returns:
            Optimal control pulses
        """
        dt = T / n_steps
        n_controls = len(H_control)
        # Initialize random controls
        controls = np.random.randn(n_controls, n_steps) * 0.1
        for iteration in range(max_iter):
            # Forward propagation
            states = np.zeros((n_steps + 1, len(initial)), dtype=complex)
            states[0] = initial
            for i in range(n_steps):
                H = H0.copy()
                for j, H_c in enumerate(H_control):
                    H += controls[j, i] * H_c
                U = expm(-1j * H * dt)
                states[i + 1] = U @ states[i]
            # Calculate fidelity
            fidelity = np.abs(np.vdot(target, states[-1]))**2
            if fidelity > 0.99:
                break
            # Backward propagation (gradient)
            lambda_states = np.zeros((n_steps + 1, len(initial)), dtype=complex)
            lambda_states[-1] = target * np.vdot(target, states[-1]).conj()
            gradients = np.zeros((n_controls, n_steps))
            for i in range(n_steps - 1, -1, -1):
                H = H0.copy()
                for j, H_c in enumerate(H_control):
                    H += controls[j, i] * H_c
                U = expm(-1j * H * dt)
                lambda_states[i] = U.conj().T @ lambda_states[i + 1]
                # Gradient for each control
                for j, H_c in enumerate(H_control):
                    gradients[j, i] = 2 * np.real(
                        1j * np.vdot(lambda_states[i + 1], H_c @ states[i])
                    )
            # Update controls
            learning_rate = 0.1 / (1 + iteration * 0.01)
            controls += learning_rate * gradients
        return controls