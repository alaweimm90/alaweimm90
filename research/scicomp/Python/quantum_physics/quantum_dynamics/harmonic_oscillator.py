#!/usr/bin/env python3
"""Quantum harmonic oscillator with eigenstates and coherent state dynamics."""
import numpy as np
import scipy.linalg
import scipy.special
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable
import warnings
try:
    from ...utils.constants import hbar, me
    from ...utils.units import energy_convert
except ImportError:
    # Fallback values
    hbar = 1.054571817e-34
    me = 9.1093837015e-31
    def energy_convert(*args, **kwargs):
        return args[0] if args else 1.0
class QuantumHarmonic:
    """
    Comprehensive quantum harmonic oscillator implementation.
    Provides analytical solutions, numerical methods, and visualization
    tools for the quantum harmonic oscillator in one dimension.
    Parameters
    ----------
    omega : float
        Angular frequency in rad/s
    mass : float, default me
        Particle mass in kg
    n_max : int, default 50
        Maximum quantum number for basis
    x_max : float, default 10.0
        Maximum position for grid (in units of x0)
    n_points : int, default 1000
        Number of grid points
    Attributes
    ----------
    omega : float
        Angular frequency
    mass : float
        Particle mass
    x0 : float
        Characteristic length scale
    E0 : float
        Zero-point energy
    """
    def __init__(self, omega: float,
                 mass: float = me,
                 n_max: int = 50,
                 x_max: float = 10.0,
                 n_points: int = 1000):
        """Initialize quantum harmonic oscillator."""
        self.omega = omega
        self.mass = mass
        self.n_max = n_max
        # Characteristic scales
        self.x0 = np.sqrt(hbar / (mass * omega))  # Length scale
        self.E0 = 0.5 * hbar * omega  # Zero-point energy
        # Position grid
        self.x_max = x_max
        self.n_points = n_points
        self.x = np.linspace(-x_max * self.x0, x_max * self.x0, n_points)
        self.dx = self.x[1] - self.x[0]
        # Dimensionless coordinate
        self.xi = self.x / self.x0
        # Precompute eigenstates if needed
        self._eigenstates_cached = {}
        self._eigenvalues_cached = {}
    def energy(self, n: int) -> float:
        """
        Calculate energy eigenvalue for quantum number n.
        Parameters
        ----------
        n : int
            Quantum number (n >= 0)
        Returns
        -------
        float
            Energy eigenvalue in Joules
        """
        if n < 0:
            raise ValueError("Quantum number must be non-negative")
        return hbar * self.omega * (n + 0.5)
    def eigenstate(self, n: int,
                  x: Optional[np.ndarray] = None,
                  normalized: bool = True) -> np.ndarray:
        """
        Calculate eigenstate wavefunction using Hermite polynomials.
        The wavefunction is given by:
        ψₙ(x) = (mω/πℏ)^(1/4) * (1/√(2ⁿn!)) * exp(-mωx²/2ℏ) * Hₙ(√(mω/ℏ)x)
        Parameters
        ----------
        n : int
            Quantum number
        x : ndarray, optional
            Position grid. If None, uses internal grid
        normalized : bool, default True
            Whether to return normalized wavefunction
        Returns
        -------
        ndarray
            Wavefunction values
        """
        if n < 0:
            raise ValueError("Quantum number must be non-negative")
        if x is None:
            x = self.x
            xi = self.xi
        else:
            xi = x / self.x0
        # Check cache
        cache_key = (n, len(x), x[0] if len(x) > 0 else 0, x[-1] if len(x) > 0 else 0)
        if cache_key in self._eigenstates_cached:
            return self._eigenstates_cached[cache_key]
        # Calculate Hermite polynomial
        H_n = scipy.special.hermite(n, monic=False)
        # Calculate wavefunction
        if normalized:
            # Normalization constant
            N = (1.0 / (np.pi**0.25 * np.sqrt(2**n * scipy.special.factorial(n))))
            psi = N * np.exp(-xi**2 / 2) * H_n(xi)
        else:
            psi = np.exp(-xi**2 / 2) * H_n(xi)
        # Cache result
        self._eigenstates_cached[cache_key] = psi
        return psi
    def coherent_state(self, alpha: complex,
                      x: Optional[np.ndarray] = None,
                      n_terms: Optional[int] = None) -> np.ndarray:
        """
        Generate coherent state |α⟩.
        Coherent states are superpositions of energy eigenstates:
        |α⟩ = exp(-|α|²/2) * Σₙ (αⁿ/√n!) |n⟩
        Parameters
        ----------
        alpha : complex
            Coherent state parameter
        x : ndarray, optional
            Position grid
        n_terms : int, optional
            Number of terms in expansion. If None, uses adaptive cutoff
        Returns
        -------
        ndarray
            Coherent state wavefunction
        """
        if x is None:
            x = self.x
        if n_terms is None:
            # Adaptive cutoff based on |α|
            n_terms = max(20, int(3 * np.abs(alpha)**2) + 20)
            n_terms = min(n_terms, self.n_max)
        # Initialize wavefunction
        psi = np.zeros(len(x), dtype=complex)
        # Coherent state expansion
        normalization = np.exp(-0.5 * np.abs(alpha)**2)
        for n in range(n_terms):
            coefficient = normalization * (alpha**n) / np.sqrt(scipy.special.factorial(n))
            psi += coefficient * self.eigenstate(n, x)
        return psi
    def time_evolution(self, psi_initial: np.ndarray,
                      t: Union[float, np.ndarray],
                      method: str = 'analytical') -> np.ndarray:
        """
        Evolve wavefunction in time.
        Parameters
        ----------
        psi_initial : ndarray
            Initial wavefunction
        t : float or ndarray
            Time(s) for evolution
        method : str, default 'analytical'
            Evolution method ('analytical', 'numerical')
        Returns
        -------
        ndarray
            Time-evolved wavefunction(s)
        """
        if method == 'analytical':
            return self._analytical_time_evolution(psi_initial, t)
        elif method == 'numerical':
            return self._numerical_time_evolution(psi_initial, t)
        else:
            raise ValueError(f"Unknown evolution method: {method}")
    def _analytical_time_evolution(self, psi_initial: np.ndarray,
                                 t: Union[float, np.ndarray]) -> np.ndarray:
        """Analytical time evolution using eigenstate expansion."""
        # Expand initial state in energy eigenbasis
        coefficients = self._expand_in_eigenbasis(psi_initial)
        # Time evolution
        if np.isscalar(t):
            psi_t = np.zeros_like(psi_initial, dtype=complex)
            for n, c_n in enumerate(coefficients):
                if np.abs(c_n) > 1e-12:  # Skip negligible terms
                    phase = np.exp(-1j * self.energy(n) * t / hbar)
                    psi_t += c_n * phase * self.eigenstate(n)
            return psi_t
        else:
            # Multiple times
            psi_t = np.zeros((len(t), len(psi_initial)), dtype=complex)
            for i, time in enumerate(t):
                for n, c_n in enumerate(coefficients):
                    if np.abs(c_n) > 1e-12:
                        phase = np.exp(-1j * self.energy(n) * time / hbar)
                        psi_t[i] += c_n * phase * self.eigenstate(n)
            return psi_t
    def _expand_in_eigenbasis(self, psi: np.ndarray,
                            n_terms: Optional[int] = None) -> np.ndarray:
        """Expand wavefunction in energy eigenbasis."""
        if n_terms is None:
            n_terms = min(50, self.n_max)
        coefficients = np.zeros(n_terms, dtype=complex)
        for n in range(n_terms):
            eigenstate_n = self.eigenstate(n)
            coefficients[n] = np.trapz(np.conj(eigenstate_n) * psi, self.x)
        return coefficients
    def expectation_value(self, psi: np.ndarray,
                         observable: str,
                         **kwargs) -> Union[float, complex]:
        """
        Calculate expectation value of observable.
        Parameters
        ----------
        psi : ndarray
            Quantum state
        observable : str
            Observable name ('x', 'p', 'x2', 'p2', 'H', 'L')
        **kwargs
            Additional parameters for observable
        Returns
        -------
        float or complex
            Expectation value
        """
        if observable == 'x':
            return self._expectation_position(psi)
        elif observable == 'p':
            return self._expectation_momentum(psi)
        elif observable == 'x2':
            return self._expectation_position_squared(psi)
        elif observable == 'p2':
            return self._expectation_momentum_squared(psi)
        elif observable == 'H':
            return self._expectation_hamiltonian(psi)
        elif observable == 'L':
            return self._expectation_angular_momentum(psi)
        else:
            raise ValueError(f"Unknown observable: {observable}")
    def _expectation_position(self, psi: np.ndarray) -> complex:
        """Calculate ⟨x⟩."""
        integrand = np.conj(psi) * self.x * psi
        return np.trapz(integrand, self.x)
    def _expectation_momentum(self, psi: np.ndarray) -> complex:
        """Calculate ⟨p⟩ using finite differences."""
        # p = -iℏ d/dx
        dpsi_dx = np.gradient(psi, self.dx)
        integrand = np.conj(psi) * (-1j * hbar * dpsi_dx)
        return np.trapz(integrand, self.x)
    def _expectation_position_squared(self, psi: np.ndarray) -> complex:
        """Calculate ⟨x²⟩."""
        integrand = np.conj(psi) * self.x**2 * psi
        return np.trapz(integrand, self.x)
    def _expectation_momentum_squared(self, psi: np.ndarray) -> complex:
        """Calculate ⟨p²⟩."""
        # p² = -ℏ² d²/dx²
        d2psi_dx2 = np.gradient(np.gradient(psi, self.dx), self.dx)
        integrand = np.conj(psi) * (-hbar**2 * d2psi_dx2)
        return np.trapz(integrand, self.x)
    def _expectation_hamiltonian(self, psi: np.ndarray) -> complex:
        """Calculate ⟨H⟩."""
        # H = p²/(2m) + (1/2)mω²x²
        T = self._expectation_momentum_squared(psi) / (2 * self.mass)
        V = 0.5 * self.mass * self.omega**2 * self._expectation_position_squared(psi)
        return T + V
    def wigner_function(self, psi: np.ndarray,
                       x_range: Optional[Tuple[float, float]] = None,
                       p_range: Optional[Tuple[float, float]] = None,
                       n_points: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Wigner function for phase space representation.
        The Wigner function is defined as:
        W(x,p) = (1/πℏ) ∫ dy ψ*(x+y/2) ψ(x-y/2) exp(ipy/ℏ)
        Parameters
        ----------
        psi : ndarray
            Quantum state wavefunction
        x_range : tuple, optional
            (x_min, x_max) for phase space grid
        p_range : tuple, optional
            (p_min, p_max) for phase space grid
        n_points : tuple, default (100, 100)
            Number of points in (x, p) directions
        Returns
        -------
        X : ndarray
            Position meshgrid
        P : ndarray
            Momentum meshgrid
        W : ndarray
            Wigner function values
        """
        if x_range is None:
            x_max = 3 * self.x0
            x_range = (-x_max, x_max)
        if p_range is None:
            p_max = 3 * hbar / self.x0
            p_range = (-p_max, p_max)
        x_w = np.linspace(x_range[0], x_range[1], n_points[0])
        p_w = np.linspace(p_range[0], p_range[1], n_points[1])
        X, P = np.meshgrid(x_w, p_w, indexing='ij')
        W = np.zeros_like(X)
        # Calculate Wigner function
        for i, x_val in enumerate(x_w):
            for j, p_val in enumerate(p_w):
                W[i, j] = self._wigner_point(psi, x_val, p_val)
        return X, P, W
    def _wigner_point(self, psi: np.ndarray, x_val: float, p_val: float) -> float:
        """Calculate Wigner function at a single point."""
        # Interpolate wavefunction to find values at x ± y/2
        y_max = min(2 * np.abs(self.x).max(), 10 * self.x0)
        y = np.linspace(-y_max, y_max, 200)
        dy = y[1] - y[0]
        # Interpolate psi at x + y/2 and x - y/2
        psi_plus = np.interp(x_val + y/2, self.x, psi, left=0, right=0)
        psi_minus = np.interp(x_val - y/2, self.x, psi, left=0, right=0)
        # Calculate integrand
        integrand = np.conj(psi_plus) * psi_minus * np.exp(1j * p_val * y / hbar)
        # Integrate
        result = np.trapz(integrand, y) / (np.pi * hbar)
        return np.real(result)
    def thermal_state(self, temperature: float,
                     n_terms: Optional[int] = None) -> np.ndarray:
        """
        Generate thermal state at given temperature.
        ρ_thermal = exp(-βH) / Tr[exp(-βH)]
        where β = 1/(kB*T)
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        n_terms : int, optional
            Number of terms in thermal expansion
        Returns
        -------
        ndarray
            Thermal state density matrix
        """
        if n_terms is None:
            # Adaptive cutoff based on temperature
            beta = 1.0 / (1.380649e-23 * temperature)  # kB in SI units
            n_terms = max(10, int(3 / (beta * hbar * self.omega)) + 10)
            n_terms = min(n_terms, self.n_max)
        # Calculate thermal weights
        beta = 1.0 / (1.380649e-23 * temperature)
        weights = np.zeros(n_terms)
        for n in range(n_terms):
            weights[n] = np.exp(-beta * self.energy(n))
        # Normalize
        Z = np.sum(weights)  # Partition function
        weights /= Z
        return weights
    def plot_eigenstate(self, n: int,
                       show_probability: bool = True,
                       figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        """
        Plot eigenstate wavefunction and probability density.
        Parameters
        ----------
        n : int
            Quantum number
        show_probability : bool, default True
            Whether to show probability density
        figsize : tuple, default (10, 6)
            Figure size
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        psi_n = self.eigenstate(n)
        fig, (ax1, ax2) = plt.subplots(1, 2 if show_probability else 1,
                                      figsize=figsize)
        # Wavefunction
        ax1.plot(self.x / self.x0, np.real(psi_n), 'b-',
                linewidth=2, label=f'Re[ψ_{n}(x)]')
        if np.any(np.imag(psi_n) != 0):
            ax1.plot(self.x / self.x0, np.imag(psi_n), 'r--',
                    linewidth=2, label=f'Im[ψ_{n}(x)]')
        ax1.set_xlabel('x/x₀')
        ax1.set_ylabel('ψₙ(x)')
        ax1.set_title(f'Eigenstate n={n}, E={self.energy(n)/self.E0:.1f}E₀')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if show_probability:
            # Probability density
            ax2.plot(self.x / self.x0, np.abs(psi_n)**2, 'g-', linewidth=2)
            ax2.fill_between(self.x / self.x0, np.abs(psi_n)**2, alpha=0.3, color='green')
            ax2.set_xlabel('x/x₀')
            ax2.set_ylabel('|ψₙ(x)|²')
            ax2.set_title('Probability Density')
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
# Convenience functions
def harmonic_eigenstate(n: int, omega: float, x: np.ndarray,
                       mass: float = me) -> np.ndarray:
    """
    Calculate harmonic oscillator eigenstate.
    Parameters
    ----------
    n : int
        Quantum number
    omega : float
        Angular frequency
    x : ndarray
        Position grid
    mass : float, default me
        Particle mass
    Returns
    -------
    ndarray
        Eigenstate wavefunction
    """
    qho = QuantumHarmonic(omega, mass)
    return qho.eigenstate(n, x)
def coherent_state(alpha: complex, omega: float, x: np.ndarray,
                  mass: float = me) -> np.ndarray:
    """
    Generate coherent state.
    Parameters
    ----------
    alpha : complex
        Coherent state parameter
    omega : float
        Angular frequency
    x : ndarray
        Position grid
    mass : float, default me
        Particle mass
    Returns
    -------
    ndarray
        Coherent state wavefunction
    """
    qho = QuantumHarmonic(omega, mass)
    return qho.coherent_state(alpha, x)
def harmonic_time_evolution(psi_initial: np.ndarray, omega: float,
                          t: Union[float, np.ndarray],
                          x: np.ndarray, mass: float = me) -> np.ndarray:
    """
    Time evolution for harmonic oscillator.
    Parameters
    ----------
    psi_initial : ndarray
        Initial wavefunction
    omega : float
        Angular frequency
    t : float or ndarray
        Time(s)
    x : ndarray
        Position grid
    mass : float, default me
        Particle mass
    Returns
    -------
    ndarray
        Time-evolved wavefunction
    """
    qho = QuantumHarmonic(omega, mass, x_max=np.max(np.abs(x))/np.sqrt(hbar/(mass*omega)),
                         n_points=len(x))
    return qho.time_evolution(psi_initial, t)
def wigner_function(psi: np.ndarray, x: np.ndarray, omega: float,
                   mass: float = me) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Wigner function.
    Parameters
    ----------
    psi : ndarray
        Quantum state
    x : ndarray
        Position grid
    omega : float
        Angular frequency
    mass : float, default me
        Particle mass
    Returns
    -------
    X, P, W : ndarray
        Position, momentum meshgrids and Wigner function
    """
    qho = QuantumHarmonic(omega, mass, x_max=np.max(np.abs(x))/np.sqrt(hbar/(mass*omega)),
                         n_points=len(x))
    return qho.wigner_function(psi)