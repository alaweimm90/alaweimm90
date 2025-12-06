"""
Spin dynamics and magnetization evolution.
This module provides tools for spin dynamics including:
- Landau-Lifshitz-Gilbert equation
- Spin precession dynamics
- Magnetic anisotropy effects
- Damping mechanisms
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Callable
from scipy.integrate import solve_ivp
from scipy.linalg import norm
import matplotlib.pyplot as plt
class LandauLifshitzGilbert:
    """Landau-Lifshitz-Gilbert equation solver."""
    def __init__(self, gamma: float = 2.21e5, alpha: float = 0.01):
        """
        Initialize LLG solver.
        Args:
            gamma: Gyromagnetic ratio (m/(A·s²))
            alpha: Gilbert damping parameter
        """
        self.gamma = gamma
        self.alpha = alpha
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space
    def effective_field(self, m: np.ndarray, **params) -> np.ndarray:
        """
        Calculate effective magnetic field.
        Args:
            m: Magnetization vector (normalized)
            **params: Field parameters
        Returns:
            Effective field vector
        """
        H_eff = np.zeros(3)
        # External field
        if 'H_ext' in params:
            H_eff += params['H_ext']
        # Uniaxial anisotropy
        if 'K_u' in params and 'u_axis' in params:
            K_u = params['K_u']
            u_axis = params['u_axis'] / norm(params['u_axis'])
            H_anis = 2 * K_u / self.mu0 * np.dot(m, u_axis) * u_axis
            H_eff += H_anis
        # Exchange field (simplified)
        if 'A_ex' in params:
            A_ex = params['A_ex']
            # Simplified exchange as second derivative (requires spatial discretization)
            if 'neighbors' in params:
                H_ex = 2 * A_ex / self.mu0 * params['neighbors']
                H_eff += H_ex
        # Demagnetization field (simplified for ellipsoid)
        if 'N_demag' in params:
            N = params['N_demag']  # Demagnetization tensor
            H_demag = -N @ m
            H_eff += H_demag
        return H_eff
    def llg_equation(self, t: float, m: np.ndarray, **params) -> np.ndarray:
        """
        LLG equation: dm/dt = -γ/(1+α²) [m×H_eff + α m×(m×H_eff)].
        Args:
            t: Time
            m: Magnetization vector
            **params: Field parameters
        Returns:
            Time derivative dm/dt
        """
        H_eff = self.effective_field(m, **params)
        # Cross products
        m_cross_H = np.cross(m, H_eff)
        m_cross_mH = np.cross(m, m_cross_H)
        # LLG equation
        prefactor = -self.gamma / (1 + self.alpha**2)
        dmdt = prefactor * (m_cross_H + self.alpha * m_cross_mH)
        return dmdt
    def solve(self, m0: np.ndarray, t_span: Tuple[float, float],
              t_eval: np.ndarray, **params) -> Dict:
        """
        Solve LLG equation.
        Args:
            m0: Initial magnetization
            t_span: Time span (start, end)
            t_eval: Time points for evaluation
            **params: Field parameters
        Returns:
            Solution dictionary
        """
        # Normalize initial magnetization
        m0 = m0 / norm(m0)
        # Solve ODE
        sol = solve_ivp(
            lambda t, m: self.llg_equation(t, m, **params),
            t_span, m0, t_eval=t_eval,
            method='RK45', rtol=1e-6
        )
        # Extract components
        mx, my, mz = sol.y
        # Calculate energies
        energies = self._calculate_energies(sol.y.T, **params)
        return {
            'time': sol.t,
            'magnetization': sol.y.T,
            'mx': mx, 'my': my, 'mz': mz,
            'energies': energies,
            'success': sol.success
        }
    def _calculate_energies(self, magnetization: np.ndarray, **params) -> Dict:
        """Calculate magnetic energies during evolution."""
        energies = {
            'zeeman': np.zeros(len(magnetization)),
            'anisotropy': np.zeros(len(magnetization)),
            'demagnetization': np.zeros(len(magnetization)),
            'total': np.zeros(len(magnetization))
        }
        for i, m in enumerate(magnetization):
            # Zeeman energy
            if 'H_ext' in params:
                energies['zeeman'][i] = -self.mu0 * np.dot(m, params['H_ext'])
            # Anisotropy energy
            if 'K_u' in params and 'u_axis' in params:
                K_u = params['K_u']
                u_axis = params['u_axis'] / norm(params['u_axis'])
                energies['anisotropy'][i] = -K_u * np.dot(m, u_axis)**2
            # Demagnetization energy
            if 'N_demag' in params:
                N = params['N_demag']
                energies['demagnetization'][i] = -0.5 * self.mu0 * m.T @ N @ m
            energies['total'][i] = (energies['zeeman'][i] +
                                  energies['anisotropy'][i] +
                                  energies['demagnetization'][i])
        return energies
class SpinWaves:
    """Spin wave analysis and dispersion relations."""
    def __init__(self, Ms: float, A_ex: float, gamma: float = 2.21e5):
        """
        Initialize spin wave analysis.
        Args:
            Ms: Saturation magnetization
            A_ex: Exchange stiffness
            gamma: Gyromagnetic ratio
        """
        self.Ms = Ms
        self.A_ex = A_ex
        self.gamma = gamma
        self.mu0 = 4 * np.pi * 1e-7
    def dispersion_relation(self, k: np.ndarray, H_ext: float = 0,
                           K_u: float = 0) -> np.ndarray:
        """
        Calculate spin wave dispersion relation.
        Args:
            k: Wave vector array
            H_ext: External field
            K_u: Uniaxial anisotropy constant
        Returns:
            Frequency array
        """
        # Exchange frequency
        omega_ex = self.gamma * 2 * self.A_ex * k**2 / (self.mu0 * self.Ms)
        # Zeeman frequency
        omega_H = self.gamma * self.mu0 * H_ext
        # Anisotropy frequency
        omega_A = self.gamma * 2 * K_u / (self.mu0 * self.Ms)
        # Total frequency
        omega = np.sqrt((omega_H + omega_A + omega_ex) * (omega_H + omega_A + omega_ex))
        return omega
    def group_velocity(self, k: np.ndarray, **params) -> np.ndarray:
        """
        Calculate group velocity v_g = dω/dk.
        Args:
            k: Wave vector array
            **params: Material parameters
        Returns:
            Group velocity array
        """
        dk = k[1] - k[0] if len(k) > 1 else 1e-6
        omega = self.dispersion_relation(k, **params)
        # Numerical derivative
        v_g = np.gradient(omega, dk)
        return v_g
    def magnon_density_of_states(self, omega: np.ndarray,
                                lattice_param: float) -> np.ndarray:
        """
        Calculate magnon density of states.
        Args:
            omega: Frequency array
            lattice_param: Lattice parameter
        Returns:
            Density of states
        """
        # For 3D simple cubic lattice
        V_BZ = (2 * np.pi / lattice_param)**3  # Brillouin zone volume
        # Approximate DOS (needs proper k-space integration)
        dos = omega**2 / (2 * np.pi**2) * (lattice_param / (2 * np.pi))**3
        return dos
class MagneticDomains:
    """Magnetic domain wall dynamics."""
    def __init__(self, Ms: float, A_ex: float, K_u: float,
                 width: float = 1e-6):
        """
        Initialize domain wall system.
        Args:
            Ms: Saturation magnetization
            A_ex: Exchange stiffness
            K_u: Uniaxial anisotropy
            width: Domain wall width
        """
        self.Ms = Ms
        self.A_ex = A_ex
        self.K_u = K_u
        self.width = width
        self.mu0 = 4 * np.pi * 1e-7
        # Domain wall parameters
        self.delta_w = np.sqrt(A_ex / K_u)  # Wall width
        self.gamma_w = 4 * np.sqrt(A_ex * K_u)  # Wall energy density
    def bloch_wall_profile(self, x: np.ndarray, x0: float = 0) -> Dict:
        """
        Bloch domain wall magnetization profile.
        Args:
            x: Position array
            x0: Wall center position
        Returns:
            Magnetization components
        """
        xi = (x - x0) / self.delta_w
        # Bloch wall profile
        mx = np.zeros_like(x)
        my = 1 / np.cosh(xi)
        mz = np.tanh(xi)
        return {'mx': mx, 'my': my, 'mz': mz, 'position': x}
    def neel_wall_profile(self, x: np.ndarray, x0: float = 0) -> Dict:
        """
        Néel domain wall magnetization profile.
        Args:
            x: Position array
            x0: Wall center position
        Returns:
            Magnetization components
        """
        xi = (x - x0) / self.delta_w
        # Néel wall profile
        mx = 1 / np.cosh(xi)
        my = np.zeros_like(x)
        mz = np.tanh(xi)
        return {'mx': mx, 'my': my, 'mz': mz, 'position': x}
    def walker_breakdown(self, H_field: float) -> Tuple[float, float]:
        """
        Calculate Walker breakdown field and velocity.
        Args:
            H_field: Applied field
        Returns:
            Breakdown field and velocity
        """
        alpha = 0.01  # Typical damping
        gamma = 2.21e5  # Gyromagnetic ratio
        # Walker breakdown field
        H_walker = alpha * 2 * self.K_u / (self.mu0 * self.Ms)
        # Walker velocity
        if H_field < H_walker:
            v_walker = gamma * self.delta_w * H_field / alpha
        else:
            # Above breakdown, velocity oscillates
            v_walker = gamma * self.delta_w * H_walker / alpha
        return H_walker, v_walker
    def domain_wall_energy(self, wall_type: str = 'bloch') -> float:
        """
        Calculate domain wall energy per unit area.
        Args:
            wall_type: 'bloch' or 'neel'
        Returns:
            Wall energy density
        """
        if wall_type.lower() == 'bloch':
            # Bloch wall energy
            energy = 4 * np.sqrt(self.A_ex * self.K_u)
        elif wall_type.lower() == 'neel':
            # Néel wall energy (approximately same as Bloch)
            energy = 4 * np.sqrt(self.A_ex * self.K_u)
        else:
            raise ValueError("Wall type must be 'bloch' or 'neel'")
        return energy
class SpinTorque:
    """Spin-transfer torque effects."""
    def __init__(self, gamma: float = 2.21e5, hbar: float = 1.055e-34):
        """
        Initialize spin torque calculations.
        Args:
            gamma: Gyromagnetic ratio
            hbar: Reduced Planck constant
        """
        self.gamma = gamma
        self.hbar = hbar
        self.e = 1.602e-19  # Elementary charge
        self.mu0 = 4 * np.pi * 1e-7
    def slonczewski_torque(self, m: np.ndarray, p: np.ndarray,
                          current: float, thickness: float,
                          spin_polarization: float = 0.4) -> np.ndarray:
        """
        Calculate Slonczewski spin-transfer torque.
        Args:
            m: Free layer magnetization
            p: Fixed layer magnetization
            current: Current density (A/m²)
            thickness: Free layer thickness
            spin_polarization: Spin polarization efficiency
        Returns:
            Torque vector
        """
        # Normalize vectors
        m = m / norm(m)
        p = p / norm(p)
        # Torque coefficient
        tau_0 = self.hbar * current * spin_polarization / (2 * self.e * thickness)
        # Slonczewski torque: τ = (ℏ/2e) * J * P * [m × (m × p)]
        m_cross_p = np.cross(m, p)
        torque = -tau_0 * np.cross(m, m_cross_p)
        return torque
    def field_like_torque(self, m: np.ndarray, p: np.ndarray,
                         current: float, beta: float = 0.1) -> np.ndarray:
        """
        Calculate field-like torque.
        Args:
            m: Free layer magnetization
            p: Fixed layer magnetization
            current: Current density
            beta: Field-like torque efficiency
        Returns:
            Field-like torque
        """
        m = m / norm(m)
        p = p / norm(p)
        # Field-like torque: τ_fl = β * (J/e) * (m × p)
        torque = beta * current / self.e * np.cross(m, p)
        return torque
    def critical_current(self, Ms: float, thickness: float,
                        alpha: float, spin_polarization: float = 0.4) -> float:
        """
        Calculate critical current for switching.
        Args:
            Ms: Saturation magnetization
            thickness: Layer thickness
            alpha: Gilbert damping
            spin_polarization: Spin polarization
        Returns:
            Critical current density
        """
        # Simplified critical current
        J_c = (2 * self.e * alpha * Ms * thickness) / (self.hbar * spin_polarization)
        return J_c
class MagnetocrystallineAnisotropy:
    """Magnetocrystalline anisotropy calculations."""
    def __init__(self, crystal_class: str = 'cubic'):
        """
        Initialize anisotropy calculations.
        Args:
            crystal_class: Crystal class ('cubic', 'uniaxial', 'orthorhombic')
        """
        self.crystal_class = crystal_class.lower()
    def anisotropy_energy(self, m: np.ndarray, **constants) -> float:
        """
        Calculate magnetocrystalline anisotropy energy.
        Args:
            m: Magnetization direction (normalized)
            **constants: Anisotropy constants
        Returns:
            Anisotropy energy density
        """
        mx, my, mz = m / norm(m)
        if self.crystal_class == 'cubic':
            # Cubic anisotropy: K₁(mx²my² + my²mz² + mz²mx²) + K₂(mx²my²mz²)
            K1 = constants.get('K1', 0)
            K2 = constants.get('K2', 0)
            E = K1 * (mx**2 * my**2 + my**2 * mz**2 + mz**2 * mx**2)
            E += K2 * (mx**2 * my**2 * mz**2)
        elif self.crystal_class == 'uniaxial':
            # Uniaxial anisotropy: K₁sin²θ + K₂sin⁴θ
            K1 = constants.get('K1', 0)
            K2 = constants.get('K2', 0)
            axis = constants.get('axis', np.array([0, 0, 1]))
            axis = axis / norm(axis)
            cos_theta = np.dot(m, axis)
            sin2_theta = 1 - cos_theta**2
            E = K1 * sin2_theta + K2 * sin2_theta**2
        elif self.crystal_class == 'orthorhombic':
            # Orthorhombic: Kₓmx² + Kymy² + Kzmz²
            Kx = constants.get('Kx', 0)
            Ky = constants.get('Ky', 0)
            Kz = constants.get('Kz', 0)
            E = Kx * mx**2 + Ky * my**2 + Kz * mz**2
        else:
            raise ValueError(f"Unknown crystal class: {self.crystal_class}")
        return E
    def anisotropy_field(self, m: np.ndarray, **constants) -> np.ndarray:
        """
        Calculate anisotropy field H_anis = -∂E/∂m.
        Args:
            m: Magnetization direction
            **constants: Anisotropy constants
        Returns:
            Anisotropy field
        """
        # Numerical gradient
        eps = 1e-8
        H = np.zeros(3)
        for i in range(3):
            m_plus = m.copy()
            m_minus = m.copy()
            m_plus[i] += eps
            m_minus[i] -= eps
            E_plus = self.anisotropy_energy(m_plus, **constants)
            E_minus = self.anisotropy_energy(m_minus, **constants)
            H[i] = -(E_plus - E_minus) / (2 * eps)
        return H
    def easy_axes(self, **constants) -> List[np.ndarray]:
        """
        Find crystallographic easy axes.
        Args:
            **constants: Anisotropy constants
        Returns:
            List of easy axis directions
        """
        if self.crystal_class == 'cubic':
            K1 = constants.get('K1', 0)
            if K1 < 0:
                # <100> easy axes
                return [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                       np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
            else:
                # <111> easy axes
                return [np.array([1, 1, 1])/np.sqrt(3), np.array([1, 1, -1])/np.sqrt(3),
                       np.array([1, -1, 1])/np.sqrt(3), np.array([-1, 1, 1])/np.sqrt(3)]
        elif self.crystal_class == 'uniaxial':
            axis = constants.get('axis', np.array([0, 0, 1]))
            return [axis / norm(axis), -axis / norm(axis)]
        else:
            # Numerical approach for complex cases
            return []