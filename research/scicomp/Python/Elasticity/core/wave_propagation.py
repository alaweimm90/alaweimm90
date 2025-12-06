"""
Elastic Wave Propagation Analysis
Comprehensive elastic wave simulation including longitudinal, transverse,
and surface waves in elastic media with various boundary conditions.
"""
import numpy as np
from typing import Tuple, Optional, Callable, List, Union
from dataclasses import dataclass
from scipy import signal, fft
import matplotlib.pyplot as plt
from .stress_strain import ElasticConstants, IsotropicElasticity
@dataclass
class WaveProperties:
    """Properties of elastic waves."""
    frequency: float  # Hz
    wavelength: float  # m
    wave_number: float  # rad/m
    velocity: float  # m/s
    amplitude: float  # displacement amplitude
    angular_frequency: float  # rad/s
    def __post_init__(self):
        """Calculate derived properties."""
        self.angular_frequency = 2 * np.pi * self.frequency
        self.wave_number = 2 * np.pi / self.wavelength
        # Consistency check
        calculated_velocity = self.frequency * self.wavelength
        if abs(calculated_velocity - self.velocity) > 1e-6:
            raise ValueError("Inconsistent wave properties: c ≠ fλ")
class ElasticWave1D:
    """
    1D elastic wave propagation in rods and bars.
    Features:
    - Longitudinal wave simulation
    - Multiple boundary conditions
    - Wave reflection and transmission
    - Dispersion analysis
    - Time-domain and frequency-domain solutions
    Examples:
        >>> wave = ElasticWave1D(length=1.0, material=steel_properties)
        >>> displacement = wave.simulate_wave(time_array, position_array, wave_source)
        >>> reflection_coeff = wave.reflection_coefficient(impedance1, impedance2)
    """
    def __init__(self, length: float, material: ElasticConstants,
                 density: float = 7850, cross_section_area: float = 1.0):
        """
        Initialize 1D elastic wave system.
        Parameters:
            length: Length of the rod (m)
            material: Elastic material properties
            density: Material density (kg/m³)
            cross_section_area: Cross-sectional area (m²)
        """
        self.length = length
        self.material = material
        self.density = density
        self.area = cross_section_area
        # Calculate wave velocity
        self.wave_velocity = np.sqrt(material.youngs_modulus / density)
        # Calculate impedance
        self.impedance = density * self.wave_velocity * cross_section_area
    def fundamental_frequency(self) -> float:
        """Calculate fundamental frequency for fixed-fixed boundary conditions."""
        return self.wave_velocity / (2 * self.length)
    def natural_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """Calculate natural frequencies for fixed-fixed boundary conditions."""
        fundamental = self.fundamental_frequency()
        return np.array([n * fundamental for n in range(1, n_modes + 1)])
    def mode_shapes(self, x: np.ndarray, n_modes: int = 5) -> np.ndarray:
        """
        Calculate mode shapes for fixed-fixed boundary conditions.
        Parameters:
            x: Position array (m)
            n_modes: Number of modes to calculate
        Returns:
            Array of mode shapes [position, mode]
        """
        modes = np.zeros((len(x), n_modes))
        for n in range(1, n_modes + 1):
            modes[:, n-1] = np.sin(n * np.pi * x / self.length)
        return modes
    def wave_equation_solution(self, x: np.ndarray, t: np.ndarray,
                             initial_displacement: Callable,
                             initial_velocity: Callable,
                             boundary_conditions: str = 'fixed-fixed') -> np.ndarray:
        """
        Solve 1D wave equation with given initial and boundary conditions.
        Parameters:
            x: Spatial grid (m)
            t: Time grid (s)
            initial_displacement: Function u(x, 0)
            initial_velocity: Function ∂u/∂t(x, 0)
            boundary_conditions: Type of boundary conditions
        Returns:
            Displacement field u(x, t)
        """
        # D'Alembert solution for infinite domain, then apply boundary conditions
        c = self.wave_velocity
        displacement = np.zeros((len(t), len(x)))
        if boundary_conditions == 'fixed-fixed':
            # Use modal superposition
            frequencies = self.natural_frequencies(50)
            mode_shapes = self.mode_shapes(x, 50)
            # Calculate modal coefficients
            for i, (freq, mode) in enumerate(zip(frequencies, mode_shapes.T)):
                omega = 2 * np.pi * freq
                # Modal initial conditions
                q0 = np.trapz(initial_displacement(x) * mode, x)
                qdot0 = np.trapz(initial_velocity(x) * mode, x)
                # Modal response
                q_t = q0 * np.cos(omega * t) + (qdot0 / omega) * np.sin(omega * t)
                # Add to total response
                for j, time_val in enumerate(t):
                    displacement[j, :] += q_t[j] * mode
        elif boundary_conditions == 'free-free':
            # Similar modal approach but with cosine modes
            frequencies = self.natural_frequencies(50)
            for i, freq in enumerate(frequencies):
                omega = 2 * np.pi * freq
                mode = np.cos(i * np.pi * x / self.length)
                q0 = np.trapz(initial_displacement(x) * mode, x)
                qdot0 = np.trapz(initial_velocity(x) * mode, x)
                q_t = q0 * np.cos(omega * t) + (qdot0 / omega) * np.sin(omega * t)
                for j, time_val in enumerate(t):
                    displacement[j, :] += q_t[j] * mode
        return displacement
    def harmonic_wave(self, x: np.ndarray, t: np.ndarray,
                     frequency: float, amplitude: float = 1.0,
                     phase: float = 0.0, direction: str = 'forward') -> np.ndarray:
        """
        Generate harmonic wave solution.
        Parameters:
            x: Spatial coordinates
            t: Time coordinates
            frequency: Wave frequency (Hz)
            amplitude: Wave amplitude
            phase: Phase shift (rad)
            direction: 'forward' or 'backward'
        Returns:
            Wave displacement field
        """
        omega = 2 * np.pi * frequency
        k = omega / self.wave_velocity
        sign = -1 if direction == 'forward' else 1
        X, T = np.meshgrid(x, t)
        return amplitude * np.cos(omega * T + sign * k * X + phase)
    def pulse_propagation(self, x: np.ndarray, t: np.ndarray,
                         pulse_function: Callable, pulse_speed: float,
                         pulse_position: float = 0.0) -> np.ndarray:
        """
        Simulate pulse propagation using method of characteristics.
        Parameters:
            x: Spatial grid
            t: Time grid
            pulse_function: Function defining pulse shape f(ξ)
            pulse_speed: Pulse propagation speed
            pulse_position: Initial pulse position
        Returns:
            Displacement field
        """
        displacement = np.zeros((len(t), len(x)))
        for i, time_val in enumerate(t):
            # Right-traveling wave
            xi_right = x - pulse_speed * time_val - pulse_position
            # Left-traveling wave (for reflections)
            xi_left = x + pulse_speed * time_val + pulse_position
            displacement[i, :] = pulse_function(xi_right) + pulse_function(xi_left)
        return displacement
    def reflection_coefficient(self, impedance1: float, impedance2: float) -> float:
        """Calculate reflection coefficient at interface."""
        return (impedance2 - impedance1) / (impedance2 + impedance1)
    def transmission_coefficient(self, impedance1: float, impedance2: float) -> float:
        """Calculate transmission coefficient at interface."""
        return 2 * impedance2 / (impedance2 + impedance1)
    def dispersion_relation(self, frequency: np.ndarray,
                          geometry: str = 'rod') -> np.ndarray:
        """
        Calculate dispersion relation for different geometries.
        Parameters:
            frequency: Frequency array (Hz)
            geometry: 'rod', 'plate', or 'cylindrical'
        Returns:
            Wave number array
        """
        omega = 2 * np.pi * frequency
        if geometry == 'rod':
            # Non-dispersive for thin rods
            return omega / self.wave_velocity
        elif geometry == 'plate':
            # Lamb waves - simplified
            h = 0.01  # plate thickness
            c_l = self.wave_velocity
            c_t = c_l / np.sqrt(3)  # approximate shear velocity
            # Simplified dispersion for symmetric mode
            k = omega / c_l
            return k * np.sqrt(1 + (omega * h / c_t)**2)
        else:
            return omega / self.wave_velocity
    def group_velocity(self, frequency: np.ndarray, geometry: str = 'rod') -> np.ndarray:
        """Calculate group velocity from dispersion relation."""
        omega = 2 * np.pi * frequency
        k = self.dispersion_relation(frequency, geometry)
        # Numerical derivative dω/dk
        domega_dk = np.gradient(omega, k)
        return domega_dk
    def energy_density(self, displacement: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate elastic energy density."""
        # Kinetic energy density
        kinetic = 0.5 * self.density * velocity**2
        # Potential energy density (strain energy)
        strain = np.gradient(displacement, axis=1)  # ∂u/∂x
        potential = 0.5 * self.material.youngs_modulus * strain**2
        return kinetic + potential
class ElasticWave2D:
    """
    2D elastic wave propagation in plates and membranes.
    Features:
    - P-waves and S-waves
    - Rayleigh surface waves
    - Lamb waves in plates
    - Seismic wave simulation
    - Wavefront analysis
    Examples:
        >>> wave2d = ElasticWave2D(domain_size=(10, 10), material_props=steel)
        >>> p_wave = wave2d.p_wave_solution(x, y, t, source_location)
        >>> s_wave = wave2d.s_wave_solution(x, y, t, source_location)
    """
    def __init__(self, domain_size: Tuple[float, float],
                 material: ElasticConstants, density: float = 7850):
        """
        Initialize 2D elastic wave system.
        Parameters:
            domain_size: (width, height) of computational domain (m)
            material: Elastic material properties
            density: Material density (kg/m³)
        """
        self.width, self.height = domain_size
        self.material = material
        self.density = density
        # Calculate wave velocities
        lam = material.lame_first
        mu = material.lame_second
        self.p_velocity = np.sqrt((lam + 2 * mu) / density)  # Longitudinal
        self.s_velocity = np.sqrt(mu / density)              # Transverse
        # Rayleigh wave velocity (approximate)
        nu = material.poissons_ratio
        self.rayleigh_velocity = self.s_velocity * (0.87 + 1.12 * nu) / (1 + nu)
    def green_function_2d(self, r: float, t: float, wave_type: str = 'p') -> float:
        """
        Calculate Green's function for 2D elastic waves.
        Parameters:
            r: Distance from source (m)
            t: Time (s)
            wave_type: 'p' for P-waves, 's' for S-waves
        Returns:
            Green's function value
        """
        if wave_type == 'p':
            c = self.p_velocity
        elif wave_type == 's':
            c = self.s_velocity
        else:
            raise ValueError("wave_type must be 'p' or 's'")
        # Causal Green's function
        if c * t < r:
            return 0.0
        else:
            # Simplified form for 2D
            return 1.0 / (2 * np.pi * np.sqrt((c * t)**2 - r**2))
    def point_source_solution(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                            source_x: float, source_y: float,
                            source_function: Callable,
                            wave_type: str = 'p') -> np.ndarray:
        """
        Solution for point source in infinite medium.
        Parameters:
            x, y: Spatial coordinate arrays
            t: Time array
            source_x, source_y: Source location
            source_function: Time-dependent source function
            wave_type: 'p' or 's' waves
        Returns:
            Displacement field [time, y, x]
        """
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - source_x)**2 + (Y - source_y)**2)
        if wave_type == 'p':
            c = self.p_velocity
        else:
            c = self.s_velocity
        displacement = np.zeros((len(t), len(y), len(x)))
        for i, time_val in enumerate(t):
            # Retarded time
            t_ret = time_val - R / c
            # Apply causality
            mask = t_ret >= 0
            # Evaluate source function at retarded times
            source_values = np.zeros_like(R)
            source_values[mask] = np.array([source_function(tr) for tr in t_ret[mask]])
            # Apply Green's function
            displacement[i, :, :] = source_values / (2 * np.pi * R + 1e-12)
        return displacement
    def plane_wave_solution(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                          wave_vector: Tuple[float, float], frequency: float,
                          amplitude: float = 1.0, wave_type: str = 'p') -> np.ndarray:
        """
        Plane wave solution in 2D.
        Parameters:
            x, y: Spatial coordinates
            t: Time array
            wave_vector: (kx, ky) wave vector components
            frequency: Wave frequency
            amplitude: Wave amplitude
            wave_type: 'p' or 's' waves
        Returns:
            Displacement field
        """
        kx, ky = wave_vector
        omega = 2 * np.pi * frequency
        if wave_type == 'p':
            c = self.p_velocity
        else:
            c = self.s_velocity
        # Dispersion relation: ω = c|k|
        k_mag = np.sqrt(kx**2 + ky**2)
        if abs(omega - c * k_mag) > 1e-6:
            warnings.warn("Dispersion relation not satisfied")
        X, Y = np.meshgrid(x, y)
        displacement = np.zeros((len(t), len(y), len(x)))
        for i, time_val in enumerate(t):
            phase = omega * time_val - kx * X - ky * Y
            displacement[i, :, :] = amplitude * np.cos(phase)
        return displacement
    def rayleigh_wave_solution(self, x: np.ndarray, z: np.ndarray, t: np.ndarray,
                             frequency: float, amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rayleigh surface wave solution.
        Parameters:
            x: Horizontal coordinate array
            z: Depth coordinate array (z=0 at surface)
            t: Time array
            frequency: Wave frequency
            amplitude: Wave amplitude
        Returns:
            Tuple of (horizontal_displacement, vertical_displacement)
        """
        omega = 2 * np.pi * frequency
        k = omega / self.rayleigh_velocity
        # Rayleigh wave decay parameters
        nu = self.material.poissons_ratio
        gamma_p = k * np.sqrt(1 - (self.rayleigh_velocity / self.p_velocity)**2)
        gamma_s = k * np.sqrt(1 - (self.rayleigh_velocity / self.s_velocity)**2)
        X, Z = np.meshgrid(x, z)
        u_x = np.zeros((len(t), len(z), len(x)))
        u_z = np.zeros((len(t), len(z), len(x)))
        for i, time_val in enumerate(t):
            phase = omega * time_val - k * X
            # Horizontal displacement
            u_x[i, :, :] = amplitude * (
                np.exp(-gamma_p * Z) -
                (gamma_s / gamma_p) * np.exp(-gamma_s * Z)
            ) * np.cos(phase)
            # Vertical displacement
            u_z[i, :, :] = amplitude * (
                (k / gamma_p) * np.exp(-gamma_p * Z) -
                (k / gamma_s) * np.exp(-gamma_s * Z)
            ) * np.sin(phase)
        return u_x, u_z
    def lamb_wave_modes(self, frequency: np.ndarray, plate_thickness: float,
                       n_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Lamb wave dispersion curves for a plate.
        Parameters:
            frequency: Frequency array
            plate_thickness: Plate thickness
            n_modes: Number of modes to calculate
        Returns:
            Tuple of (symmetric_modes, antisymmetric_modes) wave numbers
        """
        omega = 2 * np.pi * frequency
        h = plate_thickness / 2  # half thickness
        c_l = self.p_velocity
        c_t = self.s_velocity
        symmetric_modes = np.zeros((n_modes, len(frequency)))
        antisymmetric_modes = np.zeros((n_modes, len(frequency)))
        for i, f in enumerate(frequency):
            w = omega[i]
            # Search for roots of dispersion equation
            for mode in range(n_modes):
                # Simplified dispersion relation - would need full solver for accuracy
                # This is an approximation
                if mode == 0:
                    # Fundamental symmetric mode (extensional)
                    k_s = w / c_l
                    symmetric_modes[mode, i] = k_s
                    # Fundamental antisymmetric mode (flexural)
                    k_a = (w / c_t) * np.sqrt(f * h / c_t)  # approximate
                    antisymmetric_modes[mode, i] = k_a
                else:
                    # Higher modes - simplified
                    k_s = (w / c_l) * (1 + mode * 0.1)
                    k_a = (w / c_t) * (1 + mode * 0.15)
                    symmetric_modes[mode, i] = k_s
                    antisymmetric_modes[mode, i] = k_a
        return symmetric_modes, antisymmetric_modes
    def wavefront_analysis(self, displacement_field: np.ndarray,
                          x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
        """
        Analyze wavefront propagation characteristics.
        Parameters:
            displacement_field: Displacement field [time, y, x]
            x, y: Spatial coordinates
            t: Time array
        Returns:
            Dictionary with wavefront properties
        """
        # Calculate arrival times
        threshold = 0.1 * np.max(np.abs(displacement_field))
        arrival_times = np.full((len(y), len(x)), np.inf)
        for i in range(len(y)):
            for j in range(len(x)):
                time_series = displacement_field[:, i, j]
                arrival_idx = np.where(np.abs(time_series) > threshold)[0]
                if len(arrival_idx) > 0:
                    arrival_times[i, j] = t[arrival_idx[0]]
        # Calculate wavefront velocities
        X, Y = np.meshgrid(x, y)
        center_x, center_y = x[len(x)//2], y[len(y)//2]
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        # Fit velocity from arrival times vs distance
        valid_mask = np.isfinite(arrival_times) & (arrival_times > 0)
        if np.sum(valid_mask) > 10:
            coeffs = np.polyfit(distances[valid_mask], arrival_times[valid_mask], 1)
            estimated_velocity = 1.0 / coeffs[0]
        else:
            estimated_velocity = np.nan
        return {
            'arrival_times': arrival_times,
            'estimated_velocity': estimated_velocity,
            'p_velocity_theoretical': self.p_velocity,
            's_velocity_theoretical': self.s_velocity
        }
class WaveInteraction:
    """
    Analysis of wave interactions, scattering, and diffraction.
    Features:
    - Wave scattering by obstacles
    - Diffraction around edges
    - Wave interference patterns
    - Attenuation and absorption
    """
    def __init__(self, material: ElasticConstants, density: float = 7850):
        """Initialize wave interaction analyzer."""
        self.material = material
        self.density = density
        elasticity = IsotropicElasticity(material.youngs_modulus, material.poissons_ratio, density)
        self.p_velocity, self.s_velocity = elasticity.elastic_wave_velocities()
    def scattering_cross_section(self, obstacle_radius: float, frequency: float,
                               wave_type: str = 'p') -> float:
        """
        Calculate scattering cross-section for circular obstacle.
        Parameters:
            obstacle_radius: Radius of scattering obstacle
            frequency: Wave frequency
            wave_type: 'p' or 's' waves
        Returns:
            Scattering cross-section
        """
        if wave_type == 'p':
            c = self.p_velocity
        else:
            c = self.s_velocity
        k = 2 * np.pi * frequency / c
        ka = k * obstacle_radius
        # Low frequency approximation (Rayleigh scattering)
        if ka < 1:
            return np.pi * obstacle_radius**2 * (ka)**4 / 4
        # High frequency approximation (geometric scattering)
        elif ka > 10:
            return 2 * np.pi * obstacle_radius**2
        # Intermediate range - simplified Mie scattering
        else:
            return np.pi * obstacle_radius**2 * (1 - np.cos(2 * ka)) / 2
    def interference_pattern(self, source1_pos: Tuple[float, float],
                           source2_pos: Tuple[float, float],
                           x: np.ndarray, y: np.ndarray,
                           frequency: float, phase_diff: float = 0.0) -> np.ndarray:
        """
        Calculate interference pattern from two coherent sources.
        Parameters:
            source1_pos, source2_pos: Source positions (x, y)
            x, y: Observation coordinates
            frequency: Wave frequency
            phase_diff: Phase difference between sources
        Returns:
            Interference pattern
        """
        k = 2 * np.pi * frequency / self.p_velocity
        X, Y = np.meshgrid(x, y)
        # Calculate distances to each source
        r1 = np.sqrt((X - source1_pos[0])**2 + (Y - source1_pos[1])**2)
        r2 = np.sqrt((X - source2_pos[0])**2 + (Y - source2_pos[1])**2)
        # Path difference
        path_diff = r2 - r1
        # Interference pattern
        return 2 * np.cos(k * path_diff / 2 + phase_diff / 2)
    def attenuation_coefficient(self, frequency: float, quality_factor: float = 100) -> float:
        """
        Calculate attenuation coefficient due to material damping.
        Parameters:
            frequency: Wave frequency
            quality_factor: Material quality factor Q
        Returns:
            Attenuation coefficient (1/m)
        """
        k = 2 * np.pi * frequency / self.p_velocity
        return k / (2 * quality_factor)