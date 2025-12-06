"""Wave Optics Module.
This module provides comprehensive wave optics functionality including
wave propagation, diffraction, interference, and coherence analysis.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel, jv
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from .visualization import OpticsVisualizer
@dataclass
class WaveParameters:
    """Container for wave parameters."""
    wavelength: float
    frequency: float
    wavenumber: float
    angular_frequency: float
    period: float
    @classmethod
    def from_wavelength(cls, wavelength: float, c: float = 2.99792458e8):
        """Create from wavelength."""
        frequency = c / wavelength
        wavenumber = 2 * np.pi / wavelength
        angular_frequency = 2 * np.pi * frequency
        period = 1 / frequency
        return cls(wavelength, frequency, wavenumber, angular_frequency, period)
class WaveOptics:
    """Base class for wave optics calculations."""
    def __init__(self, wavelength: float, medium_index: float = 1.0):
        """Initialize wave optics calculator.
        Args:
            wavelength: Wavelength in vacuum (meters)
            medium_index: Refractive index of medium
        """
        self.wavelength_vacuum = wavelength
        self.medium_index = medium_index
        self.wavelength = wavelength / medium_index
        self.k = 2 * np.pi / self.wavelength
        self.frequency = 2.99792458e8 / wavelength
        # Berkeley color scheme
        self.berkeley_blue = '#003262'
        self.california_gold = '#FDB515'
    def validate_input(self, *args):
        """Validate input parameters."""
        for arg in args:
            if isinstance(arg, (int, float)):
                if not np.isfinite(arg):
                    raise ValueError("All parameters must be finite")
            elif isinstance(arg, np.ndarray):
                if not np.all(np.isfinite(arg)):
                    raise ValueError("All array elements must be finite")
class PlaneWave(WaveOptics):
    """Plane wave propagation and analysis."""
    def __init__(self, wavelength: float, amplitude: float = 1.0,
                 phase: float = 0.0, direction: np.ndarray = None,
                 medium_index: float = 1.0):
        """Initialize plane wave.
        Args:
            wavelength: Wavelength in vacuum (meters)
            amplitude: Wave amplitude
            phase: Initial phase (radians)
            direction: Propagation direction unit vector
            medium_index: Refractive index
        """
        super().__init__(wavelength, medium_index)
        self.amplitude = amplitude
        self.phase = phase
        self.direction = direction if direction is not None else np.array([0, 0, 1])
        self.direction = self.direction / np.linalg.norm(self.direction)
    def field_at_point(self, r: np.ndarray, t: float = 0) -> complex:
        """Calculate field at a point in space and time.
        Args:
            r: Position vector(s) [x, y, z] (meters)
            t: Time (seconds)
        Returns:
            Complex field amplitude
        """
        r = np.asarray(r)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        # Calculate phase
        k_dot_r = np.dot(r, self.k * self.direction)
        omega_t = 2 * np.pi * self.frequency * t
        return self.amplitude * np.exp(1j * (k_dot_r - omega_t + self.phase))
    def intensity_at_point(self, r: np.ndarray, t: float = 0) -> float:
        """Calculate intensity at a point.
        Args:
            r: Position vector(s) (meters)
            t: Time (seconds)
        Returns:
            Intensity (W/m²)
        """
        field = self.field_at_point(r, t)
        return np.abs(field)**2
    def propagate_distance(self, distance: float) -> 'PlaneWave':
        """Propagate plane wave by a distance.
        Args:
            distance: Propagation distance (meters)
        Returns:
            New PlaneWave object at propagated position
        """
        phase_change = self.k * distance
        new_phase = self.phase + phase_change
        return PlaneWave(
            self.wavelength_vacuum, self.amplitude, new_phase,
            self.direction, self.medium_index
        )
class SphericalWave(WaveOptics):
    """Spherical wave propagation and analysis."""
    def __init__(self, wavelength: float, source_power: float = 1.0,
                 source_position: np.ndarray = None, medium_index: float = 1.0):
        """Initialize spherical wave.
        Args:
            wavelength: Wavelength in vacuum (meters)
            source_power: Source power (watts)
            source_position: Source position [x, y, z] (meters)
            medium_index: Refractive index
        """
        super().__init__(wavelength, medium_index)
        self.source_power = source_power
        self.source_position = source_position if source_position is not None else np.zeros(3)
    def field_at_point(self, r: np.ndarray, t: float = 0) -> complex:
        """Calculate field at a point.
        Args:
            r: Position vector(s) (meters)
            t: Time (seconds)
        Returns:
            Complex field amplitude
        """
        r = np.asarray(r)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        # Distance from source
        dr = r - self.source_position
        distance = np.linalg.norm(dr, axis=-1)
        # Avoid division by zero
        distance = np.where(distance == 0, 1e-12, distance)
        # Spherical wave field
        amplitude = np.sqrt(self.source_power / (4 * np.pi)) / distance
        phase = self.k * distance - 2 * np.pi * self.frequency * t
        return amplitude * np.exp(1j * phase)
    def intensity_at_point(self, r: np.ndarray, t: float = 0) -> float:
        """Calculate intensity at a point.
        Args:
            r: Position vector(s) (meters)
            t: Time (seconds)
        Returns:
            Intensity (W/m²)
        """
        field = self.field_at_point(r, t)
        return np.abs(field)**2
class GaussianBeam(WaveOptics):
    """Gaussian beam propagation using ABCD matrix method."""
    def __init__(self, wavelength: float, waist_radius: float,
                 waist_position: float = 0, medium_index: float = 1.0,
                 power: float = 1.0):
        """Initialize Gaussian beam.
        Args:
            wavelength: Wavelength in vacuum (meters)
            waist_radius: Beam waist radius (meters)
            waist_position: Z position of waist (meters)
            medium_index: Refractive index
            power: Beam power (watts)
        """
        super().__init__(wavelength, medium_index)
        self.waist_radius = waist_radius
        self.waist_position = waist_position
        self.power = power
        # Derived parameters
        self.rayleigh_range = np.pi * waist_radius**2 / self.wavelength
        self.divergence_angle = self.wavelength / (np.pi * waist_radius)
    def beam_radius(self, z: float) -> float:
        """Calculate beam radius at position z.
        Args:
            z: Axial position (meters)
        Returns:
            Beam radius (meters)
        """
        z_rel = z - self.waist_position
        return self.waist_radius * np.sqrt(1 + (z_rel / self.rayleigh_range)**2)
    def radius_of_curvature(self, z: float) -> float:
        """Calculate radius of curvature at position z.
        Args:
            z: Axial position (meters)
        Returns:
            Radius of curvature (meters), inf at waist
        """
        z_rel = z - self.waist_position
        if abs(z_rel) < 1e-12:
            return np.inf
        return z_rel * (1 + (self.rayleigh_range / z_rel)**2)
    def gouy_phase(self, z: float) -> float:
        """Calculate Gouy phase at position z.
        Args:
            z: Axial position (meters)
        Returns:
            Gouy phase (radians)
        """
        z_rel = z - self.waist_position
        return np.arctan(z_rel / self.rayleigh_range)
    def field_profile(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        """Calculate transverse field profile at position z.
        Args:
            x, y: Transverse coordinates (meters)
            z: Axial position (meters)
        Returns:
            Complex field amplitude
        """
        x, y = np.meshgrid(x, y)
        r_squared = x**2 + y**2
        w_z = self.beam_radius(z)
        R_z = self.radius_of_curvature(z)
        zeta = self.gouy_phase(z)
        # Gaussian beam field
        amplitude = (self.waist_radius / w_z) * np.sqrt(2 * self.power /
                    (np.pi * self.waist_radius**2))
        # Phase terms
        if np.isfinite(R_z):
            phase = self.k * r_squared / (2 * R_z) - zeta
        else:
            phase = -zeta
        return amplitude * np.exp(-r_squared / w_z**2) * np.exp(1j * phase)
    def intensity_profile(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        """Calculate intensity profile at position z.
        Args:
            x, y: Transverse coordinates (meters)
            z: Axial position (meters)
        Returns:
            Intensity distribution (W/m²)
        """
        field = self.field_profile(x, y, z)
        return np.abs(field)**2
    def propagate_through_lens(self, focal_length: float,
                              distance_to_lens: float) -> 'GaussianBeam':
        """Propagate Gaussian beam through a thin lens.
        Args:
            focal_length: Lens focal length (meters)
            distance_to_lens: Distance from current waist to lens (meters)
        Returns:
            New GaussianBeam after lens
        """
        # ABCD matrix for free space + lens + free space
        z1 = distance_to_lens - self.waist_position
        # Current beam parameters at lens
        w_lens = self.beam_radius(distance_to_lens)
        R_lens = self.radius_of_curvature(distance_to_lens)
        # q parameter at lens
        if np.isfinite(R_lens):
            q_lens = 1j * np.pi * w_lens**2 / self.wavelength + 1 / R_lens
        else:
            q_lens = 1j * np.pi * w_lens**2 / self.wavelength
        # Transform through lens
        q_after_lens = q_lens / (1 - q_lens / focal_length)
        # Extract new beam parameters
        w_new_squared = -self.wavelength / (np.pi * np.imag(1 / q_after_lens))
        w_new = np.sqrt(w_new_squared)
        if np.real(1 / q_after_lens) != 0:
            R_new = 1 / np.real(1 / q_after_lens)
        else:
            R_new = np.inf
        return GaussianBeam(
            self.wavelength_vacuum, w_new, distance_to_lens,
            self.medium_index, self.power
        )
def propagate_fresnel(aperture_function: callable, wavelength: float,
                     propagation_distance: float, x_aperture: np.ndarray,
                     x_screen: np.ndarray) -> np.ndarray:
    """Propagate optical field using Fresnel diffraction.
    Args:
        aperture_function: Function defining aperture transmission
        wavelength: Wavelength (meters)
        propagation_distance: Distance to observation screen (meters)
        x_aperture: Aperture coordinates (meters)
        x_screen: Screen coordinates (meters)
    Returns:
        Complex field amplitude at screen
    """
    k = 2 * np.pi / wavelength
    # Initialize output field
    field_screen = np.zeros(len(x_screen), dtype=complex)
    # Fresnel diffraction integral
    for i, x_s in enumerate(x_screen):
        integrand = lambda x_a: (aperture_function(x_a) *
                                np.exp(1j * k * (x_s - x_a)**2 / (2 * propagation_distance)))
        # Numerical integration
        real_part = quad(lambda x: np.real(integrand(x)),
                        x_aperture[0], x_aperture[-1])[0]
        imag_part = quad(lambda x: np.imag(integrand(x)),
                        x_aperture[0], x_aperture[-1])[0]
        field_screen[i] = real_part + 1j * imag_part
    # Fresnel diffraction prefactor
    prefactor = np.sqrt(k / (2j * np.pi * propagation_distance))
    return prefactor * field_screen
def calculate_diffraction(aperture_type: str, aperture_size: float,
                         wavelength: float, screen_distance: float,
                         screen_size: float, num_points: int = 1000) -> Dict[str, Any]:
    """Calculate diffraction patterns for various apertures.
    Args:
        aperture_type: Type of aperture ('single_slit', 'double_slit', 'circular')
        aperture_size: Characteristic size (meters)
        wavelength: Wavelength (meters)
        screen_distance: Distance to screen (meters)
        screen_size: Size of observation screen (meters)
        num_points: Number of calculation points
    Returns:
        Dictionary with position and intensity arrays
    """
    x = np.linspace(-screen_size/2, screen_size/2, num_points)
    if aperture_type == 'single_slit':
        # Single slit diffraction
        theta = x / screen_distance  # Small angle approximation
        beta = np.pi * aperture_size * np.sin(theta) / wavelength
        # Avoid division by zero
        beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
        intensity = (np.sin(beta) / beta)**2
    elif aperture_type == 'double_slit':
        # Double slit with slit width aperture_size/10 and separation aperture_size
        slit_width = aperture_size / 10
        slit_separation = aperture_size
        theta = x / screen_distance
        beta = np.pi * slit_width * np.sin(theta) / wavelength
        alpha = np.pi * slit_separation * np.sin(theta) / wavelength
        beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
        envelope = (np.sin(beta) / beta)**2
        interference = np.cos(alpha)**2
        intensity = envelope * interference
    elif aperture_type == 'circular':
        # Circular aperture (Airy disk)
        theta = x / screen_distance
        u = np.pi * aperture_size * np.sin(theta) / wavelength
        # First-order Bessel function
        u = np.where(np.abs(u) < 1e-10, 1e-10, u)
        intensity = (2 * jv(1, u) / u)**2
    else:
        raise ValueError(f"Unknown aperture type: {aperture_type}")
    return {
        'position': x,
        'intensity': intensity,
        'intensity_normalized': intensity / np.max(intensity),
        'aperture_type': aperture_type,
        'aperture_size': aperture_size,
        'wavelength': wavelength,
        'screen_distance': screen_distance
    }
def analyze_interference(wavelength: float, source_separation: float,
                        screen_distance: float, pattern_type: str = 'double_slit',
                        coherence_length: float = None, num_points: int = 1000) -> Dict[str, Any]:
    """Analyze interference patterns.
    Args:
        wavelength: Wavelength (meters)
        source_separation: Separation between sources (meters)
        screen_distance: Distance to observation screen (meters)
        pattern_type: Type of interference ('double_slit', 'young', 'michelson')
        coherence_length: Coherence length for partial coherence (meters)
        num_points: Number of calculation points
    Returns:
        Dictionary with interference analysis results
    """
    # Screen coordinates
    screen_size = 10 * wavelength * screen_distance / source_separation
    x = np.linspace(-screen_size/2, screen_size/2, num_points)
    if pattern_type in ['double_slit', 'young']:
        # Young's double slit interference
        theta = x / screen_distance  # Small angle approximation
        path_difference = source_separation * np.sin(theta)
        phase_difference = 2 * np.pi * path_difference / wavelength
        # Intensity pattern
        intensity = 4 * np.cos(phase_difference / 2)**2
        # Add partial coherence effects if specified
        if coherence_length is not None:
            coherence_factor = np.exp(-np.abs(path_difference) / coherence_length)
            visibility = coherence_factor
            intensity = 2 * (1 + visibility * np.cos(phase_difference))
    elif pattern_type == 'michelson':
        # Michelson interferometer
        phase_difference = 4 * np.pi * source_separation / wavelength  # Assuming source_separation is arm difference
        intensity = 2 * (1 + np.cos(phase_difference))
    else:
        raise ValueError(f"Unknown interference type: {pattern_type}")
    # Calculate fringe parameters
    fringe_spacing = wavelength * screen_distance / source_separation
    central_max_positions = np.where(np.diff(np.sign(np.diff(intensity))) < 0)[0] + 1
    # Visibility calculation
    I_max = np.max(intensity)
    I_min = np.min(intensity)
    visibility = (I_max - I_min) / (I_max + I_min)
    return {
        'position': x,
        'intensity': intensity,
        'intensity_normalized': intensity / np.max(intensity),
        'fringe_spacing': fringe_spacing,
        'visibility': visibility,
        'central_maxima': central_max_positions,
        'pattern_type': pattern_type,
        'wavelength': wavelength,
        'source_separation': source_separation,
        'screen_distance': screen_distance
    }
def fresnel_number(aperture_radius: float, wavelength: float,
                  distance: float) -> float:
    """Calculate Fresnel number for circular aperture.
    Args:
        aperture_radius: Radius of circular aperture (meters)
        wavelength: Wavelength (meters)
        distance: Distance to observation point (meters)
    Returns:
        Fresnel number (dimensionless)
    """
    return aperture_radius**2 / (wavelength * distance)
def rayleigh_range(wavelength: float, waist_radius: float,
                  medium_index: float = 1.0) -> float:
    """Calculate Rayleigh range for Gaussian beam.
    Args:
        wavelength: Wavelength in vacuum (meters)
        waist_radius: Beam waist radius (meters)
        medium_index: Refractive index
    Returns:
        Rayleigh range (meters)
    """
    wavelength_medium = wavelength / medium_index
    return np.pi * waist_radius**2 / wavelength_medium
# Example usage and demonstrations
def demo_wave_optics():
    """Demonstrate wave optics functionality."""
    print("Wave Optics Demo")
    print("================")
    # Parameters
    wavelength = 633e-9  # HeNe laser
    # 1. Gaussian beam propagation
    print("\n1. Gaussian Beam Propagation")
    beam = GaussianBeam(wavelength, waist_radius=1e-3, power=1e-3)
    z_positions = np.linspace(-5e-3, 5e-3, 100)
    beam_radii = [beam.beam_radius(z) for z in z_positions]
    plt.figure(figsize=(10, 6))
    plt.plot(z_positions*1000, np.array(beam_radii)*1000,
             color='#003262', linewidth=2, label='Beam Radius')
    plt.axhline(beam.waist_radius*1000, color='#FDB515',
                linestyle='--', label='Waist Radius')
    plt.xlabel('Position (mm)')
    plt.ylabel('Beam Radius (mm)')
    plt.title('Gaussian Beam Propagation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Rayleigh range: {beam.rayleigh_range*1000:.2f} mm")
    print(f"Divergence angle: {beam.divergence_angle*1000:.2f} mrad")
    # 2. Single slit diffraction
    print("\n2. Single Slit Diffraction")
    diffraction = calculate_diffraction(
        'single_slit', 50e-6, wavelength, 1.0, 0.01
    )
    plt.figure(figsize=(10, 6))
    plt.plot(diffraction['position']*1000, diffraction['intensity_normalized'],
             color='#003262', linewidth=2)
    plt.xlabel('Position (mm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Single Slit Diffraction Pattern')
    plt.grid(True, alpha=0.3)
    plt.show()
    # 3. Double slit interference
    print("\n3. Double Slit Interference")
    interference = analyze_interference(wavelength, 100e-6, 1.0)
    plt.figure(figsize=(10, 6))
    plt.plot(interference['position']*1000, interference['intensity_normalized'],
             color='#FDB515', linewidth=2)
    plt.xlabel('Position (mm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Double Slit Interference Pattern')
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Fringe spacing: {interference['fringe_spacing']*1000:.3f} mm")
    print(f"Visibility: {interference['visibility']:.3f}")
if __name__ == "__main__":
    demo_wave_optics()