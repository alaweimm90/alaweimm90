"""Optics Package.
This package provides comprehensive tools for optical physics and photonics including
wave optics, ray optics, laser physics, nonlinear optics, and optical simulation.
Modules:
    wave_optics: Wave propagation, diffraction, and interference
    ray_optics: Geometric optics and ray tracing
    laser_physics: Laser cavity design, gain media, and beam analysis
    nonlinear_optics: Nonlinear optical processes and frequency conversion
    optical_materials: Material properties, dispersion, and absorption
    interferometry: Interferometric measurements and analysis
    beam_propagation: Gaussian beam propagation and ABCD matrix methods
    polarization: Polarization states and Jones/Mueller calculus
    fiber_optics: Optical fiber modes, dispersion, and propagation
    visualization: Berkeley-themed plotting for optical systems
Classes:
    OpticalSystem: Base class for optical systems
    OpticalElement: Base class for optical elements
    BeamPropagator: Beam propagation methods
    InterferometerAnalyzer: Interferometry analysis
    FiberMode: Fiber optic mode analysis
    NonlinearProcess: Nonlinear optical processes
Functions:
    propagate_beam: General beam propagation
    trace_ray: Ray tracing through optical systems
    calculate_interference: Interference pattern calculation
    analyze_spectrum: Spectral analysis
    design_cavity: Laser cavity design
Author: Berkeley SciComp Team
Date: 2024
"""
from .wave_optics import (
    WaveOptics, PlaneWave, SphericalWave, GaussianBeam,
    propagate_fresnel, calculate_diffraction, analyze_interference,
    fresnel_number, rayleigh_range
)
from .ray_optics import (
    RayOptics, OpticalRay, ThinLens, ThickLens, Mirror,
    SphericalSurface, AsphericSurface, trace_ray_through_system,
    paraxial_ray_trace, calculate_aberrations
)
from .laser_physics import (
    LaserCavity, GainMedium, LaserMode, BeamQuality,
    calculate_threshold, analyze_stability, design_resonator,
    gaussian_beam_parameters, beam_quality_factor
)
from .nonlinear_optics import (
    NonlinearCrystal, SecondHarmonic, ThirdHarmonic, FourWaveMixing,
    OpticalParametricAmplifier, calculate_phase_matching,
    efficiency_calculation, chi2_process, chi3_process
)
from .optical_materials import (
    OpticalMaterial, Sellmeier, Cauchy, LorentzDrude,
    calculate_refractive_index, dispersion_analysis,
    absorption_coefficient, group_velocity_dispersion
)
from .interferometry import (
    Interferometer, MichelsonInterferometer, MachZehnderInterferometer,
    FabryPerotInterferometer, analyze_fringe_pattern,
    visibility_measurement, phase_extraction
)
from .beam_propagation import (
    BeamPropagator, ABCDMatrix, GaussianBeamPropagation,
    propagate_through_lens, beam_waist_calculation,
    mode_matching, cavity_stability_analysis
)
from .polarization import (
    PolarizationState, JonesVector, JonesMatrix, MuellerMatrix,
    StokesVector, linear_polarizer, quarter_wave_plate,
    polarization_ellipse, degree_of_polarization
)
from .fiber_optics import (
    OpticalFiber, StepIndexFiber, GradedIndexFiber, FiberMode,
    calculate_modes, numerical_aperture, cutoff_frequency,
    dispersion_calculation, fiber_coupling_efficiency
)
from .visualization import (
    OpticsVisualizer, plot_beam_profile, plot_ray_diagram,
    plot_interference_pattern, animate_wave_propagation,
    create_optical_system_diagram
)
# Package metadata
__version__ = "1.0.0"
__author__ = "Berkeley SciComp Team"
__email__ = "scicomp@berkeley.edu"
# Package-level configuration
import numpy as np
import matplotlib.pyplot as plt
# Physical constants for optics
SPEED_OF_LIGHT = 2.99792458e8  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
REDUCED_PLANCK = PLANCK_CONSTANT / (2 * np.pi)  # ℏ
ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # H/m
VACUUM_IMPEDANCE = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)  # Ω
# Common wavelengths (m)
WAVELENGTHS = {
    'UV_A': 365e-9,      # Near UV
    'Violet': 400e-9,     # Violet light
    'Blue': 450e-9,       # Blue light
    'Green': 532e-9,      # Green laser (Nd:YAG 2nd harmonic)
    'Yellow': 589e-9,     # Sodium D-line
    'Red': 633e-9,        # HeNe laser
    'NIR': 800e-9,        # Near infrared
    'Telecom_O': 1310e-9, # Telecom O-band
    'Telecom_C': 1550e-9, # Telecom C-band
    'CO2': 10.6e-6,       # CO2 laser
}
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
# Configure matplotlib for Berkeley styling
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [10, 6],
    'axes.prop_cycle': plt.cycler('color', [BERKELEY_BLUE, CALIFORNIA_GOLD,
                                          '#3B7EA1', '#C4820E', '#00B0DA']),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10
})
# Utility functions
def wavelength_to_frequency(wavelength):
    """Convert wavelength to frequency.
    Args:
        wavelength: Wavelength in meters
    Returns:
        Frequency in Hz
    """
    return SPEED_OF_LIGHT / wavelength
def frequency_to_wavelength(frequency):
    """Convert frequency to wavelength.
    Args:
        frequency: Frequency in Hz
    Returns:
        Wavelength in meters
    """
    return SPEED_OF_LIGHT / frequency
def photon_energy(wavelength):
    """Calculate photon energy.
    Args:
        wavelength: Wavelength in meters
    Returns:
        Photon energy in Joules
    """
    return PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength
def validate_wavelength(wavelength):
    """Validate wavelength input.
    Args:
        wavelength: Wavelength in meters
    Raises:
        ValueError: If wavelength is invalid
    """
    wavelength = np.asarray(wavelength)
    if not np.all(wavelength > 0):
        raise ValueError("Wavelength must be positive")
    if not np.all(wavelength < 1):  # Sanity check: less than 1 meter
        raise ValueError("Wavelength seems too large, check units (should be meters)")
    return wavelength
def validate_refractive_index(n):
    """Validate refractive index.
    Args:
        n: Refractive index (complex)
    Raises:
        ValueError: If refractive index is invalid
    """
    n = np.asarray(n)
    if not np.all(np.real(n) >= 1):
        raise ValueError("Real part of refractive index must be >= 1")
    if not np.all(np.imag(n) >= 0):
        raise ValueError("Imaginary part of refractive index must be >= 0")
    return n
# Quick access functions
def quick_gaussian_beam(wavelength, waist_radius, distance):
    """Quick Gaussian beam propagation calculation.
    Args:
        wavelength: Wavelength in meters
        waist_radius: Beam waist radius in meters
        distance: Propagation distance in meters
    Returns:
        Dictionary with beam parameters
    """
    from .beam_propagation import GaussianBeamPropagation
    beam = GaussianBeamPropagation(wavelength, waist_radius)
    return beam.propagate(distance)
def quick_interference_pattern(wavelength, slit_separation, screen_distance, num_points=1000):
    """Quick double-slit interference pattern calculation.
    Args:
        wavelength: Wavelength in meters
        slit_separation: Slit separation in meters
        screen_distance: Distance to screen in meters
        num_points: Number of points on screen
    Returns:
        Dictionary with position and intensity arrays
    """
    from .wave_optics import analyze_interference
    return analyze_interference(
        wavelength, slit_separation, screen_distance,
        pattern_type='double_slit', num_points=num_points
    )
def quick_ray_trace(surfaces, ray_height=0, ray_angle=0, wavelength=589e-9):
    """Quick paraxial ray trace through optical system.
    Args:
        surfaces: List of surface dictionaries
        ray_height: Initial ray height in meters
        ray_angle: Initial ray angle in radians
        wavelength: Wavelength in meters
    Returns:
        Ray trace results
    """
    from .ray_optics import paraxial_ray_trace
    return paraxial_ray_trace(surfaces, ray_height, ray_angle, wavelength)
# Package info
def get_package_info():
    """Get package information."""
    return {
        'name': 'Optics',
        'version': __version__,
        'author': __author__,
        'description': 'Berkeley SciComp Optics and Photonics Package',
        'modules': {
            'wave_optics': 'Wave propagation, diffraction, interference',
            'ray_optics': 'Geometric optics and ray tracing',
            'laser_physics': 'Laser cavity design and beam analysis',
            'nonlinear_optics': 'Nonlinear optical processes',
            'optical_materials': 'Material properties and dispersion',
            'interferometry': 'Interferometric measurements',
            'beam_propagation': 'Gaussian beam propagation',
            'polarization': 'Polarization states and analysis',
            'fiber_optics': 'Optical fiber modes and propagation',
            'visualization': 'Berkeley-themed optical plotting'
        },
        'constants': {
            'c': SPEED_OF_LIGHT,
            'h': PLANCK_CONSTANT,
            'hbar': REDUCED_PLANCK,
            'e': ELEMENTARY_CHARGE,
            'epsilon0': VACUUM_PERMITTIVITY,
            'mu0': VACUUM_PERMEABILITY,
            'Z0': VACUUM_IMPEDANCE
        },
        'wavelengths': WAVELENGTHS
    }
# Convenience imports for common use cases
__all__ = [
    # Core classes
    'OpticalSystem', 'OpticalElement', 'BeamPropagator',
    # Wave optics
    'WaveOptics', 'PlaneWave', 'SphericalWave', 'GaussianBeam',
    'propagate_fresnel', 'calculate_diffraction', 'analyze_interference',
    # Ray optics
    'RayOptics', 'OpticalRay', 'ThinLens', 'ThickLens', 'Mirror',
    'trace_ray_through_system', 'paraxial_ray_trace', 'calculate_aberrations',
    # Laser physics
    'LaserCavity', 'GainMedium', 'LaserMode', 'BeamQuality',
    'calculate_threshold', 'analyze_stability', 'design_resonator',
    # Nonlinear optics
    'NonlinearCrystal', 'SecondHarmonic', 'ThirdHarmonic',
    'calculate_phase_matching', 'efficiency_calculation',
    # Materials
    'OpticalMaterial', 'Sellmeier', 'Cauchy',
    'calculate_refractive_index', 'dispersion_analysis',
    # Interferometry
    'Interferometer', 'MichelsonInterferometer', 'FabryPerotInterferometer',
    'analyze_fringe_pattern', 'visibility_measurement',
    # Beam propagation
    'ABCDMatrix', 'GaussianBeamPropagation',
    'propagate_through_lens', 'beam_waist_calculation',
    # Polarization
    'PolarizationState', 'JonesVector', 'JonesMatrix', 'MuellerMatrix',
    'linear_polarizer', 'quarter_wave_plate',
    # Fiber optics
    'OpticalFiber', 'StepIndexFiber', 'FiberMode',
    'calculate_modes', 'numerical_aperture',
    # Visualization
    'OpticsVisualizer', 'plot_beam_profile', 'plot_ray_diagram',
    'plot_interference_pattern', 'animate_wave_propagation',
    # Utilities
    'wavelength_to_frequency', 'frequency_to_wavelength', 'photon_energy',
    'quick_gaussian_beam', 'quick_interference_pattern', 'quick_ray_trace',
    'get_package_info'
]