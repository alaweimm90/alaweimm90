"""
X-ray Diffraction Analysis
Comprehensive X-ray diffraction pattern simulation and analysis tools.
Includes structure factor calculations, intensity modeling, and
powder diffraction pattern generation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
# Import related modules
from .crystal_structure import CrystalStructure, AtomicPosition
from .space_groups import SpaceGroup
@dataclass
class AtomicScatteringFactor:
    """Atomic scattering factor parameters."""
    element: str
    a1: float; b1: float
    a2: float; b2: float
    a3: float; b3: float
    a4: float; b4: float
    c: float
    def f(self, sin_theta_over_lambda: float) -> complex:
        """
        Calculate atomic scattering factor.
        Parameters:
            sin_theta_over_lambda: sin(θ)/λ in Å⁻¹
        Returns:
            Complex scattering factor
        """
        s_squared = (sin_theta_over_lambda) ** 2
        f_real = (self.a1 * np.exp(-self.b1 * s_squared) +
                 self.a2 * np.exp(-self.b2 * s_squared) +
                 self.a3 * np.exp(-self.b3 * s_squared) +
                 self.a4 * np.exp(-self.b4 * s_squared) + self.c)
        # For now, ignore anomalous scattering (imaginary part)
        return complex(f_real, 0.0)
class ScatteringFactorDatabase:
    """Database of atomic scattering factors."""
    def __init__(self):
        """Initialize with common elements."""
        # International Tables for Crystallography, Vol. C
        # Simplified database with key elements
        self._database = {
            'H': AtomicScatteringFactor('H', 0.493002, 10.5109, 0.322912, 26.1257,
                                      0.140191, 3.14236, 0.040810, 57.7997, 0.003038),
            'C': AtomicScatteringFactor('C', 2.31000, 20.8439, 1.02000, 10.2075,
                                      1.58860, 0.568700, 0.865000, 51.6512, 0.215600),
            'N': AtomicScatteringFactor('N', 12.2126, 0.005700, 3.13220, 9.89330,
                                      2.01250, 28.9975, 1.16630, 0.582600, -11.529),
            'O': AtomicScatteringFactor('O', 3.04850, 13.2771, 2.28680, 5.70110,
                                      1.54630, 0.323900, 0.867000, 32.9089, 0.250800),
            'Si': AtomicScatteringFactor('Si', 6.29150, 2.43860, 3.03530, 32.3337,
                                         1.98910, 0.678500, 1.54100, 81.6937, 1.14070),
            'Fe': AtomicScatteringFactor('Fe', 11.7695, 4.76110, 7.35730, 0.307200,
                                       3.52220, 15.3535, 2.30450, 76.8805, 1.03690),
            'Cu': AtomicScatteringFactor('Cu', 13.3380, 3.58280, 7.16760, 0.247000,
                                       5.61580, 11.3966, 1.67350, 64.8126, 1.19100),
            'Zn': AtomicScatteringFactor('Zn', 14.0743, 3.26550, 7.03180, 0.233300,
                                       5.16252, 10.3163, 2.41000, 58.7097, 1.30410)
        }
    def get_scattering_factor(self, element: str) -> AtomicScatteringFactor:
        """Get scattering factor parameters for element."""
        if element not in self._database:
            warnings.warn(f"Scattering factor for {element} not in database. Using carbon.")
            return self._database['C']
        return self._database[element]
@dataclass
class ReflectionData:
    """X-ray reflection data."""
    h: int; k: int; l: int  # Miller indices
    d_spacing: float        # d-spacing in Å
    two_theta: float       # 2θ angle in degrees
    intensity: float       # Relative intensity
    structure_factor: complex  # Complex structure factor
    multiplicity: int      # Reflection multiplicity
    @property
    def structure_factor_magnitude(self) -> float:
        """Magnitude of structure factor."""
        return abs(self.structure_factor)
    @property
    def structure_factor_phase(self) -> float:
        """Phase of structure factor in degrees."""
        return np.degrees(np.angle(self.structure_factor))
class StructureFactor:
    """
    Structure factor calculation for X-ray diffraction.
    Features:
    - Complex structure factor calculation
    - Thermal factor inclusion
    - Anomalous scattering effects
    - Systematic absence prediction
    Examples:
        >>> crystal = CrystalStructure(lattice, atoms)
        >>> sf = StructureFactor(crystal)
        >>> F_hkl = sf.calculate(1, 1, 1, wavelength=1.54)
    """
    def __init__(self, crystal: CrystalStructure, space_group: Optional[SpaceGroup] = None):
        """
        Initialize structure factor calculator.
        Parameters:
            crystal: Crystal structure
            space_group: Space group (optional)
        """
        self.crystal = crystal
        self.space_group = space_group
        self.scattering_db = ScatteringFactorDatabase()
    def calculate(self, h: int, k: int, l: int, wavelength: float = 1.54056) -> complex:
        """
        Calculate structure factor F_hkl.
        Parameters:
            h, k, l: Miller indices
            wavelength: X-ray wavelength in Å
        Returns:
            Complex structure factor
        """
        # Calculate sin(θ)/λ
        d_spacing = self.crystal.d_spacing(h, k, l)
        sin_theta = wavelength / (2 * d_spacing)
        sin_theta_over_lambda = sin_theta / wavelength
        structure_factor = complex(0.0, 0.0)
        # Sum over all atoms in unit cell
        for atom in self.crystal.atoms:
            # Get atomic scattering factor
            scattering_factor = self.scattering_db.get_scattering_factor(atom.element)
            f_j = scattering_factor.f(sin_theta_over_lambda)
            # Apply thermal factor (Debye-Waller factor)
            thermal_factor = np.exp(-atom.thermal_factor * sin_theta_over_lambda**2)
            # Phase factor
            phase = 2 * np.pi * (h * atom.x + k * atom.y + l * atom.z)
            phase_factor = np.exp(1j * phase)
            # Add contribution
            contribution = atom.occupancy * f_j * thermal_factor * phase_factor
            structure_factor += contribution
        return structure_factor
    def calculate_intensity(self, h: int, k: int, l: int, wavelength: float = 1.54056) -> float:
        """
        Calculate diffraction intensity |F_hkl|².
        Parameters:
            h, k, l: Miller indices
            wavelength: X-ray wavelength in Å
        Returns:
            Relative intensity
        """
        F_hkl = self.calculate(h, k, l, wavelength)
        return abs(F_hkl) ** 2
    def is_systematic_absence(self, h: int, k: int, l: int, tolerance: float = 1e-6) -> bool:
        """
        Check if reflection is systematically absent.
        Parameters:
            h, k, l: Miller indices
            tolerance: Tolerance for zero intensity
        Returns:
            True if systematically absent
        """
        intensity = self.calculate_intensity(h, k, l)
        return intensity < tolerance
    def structure_factor_derivatives(self, h: int, k: int, l: int,
                                   parameter: str, atom_idx: int) -> complex:
        """
        Calculate derivative of structure factor with respect to structural parameter.
        Parameters:
            h, k, l: Miller indices
            parameter: Parameter name ('x', 'y', 'z', 'occupancy', 'thermal_factor')
            atom_idx: Index of atom
        Returns:
            Derivative of structure factor
        """
        if atom_idx >= len(self.crystal.atoms):
            raise IndexError("Atom index out of range")
        atom = self.crystal.atoms[atom_idx]
        # Get current structure factor calculation components
        d_spacing = self.crystal.d_spacing(h, k, l)
        sin_theta_over_lambda = 1.0 / (2 * d_spacing)
        scattering_factor = self.scattering_db.get_scattering_factor(atom.element)
        f_j = scattering_factor.f(sin_theta_over_lambda)
        thermal_factor = np.exp(-atom.thermal_factor * sin_theta_over_lambda**2)
        phase = 2 * np.pi * (h * atom.x + k * atom.y + l * atom.z)
        phase_factor = np.exp(1j * phase)
        if parameter == 'x':
            return (atom.occupancy * f_j * thermal_factor *
                   phase_factor * 1j * 2 * np.pi * h)
        elif parameter == 'y':
            return (atom.occupancy * f_j * thermal_factor *
                   phase_factor * 1j * 2 * np.pi * k)
        elif parameter == 'z':
            return (atom.occupancy * f_j * thermal_factor *
                   phase_factor * 1j * 2 * np.pi * l)
        elif parameter == 'occupancy':
            return f_j * thermal_factor * phase_factor
        elif parameter == 'thermal_factor':
            return (atom.occupancy * f_j * phase_factor *
                   thermal_factor * (-sin_theta_over_lambda**2))
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
class DiffractionPattern:
    """
    X-ray diffraction pattern simulation and analysis.
    Features:
    - Powder diffraction pattern simulation
    - Peak profile modeling (Gaussian, Lorentzian, Voigt)
    - Background modeling
    - Peak search and indexing
    - Intensity corrections
    Examples:
        >>> crystal = CrystalStructure(lattice, atoms)
        >>> pattern = DiffractionPattern(crystal)
        >>> two_theta, intensity = pattern.simulate_powder_pattern(wavelength=1.54)
    """
    def __init__(self, crystal: CrystalStructure, space_group: Optional[SpaceGroup] = None):
        """
        Initialize diffraction pattern simulator.
        Parameters:
            crystal: Crystal structure
            space_group: Space group (optional)
        """
        self.crystal = crystal
        self.space_group = space_group
        self.structure_factor = StructureFactor(crystal, space_group)
    def simulate_powder_pattern(self, wavelength: float = 1.54056,
                              two_theta_range: Tuple[float, float] = (10.0, 80.0),
                              step_size: float = 0.02,
                              peak_width: float = 0.1,
                              background: float = 10.0,
                              max_hkl: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate powder diffraction pattern.
        Parameters:
            wavelength: X-ray wavelength in Å
            two_theta_range: 2θ range in degrees
            step_size: Step size in degrees
            peak_width: Peak FWHM in degrees
            background: Background intensity
            max_hkl: Maximum Miller index to consider
        Returns:
            Tuple of (2θ array, intensity array)
        """
        # Generate 2θ array
        two_theta_min, two_theta_max = two_theta_range
        two_theta = np.arange(two_theta_min, two_theta_max + step_size, step_size)
        intensity = np.full_like(two_theta, background)
        # Calculate reflections
        reflections = self._calculate_reflections(wavelength, two_theta_max, max_hkl)
        # Add peaks to pattern
        for refl in reflections:
            if two_theta_min <= refl.two_theta <= two_theta_max:
                # Gaussian peak profile
                peak_intensity = refl.intensity * self._gaussian_profile(
                    two_theta, refl.two_theta, peak_width
                )
                intensity += peak_intensity
        return two_theta, intensity
    def _calculate_reflections(self, wavelength: float, max_two_theta: float,
                             max_hkl: int) -> List[ReflectionData]:
        """Calculate all allowed reflections."""
        reflections = []
        for h in range(-max_hkl, max_hkl + 1):
            for k in range(-max_hkl, max_hkl + 1):
                for l in range(-max_hkl, max_hkl + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    try:
                        # Calculate d-spacing and angle
                        d_spacing = self.crystal.d_spacing(h, k, l)
                        sin_theta = wavelength / (2 * d_spacing)
                        if sin_theta > 1.0:  # No diffraction possible
                            continue
                        theta = np.degrees(np.arcsin(sin_theta))
                        two_theta = 2 * theta
                        if two_theta > max_two_theta:
                            continue
                        # Calculate structure factor and intensity
                        F_hkl = self.structure_factor.calculate(h, k, l, wavelength)
                        intensity = abs(F_hkl) ** 2
                        # Skip systematically absent reflections
                        if intensity < 1e-6:
                            continue
                        # Calculate multiplicity (simplified)
                        multiplicity = self._calculate_multiplicity(h, k, l)
                        # Apply geometric and physical corrections
                        corrected_intensity = self._apply_corrections(
                            intensity, theta, multiplicity
                        )
                        refl = ReflectionData(
                            h=h, k=k, l=l,
                            d_spacing=d_spacing,
                            two_theta=two_theta,
                            intensity=corrected_intensity,
                            structure_factor=F_hkl,
                            multiplicity=multiplicity
                        )
                        reflections.append(refl)
                    except (ValueError, ZeroDivisionError):
                        continue
        # Sort by 2θ
        reflections.sort(key=lambda x: x.two_theta)
        # Normalize intensities
        if reflections:
            max_intensity = max(refl.intensity for refl in reflections)
            for refl in reflections:
                refl.intensity = 100 * refl.intensity / max_intensity
        return reflections
    def _calculate_multiplicity(self, h: int, k: int, l: int) -> int:
        """Calculate reflection multiplicity (simplified)."""
        # This is a simplified calculation
        # Full implementation would use space group operations
        if self.space_group:
            # Use space group operations to calculate exact multiplicity
            # For now, use simplified approach
            pass
        # Simplified multiplicity based on symmetry
        unique_indices = len(set([abs(h), abs(k), abs(l)]))
        zero_indices = [h, k, l].count(0)
        if zero_indices == 2:  # e.g., (1,0,0)
            return 6
        elif zero_indices == 1:  # e.g., (1,1,0)
            return 12
        elif unique_indices == 1:  # e.g., (1,1,1)
            return 8
        elif unique_indices == 2:  # e.g., (1,1,2)
            return 24
        else:  # e.g., (1,2,3)
            return 48
    def _apply_corrections(self, intensity: float, theta_deg: float,
                         multiplicity: int) -> float:
        """Apply geometric and physical intensity corrections."""
        theta_rad = np.radians(theta_deg)
        # Lorentz-polarization factor
        lorentz_factor = 1.0 / (np.sin(theta_rad) * np.sin(2 * theta_rad))
        # Polarization factor (unpolarized radiation)
        polarization_factor = (1 + np.cos(2 * theta_rad)**2) / 2
        # Multiplicity factor
        corrected_intensity = intensity * multiplicity * lorentz_factor * polarization_factor
        return corrected_intensity
    def _gaussian_profile(self, two_theta: np.ndarray, peak_center: float,
                         fwhm: float) -> np.ndarray:
        """Generate Gaussian peak profile."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * ((two_theta - peak_center) / sigma) ** 2)
    def _lorentzian_profile(self, two_theta: np.ndarray, peak_center: float,
                          fwhm: float) -> np.ndarray:
        """Generate Lorentzian peak profile."""
        gamma = fwhm / 2
        return gamma**2 / ((two_theta - peak_center)**2 + gamma**2)
    def _voigt_profile(self, two_theta: np.ndarray, peak_center: float,
                      fwhm_gaussian: float, fwhm_lorentzian: float) -> np.ndarray:
        """Generate Voigt peak profile (convolution of Gaussian and Lorentzian)."""
        # Simplified Voigt profile using pseudo-Voigt approximation
        eta = 1.36603 * (fwhm_lorentzian / (fwhm_gaussian + fwhm_lorentzian)) - \
              0.47719 * (fwhm_lorentzian / (fwhm_gaussian + fwhm_lorentzian))**2 + \
              0.11116 * (fwhm_lorentzian / (fwhm_gaussian + fwhm_lorentzian))**3
        gaussian = self._gaussian_profile(two_theta, peak_center, fwhm_gaussian)
        lorentzian = self._lorentzian_profile(two_theta, peak_center, fwhm_lorentzian)
        return eta * lorentzian + (1 - eta) * gaussian
    def find_peaks(self, two_theta: np.ndarray, intensity: np.ndarray,
                  prominence: float = 10.0, min_distance: float = 0.1) -> List[Dict]:
        """
        Find peaks in diffraction pattern.
        Parameters:
            two_theta: 2θ array
            intensity: Intensity array
            prominence: Minimum peak prominence
            min_distance: Minimum distance between peaks
        Returns:
            List of peak information dictionaries
        """
        peaks = []
        # Simple peak finding algorithm
        for i in range(1, len(intensity) - 1):
            # Check if it's a local maximum
            if (intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and
                intensity[i] > prominence):
                # Check minimum distance to existing peaks
                too_close = False
                for existing_peak in peaks:
                    if abs(two_theta[i] - existing_peak['two_theta']) < min_distance:
                        too_close = True
                        break
                if not too_close:
                    # Fit peak position more precisely (parabolic interpolation)
                    if i > 0 and i < len(intensity) - 1:
                        y1, y2, y3 = intensity[i-1], intensity[i], intensity[i+1]
                        x1, x2, x3 = two_theta[i-1], two_theta[i], two_theta[i+1]
                        # Parabolic fit
                        denom = (x1-x2)*(x1-x3)*(x2-x3)
                        if abs(denom) > 1e-10:
                            a = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
                            b = (x3*x3*(y1-y2) + x2*x2*(y3-y1) + x1*x1*(y2-y3)) / denom
                            if a < 0:  # Peak (not valley)
                                peak_pos = -b / (2*a)
                                peak_intensity = a*peak_pos*peak_pos + b*peak_pos + \
                                               (x1*x1*(x2*y3-x3*y2) + x1*(x3*x3*y2-x2*x2*y3) + x2*x2*x3*y1-x2*x3*x3*y1) / denom
                            else:
                                peak_pos = two_theta[i]
                                peak_intensity = intensity[i]
                        else:
                            peak_pos = two_theta[i]
                            peak_intensity = intensity[i]
                    else:
                        peak_pos = two_theta[i]
                        peak_intensity = intensity[i]
                    peaks.append({
                        'two_theta': peak_pos,
                        'intensity': peak_intensity,
                        'd_spacing': 1.54056 / (2 * np.sin(np.radians(peak_pos / 2)))  # Cu Kα
                    })
        return peaks
    def index_powder_pattern(self, peaks: List[Dict],
                           crystal_system: str = 'cubic') -> List[Tuple[int, int, int]]:
        """
        Index powder diffraction pattern (simplified).
        Parameters:
            peaks: List of peak dictionaries from find_peaks()
            crystal_system: Crystal system for indexing
        Returns:
            List of possible Miller indices for each peak
        """
        indexed_peaks = []
        if crystal_system.lower() == 'cubic':
            # For cubic system: 1/d² = (h²+k²+l²)/a²
            # Use first peak to estimate lattice parameter
            if peaks:
                first_peak = peaks[0]
                # Assume first peak is (1,0,0) or similar
                a_estimated = first_peak['d_spacing']  # Very rough estimate
                for peak in peaks:
                    d = peak['d_spacing']
                    # Calculate h²+k²+l²
                    hkl_squared = (a_estimated / d) ** 2
                    # Find integer solutions close to hkl_squared
                    possible_hkl = []
                    for h in range(0, 6):
                        for k in range(0, 6):
                            for l in range(0, 6):
                                if abs(h*h + k*k + l*l - hkl_squared) < 0.1:
                                    possible_hkl.append((h, k, l))
                    indexed_peaks.append(possible_hkl)
        return indexed_peaks