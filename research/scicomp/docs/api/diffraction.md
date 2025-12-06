# diffraction
**Module:** `Python/Crystallography/core/diffraction.py`
## Overview
X-ray Diffraction Analysis
Comprehensive X-ray diffraction pattern simulation and analysis tools.
Includes structure factor calculations, intensity modeling, and
powder diffraction pattern generation.
## Classes
### `AtomicScatteringFactor`
Atomic scattering factor parameters.
#### Methods
##### `f(self, sin_theta_over_lambda)`
Calculate atomic scattering factor.
Parameters:
sin_theta_over_lambda: sin(θ)/λ in Å⁻¹
Returns:
Complex scattering factor
**Source:** [Line 29](Python/Crystallography/core/diffraction.py#L29)
**Class Source:** [Line 20](Python/Crystallography/core/diffraction.py#L20)
### `ScatteringFactorDatabase`
Database of atomic scattering factors.
#### Methods
##### `__init__(self)`
Initialize with common elements.
**Source:** [Line 53](Python/Crystallography/core/diffraction.py#L53)
##### `get_scattering_factor(self, element)`
Get scattering factor parameters for element.
**Source:** [Line 76](Python/Crystallography/core/diffraction.py#L76)
**Class Source:** [Line 50](Python/Crystallography/core/diffraction.py#L50)
### `ReflectionData`
X-ray reflection data.
#### Methods
##### `structure_factor_magnitude(self)`
Magnitude of structure factor.
**Source:** [Line 95](Python/Crystallography/core/diffraction.py#L95)
##### `structure_factor_phase(self)`
Phase of structure factor in degrees.
**Source:** [Line 100](Python/Crystallography/core/diffraction.py#L100)
**Class Source:** [Line 85](Python/Crystallography/core/diffraction.py#L85)
### `StructureFactor`
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
#### Methods
##### `__init__(self, crystal, space_group)`
Initialize structure factor calculator.
Parameters:
crystal: Crystal structure
space_group: Space group (optional)
**Source:** [Line 121](Python/Crystallography/core/diffraction.py#L121)
##### `calculate(self, h, k, l, wavelength)`
Calculate structure factor F_hkl.
Parameters:
h, k, l: Miller indices
wavelength: X-ray wavelength in Å
Returns:
Complex structure factor
**Source:** [Line 133](Python/Crystallography/core/diffraction.py#L133)
##### `calculate_intensity(self, h, k, l, wavelength)`
Calculate diffraction intensity |F_hkl|².
Parameters:
h, k, l: Miller indices
wavelength: X-ray wavelength in Å
Returns:
Relative intensity
**Source:** [Line 170](Python/Crystallography/core/diffraction.py#L170)
##### `is_systematic_absence(self, h, k, l, tolerance)`
Check if reflection is systematically absent.
Parameters:
h, k, l: Miller indices
tolerance: Tolerance for zero intensity
Returns:
True if systematically absent
**Source:** [Line 184](Python/Crystallography/core/diffraction.py#L184)
##### `structure_factor_derivatives(self, h, k, l, parameter, atom_idx)`
Calculate derivative of structure factor with respect to structural parameter.
Parameters:
h, k, l: Miller indices
parameter: Parameter name ('x', 'y', 'z', 'occupancy', 'thermal_factor')
atom_idx: Index of atom
Returns:
Derivative of structure factor
**Source:** [Line 198](Python/Crystallography/core/diffraction.py#L198)
**Class Source:** [Line 105](Python/Crystallography/core/diffraction.py#L105)
### `DiffractionPattern`
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
#### Methods
##### `__init__(self, crystal, space_group)`
Initialize diffraction pattern simulator.
Parameters:
crystal: Crystal structure
space_group: Space group (optional)
**Source:** [Line 261](Python/Crystallography/core/diffraction.py#L261)
##### `simulate_powder_pattern(self, wavelength, two_theta_range, step_size, peak_width, background, max_hkl)`
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
**Source:** [Line 273](Python/Crystallography/core/diffraction.py#L273)
##### `_calculate_reflections(self, wavelength, max_two_theta, max_hkl)`
Calculate all allowed reflections.
**Source:** [Line 312](Python/Crystallography/core/diffraction.py#L312)
##### `_calculate_multiplicity(self, h, k, l)`
Calculate reflection multiplicity (simplified).
**Source:** [Line 378](Python/Crystallography/core/diffraction.py#L378)
##### `_apply_corrections(self, intensity, theta_deg, multiplicity)`
Apply geometric and physical intensity corrections.
**Source:** [Line 403](Python/Crystallography/core/diffraction.py#L403)
##### `_gaussian_profile(self, two_theta, peak_center, fwhm)`
Generate Gaussian peak profile.
**Source:** [Line 419](Python/Crystallography/core/diffraction.py#L419)
##### `_lorentzian_profile(self, two_theta, peak_center, fwhm)`
Generate Lorentzian peak profile.
**Source:** [Line 425](Python/Crystallography/core/diffraction.py#L425)
##### `_voigt_profile(self, two_theta, peak_center, fwhm_gaussian, fwhm_lorentzian)`
Generate Voigt peak profile (convolution of Gaussian and Lorentzian).
**Source:** [Line 431](Python/Crystallography/core/diffraction.py#L431)
##### `find_peaks(self, two_theta, intensity, prominence, min_distance)`
Find peaks in diffraction pattern.
Parameters:
two_theta: 2θ array
intensity: Intensity array
prominence: Minimum peak prominence
min_distance: Minimum distance between peaks
Returns:
List of peak information dictionaries
**Source:** [Line 444](Python/Crystallography/core/diffraction.py#L444)
##### `index_powder_pattern(self, peaks, crystal_system)`
Index powder diffraction pattern (simplified).
Parameters:
peaks: List of peak dictionaries from find_peaks()
crystal_system: Crystal system for indexing
Returns:
List of possible Miller indices for each peak
**Source:** [Line 507](Python/Crystallography/core/diffraction.py#L507)
**Class Source:** [Line 244](Python/Crystallography/core/diffraction.py#L244)