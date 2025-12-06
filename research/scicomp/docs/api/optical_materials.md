# optical_materials
**Module:** `Python/Optics/optical_materials.py`
## Overview
Optical Materials Module.
This module provides comprehensive optical material properties including
refractive index models, dispersion analysis, and material databases.
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `create_material_database()`
Create database of common optical materials.
**Source:** [Line 278](Python/Optics/optical_materials.py#L278)
### `calculate_refractive_index(material, wavelength, temperature)`
Calculate refractive index for material.
Args:
material: Material name or OpticalMaterial object
wavelength: Wavelength (meters)
temperature: Temperature (Kelvin)
Returns:
Refractive index
**Source:** [Line 343](Python/Optics/optical_materials.py#L343)
### `dispersion_analysis(material, wavelength_range)`
Perform comprehensive dispersion analysis.
Args:
material: OpticalMaterial object
wavelength_range: Wavelength range for analysis (meters)
Returns:
Dictionary with dispersion analysis results
**Source:** [Line 376](Python/Optics/optical_materials.py#L376)
### `absorption_coefficient(material, wavelength)`
Calculate absorption coefficient for material.
Args:
material: Material name
wavelength: Wavelength (meters)
Returns:
Absorption coefficient (1/m)
**Source:** [Line 428](Python/Optics/optical_materials.py#L428)
### `group_velocity_dispersion(material, wavelength)`
Calculate group velocity dispersion.
Args:
material: Material name or OpticalMaterial object
wavelength: Wavelength (meters)
Returns:
GVD in ps²/km
**Source:** [Line 453](Python/Optics/optical_materials.py#L453)
### `demo_optical_materials()`
Demonstrate optical materials functionality.
**Source:** [Line 477](Python/Optics/optical_materials.py#L477)
## Classes
### `MaterialProperties`
Container for optical material properties.
**Class Source:** [Line 20](Python/Optics/optical_materials.py#L20)
### `DispersionModel`
Abstract base class for dispersion models.
#### Methods
##### `refractive_index(self, wavelength, temperature)`
Calculate refractive index at given wavelength and temperature.
Args:
wavelength: Wavelength in meters
temperature: Temperature in Kelvin
Returns:
Refractive index
**Source:** [Line 37](Python/Optics/optical_materials.py#L37)
##### `group_index(self, wavelength)`
Calculate group index.
Args:
wavelength: Wavelength in meters
Returns:
Group index
**Source:** [Line 50](Python/Optics/optical_materials.py#L50)
**Class Source:** [Line 33](Python/Optics/optical_materials.py#L33)
### `Sellmeier`
Sellmeier dispersion model.
#### Methods
##### `__init__(self, B1, B2, B3, C1, C2, C3)`
Initialize Sellmeier model.
Args:
B1, B2, B3: Sellmeier B coefficients
C1, C2, C3: Sellmeier C coefficients (μm²)
**Source:** [Line 65](Python/Optics/optical_materials.py#L65)
##### `refractive_index(self, wavelength, temperature)`
Calculate refractive index using Sellmeier equation.
n²(λ) = 1 + B₁λ²/(λ² - C₁) + B₂λ²/(λ² - C₂) + B₃λ²/(λ² - C₃)
**Source:** [Line 76](Python/Optics/optical_materials.py#L76)
##### `group_index(self, wavelength)`
Calculate group index ng = n - λ(dn/dλ).
**Source:** [Line 91](Python/Optics/optical_materials.py#L91)
**Class Source:** [Line 62](Python/Optics/optical_materials.py#L62)
### `Cauchy`
Cauchy dispersion model.
#### Methods
##### `__init__(self, A, B, C)`
Initialize Cauchy model.
Args:
A, B, C: Cauchy coefficients
**Source:** [Line 106](Python/Optics/optical_materials.py#L106)
##### `refractive_index(self, wavelength, temperature)`
Calculate refractive index using Cauchy equation.
n(λ) = A + B/λ² + C/λ⁴
**Source:** [Line 114](Python/Optics/optical_materials.py#L114)
##### `group_index(self, wavelength)`
Calculate group index.
**Source:** [Line 122](Python/Optics/optical_materials.py#L122)
**Class Source:** [Line 103](Python/Optics/optical_materials.py#L103)
### `LorentzDrude`
Lorentz-Drude dispersion model for metals.
#### Methods
##### `__init__(self, epsilon_inf, oscillators)`
Initialize Lorentz-Drude model.
Args:
epsilon_inf: High-frequency dielectric constant
oscillators: List of oscillator parameters
**Source:** [Line 137](Python/Optics/optical_materials.py#L137)
##### `dielectric_function(self, wavelength)`
Calculate complex dielectric function.
**Source:** [Line 147](Python/Optics/optical_materials.py#L147)
##### `refractive_index(self, wavelength, temperature)`
Calculate complex refractive index.
**Source:** [Line 163](Python/Optics/optical_materials.py#L163)
##### `group_index(self, wavelength)`
Calculate group index (real part only).
**Source:** [Line 168](Python/Optics/optical_materials.py#L168)
**Class Source:** [Line 134](Python/Optics/optical_materials.py#L134)
### `OpticalMaterial`
Comprehensive optical material class.
#### Methods
##### `__init__(self, name, dispersion_model, wavelength_range, properties)`
Initialize optical material.
Args:
name: Material name
dispersion_model: Dispersion model object
wavelength_range: Valid wavelength range (meters)
properties: Additional material properties
**Source:** [Line 177](Python/Optics/optical_materials.py#L177)
##### `refractive_index(self, wavelength, temperature)`
Get refractive index at wavelength(s).
**Source:** [Line 193](Python/Optics/optical_materials.py#L193)
##### `group_index(self, wavelength)`
Get group index at wavelength(s).
**Source:** [Line 199](Python/Optics/optical_materials.py#L199)
##### `group_velocity_dispersion(self, wavelength)`
Calculate group velocity dispersion (GVD).
Args:
wavelength: Wavelength (meters)
Returns:
GVD in ps²/km
**Source:** [Line 204](Python/Optics/optical_materials.py#L204)
##### `chromatic_dispersion(self, wavelength)`
Calculate chromatic dispersion parameter D.
Args:
wavelength: Wavelength (meters)
Returns:
Dispersion parameter in ps/(nm⋅km)
**Source:** [Line 229](Python/Optics/optical_materials.py#L229)
##### `plot_dispersion(self, wavelength_range, num_points)`
Plot dispersion curve.
**Source:** [Line 242](Python/Optics/optical_materials.py#L242)
**Class Source:** [Line 174](Python/Optics/optical_materials.py#L174)