# wave_optics
**Module:** `Python/Optics/wave_optics.py`
## Overview
Wave Optics Module.
This module provides comprehensive wave optics functionality including
wave propagation, diffraction, interference, and coherence analysis.
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `propagate_fresnel(aperture_function, wavelength, propagation_distance, x_aperture, x_screen)`
Propagate optical field using Fresnel diffraction.
Args:
aperture_function: Function defining aperture transmission
wavelength: Wavelength (meters)
propagation_distance: Distance to observation screen (meters)
x_aperture: Aperture coordinates (meters)
x_screen: Screen coordinates (meters)
Returns:
Complex field amplitude at screen
**Source:** [Line 348](Python/Optics/wave_optics.py#L348)
### `calculate_diffraction(aperture_type, aperture_size, wavelength, screen_distance, screen_size, num_points)`
Calculate diffraction patterns for various apertures.
Args:
aperture_type: Type of aperture ('single_slit', 'double_slit', 'circular')
aperture_size: Characteristic size (meters)
wavelength: Wavelength (meters)
screen_distance: Distance to screen (meters)
screen_size: Size of observation screen (meters)
num_points: Number of calculation points
Returns:
Dictionary with position and intensity arrays
**Source:** [Line 387](Python/Optics/wave_optics.py#L387)
### `analyze_interference(wavelength, source_separation, screen_distance, pattern_type, coherence_length, num_points)`
Analyze interference patterns.
Args:
wavelength: Wavelength (meters)
source_separation: Separation between sources (meters)
screen_distance: Distance to observation screen (meters)
pattern_type: Type of interference ('double_slit', 'young', 'michelson')
coherence_length: Coherence length for partial coherence (meters)
num_points: Number of calculation points
Returns:
Dictionary with interference analysis results
**Source:** [Line 451](Python/Optics/wave_optics.py#L451)
### `fresnel_number(aperture_radius, wavelength, distance)`
Calculate Fresnel number for circular aperture.
Args:
aperture_radius: Radius of circular aperture (meters)
wavelength: Wavelength (meters)
distance: Distance to observation point (meters)
Returns:
Fresnel number (dimensionless)
**Source:** [Line 517](Python/Optics/wave_optics.py#L517)
### `rayleigh_range(wavelength, waist_radius, medium_index)`
Calculate Rayleigh range for Gaussian beam.
Args:
wavelength: Wavelength in vacuum (meters)
waist_radius: Beam waist radius (meters)
medium_index: Refractive index
Returns:
Rayleigh range (meters)
**Source:** [Line 532](Python/Optics/wave_optics.py#L532)
### `demo_wave_optics()`
Demonstrate wave optics functionality.
**Source:** [Line 549](Python/Optics/wave_optics.py#L549)
## Classes
### `WaveParameters`
Container for wave parameters.
#### Methods
##### `from_wavelength(cls, wavelength, c)`
Create from wavelength.
**Source:** [Line 31](Python/Optics/wave_optics.py#L31)
**Class Source:** [Line 22](Python/Optics/wave_optics.py#L22)
### `WaveOptics`
Base class for wave optics calculations.
#### Methods
##### `__init__(self, wavelength, medium_index)`
Initialize wave optics calculator.
Args:
wavelength: Wavelength in vacuum (meters)
medium_index: Refractive index of medium
**Source:** [Line 44](Python/Optics/wave_optics.py#L44)
##### `validate_input(self)`
Validate input parameters.
**Source:** [Line 61](Python/Optics/wave_optics.py#L61)
**Class Source:** [Line 41](Python/Optics/wave_optics.py#L41)
### `PlaneWave`
Plane wave propagation and analysis.
#### Methods
##### `__init__(self, wavelength, amplitude, phase, direction, medium_index)`
Initialize plane wave.
Args:
wavelength: Wavelength in vacuum (meters)
amplitude: Wave amplitude
phase: Initial phase (radians)
direction: Propagation direction unit vector
medium_index: Refractive index
**Source:** [Line 75](Python/Optics/wave_optics.py#L75)
##### `field_at_point(self, r, t)`
Calculate field at a point in space and time.
Args:
r: Position vector(s) [x, y, z] (meters)
t: Time (seconds)
Returns:
Complex field amplitude
**Source:** [Line 93](Python/Optics/wave_optics.py#L93)
##### `intensity_at_point(self, r, t)`
Calculate intensity at a point.
Args:
r: Position vector(s) (meters)
t: Time (seconds)
Returns:
Intensity (W/m²)
**Source:** [Line 113](Python/Optics/wave_optics.py#L113)
##### `propagate_distance(self, distance)`
Propagate plane wave by a distance.
Args:
distance: Propagation distance (meters)
Returns:
New PlaneWave object at propagated position
**Source:** [Line 126](Python/Optics/wave_optics.py#L126)
**Class Source:** [Line 72](Python/Optics/wave_optics.py#L72)
### `SphericalWave`
Spherical wave propagation and analysis.
#### Methods
##### `__init__(self, wavelength, source_power, source_position, medium_index)`
Initialize spherical wave.
Args:
wavelength: Wavelength in vacuum (meters)
source_power: Source power (watts)
source_position: Source position [x, y, z] (meters)
medium_index: Refractive index
**Source:** [Line 147](Python/Optics/wave_optics.py#L147)
##### `field_at_point(self, r, t)`
Calculate field at a point.
Args:
r: Position vector(s) (meters)
t: Time (seconds)
Returns:
Complex field amplitude
**Source:** [Line 161](Python/Optics/wave_optics.py#L161)
##### `intensity_at_point(self, r, t)`
Calculate intensity at a point.
Args:
r: Position vector(s) (meters)
t: Time (seconds)
Returns:
Intensity (W/m²)
**Source:** [Line 188](Python/Optics/wave_optics.py#L188)
**Class Source:** [Line 144](Python/Optics/wave_optics.py#L144)
### `GaussianBeam`
Gaussian beam propagation using ABCD matrix method.
#### Methods
##### `__init__(self, wavelength, waist_radius, waist_position, medium_index, power)`
Initialize Gaussian beam.
Args:
wavelength: Wavelength in vacuum (meters)
waist_radius: Beam waist radius (meters)
waist_position: Z position of waist (meters)
medium_index: Refractive index
power: Beam power (watts)
**Source:** [Line 205](Python/Optics/wave_optics.py#L205)
##### `beam_radius(self, z)`
Calculate beam radius at position z.
Args:
z: Axial position (meters)
Returns:
Beam radius (meters)
**Source:** [Line 226](Python/Optics/wave_optics.py#L226)
##### `radius_of_curvature(self, z)`
Calculate radius of curvature at position z.
Args:
z: Axial position (meters)
Returns:
Radius of curvature (meters), inf at waist
**Source:** [Line 238](Python/Optics/wave_optics.py#L238)
##### `gouy_phase(self, z)`
Calculate Gouy phase at position z.
Args:
z: Axial position (meters)
Returns:
Gouy phase (radians)
**Source:** [Line 252](Python/Optics/wave_optics.py#L252)
##### `field_profile(self, x, y, z)`
Calculate transverse field profile at position z.
Args:
x, y: Transverse coordinates (meters)
z: Axial position (meters)
Returns:
Complex field amplitude
**Source:** [Line 264](Python/Optics/wave_optics.py#L264)
##### `intensity_profile(self, x, y, z)`
Calculate intensity profile at position z.
Args:
x, y: Transverse coordinates (meters)
z: Axial position (meters)
Returns:
Intensity distribution (W/m²)
**Source:** [Line 293](Python/Optics/wave_optics.py#L293)
##### `propagate_through_lens(self, focal_length, distance_to_lens)`
Propagate Gaussian beam through a thin lens.
Args:
focal_length: Lens focal length (meters)
distance_to_lens: Distance from current waist to lens (meters)
Returns:
New GaussianBeam after lens
**Source:** [Line 306](Python/Optics/wave_optics.py#L306)
**Class Source:** [Line 202](Python/Optics/wave_optics.py#L202)