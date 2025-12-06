# wave_propagation
**Module:** `Python/Elasticity/core/wave_propagation.py`
## Overview
Elastic Wave Propagation Analysis
Comprehensive elastic wave simulation including longitudinal, transverse,
and surface waves in elastic media with various boundary conditions.
## Classes
### `WaveProperties`
Properties of elastic waves.
#### Methods
##### `__post_init__(self)`
Calculate derived properties.
**Source:** [Line 27](Python/Elasticity/core/wave_propagation.py#L27)
**Class Source:** [Line 18](Python/Elasticity/core/wave_propagation.py#L18)
### `ElasticWave1D`
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
#### Methods
##### `__init__(self, length, material, density, cross_section_area)`
Initialize 1D elastic wave system.
Parameters:
length: Length of the rod (m)
material: Elastic material properties
density: Material density (kg/m³)
cross_section_area: Cross-sectional area (m²)
**Source:** [Line 55](Python/Elasticity/core/wave_propagation.py#L55)
##### `fundamental_frequency(self)`
Calculate fundamental frequency for fixed-fixed boundary conditions.
**Source:** [Line 77](Python/Elasticity/core/wave_propagation.py#L77)
##### `natural_frequencies(self, n_modes)`
Calculate natural frequencies for fixed-fixed boundary conditions.
**Source:** [Line 81](Python/Elasticity/core/wave_propagation.py#L81)
##### `mode_shapes(self, x, n_modes)`
Calculate mode shapes for fixed-fixed boundary conditions.
Parameters:
x: Position array (m)
n_modes: Number of modes to calculate
Returns:
Array of mode shapes [position, mode]
**Source:** [Line 86](Python/Elasticity/core/wave_propagation.py#L86)
##### `wave_equation_solution(self, x, t, initial_displacement, initial_velocity, boundary_conditions)`
Solve 1D wave equation with given initial and boundary conditions.
Parameters:
x: Spatial grid (m)
t: Time grid (s)
initial_displacement: Function u(x, 0)
initial_velocity: Function ∂u/∂t(x, 0)
boundary_conditions: Type of boundary conditions
Returns:
Displacement field u(x, t)
**Source:** [Line 104](Python/Elasticity/core/wave_propagation.py#L104)
##### `harmonic_wave(self, x, t, frequency, amplitude, phase, direction)`
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
**Source:** [Line 163](Python/Elasticity/core/wave_propagation.py#L163)
##### `pulse_propagation(self, x, t, pulse_function, pulse_speed, pulse_position)`
Simulate pulse propagation using method of characteristics.
Parameters:
x: Spatial grid
t: Time grid
pulse_function: Function defining pulse shape f(ξ)
pulse_speed: Pulse propagation speed
pulse_position: Initial pulse position
Returns:
Displacement field
**Source:** [Line 188](Python/Elasticity/core/wave_propagation.py#L188)
##### `reflection_coefficient(self, impedance1, impedance2)`
Calculate reflection coefficient at interface.
**Source:** [Line 216](Python/Elasticity/core/wave_propagation.py#L216)
##### `transmission_coefficient(self, impedance1, impedance2)`
Calculate transmission coefficient at interface.
**Source:** [Line 220](Python/Elasticity/core/wave_propagation.py#L220)
##### `dispersion_relation(self, frequency, geometry)`
Calculate dispersion relation for different geometries.
Parameters:
frequency: Frequency array (Hz)
geometry: 'rod', 'plate', or 'cylindrical'
Returns:
Wave number array
**Source:** [Line 224](Python/Elasticity/core/wave_propagation.py#L224)
##### `group_velocity(self, frequency, geometry)`
Calculate group velocity from dispersion relation.
**Source:** [Line 255](Python/Elasticity/core/wave_propagation.py#L255)
##### `energy_density(self, displacement, velocity)`
Calculate elastic energy density.
**Source:** [Line 265](Python/Elasticity/core/wave_propagation.py#L265)
**Class Source:** [Line 38](Python/Elasticity/core/wave_propagation.py#L38)
### `ElasticWave2D`
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
#### Methods
##### `__init__(self, domain_size, material, density)`
Initialize 2D elastic wave system.
Parameters:
domain_size: (width, height) of computational domain (m)
material: Elastic material properties
density: Material density (kg/m³)
**Source:** [Line 294](Python/Elasticity/core/wave_propagation.py#L294)
##### `green_function_2d(self, r, t, wave_type)`
Calculate Green's function for 2D elastic waves.
Parameters:
r: Distance from source (m)
t: Time (s)
wave_type: 'p' for P-waves, 's' for S-waves
Returns:
Green's function value
**Source:** [Line 319](Python/Elasticity/core/wave_propagation.py#L319)
##### `point_source_solution(self, x, y, t, source_x, source_y, source_function, wave_type)`
Solution for point source in infinite medium.
Parameters:
x, y: Spatial coordinate arrays
t: Time array
source_x, source_y: Source location
source_function: Time-dependent source function
wave_type: 'p' or 's' waves
Returns:
Displacement field [time, y, x]
**Source:** [Line 345](Python/Elasticity/core/wave_propagation.py#L345)
##### `plane_wave_solution(self, x, y, t, wave_vector, frequency, amplitude, wave_type)`
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
**Source:** [Line 388](Python/Elasticity/core/wave_propagation.py#L388)
##### `rayleigh_wave_solution(self, x, z, t, frequency, amplitude)`
Rayleigh surface wave solution.
Parameters:
x: Horizontal coordinate array
z: Depth coordinate array (z=0 at surface)
t: Time array
frequency: Wave frequency
amplitude: Wave amplitude
Returns:
Tuple of (horizontal_displacement, vertical_displacement)
**Source:** [Line 427](Python/Elasticity/core/wave_propagation.py#L427)
##### `lamb_wave_modes(self, frequency, plate_thickness, n_modes)`
Calculate Lamb wave dispersion curves for a plate.
Parameters:
frequency: Frequency array
plate_thickness: Plate thickness
n_modes: Number of modes to calculate
Returns:
Tuple of (symmetric_modes, antisymmetric_modes) wave numbers
**Source:** [Line 472](Python/Elasticity/core/wave_propagation.py#L472)
##### `wavefront_analysis(self, displacement_field, x, y, t)`
Analyze wavefront propagation characteristics.
Parameters:
displacement_field: Displacement field [time, y, x]
x, y: Spatial coordinates
t: Time array
Returns:
Dictionary with wavefront properties
**Source:** [Line 518](Python/Elasticity/core/wave_propagation.py#L518)
**Class Source:** [Line 277](Python/Elasticity/core/wave_propagation.py#L277)
### `WaveInteraction`
Analysis of wave interactions, scattering, and diffraction.
Features:
- Wave scattering by obstacles
- Diffraction around edges
- Wave interference patterns
- Attenuation and absorption
#### Methods
##### `__init__(self, material, density)`
Initialize wave interaction analyzer.
**Source:** [Line 574](Python/Elasticity/core/wave_propagation.py#L574)
##### `scattering_cross_section(self, obstacle_radius, frequency, wave_type)`
Calculate scattering cross-section for circular obstacle.
Parameters:
obstacle_radius: Radius of scattering obstacle
frequency: Wave frequency
wave_type: 'p' or 's' waves
Returns:
Scattering cross-section
**Source:** [Line 582](Python/Elasticity/core/wave_propagation.py#L582)
##### `interference_pattern(self, source1_pos, source2_pos, x, y, frequency, phase_diff)`
Calculate interference pattern from two coherent sources.
Parameters:
source1_pos, source2_pos: Source positions (x, y)
x, y: Observation coordinates
frequency: Wave frequency
phase_diff: Phase difference between sources
Returns:
Interference pattern
**Source:** [Line 615](Python/Elasticity/core/wave_propagation.py#L615)
##### `attenuation_coefficient(self, frequency, quality_factor)`
Calculate attenuation coefficient due to material damping.
Parameters:
frequency: Wave frequency
quality_factor: Material quality factor Q
Returns:
Attenuation coefficient (1/m)
**Source:** [Line 645](Python/Elasticity/core/wave_propagation.py#L645)
**Class Source:** [Line 563](Python/Elasticity/core/wave_propagation.py#L563)