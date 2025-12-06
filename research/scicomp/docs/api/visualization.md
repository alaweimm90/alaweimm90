# visualization
**Module:** `Python/Optics/visualization.py`
## Overview
Visualization Module for Optics.
This module provides Berkeley-themed visualization tools for optical systems,
beam profiles, ray diagrams, and interference patterns.
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `plot_beam_profile(x, y, z, intensity)`
Plot beam profile using Berkeley styling.
**Source:** [Line 439](Python/Optics/visualization.py#L439)
### `plot_ray_diagram(rays, surfaces)`
Plot ray diagram using Berkeley styling.
**Source:** [Line 446](Python/Optics/visualization.py#L446)
### `plot_interference_pattern(x, intensity, pattern_info)`
Plot interference pattern using Berkeley styling.
**Source:** [Line 452](Python/Optics/visualization.py#L452)
### `animate_wave_propagation(x, z, field_function, wavelength)`
Create wave propagation animation using Berkeley styling.
**Source:** [Line 459](Python/Optics/visualization.py#L459)
### `create_optical_system_diagram(elements)`
Create optical system diagram using Berkeley styling.
**Source:** [Line 467](Python/Optics/visualization.py#L467)
### `demo_visualization()`
Demonstrate visualization capabilities.
**Source:** [Line 474](Python/Optics/visualization.py#L474)
## Classes
### `OpticsVisualizer`
Berkeley-themed optical visualization class.
#### Methods
##### `__init__(self)`
Initialize visualizer with Berkeley styling.
**Source:** [Line 22](Python/Optics/visualization.py#L22)
##### `setup_berkeley_style(self)`
Setup Berkeley visual styling for matplotlib.
**Source:** [Line 43](Python/Optics/visualization.py#L43)
##### `plot_beam_profile(self, x, y, z, intensity, title, figsize)`
Plot 2D beam intensity profile.
Args:
x, y: Coordinate arrays (meters)
z: Axial position (meters)
intensity: 2D intensity array (W/m²)
title: Plot title
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 62](Python/Optics/visualization.py#L62)
##### `plot_ray_diagram(self, rays, surfaces, title, figsize)`
Plot ray diagram with optical elements.
Args:
rays: List of ray objects or ray trace results
surfaces: List of optical surfaces
title: Plot title
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 109](Python/Optics/visualization.py#L109)
##### `_draw_optical_element(self, ax, surface)`
Draw optical element on ray diagram.
**Source:** [Line 161](Python/Optics/visualization.py#L161)
##### `plot_interference_pattern(self, x, intensity, pattern_info, title, figsize)`
Plot interference pattern.
Args:
x: Position array (meters)
intensity: Intensity array
pattern_info: Additional pattern information
title: Plot title
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 188](Python/Optics/visualization.py#L188)
##### `plot_diffraction_pattern(self, x, intensity, aperture_info, title, figsize)`
Plot diffraction pattern with aperture information.
Args:
x: Position array (meters)
intensity: Intensity array
aperture_info: Aperture parameters
title: Plot title
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 234](Python/Optics/visualization.py#L234)
##### `animate_wave_propagation(self, x, z, field_function, wavelength, title, save_path)`
Create animation of wave propagation.
Args:
x: Spatial coordinates (meters)
z: Propagation distances (meters)
field_function: Function(x, z, t) returning field amplitude
wavelength: Wavelength (meters)
title: Animation title
save_path: Path to save animation (optional)
Returns:
Animation object
**Source:** [Line 286](Python/Optics/visualization.py#L286)
##### `create_optical_system_diagram(self, elements, title, figsize)`
Create comprehensive optical system diagram.
Args:
elements: List of optical element dictionaries
title: Diagram title
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 354](Python/Optics/visualization.py#L354)
##### `_draw_system_element(self, ax, element, index)`
Draw individual optical element in system diagram.
**Source:** [Line 389](Python/Optics/visualization.py#L389)
**Class Source:** [Line 19](Python/Optics/visualization.py#L19)