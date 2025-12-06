# ray_optics
**Module:** `Python/Optics/ray_optics.py`
## Overview
Ray Optics Module.
This module provides comprehensive ray optics functionality including
ray tracing, lens design, aberration analysis, and optical system design.
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `paraxial_ray_trace(surfaces, ray_height, ray_angle, wavelength)`
Perform paraxial ray trace through optical system.
Args:
surfaces: List of surface dictionaries with keys:
'position', 'radius', 'material_before', 'material_after'
ray_height: Initial ray height (meters)
ray_angle: Initial ray angle (radians)
wavelength: Wavelength (meters)
Returns:
Dictionary with ray trace results
**Source:** [Line 435](Python/Optics/ray_optics.py#L435)
### `trace_ray_through_system(ray, surfaces)`
Trace ray through list of optical surfaces.
Args:
ray: Input ray
surfaces: List of optical surfaces
Returns:
Ray trace results
**Source:** [Line 487](Python/Optics/ray_optics.py#L487)
### `calculate_aberrations(system, field_points, pupil_points, wavelength)`
Calculate Seidel aberrations for optical system.
Args:
system: Optical system
field_points: Field positions (meters)
pupil_points: Pupil coordinates (meters)
wavelength: Wavelength (meters)
Returns:
Dictionary with aberration coefficients
**Source:** [Line 504](Python/Optics/ray_optics.py#L504)
### `demo_ray_optics()`
Demonstrate ray optics functionality.
**Source:** [Line 522](Python/Optics/ray_optics.py#L522)
## Classes
### `Ray`
Optical ray representation.
#### Methods
##### `__post_init__(self)`
Normalize direction vector.
**Source:** [Line 29](Python/Optics/ray_optics.py#L29)
**Class Source:** [Line 21](Python/Optics/ray_optics.py#L21)
### `RayTraceResult`
Results of ray trace analysis.
**Class Source:** [Line 35](Python/Optics/ray_optics.py#L35)
### `OpticalSurface`
Abstract base class for optical surfaces.
#### Methods
##### `__init__(self, position, radius, material_before, material_after, diameter)`
Initialize optical surface.
Args:
position: Z position of surface (meters)
radius: Radius of curvature (meters, positive for convex)
material_before: Material before surface
material_after: Material after surface
diameter: Clear diameter (meters)
**Source:** [Line 48](Python/Optics/ray_optics.py#L48)
##### `intersect_ray(self, ray)`
Find intersection point of ray with surface.
Args:
ray: Input ray
Returns:
Tuple of (intersection_found, intersection_point)
**Source:** [Line 66](Python/Optics/ray_optics.py#L66)
##### `refract_ray(self, ray, intersection_point, wavelength)`
Refract ray at surface according to Snell's law.
Args:
ray: Incident ray
intersection_point: Point of intersection
wavelength: Wavelength for dispersion calculation
Returns:
Refracted ray
**Source:** [Line 78](Python/Optics/ray_optics.py#L78)
**Class Source:** [Line 45](Python/Optics/ray_optics.py#L45)
### `SphericalSurface`
Spherical refracting surface.
#### Methods
##### `intersect_ray(self, ray)`
Find intersection with spherical surface.
**Source:** [Line 96](Python/Optics/ray_optics.py#L96)
##### `surface_normal(self, point)`
Calculate surface normal at point.
**Source:** [Line 139](Python/Optics/ray_optics.py#L139)
##### `refract_ray(self, ray, intersection_point, wavelength)`
Refract ray using Snell's law.
**Source:** [Line 148](Python/Optics/ray_optics.py#L148)
**Class Source:** [Line 93](Python/Optics/ray_optics.py#L93)
### `ThinLens`
Thin lens element.
#### Methods
##### `__init__(self, position, focal_length, diameter, material)`
Initialize thin lens.
Args:
position: Z position (meters)
focal_length: Focal length (meters)
diameter: Clear diameter (meters)
material: Surrounding medium
**Source:** [Line 192](Python/Optics/ray_optics.py#L192)
##### `intersect_ray(self, ray)`
Find intersection with lens plane.
**Source:** [Line 205](Python/Optics/ray_optics.py#L205)
##### `refract_ray(self, ray, intersection_point, wavelength)`
Apply thin lens equation.
**Source:** [Line 223](Python/Optics/ray_optics.py#L223)
**Class Source:** [Line 189](Python/Optics/ray_optics.py#L189)
### `ThickLens`
Thick lens with two spherical surfaces.
#### Methods
##### `__init__(self, position, thickness, r1, r2, material, diameter, medium)`
Initialize thick lens.
Args:
position: Z position of first surface (meters)
thickness: Center thickness (meters)
r1: Radius of first surface (meters)
r2: Radius of second surface (meters)
material: Lens material
diameter: Clear diameter (meters)
medium: Surrounding medium
**Source:** [Line 248](Python/Optics/ray_optics.py#L248)
##### `trace_ray(self, ray)`
Trace ray through thick lens.
**Source:** [Line 267](Python/Optics/ray_optics.py#L267)
**Class Source:** [Line 245](Python/Optics/ray_optics.py#L245)
### `RayOptics`
Ray optics system for ray tracing and analysis.
#### Methods
##### `__init__(self)`
Initialize ray optics system.
**Source:** [Line 289](Python/Optics/ray_optics.py#L289)
##### `add_surface(self, surface)`
Add optical surface to system.
**Source:** [Line 295](Python/Optics/ray_optics.py#L295)
##### `trace_ray(self, ray)`
Trace single ray through optical system.
Args:
ray: Input ray
Returns:
Ray trace results
**Source:** [Line 299](Python/Optics/ray_optics.py#L299)
##### `trace_parallel_rays(self, heights, wavelength, direction)`
Trace parallel rays at different heights.
Args:
heights: Ray heights (meters)
wavelength: Wavelength (meters)
direction: Ray direction (default: +z)
Returns:
List of ray trace results
**Source:** [Line 331](Python/Optics/ray_optics.py#L331)
##### `calculate_focal_length(self, ray_heights, wavelength)`
Calculate effective focal length using paraxial rays.
Args:
ray_heights: Heights for ray tracing (meters)
wavelength: Wavelength (meters)
Returns:
Effective focal length (meters)
**Source:** [Line 354](Python/Optics/ray_optics.py#L354)
##### `analyze_aberrations(self, field_angles, ray_heights, wavelength)`
Analyze optical aberrations.
Args:
field_angles: Field angles (radians)
ray_heights: Ray heights (meters)
wavelength: Wavelength (meters)
Returns:
Dictionary with aberration analysis
**Source:** [Line 386](Python/Optics/ray_optics.py#L386)
**Class Source:** [Line 286](Python/Optics/ray_optics.py#L286)