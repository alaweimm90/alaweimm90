"""Ray Optics Module.
This module provides comprehensive ray optics functionality including
ray tracing, lens design, aberration analysis, and optical system design.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
from .optical_materials import calculate_refractive_index
@dataclass
class Ray:
    """Optical ray representation."""
    position: np.ndarray  # [x, y, z] position
    direction: np.ndarray  # [l, m, n] direction cosines
    wavelength: float     # wavelength in meters
    intensity: float = 1.0  # relative intensity
    optical_path_length: float = 0.0  # accumulated optical path
    def __post_init__(self):
        """Normalize direction vector."""
        self.direction = self.direction / np.linalg.norm(self.direction)
@dataclass
class RayTraceResult:
    """Results of ray trace analysis."""
    rays: List[Ray]
    surfaces_hit: List[int]
    success: bool
    message: str
    aberrations: Dict[str, float] = None
    spot_diagram: np.ndarray = None
class OpticalSurface(ABC):
    """Abstract base class for optical surfaces."""
    def __init__(self, position: float, radius: float, material_before: str,
                 material_after: str, diameter: float = np.inf):
        """Initialize optical surface.
        Args:
            position: Z position of surface (meters)
            radius: Radius of curvature (meters, positive for convex)
            material_before: Material before surface
            material_after: Material after surface
            diameter: Clear diameter (meters)
        """
        self.position = position
        self.radius = radius
        self.material_before = material_before
        self.material_after = material_after
        self.diameter = diameter
    @abstractmethod
    def intersect_ray(self, ray: Ray) -> Tuple[bool, np.ndarray]:
        """Find intersection point of ray with surface.
        Args:
            ray: Input ray
        Returns:
            Tuple of (intersection_found, intersection_point)
        """
        pass
    @abstractmethod
    def refract_ray(self, ray: Ray, intersection_point: np.ndarray,
                   wavelength: float) -> Ray:
        """Refract ray at surface according to Snell's law.
        Args:
            ray: Incident ray
            intersection_point: Point of intersection
            wavelength: Wavelength for dispersion calculation
        Returns:
            Refracted ray
        """
        pass
class SphericalSurface(OpticalSurface):
    """Spherical refracting surface."""
    def intersect_ray(self, ray: Ray) -> Tuple[bool, np.ndarray]:
        """Find intersection with spherical surface."""
        # Ray equation: P = P0 + t * D
        # Sphere equation: (x - xc)² + (y - yc)² + (z - zc)² = R²
        # Center of sphere
        if abs(self.radius) < 1e-12:  # Flat surface
            t = (self.position - ray.position[2]) / ray.direction[2]
            if t <= 0:
                return False, None
            intersection = ray.position + t * ray.direction
        else:
            center = np.array([0, 0, self.position + self.radius])
            # Quadratic equation coefficients
            a = np.dot(ray.direction, ray.direction)
            b = 2 * np.dot(ray.direction, ray.position - center)
            c = np.dot(ray.position - center, ray.position - center) - self.radius**2
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                return False, None
            # Choose appropriate root (closest positive intersection)
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            if t1 > 1e-12:
                t = t1
            elif t2 > 1e-12:
                t = t2
            else:
                return False, None
            intersection = ray.position + t * ray.direction
        # Check if intersection is within clear aperture
        r_aperture = np.sqrt(intersection[0]**2 + intersection[1]**2)
        if r_aperture > self.diameter / 2:
            return False, None
        return True, intersection
    def surface_normal(self, point: np.ndarray) -> np.ndarray:
        """Calculate surface normal at point."""
        if abs(self.radius) < 1e-12:  # Flat surface
            return np.array([0, 0, -1])
        else:
            center = np.array([0, 0, self.position + self.radius])
            normal = (point - center) / self.radius
            return normal
    def refract_ray(self, ray: Ray, intersection_point: np.ndarray,
                   wavelength: float) -> Ray:
        """Refract ray using Snell's law."""
        # Get refractive indices
        n1 = calculate_refractive_index(self.material_before, wavelength)
        n2 = calculate_refractive_index(self.material_after, wavelength)
        # Surface normal
        normal = self.surface_normal(intersection_point)
        # Ensure normal points into second medium
        if np.dot(normal, ray.direction) > 0:
            normal = -normal
        # Snell's law in vector form
        cos_theta1 = -np.dot(ray.direction, normal)
        sin_theta1_squared = 1 - cos_theta1**2
        # Check for total internal reflection
        if n1 > n2:
            sin_theta_c_squared = (n2/n1)**2
            if sin_theta1_squared > sin_theta_c_squared:
                # Total internal reflection
                reflected_direction = ray.direction + 2 * cos_theta1 * normal
                return Ray(intersection_point, reflected_direction, wavelength,
                          ray.intensity, ray.optical_path_length)
        # Refracted ray direction
        n_ratio = n1 / n2
        cos_theta2 = np.sqrt(1 - n_ratio**2 * sin_theta1_squared)
        refracted_direction = (n_ratio * ray.direction +
                             (n_ratio * cos_theta1 - cos_theta2) * normal)
        # Update optical path length
        distance = np.linalg.norm(intersection_point - ray.position)
        new_opl = ray.optical_path_length + n1 * distance
        return Ray(intersection_point, refracted_direction, wavelength,
                  ray.intensity, new_opl)
class ThinLens(OpticalSurface):
    """Thin lens element."""
    def __init__(self, position: float, focal_length: float, diameter: float,
                 material: str = 'air'):
        """Initialize thin lens.
        Args:
            position: Z position (meters)
            focal_length: Focal length (meters)
            diameter: Clear diameter (meters)
            material: Surrounding medium
        """
        super().__init__(position, np.inf, material, material, diameter)
        self.focal_length = focal_length
    def intersect_ray(self, ray: Ray) -> Tuple[bool, np.ndarray]:
        """Find intersection with lens plane."""
        if abs(ray.direction[2]) < 1e-12:
            return False, None
        t = (self.position - ray.position[2]) / ray.direction[2]
        if t <= 0:
            return False, None
        intersection = ray.position + t * ray.direction
        # Check aperture
        r = np.sqrt(intersection[0]**2 + intersection[1]**2)
        if r > self.diameter / 2:
            return False, None
        return True, intersection
    def refract_ray(self, ray: Ray, intersection_point: np.ndarray,
                   wavelength: float) -> Ray:
        """Apply thin lens equation."""
        h = np.sqrt(intersection_point[0]**2 + intersection_point[1]**2)
        # Thin lens deflection
        deflection_angle = -h / self.focal_length
        # New direction (paraxial approximation)
        new_direction = ray.direction.copy()
        if intersection_point[0] != 0:
            new_direction[0] += deflection_angle * intersection_point[0] / h
        if intersection_point[1] != 0:
            new_direction[1] += deflection_angle * intersection_point[1] / h
        # Normalize
        new_direction = new_direction / np.linalg.norm(new_direction)
        return Ray(intersection_point, new_direction, wavelength,
                  ray.intensity, ray.optical_path_length)
class ThickLens:
    """Thick lens with two spherical surfaces."""
    def __init__(self, position: float, thickness: float, r1: float, r2: float,
                 material: str, diameter: float, medium: str = 'air'):
        """Initialize thick lens.
        Args:
            position: Z position of first surface (meters)
            thickness: Center thickness (meters)
            r1: Radius of first surface (meters)
            r2: Radius of second surface (meters)
            material: Lens material
            diameter: Clear diameter (meters)
            medium: Surrounding medium
        """
        self.surface1 = SphericalSurface(position, r1, medium, material, diameter)
        self.surface2 = SphericalSurface(position + thickness, r2, material, medium, diameter)
        self.thickness = thickness
        self.material = material
        self.medium = medium
    def trace_ray(self, ray: Ray) -> Tuple[bool, Ray]:
        """Trace ray through thick lens."""
        # First surface
        hit1, point1 = self.surface1.intersect_ray(ray)
        if not hit1:
            return False, ray
        ray1 = self.surface1.refract_ray(ray, point1, ray.wavelength)
        # Second surface
        hit2, point2 = self.surface2.intersect_ray(ray1)
        if not hit2:
            return False, ray1
        ray2 = self.surface2.refract_ray(ray1, point2, ray.wavelength)
        return True, ray2
class RayOptics:
    """Ray optics system for ray tracing and analysis."""
    def __init__(self):
        """Initialize ray optics system."""
        self.surfaces = []
        self.berkeley_blue = '#003262'
        self.california_gold = '#FDB515'
    def add_surface(self, surface: OpticalSurface):
        """Add optical surface to system."""
        self.surfaces.append(surface)
    def trace_ray(self, ray: Ray) -> RayTraceResult:
        """Trace single ray through optical system.
        Args:
            ray: Input ray
        Returns:
            Ray trace results
        """
        rays = [ray]
        surfaces_hit = []
        current_ray = ray
        try:
            for i, surface in enumerate(self.surfaces):
                # Find intersection
                hit, intersection = surface.intersect_ray(current_ray)
                if not hit:
                    break
                surfaces_hit.append(i)
                # Refract ray
                new_ray = surface.refract_ray(current_ray, intersection, ray.wavelength)
                rays.append(new_ray)
                current_ray = new_ray
            return RayTraceResult(rays, surfaces_hit, True, "Ray trace successful")
        except Exception as e:
            return RayTraceResult(rays, surfaces_hit, False, f"Ray trace failed: {str(e)}")
    def trace_parallel_rays(self, heights: np.ndarray, wavelength: float = 589e-9,
                           direction: np.ndarray = None) -> List[RayTraceResult]:
        """Trace parallel rays at different heights.
        Args:
            heights: Ray heights (meters)
            wavelength: Wavelength (meters)
            direction: Ray direction (default: +z)
        Returns:
            List of ray trace results
        """
        if direction is None:
            direction = np.array([0, 0, 1])
        results = []
        for h in heights:
            ray = Ray(np.array([h, 0, -1]), direction, wavelength)
            result = self.trace_ray(ray)
            results.append(result)
        return results
    def calculate_focal_length(self, ray_heights: np.ndarray = None,
                              wavelength: float = 589e-9) -> float:
        """Calculate effective focal length using paraxial rays.
        Args:
            ray_heights: Heights for ray tracing (meters)
            wavelength: Wavelength (meters)
        Returns:
            Effective focal length (meters)
        """
        if ray_heights is None:
            ray_heights = np.linspace(0.1e-3, 1e-3, 10)
        # Trace paraxial rays
        focal_points = []
        for h in ray_heights:
            ray = Ray(np.array([h, 0, -10]), np.array([0, 0, 1]), wavelength)
            result = self.trace_ray(ray)
            if result.success and len(result.rays) > 1:
                # Find focus by extrapolating final ray
                final_ray = result.rays[-1]
                if abs(final_ray.direction[2]) > 1e-12:
                    z_focus = final_ray.position[2] - final_ray.position[0] / final_ray.direction[0] * final_ray.direction[2]
                    focal_points.append(z_focus)
        if focal_points:
            return np.mean(focal_points) - self.surfaces[0].position
        else:
            return np.inf
    def analyze_aberrations(self, field_angles: np.ndarray,
                           ray_heights: np.ndarray,
                           wavelength: float = 589e-9) -> Dict[str, Any]:
        """Analyze optical aberrations.
        Args:
            field_angles: Field angles (radians)
            ray_heights: Ray heights (meters)
            wavelength: Wavelength (meters)
        Returns:
            Dictionary with aberration analysis
        """
        aberrations = {
            'spherical': [],
            'coma': [],
            'astigmatism': [],
            'field_curvature': [],
            'distortion': []
        }
        # Trace rays for aberration analysis
        for angle in field_angles:
            direction = np.array([np.sin(angle), 0, np.cos(angle)])
            ray_results = []
            for h in ray_heights:
                ray = Ray(np.array([h, 0, -10]), direction, wavelength)
                result = self.trace_ray(ray)
                if result.success:
                    ray_results.append(result.rays[-1])
            # Calculate aberrations (simplified analysis)
            if ray_results:
                # Spherical aberration
                focal_shifts = []
                for ray_result in ray_results:
                    if abs(ray_result.direction[2]) > 1e-12:
                        z_focus = (ray_result.position[2] -
                                 ray_result.position[0] / ray_result.direction[0] * ray_result.direction[2])
                        focal_shifts.append(z_focus)
                if focal_shifts:
                    spherical_aberration = np.std(focal_shifts)
                    aberrations['spherical'].append(spherical_aberration)
        return aberrations
def paraxial_ray_trace(surfaces: List[Dict], ray_height: float = 0,
                      ray_angle: float = 0, wavelength: float = 589e-9) -> Dict[str, Any]:
    """Perform paraxial ray trace through optical system.
    Args:
        surfaces: List of surface dictionaries with keys:
                 'position', 'radius', 'material_before', 'material_after'
        ray_height: Initial ray height (meters)
        ray_angle: Initial ray angle (radians)
        wavelength: Wavelength (meters)
    Returns:
        Dictionary with ray trace results
    """
    # Initial ray parameters
    h = ray_height
    u = ray_angle  # ray angle
    heights = [h]
    angles = [u]
    positions = [surfaces[0]['position'] - 1]  # Start before first surface
    for i, surface in enumerate(surfaces):
        # Propagate to surface
        distance = surface['position'] - positions[-1]
        h = h + u * distance
        heights.append(h)
        angles.append(u)
        positions.append(surface['position'])
        # Refraction at surface
        n1 = calculate_refractive_index(surface['material_before'], wavelength)
        n2 = calculate_refractive_index(surface['material_after'], wavelength)
        if abs(surface['radius']) > 1e12:  # Flat surface
            power = 0
        else:
            power = (n2 - n1) / surface['radius']
        # Paraxial refraction
        u_new = u - h * power / n2
        u = u_new
    return {
        'heights': np.array(heights),
        'angles': np.array(angles),
        'positions': np.array(positions),
        'success': True
    }
def trace_ray_through_system(ray: Ray, surfaces: List[OpticalSurface]) -> RayTraceResult:
    """Trace ray through list of optical surfaces.
    Args:
        ray: Input ray
        surfaces: List of optical surfaces
    Returns:
        Ray trace results
    """
    system = RayOptics()
    for surface in surfaces:
        system.add_surface(surface)
    return system.trace_ray(ray)
def calculate_aberrations(system: RayOptics, field_points: np.ndarray,
                         pupil_points: np.ndarray,
                         wavelength: float = 589e-9) -> Dict[str, Any]:
    """Calculate Seidel aberrations for optical system.
    Args:
        system: Optical system
        field_points: Field positions (meters)
        pupil_points: Pupil coordinates (meters)
        wavelength: Wavelength (meters)
    Returns:
        Dictionary with aberration coefficients
    """
    return system.analyze_aberrations(field_points, pupil_points, wavelength)
# Demonstration and examples
def demo_ray_optics():
    """Demonstrate ray optics functionality."""
    print("Ray Optics Demo")
    print("===============")
    # Create simple lens system
    system = RayOptics()
    # Add a positive thin lens
    lens = ThinLens(position=0, focal_length=0.1, diameter=0.02)
    system.add_surface(lens)
    print("\n1. Single Lens System")
    print(f"Focal length: {lens.focal_length*1000:.1f} mm")
    # Trace parallel rays
    heights = np.linspace(-5e-3, 5e-3, 11)
    wavelength = 589e-9
    results = system.trace_parallel_rays(heights, wavelength)
    # Extract ray paths for plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Ray diagram
    for i, result in enumerate(results):
        if result.success:
            z_positions = []
            y_positions = []
            for ray in result.rays:
                z_positions.append(ray.position[2])
                y_positions.append(ray.position[0])
            # Extend final ray to show convergence
            final_ray = result.rays[-1]
            z_extend = np.linspace(final_ray.position[2], 0.2, 50)
            y_extend = (final_ray.position[0] +
                       final_ray.direction[0] / final_ray.direction[2] *
                       (z_extend - final_ray.position[2]))
            ax1.plot(np.array(z_positions)*1000, np.array(y_positions)*1000,
                    'b-', alpha=0.7)
            ax1.plot(z_extend*1000, y_extend*1000, 'b--', alpha=0.5)
    # Lens representation
    ax1.axvline(0, color='#FDB515', linewidth=3, label='Lens')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Height (mm)')
    ax1.set_title('Ray Diagram')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Spot diagram at focus
    focal_length = system.calculate_focal_length(heights, wavelength)
    print(f"Calculated focal length: {focal_length*1000:.2f} mm")
    # Calculate spot positions
    spot_x = []
    spot_y = []
    for result in results:
        if result.success:
            final_ray = result.rays[-1]
            # Propagate to focal plane
            t = (focal_length - final_ray.position[2]) / final_ray.direction[2]
            spot_pos = final_ray.position + t * final_ray.direction
            spot_x.append(spot_pos[0])
            spot_y.append(spot_pos[1])
    ax2.scatter(np.array(spot_x)*1e6, np.array(spot_y)*1e6,
               c='#003262', s=50, alpha=0.7)
    ax2.set_xlabel('X Position (μm)')
    ax2.set_ylabel('Y Position (μm)')
    ax2.set_title('Spot Diagram at Focus')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    plt.tight_layout()
    plt.show()
    # Aberration analysis
    print("\n2. Aberration Analysis")
    field_angles = np.linspace(0, 0.1, 5)  # Small field angles
    ray_heights = np.linspace(1e-3, 8e-3, 8)
    aberrations = system.analyze_aberrations(field_angles, ray_heights, wavelength)
    if aberrations['spherical']:
        print(f"Spherical aberration (RMS): {np.mean(aberrations['spherical'])*1e6:.2f} μm")
if __name__ == "__main__":
    demo_ray_optics()