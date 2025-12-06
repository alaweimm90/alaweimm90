#!/usr/bin/env python3
"""Basic Optics Examples - Beginner Level.
This script demonstrates fundamental optics concepts using the Berkeley SciComp
Optics package including geometric optics, wave optics, and material properties.
Author: Berkeley SciComp Team
Date: 2024
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from Optics.wave_optics import GaussianBeam, analyze_interference, calculate_diffraction
from Optics.ray_optics import RayOptics, ThinLens, Ray
from Optics.optical_materials import create_material_database, calculate_refractive_index
from Optics.visualization import OpticsVisualizer
def example_1_snells_law():
    """Example 1: Snell's Law and Refraction."""
    print("=" * 60)
    print("Example 1: Snell's Law and Refraction")
    print("=" * 60)
    # Parameters
    wavelength = 589.3e-9  # Sodium D-line
    incident_angle = 30  # degrees
    # Materials
    materials_db = create_material_database()
    n_air = calculate_refractive_index('air', wavelength)
    n_glass = calculate_refractive_index('BK7', wavelength)
    n_water = calculate_refractive_index('H2O', wavelength)
    print(f"Wavelength: {wavelength*1e9:.1f} nm")
    print(f"Incident angle: {incident_angle}°")
    print(f"\nRefractive indices:")
    print(f"Air:   n = {n_air:.6f}")
    print(f"BK7:   n = {n_glass:.6f}")
    print(f"Water: n = {n_water:.6f}")
    # Calculate refraction angles using Snell's law
    theta_i_rad = np.radians(incident_angle)
    # Air to glass
    sin_theta_r_glass = n_air * np.sin(theta_i_rad) / n_glass
    theta_r_glass = np.degrees(np.arcsin(sin_theta_r_glass))
    # Air to water
    sin_theta_r_water = n_air * np.sin(theta_i_rad) / n_water
    theta_r_water = np.degrees(np.arcsin(sin_theta_r_water))
    print(f"\nRefraction angles:")
    print(f"Air → BK7:   {theta_r_glass:.2f}°")
    print(f"Air → Water: {theta_r_water:.2f}°")
    # Critical angle for total internal reflection
    theta_c_glass = np.degrees(np.arcsin(n_air / n_glass))
    theta_c_water = np.degrees(np.arcsin(n_air / n_water))
    print(f"\nCritical angles (for total internal reflection):")
    print(f"BK7 → Air:   {theta_c_glass:.2f}°")
    print(f"Water → Air: {theta_c_water:.2f}°")
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    # Interface
    ax.axhline(y=0, color='black', linewidth=2, label='Interface')
    ax.axhline(y=0, xmin=0, xmax=1, color='lightblue', linewidth=10, alpha=0.3)
    ax.text(0.5, 0.1, 'BK7 Glass', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.9, 'Air', ha='center', transform=ax.transAxes, fontsize=12)
    # Incident ray
    x_incident = [-1, 0]
    y_incident = [np.tan(theta_i_rad), 0]
    ax.plot(x_incident, y_incident, 'r-', linewidth=3, label=f'Incident ({incident_angle}°)')
    # Refracted ray
    x_refracted = [0, 1]
    y_refracted = [0, -np.tan(np.radians(theta_r_glass))]
    ax.plot(x_refracted, y_refracted, 'b-', linewidth=3, label=f'Refracted ({theta_r_glass:.1f}°)')
    # Normal line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Normal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Snell\'s Law: Refraction at Air-Glass Interface',
                fontsize=14, color='#003262', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
def example_2_thin_lens():
    """Example 2: Thin Lens Ray Tracing."""
    print("=" * 60)
    print("Example 2: Thin Lens Ray Tracing")
    print("=" * 60)
    # Lens parameters
    focal_length = 0.1  # 100 mm
    lens_diameter = 0.025  # 25 mm
    object_distance = 0.15  # 150 mm
    print(f"Lens focal length: {focal_length*1000:.0f} mm")
    print(f"Object distance: {object_distance*1000:.0f} mm")
    # Calculate image distance using thin lens equation
    image_distance = 1 / (1/focal_length - 1/object_distance)
    magnification = -image_distance / object_distance
    print(f"Image distance: {image_distance*1000:.1f} mm")
    print(f"Magnification: {magnification:.2f}x")
    # Setup ray tracing system
    system = RayOptics()
    lens = ThinLens(position=0, focal_length=focal_length, diameter=lens_diameter)
    system.add_surface(lens)
    # Trace rays from object
    object_height = 0.005  # 5 mm
    ray_heights = np.linspace(-object_height, object_height, 11)
    wavelength = 589e-9
    # Store ray paths for plotting
    ray_paths = []
    for h in ray_heights:
        # Ray from object point
        ray_direction = np.array([0, 0, 1])  # Parallel to optical axis initially
        ray_start = np.array([h, 0, -object_distance])
        ray = Ray(ray_start, ray_direction, wavelength)
        result = system.trace_ray(ray)
        if result.success:
            ray_paths.append(result.rays)
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    # Plot rays
    colors = ['#003262', '#FDB515', '#3B7EA1', '#C4820E', '#00B0DA']
    for i, ray_path in enumerate(ray_paths):
        z_positions = []
        x_positions = []
        for ray in ray_path:
            z_positions.append(ray.position[2])
            x_positions.append(ray.position[0])
        # Extend final ray to image plane
        if len(ray_path) > 1:
            final_ray = ray_path[-1]
            z_extend = np.linspace(final_ray.position[2], image_distance, 50)
            x_extend = (final_ray.position[0] +
                       final_ray.direction[0] / final_ray.direction[2] *
                       (z_extend - final_ray.position[2]))
            color = colors[i % len(colors)]
            ax.plot(np.array(z_positions)*1000, np.array(x_positions)*1000,
                   color=color, linewidth=1.5, alpha=0.8)
            ax.plot(z_extend*1000, x_extend*1000, color=color,
                   linewidth=1.5, alpha=0.6, linestyle='--')
    # Draw lens
    lens_height = lens_diameter * 1000 / 2
    ax.plot([0, 0], [-lens_height, lens_height], 'k-', linewidth=4,
           label='Lens', color='#FDB515')
    # Draw object and image
    ax.plot([-object_distance*1000, -object_distance*1000],
           [-object_height*1000, object_height*1000], 'r-', linewidth=3,
           label='Object')
    image_height = object_height * abs(magnification)
    ax.plot([image_distance*1000, image_distance*1000],
           [-image_height*1000, image_height*1000], 'g-', linewidth=3,
           label='Image')
    # Focal points
    ax.plot([-focal_length*1000, focal_length*1000], [0, 0], 'o',
           color='#003262', markersize=8, label='Focal Points')
    # Optical axis
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Thin Lens Ray Tracing', fontsize=14, color='#003262', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def example_3_gaussian_beam():
    """Example 3: Gaussian Beam Propagation."""
    print("=" * 60)
    print("Example 3: Gaussian Beam Propagation")
    print("=" * 60)
    # Beam parameters
    wavelength = 633e-9  # HeNe laser
    waist_radius = 1e-3  # 1 mm
    power = 1e-3  # 1 mW
    beam = GaussianBeam(wavelength, waist_radius, power=power)
    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Beam waist: {waist_radius*1000:.1f} mm")
    print(f"Rayleigh range: {beam.rayleigh_range*1000:.2f} mm")
    print(f"Divergence angle: {beam.divergence_angle*1000:.2f} mrad")
    print(f"Power: {power*1000:.1f} mW")
    # Propagation distances
    z_positions = np.linspace(-5*beam.rayleigh_range, 5*beam.rayleigh_range, 200)
    beam_radii = [beam.beam_radius(z) for z in z_positions]
    # Beam profile at different positions
    x_profile = np.linspace(-5e-3, 5e-3, 200)
    y_profile = np.linspace(-5e-3, 5e-3, 200)
    # Calculate intensity profiles at waist and at 2*zR
    z_waist = 0
    z_diverged = 2 * beam.rayleigh_range
    intensity_waist = beam.intensity_profile(x_profile, np.array([0]), z_waist)
    intensity_diverged = beam.intensity_profile(x_profile, np.array([0]), z_diverged)
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # Beam propagation
    ax1.plot(z_positions*1000, np.array(beam_radii)*1000,
            color='#003262', linewidth=3, label='Beam Radius')
    ax1.plot(z_positions*1000, -np.array(beam_radii)*1000,
            color='#003262', linewidth=3)
    ax1.axhline(waist_radius*1000, color='#FDB515', linestyle='--',
               linewidth=2, label='Waist Radius')
    ax1.axhline(-waist_radius*1000, color='#FDB515', linestyle='--', linewidth=2)
    ax1.axvline(beam.rayleigh_range*1000, color='gray', linestyle=':',
               alpha=0.7, label='Rayleigh Range')
    ax1.axvline(-beam.rayleigh_range*1000, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Beam Radius (mm)')
    ax1.set_title('Gaussian Beam Propagation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Intensity profile at waist
    ax2.plot(x_profile*1000, intensity_waist[0, :], color='#003262', linewidth=3)
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Intensity (W/m²)')
    ax2.set_title('Intensity Profile at Waist')
    ax2.grid(True, alpha=0.3)
    # Intensity profile at 2*zR
    ax3.plot(x_profile*1000, intensity_diverged[0, :], color='#FDB515', linewidth=3)
    ax3.set_xlabel('Position (mm)')
    ax3.set_ylabel('Intensity (W/m²)')
    ax3.set_title(f'Intensity Profile at z = 2zR = {2*beam.rayleigh_range*1000:.1f} mm')
    ax3.grid(True, alpha=0.3)
    # Beam evolution (3D-like plot)
    Z, X = np.meshgrid(z_positions, x_profile)
    intensity_evolution = np.zeros_like(Z)
    for i, z in enumerate(z_positions):
        intensity_evolution[:, i] = beam.intensity_profile(x_profile, np.array([0]), z)[0, :]
    contour = ax4.contourf(Z*1000, X*1000, intensity_evolution, levels=20, cmap='Blues')
    ax4.set_xlabel('Propagation Distance (mm)')
    ax4.set_ylabel('Beam Radius (mm)')
    ax4.set_title('Beam Intensity Evolution')
    plt.colorbar(contour, ax=ax4, label='Intensity (W/m²)')
    plt.tight_layout()
    plt.show()
def example_4_interference():
    """Example 4: Young's Double Slit Interference."""
    print("=" * 60)
    print("Example 4: Young's Double Slit Interference")
    print("=" * 60)
    # Parameters
    wavelength = 550e-9  # Green light
    slit_separation = 100e-6  # 100 μm
    screen_distance = 1.0  # 1 m
    screen_width = 0.01  # 10 mm
    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Slit separation: {slit_separation*1e6:.0f} μm")
    print(f"Screen distance: {screen_distance:.1f} m")
    # Calculate interference pattern
    interference = analyze_interference(
        wavelength, slit_separation, screen_distance,
        pattern_type='double_slit'
    )
    print(f"Fringe spacing: {interference['fringe_spacing']*1000:.3f} mm")
    print(f"Visibility: {interference['visibility']:.3f}")
    # Theoretical fringe spacing
    fringe_spacing_theory = wavelength * screen_distance / slit_separation
    print(f"Theoretical fringe spacing: {fringe_spacing_theory*1000:.3f} mm")
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Interference pattern
    position_mm = interference['position'] * 1000
    ax1.plot(position_mm, interference['intensity_normalized'],
            color='#003262', linewidth=2)
    ax1.fill_between(position_mm, 0, interference['intensity_normalized'],
                    alpha=0.3, color='#003262')
    # Mark maxima
    maxima_indices = []
    for i in range(1, len(interference['intensity_normalized'])-1):
        if (interference['intensity_normalized'][i] > interference['intensity_normalized'][i-1] and
            interference['intensity_normalized'][i] > interference['intensity_normalized'][i+1] and
            interference['intensity_normalized'][i] > 0.8):
            maxima_indices.append(i)
    if maxima_indices:
        ax1.plot(position_mm[maxima_indices],
                interference['intensity_normalized'][maxima_indices],
                'o', color='#FDB515', markersize=8, label='Maxima')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Young\'s Double Slit Interference Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Zoom in on central fringes
    central_range = 5e-3  # ±5 mm
    central_mask = np.abs(interference['position']) <= central_range
    ax2.plot(position_mm[central_mask],
            interference['intensity_normalized'][central_mask],
            color='#003262', linewidth=3)
    ax2.fill_between(position_mm[central_mask], 0,
                    interference['intensity_normalized'][central_mask],
                    alpha=0.3, color='#003262')
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Central Interference Fringes (Zoomed)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def example_5_single_slit_diffraction():
    """Example 5: Single Slit Diffraction."""
    print("=" * 60)
    print("Example 5: Single Slit Diffraction")
    print("=" * 60)
    # Parameters
    wavelength = 633e-9  # HeNe laser
    slit_width = 50e-6  # 50 μm
    screen_distance = 2.0  # 2 m
    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Slit width: {slit_width*1e6:.0f} μm")
    print(f"Screen distance: {screen_distance:.1f} m")
    # Calculate diffraction pattern
    diffraction = calculate_diffraction(
        'single_slit', slit_width, wavelength, screen_distance, 0.02
    )
    # Theoretical first minimum position
    first_minimum = wavelength * screen_distance / slit_width
    print(f"First minimum position: ±{first_minimum*1000:.2f} mm")
    # Find actual minima
    intensity = diffraction['intensity_normalized']
    position = diffraction['position']
    # Find local minima (excluding center)
    minima_indices = []
    for i in range(1, len(intensity)-1):
        if (intensity[i] < intensity[i-1] and intensity[i] < intensity[i+1] and
            abs(position[i]) > 1e-3):  # Exclude central region
            minima_indices.append(i)
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Full diffraction pattern
    position_mm = position * 1000
    ax1.plot(position_mm, intensity, color='#003262', linewidth=2)
    ax1.fill_between(position_mm, 0, intensity, alpha=0.3, color='#003262')
    # Mark theoretical minima
    theory_minima = [-first_minimum, first_minimum]
    for pos in theory_minima:
        ax1.axvline(pos*1000, color='#FDB515', linestyle='--',
                   label='Theoretical Minima' if pos == theory_minima[0] else '')
    # Mark actual minima
    if minima_indices:
        ax1.plot(position_mm[minima_indices], intensity[minima_indices],
                'o', color='red', markersize=6, label='Observed Minima')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Single Slit Diffraction Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Logarithmic scale to show side lobes
    ax2.semilogy(position_mm, intensity, color='#FDB515', linewidth=2)
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Normalized Intensity (log scale)')
    ax2.set_title('Single Slit Diffraction (Log Scale)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def main():
    """Run all basic optics examples."""
    print("BERKELEY SCICOMP - BASIC OPTICS EXAMPLES")
    print("========================================")
    print("This script demonstrates fundamental optics concepts including:")
    print("1. Snell's Law and Refraction")
    print("2. Thin Lens Ray Tracing")
    print("3. Gaussian Beam Propagation")
    print("4. Young's Double Slit Interference")
    print("5. Single Slit Diffraction")
    print()
    try:
        example_1_snells_law()
        input("\nPress Enter to continue to Example 2...")
        example_2_thin_lens()
        input("\nPress Enter to continue to Example 3...")
        example_3_gaussian_beam()
        input("\nPress Enter to continue to Example 4...")
        example_4_interference()
        input("\nPress Enter to continue to Example 5...")
        example_5_single_slit_diffraction()
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Learning Points:")
        print("• Snell's law governs refraction at interfaces")
        print("• Thin lenses follow predictable ray tracing rules")
        print("• Gaussian beams have characteristic propagation properties")
        print("• Interference creates predictable fringe patterns")
        print("• Diffraction limits resolution in optical systems")
        print("\nFor more advanced examples, see intermediate and advanced tutorials.")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError in demo: {e}")
        raise
if __name__ == "__main__":
    main()