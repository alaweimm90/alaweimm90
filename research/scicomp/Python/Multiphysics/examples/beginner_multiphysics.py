"""Beginner Multiphysics Examples.
This script demonstrates basic multiphysics coupling concepts including
thermal-mechanical coupling, fluid-structure interaction basics,
and electromagnetic heating fundamentals.
Topics covered:
- Thermal expansion and stress
- Basic FSI concepts
- Joule heating basics
- Simple coupling strategies
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thermal_mechanical import (ThermalExpansion, ThermalProperties,
                               MechanicalProperties, thermal_stress_analysis)
from electromagnetic import (electromagnetic_heating, JouleHeating,
                           ElectromagneticProperties)
from fluid_structure import FluidProperties, StructuralProperties
from visualization import MultiphysicsVisualizer
def thermal_expansion_demo():
    """Demonstrate thermal expansion and stress calculation."""
    print("=== Thermal Expansion Demo ===")
    # Material properties for steel
    thermal_props = ThermalProperties(
        conductivity=50.0,          # W/(m·K)
        specific_heat=500.0,        # J/(kg·K)
        density=7800.0,             # kg/m³
        thermal_expansion=1.2e-5,   # 1/K
        reference_temperature=20.0   # °C
    )
    mechanical_props = MechanicalProperties(
        youngs_modulus=200e9,       # Pa
        poissons_ratio=0.3,
        density=7800.0
    )
    # Create thermal expansion model
    expansion_model = ThermalExpansion("isotropic")
    # Temperature scenarios
    temperatures = np.array([20, 100, 200, 300, 400])  # °C
    print(f"Reference temperature: {thermal_props.reference_temperature}°C")
    print(f"Thermal expansion coefficient: {thermal_props.thermal_expansion:.2e} 1/K")
    print()
    for temp in temperatures:
        # Compute thermal strain
        thermal_strain = expansion_model.compute_thermal_strain(
            temp, thermal_props
        )
        # Compute thermal stress (constrained case)
        thermal_stress = expansion_model.compute_thermal_stress(
            temp, thermal_props, mechanical_props, "constrained"
        )
        delta_T = temp - thermal_props.reference_temperature
        strain_magnitude = np.abs(thermal_strain[0, 0])
        stress_magnitude = np.abs(thermal_stress[0, 0]) / 1e6  # Convert to MPa
        print(f"T = {temp}°C (ΔT = {delta_T}°C):")
        print(f"  Thermal strain: {strain_magnitude:.2e}")
        print(f"  Thermal stress: {stress_magnitude:.1f} MPa")
        print()
def plate_thermal_stress_example():
    """Analyze thermal stress in a heated plate."""
    print("=== Heated Plate Analysis ===")
    # Geometry
    geometry = {
        'plate': True,
        'length': 1.0,   # m
        'width': 0.5     # m
    }
    # Temperature distribution (linear heating)
    def temperature_field(x, y):
        T_cold = 20.0   # °C
        T_hot = 200.0   # °C
        return T_cold + (T_hot - T_cold) * x / geometry['length']
    # Material properties (aluminum)
    material = {
        'E': 70e9,           # Pa
        'nu': 0.33,
        'rho': 2700,         # kg/m³
        'alpha': 23e-6,      # 1/K
        'k': 237,            # W/(m·K)
        'cp': 900            # J/(kg·K)
    }
    # Analyze thermal stresses
    result = thermal_stress_analysis(
        geometry, temperature_field, material, "constrained"
    )
    print(f"Plate dimensions: {geometry['length']}m × {geometry['width']}m")
    print(f"Temperature range: 20°C to 200°C")
    print(f"Maximum thermal stress: {result['max_stress']/1e6:.1f} MPa")
    # Visualize results
    visualizer = MultiphysicsVisualizer()
    fig = plt.figure(figsize=(12, 5))
    # Temperature distribution
    plt.subplot(1, 2, 1)
    nodes = result['nodes']
    temps = result['temperature']
    plt.scatter(nodes[:, 0], nodes[:, 1], c=temps, cmap='hot', s=20)
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    # Stress distribution
    plt.subplot(1, 2, 2)
    stress = result['stress']
    stress_magnitude = np.linalg.norm(stress.reshape(len(stress), -1), axis=1)
    plt.scatter(nodes[:, 0], nodes[:, 1], c=stress_magnitude/1e6, cmap='viridis', s=20)
    plt.colorbar(label='Stress (MPa)')
    plt.title('Thermal Stress')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    plt.show()
def joule_heating_demo():
    """Demonstrate electromagnetic heating calculation."""
    print("=== Joule Heating Demo ===")
    # Material properties for copper wire
    material = {
        'conductivity': 5.96e7,     # S/m
        'permeability': 4*np.pi*1e-7,  # H/m (vacuum)
    }
    # Wire geometry
    wire_geometry = {
        'wire': True,
        'radius': 0.001,    # 1 mm radius
        'length': 1.0       # 1 m length
    }
    # Current and frequency scenarios
    current = 10.0      # A
    frequencies = [0, 50, 1000, 10000]  # Hz (DC, 50Hz, 1kHz, 10kHz)
    print(f"Copper wire: radius = {wire_geometry['radius']*1000:.1f} mm")
    print(f"Current: {current} A")
    print()
    for frequency in frequencies:
        result = electromagnetic_heating(
            wire_geometry, current, frequency, material
        )
        power = result['power']
        resistance = result['resistance']
        skin_depth = result['skin_depth']
        if frequency == 0:
            freq_str = "DC"
        else:
            freq_str = f"{frequency} Hz"
        print(f"Frequency: {freq_str}")
        print(f"  Resistance: {resistance*1000:.2f} mΩ")
        print(f"  Power: {power:.2f} W")
        if frequency > 0:
            print(f"  Skin depth: {skin_depth*1000:.2f} mm")
        print()
def basic_fsi_concept():
    """Demonstrate basic fluid-structure interaction concepts."""
    print("=== Basic FSI Concepts ===")
    # Fluid properties (water)
    fluid = FluidProperties(
        density=1000.0,      # kg/m³
        viscosity=1e-3,      # Pa·s
        bulk_modulus=2.2e9   # Pa
    )
    # Structure properties (rubber membrane)
    structure = StructuralProperties(
        density=1200.0,          # kg/m³
        youngs_modulus=1e6,      # Pa (soft material)
        poissons_ratio=0.45,
        thickness=0.001          # 1 mm
    )
    print("Fluid-Structure Interaction Example:")
    print(f"Fluid: Water (ρ = {fluid.density} kg/m³, μ = {fluid.viscosity*1000:.1f} cP)")
    print(f"Structure: Rubber (E = {structure.youngs_modulus/1e6:.1f} MPa)")
    print()
    # Demonstrate coupling effects
    pressure_load = 1000.0  # Pa
    membrane_area = 0.01    # m²
    # Force on membrane
    force = pressure_load * membrane_area
    # Estimate deflection (simplified)
    # δ = F*L³/(3*E*I) for beam, simplified for membrane
    length = 0.1  # m
    width = 0.1   # m
    # Membrane stiffness (simplified)
    membrane_stiffness = structure.youngs_modulus * structure.thickness / length
    deflection = force / membrane_stiffness
    print(f"Applied pressure: {pressure_load} Pa")
    print(f"Force on membrane: {force:.1f} N")
    print(f"Estimated deflection: {deflection*1000:.2f} mm")
    print()
    # Coupling considerations
    print("FSI Coupling Considerations:")
    print("- Fluid pressure causes structural deformation")
    print("- Structural motion changes fluid domain")
    print("- Iterative solution required for strong coupling")
    print("- Mesh motion needed for large deformations")
def simple_coupling_strategies():
    """Demonstrate different coupling strategies."""
    print("=== Coupling Strategies ===")
    strategies = [
        "One-way coupling",
        "Two-way coupling (staggered)",
        "Two-way coupling (monolithic)"
    ]
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}:")
        if "One-way" in strategy:
            print("   - Solve physics A → transfer data → solve physics B")
            print("   - Fast and stable")
            print("   - Limited accuracy for strong coupling")
            print("   - Example: Thermal → Structural")
        elif "staggered" in strategy:
            print("   - Alternately solve physics A and B")
            print("   - Iterate until convergence")
            print("   - Good balance of accuracy and efficiency")
            print("   - Example: Fluid ↔ Structure")
        elif "monolithic" in strategy:
            print("   - Solve all physics simultaneously")
            print("   - Highest accuracy")
            print("   - Computationally expensive")
            print("   - Example: All fields in single matrix")
        print()
def convergence_monitoring_demo():
    """Demonstrate convergence monitoring for coupled systems."""
    print("=== Convergence Monitoring ===")
    # Simulate convergence history
    max_iterations = 20
    target_tolerance = 1e-6
    # Different convergence behaviors
    behaviors = ["good", "slow", "oscillatory"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, behavior in enumerate(behaviors):
        if behavior == "good":
            # Exponential convergence
            residuals = [1.0 * (0.5)**k for k in range(max_iterations)]
        elif behavior == "slow":
            # Linear convergence
            residuals = [1.0 * (0.9)**k for k in range(max_iterations)]
        else:  # oscillatory
            # Oscillatory convergence
            residuals = [1.0 * (0.8)**k * (1 + 0.3*np.sin(k)) for k in range(max_iterations)]
        # Plot
        axes[i].semilogy(range(len(residuals)), residuals, 'o-',
                        color='#003262', linewidth=2, markersize=6)
        axes[i].axhline(y=target_tolerance, color='#FDB515',
                       linestyle='--', linewidth=2, label='Tolerance')
        axes[i].set_title(f'{behavior.title()} Convergence')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Residual')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        # Convergence analysis
        if residuals[-1] < target_tolerance:
            converged_iter = next(k for k, r in enumerate(residuals) if r < target_tolerance)
            print(f"{behavior.title()} convergence: converged at iteration {converged_iter}")
        else:
            print(f"{behavior.title()} convergence: not converged")
    plt.tight_layout()
    plt.show()
def main():
    """Run all beginner multiphysics examples."""
    print("Berkeley SciComp - Beginner Multiphysics Examples")
    print("=" * 50)
    print()
    # Run examples
    thermal_expansion_demo()
    print()
    plate_thermal_stress_example()
    print()
    joule_heating_demo()
    print()
    basic_fsi_concept()
    print()
    simple_coupling_strategies()
    print()
    convergence_monitoring_demo()
    print()
    print("Examples completed successfully!")
    print("\nKey takeaways:")
    print("- Multiphysics involves coupling between different physical phenomena")
    print("- Thermal expansion can cause significant stresses in constrained structures")
    print("- Electromagnetic heating depends on frequency through skin effect")
    print("- FSI requires careful treatment of moving boundaries")
    print("- Coupling strategies balance accuracy and computational cost")
    print("- Monitoring convergence is crucial for iterative coupling")
if __name__ == "__main__":
    main()