"""
Beginner Example: Basic Stress-Strain Analysis
This example demonstrates fundamental stress and strain tensor operations,
elastic material behavior, and simple stress analysis for engineering applications.
Learning Objectives:
- Understand stress and strain tensor concepts
- Apply Hooke's law for elastic materials
- Calculate principal stresses and strains
- Perform basic failure analysis using von Mises criterion
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# Add Elasticity module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.stress_strain import (StressTensor, StrainTensor, IsotropicElasticity,
                               ElasticConstants, create_material_database)
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
BERKELEY_GOLD = '#FDB515'
def create_steel_material():
    """Create steel material with typical properties."""
    print("Creating Steel Material Properties")
    print("-" * 40)
    # Typical mild steel properties
    youngs_modulus = 200e9  # Pa (200 GPa)
    poissons_ratio = 0.30
    density = 7850  # kg/m³
    steel = IsotropicElasticity(youngs_modulus, poissons_ratio, density)
    print(f"Young's modulus: {youngs_modulus/1e9:.1f} GPa")
    print(f"Poisson's ratio: {poissons_ratio:.2f}")
    print(f"Shear modulus: {steel.constants.shear_modulus/1e9:.1f} GPa")
    print(f"Bulk modulus: {steel.constants.bulk_modulus/1e9:.1f} GPa")
    print(f"Density: {density} kg/m³")
    return steel
def demonstrate_stress_tensor_operations():
    """Demonstrate stress tensor creation and operations."""
    print("\n\nStress Tensor Operations")
    print("=" * 30)
    # Create stress tensor for uniaxial tension
    print("\n1. Uniaxial Tension (σx = 100 MPa)")
    print("-" * 35)
    stress_uniaxial = StressTensor(sigma_xx=100e6)  # 100 MPa in x-direction
    print(f"Stress tensor:\n{stress_uniaxial.tensor/1e6:.1f} (MPa)")
    # Calculate principal stresses
    principal_stresses, principal_directions = stress_uniaxial.principal_stresses()
    print(f"\nPrincipal stresses: {principal_stresses/1e6:.1f} MPa")
    print(f"von Mises stress: {stress_uniaxial.von_mises_stress()/1e6:.1f} MPa")
    # Create stress tensor for pure shear
    print("\n2. Pure Shear (τxy = 50 MPa)")
    print("-" * 30)
    stress_shear = StressTensor(sigma_xy=50e6)  # 50 MPa shear
    print(f"Stress tensor:\n{stress_shear.tensor/1e6:.1f} (MPa)")
    principal_stresses, _ = stress_shear.principal_stresses()
    print(f"Principal stresses: {principal_stresses/1e6:.1f} MPa")
    print(f"von Mises stress: {stress_shear.von_mises_stress()/1e6:.1f} MPa")
    print(f"Maximum shear stress: {stress_shear.maximum_shear_stress()/1e6:.1f} MPa")
    # Create complex stress state
    print("\n3. Complex Stress State")
    print("-" * 25)
    stress_complex = StressTensor(
        sigma_xx=80e6,   # 80 MPa
        sigma_yy=40e6,   # 40 MPa
        sigma_zz=20e6,   # 20 MPa
        sigma_xy=30e6    # 30 MPa shear
    )
    print(f"Stress tensor:\n{stress_complex.tensor/1e6:.1f} (MPa)")
    principal_stresses, _ = stress_complex.principal_stresses()
    print(f"Principal stresses: {principal_stresses/1e6:.1f} MPa")
    print(f"von Mises stress: {stress_complex.von_mises_stress()/1e6:.1f} MPa")
    # Stress invariants
    I1, I2, I3 = stress_complex.invariants()
    print(f"Stress invariants:")
    print(f"  I₁ = {I1/1e6:.1f} MPa")
    print(f"  I₂ = {I2/1e12:.1f} (MPa)²")
    print(f"  I₃ = {I3/1e18:.1f} (MPa)³")
    return stress_uniaxial, stress_shear, stress_complex
def demonstrate_strain_tensor_operations():
    """Demonstrate strain tensor creation and operations."""
    print("\n\nStrain Tensor Operations")
    print("=" * 30)
    # Create strain tensor
    print("\n1. Simple Strain State")
    print("-" * 22)
    strain = StrainTensor(
        epsilon_xx=0.001,    # 1000 microstrain
        epsilon_yy=0.0005,   # 500 microstrain
        gamma_xy=0.0008      # 800 microstrain engineering shear
    )
    print(f"Strain tensor:\n{strain.tensor*1e6:.0f} (microstrain)")
    # Calculate principal strains
    principal_strains, _ = strain.principal_strains()
    print(f"Principal strains: {principal_strains*1e6:.0f} microstrain")
    # Volumetric and deviatoric strains
    volumetric_strain = strain.volumetric_strain()
    deviatoric_strain = strain.deviatoric()
    equivalent_strain = strain.equivalent_strain()
    print(f"Volumetric strain: {volumetric_strain*1e6:.0f} microstrain")
    print(f"Equivalent strain: {equivalent_strain*1e6:.0f} microstrain")
    return strain
def demonstrate_hookes_law(steel, strain):
    """Demonstrate Hooke's law relationships."""
    print("\n\nHooke's Law Application")
    print("=" * 25)
    # Calculate stress from strain
    stress_from_strain = steel.stress_from_strain(strain)
    print(f"Stress from strain (Hooke's law):")
    print(f"Stress tensor:\n{stress_from_strain.tensor/1e6:.1f} (MPa)")
    # Calculate strain from stress
    strain_from_stress = steel.strain_from_stress(stress_from_strain)
    print(f"\nStrain from stress (compliance):")
    print(f"Strain tensor:\n{strain_from_stress.tensor*1e6:.0f} (microstrain)")
    # Verify round-trip accuracy
    error = np.max(np.abs(strain.tensor - strain_from_stress.tensor))
    print(f"\nRound-trip error: {error*1e6:.2e} microstrain")
    # Display material matrices
    print(f"\nStiffness matrix (GPa):")
    stiffness = steel.stiffness_matrix / 1e9
    print(f"{stiffness}")
    print(f"\nCompliance matrix (1/TPa):")
    compliance = steel.compliance_matrix * 1e12
    print(f"{compliance}")
    return stress_from_strain
def analyze_elastic_wave_velocities(steel):
    """Analyze elastic wave velocities in material."""
    print("\n\nElastic Wave Velocities")
    print("=" * 25)
    v_p, v_s = steel.elastic_wave_velocities()
    print(f"Longitudinal wave velocity (P-wave): {v_p:.0f} m/s")
    print(f"Transverse wave velocity (S-wave): {v_s:.0f} m/s")
    print(f"Velocity ratio (vₚ/vₛ): {v_p/v_s:.2f}")
    # Calculate Poisson's ratio from wave velocities
    nu_from_waves = (v_p**2 - 2*v_s**2) / (2 * (v_p**2 - v_s**2))
    print(f"Poisson's ratio from wave velocities: {nu_from_waves:.3f}")
    print(f"Original Poisson's ratio: {steel.constants.poissons_ratio:.3f}")
    return v_p, v_s
def failure_analysis_example():
    """Demonstrate failure analysis using von Mises criterion."""
    print("\n\nFailure Analysis Example")
    print("=" * 30)
    # Material database
    materials = create_material_database()
    steel_props = materials['steel']
    print(f"Material: Steel")
    print(f"Yield strength: {steel_props.youngs_modulus/500/1e6:.0f} MPa (estimated)")
    # Create various stress states
    stress_states = [
        ("Uniaxial tension", StressTensor(sigma_xx=200e6)),
        ("Pure shear", StressTensor(sigma_xy=120e6)),
        ("Biaxial tension", StressTensor(sigma_xx=150e6, sigma_yy=100e6)),
        ("Triaxial stress", StressTensor(sigma_xx=180e6, sigma_yy=120e6, sigma_zz=80e6))
    ]
    yield_strength = 250e6  # Typical mild steel yield strength
    print(f"\nFailure Analysis Results:")
    print(f"{'Load Case':<20} {'von Mises':<12} {'Safety Factor':<15} {'Status'}")
    print("-" * 65)
    for name, stress in stress_states:
        von_mises = stress.von_mises_stress()
        safety_factor = yield_strength / von_mises
        status = "SAFE" if safety_factor > 1.0 else "FAILURE"
        print(f"{name:<20} {von_mises/1e6:>8.1f} MPa {safety_factor:>10.2f}     {status}")
    return stress_states
def plot_stress_analysis_results(stress_states):
    """Plot stress analysis results."""
    print("\n\nGenerating Stress Analysis Plots")
    print("-" * 35)
    # Extract data for plotting
    load_cases = [case[0] for case in stress_states]
    von_mises_stresses = [case[1].von_mises_stress()/1e6 for case in stress_states]
    yield_strength = 250  # MPa
    # Create figure with Berkeley styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot 1: von Mises stress comparison
    bars = ax1.bar(range(len(load_cases)), von_mises_stresses,
                   color=BERKELEY_BLUE, alpha=0.7, edgecolor='black')
    ax1.axhline(y=yield_strength, color=BERKELEY_GOLD, linestyle='--', linewidth=2,
                label=f'Yield Strength ({yield_strength} MPa)')
    ax1.set_xlabel('Load Case')
    ax1.set_ylabel('von Mises Stress (MPa)')
    ax1.set_title('von Mises Stress Analysis')
    ax1.set_xticks(range(len(load_cases)))
    ax1.set_xticklabels([case.replace(' ', '\n') for case in load_cases], rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom')
    # Plot 2: Safety factors
    safety_factors = [yield_strength/stress for stress in von_mises_stresses]
    colors = [BERKELEY_BLUE if sf > 1.0 else 'red' for sf in safety_factors]
    bars2 = ax2.bar(range(len(load_cases)), safety_factors,
                    color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color=BERKELEY_GOLD, linestyle='--', linewidth=2,
                label='Safety Limit (SF = 1.0)')
    ax2.set_xlabel('Load Case')
    ax2.set_ylabel('Safety Factor')
    ax2.set_title('Safety Factor Analysis')
    ax2.set_xticks(range(len(load_cases)))
    ax2.set_xticklabels([case.replace(' ', '\n') for case in load_cases], rotation=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    return fig
def demonstrate_coordinate_transformation():
    """Demonstrate stress tensor coordinate transformation."""
    print("\n\nCoordinate Transformation Example")
    print("=" * 40)
    # Original stress state in xy coordinate system
    stress_original = StressTensor(sigma_xx=100e6, sigma_yy=50e6, sigma_xy=30e6)
    print(f"Original stress tensor (MPa):")
    print(f"{stress_original.tensor/1e6}")
    # Rotation by 45 degrees about z-axis
    angle = np.pi/4  # 45 degrees
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    # Transform stress tensor
    stress_rotated = stress_original.transform(rotation_matrix)
    print(f"\nRotated stress tensor (45° rotation, MPa):")
    print(f"{stress_rotated.tensor/1e6}")
    # Principal stresses should remain invariant
    principal_orig, _ = stress_original.principal_stresses()
    principal_rot, _ = stress_rotated.principal_stresses()
    print(f"\nPrincipal stress invariance check:")
    print(f"Original: {principal_orig/1e6:.1f} MPa")
    print(f"Rotated:  {principal_rot/1e6:.1f} MPa")
    print(f"Difference: {np.max(np.abs(principal_orig - principal_rot))/1e6:.6f} MPa")
    return stress_original, stress_rotated
def energy_analysis_example(steel, stress, strain):
    """Demonstrate elastic energy calculations."""
    print("\n\nElastic Energy Analysis")
    print("=" * 25)
    # Calculate elastic energy density
    energy_density = steel.elastic_energy_density(stress, strain)
    print(f"Elastic energy density: {energy_density/1e3:.2f} kJ/m³")
    # Compare with theoretical calculation
    # U = (1/2E)[σ₁² + σ₂² + σ₃² - 2ν(σ₁σ₂ + σ₂σ₃ + σ₃σ₁)]
    principal_stresses, _ = stress.principal_stresses()
    s1, s2, s3 = principal_stresses
    E = steel.constants.youngs_modulus
    nu = steel.constants.poissons_ratio
    theoretical_energy = (1/(2*E)) * (s1**2 + s2**2 + s3**2 -
                                     2*nu*(s1*s2 + s2*s3 + s3*s1))
    print(f"Theoretical energy density: {theoretical_energy/1e3:.2f} kJ/m³")
    print(f"Relative error: {abs(energy_density - theoretical_energy)/theoretical_energy*100:.3f}%")
    return energy_density
def main():
    """Main stress-strain analysis demonstration."""
    print("Basic Stress-Strain Analysis")
    print("=" * 50)
    print("This example demonstrates fundamental concepts in elasticity theory")
    print("including stress/strain tensors, Hooke's law, and failure analysis.\n")
    # Create material
    steel = create_steel_material()
    # Demonstrate stress tensor operations
    stress_uniaxial, stress_shear, stress_complex = demonstrate_stress_tensor_operations()
    # Demonstrate strain tensor operations
    strain = demonstrate_strain_tensor_operations()
    # Apply Hooke's law
    stress_from_strain = demonstrate_hookes_law(steel, strain)
    # Analyze wave velocities
    v_p, v_s = analyze_elastic_wave_velocities(steel)
    # Coordinate transformation
    stress_orig, stress_rot = demonstrate_coordinate_transformation()
    # Energy analysis
    energy_density = energy_analysis_example(steel, stress_complex, strain)
    # Failure analysis
    stress_states = failure_analysis_example()
    # Generate plots
    fig = plot_stress_analysis_results(stress_states)
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("\nKey Results Summary:")
    print(f"• Steel Young's modulus: {steel.constants.youngs_modulus/1e9:.0f} GPa")
    print(f"• P-wave velocity: {v_p:.0f} m/s")
    print(f"• S-wave velocity: {v_s:.0f} m/s")
    print(f"• Elastic energy density: {energy_density/1e3:.1f} kJ/m³")
    print(f"• Maximum von Mises stress: {max([s[1].von_mises_stress() for s in stress_states])/1e6:.1f} MPa")
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Configure matplotlib for Berkeley style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    main()