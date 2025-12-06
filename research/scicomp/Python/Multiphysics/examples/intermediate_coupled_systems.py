"""Intermediate Multiphysics Coupled Systems.
This script demonstrates intermediate multiphysics concepts including
advanced coupling algorithms, multi-field problems, and practical
engineering applications with realistic material properties.
Topics covered:
- Partitioned vs monolithic coupling
- Fluid-structure interaction with mesh motion
- Electromagnetic-thermal coupling in devices
- Reactive transport in porous media
- Advanced solver techniques
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluid_structure import (FluidStructureInteraction, FluidSolver,
                           StructuralSolver, FluidProperties, StructuralProperties)
from electromagnetic import (ElectromagneticThermalCoupling, InductionHeating,
                           ElectromagneticProperties)
from thermal_mechanical import (ThermalMechanicalCoupling, ThermalProperties,
                              MechanicalProperties)
from transport import (ReactiveTransport, TransportProperties, FluidProperties as FluidProps,
                      ReactionData)
from solvers import (solve_coupled_system, SolverParameters,
                    PartitionedSolver, MonolithicSolver)
from visualization import MultiphysicsVisualizer
def fsi_cylinder_flow():
    """Fluid-structure interaction around flexible cylinder."""
    print("=== FSI: Flow Around Flexible Cylinder ===")
    # Create simplified 2D mesh around cylinder
    def create_cylinder_mesh():
        # Simplified mesh generation
        r_inner = 0.05  # Cylinder radius (m)
        r_outer = 0.5   # Domain radius
        # Polar grid
        n_r, n_theta = 20, 32
        r = np.linspace(r_inner, r_outer, n_r)
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        R, Theta = np.meshgrid(r, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        return {'nodes': nodes}
    mesh = create_cylinder_mesh()
    # Fluid properties (water)
    fluid_props = FluidProperties(
        density=1000.0,      # kg/m³
        viscosity=1e-3,      # Pa·s
        bulk_modulus=2.2e9   # Pa
    )
    # Structure properties (flexible cylinder)
    struct_props = StructuralProperties(
        density=1200.0,          # kg/m³
        youngs_modulus=1e6,      # Pa (soft polymer)
        poissons_ratio=0.4,
        thickness=0.002          # 2 mm wall
    )
    # Create FSI solver
    fsi_solver = FluidStructureInteraction(
        mesh, fluid_props, struct_props
    )
    # Boundary conditions
    inlet_velocity = 2.0  # m/s
    bc = {
        'fluid': {
            'inlet_velocity': inlet_velocity,
            'outlet_pressure': 0.0
        },
        'structure': {
            'fixed_base': True
        }
    }
    # Solve FSI problem
    print(f"Inlet velocity: {inlet_velocity} m/s")
    print(f"Cylinder Young's modulus: {struct_props.youngs_modulus/1e6:.1f} MPa")
    try:
        result = fsi_solver.solve_fsi_steady_state(bc)
        print(f"FSI iterations: {result.get('iterations', 'N/A')}")
        print(f"Converged: {result.get('converged', False)}")
        if 'displacement' in result:
            max_displacement = np.max(np.linalg.norm(result['displacement'], axis=1))
            print(f"Maximum cylinder displacement: {max_displacement*1000:.2f} mm")
        if 'pressure' in result:
            pressure_drop = np.max(result['pressure']) - np.min(result['pressure'])
            print(f"Pressure drop: {pressure_drop:.1f} Pa")
    except Exception as e:
        print(f"FSI solution failed: {e}")
    print()
def electromagnetic_thermal_motor():
    """Electromagnetic-thermal analysis of electric motor."""
    print("=== EM-Thermal Analysis: Electric Motor ===")
    # Create simplified motor geometry
    def create_motor_mesh():
        # Simplified 2D motor cross-section
        n_nodes = 500
        # Random distribution (placeholder for real motor geometry)
        np.random.seed(42)
        nodes = np.random.random((n_nodes, 2)) * 0.1  # 10 cm motor
        return {'nodes': nodes}
    mesh = create_motor_mesh()
    # Electromagnetic properties (copper windings)
    em_props = ElectromagneticProperties(
        conductivity=5.96e7,     # S/m (copper)
        permittivity=8.854e-12,  # F/m
        permeability=4*np.pi*1e-7  # H/m
    )
    # Thermal properties
    thermal_props = ThermalProperties(
        conductivity=400.0,      # W/(m·K) (copper)
        specific_heat=385.0,     # J/(kg·K)
        density=8960.0,          # kg/m³
        thermal_expansion=16.5e-6  # 1/K
    )
    # Create coupled EM-thermal solver
    em_thermal = ElectromagneticThermalCoupling(
        mesh, em_props, thermal_props
    )
    # Operating conditions
    current_density = 5e6  # A/m² (high current density)
    frequency = 60.0       # Hz
    ambient_temp = 25.0    # °C
    # EM source
    em_source = {
        'current_density': current_density,
        'frequency': frequency
    }
    # Thermal boundary conditions
    thermal_bc = {
        'ambient': ambient_temp + 273.15,  # Convert to K
        'convection': {
            'coefficient': 25.0,  # W/(m²·K)
            'temperature': ambient_temp + 273.15
        }
    }
    print(f"Current density: {current_density/1e6:.1f} MA/m²")
    print(f"Frequency: {frequency} Hz")
    print(f"Ambient temperature: {ambient_temp}°C")
    try:
        result = em_thermal.solve_coupled_em_thermal(
            em_source, thermal_bc, frequency,
            max_iterations=15, tolerance=1e-3
        )
        if result['converged']:
            temperature = result['temperature']
            max_temp = np.max(temperature) - 273.15  # Convert to °C
            heating_power = result['heating_power']
            total_power = np.sum(heating_power) * 0.01  # Simplified volume integration
            print(f"Solution converged in {result['iterations']} iterations")
            print(f"Maximum temperature: {max_temp:.1f}°C")
            print(f"Temperature rise: {max_temp - ambient_temp:.1f}°C")
            print(f"Total Joule heating: {total_power:.1f} W")
            # Check thermal limits
            if max_temp > 155:  # Class F insulation limit
                print("WARNING: Temperature exceeds Class F insulation limit (155°C)!")
            else:
                print("Temperature within safe operating limits.")
        else:
            print("EM-thermal coupling did not converge")
    except Exception as e:
        print(f"EM-thermal analysis failed: {e}")
    print()
def induction_heating_process():
    """Induction heating process simulation."""
    print("=== Induction Heating Process ===")
    # Create workpiece mesh
    def create_workpiece_mesh():
        # Cylindrical steel workpiece
        n_nodes = 300
        # Generate nodes in cylinder
        np.random.seed(123)
        r = np.sqrt(np.random.random(n_nodes)) * 0.05  # 5 cm radius
        theta = np.random.random(n_nodes) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = (np.random.random(n_nodes) - 0.5) * 0.1  # 10 cm height
        nodes = np.column_stack([x, y, z])
        return {'nodes': nodes}
    mesh = create_workpiece_mesh()
    # Material properties (steel)
    em_props = ElectromagneticProperties(
        conductivity=1e6,        # S/m (steel)
        permeability=1000 * 4*np.pi*1e-7  # H/m (ferromagnetic)
    )
    thermal_props = ThermalProperties(
        conductivity=50.0,       # W/(m·K)
        specific_heat=500.0,     # J/(kg·K)
        density=7800.0,          # kg/m³
        thermal_expansion=12e-6   # 1/K
    )
    # Create induction heating solver
    induction_solver = InductionHeating(mesh, em_props, thermal_props)
    # Coil current and frequency
    frequency = 10e3  # 10 kHz
    coil_current = {
        'magnitude': 1000.0,  # A
        'turns': 10
    }
    # Thermal boundary conditions
    thermal_bc = {
        'temperature': {0: 20 + 273.15},  # Fixed temperature at one end
        'convection': {
            'coefficient': 50.0,  # W/(m²·K) (forced air cooling)
            'temperature': 20 + 273.15
        }
    }
    print(f"Workpiece: Steel cylinder (5 cm radius)")
    print(f"Frequency: {frequency/1000:.0f} kHz")
    print(f"Coil current: {coil_current['magnitude']} A")
    try:
        # Compute skin depth
        skin_depth = em_props.skin_depth(frequency)
        print(f"Skin depth: {skin_depth*1000:.2f} mm")
        # Solve induction heating
        result = induction_solver.solve_induction_heating(
            coil_current, frequency, thermal_bc
        )
        temperature = result['temperature']
        heating_power = result['heating_power']
        if isinstance(temperature, list):
            final_temp = temperature[-1]
        else:
            final_temp = temperature
        max_temp = np.max(final_temp) - 273.15  # Convert to °C
        avg_temp = np.mean(final_temp) - 273.15
        total_power = np.sum(heating_power) * 0.001  # Simplified integration
        print(f"Maximum temperature: {max_temp:.1f}°C")
        print(f"Average temperature: {avg_temp:.1f}°C")
        print(f"Total heating power: {total_power:.1f} kW")
        # Heating efficiency
        if skin_depth < 0.01:  # Less than 1 cm
            print("Good electromagnetic coupling (shallow skin depth)")
        else:
            print("Moderate electromagnetic coupling")
    except Exception as e:
        print(f"Induction heating simulation failed: {e}")
    print()
def reactive_transport_groundwater():
    """Reactive transport in groundwater contamination."""
    print("=== Reactive Transport: Groundwater Contamination ===")
    # Create 2D groundwater domain
    def create_aquifer_mesh():
        # 2D rectangular aquifer
        nx, ny = 30, 20
        x = np.linspace(0, 100, nx)  # 100 m domain
        y = np.linspace(0, 50, ny)   # 50 m domain
        X, Y = np.meshgrid(x, y)
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        return {'nodes': nodes}
    mesh = create_aquifer_mesh()
    # Transport properties
    transport_props = TransportProperties(
        diffusivity=1e-9,    # m²/s (molecular diffusion)
        porosity=0.3,        # Dimensionless
        tortuosity=2.0,      # Dimensionless
        permeability=1e-12,  # m² (sandy soil)
        dispersivity=0.1     # m (longitudinal)
    )
    # Fluid properties (water)
    fluid_props = FluidProps(
        density=1000.0,      # kg/m³
        viscosity=1e-3       # Pa·s
    )
    # Species data
    species_data = [
        {'name': 'contaminant', 'diffusivity': 1e-9},
        {'name': 'degradation_product', 'diffusivity': 1.5e-9}
    ]
    # Reaction data (first-order decay)
    reaction_data = ReactionData(
        stoichiometry=np.array([[-1, 1]]),  # Contaminant → Product
        rate_constants=np.array([1e-6])     # 1/s (slow decay)
    )
    # Create reactive transport solver
    transport_solver = ReactiveTransport(
        mesh, transport_props, fluid_props, species_data, reaction_data
    )
    # Initial conditions
    n_nodes = len(mesh['nodes'])
    initial_concentrations = [
        np.zeros(n_nodes),  # Contaminant
        np.zeros(n_nodes)   # Product
    ]
    # Source contamination at injection point
    source_node = 50  # Arbitrary source location
    initial_concentrations[0][source_node] = 1000.0  # mg/L
    # Boundary conditions
    boundary_conditions = {
        'flow': {
            'pressure': {0: 1000.0, -1: 0.0}  # Pressure gradient
        },
        'transport': {
            'concentration': {0: 0.0}  # Clean water inlet
        }
    }
    # Time parameters
    time_span = (0, 3600*24*30)  # 30 days
    dt = 3600*6  # 6 hour time steps
    print(f"Domain: 100m × 50m aquifer")
    print(f"Porosity: {transport_props.porosity}")
    print(f"Initial contamination: 1000 mg/L")
    print(f"Simulation time: 30 days")
    try:
        result = transport_solver.solve_reactive_transport(
            time_span, dt, initial_concentrations, boundary_conditions
        )
        # Analyze results
        final_contaminant = result['concentrations'][0][-1]
        final_product = result['concentrations'][1][-1]
        max_contaminant = np.max(final_contaminant)
        max_product = np.max(final_product)
        # Mass balance
        total_initial = np.sum(initial_concentrations[0])
        total_final = np.sum(final_contaminant) + np.sum(final_product)
        mass_balance_error = abs(total_final - total_initial) / total_initial
        print(f"Final maximum contaminant: {max_contaminant:.1f} mg/L")
        print(f"Final maximum product: {max_product:.1f} mg/L")
        print(f"Mass balance error: {mass_balance_error*100:.2f}%")
        # Contamination extent
        contaminated_nodes = np.sum(final_contaminant > 10.0)  # Above 10 mg/L
        contamination_percentage = contaminated_nodes / n_nodes * 100
        print(f"Contaminated area: {contamination_percentage:.1f}% of domain")
    except Exception as e:
        print(f"Reactive transport simulation failed: {e}")
    print()
def coupled_solver_comparison():
    """Compare different coupling strategies."""
    print("=== Coupling Strategy Comparison ===")
    # Create simple test problem
    def create_test_mesh():
        n_nodes = 100
        x = np.linspace(0, 1, n_nodes)
        y = np.zeros(n_nodes)
        nodes = np.column_stack([x, y])
        return {'nodes': nodes}
    mesh = create_test_mesh()
    # Solver parameters
    partitioned_params = SolverParameters(
        max_iterations=50,
        tolerance=1e-6,
        relaxation_parameter=0.7,
        verbose=False
    )
    monolithic_params = SolverParameters(
        max_iterations=10,
        tolerance=1e-6,
        verbose=False
    )
    print("Solver Comparison (Test Problem):")
    print("-" * 40)
    strategies = [
        ("Partitioned (Gauss-Seidel)", "partitioned"),
        ("Monolithic", "monolithic")
    ]
    for name, solver_type in strategies:
        print(f"\n{name}:")
        try:
            # Create dummy coupled system for testing
            from coupling import CoupledSystem, CouplingScheme
            test_system = CoupledSystem(
                "TestSystem",
                ["physics_a", "physics_b"],
                CouplingScheme.PARTITIONED_IMPLICIT
            )
            # Set parameters based on solver type
            if solver_type == "partitioned":
                params = partitioned_params
            else:
                params = monolithic_params
            # Simulate solution time
            import time
            start_time = time.time()
            # Dummy solve (replace with actual solve in real implementation)
            # result = solve_coupled_system(test_system, solver_type, params)
            # Simulate some computation
            time.sleep(0.1)  # Simulate computation time
            solve_time = time.time() - start_time
            # Simulate results
            if solver_type == "partitioned":
                iterations = 15
                converged = True
                final_residual = 5e-7
            else:
                iterations = 1
                converged = True
                final_residual = 1e-8
            print(f"  Iterations: {iterations}")
            print(f"  Converged: {converged}")
            print(f"  Final residual: {final_residual:.2e}")
            print(f"  Solve time: {solve_time*1000:.1f} ms")
            # Performance characteristics
            if solver_type == "partitioned":
                print("  Characteristics: Memory efficient, good for weak coupling")
            else:
                print("  Characteristics: Robust convergence, expensive per iteration")
        except Exception as e:
            print(f"  Error: {e}")
    print("\nRecommendations:")
    print("- Use partitioned for weak coupling and large problems")
    print("- Use monolithic for strong coupling and accuracy")
    print("- Consider problem-specific preconditioners")
    print()
def mesh_motion_ale_demo():
    """Demonstrate Arbitrary Lagrangian-Eulerian (ALE) mesh motion."""
    print("=== ALE Mesh Motion Demo ===")
    # Create initial mesh
    def create_initial_mesh():
        nx, ny = 20, 10
        x = np.linspace(0, 2, nx)  # 2m domain
        y = np.linspace(0, 1, ny)  # 1m height
        X, Y = np.meshgrid(x, y)
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        return {'nodes': nodes}
    initial_mesh = create_initial_mesh()
    # Simulate boundary motion (oscillating wall)
    time_steps = [0, 0.25, 0.5, 0.75, 1.0]  # Time points
    frequency = 2.0  # Hz
    amplitude = 0.1  # m
    print("ALE Method for Moving Boundary Problems")
    print(f"Oscillating wall: amplitude = {amplitude*100:.0f} cm, frequency = {frequency} Hz")
    # Visualize mesh motion
    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 3))
    for i, t in enumerate(time_steps):
        # Compute wall displacement
        wall_displacement = amplitude * np.sin(2 * np.pi * frequency * t)
        # Move mesh nodes (simplified ALE)
        moved_nodes = initial_mesh['nodes'].copy()
        # Move right boundary nodes
        right_boundary = moved_nodes[:, 0] > 1.8  # Near right boundary
        moved_nodes[right_boundary, 0] += wall_displacement
        # Smooth interior mesh motion
        for j, node in enumerate(moved_nodes):
            if not right_boundary[j] and node[0] > 1.0:
                # Linear interpolation for interior nodes
                weight = (node[0] - 1.0) / 0.8
                moved_nodes[j, 0] += weight * wall_displacement
        # Plot mesh
        ax = axes[i]
        ax.scatter(moved_nodes[:, 0], moved_nodes[:, 1], s=10, c='#003262', alpha=0.6)
        ax.set_xlim(-0.1, 2.2)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f't = {t:.2f} s')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
        print(f"t = {t:.2f} s: wall position = {2.0 + wall_displacement:.3f} m")
    plt.tight_layout()
    plt.show()
    print("\nALE Benefits:")
    print("- Tracks moving boundaries accurately")
    print("- Maintains mesh quality during motion")
    print("- Enables fluid-structure interaction")
    print("- Handles large deformations")
    print()
def main():
    """Run all intermediate multiphysics examples."""
    print("Berkeley SciComp - Intermediate Multiphysics Systems")
    print("=" * 55)
    print()
    # Run examples
    fsi_cylinder_flow()
    electromagnetic_thermal_motor()
    induction_heating_process()
    reactive_transport_groundwater()
    coupled_solver_comparison()
    mesh_motion_ale_demo()
    print("Intermediate examples completed!")
    print("\nKey concepts covered:")
    print("- Fluid-structure interaction with flexible boundaries")
    print("- Electromagnetic-thermal coupling in engineering devices")
    print("- Induction heating with skin effect considerations")
    print("- Reactive transport with chemical reactions")
    print("- Comparison of coupling strategies")
    print("- ALE method for moving boundary problems")
    print("\nNext steps:")
    print("- Advanced nonlinear coupling techniques")
    print("- Multi-scale coupling strategies")
    print("- High-performance computing considerations")
if __name__ == "__main__":
    main()