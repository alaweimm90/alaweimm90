"""
Simple Truss Analysis - Beginner FEM Example
This example demonstrates basic finite element analysis of a simple truss structure.
It covers fundamental concepts including mesh creation, boundary conditions,
loading, and result visualization.
Learning Objectives:
- Understand basic FEM workflow
- Create simple 1D truss elements
- Apply boundary conditions and loads
- Interpret displacement and stress results
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add FEM package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.mesh_generation import Mesh
from core.assembly import GlobalAssembly
from core.solvers import StaticSolver
from core.post_processing import FEMPostProcessor
from utils.material_properties import MaterialLibrary
def main():
    """Run simple truss analysis example."""
    print("Simple Truss Analysis - Beginner FEM Example")
    print("=" * 50)
    print("This example analyzes a 3-member truss under point loading")
    print("Learning: Basic FEM workflow, boundary conditions, result interpretation\n")
    # Create materials
    steel = MaterialLibrary.steel_mild()
    print(f"Material: {steel.name}")
    print(f"Young's Modulus: {steel.youngs_modulus/1e9:.1f} GPa")
    print(f"Density: {steel.density} kg/m³\n")
    # Create simple truss mesh
    mesh = create_simple_truss()
    print(f"Mesh created: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    # Visualize mesh
    plot_truss_geometry(mesh)
    # Setup materials dictionary
    materials = {0: steel}  # All elements use material ID 0
    # Create assembly system
    assembly = GlobalAssembly(mesh, materials)
    # Assemble global matrices
    print("\nAssembling global matrices...")
    K_global = assembly.assemble_global_stiffness()
    M_global = assembly.assemble_global_mass()
    print(f"Global stiffness matrix: {K_global.shape}")
    print(f"Global mass matrix: {M_global.shape}")
    # Apply boundary conditions
    print("\nApplying boundary conditions...")
    boundary_conditions = {
        (0, 0): 0.0,  # Node 0, x-direction: fixed
        (3, 0): 0.0   # Node 3, x-direction: fixed
    }
    assembly.apply_boundary_conditions(boundary_conditions)
    print(f"Applied {len(boundary_conditions)} displacement constraints")
    # Apply loads
    print("\nApplying loads...")
    point_loads = {
        (1, 1): -10000.0,  # Node 1, y-direction: 10 kN downward
        (2, 1): -5000.0    # Node 2, y-direction: 5 kN downward
    }
    load_vector = assembly.apply_point_loads(point_loads)
    print(f"Applied {len(point_loads)} point loads")
    print(f"Total applied force: {np.sum(np.abs(list(point_loads.values()))):.0f} N")
    # Solve static problem
    print("\nSolving static equilibrium...")
    solver = StaticSolver(assembly)
    displacement = solver.solve_direct()
    # Analyze results
    analyze_results(assembly, displacement)
    # Post-process and visualize results
    post_processor = FEMPostProcessor(assembly, displacement)
    # Plot deformed shape
    fig_deformed = post_processor.plot_deformed_shape(scale_factor=1000, show_undeformed=True)
    plt.show()
    # Generate comprehensive results
    results_summary = post_processor.generate_results_summary()
    print_results_summary(results_summary)
    # Demonstrate stress calculations
    element_stresses = assembly.compute_element_stresses(displacement)
    print_stress_results(element_stresses, steel)
    print("\n" + "=" * 50)
    print("Truss Analysis Complete!")
    print("Key Learning Points:")
    print("• FEM converts continuous problems to discrete matrix equations")
    print("• Boundary conditions constrain the structure")
    print("• Loads create internal forces and deformations")
    print("• Results must be interpreted in engineering context")
def create_simple_truss():
    """Create a simple 3-member truss structure."""
    mesh = Mesh()
    mesh.dimension = 2  # 2D truss
    # Define nodes (x, y coordinates in meters)
    nodes = [
        (0.0, 0.0),    # Node 0: Left support
        (1.0, 1.0),    # Node 1: Top left
        (2.0, 1.0),    # Node 2: Top right
        (3.0, 0.0)     # Node 3: Right support
    ]
    # Add nodes to mesh
    for i, (x, y) in enumerate(nodes):
        mesh.add_node(np.array([x, y]), node_id=i)
    # Define elements (truss members)
    # Each element connects two nodes
    cross_section_area = 0.01  # 100 cm² = 0.01 m²
    elements = [
        (0, 1),  # Element 0: Left diagonal member
        (1, 2),  # Element 1: Top horizontal member
        (2, 3)   # Element 2: Right diagonal member
    ]
    # Add elements to mesh
    for i, (node1, node2) in enumerate(elements):
        mesh.add_element(
            element_type='bar1d',
            node_ids=[node1, node2],
            material_id=0,
            element_id=i,
            cross_section_area=cross_section_area
        )
    return mesh
def plot_truss_geometry(mesh):
    """Plot the truss geometry."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot elements
    for element in mesh.elements.values():
        node_coords = np.array([mesh.nodes[nid].coordinates for nid in element.node_ids])
        ax.plot(node_coords[:, 0], node_coords[:, 1], 'b-', linewidth=3,
               label='Truss Member' if element.id == 0 else "")
    # Plot nodes
    coords = mesh.get_node_coordinates()
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5,
              edgecolors='black', label='Joints')
    # Add node labels
    for node_id, node in mesh.nodes.items():
        ax.annotate(f'Node {node_id}', (node.coordinates[0], node.coordinates[1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    # Add support symbols
    support_size = 0.1
    # Left support (pin)
    ax.plot([0, 0], [-support_size, 0], 'k-', linewidth=4)
    ax.plot([-support_size, support_size], [-support_size, -support_size], 'k-', linewidth=2)
    # Right support (roller)
    ax.plot([3, 3], [-support_size, 0], 'k-', linewidth=4)
    circle = plt.Circle((3, -support_size/2), support_size/3, fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Simple Truss Geometry')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
def analyze_results(assembly, displacement):
    """Analyze and display key results."""
    print("\nDisplacement Results:")
    print("-" * 30)
    for node_id in sorted(assembly.mesh.nodes.keys()):
        node_disp = assembly.get_displacement_at_node(displacement, node_id)
        disp_magnitude = np.linalg.norm(node_disp)
        print(f"Node {node_id}: dx = {node_disp[0]*1000:7.3f} mm, "
              f"dy = {node_disp[1]*1000:7.3f} mm, "
              f"|d| = {disp_magnitude*1000:7.3f} mm")
    # Find maximum displacement
    max_disp = np.max(np.abs(displacement))
    max_disp_dof = np.argmax(np.abs(displacement))
    max_disp_node, max_disp_local = assembly.global_dof_to_node[max_disp_dof]
    print(f"\nMaximum displacement: {max_disp*1000:.3f} mm")
    print(f"Location: Node {max_disp_node}, {'X' if max_disp_local == 0 else 'Y'} direction")
    # Calculate reaction forces
    reactions = assembly.compute_reaction_forces(displacement)
    print("\nReaction Forces:")
    print("-" * 20)
    total_reaction = np.array([0.0, 0.0])
    for node_id, reaction in reactions.items():
        print(f"Node {node_id}: Rx = {reaction[0]:8.1f} N, Ry = {reaction[1]:8.1f} N")
        total_reaction += reaction
    print(f"Total reaction: Rx = {total_reaction[0]:8.1f} N, Ry = {total_reaction[1]:8.1f} N")
    # Verify equilibrium
    applied_loads = np.array([0.0, -15000.0])  # Total applied loads
    equilibrium_error = np.linalg.norm(total_reaction + applied_loads)
    print(f"Equilibrium check (should be ~0): {equilibrium_error:.2e} N")
def print_results_summary(results):
    """Print comprehensive results summary."""
    print("\nResults Summary:")
    print("=" * 30)
    print(f"Mesh: {results['mesh_info']['num_nodes']} nodes, "
          f"{results['mesh_info']['num_elements']} elements")
    print(f"Max displacement: {results['displacement_stats']['max_displacement']*1000:.3f} mm")
    print(f"RMS displacement: {results['displacement_stats']['rms_displacement']*1000:.3f} mm")
    if 'energy_info' in results:
        print(f"Strain energy: {results['energy_info']['strain_energy']:.2f} J")
        print(f"Total potential energy: {results['energy_info']['total_potential']:.2f} J")
    if 'stress_stats' in results and results['stress_stats']:
        print(f"Max stress: {results['stress_stats']['max_von_mises']/1e6:.2f} MPa")
        print(f"Average stress: {results['stress_stats']['avg_von_mises']/1e6:.2f} MPa")
def print_stress_results(element_stresses, material):
    """Print element stress results."""
    print("\nElement Stress Results:")
    print("-" * 30)
    yield_strength = 250e6  # Typical mild steel yield strength (Pa)
    for element_id, stress in element_stresses.items():
        stress_value = stress[0]  # Axial stress for truss elements
        stress_mpa = stress_value / 1e6
        safety_factor = yield_strength / abs(stress_value) if abs(stress_value) > 1e-6 else float('inf')
        stress_type = "Tension" if stress_value > 0 else "Compression"
        print(f"Element {element_id}: {stress_mpa:8.2f} MPa ({stress_type}), "
              f"Safety Factor: {safety_factor:.1f}")
    print(f"\nMaterial yield strength: {yield_strength/1e6:.0f} MPa")
    print("Safety Factor > 1.0 indicates safe design")
if __name__ == "__main__":
    main()