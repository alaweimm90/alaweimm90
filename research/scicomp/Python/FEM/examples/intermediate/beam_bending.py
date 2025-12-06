"""
Beam Bending Analysis - Intermediate FEM Example
This example demonstrates finite element analysis of beam bending using 2D plane stress elements.
It covers mesh generation, stress analysis, convergence studies, and comparison with analytical solutions.
Learning Objectives:
- Understand 2D finite element modeling
- Perform mesh convergence studies
- Compare FEM results with analytical beam theory
- Analyze stress distributions and concentrations
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add FEM package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.mesh_generation import Mesh, StructuredMeshGenerator
from core.assembly import GlobalAssembly
from core.solvers import StaticSolver
from core.post_processing import FEMPostProcessor
from utils.material_properties import MaterialLibrary
def main():
    """Run beam bending analysis example."""
    print("Beam Bending Analysis - Intermediate FEM Example")
    print("=" * 55)
    print("This example analyzes a cantilever beam under end loading")
    print("Learning: 2D elements, mesh convergence, analytical comparison\n")
    # Beam geometry and material properties
    beam_length = 1.0  # m
    beam_height = 0.1  # m
    beam_thickness = 0.01  # m
    end_load = 1000.0  # N
    steel = MaterialLibrary.steel_mild()
    print(f"Material: {steel.name}")
    print(f"Young's Modulus: {steel.youngs_modulus/1e9:.1f} GPa")
    print(f"Poisson's Ratio: {steel.poissons_ratio}")
    print(f"\nBeam Geometry:")
    print(f"Length: {beam_length} m")
    print(f"Height: {beam_height} m")
    print(f"Thickness: {beam_thickness} m")
    print(f"End Load: {end_load} N\n")
    # Analytical solution for comparison
    analytical_results = calculate_analytical_solution(beam_length, beam_height, beam_thickness,
                                                     end_load, steel.youngs_modulus)
    print("Analytical Solution (Euler-Bernoulli Beam Theory):")
    print(f"Maximum displacement: {analytical_results['max_displacement']*1000:.3f} mm")
    print(f"Maximum stress: {analytical_results['max_stress']/1e6:.2f} MPa\n")
    # Perform convergence study
    print("Performing mesh convergence study...")
    convergence_results = perform_convergence_study(beam_length, beam_height, beam_thickness,
                                                   end_load, steel, analytical_results)
    # Detailed analysis with fine mesh
    print("\nDetailed Analysis with Fine Mesh:")
    print("-" * 40)
    # Create fine mesh for detailed analysis
    mesh_fine = create_beam_mesh(beam_length, beam_height, nx=40, ny=8, element_type='quad2d')
    materials = {0: steel}
    # Solve with fine mesh
    assembly_fine = GlobalAssembly(mesh_fine, materials)
    displacement_fine, assembly_fine = solve_beam_problem(assembly_fine, beam_thickness, end_load)
    # Post-process results
    post_processor = FEMPostProcessor(assembly_fine, displacement_fine)
    # Generate plots
    create_analysis_plots(post_processor, convergence_results, analytical_results,
                         beam_length, beam_height)
    # Detailed stress analysis
    perform_stress_analysis(assembly_fine, displacement_fine, steel, analytical_results)
    print("\n" + "=" * 55)
    print("Beam Bending Analysis Complete!")
    print("Key Learning Points:")
    print("• Mesh density affects solution accuracy")
    print("• FEM converges to analytical solution with refinement")
    print("• Stress concentrations occur at geometric discontinuities")
    print("• Element choice impacts accuracy and computational cost")
def create_beam_mesh(length, height, nx, ny, element_type='quad2d'):
    """Create structured mesh for cantilever beam."""
    mesh_generator = StructuredMeshGenerator()
    mesh = mesh_generator.generate_rectangle_mesh(length, height, nx, ny, element_type)
    # Set element thickness property
    for element in mesh.elements.values():
        element.thickness = 0.01  # Will be set properly in assembly
    return mesh
def solve_beam_problem(assembly, thickness, end_load):
    """Solve the beam bending problem."""
    # Set element thickness
    for element in assembly.mesh.elements.values():
        element.thickness = thickness
    # Assemble matrices
    K_global = assembly.assemble_global_stiffness()
    M_global = assembly.assemble_global_mass()
    # Apply boundary conditions (fixed at left end)
    boundary_conditions = {}
    # Find nodes at x = 0 (left end)
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0]) < 1e-10:  # At x = 0
            boundary_conditions[(node_id, 0)] = 0.0  # Fix x-displacement
            boundary_conditions[(node_id, 1)] = 0.0  # Fix y-displacement
    assembly.apply_boundary_conditions(boundary_conditions)
    # Apply end load (distributed over right end nodes)
    point_loads = {}
    right_end_nodes = []
    # Find nodes at right end (x = length)
    coords = assembly.mesh.get_node_coordinates()
    max_x = np.max(coords[:, 0])
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0] - max_x) < 1e-10:  # At right end
            right_end_nodes.append(node_id)
    # Distribute load equally among right end nodes
    load_per_node = -end_load / len(right_end_nodes)  # Negative for downward
    for node_id in right_end_nodes:
        point_loads[(node_id, 1)] = load_per_node  # y-direction
    load_vector = assembly.apply_point_loads(point_loads)
    # Solve
    solver = StaticSolver(assembly)
    displacement = solver.solve_direct()
    return displacement, assembly
def calculate_analytical_solution(length, height, thickness, load, E):
    """Calculate analytical solution using Euler-Bernoulli beam theory."""
    # Second moment of area
    I = thickness * height**3 / 12
    # Maximum displacement (at free end)
    max_displacement = load * length**3 / (3 * E * I)
    # Maximum stress (at fixed end, top/bottom fibers)
    max_moment = load * length
    max_stress = max_moment * (height/2) / I
    return {
        'max_displacement': max_displacement,
        'max_stress': max_stress,
        'moment_of_inertia': I
    }
def perform_convergence_study(length, height, thickness, load, material, analytical):
    """Perform mesh convergence study."""
    mesh_sizes = [(5, 2), (10, 4), (20, 4), (40, 8), (80, 16)]
    element_types = ['triangle2d', 'quad2d']
    convergence_data = {
        'triangle2d': {'elements': [], 'displacement': [], 'stress': []},
        'quad2d': {'elements': [], 'displacement': [], 'stress': []}
    }
    for element_type in element_types:
        print(f"\nConvergence study for {element_type} elements:")
        print("Elements | Max Disp (mm) | Max Stress (MPa) | Disp Error (%) | Stress Error (%)")
        print("-" * 80)
        for nx, ny in mesh_sizes:
            # Create mesh
            mesh = create_beam_mesh(length, height, nx, ny, element_type)
            materials = {0: material}
            # Solve
            assembly = GlobalAssembly(mesh, materials)
            displacement, assembly = solve_beam_problem(assembly, thickness, load)
            # Calculate maximum displacement and stress
            max_disp = np.max(np.abs(displacement))
            element_stresses = assembly.compute_element_stresses(displacement)
            max_stress = 0.0
            for stress in element_stresses.values():
                if len(stress) >= 3:  # 2D stress state
                    von_mises = np.sqrt(stress[0]**2 + stress[1]**2 - stress[0]*stress[1] + 3*stress[2]**2)
                    max_stress = max(max_stress, von_mises)
            # Calculate errors
            disp_error = abs(max_disp - analytical['max_displacement']) / analytical['max_displacement'] * 100
            stress_error = abs(max_stress - analytical['max_stress']) / analytical['max_stress'] * 100
            # Store results
            num_elements = len(mesh.elements)
            convergence_data[element_type]['elements'].append(num_elements)
            convergence_data[element_type]['displacement'].append(max_disp)
            convergence_data[element_type]['stress'].append(max_stress)
            print(f"{num_elements:8d} | {max_disp*1000:11.3f} | {max_stress/1e6:14.2f} | "
                  f"{disp_error:10.2f} | {stress_error:12.2f}")
    return convergence_data
def create_analysis_plots(post_processor, convergence_data, analytical, length, height):
    """Create comprehensive analysis plots."""
    # Plot 1: Convergence study
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Displacement convergence
    for element_type, data in convergence_data.items():
        ax1.loglog(data['elements'], np.array(data['displacement'])*1000, 'o-',
                  label=f'{element_type}', linewidth=2, markersize=6)
    ax1.axhline(analytical['max_displacement']*1000, color='red', linestyle='--',
               label='Analytical', linewidth=2)
    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel('Maximum Displacement (mm)')
    ax1.set_title('Displacement Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Stress convergence
    for element_type, data in convergence_data.items():
        ax2.loglog(data['elements'], np.array(data['stress'])/1e6, 'o-',
                  label=f'{element_type}', linewidth=2, markersize=6)
    ax2.axhline(analytical['max_stress']/1e6, color='red', linestyle='--',
               label='Analytical', linewidth=2)
    ax2.set_xlabel('Number of Elements')
    ax2.set_ylabel('Maximum Stress (MPa)')
    ax2.set_title('Stress Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    # Plot 2: Deformed shape
    fig_deformed = post_processor.plot_deformed_shape(scale_factor=10000, show_undeformed=True)
    plt.show()
    # Plot 3: Stress contours
    fig_stress = post_processor.plot_stress_contours('von_mises')
    plt.show()
    # Plot 4: Displacement vectors
    fig_vectors = post_processor.plot_displacement_vectors(scale_factor=5000)
    plt.show()
def perform_stress_analysis(assembly, displacement, material, analytical):
    """Perform detailed stress analysis."""
    print("\nDetailed Stress Analysis:")
    print("-" * 30)
    element_stresses = assembly.compute_element_stresses(displacement)
    # Find maximum stresses
    max_von_mises = 0.0
    max_principal = 0.0
    stress_data = []
    for element_id, stress in element_stresses.items():
        if len(stress) >= 3:  # 2D stress state
            sxx, syy, sxy = stress[0], stress[1], stress[2]
            # von Mises stress
            von_mises = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)
            max_von_mises = max(max_von_mises, von_mises)
            # Principal stresses
            s_mean = (sxx + syy) / 2
            s_diff = (sxx - syy) / 2
            s1 = s_mean + np.sqrt(s_diff**2 + sxy**2)  # Maximum principal stress
            s2 = s_mean - np.sqrt(s_diff**2 + sxy**2)  # Minimum principal stress
            max_principal = max(max_principal, abs(s1), abs(s2))
            stress_data.append({
                'element_id': element_id,
                'von_mises': von_mises,
                'principal_max': s1,
                'principal_min': s2,
                'normal_x': sxx,
                'normal_y': syy,
                'shear_xy': sxy
            })
    print(f"Maximum von Mises stress: {max_von_mises/1e6:.2f} MPa")
    print(f"Analytical maximum stress: {analytical['max_stress']/1e6:.2f} MPa")
    print(f"Error: {abs(max_von_mises - analytical['max_stress'])/analytical['max_stress']*100:.2f}%")
    # Material safety analysis
    yield_strength = 250e6  # Typical mild steel yield strength
    safety_factor = yield_strength / max_von_mises
    print(f"\nSafety Analysis:")
    print(f"Material yield strength: {yield_strength/1e6:.0f} MPa")
    print(f"Safety factor: {safety_factor:.1f}")
    if safety_factor > 2.0:
        print("Design is safe with good margin")
    elif safety_factor > 1.0:
        print("Design is safe but margin is low")
    else:
        print("WARNING: Design exceeds material limits!")
    # Stress distribution analysis
    analyze_stress_distribution(stress_data, assembly.mesh)
def analyze_stress_distribution(stress_data, mesh):
    """Analyze stress distribution along beam."""
    print("\nStress Distribution Analysis:")
    print("-" * 35)
    # Group elements by x-coordinate (along beam length)
    x_positions = []
    avg_stresses = []
    # Get element centroids
    coords = mesh.get_node_coordinates()
    # Create bins along beam length
    x_bins = np.linspace(0, 1.0, 11)  # 10 sections along beam
    for i in range(len(x_bins)-1):
        x_start, x_end = x_bins[i], x_bins[i+1]
        x_center = (x_start + x_end) / 2
        # Find elements in this section
        section_stresses = []
        for stress_info in stress_data:
            element = mesh.elements[stress_info['element_id']]
            # Calculate element centroid
            element_coords = [coords[nid] for nid in element.node_ids]
            centroid_x = np.mean([coord[0] for coord in element_coords])
            if x_start <= centroid_x < x_end:
                section_stresses.append(stress_info['von_mises'])
        if section_stresses:
            avg_stress = np.mean(section_stresses)
            x_positions.append(x_center)
            avg_stresses.append(avg_stress)
            print(f"x = {x_center:.2f} m: Average stress = {avg_stress/1e6:.2f} MPa")
    # Plot stress distribution
    if x_positions:
        plt.figure(figsize=(10, 6))
        plt.plot(x_positions, np.array(avg_stresses)/1e6, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Position along beam (m)')
        plt.ylabel('Average von Mises Stress (MPa)')
        plt.title('Stress Distribution along Cantilever Beam')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()