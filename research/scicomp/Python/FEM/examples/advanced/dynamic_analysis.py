"""
Dynamic Analysis - Advanced FEM Example
This example demonstrates advanced finite element analysis including modal analysis,
time-domain response, and frequency response analysis of structures.
Learning Objectives:
- Understand dynamic finite element formulations
- Perform modal analysis to find natural frequencies and mode shapes
- Analyze transient response to time-varying loads
- Compute frequency response functions
- Study damping effects and resonance phenomena
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
# Add FEM package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.mesh_generation import Mesh, StructuredMeshGenerator
from core.assembly import GlobalAssembly
from core.solvers import DynamicSolver
from core.post_processing import FEMPostProcessor
from utils.material_properties import MaterialLibrary
def main():
    """Run dynamic analysis example."""
    print("Dynamic Analysis - Advanced FEM Example")
    print("=" * 45)
    print("This example analyzes dynamic response of a cantilever beam")
    print("Learning: Modal analysis, time integration, frequency response\n")
    # Structure parameters
    beam_length = 1.0  # m
    beam_height = 0.05  # m
    beam_thickness = 0.01  # m
    # Load parameters
    impulse_magnitude = 1000.0  # N
    impulse_duration = 0.001  # s
    harmonic_frequency = 50.0  # Hz
    harmonic_amplitude = 100.0  # N
    # Analysis parameters
    total_time = 0.2  # s
    time_step = 0.0001  # s
    # Material
    aluminum = MaterialLibrary.aluminum_6061()
    print(f"Material: {aluminum.name}")
    print(f"Young's Modulus: {aluminum.youngs_modulus/1e9:.1f} GPa")
    print(f"Density: {aluminum.density} kg/m³\n")
    # Create mesh
    print("Creating mesh...")
    mesh = create_beam_mesh(beam_length, beam_height, nx=20, ny=4)
    materials = {0: aluminum}
    print(f"Mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    # Setup assembly
    assembly = GlobalAssembly(mesh, materials)
    setup_beam_boundary_conditions(assembly, beam_thickness)
    print("\nAssembling system matrices...")
    K_global = assembly.assemble_global_stiffness()
    M_global = assembly.assemble_global_mass()
    print(f"System size: {K_global.shape[0]} DOFs")
    # Modal analysis
    print("\nPerforming modal analysis...")
    modal_results = perform_modal_analysis(assembly)
    # Transient analysis - impulse response
    print("\nPerforming impulse response analysis...")
    impulse_results = perform_impulse_analysis(assembly, impulse_magnitude,
                                             impulse_duration, total_time, time_step)
    # Transient analysis - harmonic response
    print("\nPerforming harmonic response analysis...")
    harmonic_results = perform_harmonic_analysis(assembly, harmonic_frequency,
                                                harmonic_amplitude, total_time, time_step)
    # Frequency response analysis
    print("\nPerforming frequency response analysis...")
    frequency_results = perform_frequency_response_analysis(assembly)
    # Generate comprehensive plots
    create_dynamic_analysis_plots(modal_results, impulse_results, harmonic_results,
                                frequency_results, mesh)
    # Advanced analysis
    perform_damping_analysis(assembly, modal_results)
    analyze_resonance_effects(frequency_results, modal_results)
    print("\n" + "=" * 45)
    print("Dynamic Analysis Complete!")
    print("Key Learning Points:")
    print("• Natural frequencies determine structural dynamic behavior")
    print("• Damping affects response amplitude and decay")
    print("• Resonance occurs when excitation matches natural frequencies")
    print("• Time integration captures transient response")
    print("• Modal superposition enables efficient dynamic analysis")
def create_beam_mesh(length, height, nx, ny):
    """Create structured mesh for cantilever beam."""
    mesh_generator = StructuredMeshGenerator()
    mesh = mesh_generator.generate_rectangle_mesh(length, height, nx, ny, element_type='quad2d')
    # Set element properties
    for element in mesh.elements.values():
        element.thickness = 0.01
        element.properties['plane_stress'] = True
    return mesh
def setup_beam_boundary_conditions(assembly, thickness):
    """Setup boundary conditions for cantilever beam."""
    # Set element thickness
    for element in assembly.mesh.elements.values():
        element.thickness = thickness
    # Fixed boundary condition at x = 0
    boundary_conditions = {}
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0]) < 1e-10:  # At x = 0
            boundary_conditions[(node_id, 0)] = 0.0  # Fix x-displacement
            boundary_conditions[(node_id, 1)] = 0.0  # Fix y-displacement
    assembly.apply_boundary_conditions(boundary_conditions)
    print(f"Applied boundary conditions to {len(boundary_conditions)} DOFs")
def perform_modal_analysis(assembly):
    """Perform modal analysis."""
    solver = DynamicSolver(assembly)
    # Extract first 10 modes
    eigenvalues, eigenvectors = solver.modal_analysis(num_modes=10)
    # Convert to frequencies and periods
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Hz
    periods = 1.0 / frequencies  # s
    print("Natural Frequencies and Mode Shapes:")
    print("-" * 45)
    print("Mode | Frequency (Hz) | Period (s)")
    print("-" * 45)
    for i in range(len(frequencies)):
        print(f"{i+1:4d} | {frequencies[i]:12.2f} | {periods[i]:8.4f}")
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'frequencies': frequencies,
        'periods': periods
    }
def perform_impulse_analysis(assembly, magnitude, duration, total_time, time_step):
    """Perform impulse response analysis."""
    def impulse_force(t):
        """Impulse load function."""
        force_vector = np.zeros(assembly.num_dofs)
        # Apply impulse at free end (tip of beam)
        coords = assembly.mesh.get_node_coordinates()
        max_x = np.max(coords[:, 0])
        # Find tip node (right end, middle height)
        tip_node = None
        min_distance = float('inf')
        for node_id, node in assembly.mesh.nodes.items():
            if abs(node.coordinates[0] - max_x) < 1e-10:
                distance = abs(node.coordinates[1] - coords[:, 1].max()/2)
                if distance < min_distance:
                    min_distance = distance
                    tip_node = node_id
        if tip_node is not None and t <= duration:
            global_dof = assembly.dof_map[tip_node][1]  # y-direction
            force_vector[global_dof] = -magnitude  # Downward force
        return force_vector
    # Initial conditions (at rest)
    initial_conditions = {
        'displacement': np.zeros(assembly.num_dofs),
        'velocity': np.zeros(assembly.num_dofs),
        'acceleration': np.zeros(assembly.num_dofs)
    }
    # Time integration
    solver = DynamicSolver(assembly)
    time_history = solver.newmark_integration(
        time_span=(0.0, total_time),
        time_step=time_step,
        initial_conditions=initial_conditions,
        force_function=impulse_force
    )
    # Extract tip displacement
    coords = assembly.mesh.get_node_coordinates()
    max_x = np.max(coords[:, 0])
    tip_node = None
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0] - max_x) < 1e-10:
            tip_node = node_id
            break
    if tip_node is not None:
        tip_displacement = time_history['displacement'][assembly.dof_map[tip_node][1], :]
    else:
        tip_displacement = np.zeros(len(time_history['time']))
    return {
        'time': time_history['time'],
        'displacement': time_history['displacement'],
        'velocity': time_history['velocity'],
        'acceleration': time_history['acceleration'],
        'tip_displacement': tip_displacement
    }
def perform_harmonic_analysis(assembly, frequency, amplitude, total_time, time_step):
    """Perform harmonic response analysis."""
    omega = 2 * np.pi * frequency
    def harmonic_force(t):
        """Harmonic load function."""
        force_vector = np.zeros(assembly.num_dofs)
        # Apply harmonic force at free end
        coords = assembly.mesh.get_node_coordinates()
        max_x = np.max(coords[:, 0])
        tip_node = None
        for node_id, node in assembly.mesh.nodes.items():
            if abs(node.coordinates[0] - max_x) < 1e-10:
                tip_node = node_id
                break
        if tip_node is not None:
            global_dof = assembly.dof_map[tip_node][1]
            force_vector[global_dof] = -amplitude * np.sin(omega * t)
        return force_vector
    # Initial conditions
    initial_conditions = {
        'displacement': np.zeros(assembly.num_dofs),
        'velocity': np.zeros(assembly.num_dofs),
        'acceleration': np.zeros(assembly.num_dofs)
    }
    # Time integration
    solver = DynamicSolver(assembly)
    time_history = solver.newmark_integration(
        time_span=(0.0, total_time),
        time_step=time_step,
        initial_conditions=initial_conditions,
        force_function=harmonic_force
    )
    # Extract tip displacement
    coords = assembly.mesh.get_node_coordinates()
    max_x = np.max(coords[:, 0])
    tip_node = None
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0] - max_x) < 1e-10:
            tip_node = node_id
            break
    if tip_node is not None:
        tip_displacement = time_history['displacement'][assembly.dof_map[tip_node][1], :]
    else:
        tip_displacement = np.zeros(len(time_history['time']))
    return {
        'time': time_history['time'],
        'displacement': time_history['displacement'],
        'tip_displacement': tip_displacement,
        'excitation_frequency': frequency
    }
def perform_frequency_response_analysis(assembly):
    """Perform frequency response analysis."""
    # Find tip nodes for excitation and response
    coords = assembly.mesh.get_node_coordinates()
    max_x = np.max(coords[:, 0])
    tip_node = None
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0] - max_x) < 1e-10:
            tip_node = node_id
            break
    if tip_node is None:
        print("Warning: Could not find tip node for frequency response")
        return None
    excitation_dof = assembly.dof_map[tip_node][1]  # y-direction
    response_dof = excitation_dof
    solver = DynamicSolver(assembly)
    # Frequency range from 0.1 Hz to 200 Hz
    frequency_response = solver.frequency_response(
        frequency_range=(0.1, 200.0),
        num_frequencies=500,
        damping_ratio=0.02,  # 2% modal damping
        excitation_dof=excitation_dof,
        response_dof=response_dof
    )
    return frequency_response
def create_dynamic_analysis_plots(modal_results, impulse_results, harmonic_results,
                                 frequency_results, mesh):
    """Create comprehensive dynamic analysis plots."""
    # Plot 1: Mode shapes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i in range(min(4, len(modal_results['frequencies']))):
        ax = axes[i]
        # Create a simple mode shape visualization
        # This is simplified - in practice would use proper modal displacement plotting
        coords = mesh.get_node_coordinates()
        mode_shape = modal_results['eigenvectors'][:, i]
        # Extract displacement for visualization (simplified)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        # Plot simplified mode shape
        ax.plot(x_coords, y_coords, 'b-', alpha=0.3, label='Undeformed')
        ax.set_title(f'Mode {i+1}: {modal_results["frequencies"][i]:.2f} Hz')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    # Plot 2: Impulse response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # Tip displacement time history
    ax1.plot(impulse_results['time'], impulse_results['tip_displacement']*1000, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tip Displacement (mm)')
    ax1.set_title('Impulse Response - Tip Displacement')
    ax1.grid(True, alpha=0.3)
    # Frequency content (FFT)
    dt = impulse_results['time'][1] - impulse_results['time'][0]
    fft_tip = np.fft.rfft(impulse_results['tip_displacement'])
    freqs = np.fft.rfftfreq(len(impulse_results['tip_displacement']), dt)
    ax2.semilogy(freqs, np.abs(fft_tip), 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Displacement Amplitude')
    ax2.set_title('Frequency Content of Impulse Response')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    # Mark natural frequencies
    for i, freq in enumerate(modal_results['frequencies'][:5]):
        if freq < 100:
            ax2.axvline(freq, color='green', linestyle='--', alpha=0.7,
                       label=f'Mode {i+1}' if i < 3 else "")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    # Plot 3: Harmonic response
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot steady-state portion of response
    steady_start = int(0.8 * len(harmonic_results['time']))  # Last 20% of simulation
    time_steady = harmonic_results['time'][steady_start:]
    disp_steady = harmonic_results['tip_displacement'][steady_start:] * 1000
    ax.plot(time_steady, disp_steady, 'b-', linewidth=2, label='Response')
    # Overlay excitation (scaled for visualization)
    omega = 2 * np.pi * harmonic_results['excitation_frequency']
    excitation_scaled = 0.1 * np.sin(omega * time_steady)  # Scaled down
    ax.plot(time_steady, excitation_scaled, 'r--', linewidth=1, alpha=0.7, label='Excitation (scaled)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tip Displacement (mm)')
    ax.set_title(f'Harmonic Response at {harmonic_results["excitation_frequency"]:.1f} Hz')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Plot 4: Frequency response function
    if frequency_results is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Magnitude
        ax1.loglog(frequency_results['frequency'], frequency_results['magnitude'], 'b-', linewidth=2)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Displacement/Force (m/N)')
        ax1.set_title('Frequency Response Function - Magnitude')
        ax1.grid(True, alpha=0.3)
        # Mark natural frequencies
        for i, freq in enumerate(modal_results['frequencies'][:5]):
            if freq < 200:
                ax1.axvline(freq, color='red', linestyle='--', alpha=0.7,
                           label=f'Mode {i+1}' if i < 3 else "")
        ax1.legend()
        # Phase
        ax2.semilogx(frequency_results['frequency'], frequency_results['phase'], 'g-', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Frequency Response Function - Phase')
        ax2.grid(True, alpha=0.3)
        # Mark natural frequencies
        for i, freq in enumerate(modal_results['frequencies'][:5]):
            if freq < 200:
                ax2.axvline(freq, color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
def perform_damping_analysis(assembly, modal_results):
    """Analyze the effects of damping on dynamic response."""
    print("\nDamping Analysis:")
    print("-" * 20)
    # Calculate modal damping effects
    damping_ratios = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
    first_frequency = modal_results['frequencies'][0]
    print(f"First natural frequency: {first_frequency:.2f} Hz")
    print("Damping effects on first mode:")
    print("Damping Ratio | Damped Freq (Hz) | Quality Factor")
    print("-" * 50)
    for zeta in damping_ratios:
        # Damped frequency
        if zeta < 1.0:
            damped_freq = first_frequency * np.sqrt(1 - zeta**2)
            quality_factor = 1 / (2 * zeta)
            print(f"{zeta:11.2f} | {damped_freq:13.2f} | {quality_factor:12.1f}")
        else:
            print(f"{zeta:11.2f} | Overdamped      | N/A")
    print(f"\nNote: Higher damping reduces resonance peaks but also reduces frequency")
def analyze_resonance_effects(frequency_results, modal_results):
    """Analyze resonance effects and peak responses."""
    if frequency_results is None:
        return
    print("\nResonance Analysis:")
    print("-" * 25)
    # Find peaks in frequency response
    magnitude = frequency_results['magnitude']
    frequencies = frequency_results['frequency']
    # Find local maxima
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(magnitude, height=np.max(magnitude)*0.1, distance=10)
    print("Resonance Peaks in Frequency Response:")
    print("Peak # | Frequency (Hz) | Magnitude | Amplification")
    print("-" * 55)
    static_compliance = magnitude[0] if len(magnitude) > 0 else 1.0
    for i, peak_idx in enumerate(peaks[:5]):  # Show first 5 peaks
        peak_freq = frequencies[peak_idx]
        peak_magnitude = magnitude[peak_idx]
        amplification = peak_magnitude / static_compliance
        print(f"{i+1:6d} | {peak_freq:12.2f} | {peak_magnitude:.2e} | {amplification:11.1f}")
    # Compare with natural frequencies
    print(f"\nComparison with Natural Frequencies:")
    print("Natural frequencies from modal analysis:")
    for i, freq in enumerate(modal_results['frequencies'][:5]):
        print(f"Mode {i+1}: {freq:.2f} Hz")
    print("\nNote: Resonance peaks should align with natural frequencies")
    print("Differences may be due to damping and modeling approximations")
if __name__ == "__main__":
    main()