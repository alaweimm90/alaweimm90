#!/usr/bin/env python3
"""
Basic Triangle Simulation Example

This script demonstrates how to run a basic single triangle simulation
using the MagLogic framework. It serves as a minimal working example
for new users.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add MagLogic to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

try:
    import maglogic
    from maglogic.simulation.oommf_runner import OOMMFRunner
    from maglogic.visualization.berkeley_style import berkeley_style
    from maglogic.core.constants import MATERIAL_CONSTANTS
except ImportError as e:
    print(f"MagLogic import error: {e}")
    print("Please ensure MagLogic is properly installed:")
    print("  pip install -e .")
    sys.exit(1)

def main():
    """Run basic triangle simulation example."""
    print("="*60)
    print("MagLogic: Basic Triangle Simulation Example")
    print("="*60)
    print(f"MagLogic version: {maglogic.__version__}")
    print(f"Author: {maglogic.__author__}")
    print("")
    
    # Setup Berkeley styling
    try:
        berkeley_style.setup()
        print("✓ Berkeley styling applied")
    except Exception as e:
        print(f"⚠ Could not apply Berkeley styling: {e}")
    
    # Simulation parameters
    params = {
        'edge_length': 100e-9,      # 100 nm triangle
        'thickness': 10e-9,         # 10 nm thickness
        'temperature': 300.0,       # Room temperature
        'cell_size': 2e-9,          # 2 nm mesh
        'final_time': 5e-9,         # 5 ns simulation
        'applied_field': 0.0,       # No applied field
        'field_angle': 0.0,         # Field angle
        'output_dir': './triangle_output'
    }
    
    # Material parameters (Permalloy)
    try:
        material_params = MATERIAL_CONSTANTS['permalloy_ni80fe20'].copy()
        print(f"✓ Using material: Permalloy (Ni80Fe20)")
        print(f"  Ms = {material_params['Ms']:.0f} A/m")
        print(f"  A_ex = {material_params['A_ex']:.2e} J/m")
        print(f"  α = {material_params['alpha']:.3f}")
    except KeyError:
        print("⚠ Could not load material parameters, using defaults")
        material_params = {
            'Ms': 860e3,
            'A_ex': 13e-12,
            'alpha': 0.01,
            'gamma': 2.211e5
        }
    
    # Create output directory
    output_dir = Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    
    # Check for OOMMF availability
    try:
        runner = OOMMFRunner(verbose=True)
        oommf_available = runner.check_installation()
        print(f"✓ OOMMF available: {oommf_available}")
    except Exception as e:
        print(f"⚠ OOMMF runner error: {e}")
        print("Continuing with parameter analysis only...")
        oommf_available = False
    
    # Analyze simulation parameters
    print("\n" + "="*40)
    print("Simulation Parameter Analysis")
    print("="*40)
    
    # Calculate characteristic lengths
    mu_0 = 4 * np.pi * 1e-7
    Ms = material_params['Ms']
    A_ex = material_params['A_ex']
    
    # Exchange length
    l_ex = np.sqrt(2 * A_ex / (mu_0 * Ms**2))
    print(f"Exchange length: {l_ex*1e9:.1f} nm")
    
    # Domain wall width (approximate)
    delta_w = np.pi * np.sqrt(A_ex / (mu_0 * Ms**2 / 2))
    print(f"Domain wall width: {delta_w*1e9:.1f} nm")
    
    # Mesh quality check
    mesh_quality = params['cell_size'] / l_ex
    print(f"Mesh quality (cell_size/l_ex): {mesh_quality:.3f}")
    if mesh_quality > 0.1:
        print("⚠ Warning: Mesh may be too coarse for accurate exchange calculation")
    else:
        print("✓ Mesh quality is adequate")
    
    # Time scale analysis
    alpha = material_params['alpha']
    gamma = material_params.get('gamma', 2.211e5)
    
    # Precession time scale
    H_typical = Ms  # Typical internal field ~ Ms
    f_prec = gamma * H_typical / (2 * np.pi)
    T_prec = 1 / f_prec
    print(f"Precession period: {T_prec*1e12:.1f} ps")
    
    # Relaxation time scale
    tau_relax = 1 / (alpha * gamma * Ms)
    print(f"Relaxation time: {tau_relax*1e9:.1f} ns")
    
    # Run simulation if OOMMF is available
    if oommf_available:
        print("\n" + "="*40)
        print("Running OOMMF Simulation")
        print("="*40)
        
        try:
            # Create simulation configuration
            config = {
                'geometry': {
                    'shape': 'triangle',
                    'edge_length': params['edge_length'],
                    'thickness': params['thickness']
                },
                'material': material_params,
                'simulation': {
                    'final_time': params['final_time'],
                    'cell_size': params['cell_size'],
                    'temperature': params['temperature']
                },
                'output': {
                    'directory': str(output_dir),
                    'basename': 'basic_triangle'
                }
            }
            
            # Run simulation
            print("Starting simulation...")
            result = runner.run_simulation(config)
            
            if result['success']:
                print("✓ Simulation completed successfully!")
                print(f"  Runtime: {result.get('runtime', 'unknown')}")
                print(f"  Output files: {len(result.get('output_files', []))}")
                
                # Analyze results
                analyze_results(result, output_dir)
                
            else:
                print(f"✗ Simulation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Simulation error: {e}")
            print("This may be due to:")
            print("  - OOMMF not properly installed")
            print("  - Missing simulation files")
            print("  - Insufficient permissions")
    
    # Create educational plots
    create_educational_plots(params, material_params, output_dir)
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    print("Next steps:")
    print("1. Examine the generated plots in:", output_dir)
    print("2. Try modifying simulation parameters")
    print("3. Run more complex examples:")
    print("   - python examples/compare_simulators.py")
    print("   - python examples/generate_logic_demo.py")
    print("4. Explore Jupyter notebooks in python/notebooks/")

def analyze_results(result, output_dir):
    """Analyze simulation results and create plots."""
    print("\nAnalyzing simulation results...")
    
    try:
        # Load magnetization data
        if 'magnetization_files' in result:
            mag_files = result['magnetization_files']
            print(f"Found {len(mag_files)} magnetization files")
            
            # Analyze final state
            if mag_files:
                final_file = mag_files[-1]
                print(f"Analyzing final state: {Path(final_file).name}")
                
                # This would require implementing OVF file reading
                # For now, create placeholder analysis
                create_analysis_plots(output_dir)
        
        # Load time series data  
        if 'table_file' in result:
            table_file = result['table_file']
            print(f"Loading time series: {Path(table_file).name}")
            
            # This would require implementing ODT file reading
            # For now, create placeholder plots
            create_timeseries_plots(output_dir)
            
    except Exception as e:
        print(f"Analysis error: {e}")

def create_analysis_plots(output_dir):
    """Create analysis plots (placeholder implementation)."""
    print("Creating analysis plots...")
    
    # Create example magnetization pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Magnetization pattern
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create triangular mask
    mask = (np.abs(X) < 50) & (Y > -30) & (Y < 30)
    triangle_mask = mask & (Y > -X*np.sqrt(3) - 30) & (Y > X*np.sqrt(3) - 30)
    
    # Example magnetization state (Y-state)
    Mx = np.zeros_like(X)
    My = np.zeros_like(X) 
    Mx[triangle_mask] = np.cos(np.pi/6)  # 30° from x-axis
    My[triangle_mask] = np.sin(np.pi/6)
    
    # Plot magnetization
    ax1.quiver(X[triangle_mask][::5], Y[triangle_mask][::5], 
               Mx[triangle_mask][::5], My[triangle_mask][::5],
               scale=10, alpha=0.8, color=berkeley_style.colors['primary']['berkeley_blue'])
    ax1.set_xlim(-60, 60)
    ax1.set_ylim(-40, 40)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title('Magnetization Pattern (Y-State)', 
                 color=berkeley_style.colors['primary']['berkeley_blue'])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Energy landscape
    theta = np.linspace(0, 2*np.pi, 100)
    # Simplified 6-fold energy landscape for triangle
    energy = -np.cos(6*theta) + 0.1*np.cos(2*theta)
    
    ax2.plot(theta*180/np.pi, energy, 
             color=berkeley_style.colors['primary']['california_gold'], 
             linewidth=3)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Energy (normalized)')
    ax2.set_title('Energy Landscape (6-fold)', 
                 color=berkeley_style.colors['primary']['berkeley_blue'])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    
    # Mark stable states
    stable_angles = [0, 60, 120, 180, 240, 300]
    for angle in stable_angles:
        ax2.axvline(angle, color=berkeley_style.colors['secondary']['red_dark'], 
                   alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    
    # Apply Berkeley styling
    berkeley_style.apply_to_figure(fig)
    
    # Save plot
    output_file = output_dir / 'magnetization_analysis.png'
    berkeley_style.save_figure(fig, output_file, dpi=300, format='png')
    plt.close()
    
    print(f"✓ Analysis plot saved: {output_file}")

def create_timeseries_plots(output_dir):
    """Create time series plots (placeholder implementation)."""
    print("Creating time series plots...")
    
    # Generate example time series
    t = np.linspace(0, 5, 500)  # 5 ns
    
    # Example relaxation dynamics
    tau = 0.5  # ns
    mx = 0.8 * np.exp(-t/tau) * np.cos(2*np.pi*t/0.1) + 0.1 * np.random.normal(0, 0.01, len(t))
    my = 0.6 * np.exp(-t/tau) * np.sin(2*np.pi*t/0.1) + 0.1 * np.random.normal(0, 0.01, len(t))
    mz = 0.1 * np.random.normal(0, 0.01, len(t))
    
    # Total energy (decreasing)
    energy = np.exp(-t/tau) + 0.1 * np.random.normal(0, 0.01, len(t))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnetization components
    colors = berkeley_style.get_color_cycle()
    ax1.plot(t, mx, color=colors[0], linewidth=2, label='$m_x$')
    ax1.plot(t, my, color=colors[1], linewidth=2, label='$m_y$')  
    ax1.plot(t, mz, color=colors[2], linewidth=2, label='$m_z$')
    
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Magnetization')
    ax1.set_title('Magnetization Dynamics', 
                 color=berkeley_style.colors['primary']['berkeley_blue'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy evolution
    ax2.plot(t, energy, color=berkeley_style.colors['primary']['california_gold'], 
             linewidth=2.5, label='Total Energy')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Energy (normalized)')
    ax2.set_title('Energy Relaxation', 
                 color=berkeley_style.colors['primary']['berkeley_blue'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Apply Berkeley styling
    berkeley_style.apply_to_figure(fig)
    
    # Save plot
    output_file = output_dir / 'time_series.png'
    berkeley_style.save_figure(fig, output_file, dpi=300, format='png')
    plt.close()
    
    print(f"✓ Time series plot saved: {output_file}")

def create_educational_plots(params, material_params, output_dir):
    """Create educational plots about triangle physics."""
    print("Creating educational plots...")
    
    # Plot 1: Triangle geometry and states
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Triangle geometry
    ax = axes[0, 0]
    triangle_x = np.array([-0.5, 0.5, 0, -0.5]) * params['edge_length'] * 1e9
    triangle_y = np.array([-0.289, -0.289, 0.577, -0.289]) * params['edge_length'] * 1e9
    
    ax.plot(triangle_x, triangle_y, 'k-', linewidth=3)
    ax.fill(triangle_x[:-1], triangle_y[:-1], alpha=0.3, 
            color=berkeley_style.colors['primary']['berkeley_blue'])
    ax.set_xlim(-60, 60) 
    ax.set_ylim(-40, 40)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title('Triangle Geometry')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Six magnetic states
    state_names = ['Y1 (0°)', 'Y2 (120°)', 'Y3 (240°)', 'V1', 'V2', 'V3']
    state_angles = [0, 120, 240, 60, 180, 300]
    
    for i, (name, angle) in enumerate(zip(state_names, state_angles)):
        row, col = divmod(i+1, 3)
        if row >= 2:
            break
            
        ax = axes[row, col]
        
        # Draw triangle
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Draw magnetization pattern
        if 'Y' in name:
            # Y-state: uniform magnetization
            arrow_x = 0.8 * np.cos(np.radians(angle))
            arrow_y = 0.8 * np.sin(np.radians(angle))
            ax.arrow(0, 0, arrow_x*30, arrow_y*30, head_width=5, head_length=5,
                    fc=berkeley_style.colors['primary']['california_gold'], 
                    ec=berkeley_style.colors['primary']['california_gold'])
        else:
            # Vortex state: circular arrows
            theta_circ = np.linspace(0, 2*np.pi, 8)
            for j, th in enumerate(theta_circ[:-1]):
                x_start = 15 * np.cos(th)
                y_start = 15 * np.sin(th)
                dx = -5 * np.sin(th)
                dy = 5 * np.cos(th)
                ax.arrow(x_start, y_start, dx, dy, head_width=2, head_length=2,
                        fc=berkeley_style.colors['secondary']['green_dark'],
                        ec=berkeley_style.colors['secondary']['green_dark'])
        
        ax.set_xlim(-60, 60)
        ax.set_ylim(-40, 40)
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    berkeley_style.apply_to_figure(fig, title='Triangular Nanomagnetic States')
    
    # Save plot
    output_file = output_dir / 'triangle_states.png'
    berkeley_style.save_figure(fig, output_file, dpi=300, format='png')
    plt.close()
    
    print(f"✓ Educational plot saved: {output_file}")
    
    # Plot 2: Parameter space
    create_parameter_space_plot(params, material_params, output_dir)

def create_parameter_space_plot(params, material_params, output_dir):
    """Create parameter space analysis plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Size vs Exchange length
    ax = axes[0, 0]
    sizes = np.logspace(1, 3, 50)  # 10 nm to 1 μm
    mu_0 = 4 * np.pi * 1e-7
    Ms = material_params['Ms']
    A_ex = material_params['A_ex']
    l_ex = np.sqrt(2 * A_ex / (mu_0 * Ms**2)) * 1e9  # nm
    
    ax.loglog(sizes, np.ones_like(sizes) * l_ex, 'k--', linewidth=2, 
              label=f'Exchange length = {l_ex:.1f} nm')
    ax.axvline(params['edge_length']*1e9, color=berkeley_style.colors['primary']['berkeley_blue'], 
               linewidth=3, label='Our triangle')
    ax.fill_between([l_ex, 1000], [1, 1], [1000, 1000], alpha=0.3, 
                    color=berkeley_style.colors['primary']['california_gold'],
                    label='Good resolution')
    
    ax.set_xlabel('Triangle size (nm)')
    ax.set_ylabel('Length scale (nm)')  
    ax.set_title('Size vs Exchange Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature vs Thermal energy
    ax = axes[0, 1]
    temps = np.linspace(0, 500, 100)
    k_B = 1.38064852e-23
    thermal_energy = k_B * temps / 1.602176634e-19  # Convert to eV
    
    # Estimate magnetic energy scale
    volume = (params['edge_length']**2 * np.sqrt(3) / 4) * params['thickness']
    E_mag = mu_0 * Ms**2 * volume / 2 / 1.602176634e-19  # eV
    
    ax.plot(temps, thermal_energy, color=berkeley_style.colors['secondary']['red_dark'], 
            linewidth=2, label='Thermal energy')
    ax.axhline(E_mag, color=berkeley_style.colors['primary']['berkeley_blue'], 
               linewidth=2, label=f'Magnetic energy ≈ {E_mag*1e6:.0f} μeV')
    ax.axvline(params['temperature'], color=berkeley_style.colors['primary']['california_gold'],
               linewidth=2, label='Our temperature')
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Temperature vs Energy Scales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Damping vs Time scales
    ax = axes[1, 0]
    alphas = np.logspace(-4, 0, 100)
    gamma = material_params.get('gamma', 2.211e5)
    
    tau_relax = 1 / (alphas * gamma * Ms) * 1e9  # ns
    T_prec = 2 * np.pi / (gamma * Ms) * 1e12  # ps
    
    ax.loglog(alphas, tau_relax, color=berkeley_style.colors['primary']['berkeley_blue'],
              linewidth=2, label='Relaxation time')
    ax.axhline(T_prec, color=berkeley_style.colors['secondary']['green_dark'],
               linewidth=2, label=f'Precession period ≈ {T_prec:.1f} ps')
    ax.axvline(material_params['alpha'], color=berkeley_style.colors['primary']['california_gold'],
               linewidth=2, label='Our damping')
    
    ax.set_xlabel('Damping parameter α')
    ax.set_ylabel('Time (ns, ps)')
    ax.set_title('Damping vs Time Scales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Field vs Energy
    ax = axes[1, 1]
    fields = np.linspace(0, 100e3, 100)  # A/m
    zeeman_energy = mu_0 * Ms * fields * volume / 1.602176634e-19  # eV
    
    ax.plot(fields/1000, zeeman_energy*1e6, color=berkeley_style.colors['secondary']['purple_dark'],
            linewidth=2, label='Zeeman energy')
    ax.axhline(E_mag*1e6, color=berkeley_style.colors['primary']['berkeley_blue'],
               linewidth=2, label='Shape anisotropy')
    
    ax.set_xlabel('Applied field (kA/m)')
    ax.set_ylabel('Energy (μeV)')
    ax.set_title('Field vs Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    berkeley_style.apply_to_figure(fig, title='Parameter Space Analysis')
    
    # Save plot
    output_file = output_dir / 'parameter_space.png'
    berkeley_style.save_figure(fig, output_file, dpi=300, format='png')
    plt.close()
    
    print(f"✓ Parameter space plot saved: {output_file}")

if __name__ == "__main__":
    main()