#!/usr/bin/env python3
"""SciComp getting started tutorial with quantum states and signal processing."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Add the Python package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Python"))
def main():
    """Main tutorial function."""
    print("SciComp Tutorial")
    print("=" * 20)
    print()
    # Create output directory
    output_dir = Path("output/getting_started")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Section 1: Basic quantum mechanics
    print("📊 Section 1: Quantum States and Bell States")
    tutorial_quantum_basics(output_dir)
    # Section 2: Signal processing
    print("\n🌊 Section 2: Signal Processing Basics")
    tutorial_signal_processing(output_dir)
    # Section 3: Optimization
    print("\n🎯 Section 3: Optimization Basics")
    tutorial_optimization_basics(output_dir)
    # Section 4: Thermal physics
    print("\n🔥 Section 4: Heat Transfer Basics")
    tutorial_heat_transfer(output_dir)
    print(f"\nTutorial completed. Results in: {output_dir}")
def tutorial_quantum_basics(output_dir: Path):
    """Tutorial on quantum state basics."""
    print("   Creating and manipulating quantum states...")
    try:
        from Quantum.core.quantum_states import QuantumState, BellStates, EntanglementMeasures
        # Create a simple qubit state |+⟩ = (|0⟩ + |1⟩)/√2
        plus_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])
        print(f"   |+⟩ state created: {plus_state.state_vector}")
        print(f"   State is normalized: {plus_state.is_normalized()}")
        # Create Bell states
        phi_plus = BellStates.phi_plus()
        print(f"   Bell state |Φ+⟩ created")
        # Measure entanglement
        concurrence = EntanglementMeasures.concurrence(phi_plus)
        print(f"   Concurrence of |Φ+⟩: {concurrence:.3f}")
        print("   (Perfect entanglement = 1.0)")
        # Demonstrate superposition
        alpha, beta = 0.6, 0.8
        superposition = QuantumState([alpha, beta])
        prob_0 = superposition.measurement_probability(0)
        prob_1 = superposition.measurement_probability(1)
        print(f"   Superposition state |ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
        print(f"   P(|0⟩) = {prob_0:.3f}, P(|1⟩) = {prob_1:.3f}")
    except ImportError as e:
        print(f"   Quantum module not available: {e}")
        print("   This is expected for basic installation. Full quantum features coming soon!")
    except Exception as e:
        print(f"   Error in quantum tutorial: {e}")
        print("   Skipping quantum demonstration...")
def tutorial_signal_processing(output_dir: Path):
    """Tutorial on signal processing basics."""
    print("   Analyzing signals with Fourier transforms...")
    try:
        from Signal_Processing.core.fourier_transforms import FFT, SpectralAnalysis
        # Create a test signal: sine wave with noise
        t = np.linspace(0, 1, 1000)
        frequency = 50  # Hz
        signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
        # Perform FFT
        fft_processor = FFT()
        frequencies, spectrum = fft_processor.compute_fft(signal, sample_rate=1000)
        # Find dominant frequency
        dominant_freq_idx = np.argmax(np.abs(spectrum))
        dominant_freq = frequencies[dominant_freq_idx]
        print(f"   Signal created with frequency: {frequency} Hz")
        print(f"   FFT detected dominant frequency: {dominant_freq:.1f} Hz")
        # Spectral analysis
        spectral = SpectralAnalysis()
        power_spectrum = spectral.power_spectral_density(signal, sample_rate=1000)
        # Plot signal and spectrum
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t[:200], signal[:200], 'b-', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Time Domain Signal')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(frequencies[:len(frequencies)//2],
                np.abs(spectrum[:len(spectrum)//2]), 'r-', linewidth=1.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "signal_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError as e:
        print(f"   Signal processing module not available: {e}")
        # Create a simple example without the module
        print("   Creating basic sine wave example...")
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t)
        plt.figure(figsize=(10, 4))
        plt.plot(t[:200], signal[:200], 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('50 Hz Sine Wave')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "basic_signal.png", dpi=150, bbox_inches='tight')
        plt.close()
def tutorial_optimization_basics(output_dir: Path):
    """Tutorial on optimization basics."""
    print("   Solving optimization problems...")
    try:
        from Optimization.unconstrained import BFGS
        # Define a simple quadratic function to minimize
        def quadratic_function(x):
            return x[0]**2 + x[1]**2 + 2*x[0]*x[1] + 3*x[0] - x[1]
        def quadratic_gradient(x):
            return np.array([2*x[0] + 2*x[1] + 3, 2*x[1] + 2*x[0] - 1])
        # Initialize optimizer
        optimizer = BFGS(tolerance=1e-8)
        # Starting point
        x0 = np.array([5.0, 5.0])
        # Optimize
        result = optimizer.minimize(quadratic_function, x0, gradient=quadratic_gradient)
        print(f"   Optimization problem: min f(x,y) = x² + y² + 2xy + 3x - y")
        print(f"   Starting point: ({x0[0]}, {x0[1]})")
        print(f"   Optimal solution: ({result.x[0]:.4f}, {result.x[1]:.4f})")
        print(f"   Minimum value: {result.fun:.6f}")
        print(f"   Converged in {result.nit} iterations")
        # Visualize the optimization
        x_range = np.linspace(-2, 6, 100)
        y_range = np.linspace(-2, 6, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = X**2 + Y**2 + 2*X*Y + 3*X - Y
        plt.figure(figsize=(10, 8))
        contours = plt.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6)
        plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Function Value')
        # Plot optimization path if available
        if hasattr(result, 'path') and result.path is not None:
            path = np.array(result.path)
            plt.plot(path[:, 0], path[:, 1], 'r-o', linewidth=2,
                    markersize=6, label='Optimization Path')
        plt.plot(x0[0], x0[1], 'ro', markersize=10, label='Start')
        plt.plot(result.x[0], result.x[1], 'r*', markersize=15, label='Optimum')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Optimization of Quadratic Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "optimization.png", dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError as e:
        print(f"   Optimization module not available: {e}")
        print("   Creating simple optimization visualization...")
        # Create a simple contour plot
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = X**2 + Y**2
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='f(x,y) = x² + y²')
        plt.plot(0, 0, 'r*', markersize=15, label='Global Minimum')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Simple Optimization Problem')
        plt.legend()
        plt.savefig(output_dir / "basic_optimization.png", dpi=150, bbox_inches='tight')
        plt.close()
def tutorial_heat_transfer(output_dir: Path):
    """Tutorial on heat transfer basics."""
    print("   Solving heat equation...")
    try:
        from Thermal_Transport.core.heat_equation import HeatEquationSolver1D
        # Set up 1D heat equation problem
        L = 1.0  # Length
        nx = 100  # Grid points
        alpha = 0.01  # Thermal diffusivity
        solver = HeatEquationSolver1D(L=L, nx=nx, alpha=alpha)
        # Initial condition: Gaussian temperature distribution
        x = solver.grid
        T_initial = 100 * np.exp(-((x - 0.5) / 0.1)**2)
        # Solve for different times
        times = [0, 0.1, 0.5, 1.0, 2.0]
        print(f"   Solving 1D heat equation:")
        print(f"   Domain: [0, {L}] with {nx} grid points")
        print(f"   Thermal diffusivity: α = {alpha}")
        print(f"   Initial condition: Gaussian profile centered at x = 0.5")
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(times):
            if t == 0:
                T = T_initial
            else:
                T = solver.solve(T_initial, t_final=t, dt=0.001)
            plt.plot(x, T, linewidth=2, label=f't = {t:.1f}s')
            if i == 0:
                print(f"   Initial max temperature: {np.max(T):.1f}°C")
            else:
                print(f"   Max temperature at t={t:.1f}s: {np.max(T):.1f}°C")
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (°C)')
        plt.title('1D Heat Diffusion')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "heat_diffusion.png", dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError as e:
        print(f"   Heat transfer module not available: {e}")
        print("   Creating basic diffusion visualization...")
        # Create a simple diffusion example
        x = np.linspace(0, 1, 100)
        times = [0, 0.1, 0.5, 1.0]
        plt.figure(figsize=(10, 6))
        for t in times:
            # Analytical solution for Gaussian initial condition
            if t == 0:
                T = 100 * np.exp(-((x - 0.5) / 0.1)**2)
            else:
                sigma = np.sqrt(0.01**2 + 2*0.01*t)  # Spreading due to diffusion
                T = 100 * (0.1/sigma) * np.exp(-((x - 0.5) / sigma)**2)
            plt.plot(x, T, linewidth=2, label=f't = {t:.1f}s')
        plt.xlabel('Position')
        plt.ylabel('Temperature')
        plt.title('Heat Diffusion Example')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "basic_diffusion.png", dpi=150, bbox_inches='tight')
        plt.close()
if __name__ == "__main__":
    main()