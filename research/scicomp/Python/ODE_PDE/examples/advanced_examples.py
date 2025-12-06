"""Advanced Examples for ODE and PDE Solving.
This module contains advanced examples demonstrating sophisticated ODE and PDE
solving techniques including stiff systems, nonlinear PDEs, spectral methods,
and multiphysics coupling.
Examples:
    - Stiff chemical kinetics (Robertson problem)
    - Brusselator reaction-diffusion system
    - Burgers' equation (nonlinear PDE)
    - Spectral methods for PDEs
    - Adaptive mesh refinement
    - Continuation and bifurcation analysis
    - Multiphysics coupling example
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Dict, Any, Tuple
# Import our ODE/PDE modules
from ..adaptive_methods import AdaptiveTimeStepper, AdaptiveMeshRefiner
from ..spectral_methods import FourierSpectral, ChebyshevSpectral
from ..nonlinear_solvers import ContinuationSolver, newton_raphson
from ..stability_analysis import von_neumann_analysis, LinearStabilityAnalyzer
from ..visualization import ODEPDEVisualizer
def example_1_robertson_problem():
    """Example 1: Robertson chemical kinetics (stiff ODE system)."""
    print("Example 1: Robertson chemical kinetics (stiff system)")
    print("=" * 60)
    # Robertson problem: A classic stiff ODE test case
    # dy1/dt = -0.04*y1 + 1e4*y2*y3
    # dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2²
    # dy3/dt = 3e7*y2²
    # Initial conditions: y1(0) = 1, y2(0) = 0, y3(0) = 0
    # Conservation: y1 + y2 + y3 = 1
    def robertson_ode(t, y):
        y1, y2, y3 = y
        dy1dt = -0.04 * y1 + 1e4 * y2 * y3
        dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        dy3dt = 3e7 * y2**2
        return np.array([dy1dt, dy2dt, dy3dt])
    def robertson_jacobian(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4*y3, 1e4*y2],
            [0.04, -1e4*y3 - 6e7*y2, -1e4*y2],
            [0, 6e7*y2, 0]
        ])
        return J
    # Initial conditions
    y0 = np.array([1.0, 0.0, 0.0])
    # Solve with very tight tolerances due to stiffness
    adaptive_stepper = AdaptiveTimeStepper(
        rtol=1e-8,
        atol=1e-12,
        dt_min=1e-15,
        dt_max=1e3
    )
    # Solve over multiple time scales
    time_spans = [(0, 4e-1), (4e-1, 4e1), (4e1, 4e5)]
    solutions = []
    for i, t_span in enumerate(time_spans):
        if i == 0:
            initial = y0
        else:
            initial = solutions[-1].y[-1]
        result = adaptive_stepper.integrate(robertson_ode, t_span, initial)
        solutions.append(result)
        if result.success:
            print(f"Time span {t_span}: {result.total_steps} steps, "
                  f"{result.rejections} rejections")
        else:
            print(f"Failed for time span {t_span}: {result.message}")
            break
    if all(sol.success for sol in solutions):
        # Combine solutions
        all_t = np.concatenate([sol.t for sol in solutions])
        all_y = np.concatenate([sol.y for sol in solutions])
        # Plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Linear scale
        ax1.plot(all_t, all_y[:, 0], 'b-', label='A (y₁)', linewidth=2)
        ax1.plot(all_t, all_y[:, 1], 'r-', label='B (y₂)', linewidth=2)
        ax1.plot(all_t, all_y[:, 2], 'g-', label='C (y₃)', linewidth=2)
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Robertson Problem - Linear Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Log scale
        ax2.semilogx(all_t, all_y[:, 0], 'b-', label='A (y₁)', linewidth=2)
        ax2.semilogx(all_t, all_y[:, 1], 'r-', label='B (y₂)', linewidth=2)
        ax2.semilogx(all_t, all_y[:, 2], 'g-', label='C (y₃)', linewidth=2)
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Concentration')
        ax2.set_title('Robertson Problem - Log Time Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Conservation check
        conservation = np.sum(all_y, axis=1)
        ax3.semilogx(all_t, conservation - 1, 'purple', linewidth=2)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Conservation error')
        ax3.set_title('Mass Conservation (y₁ + y₂ + y₃ - 1)')
        ax3.grid(True, alpha=0.3)
        # Time step evolution
        all_dt = np.concatenate([np.diff(sol.t) for sol in solutions])
        all_t_dt = np.concatenate([sol.t[1:] for sol in solutions])
        ax4.loglog(all_t_dt, all_dt, 'g-', linewidth=2)
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('Time step Δt')
        ax4.set_title('Adaptive Time Step Evolution')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Final concentrations: A={all_y[-1, 0]:.6f}, "
              f"B={all_y[-1, 1]:.6e}, C={all_y[-1, 2]:.6f}")
        print(f"Conservation error: {abs(conservation[-1] - 1):.2e}")
        print(f"Total steps: {sum(sol.total_steps for sol in solutions)}")
        print(f"Total rejections: {sum(sol.rejections for sol in solutions)}")
    print()
def example_2_brusselator_system():
    """Example 2: Brusselator reaction-diffusion system."""
    print("Example 2: Brusselator reaction-diffusion system")
    print("=" * 60)
    # Brusselator equations (simplified 1D version):
    # ∂u/∂t = Du∇²u + A - (B+1)u + u²v
    # ∂v/∂t = Dv∇²v + Bu - u²v
    # This is a classic pattern-forming system
    # Parameters
    A, B = 1.0, 3.0
    Du, Dv = 1.0, 16.0
    L = 20.0  # Domain length
    N = 256   # Grid points
    # Spatial grid
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    # Wave numbers for spectral differentiation
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    k2 = k**2
    def brusselator_rhs(t, uv):
        # Split into u and v components
        u = uv[:N]
        v = uv[N:]
        # Compute second derivatives using FFT
        u_hat = np.fft.fft(u)
        v_hat = np.fft.fft(v)
        u_xx = np.real(np.fft.ifft(-k2 * u_hat))
        v_xx = np.real(np.fft.ifft(-k2 * v_hat))
        # Reaction terms
        u2v = u**2 * v
        dudt = Du * u_xx + A - (B + 1) * u + u2v
        dvdt = Dv * v_xx + B * u - u2v
        return np.concatenate([dudt, dvdt])
    # Initial conditions: small random perturbations around steady state
    u_steady = A
    v_steady = B / A
    np.random.seed(42)  # For reproducibility
    u0 = u_steady + 0.1 * np.random.randn(N)
    v0 = v_steady + 0.1 * np.random.randn(N)
    initial_conditions = np.concatenate([u0, v0])
    # Solve the system
    adaptive_stepper = AdaptiveTimeStepper(rtol=1e-6, atol=1e-8, dt_max=0.1)
    result = adaptive_stepper.integrate(brusselator_rhs, (0, 20), initial_conditions)
    if result.success:
        print(f"Brusselator system solved successfully")
        print(f"Grid points: {N}")
        print(f"Time steps: {len(result.t)}")
        print(f"Final time: {result.t[-1]:.2f}")
        # Extract u and v solutions
        u_solution = result.y[:, :N]
        v_solution = result.y[:, N:]
        # Create plots
        fig = plt.figure(figsize=(16, 12))
        # Spacetime plots
        ax1 = plt.subplot(2, 3, 1)
        X, T = np.meshgrid(x, result.t)
        contour1 = ax1.contourf(X, T, u_solution, levels=50, cmap='viridis')
        plt.colorbar(contour1, ax=ax1, label='u concentration')
        ax1.set_xlabel('Space x')
        ax1.set_ylabel('Time t')
        ax1.set_title('Species u Evolution')
        ax2 = plt.subplot(2, 3, 2)
        contour2 = ax2.contourf(X, T, v_solution, levels=50, cmap='plasma')
        plt.colorbar(contour2, ax=ax2, label='v concentration')
        ax2.set_xlabel('Space x')
        ax2.set_ylabel('Time t')
        ax2.set_title('Species v Evolution')
        # Final spatial patterns
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(x, u_solution[-1], 'b-', linewidth=2, label='u(x)')
        ax3.plot(x, v_solution[-1], 'r-', linewidth=2, label='v(x)')
        ax3.axhline(y=u_steady, color='b', linestyle='--', alpha=0.7, label='u steady')
        ax3.axhline(y=v_steady, color='r', linestyle='--', alpha=0.7, label='v steady')
        ax3.set_xlabel('Space x')
        ax3.set_ylabel('Concentration')
        ax3.set_title('Final Spatial Pattern')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # Time series at specific location
        ax4 = plt.subplot(2, 3, 4)
        mid_idx = N // 2
        ax4.plot(result.t, u_solution[:, mid_idx], 'b-', linewidth=2, label='u(L/2)')
        ax4.plot(result.t, v_solution[:, mid_idx], 'r-', linewidth=2, label='v(L/2)')
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('Concentration')
        ax4.set_title('Temporal Evolution at x=L/2')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        # Phase portrait
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(u_solution[:, mid_idx], v_solution[:, mid_idx], 'purple', linewidth=2)
        ax5.plot(u_steady, v_steady, 'ko', markersize=8, label='Steady state')
        ax5.set_xlabel('u concentration')
        ax5.set_ylabel('v concentration')
        ax5.set_title('Phase Portrait at x=L/2')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        # Fourier analysis of final pattern
        ax6 = plt.subplot(2, 3, 6)
        u_final_fft = np.abs(np.fft.fft(u_solution[-1]))
        v_final_fft = np.abs(np.fft.fft(v_solution[-1]))
        freq = np.fft.fftfreq(N, dx)
        positive_freq = freq[:N//2]
        ax6.semilogy(positive_freq, u_final_fft[:N//2], 'b-', linewidth=2, label='u spectrum')
        ax6.semilogy(positive_freq, v_final_fft[:N//2], 'r-', linewidth=2, label='v spectrum')
        ax6.set_xlabel('Frequency')
        ax6.set_ylabel('Amplitude')
        ax6.set_title('Spatial Frequency Spectrum')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Pattern analysis
        u_var = np.var(u_solution[-1])
        v_var = np.var(v_solution[-1])
        print(f"Final pattern variance: u = {u_var:.4f}, v = {v_var:.4f}")
        # Dominant wavelength
        dominant_idx = np.argmax(u_final_fft[1:N//2]) + 1
        dominant_wavelength = L / dominant_idx
        print(f"Dominant wavelength: {dominant_wavelength:.2f}")
    else:
        print(f"Brusselator solve failed: {result.message}")
    print()
def example_3_burgers_equation():
    """Example 3: Burgers' equation (nonlinear PDE)."""
    print("Example 3: Burgers' equation")
    print("=" * 60)
    # Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    # This is a classic nonlinear PDE that can develop shock waves
    # Parameters
    nu = 0.01  # Viscosity (small for shock formation)
    L = 2 * np.pi
    N = 512
    # Spatial grid
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    # Spectral method setup
    spectral_solver = FourierSpectral(N, (0, L))
    def burgers_rhs(t, u):
        # Spectral derivatives
        dudx = spectral_solver.fourier_derivative(u, order=1)
        d2udx2 = spectral_solver.fourier_derivative(u, order=2)
        # Burgers' equation RHS
        dudt = -u * dudx + nu * d2udx2
        return dudt
    # Initial condition: sine wave (will develop into shock)
    def initial_condition(x):
        return np.sin(x)
    u0 = initial_condition(x)
    # Solve with adaptive time stepping
    adaptive_stepper = AdaptiveTimeStepper(rtol=1e-6, atol=1e-8, dt_max=0.01)
    result = adaptive_stepper.integrate(burgers_rhs, (0, 3.0), u0)
    if result.success:
        print(f"Burgers' equation solved successfully")
        print(f"Grid points: {N}")
        print(f"Time steps: {len(result.t)}")
        print(f"Reynolds number: {1/nu:.1f}")
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Snapshots showing shock development
        time_indices = [0, len(result.t)//6, len(result.t)//3, len(result.t)//2, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for t_idx, color in zip(time_indices, colors):
            ax1.plot(x, result.y[t_idx], color=color, linewidth=2,
                    label=f't = {result.t[t_idx]:.2f}')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Velocity u(x,t)')
        ax1.set_title('Burgers\' Equation: Shock Development')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Spacetime contour
        X, T = np.meshgrid(x, result.t)
        contour = ax2.contourf(X, T, result.y, levels=50, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax2, label='Velocity u')
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Time t')
        ax2.set_title('Spacetime Evolution')
        # Energy evolution
        energy = np.sum(result.y**2, axis=1) * dx / 2
        ax3.plot(result.t, energy, 'b-', linewidth=2)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Kinetic energy')
        ax3.set_title('Energy Dissipation')
        ax3.grid(True, alpha=0.3)
        # Shock steepening analysis
        gradients = []
        for i in range(len(result.t)):
            dudx = spectral_solver.fourier_derivative(result.y[i], order=1)
            max_gradient = np.max(np.abs(dudx))
            gradients.append(max_gradient)
        ax4.semilogy(result.t, gradients, 'r-', linewidth=2)
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('Max |∂u/∂x|')
        ax4.set_title('Shock Steepening')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Initial energy: {energy[0]:.6f}")
        print(f"Final energy: {energy[-1]:.6f}")
        print(f"Energy dissipated: {(energy[0] - energy[-1])/energy[0]*100:.1f}%")
        print(f"Maximum gradient: {np.max(gradients):.1f}")
        # Check for shock formation
        if np.max(gradients) > 10:
            shock_time_idx = np.where(np.array(gradients) > 5)[0][0]
            print(f"Shock formation time: ~{result.t[shock_time_idx]:.3f}")
    else:
        print(f"Burgers' equation solve failed: {result.message}")
    print()
def example_4_spectral_method_comparison():
    """Example 4: Spectral method comparison for PDEs."""
    print("Example 4: Spectral method comparison")
    print("=" * 60)
    # Compare Fourier vs Chebyshev spectral methods for solving
    # -d²u/dx² = f(x) with different boundary conditions
    # Test problem 1: Periodic (Fourier)
    print("Periodic problem (Fourier spectral):")
    # Domain and discretization
    N = 64
    L = 2 * np.pi
    fourier_solver = FourierSpectral(N, (0, L))
    # Source term and analytical solution for periodic problem
    def source_periodic(x):
        return np.sin(2*x) + 0.5*np.cos(3*x)
    def analytical_periodic(x):
        return -np.sin(2*x)/4 - 0.5*np.cos(3*x)/9
    # Solve with Fourier spectral method
    boundary_conditions = {'periodic': True}
    result_fourier = fourier_solver.solve_pde(None, boundary_conditions)
    x_fourier = fourier_solver.x
    u_analytical_fourier = analytical_periodic(x_fourier)
    # Test problem 2: Non-periodic (Chebyshev)
    print("Non-periodic problem (Chebyshev spectral):")
    chebyshev_solver = ChebyshevSpectral(N, (-1, 1))
    # Source term and analytical solution for Chebyshev problem
    def source_chebyshev(x):
        return 12*x**2 - 2
    def analytical_chebyshev(x):
        return x**4 - x**2  # Satisfies u(-1) = u(1) = 0
    # Boundary conditions
    boundary_conditions_cheb = {
        'dirichlet': {'left': 0.0, 'right': 0.0}
    }
    result_chebyshev = chebyshev_solver.solve_pde(None, boundary_conditions_cheb)
    x_chebyshev = chebyshev_solver.x
    u_analytical_chebyshev = analytical_chebyshev(x_chebyshev)
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # Fourier solution
    if result_fourier.success:
        ax1.plot(x_fourier, u_analytical_fourier, 'k-', linewidth=2, label='Analytical')
        ax1.plot(x_fourier, result_fourier.u, 'r--', linewidth=2, label='Fourier spectral')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title('Periodic Problem: Fourier Spectral')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Error
        error_fourier = np.abs(result_fourier.u - u_analytical_fourier)
        ax2.semilogy(x_fourier, error_fourier, 'b-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute error')
        ax2.set_title('Fourier Spectral Error')
        ax2.grid(True, alpha=0.3)
        print(f"Fourier max error: {np.max(error_fourier):.2e}")
    # Chebyshev solution
    if result_chebyshev.success:
        ax3.plot(x_chebyshev, u_analytical_chebyshev, 'k-', linewidth=2, label='Analytical')
        ax3.plot(x_chebyshev, result_chebyshev.u, 'r--', linewidth=2, label='Chebyshev spectral')
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x)')
        ax3.set_title('Non-periodic Problem: Chebyshev Spectral')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # Error
        error_chebyshev = np.abs(result_chebyshev.u - u_analytical_chebyshev)
        ax4.semilogy(x_chebyshev, error_chebyshev, 'g-', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Absolute error')
        ax4.set_title('Chebyshev Spectral Error')
        ax4.grid(True, alpha=0.3)
        print(f"Chebyshev max error: {np.max(error_chebyshev):.2e}")
    plt.tight_layout()
    plt.show()
    # Spectral convergence analysis
    print("\nSpectral convergence analysis:")
    N_values = [8, 16, 32, 64, 128]
    fourier_errors = []
    chebyshev_errors = []
    for n in N_values:
        # Fourier
        fs = FourierSpectral(n, (0, L))
        result_f = fs.solve_pde(None, {'periodic': True})
        if result_f.success:
            u_exact_f = analytical_periodic(fs.x)
            error_f = np.max(np.abs(result_f.u - u_exact_f))
            fourier_errors.append(error_f)
        else:
            fourier_errors.append(np.nan)
        # Chebyshev
        cs = ChebyshevSpectral(n, (-1, 1))
        result_c = cs.solve_pde(None, {'dirichlet': {'left': 0.0, 'right': 0.0}})
        if result_c.success:
            u_exact_c = analytical_chebyshev(cs.x)
            error_c = np.max(np.abs(result_c.u - u_exact_c))
            chebyshev_errors.append(error_c)
        else:
            chebyshev_errors.append(np.nan)
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, fourier_errors, 'ro-', linewidth=2, label='Fourier')
    plt.semilogy(N_values, chebyshev_errors, 'bs-', linewidth=2, label='Chebyshev')
    plt.xlabel('Number of modes N')
    plt.ylabel('Maximum error')
    plt.title('Spectral Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print()
def example_5_adaptive_mesh_refinement():
    """Example 5: Adaptive mesh refinement for boundary layer problem."""
    print("Example 5: Adaptive mesh refinement")
    print("=" * 60)
    # Boundary layer problem: -εu'' + u' = 0, u(0) = 0, u(1) = 1
    # Analytical solution: u(x) = (1 - exp(x/ε)) / (1 - exp(1/ε))
    # This has a boundary layer near x = 0 for small ε
    epsilon = 0.01  # Small parameter -> boundary layer
    def analytical_solution(x):
        return (1 - np.exp(x/epsilon)) / (1 - np.exp(1/epsilon))
    def residual_based_indicator(x_left, x_right, x_mid):
        # Simple residual-based refinement criterion
        # Refine where solution changes rapidly
        u_left = analytical_solution(x_left)
        u_right = analytical_solution(x_right)
        u_mid = analytical_solution(x_mid)
        # Check if linear interpolation is insufficient
        u_interp = 0.5 * (u_left + u_right)
        error = abs(u_mid - u_interp)
        return error > 0.01  # Refinement threshold
    # Adaptive mesh refinement
    refiner = AdaptiveMeshRefiner(
        refinement_criterion='gradient',
        refinement_threshold=0.1,
        max_refinement_levels=5
    )
    # Start with coarse uniform mesh
    x_coarse = np.linspace(0, 1, 11)
    u_coarse = analytical_solution(x_coarse)
    # Perform several refinement steps
    meshes = [x_coarse]
    solutions = [u_coarse]
    for level in range(4):
        print(f"Refinement level {level + 1}:")
        result = refiner.refine_mesh(meshes[-1], solutions[-1])
        if result.success:
            meshes.append(result.new_mesh)
            solutions.append(result.refined_solution)
            print(f"  Mesh points: {len(meshes[-2])} -> {len(meshes[-1])}")
            print(f"  Refinement ratio: {result.refinement_ratio:.2f}")
        else:
            print(f"  Refinement failed: {result.message}")
            break
    # Plot mesh evolution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # Solution on different meshes
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    x_fine = np.linspace(0, 1, 1000)
    u_exact = analytical_solution(x_fine)
    ax1.plot(x_fine, u_exact, 'k-', linewidth=2, label='Exact')
    for i, (x_mesh, u_mesh, color) in enumerate(zip(meshes, solutions, colors)):
        ax1.plot(x_mesh, u_mesh, 'o-', color=color, linewidth=2,
                label=f'Level {i} ({len(x_mesh)} pts)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('Adaptive Mesh Refinement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Zoom into boundary layer
    ax2.plot(x_fine, u_exact, 'k-', linewidth=2, label='Exact')
    for i, (x_mesh, u_mesh, color) in enumerate(zip(meshes, solutions, colors)):
        mask = x_mesh <= 0.2  # Focus on boundary layer
        ax2.plot(x_mesh[mask], u_mesh[mask], 'o-', color=color, linewidth=2,
                label=f'Level {i}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title('Boundary Layer Detail')
    ax2.set_xlim(0, 0.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Mesh point distribution
    ax3.set_title('Mesh Point Distribution')
    for i, (x_mesh, color) in enumerate(zip(meshes, colors)):
        y_pos = np.full_like(x_mesh, i)
        ax3.plot(x_mesh, y_pos, 'o', color=color, markersize=4, label=f'Level {i}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Refinement level')
    ax3.set_title('Mesh Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Error analysis
    errors = []
    for x_mesh, u_mesh in zip(meshes, solutions):
        u_exact_mesh = analytical_solution(x_mesh)
        error = np.max(np.abs(u_mesh - u_exact_mesh))
        errors.append(error)
    ax4.semilogy(range(len(errors)), errors, 'ro-', linewidth=2)
    ax4.set_xlabel('Refinement level')
    ax4.set_ylabel('Maximum error')
    ax4.set_title('Error Reduction')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Print mesh statistics
    print("\nMesh statistics:")
    for i, x_mesh in enumerate(meshes):
        h_min = np.min(np.diff(x_mesh))
        h_max = np.max(np.diff(x_mesh))
        print(f"Level {i}: {len(x_mesh)} points, h_min = {h_min:.6f}, "
              f"h_max = {h_max:.6f}, ratio = {h_max/h_min:.1f}")
    print(f"\nFinal error: {errors[-1]:.2e}")
    print()
def example_6_continuation_bifurcation():
    """Example 6: Continuation and bifurcation analysis."""
    print("Example 6: Continuation and bifurcation analysis")
    print("=" * 60)
    # Pitchfork bifurcation example: f(x, μ) = μx - x³ = 0
    # This has a pitchfork bifurcation at μ = 0
    def pitchfork_equation(x, mu):
        return mu * x - x**3
    def pitchfork_jacobian(x, mu):
        return np.array([[mu - 3*x**2]])
    # Continuation solver
    continuation_solver = ContinuationSolver()
    # Start from stable solution at μ = -1
    x0 = np.array([0.0])  # x = 0 is solution for all μ
    mu_range = (-1.0, 1.0)
    # Perform continuation
    result = continuation_solver.solve_parameter_continuation(
        f=pitchfork_equation,
        jacobian=pitchfork_jacobian,
        x0=x0,
        param_range=mu_range,
        n_steps=100
    )
    if result.success:
        print(f"Continuation completed successfully")
        print(f"Number of steps: {len(result.parameter_values)}")
        print(f"Bifurcation points detected: {len(result.bifurcation_points)}")
        # Also solve for non-trivial branch (x ≠ 0)
        # For μ > 0, solutions are x = ±√μ
        mu_positive = result.parameter_values[result.parameter_values > 0]
        x_plus = np.sqrt(mu_positive)
        x_minus = -np.sqrt(mu_positive)
        # Plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Bifurcation diagram
        ax1.plot(result.parameter_values, result.solutions[:, 0], 'b-',
                linewidth=2, label='Trivial branch (x = 0)')
        ax1.plot(mu_positive, x_plus, 'r-', linewidth=2, label='x = +√μ')
        ax1.plot(mu_positive, x_minus, 'r-', linewidth=2, label='x = -√μ')
        # Mark bifurcation points
        for bp in result.bifurcation_points:
            ax1.axvline(x=result.parameter_values[bp], color='k',
                       linestyle='--', alpha=0.7)
            ax1.plot(result.parameter_values[bp], result.solutions[bp, 0],
                    'ko', markersize=8)
        ax1.set_xlabel('Parameter μ')
        ax1.set_ylabel('Solution x')
        ax1.set_title('Pitchfork Bifurcation Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        # Convergence history for selected points
        selected_steps = [10, 25, 50, 75]
        for i, step in enumerate(selected_steps):
            if step < len(result.convergence_data):
                conv_data = result.convergence_data[step]
                ax2.semilogy(conv_data.convergence_history,
                           color=colors[i % len(colors)], linewidth=2,
                           label=f'μ = {result.parameter_values[step]:.2f}')
        ax2.set_xlabel('Newton iteration')
        ax2.set_ylabel('Residual norm')
        ax2.set_title('Newton Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Linear stability analysis
        eigenvalues = []
        for i, (x, mu) in enumerate(zip(result.solutions[:, 0], result.parameter_values)):
            # Jacobian eigenvalue = μ - 3x²
            eigenval = mu - 3*x**2
            eigenvalues.append(eigenval)
        ax3.plot(result.parameter_values, eigenvalues, 'g-', linewidth=2)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Stability boundary')
        ax3.fill_between(result.parameter_values, -2, 0, alpha=0.3, color='red',
                        label='Unstable')
        ax3.fill_between(result.parameter_values, 0, 2, alpha=0.3, color='green',
                        label='Stable')
        ax3.set_xlabel('Parameter μ')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Linear Stability Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-2, 2)
        # Solution manifold in 3D
        ax4 = fig.add_subplot(224, projection='3d')
        # Trivial branch
        ax4.plot(result.parameter_values, result.solutions[:, 0],
                np.zeros_like(result.parameter_values), 'b-', linewidth=3,
                label='Trivial branch')
        # Non-trivial branches
        ax4.plot(mu_positive, x_plus, np.zeros_like(mu_positive), 'r-',
                linewidth=3, label='Non-trivial branch')
        ax4.plot(mu_positive, x_minus, np.zeros_like(mu_positive), 'r-',
                linewidth=3)
        # Vector field
        mu_grid = np.linspace(-1, 1, 20)
        x_grid = np.linspace(-1.5, 1.5, 20)
        MU, X = np.meshgrid(mu_grid, x_grid)
        F = MU * X - X**3
        # Skip some arrows for clarity
        skip = 2
        ax4.quiver(MU[::skip, ::skip], X[::skip, ::skip],
                  np.zeros_like(F[::skip, ::skip]), F[::skip, ::skip],
                  length=0.1, alpha=0.6)
        ax4.set_xlabel('Parameter μ')
        ax4.set_ylabel('Solution x')
        ax4.set_zlabel('f(x,μ)')
        ax4.set_title('Solution Manifold')
        plt.tight_layout()
        plt.show()
        # Analysis
        if result.bifurcation_points:
            bp_mu = result.parameter_values[result.bifurcation_points[0]]
            print(f"Bifurcation point detected at μ ≈ {bp_mu:.6f}")
            print(f"Theoretical bifurcation point: μ = 0")
            print(f"Error: {abs(bp_mu):.6f}")
        print(f"Stability changes at μ = 0")
    else:
        print(f"Continuation failed: {result.message}")
    print()
def run_all_examples():
    """Run all advanced examples."""
    print("Running Advanced ODE/PDE Examples")
    print("=" * 70)
    print()
    example_1_robertson_problem()
    example_2_brusselator_system()
    example_3_burgers_equation()
    example_4_spectral_method_comparison()
    example_5_adaptive_mesh_refinement()
    example_6_continuation_bifurcation()
    print("All advanced examples completed!")
if __name__ == "__main__":
    run_all_examples()