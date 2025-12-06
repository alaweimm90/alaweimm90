"""Intermediate Examples for ODE and PDE Solving.
This module contains intermediate-level examples demonstrating advanced ODE and PDE
solving techniques including adaptive methods, nonlinear problems, and systems.
Examples:
    - Adaptive time stepping with error control
    - Nonlinear ODE systems (predator-prey, pendulum)
    - 2D heat equation with mixed boundary conditions
    - Advection-diffusion equation
    - Nonlinear PDE (Burgers' equation)
    - Finite element method examples
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any
# Import our ODE/PDE modules
from ..adaptive_methods import AdaptiveTimeStepper
from ..pde_solvers import AdvectionDiffusionSolver, PDEOptions
from ..finite_element import solve_fem_poisson, fem_convergence_study
from ..nonlinear_solvers import newton_raphson
from ..visualization import ODEPDEVisualizer
def example_1_adaptive_ode():
    """Example 1: Adaptive time stepping for stiff ODE."""
    print("Example 1: Adaptive time stepping for stiff ODE")
    print("=" * 50)
    # Stiff ODE: dy/dt = -1000y + 1000cos(t), y(0) = 0
    # Analytical solution: y(t) = sin(t)
    def dydt(t, y):
        return -1000 * y + 1000 * np.cos(t)
    def analytical_solution(t):
        return np.sin(t)
    # Adaptive solver
    adaptive_stepper = AdaptiveTimeStepper(
        rtol=1e-6,
        atol=1e-8,
        dt_min=1e-8,
        dt_max=0.1
    )
    # Solve with adaptive stepping
    result = adaptive_stepper.integrate(dydt, (0, 2*np.pi), np.array([0.0]))
    if result.success:
        print(f"Adaptive integration successful")
        print(f"Total steps: {result.total_steps}")
        print(f"Rejections: {result.rejections}")
        print(f"Function evaluations: {result.function_evaluations}")
        # Compare with analytical solution
        y_exact = analytical_solution(result.t)
        error = np.abs(result.y[:, 0] - y_exact)
        # Plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Solution comparison
        ax1.plot(result.t, y_exact, 'k-', label='Analytical', linewidth=2)
        ax1.plot(result.t, result.y[:, 0], 'r--', label='Adaptive RK45', linewidth=2)
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('Solution y(t)')
        ax1.set_title('Stiff ODE Solution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Error
        ax2.semilogy(result.t, error, 'b-', linewidth=2)
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Absolute error')
        ax2.set_title('Solution Error')
        ax2.grid(True, alpha=0.3)
        # Time step evolution
        ax3.semilogy(result.t[1:], result.dt_history, 'g-', linewidth=2)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Time step Δt')
        ax3.set_title('Adaptive Time Step')
        ax3.grid(True, alpha=0.3)
        # Error control
        ax4.semilogy(result.t[1:], result.error_history, 'purple', linewidth=2)
        ax4.axhline(y=1.0, color='r', linestyle='--', label='Tolerance')
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('Error estimate')
        ax4.set_title('Error Control')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Final error: {error[-1]:.8f}")
        print(f"Average step size: {np.mean(result.dt_history):.8f}")
        print(f"Min step size: {np.min(result.dt_history):.8f}")
        print(f"Max step size: {np.max(result.dt_history):.8f}")
    else:
        print(f"Adaptive integration failed: {result.message}")
    print()
def example_2_predator_prey():
    """Example 2: Predator-prey system (Lotka-Volterra)."""
    print("Example 2: Predator-prey system (Lotka-Volterra)")
    print("=" * 50)
    # Lotka-Volterra equations:
    # dx/dt = ax - bxy  (prey)
    # dy/dt = cxy - dy  (predator)
    # Parameters
    a, b, c, d = 1.0, 0.5, 0.5, 1.0
    def lotka_volterra(t, z):
        x, y = z
        dxdt = a * x - b * x * y
        dydt = c * x * y - d * y
        return np.array([dxdt, dydt])
    # Initial conditions
    x0, y0 = 2.0, 1.0
    initial_conditions = np.array([x0, y0])
    # Solve system
    adaptive_stepper = AdaptiveTimeStepper(rtol=1e-8, atol=1e-10)
    result = adaptive_stepper.integrate(lotka_volterra, (0, 15), initial_conditions)
    if result.success:
        x_solution = result.y[:, 0]
        y_solution = result.y[:, 1]
        # Plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Time series
        ax1.plot(result.t, x_solution, 'b-', label='Prey (x)', linewidth=2)
        ax1.plot(result.t, y_solution, 'r-', label='Predator (y)', linewidth=2)
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('Population')
        ax1.set_title('Predator-Prey Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Phase portrait
        ax2.plot(x_solution, y_solution, 'purple', linewidth=2)
        ax2.plot(x0, y0, 'go', markersize=8, label='Initial condition')
        ax2.set_xlabel('Prey (x)')
        ax2.set_ylabel('Predator (y)')
        ax2.set_title('Phase Portrait')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Conservation quantity (H = cx - d*ln(x) + ay - b*ln(y))
        H = c * x_solution - d * np.log(x_solution) + a * y_solution - b * np.log(y_solution)
        ax3.plot(result.t, H, 'g-', linewidth=2)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Conserved quantity H')
        ax3.set_title('Conservation Check')
        ax3.grid(True, alpha=0.3)
        # Vector field
        x_range = np.linspace(0.5, 4, 20)
        y_range = np.linspace(0.5, 3, 15)
        X, Y = np.meshgrid(x_range, y_range)
        DX = a * X - b * X * Y
        DY = c * X * Y - d * Y
        ax4.quiver(X, Y, DX, DY, alpha=0.6)
        ax4.plot(x_solution, y_solution, 'purple', linewidth=2)
        ax4.set_xlabel('Prey (x)')
        ax4.set_ylabel('Predator (y)')
        ax4.set_title('Vector Field')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Analysis
        print(f"Conservation quantity variation: {np.std(H):.10f}")
        print(f"Period estimate: {result.t[-1]/3:.2f} time units")  # Approximate
        print(f"Max prey population: {np.max(x_solution):.3f}")
        print(f"Max predator population: {np.max(y_solution):.3f}")
    else:
        print(f"Solution failed: {result.message}")
    print()
def example_3_2d_heat_equation():
    """Example 3: 2D heat equation with mixed boundary conditions."""
    print("Example 3: 2D heat equation with mixed BCs")
    print("=" * 50)
    # Problem: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) on [0,1]×[0,1]
    # BC: u(0,y,t) = 0, u(1,y,t) = 0, ∂u/∂y(x,0,t) = 0, u(x,1,t) = sin(πx)
    from ..pde_solvers import HeatEquationSolver, PDEOptions
    # Domain
    nx, ny = 21, 21
    domain = {
        'x': np.linspace(0, 1, nx),
        'y': np.linspace(0, 1, ny)
    }
    # Mixed boundary conditions (simplified for demonstration)
    boundary_conditions = {
        'dirichlet': {
            # Bottom boundary nodes (y=0): Neumann-like, set to initial values
            # Top boundary (y=1): u = sin(πx)
            # Left/right boundaries (x=0,1): u = 0
        }
    }
    # This is a simplified 2D implementation
    # In practice, you'd need a full 2D heat solver
    print("2D heat equation requires full 2D implementation.")
    print("This would involve:")
    print("- 2D finite difference discretization")
    print("- Proper mixed boundary condition handling")
    print("- 2D visualization with contours/surface plots")
    print("- Kronecker product assembly for 2D Laplacian")
    print()
def example_4_advection_diffusion():
    """Example 4: Advection-diffusion equation."""
    print("Example 4: Advection-diffusion equation")
    print("=" * 50)
    # Problem: ∂u/∂t + v·∇u = D∇²u + f
    # 1D: ∂u/∂t + v∂u/∂x = D∂²u/∂x²
    # Domain
    domain = {'x': np.linspace(0, 10, 101)}
    # Parameters
    velocity = 1.0  # Advection velocity
    diffusivity = 0.1  # Diffusion coefficient
    # Boundary conditions
    boundary_conditions = {
        'dirichlet': {0: 0.0, 100: 0.0}
    }
    # Options
    options = PDEOptions(
        method='finite_difference',
        time_scheme='implicit',
        theta=0.5  # Crank-Nicolson
    )
    # Create solver
    solver = AdvectionDiffusionSolver(
        domain=domain,
        boundary_conditions=boundary_conditions,
        velocity=velocity,
        diffusivity=diffusivity,
        options=options
    )
    # Initial condition: Gaussian pulse
    def initial_condition(x):
        return np.exp(-2 * (x - 2)**2)
    # Solve transient problem
    result = solver.solve_transient(
        initial_condition=initial_condition,
        time_span=(0, 5),
        dt=0.02
    )
    if result.success:
        print(f"Advection-diffusion solved successfully")
        print(f"Grid points: {len(result.x)}")
        print(f"Time steps: {len(result.t)}")
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Snapshots
        time_indices = [0, len(result.t)//4, len(result.t)//2, 3*len(result.t)//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for t_idx, color in zip(time_indices, colors):
            ax1.plot(result.x, result.u[t_idx], color=color, linewidth=2,
                    label=f't = {result.t[t_idx]:.2f}')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Concentration u(x,t)')
        ax1.set_title('Advection-Diffusion: Snapshots')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Spacetime contour
        X, T = np.meshgrid(result.x, result.t)
        contour = ax2.contourf(X, T, result.u, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax2, label='Concentration')
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Time t')
        ax2.set_title('Spacetime Evolution')
        # Peak location vs time
        peak_locations = []
        for i in range(len(result.t)):
            peak_idx = np.argmax(result.u[i])
            peak_locations.append(result.x[peak_idx])
        ax3.plot(result.t, peak_locations, 'b-', linewidth=2)
        ax3.plot(result.t, result.t * velocity + 2, 'r--', linewidth=2, label='Theoretical')
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Peak location')
        ax3.set_title('Peak Transport')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # Mass conservation
        dx = result.x[1] - result.x[0]
        total_mass = np.sum(result.u, axis=1) * dx
        ax4.plot(result.t, total_mass, 'g-', linewidth=2)
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('Total mass')
        ax4.set_title('Mass Conservation')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Initial mass: {total_mass[0]:.6f}")
        print(f"Final mass: {total_mass[-1]:.6f}")
        print(f"Mass loss: {(total_mass[0] - total_mass[-1])/total_mass[0]*100:.2f}%")
        # Peclet number
        peclet = velocity * (result.x[-1] - result.x[0]) / diffusivity
        print(f"Peclet number: {peclet:.2f}")
    else:
        print(f"Advection-diffusion solve failed: {result.message}")
    print()
def example_5_nonlinear_pendulum():
    """Example 5: Nonlinear pendulum with large amplitude."""
    print("Example 5: Nonlinear pendulum")
    print("=" * 50)
    # Nonlinear pendulum: d²θ/dt² + (g/L)sin(θ) = 0
    # Convert to first-order system: θ' = ω, ω' = -(g/L)sin(θ)
    # Parameters
    g = 9.81  # gravity
    L = 1.0   # length
    omega0 = np.sqrt(g/L)  # small angle frequency
    def pendulum_ode(t, z):
        theta, omega = z
        dtheta_dt = omega
        domega_dt = -(g/L) * np.sin(theta)
        return np.array([dtheta_dt, domega_dt])
    # Different initial conditions
    initial_angles = [np.pi/6, np.pi/3, np.pi/2, 0.9*np.pi]  # 30°, 60°, 90°, 162°
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    periods = []
    colors = ['blue', 'green', 'red', 'purple']
    for i, theta0 in enumerate(initial_angles):
        initial_conditions = np.array([theta0, 0.0])  # Start from rest
        # Solve with adaptive stepping
        adaptive_stepper = AdaptiveTimeStepper(rtol=1e-10, atol=1e-12)
        result = adaptive_stepper.integrate(pendulum_ode, (0, 20), initial_conditions)
        if result.success:
            theta_solution = result.y[:, 0]
            omega_solution = result.y[:, 1]
            # Plot angle vs time
            ax1.plot(result.t, theta_solution * 180/np.pi, color=colors[i],
                    linewidth=2, label=f'θ₀ = {theta0*180/np.pi:.0f}°')
            # Phase portrait
            ax2.plot(theta_solution * 180/np.pi, omega_solution, color=colors[i],
                    linewidth=2, label=f'θ₀ = {theta0*180/np.pi:.0f}°')
            # Energy (should be conserved)
            kinetic = 0.5 * L**2 * omega_solution**2
            potential = g * L * (1 - np.cos(theta_solution))
            total_energy = kinetic + potential
            ax3.plot(result.t, total_energy, color=colors[i], linewidth=2,
                    label=f'θ₀ = {theta0*180/np.pi:.0f}°')
            # Estimate period
            # Find zero crossings with positive velocity
            zero_crossings = []
            for j in range(1, len(theta_solution)):
                if (theta_solution[j-1] < 0 and theta_solution[j] >= 0 and
                    omega_solution[j] > 0):
                    # Linear interpolation for more accurate crossing
                    t_cross = result.t[j-1] + (0 - theta_solution[j-1]) * \
                             (result.t[j] - result.t[j-1]) / (theta_solution[j] - theta_solution[j-1])
                    zero_crossings.append(t_cross)
            if len(zero_crossings) >= 2:
                period = np.mean(np.diff(zero_crossings))
                periods.append(period)
                print(f"θ₀ = {theta0*180/np.pi:3.0f}°: Period = {period:.3f} s "
                      f"(vs {2*np.pi/omega0:.3f} s small angle)")
            else:
                periods.append(np.nan)
    # Period vs amplitude
    ax4.plot([theta*180/np.pi for theta in initial_angles], periods, 'bo-', linewidth=2)
    ax4.axhline(y=2*np.pi/omega0, color='r', linestyle='--',
               label='Small angle period')
    ax4.set_xlabel('Initial angle (degrees)')
    ax4.set_ylabel('Period (s)')
    ax4.set_title('Period vs Amplitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # Format other plots
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Nonlinear Pendulum Motion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Angular velocity (rad/s)')
    ax2.set_title('Phase Portrait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J/kg)')
    ax3.set_title('Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print()
def example_6_fem_convergence():
    """Example 6: Finite element method convergence study."""
    print("Example 6: FEM convergence study")
    print("=" * 50)
    # Problem: -d²u/dx² = π²sin(πx), u(0) = u(1) = 0
    # Analytical solution: u(x) = sin(πx)
    def u_exact(x):
        return np.sin(np.pi * x)
    def source(x):
        return np.pi**2 * np.sin(np.pi * x)
    boundary_conditions = {'dirichlet': {0: 0.0, -1: 0.0}}  # u(0) = u(1) = 0
    # Convergence study
    element_counts = [10, 20, 40, 80, 160]
    convergence_results = fem_convergence_study(
        domain=(0, 1),
        u_exact=u_exact,
        boundary_conditions=boundary_conditions,
        source=source,
        element_counts=element_counts,
        element_type="linear"
    )
    if convergence_results and len(convergence_results['errors']) > 0:
        print(f"FEM convergence study completed")
        print(f"Convergence rate: {convergence_results['convergence_rate']:.2f}")
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Solution for finest mesh
        finest_result = solve_fem_poisson(
            domain=(0, 1),
            n_elements=element_counts[-1],
            boundary_conditions=boundary_conditions,
            source=source,
            element_type="linear"
        )
        if finest_result.success:
            u_analytical = u_exact(finest_result.nodes)
            ax1.plot(finest_result.nodes, u_analytical, 'k-', linewidth=2,
                    label='Analytical')
            ax1.plot(finest_result.nodes, finest_result.solution, 'r--',
                    linewidth=2, label='FEM')
            ax1.set_xlabel('Position x')
            ax1.set_ylabel('Solution u(x)')
            ax1.set_title('FEM Solution vs Analytical')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        # Convergence plot
        h_values = convergence_results['h_values']
        errors = convergence_results['errors']
        ax2.loglog(h_values, errors, 'bo-', linewidth=2, label='FEM Error')
        # Theoretical slope
        if len(h_values) >= 2:
            slope = convergence_results['convergence_rate']
            C = errors[0] / (h_values[0]**slope)
            theoretical = C * np.array(h_values)**slope
            ax2.loglog(h_values, theoretical, 'r--', linewidth=2,
                      label=f'O(h^{slope:.1f})')
        ax2.set_xlabel('Element size h')
        ax2.set_ylabel('L2 Error')
        ax2.set_title('FEM Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Print convergence data
        print("Element count | h        | L2 Error  | Rate")
        print("-" * 45)
        for i, (n_elem, h, error) in enumerate(zip(element_counts, h_values, errors)):
            if i == 0:
                print(f"{n_elem:11d} | {h:.6f} | {error:.2e} |     -")
            else:
                rate = np.log(errors[i-1]/error) / np.log(h_values[i-1]/h)
                print(f"{n_elem:11d} | {h:.6f} | {error:.2e} | {rate:.2f}")
    else:
        print("FEM convergence study failed")
    print()
def run_all_examples():
    """Run all intermediate examples."""
    print("Running Intermediate ODE/PDE Examples")
    print("=" * 60)
    print()
    example_1_adaptive_ode()
    example_2_predator_prey()
    example_3_2d_heat_equation()
    example_4_advection_diffusion()
    example_5_nonlinear_pendulum()
    example_6_fem_convergence()
    print("All intermediate examples completed!")
if __name__ == "__main__":
    run_all_examples()