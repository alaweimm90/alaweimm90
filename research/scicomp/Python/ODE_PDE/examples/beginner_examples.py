"""Beginner Examples for ODE and PDE Solving.
This module contains introductory examples demonstrating basic ODE and PDE
solving techniques using the SciComp ODE_PDE package.
Examples:
    - Simple first-order ODE
    - Second-order ODE (harmonic oscillator)
    - Basic heat equation
    - Simple wave equation
    - Poisson equation with Dirichlet BC
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
# Import our ODE/PDE modules
from ..ode_solvers import ExplicitEuler, RungeKutta4
from ..pde_solvers import solve_heat_equation, solve_wave_equation, solve_pde
from ..visualization import plot_ode_solution, plot_pde_solution
def example_1_exponential_decay():
    """Example 1: Exponential decay ODE dy/dt = -λy."""
    print("Example 1: Exponential decay ODE")
    print("=" * 40)
    # Problem: dy/dt = -λy, y(0) = y0
    # Analytical solution: y(t) = y0 * exp(-λt)
    # Parameters
    lambda_param = 2.0
    y0 = 1.0
    t_span = (0, 2)
    dt = 0.01
    # Define ODE
    def dydt(t, y):
        return -lambda_param * y
    # Analytical solution
    def analytical_solution(t):
        return y0 * np.exp(-lambda_param * t)
    # Solve with Euler method
    euler_solver = ExplicitEuler()
    result_euler = euler_solver.solve(dydt, y0, t_span, dt)
    # Solve with RK4
    rk4_solver = RungeKutta4()
    result_rk4 = rk4_solver.solve(dydt, y0, t_span, dt)
    # Analytical solution
    t_exact = result_euler.t
    y_exact = analytical_solution(t_exact)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t_exact, y_exact, 'k-', label='Analytical', linewidth=2)
    plt.plot(result_euler.t, result_euler.y, 'r--', label='Euler', linewidth=2)
    plt.plot(result_rk4.t, result_rk4.y, 'b:', label='RK4', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Solution y(t)')
    plt.title('Exponential Decay: dy/dt = -λy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # Error analysis
    error_euler = np.abs(result_euler.y - y_exact)
    error_rk4 = np.abs(result_rk4.y - y_exact)
    print(f"Final time: {result_euler.t[-1]:.2f}")
    print(f"Euler final error: {error_euler[-1]:.6f}")
    print(f"RK4 final error: {error_rk4[-1]:.6f}")
    print(f"Error ratio (Euler/RK4): {error_euler[-1]/error_rk4[-1]:.1f}")
    print()
def example_2_harmonic_oscillator():
    """Example 2: Harmonic oscillator d²y/dt² + ω²y = 0."""
    print("Example 2: Harmonic oscillator")
    print("=" * 40)
    # Convert to first-order system:
    # y1 = y, y2 = dy/dt
    # dy1/dt = y2
    # dy2/dt = -ω²y1
    # Parameters
    omega = 2.0  # Angular frequency
    y0 = 1.0     # Initial position
    v0 = 0.0     # Initial velocity
    t_span = (0, 2*np.pi/omega)  # One period
    dt = 0.01
    # Define system of ODEs
    def dydt(t, y):
        y1, y2 = y
        return np.array([y2, -omega**2 * y1])
    # Initial conditions
    initial_conditions = np.array([y0, v0])
    # Analytical solution
    def analytical_solution(t):
        y = y0 * np.cos(omega * t) + (v0/omega) * np.sin(omega * t)
        dydt = -y0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        return np.array([y, dydt])
    # Solve with RK4
    rk4_solver = RungeKutta4()
    result = rk4_solver.solve(dydt, initial_conditions, t_span, dt)
    # Analytical solution
    t_exact = result.t
    y_exact = np.array([analytical_solution(t) for t in t_exact]).T
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Position vs time
    ax1.plot(t_exact, y_exact[0], 'k-', label='Analytical', linewidth=2)
    ax1.plot(result.t, result.y[:, 0], 'r--', label='RK4', linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Position y(t)')
    ax1.set_title('Harmonic Oscillator: Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Phase portrait
    ax2.plot(y_exact[0], y_exact[1], 'k-', label='Analytical', linewidth=2)
    ax2.plot(result.y[:, 0], result.y[:, 1], 'r--', label='RK4', linewidth=2)
    ax2.set_xlabel('Position y')
    ax2.set_ylabel('Velocity dy/dt')
    ax2.set_title('Phase Portrait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    plt.tight_layout()
    plt.show()
    # Energy conservation check
    kinetic_energy = 0.5 * result.y[:, 1]**2
    potential_energy = 0.5 * omega**2 * result.y[:, 0]**2
    total_energy = kinetic_energy + potential_energy
    print(f"Initial energy: {total_energy[0]:.6f}")
    print(f"Final energy: {total_energy[-1]:.6f}")
    print(f"Energy variation: {np.std(total_energy):.8f}")
    print()
def example_3_heat_equation_1d():
    """Example 3: 1D heat equation ∂u/∂t = α∂²u/∂x²."""
    print("Example 3: 1D Heat equation")
    print("=" * 40)
    # Problem setup
    domain = {'x': np.linspace(0, 1, 51)}
    thermal_diffusivity = 0.1
    # Boundary conditions: u(0,t) = 0, u(1,t) = 0
    boundary_conditions = {
        'dirichlet': {0: 0.0, 50: 0.0}  # Node indices and values
    }
    # Initial condition: u(x,0) = sin(πx)
    def initial_condition(x):
        return np.sin(np.pi * x)
    # Solve
    result = solve_heat_equation(
        domain=domain,
        boundary_conditions=boundary_conditions,
        initial_condition=initial_condition,
        time_span=(0, 0.5),
        thermal_diffusivity=thermal_diffusivity
    )
    if result.success:
        print(f"Heat equation solved successfully")
        print(f"Grid points: {len(result.x)}")
        print(f"Time steps: {len(result.t)}")
        # Plot solution at different times
        plt.figure(figsize=(12, 8))
        # Snapshots
        time_indices = [0, len(result.t)//4, len(result.t)//2, 3*len(result.t)//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
            plt.subplot(2, 2, 1)
            plt.plot(result.x, result.u[t_idx], color=color, linewidth=2,
                    label=f't = {result.t[t_idx]:.3f}')
        plt.xlabel('Position x')
        plt.ylabel('Temperature u(x,t)')
        plt.title('Heat Equation: Temperature Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Contour plot
        plt.subplot(2, 2, 2)
        X, T = np.meshgrid(result.x, result.t)
        contour = plt.contourf(X, T, result.u, levels=20, cmap='hot')
        plt.colorbar(contour, label='Temperature')
        plt.xlabel('Position x')
        plt.ylabel('Time t')
        plt.title('Heat Equation: Spacetime Evolution')
        # Central temperature vs time
        plt.subplot(2, 2, 3)
        center_idx = len(result.x) // 2
        plt.plot(result.t, result.u[:, center_idx], 'b-', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Temperature at center')
        plt.title('Temperature at x = 0.5')
        plt.grid(True, alpha=0.3)
        # Analytical comparison (for this simple case)
        plt.subplot(2, 2, 4)
        # Analytical solution: u(x,t) = sin(πx)exp(-π²αt)
        u_analytical = np.sin(np.pi * result.x) * np.exp(-np.pi**2 * thermal_diffusivity * result.t[-1])
        plt.plot(result.x, result.u[-1], 'r-', linewidth=2, label='Numerical')
        plt.plot(result.x, u_analytical, 'k--', linewidth=2, label='Analytical')
        plt.xlabel('Position x')
        plt.ylabel('Temperature')
        plt.title(f'Final Solution at t = {result.t[-1]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Error analysis
        error = np.abs(result.u[-1] - u_analytical)
        print(f"Maximum error: {np.max(error):.6f}")
        print(f"L2 error: {np.sqrt(np.mean(error**2)):.6f}")
    else:
        print(f"Heat equation solve failed: {result.message}")
    print()
def example_4_poisson_equation():
    """Example 4: Poisson equation -∇²u = f with Dirichlet BC."""
    print("Example 4: Poisson equation")
    print("=" * 40)
    # Problem: -d²u/dx² = f(x), u(0) = 0, u(1) = 0
    # Source term: f(x) = π²sin(πx)
    # Analytical solution: u(x) = sin(πx)
    # Domain and grid
    domain = {'x': np.linspace(0, 1, 101)}
    # Boundary conditions
    boundary_conditions = {
        'dirichlet': {0: 0.0, 100: 0.0}  # u(0) = 0, u(1) = 0
    }
    # Source term
    def source_term(x):
        return np.pi**2 * np.sin(np.pi * x)
    # Solve Poisson equation
    result = solve_pde(
        pde_type='poisson',
        domain=domain,
        boundary_conditions=boundary_conditions,
        source_term=source_term
    )
    if result.success:
        print(f"Poisson equation solved successfully")
        print(f"Grid points: {len(result.x)}")
        # Analytical solution
        u_analytical = np.sin(np.pi * result.x)
        # Plot results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(result.x, result.u, 'b-', linewidth=2, label='Numerical')
        plt.plot(result.x, u_analytical, 'r--', linewidth=2, label='Analytical')
        plt.xlabel('Position x')
        plt.ylabel('Solution u(x)')
        plt.title('Poisson Equation Solution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Error plot
        plt.subplot(1, 2, 2)
        error = np.abs(result.u - u_analytical)
        plt.semilogy(result.x, error, 'g-', linewidth=2)
        plt.xlabel('Position x')
        plt.ylabel('Absolute error')
        plt.title('Solution Error')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Error analysis
        max_error = np.max(error)
        l2_error = np.sqrt(np.mean(error**2))
        print(f"Maximum error: {max_error:.8f}")
        print(f"L2 error: {l2_error:.8f}")
        print(f"Relative L2 error: {l2_error/np.sqrt(np.mean(u_analytical**2)):.8f}")
    else:
        print(f"Poisson equation solve failed: {result.message}")
    print()
def example_5_wave_equation():
    """Example 5: 1D wave equation ∂²u/∂t² = c²∂²u/∂x²."""
    print("Example 5: 1D Wave equation")
    print("=" * 40)
    # Problem setup
    domain = {'x': np.linspace(0, 1, 101)}
    wave_speed = 1.0
    # Boundary conditions: u(0,t) = 0, u(1,t) = 0
    boundary_conditions = {
        'dirichlet': {0: 0.0, 100: 0.0}
    }
    # Initial conditions
    def initial_condition(x):
        # Gaussian pulse
        return np.exp(-50 * (x - 0.3)**2)
    def initial_velocity(x):
        return np.zeros_like(x)
    # Solve wave equation
    result = solve_wave_equation(
        domain=domain,
        boundary_conditions=boundary_conditions,
        initial_condition=initial_condition,
        initial_velocity=initial_velocity,
        time_span=(0, 2.0),
        wave_speed=wave_speed
    )
    if result.success:
        print(f"Wave equation solved successfully")
        print(f"Grid points: {len(result.x)}")
        print(f"Time steps: {len(result.t)}")
        # Plot solution snapshots
        plt.figure(figsize=(12, 8))
        # Snapshots
        plt.subplot(2, 2, 1)
        time_indices = [0, len(result.t)//6, len(result.t)//3, len(result.t)//2]
        colors = ['blue', 'green', 'orange', 'red']
        for t_idx, color in zip(time_indices, colors):
            plt.plot(result.x, result.u[t_idx], color=color, linewidth=2,
                    label=f't = {result.t[t_idx]:.3f}')
        plt.xlabel('Position x')
        plt.ylabel('Displacement u(x,t)')
        plt.title('Wave Equation: Snapshots')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Spacetime plot
        plt.subplot(2, 2, 2)
        X, T = np.meshgrid(result.x, result.t)
        contour = plt.contourf(X, T, result.u, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label='Displacement')
        plt.xlabel('Position x')
        plt.ylabel('Time t')
        plt.title('Wave Equation: Spacetime')
        # Wave at specific position
        plt.subplot(2, 2, 3)
        pos_idx = len(result.x) // 4  # x = 0.25
        plt.plot(result.t, result.u[:, pos_idx], 'b-', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Displacement')
        plt.title(f'Wave at x = {result.x[pos_idx]:.2f}')
        plt.grid(True, alpha=0.3)
        # Energy (should be conserved)
        plt.subplot(2, 2, 4)
        # Approximate energy calculation
        dx = result.x[1] - result.x[0]
        dt = result.t[1] - result.t[0]
        kinetic_energy = []
        potential_energy = []
        for i in range(1, len(result.t)):
            # Kinetic energy: (1/2) ∫ (∂u/∂t)² dx
            dudt = (result.u[i] - result.u[i-1]) / dt
            ke = 0.5 * np.sum(dudt**2) * dx
            kinetic_energy.append(ke)
            # Potential energy: (1/2) c² ∫ (∂u/∂x)² dx
            dudx = np.gradient(result.u[i], dx)
            pe = 0.5 * wave_speed**2 * np.sum(dudx**2) * dx
            potential_energy.append(pe)
        total_energy = np.array(kinetic_energy) + np.array(potential_energy)
        plt.plot(result.t[1:], kinetic_energy, 'r-', label='Kinetic', linewidth=2)
        plt.plot(result.t[1:], potential_energy, 'b-', label='Potential', linewidth=2)
        plt.plot(result.t[1:], total_energy, 'k--', label='Total', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Energy')
        plt.title('Energy Conservation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Initial total energy: {total_energy[0]:.6f}")
        print(f"Final total energy: {total_energy[-1]:.6f}")
        print(f"Energy variation: {np.std(total_energy):.8f}")
    else:
        print(f"Wave equation solve failed: {result.message}")
    print()
def run_all_examples():
    """Run all beginner examples."""
    print("Running Beginner ODE/PDE Examples")
    print("=" * 50)
    print()
    example_1_exponential_decay()
    example_2_harmonic_oscillator()
    example_3_heat_equation_1d()
    example_4_poisson_equation()
    example_5_wave_equation()
    print("All beginner examples completed!")
if __name__ == "__main__":
    run_all_examples()