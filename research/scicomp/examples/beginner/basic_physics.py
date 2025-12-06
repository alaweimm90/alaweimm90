#!/usr/bin/env python3
"""
Basic Physics Examples
======================
Simple physics examples using the SciComp.
Perfect for students learning computational physics or researchers
getting started with numerical methods.
Topics covered:
- Classical mechanics (projectile motion, oscillators)
- Electromagnetism (electric fields, circuits)
- Quantum mechanics (particle in a box, tunneling)
- Thermodynamics (gas laws, heat engines)
Author: UC Berkeley SciComp Team
Date: 2025
License: MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def main():
    """Run basic physics demonstrations."""
    print("⚛️  Basic Physics with Berkeley SciComp")
    print("=" * 45)
    # Create output directory
    output_dir = Path("output/basic_physics")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Classical mechanics
    print("\n🌍 Classical Mechanics")
    demo_projectile_motion(output_dir)
    demo_harmonic_oscillator(output_dir)
    # Electromagnetism
    print("\n⚡ Electromagnetism")
    demo_electric_field(output_dir)
    demo_rc_circuit(output_dir)
    # Quantum mechanics
    print("\n🔬 Quantum Mechanics")
    demo_particle_in_box(output_dir)
    demo_quantum_tunneling(output_dir)
    # Thermodynamics
    print("\n🌡️  Thermodynamics")
    demo_ideal_gas(output_dir)
    demo_heat_engine(output_dir)
    print(f"\n✅ All demonstrations completed!")
    print(f"📁 Results saved to: {output_dir}")
def demo_projectile_motion(output_dir: Path):
    """Demonstrate projectile motion."""
    print("   Projectile motion simulation...")
    # Parameters
    v0 = 50.0  # m/s initial velocity
    angle = 45.0  # degrees
    g = 9.81  # m/s^2
    # Convert angle to radians
    theta = np.radians(angle)
    # Time of flight
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, 100)
    # Position equations
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    # Velocity components
    vx = v0 * np.cos(theta) * np.ones_like(t)
    vy = v0 * np.sin(theta) - g * t
    # Calculate trajectory properties
    max_height = (v0 * np.sin(theta))**2 / (2 * g)
    range_proj = v0**2 * np.sin(2 * theta) / g
    print(f"      Initial velocity: {v0} m/s at {angle}°")
    print(f"      Maximum height: {max_height:.1f} m")
    print(f"      Range: {range_proj:.1f} m")
    print(f"      Time of flight: {t_flight:.2f} s")
    # Plot trajectory
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Trajectory
    ax1.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    ax1.axhline(y=0, color='brown', linewidth=3, label='Ground')
    ax1.plot(x[np.argmax(y)], max_height, 'ro', markersize=8, label='Max Height')
    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Projectile Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    # Velocity components vs time
    ax2.plot(t, vx, 'r-', linewidth=2, label='vₓ (horizontal)')
    ax2.plot(t, vy, 'b-', linewidth=2, label='vᵧ (vertical)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Height vs time
    ax3.plot(t, y, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Height vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    # Energy analysis
    kinetic_energy = 0.5 * (vx**2 + vy**2)  # Mass = 1 kg
    potential_energy = g * y  # Mass = 1 kg
    total_energy = kinetic_energy + potential_energy
    ax4.plot(t, kinetic_energy, 'r-', linewidth=2, label='Kinetic')
    ax4.plot(t, potential_energy, 'b-', linewidth=2, label='Potential')
    ax4.plot(t, total_energy, 'k--', linewidth=2, label='Total')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy (J/kg)')
    ax4.set_title('Energy Conservation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "projectile_motion.png", dpi=150, bbox_inches='tight')
    plt.close()
def demo_harmonic_oscillator(output_dir: Path):
    """Demonstrate simple harmonic oscillator."""
    print("   Simple harmonic oscillator...")
    # Parameters
    omega = 2.0  # rad/s angular frequency
    A = 1.0  # amplitude
    phi = np.pi/4  # phase
    # Time array
    t = np.linspace(0, 4*np.pi/omega, 200)
    # Position, velocity, acceleration
    x = A * np.cos(omega * t + phi)
    v = -A * omega * np.sin(omega * t + phi)
    a = -A * omega**2 * np.cos(omega * t + phi)
    print(f"      Angular frequency: ω = {omega} rad/s")
    print(f"      Period: T = {2*np.pi/omega:.2f} s")
    print(f"      Frequency: f = {omega/(2*np.pi):.2f} Hz")
    # Phase space plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Position vs time
    ax1.plot(t, x, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position vs Time')
    ax1.grid(True, alpha=0.3)
    # Phase space (position vs velocity)
    ax2.plot(x, v, 'r-', linewidth=2)
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Phase Space')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    # All three quantities vs time
    ax3.plot(t, x, 'b-', linewidth=2, label='Position')
    ax3.plot(t, v/omega, 'r-', linewidth=2, label='Velocity/ω')
    ax3.plot(t, a/omega**2, 'g-', linewidth=2, label='Acceleration/ω²')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.set_title('Position, Velocity, Acceleration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Energy vs time
    kinetic = 0.5 * v**2  # Mass = 1 kg
    potential = 0.5 * omega**2 * x**2  # Spring constant k = ω²
    total = kinetic + potential
    ax4.plot(t, kinetic, 'r-', linewidth=2, label='Kinetic')
    ax4.plot(t, potential, 'b-', linewidth=2, label='Potential')
    ax4.plot(t, total, 'k--', linewidth=2, label='Total')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy (J)')
    ax4.set_title('Energy in SHO')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "harmonic_oscillator.png", dpi=150, bbox_inches='tight')
    plt.close()
def demo_electric_field(output_dir: Path):
    """Demonstrate electric field of point charges."""
    print("   Electric field visualization...")
    # Set up grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    # Point charges: [charge, x_position, y_position]
    charges = [
        [1.0, -1.0, 0.0],   # Positive charge
        [-1.0, 1.0, 0.0],   # Negative charge
    ]
    # Calculate electric field
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    V = np.zeros_like(X)  # Electric potential
    k = 8.99e9  # Coulomb's constant (simplified to 1 for visualization)
    k = 1.0  # For cleaner plots
    for charge, qx, qy in charges:
        # Distance from charge to each grid point
        R = np.sqrt((X - qx)**2 + (Y - qy)**2 + 0.1**2)  # Small offset to avoid singularity
        # Electric field components
        Ex += k * charge * (X - qx) / R**3
        Ey += k * charge * (Y - qy) / R**3
        # Electric potential
        V += k * charge / R
    print(f"      Simulating {len(charges)} point charges")
    print(f"      Positive charge at (-1, 0)")
    print(f"      Negative charge at (1, 0)")
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Electric field vectors
    ax1.quiver(X, Y, Ex, Ey, np.sqrt(Ex**2 + Ey**2), cmap='viridis', scale=50)
    # Mark charge positions
    for charge, qx, qy in charges:
        color = 'red' if charge > 0 else 'blue'
        marker = '+' if charge > 0 else 'o'
        ax1.plot(qx, qy, marker, markersize=15, color=color, markeredgewidth=3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Electric Field Vectors')
    ax1.set_aspect('equal')
    # Electric potential contours
    contours = ax2.contour(X, Y, V, levels=20, colors='black', alpha=0.6)
    ax2.contourf(X, Y, V, levels=50, cmap='RdBu_r')
    ax2.clabel(contours, inline=True, fontsize=8)
    # Mark charges
    for charge, qx, qy in charges:
        color = 'red' if charge > 0 else 'blue'
        marker = '+' if charge > 0 else 'o'
        ax2.plot(qx, qy, marker, markersize=15, color=color, markeredgewidth=3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Electric Potential')
    ax2.set_aspect('equal')
    # Field lines (simplified streamplot)
    ax3.streamplot(X, Y, Ex, Ey, density=2, color='blue', linewidth=1.5)
    # Mark charges
    for charge, qx, qy in charges:
        color = 'red' if charge > 0 else 'blue'
        marker = '+' if charge > 0 else 'o'
        ax3.plot(qx, qy, marker, markersize=15, color=color, markeredgewidth=3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Electric Field Lines')
    ax3.set_aspect('equal')
    # Field magnitude along x-axis (y=0)
    x_line = np.linspace(-3, 3, 200)
    y_line = 0
    Ex_line = np.zeros_like(x_line)
    for charge, qx, qy in charges:
        R_line = np.sqrt((x_line - qx)**2 + (y_line - qy)**2 + 0.01**2)
        Ex_line += k * charge * (x_line - qx) / R_line**3
    ax4.plot(x_line, Ex_line, 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Ex (along y=0)')
    ax4.set_title('Electric Field along x-axis')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "electric_field.png", dpi=150, bbox_inches='tight')
    plt.close()
def demo_rc_circuit(output_dir: Path):
    """Demonstrate RC circuit charging and discharging."""
    print("   RC circuit analysis...")
    # Circuit parameters
    R = 1000.0  # Resistance in Ohms
    C = 1e-6    # Capacitance in Farads
    V0 = 5.0    # Supply voltage in Volts
    # Time constant
    tau = R * C
    print(f"      Resistance: R = {R/1000:.1f} kΩ")
    print(f"      Capacitance: C = {C*1e6:.1f} µF")
    print(f"      Time constant: τ = RC = {tau*1000:.1f} ms")
    # Time arrays
    t1 = np.linspace(0, 5*tau, 200)  # Charging
    t2 = np.linspace(5*tau, 10*tau, 200)  # Discharging
    # Charging phase (0 to 5τ)
    Vc1 = V0 * (1 - np.exp(-t1/tau))  # Capacitor voltage
    Ic1 = (V0/R) * np.exp(-t1/tau)    # Capacitor current
    # Discharging phase (5τ to 10τ)
    t2_shifted = t2 - 5*tau
    Vc2 = V0 * np.exp(-t2_shifted/tau)  # Capacitor voltage
    Ic2 = -(V0/R) * np.exp(-t2_shifted/tau)  # Capacitor current (negative)
    # Combined arrays
    t_total = np.concatenate([t1, t2])
    Vc_total = np.concatenate([Vc1, Vc2])
    Ic_total = np.concatenate([Ic1, Ic2])
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Capacitor voltage vs time
    ax1.plot(t_total*1000, Vc_total, 'b-', linewidth=2, label='Vc')
    ax1.axhline(y=V0, color='r', linestyle='--', alpha=0.7, label='Supply voltage')
    ax1.axhline(y=V0*0.632, color='g', linestyle='--', alpha=0.7, label='63.2% of V₀')
    ax1.axvline(x=tau*1000, color='k', linestyle=':', alpha=0.7, label='τ')
    ax1.axvline(x=5*tau*1000, color='orange', linestyle=':', alpha=0.7, label='Switch time')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Capacitor Voltage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Capacitor current vs time
    ax2.plot(t_total*1000, Ic_total*1000, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=5*tau*1000, color='orange', linestyle=':', alpha=0.7, label='Switch time')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Capacitor Current')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Energy stored in capacitor
    energy = 0.5 * C * Vc_total**2
    max_energy = 0.5 * C * V0**2
    ax3.plot(t_total*1000, energy*1e6, 'g-', linewidth=2)
    ax3.axhline(y=max_energy*1e6, color='r', linestyle='--', alpha=0.7, label='Maximum energy')
    ax3.axvline(x=5*tau*1000, color='orange', linestyle=':', alpha=0.7, label='Switch time')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Energy (µJ)')
    ax3.set_title('Energy Stored in Capacitor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Power vs time
    power = Vc_total * Ic_total
    ax4.plot(t_total*1000, power*1000, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=5*tau*1000, color='orange', linestyle=':', alpha=0.7, label='Switch time')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Power (mW)')
    ax4.set_title('Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rc_circuit.png", dpi=150, bbox_inches='tight')
    plt.close()
def demo_particle_in_box(output_dir: Path):
    """Demonstrate particle in a 1D infinite potential well."""
    print("   Particle in a box quantum mechanics...")
    # Box parameters
    L = 1.0  # Box length
    n_points = 1000
    x = np.linspace(0, L, n_points)
    # Energy levels and wavefunctions for first few states
    n_states = 4
    colors = ['blue', 'red', 'green', 'orange']
    print(f"      1D infinite potential well, L = {L}")
    print(f"      Energy levels: En = n²π²ℏ²/(2mL²)")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Plot wavefunctions
    for n in range(1, n_states + 1):
        # Normalized wavefunction
        psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        # Energy (in units of π²ℏ²/(2mL²))
        E_n = n**2
        ax1.plot(x, psi + E_n, color=colors[n-1], linewidth=2, label=f'n={n}')
        ax1.axhline(y=E_n, color=colors[n-1], linestyle='--', alpha=0.5)
    ax1.set_xlabel('Position (L)')
    ax1.set_ylabel('ψ(x) + En')
    ax1.set_title('Energy Levels and Wavefunctions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L)
    # Plot probability densities
    for n in range(1, n_states + 1):
        psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        prob_density = psi**2
        ax2.plot(x, prob_density, color=colors[n-1], linewidth=2, label=f'|ψ_{n}|²')
        ax2.fill_between(x, prob_density, alpha=0.2, color=colors[n-1])
    ax2.set_xlabel('Position (L)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Probability Densities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, L)
    # Energy level diagram
    n_levels = np.arange(1, 8)
    E_levels = n_levels**2
    for n, E in zip(n_levels, E_levels):
        ax3.plot([0, 1], [E, E], 'b-', linewidth=3)
        ax3.text(1.1, E, f'n={n}', verticalalignment='center')
    ax3.set_xlim(-0.2, 2)
    ax3.set_ylim(0, 50)
    ax3.set_ylabel('Energy (π²ℏ²/2mL²)')
    ax3.set_title('Energy Level Diagram')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3)
    # Expectation value of position vs quantum number
    n_vals = np.arange(1, 11)
    x_expectation = L/2 * np.ones_like(n_vals)  # Always L/2 for particle in box
    # Position uncertainty
    x_squared_expectation = L**2 * (1/3 - 1/(2*np.pi**2*n_vals**2))
    x_uncertainty = np.sqrt(x_squared_expectation - (L/2)**2)
    ax4.plot(n_vals, x_expectation, 'bo-', linewidth=2, markersize=6, label='⟨x⟩')
    ax4.plot(n_vals, x_uncertainty, 'r^-', linewidth=2, markersize=6, label='Δx')
    ax4.axhline(y=L/2, color='b', linestyle='--', alpha=0.5, label='L/2')
    ax4.set_xlabel('Quantum Number n')
    ax4.set_ylabel('Position (L)')
    ax4.set_title('Position Statistics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "particle_in_box.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Ground state energy: E₁ = π²ℏ²/(2mL²)")
    print(f"      All states have ⟨x⟩ = L/2 (symmetry)")
def demo_quantum_tunneling(output_dir: Path):
    """Demonstrate quantum tunneling through a barrier."""
    print("   Quantum tunneling simulation...")
    # Barrier parameters
    V0 = 2.0  # Barrier height (in energy units)
    a = 0.5   # Barrier width
    L = 4.0   # Total length
    # Energy of incident particle
    E = 0.5   # Less than V0, so classically forbidden
    print(f"      Barrier height: V₀ = {V0}")
    print(f"      Barrier width: a = {a}")
    print(f"      Particle energy: E = {E} < V₀")
    print(f"      Classical prediction: 100% reflection")
    # Position array
    x = np.linspace(-L/2, L/2, 1000)
    # Wave numbers
    k1 = np.sqrt(2 * E)  # Outside barrier (ℏ²/2m = 1)
    k2 = np.sqrt(2 * (V0 - E))  # Inside barrier (imaginary momentum)
    # Transmission coefficient (quantum mechanical result)
    transmission = 1 / (1 + (V0**2 * np.sinh(k2 * a)**2) / (4 * E * (V0 - E)))
    reflection = 1 - transmission
    print(f"      Quantum transmission probability: T = {transmission:.4f}")
    print(f"      Quantum reflection probability: R = {reflection:.4f}")
    # Construct wavefunction (simplified)
    psi = np.zeros_like(x, dtype=complex)
    # Region I: x < -a/2 (incident + reflected waves)
    region1 = x < -a/2
    psi[region1] = np.exp(1j * k1 * x[region1]) + np.sqrt(reflection) * np.exp(-1j * k1 * x[region1])
    # Region II: -a/2 < x < a/2 (evanescent waves)
    region2 = (x >= -a/2) & (x <= a/2)
    # Simplified: just show exponential decay
    psi[region2] = np.exp(-k2 * (x[region2] + a/2)) * np.exp(1j * k1 * (-a/2))
    # Region III: x > a/2 (transmitted wave)
    region3 = x > a/2
    psi[region3] = np.sqrt(transmission) * np.exp(1j * k1 * x[region3])
    # Potential
    V = np.zeros_like(x)
    barrier_region = (x >= -a/2) & (x <= a/2)
    V[barrier_region] = V0
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Potential and energy
    ax1.plot(x, V, 'k-', linewidth=3, label='Potential V(x)')
    ax1.axhline(y=E, color='r', linestyle='--', linewidth=2, label=f'Energy E = {E}')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Energy')
    ax1.set_title('Potential Barrier')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 2.5)
    # Real part of wavefunction
    ax2.plot(x, np.real(psi), 'b-', linewidth=2, label='Re[ψ(x)]')
    ax2.plot(x, V/5, 'k-', linewidth=2, alpha=0.5, label='V(x)/5')
    ax2.axvline(x=-a/2, color='k', linestyle='--', alpha=0.7)
    ax2.axvline(x=a/2, color='k', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Wavefunction')
    ax2.set_title('Real Part of Wavefunction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Probability density
    prob_density = np.abs(psi)**2
    ax3.plot(x, prob_density, 'r-', linewidth=2, label='|ψ(x)|²')
    ax3.fill_between(x, prob_density, alpha=0.3, color='red')
    ax3.plot(x, V/5, 'k-', linewidth=2, alpha=0.5, label='V(x)/5')
    ax3.axvline(x=-a/2, color='k', linestyle='--', alpha=0.7)
    ax3.axvline(x=a/2, color='k', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Probability Density |ψ(x)|²')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Transmission vs energy
    E_range = np.linspace(0.1, 2*V0, 200)
    T_range = np.zeros_like(E_range)
    for i, E_val in enumerate(E_range):
        if E_val < V0:
            k2_val = np.sqrt(2 * (V0 - E_val))
            T_range[i] = 1 / (1 + (V0**2 * np.sinh(k2_val * a)**2) / (4 * E_val * (V0 - E_val)))
        else:
            # E > V0: different formula (transmission resonances)
            k2_val = np.sqrt(2 * (E_val - V0))
            T_range[i] = 1 / (1 + (V0**2 * np.sin(k2_val * a)**2) / (4 * E_val * (E_val - V0)))
    ax4.plot(E_range, T_range, 'g-', linewidth=2, label='Transmission T(E)')
    ax4.axvline(x=V0, color='k', linestyle='--', alpha=0.7, label=f'V₀ = {V0}')
    ax4.axvline(x=E, color='r', linestyle='--', alpha=0.7, label=f'E = {E}')
    ax4.plot(E, transmission, 'ro', markersize=8)
    ax4.set_xlabel('Energy')
    ax4.set_ylabel('Transmission Probability')
    ax4.set_title('Transmission vs Energy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "quantum_tunneling.png", dpi=150, bbox_inches='tight')
    plt.close()
def demo_ideal_gas(output_dir: Path):
    """Demonstrate ideal gas law and kinetic theory."""
    print("   Ideal gas law and molecular motion...")
    # Constants
    R = 8.314  # J/(mol·K) Gas constant
    N_A = 6.022e23  # Avogadro's number
    k_B = R / N_A  # Boltzmann constant
    # Gas parameters
    n_moles = 1.0  # Amount of gas
    T_range = np.linspace(200, 400, 100)  # Temperature range (K)
    V_range = np.linspace(0.01, 0.05, 100)  # Volume range (m³)
    print(f"      Amount of gas: n = {n_moles} mol")
    print(f"      Temperature range: {T_range[0]:.0f} - {T_range[-1]:.0f} K")
    # PV = nRT calculations
    P_fixed_V = n_moles * R * T_range / 0.024  # Fixed volume = 24 L
    P_fixed_T = n_moles * R * 298 / V_range    # Fixed temperature = 298 K
    # Maxwell-Boltzmann distribution for molecular speeds
    def maxwell_boltzmann(v, T, M):
        """Maxwell-Boltzmann speed distribution."""
        return 4 * np.pi * (M / (2 * np.pi * R * T))**(3/2) * v**2 * np.exp(-M * v**2 / (2 * R * T))
    # For nitrogen gas (M = 28.014 g/mol = 0.028014 kg/mol)
    M_N2 = 0.028014  # kg/mol
    v = np.linspace(0, 1500, 200)  # Speed range (m/s)
    temperatures = [200, 300, 400]  # K
    colors = ['blue', 'red', 'green']
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # Pressure vs Temperature (constant volume)
    ax1.plot(T_range, P_fixed_V/1000, 'b-', linewidth=2)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Pressure (kPa)')
    ax1.set_title('Pressure vs Temperature (V = constant)')
    ax1.grid(True, alpha=0.3)
    # Pressure vs Volume (constant temperature)
    ax2.plot(V_range*1000, P_fixed_T/1000, 'r-', linewidth=2)
    ax2.set_xlabel('Volume (L)')
    ax2.set_ylabel('Pressure (kPa)')
    ax2.set_title('Pressure vs Volume (T = constant)')
    ax2.grid(True, alpha=0.3)
    # Maxwell-Boltzmann distributions
    for T, color in zip(temperatures, colors):
        f_v = maxwell_boltzmann(v, T, M_N2)
        ax3.plot(v, f_v, color=color, linewidth=2, label=f'T = {T} K')
        # Most probable speed
        v_mp = np.sqrt(2 * R * T / M_N2)
        ax3.axvline(x=v_mp, color=color, linestyle='--', alpha=0.7)
    ax3.set_xlabel('Speed (m/s)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Maxwell-Boltzmann Speed Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Average speeds vs temperature
    T_speeds = np.linspace(200, 400, 50)
    v_mp_array = np.sqrt(2 * R * T_speeds / M_N2)  # Most probable
    v_avg_array = np.sqrt(8 * R * T_speeds / (np.pi * M_N2))  # Average
    v_rms_array = np.sqrt(3 * R * T_speeds / M_N2)  # RMS
    ax4.plot(T_speeds, v_mp_array, 'b-', linewidth=2, label='Most probable')
    ax4.plot(T_speeds, v_avg_array, 'r-', linewidth=2, label='Average')
    ax4.plot(T_speeds, v_rms_array, 'g-', linewidth=2, label='RMS')
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Molecular Speeds vs Temperature')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ideal_gas.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Print some calculations
    T = 298  # K (room temperature)
    v_avg = np.sqrt(8 * R * T / (np.pi * M_N2))
    v_rms = np.sqrt(3 * R * T / M_N2)
    print(f"      At room temperature (T = {T} K):")
    print(f"      Average speed of N₂: {v_avg:.0f} m/s")
    print(f"      RMS speed of N₂: {v_rms:.0f} m/s")
def demo_heat_engine(output_dir: Path):
    """Demonstrate heat engine thermodynamic cycle."""
    print("   Carnot heat engine cycle...")
    # Carnot cycle parameters
    T_hot = 500  # K (hot reservoir)
    T_cold = 300  # K (cold reservoir)
    V1 = 1.0  # Initial volume
    V2 = 2.0  # Volume after isothermal expansion
    # Calculate efficiency
    eta_carnot = 1 - T_cold/T_hot
    print(f"      Hot reservoir: T_h = {T_hot} K")
    print(f"      Cold reservoir: T_c = {T_cold} K")
    print(f"      Carnot efficiency: η = {eta_carnot:.3f} = {eta_carnot*100:.1f}%")
    # Volume ratios for adiabatic processes
    gamma = 1.4  # Heat capacity ratio for diatomic gas
    V3 = V2 * (T_hot/T_cold)**(1/(gamma-1))
    V4 = V1 * (T_hot/T_cold)**(1/(gamma-1))
    # Pressure calculations (using PV = nRT, assume n = 1)
    R = 8.314  # J/(mol·K)
    n = 1.0  # mol
    P1 = n * R * T_hot / V1
    P2 = n * R * T_hot / V2
    P3 = n * R * T_cold / V3
    P4 = n * R * T_cold / V4
    # Create detailed P-V diagram
    V_iso_hot = np.linspace(V1, V2, 100)  # Isothermal expansion (hot)
    P_iso_hot = n * R * T_hot / V_iso_hot
    V_adiab1 = np.linspace(V2, V3, 100)  # Adiabatic expansion
    P_adiab1 = P2 * (V2/V_adiab1)**gamma
    V_iso_cold = np.linspace(V3, V4, 100)  # Isothermal compression (cold)
    P_iso_cold = n * R * T_cold / V_iso_cold
    V_adiab2 = np.linspace(V4, V1, 100)  # Adiabatic compression
    P_adiab2 = P4 * (V4/V_adiab2)**gamma
    # Work calculations
    W_12 = n * R * T_hot * np.log(V2/V1)  # Isothermal expansion
    W_34 = n * R * T_cold * np.log(V4/V3)  # Isothermal compression (negative)
    W_23 = n * R * (T_cold - T_hot) / (1 - gamma)  # Adiabatic expansion
    W_41 = n * R * (T_hot - T_cold) / (1 - gamma)  # Adiabatic compression
    W_net = W_12 + W_23 + W_34 + W_41
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # P-V diagram
    ax1.plot(V_iso_hot, P_iso_hot/1000, 'r-', linewidth=3, label='1→2: Isothermal (hot)')
    ax1.plot(V_adiab1, P_adiab1/1000, 'b-', linewidth=3, label='2→3: Adiabatic expansion')
    ax1.plot(V_iso_cold, P_iso_cold/1000, 'g-', linewidth=3, label='3→4: Isothermal (cold)')
    ax1.plot(V_adiab2, P_adiab2/1000, 'm-', linewidth=3, label='4→1: Adiabatic compression')
    # Mark the four states
    ax1.plot(V1, P1/1000, 'ko', markersize=8)
    ax1.plot(V2, P2/1000, 'ko', markersize=8)
    ax1.plot(V3, P3/1000, 'ko', markersize=8)
    ax1.plot(V4, P4/1000, 'ko', markersize=8)
    ax1.text(V1, P1/1000 + 0.5, '1', fontsize=12, ha='center')
    ax1.text(V2, P2/1000 + 0.5, '2', fontsize=12, ha='center')
    ax1.text(V3, P3/1000 + 0.5, '3', fontsize=12, ha='center')
    ax1.text(V4, P4/1000 + 0.5, '4', fontsize=12, ha='center')
    ax1.set_xlabel('Volume (m³)')
    ax1.set_ylabel('Pressure (kPa)')
    ax1.set_title('Carnot Cycle P-V Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # T-S diagram (simplified)
    # Entropy changes
    S_12 = W_12 / T_hot  # Isothermal process
    S_34 = -W_12 / T_cold  # Isothermal process (reverse)
    S = [0, S_12, S_12, 0, 0]  # Entropy values at states 1, 2, 3, 4, 1
    T = [T_hot, T_hot, T_cold, T_cold, T_hot]  # Temperature values
    ax2.plot(S, T, 'k-', linewidth=3, marker='o', markersize=8)
    ax2.set_xlabel('Entropy (J/K)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Carnot Cycle T-S Diagram')
    ax2.grid(True, alpha=0.3)
    # Fill the cycle area (represents work done)
    ax2.fill(S[:-1], T[:-1], alpha=0.3, color='yellow', label='Work done')
    ax2.legend()
    # Efficiency vs temperature ratio
    T_cold_range = np.linspace(200, T_hot-50, 100)
    eta_range = 1 - T_cold_range/T_hot
    ax3.plot(T_cold_range, eta_range*100, 'b-', linewidth=2)
    ax3.axvline(x=T_cold, color='r', linestyle='--', alpha=0.7, label=f'T_cold = {T_cold} K')
    ax3.axhline(y=eta_carnot*100, color='r', linestyle='--', alpha=0.7, label=f'η = {eta_carnot*100:.1f}%')
    ax3.plot(T_cold, eta_carnot*100, 'ro', markersize=8)
    ax3.set_xlabel('Cold Reservoir Temperature (K)')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Carnot Efficiency vs Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Work and heat flows
    processes = ['1→2\n(Iso. Hot)', '2→3\n(Adiabatic)', '3→4\n(Iso. Cold)', '4→1\n(Adiabatic)']
    works = [W_12, W_23, W_34, W_41]
    colors = ['red', 'blue', 'green', 'magenta']
    bars = ax4.bar(processes, np.array(works)/1000, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax4.set_ylabel('Work (kJ)')
    ax4.set_title('Work in Each Process')
    ax4.grid(True, alpha=0.3, axis='y')
    # Add net work annotation
    ax4.text(1.5, max(works)/2000, f'Net Work:\n{W_net/1000:.2f} kJ',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "heat_engine.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Net work per cycle: W_net = {W_net/1000:.2f} kJ")
    print(f"      Heat input: Q_h = {W_12/1000:.2f} kJ")
    print(f"      Actual efficiency: η = W_net/Q_h = {W_net/W_12:.3f}")
if __name__ == "__main__":
    main()