"""
Beginner Example: PID Temperature Control
A simple temperature control system using PID controller.
This example demonstrates basic PID tuning and performance analysis.
Learning Objectives:
- Understand PID controller components
- Learn basic tuning methods
- Analyze step response and stability
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# Add Control module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.pid_controller import PIDController, PIDConfig, simulate_pid_system
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
BERKELEY_GOLD = '#FDB515'
def temperature_plant(u: float, y_prev: float, dt: float) -> float:
    """
    Simple first-order temperature plant model.
    Represents: τ dy/dt + y = K*u
    where τ = time constant, K = gain
    Parameters:
        u: Control input (heater power 0-100%)
        y_prev: Previous temperature
        dt: Time step
    Returns:
        New temperature
    """
    # Plant parameters
    tau = 60.0  # Time constant (seconds)
    K = 2.0     # Steady-state gain (°C per % power)
    # Discretize: y[k+1] = a*y[k] + b*u[k]
    a = np.exp(-dt / tau)
    b = K * (1 - a)
    return a * y_prev + b * u
def main():
    """Main temperature control simulation."""
    print("PID Temperature Control Example")
    print("=" * 40)
    # Simulation parameters
    dt = 1.0        # Sample time (seconds)
    duration = 600  # Simulation time (seconds)
    setpoint = 75.0 # Target temperature (°C)
    # Initial PID tuning (conservative)
    config = PIDConfig(
        kp=2.0,     # Proportional gain
        ki=0.05,    # Integral gain
        kd=20.0,    # Derivative gain
        dt=dt,
        output_min=0.0,   # Minimum heater power (%)
        output_max=100.0, # Maximum heater power (%)
        derivative_filter_tau=10.0  # Derivative filter
    )
    # Create controller
    controller = PIDController(config)
    print(f"Initial PID gains: Kp={config.kp}, Ki={config.ki}, Kd={config.kd}")
    print(f"Target temperature: {setpoint}°C")
    print(f"Simulation time: {duration} seconds")
    # Run simulation
    results = simulate_pid_system(
        controller=controller,
        plant_transfer_function=temperature_plant,
        setpoint=setpoint,
        duration=duration,
        dt=dt,
        noise_std=0.5  # Measurement noise
    )
    # Performance analysis
    final_error = abs(results['setpoint'][-1] - results['output'][-1])
    settling_time = calculate_settling_time(results['time'], results['output'], setpoint)
    overshoot = calculate_overshoot(results['output'], setpoint)
    print(f"\nPerformance Analysis:")
    print(f"Final error: {final_error:.2f}°C")
    print(f"Settling time (2%): {settling_time:.1f} seconds")
    print(f"Overshoot: {overshoot:.1f}%")
    # Plot results
    create_plots(results, setpoint)
    # Try different tuning
    print("\nTrying Ziegler-Nichols tuning...")
    ku = 8.0   # Ultimate gain (estimated)
    tu = 120.0 # Ultimate period (estimated)
    zn_config = controller.tune_ziegler_nichols(ku, tu, method='classic')
    print(f"Z-N gains: Kp={zn_config.kp:.2f}, Ki={zn_config.ki:.4f}, Kd={zn_config.kd:.2f}")
    # Test new tuning
    controller_zn = PIDController(zn_config)
    results_zn = simulate_pid_system(
        controller=controller_zn,
        plant_transfer_function=temperature_plant,
        setpoint=setpoint,
        duration=duration,
        dt=dt,
        noise_std=0.5
    )
    # Compare performance
    compare_controllers(results, results_zn, setpoint)
def calculate_settling_time(time: np.ndarray,
                          output: np.ndarray,
                          setpoint: float,
                          tolerance: float = 0.02) -> float:
    """Calculate 2% settling time."""
    error_band = setpoint * tolerance
    for i in reversed(range(len(output))):
        if abs(output[i] - setpoint) > error_band:
            if i < len(time) - 1:
                return time[i + 1]
            else:
                return time[-1]
    return 0.0
def calculate_overshoot(output: np.ndarray, setpoint: float) -> float:
    """Calculate percentage overshoot."""
    max_value = np.max(output)
    if max_value > setpoint:
        return 100 * (max_value - setpoint) / setpoint
    return 0.0
def create_plots(results: dict, setpoint: float):
    """Create comprehensive plots of PID performance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PID Temperature Control Performance', fontsize=16, fontweight='bold')
    # Temperature response
    axes[0, 0].plot(results['time'], results['output'],
                   color=BERKELEY_BLUE, linewidth=2, label='Temperature')
    axes[0, 0].plot(results['time'], results['setpoint'],
                   color=BERKELEY_GOLD, linewidth=2, linestyle='--', label='Setpoint')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].set_title('Temperature Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Control signal
    axes[0, 1].plot(results['time'], results['control'],
                   color=BERKELEY_BLUE, linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Heater Power (%)')
    axes[0, 1].set_title('Control Signal')
    axes[0, 1].grid(True, alpha=0.3)
    # Error signal
    error = results['setpoint'] - results['output']
    axes[1, 0].plot(results['time'], error,
                   color=BERKELEY_BLUE, linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (°C)')
    axes[1, 0].set_title('Tracking Error')
    axes[1, 0].grid(True, alpha=0.3)
    # Control effort histogram
    axes[1, 1].hist(results['control'], bins=30, color=BERKELEY_GOLD, alpha=0.7)
    axes[1, 1].set_xlabel('Heater Power (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Control Effort Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def compare_controllers(results1: dict, results2: dict, setpoint: float):
    """Compare two controller performances."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Temperature comparison
    axes[0].plot(results1['time'], results1['output'],
                color=BERKELEY_BLUE, linewidth=2, label='Conservative Tuning')
    axes[0].plot(results2['time'], results2['output'],
                color=BERKELEY_GOLD, linewidth=2, label='Ziegler-Nichols')
    axes[0].axhline(y=setpoint, color='k', linestyle='--', alpha=0.5, label='Setpoint')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Controller Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Control effort comparison
    axes[1].plot(results1['time'], results1['control'],
                color=BERKELEY_BLUE, linewidth=2, label='Conservative')
    axes[1].plot(results2['time'], results2['control'],
                color=BERKELEY_GOLD, linewidth=2, label='Ziegler-Nichols')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Heater Power (%)')
    axes[1].set_title('Control Effort Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Configure matplotlib for Berkeley style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 12
    main()