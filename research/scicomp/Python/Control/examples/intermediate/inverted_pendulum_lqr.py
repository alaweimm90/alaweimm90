"""
Intermediate Example: Inverted Pendulum LQR Control
Linear Quadratic Regulator (LQR) control of an inverted pendulum on a cart.
This example demonstrates state-space control design and optimal control theory.
Learning Objectives:
- Understand state-space representation
- Learn LQR design methodology
- Analyze controllability and observability
- Compare LQR with pole placement
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# Add Control module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.state_space import StateSpaceSystem, LinearQuadraticRegulator
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
BERKELEY_GOLD = '#FDB515'
class InvertedPendulumSystem:
    """Inverted pendulum on cart system model."""
    def __init__(self):
        """Initialize system parameters."""
        # Physical parameters
        self.M = 2.0    # Cart mass (kg)
        self.m = 0.5    # Pendulum mass (kg)
        self.l = 1.0    # Pendulum length (m)
        self.g = 9.81   # Gravity (m/s^2)
        self.b = 0.1    # Cart friction coefficient
        # Linearization point: pendulum upright (θ = 0)
        self._compute_state_space_matrices()
    def _compute_state_space_matrices(self):
        """Compute linearized state-space matrices."""
        # State vector: [x, x_dot, theta, theta_dot]
        # Control input: force on cart (u)
        # Denominators for linearization
        den1 = (self.M + self.m) * self.l - self.m * self.l
        den2 = self.l * den1
        # System matrix A (4x4)
        self.A = np.array([
            [0, 1, 0, 0],
            [0, -self.b/self.M, -self.m*self.g/self.M, 0],
            [0, 0, 0, 1],
            [0, self.b/(self.M*self.l), (self.M+self.m)*self.g/(self.M*self.l), 0]
        ])
        # Input matrix B (4x1)
        self.B = np.array([
            [0],
            [1/self.M],
            [0],
            [1/(self.M*self.l)]
        ])
        # Output matrix C (2x4) - measure cart position and pendulum angle
        self.C = np.array([
            [1, 0, 0, 0],  # Cart position
            [0, 0, 1, 0]   # Pendulum angle
        ])
        # Feedthrough matrix D (2x1)
        self.D = np.array([
            [0],
            [0]
        ])
    def get_state_space_system(self) -> StateSpaceSystem:
        """Return StateSpaceSystem object."""
        return StateSpaceSystem(self.A, self.B, self.C, self.D)
    def nonlinear_dynamics(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Nonlinear dynamics for simulation.
        Parameters:
            state: [x, x_dot, theta, theta_dot]
            u: Control force
        Returns:
            State derivatives
        """
        x, x_dot, theta, theta_dot = state
        # Trigonometric terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        # Denominators
        den = self.M + self.m * sin_theta**2
        # Cart acceleration
        num_x = u - self.m * self.l * theta_dot**2 * sin_theta - self.b * x_dot + self.m * self.g * sin_theta * cos_theta
        x_ddot = num_x / den
        # Pendulum angular acceleration
        num_theta = -u * cos_theta + self.m * self.l * theta_dot**2 * sin_theta * cos_theta + (self.M + self.m) * self.g * sin_theta + self.b * x_dot * cos_theta
        theta_ddot = num_theta / (self.l * den)
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
def main():
    """Main LQR control design and simulation."""
    print("Inverted Pendulum LQR Control")
    print("=" * 40)
    # Create system
    pendulum = InvertedPendulumSystem()
    sys = pendulum.get_state_space_system()
    # System analysis
    print("System Analysis:")
    print(f"Number of states: {sys.n_states}")
    print(f"Number of inputs: {sys.n_inputs}")
    print(f"Number of outputs: {sys.n_outputs}")
    # Check poles (stability)
    poles = sys.poles()
    print(f"\nOpen-loop poles: {poles}")
    print(f"System is {'stable' if sys.is_stable() else 'unstable'}")
    # Controllability analysis
    print(f"\nControllability: {'Yes' if sys.is_controllable() else 'No'}")
    if sys.is_controllable():
        Wc = sys.controllability_matrix()
        print(f"Controllability matrix rank: {np.linalg.matrix_rank(Wc)}/{sys.n_states}")
    # Observability analysis
    print(f"Observability: {'Yes' if sys.is_observable() else 'No'}")
    if sys.is_observable():
        Wo = sys.observability_matrix()
        print(f"Observability matrix rank: {np.linalg.matrix_rank(Wo)}/{sys.n_states}")
    # LQR Design
    print("\nLQR Controller Design:")
    # Design weights
    Q = np.diag([10, 1, 100, 1])  # State penalties [x, x_dot, theta, theta_dot]
    R = np.array([[1]])            # Control penalty
    print(f"State weights Q: diag({np.diag(Q)})")
    print(f"Control weight R: {R[0,0]}")
    # Design LQR controller
    lqr = LinearQuadraticRegulator(sys, Q, R)
    K = lqr.gain_matrix()
    print(f"LQR gain matrix K: {K.flatten()}")
    # Closed-loop analysis
    sys_cl = lqr.closed_loop_system()
    cl_poles = sys_cl.poles()
    print(f"Closed-loop poles: {cl_poles}")
    print(f"Closed-loop system is {'stable' if sys_cl.is_stable() else 'unstable'}")
    # Simulation
    print("\nRunning simulation...")
    simulate_pendulum_control(pendulum, K, lqr)
    # Weight sensitivity analysis
    print("\nWeight Sensitivity Analysis:")
    analyze_weight_sensitivity(sys, pendulum)
def simulate_pendulum_control(pendulum: InvertedPendulumSystem,
                            K: np.ndarray,
                            lqr: LinearQuadraticRegulator):
    """Simulate closed-loop pendulum control."""
    # Simulation parameters
    dt = 0.01
    t_final = 10.0
    t = np.arange(0, t_final, dt)
    # Initial conditions: pendulum slightly off vertical
    x0 = np.array([0.0, 0.0, 0.2, 0.0])  # 0.2 rad ≈ 11.5 degrees
    # Storage arrays
    states = np.zeros((len(t), 4))
    controls = np.zeros(len(t))
    states[0] = x0
    # Simulation loop
    for i in range(1, len(t)):
        # Current state
        x = states[i-1]
        # LQR control law: u = -K*x
        u = -K @ x
        # Apply control limits
        u_max = 50.0  # Maximum force (N)
        u = np.clip(u, -u_max, u_max)[0]
        controls[i-1] = u
        # Integrate nonlinear dynamics
        x_dot = pendulum.nonlinear_dynamics(x, u)
        states[i] = x + x_dot * dt
    controls[-1] = controls[-2]  # Last control value
    # Performance metrics
    settling_time = calculate_settling_time(t, states[:, 2], tolerance=0.05)  # 5% of pi
    max_force = np.max(np.abs(controls))
    final_error = np.linalg.norm(states[-1])
    print(f"Performance Metrics:")
    print(f"Settling time (5%): {settling_time:.2f} seconds")
    print(f"Maximum control force: {max_force:.2f} N")
    print(f"Final state error: {final_error:.4f}")
    # Plot results
    plot_simulation_results(t, states, controls)
def calculate_settling_time(time: np.ndarray,
                          angle: np.ndarray,
                          tolerance: float = 0.05) -> float:
    """Calculate settling time for pendulum angle."""
    for i in reversed(range(len(angle))):
        if abs(angle[i]) > tolerance:
            if i < len(time) - 1:
                return time[i + 1]
            else:
                return time[-1]
    return 0.0
def plot_simulation_results(t: np.ndarray, states: np.ndarray, controls: np.ndarray):
    """Plot simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Inverted Pendulum LQR Control Results', fontsize=16, fontweight='bold')
    # Cart position
    axes[0, 0].plot(t, states[:, 0], color=BERKELEY_BLUE, linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cart Position (m)')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].grid(True, alpha=0.3)
    # Pendulum angle
    axes[0, 1].plot(t, states[:, 2] * 180/np.pi, color=BERKELEY_GOLD, linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pendulum Angle (deg)')
    axes[0, 1].set_title('Pendulum Angle')
    axes[0, 1].grid(True, alpha=0.3)
    # Control force
    axes[1, 0].plot(t, controls, color=BERKELEY_BLUE, linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Force (N)')
    axes[1, 0].set_title('Control Signal')
    axes[1, 0].grid(True, alpha=0.3)
    # Phase portrait (angle vs angular velocity)
    axes[1, 1].plot(states[:, 2] * 180/np.pi, states[:, 3] * 180/np.pi,
                   color=BERKELEY_GOLD, linewidth=2)
    axes[1, 1].plot(states[0, 2] * 180/np.pi, states[0, 3] * 180/np.pi,
                   'ro', markersize=8, label='Start')
    axes[1, 1].plot(0, 0, 'kx', markersize=10, label='Target')
    axes[1, 1].set_xlabel('Angle (deg)')
    axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[1, 1].set_title('Phase Portrait')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def analyze_weight_sensitivity(sys: StateSpaceSystem, pendulum: InvertedPendulumSystem):
    """Analyze sensitivity to LQR weights."""
    # Test different Q weights for pendulum angle
    theta_weights = [1, 10, 100, 1000]
    settling_times = []
    max_forces = []
    R = np.array([[1]])
    for q_theta in theta_weights:
        Q = np.diag([10, 1, q_theta, 1])
        lqr = LinearQuadraticRegulator(sys, Q, R)
        K = lqr.gain_matrix()
        # Quick simulation
        dt = 0.01
        t_sim = np.arange(0, 5.0, dt)
        x0 = np.array([0.0, 0.0, 0.2, 0.0])
        states = np.zeros((len(t_sim), 4))
        controls = np.zeros(len(t_sim))
        states[0] = x0
        for i in range(1, len(t_sim)):
            x = states[i-1]
            u = np.clip(-K @ x, -50, 50)[0]
            controls[i-1] = u
            x_dot = pendulum.nonlinear_dynamics(x, u)
            states[i] = x + x_dot * dt
        settling_time = calculate_settling_time(t_sim, states[:, 2], tolerance=0.05)
        max_force = np.max(np.abs(controls))
        settling_times.append(settling_time)
        max_forces.append(max_force)
        print(f"Q_theta = {q_theta:4d}: Settling time = {settling_time:.2f}s, Max force = {max_force:.1f}N")
    # Plot sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogx(theta_weights, settling_times, 'o-', color=BERKELEY_BLUE, linewidth=2, markersize=8)
    ax1.set_xlabel('Pendulum Angle Weight (Q_theta)')
    ax1.set_ylabel('Settling Time (s)')
    ax1.set_title('Settling Time vs Weight')
    ax1.grid(True, alpha=0.3)
    ax2.semilogx(theta_weights, max_forces, 'o-', color=BERKELEY_GOLD, linewidth=2, markersize=8)
    ax2.set_xlabel('Pendulum Angle Weight (Q_theta)')
    ax2.set_ylabel('Maximum Force (N)')
    ax2.set_title('Control Effort vs Weight')
    ax2.grid(True, alpha=0.3)
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