"""
Advanced Example: Quadcopter Model Predictive Control
Model Predictive Control (MPC) for quadcopter trajectory tracking.
This example demonstrates advanced control concepts including:
- Nonlinear model predictive control
- Trajectory optimization
- Constraint handling
- Real-time implementation considerations
Learning Objectives:
- Understand MPC formulation and implementation
- Learn trajectory tracking control
- Handle physical constraints (thrust limits, angular rates)
- Analyze computational requirements
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import time
# Add Control module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.optimal_control import ModelPredictiveController, MPCConfig
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
BERKELEY_GOLD = '#FDB515'
class QuadcopterModel:
    """Nonlinear quadcopter dynamics model."""
    def __init__(self):
        """Initialize quadcopter parameters."""
        # Physical parameters
        self.m = 1.5        # Mass (kg)
        self.g = 9.81       # Gravity (m/s^2)
        self.Ixx = 0.029    # Moment of inertia x-axis (kg⋅m²)
        self.Iyy = 0.029    # Moment of inertia y-axis (kg⋅m²)
        self.Izz = 0.055    # Moment of inertia z-axis (kg⋅m²)
        self.l = 0.25       # Arm length (m)
        # Drag coefficients
        self.kd_linear = 0.1   # Linear drag coefficient
        self.kd_angular = 0.01 # Angular drag coefficient
        # Motor parameters
        self.thrust_max = 20.0  # Maximum total thrust (N)
        self.thrust_min = 0.0   # Minimum thrust (N)
        # Linearization for MPC
        self._compute_linearized_model()
    def _compute_linearized_model(self):
        """Compute linearized model around hover condition."""
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # Control: [thrust, torque_x, torque_y, torque_z]
        # Linearization around hover: all states zero except z position
        n_states = 12
        n_inputs = 4
        # System matrix A (12x12)
        A = np.zeros((n_states, n_inputs))
        # Position dynamics: dp/dt = v
        A[0, 3] = 1.0  # dx/dt = vx
        A[1, 4] = 1.0  # dy/dt = vy
        A[2, 5] = 1.0  # dz/dt = vz
        # Velocity dynamics (linearized)
        A[3, 7] = self.g   # dvx/dt = g*theta
        A[4, 6] = -self.g  # dvy/dt = -g*phi
        # dvz/dt = thrust/m - g (handled in B matrix)
        # Add drag terms
        A[3, 3] = -self.kd_linear / self.m  # vx drag
        A[4, 4] = -self.kd_linear / self.m  # vy drag
        A[5, 5] = -self.kd_linear / self.m  # vz drag
        # Attitude dynamics: dangle/dt = angular_velocity
        A[6, 9] = 1.0   # dphi/dt = p
        A[7, 10] = 1.0  # dtheta/dt = q
        A[8, 11] = 1.0  # dpsi/dt = r
        # Angular velocity dynamics (with drag)
        A[9, 9] = -self.kd_angular / self.Ixx    # dp/dt
        A[10, 10] = -self.kd_angular / self.Iyy  # dq/dt
        A[11, 11] = -self.kd_angular / self.Izz  # dr/dt
        self.A_linear = A
        # Input matrix B (12x4)
        B = np.zeros((n_states, n_inputs))
        # Thrust affects z velocity
        B[5, 0] = 1.0 / self.m
        # Torques affect angular accelerations
        B[9, 1] = 1.0 / self.Ixx   # Torque x -> p_dot
        B[10, 2] = 1.0 / self.Iyy  # Torque y -> q_dot
        B[11, 3] = 1.0 / self.Izz  # Torque z -> r_dot
        self.B_linear = B
    def nonlinear_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Nonlinear quadcopter dynamics.
        State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        Control: [thrust, torque_x, torque_y, torque_z]
        """
        # Extract states
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        # Extract controls
        thrust, tau_x, tau_y, tau_z = control
        # Rotation matrix (body to inertial)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta, s_phi*c_theta, c_phi*c_theta]
        ])
        # Forces in body frame
        F_body = np.array([0, 0, thrust])
        # Transform to inertial frame
        F_inertial = R @ F_body
        # Add gravity
        F_inertial[2] -= self.m * self.g
        # Add drag
        drag = -self.kd_linear * np.array([vx, vy, vz])
        F_total = F_inertial + drag
        # Linear accelerations
        ax, ay, az = F_total / self.m
        # Angular velocity dynamics
        I = np.diag([self.Ixx, self.Iyy, self.Izz])
        omega = np.array([p, q, r])
        tau = np.array([tau_x, tau_y, tau_z])
        # Gyroscopic effects
        gyro = -np.cross(omega, I @ omega)
        # Angular accelerations
        omega_dot = np.linalg.solve(I, tau + gyro - self.kd_angular * omega)
        # Attitude kinematics
        W = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        euler_dot = W @ omega
        # Combine derivatives
        state_dot = np.array([
            vx, vy, vz,           # Position derivatives
            ax, ay, az,           # Velocity derivatives
            euler_dot[0], euler_dot[1], euler_dot[2],  # Angle derivatives
            omega_dot[0], omega_dot[1], omega_dot[2]   # Angular velocity derivatives
        ])
        return state_dot
def generate_trajectory(t: np.ndarray) -> np.ndarray:
    """Generate reference trajectory - figure-8 pattern."""
    trajectory = np.zeros((len(t), 12))  # 12 states
    # Figure-8 parameters
    A = 3.0      # Amplitude
    omega = 0.2  # Frequency
    z_ref = 5.0  # Reference height
    for i, time in enumerate(t):
        # Figure-8 in x-y plane
        trajectory[i, 0] = A * np.sin(omega * time)              # x
        trajectory[i, 1] = A * np.sin(2 * omega * time) / 2      # y
        trajectory[i, 2] = z_ref                                 # z
        # Velocity references (derivatives)
        trajectory[i, 3] = A * omega * np.cos(omega * time)      # vx
        trajectory[i, 4] = A * omega * np.cos(2 * omega * time)  # vy
        trajectory[i, 5] = 0.0                                   # vz
        # Attitude references (computed from desired acceleration)
        ax_des = -A * omega**2 * np.sin(omega * time)
        ay_des = -2 * A * omega**2 * np.sin(2 * omega * time)
        # Desired pitch and roll (small angle approximation)
        trajectory[i, 6] = -ay_des / 9.81  # phi (roll)
        trajectory[i, 7] = ax_des / 9.81   # theta (pitch)
        trajectory[i, 8] = 0.0             # psi (yaw)
        # Angular velocities (set to zero for simplicity)
        trajectory[i, 9:12] = 0.0
    return trajectory
def main():
    """Main MPC quadcopter control simulation."""
    print("Quadcopter Model Predictive Control")
    print("=" * 40)
    # Create quadcopter model
    quad = QuadcopterModel()
    print("Quadcopter Parameters:")
    print(f"Mass: {quad.m} kg")
    print(f"Max thrust: {quad.thrust_max} N")
    print(f"Arm length: {quad.l} m")
    # MPC Configuration
    config = MPCConfig(
        prediction_horizon=20,
        control_horizon=10,
        u_min=np.array([0.0, -5.0, -5.0, -2.0]),     # [thrust, torques]
        u_max=np.array([20.0, 5.0, 5.0, 2.0]),
        Q=np.diag([10, 10, 100, 1, 1, 10, 50, 50, 20, 1, 1, 1]),  # State weights
        R=np.diag([1, 10, 10, 10]),                    # Control weights
        Qf=np.diag([20, 20, 200, 2, 2, 20, 100, 100, 40, 2, 2, 2])  # Terminal weights
    )
    print(f"\nMPC Configuration:")
    print(f"Prediction horizon: {config.prediction_horizon}")
    print(f"Control horizon: {config.control_horizon}")
    # Create MPC controller
    mpc = ModelPredictiveController(quad.A_linear, quad.B_linear, config)
    # Simulation parameters
    dt = 0.1
    t_final = 30.0
    t = np.arange(0, t_final, dt)
    # Generate reference trajectory
    ref_trajectory = generate_trajectory(t)
    # Initial conditions (start at origin)
    x0 = np.zeros(12)
    x0[2] = 5.0  # Start at reference height
    print(f"\nSimulation Parameters:")
    print(f"Time step: {dt} s")
    print(f"Simulation time: {t_final} s")
    print(f"Total steps: {len(t)}")
    # Run simulation
    states, controls, comp_times = simulate_mpc_control(quad, mpc, x0, ref_trajectory, dt)
    # Performance analysis
    analyze_performance(t, states, controls, ref_trajectory, comp_times)
    # Visualization
    plot_3d_trajectory(t, states, ref_trajectory)
    plot_tracking_performance(t, states, controls, ref_trajectory)
def simulate_mpc_control(quad: QuadcopterModel,
                        mpc: ModelPredictiveController,
                        x0: np.ndarray,
                        ref_trajectory: np.ndarray,
                        dt: float) -> tuple:
    """Simulate MPC control of quadcopter."""
    n_steps = len(ref_trajectory)
    n_states = 12
    n_inputs = 4
    # Storage arrays
    states = np.zeros((n_steps, n_states))
    controls = np.zeros((n_steps, n_inputs))
    comp_times = np.zeros(n_steps)
    # Initial state
    states[0] = x0
    print("Running MPC simulation...")
    for k in range(n_steps - 1):
        if k % 50 == 0:
            print(f"Step {k}/{n_steps-1} ({100*k/(n_steps-1):.1f}%)")
        # Current state and reference
        x_current = states[k]
        # Reference for prediction horizon
        ref_horizon = ref_trajectory[k:min(k+mpc.config.prediction_horizon, n_steps)]
        if len(ref_horizon) < mpc.config.prediction_horizon:
            # Extend with last reference
            ref_extended = np.zeros((mpc.config.prediction_horizon, n_states))
            ref_extended[:len(ref_horizon)] = ref_horizon
            ref_extended[len(ref_horizon):] = ref_trajectory[-1]
            ref_horizon = ref_extended
        # Solve MPC optimization
        start_time = time.time()
        u_sequence = mpc.solve(x_current, ref_horizon)
        comp_times[k] = time.time() - start_time
        # Apply first control
        u_current = u_sequence[0] if len(u_sequence) > 0 else np.zeros(n_inputs)
        controls[k] = u_current
        # Add hover thrust bias (nonlinear compensation)
        u_compensated = u_current.copy()
        u_compensated[0] += quad.m * quad.g  # Add weight compensation
        # Integrate nonlinear dynamics
        x_dot = quad.nonlinear_dynamics(states[k], u_compensated)
        states[k+1] = states[k] + x_dot * dt
    controls[-1] = controls[-2]  # Last control
    comp_times[-1] = comp_times[-2]
    print("Simulation complete!")
    return states, controls, comp_times
def analyze_performance(t: np.ndarray,
                       states: np.ndarray,
                       controls: np.ndarray,
                       ref_trajectory: np.ndarray,
                       comp_times: np.ndarray):
    """Analyze tracking and computational performance."""
    print("\nPerformance Analysis:")
    print("=" * 20)
    # Tracking errors
    position_error = np.linalg.norm(states[:, 0:3] - ref_trajectory[:, 0:3], axis=1)
    attitude_error = np.linalg.norm(states[:, 6:9] - ref_trajectory[:, 6:9], axis=1)
    print(f"Position tracking:")
    print(f"  RMS error: {np.sqrt(np.mean(position_error**2)):.3f} m")
    print(f"  Max error: {np.max(position_error):.3f} m")
    print(f"  Final error: {position_error[-1]:.3f} m")
    print(f"\nAttitude tracking:")
    print(f"  RMS error: {np.sqrt(np.mean(attitude_error**2)) * 180/np.pi:.2f} deg")
    print(f"  Max error: {np.max(attitude_error) * 180/np.pi:.2f} deg")
    # Control effort
    thrust = controls[:, 0]
    torques = controls[:, 1:4]
    print(f"\nControl effort:")
    print(f"  Avg thrust: {np.mean(thrust):.2f} N")
    print(f"  Max thrust: {np.max(thrust):.2f} N")
    print(f"  Max torque: {np.max(np.abs(torques)):.3f} N⋅m")
    # Computational performance
    print(f"\nComputational performance:")
    print(f"  Avg computation time: {np.mean(comp_times)*1000:.2f} ms")
    print(f"  Max computation time: {np.max(comp_times)*1000:.2f} ms")
    print(f"  Real-time factor: {np.mean(comp_times)/0.1:.2f} (should be < 1)")
def plot_3d_trajectory(t: np.ndarray, states: np.ndarray, ref_trajectory: np.ndarray):
    """Plot 3D trajectory visualization."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot trajectories
    ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2],
           color=BERKELEY_GOLD, linewidth=3, linestyle='--', label='Reference')
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
           color=BERKELEY_BLUE, linewidth=2, label='Actual')
    # Mark start and end points
    ax.scatter(states[0, 0], states[0, 1], states[0, 2],
              color='green', s=100, label='Start')
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2],
              color='red', s=100, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Quadcopter 3D Trajectory Tracking')
    ax.legend()
    # Equal aspect ratio
    max_range = np.array([states[:, 0].max()-states[:, 0].min(),
                         states[:, 1].max()-states[:, 1].min(),
                         states[:, 2].max()-states[:, 2].min()]).max() / 2.0
    mid_x = (states[:, 0].max()+states[:, 0].min()) * 0.5
    mid_y = (states[:, 1].max()+states[:, 1].min()) * 0.5
    mid_z = (states[:, 2].max()+states[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()
def plot_tracking_performance(t: np.ndarray,
                            states: np.ndarray,
                            controls: np.ndarray,
                            ref_trajectory: np.ndarray):
    """Plot detailed tracking performance."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Quadcopter MPC Tracking Performance', fontsize=16, fontweight='bold')
    # Position tracking
    for i, (label, unit) in enumerate([('X', 'm'), ('Y', 'm'), ('Z', 'm')]):
        axes[i, 0].plot(t, ref_trajectory[:, i], color=BERKELEY_GOLD,
                       linewidth=2, linestyle='--', label='Reference')
        axes[i, 0].plot(t, states[:, i], color=BERKELEY_BLUE,
                       linewidth=2, label='Actual')
        axes[i, 0].set_ylabel(f'{label} Position ({unit})')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        if i == 2:
            axes[i, 0].set_xlabel('Time (s)')
    # Control signals
    control_labels = ['Thrust (N)', 'Torque X (N⋅m)', 'Torque Y (N⋅m)', 'Torque Z (N⋅m)']
    axes[0, 1].plot(t, controls[:, 0], color=BERKELEY_BLUE, linewidth=2)
    axes[0, 1].set_ylabel('Thrust (N)')
    axes[0, 1].set_title('Control Signals')
    axes[0, 1].grid(True, alpha=0.3)
    # Torques
    for i in range(3):
        axes[i+1, 1].plot(t, controls[:, i+1], color=BERKELEY_BLUE, linewidth=2)
        axes[i+1, 1].set_ylabel(f'Torque {["X", "Y", "Z"][i]} (N⋅m)')
        axes[i+1, 1].grid(True, alpha=0.3)
        if i == 2:
            axes[i+1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Configure matplotlib for Berkeley style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 11
    main()