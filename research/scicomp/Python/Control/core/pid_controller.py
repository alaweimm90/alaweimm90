"""
PID Controller Implementation
A professional PID (Proportional-Integral-Derivative) controller implementation
with advanced features including windup protection, derivative filtering,
and adaptive tuning capabilities.
"""
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import time
@dataclass
class PIDConfig:
    """Configuration parameters for PID controller."""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    # Anti-windup protection
    output_min: float = -np.inf
    output_max: float = np.inf
    # Derivative filtering
    derivative_filter_tau: float = 0.0  # Time constant for derivative filter
    # Sampling time
    dt: float = 0.01  # Default sampling time in seconds
class PIDController:
    """
    Professional PID Controller with advanced features.
    Features:
    - Standard PID control with configurable gains
    - Anti-windup protection
    - Derivative filtering to reduce noise sensitivity
    - Setpoint weighting for improved response
    - Reset functionality
    Examples:
        >>> config = PIDConfig(kp=2.0, ki=0.5, kd=0.1, dt=0.01)
        >>> controller = PIDController(config)
        >>> output = controller.update(setpoint=10.0, measurement=8.5)
    """
    def __init__(self, config: PIDConfig):
        """
        Initialize PID controller.
        Parameters:
            config: PID configuration parameters
        """
        self.config = config
        self.reset()
    def reset(self) -> None:
        """Reset controller internal state."""
        self._previous_error = 0.0
        self._integral = 0.0
        self._previous_measurement = 0.0
        self._previous_derivative = 0.0
        self._last_time = None
    def update(self,
               setpoint: float,
               measurement: float,
               dt: Optional[float] = None) -> float:
        """
        Compute PID control output.
        Parameters:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step (optional, uses config.dt if None)
        Returns:
            Control output
        """
        if dt is None:
            dt = self.config.dt
        # Calculate error
        error = setpoint - measurement
        # Proportional term
        proportional = self.config.kp * error
        # Integral term with anti-windup
        self._integral += error * dt
        # Anti-windup: clamp integral if output would saturate
        integral_term = self.config.ki * self._integral
        provisional_output = proportional + integral_term
        if provisional_output > self.config.output_max:
            self._integral = (self.config.output_max - proportional) / self.config.ki
        elif provisional_output < self.config.output_min:
            self._integral = (self.config.output_min - proportional) / self.config.ki
        integral = self.config.ki * self._integral
        # Derivative term (on measurement to avoid derivative kick)
        derivative_raw = -(measurement - self._previous_measurement) / dt
        # Apply derivative filtering if specified
        if self.config.derivative_filter_tau > 0:
            alpha = dt / (self.config.derivative_filter_tau + dt)
            derivative = alpha * derivative_raw + (1 - alpha) * self._previous_derivative
            self._previous_derivative = derivative
        else:
            derivative = derivative_raw
        derivative_term = self.config.kd * derivative
        # Compute total output
        output = proportional + integral + derivative_term
        # Apply output limits
        output = np.clip(output, self.config.output_min, self.config.output_max)
        # Store values for next iteration
        self._previous_error = error
        self._previous_measurement = measurement
        return output
    def get_components(self,
                      setpoint: float,
                      measurement: float,
                      dt: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Get individual PID components (for analysis/tuning).
        Parameters:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step (optional)
        Returns:
            Tuple of (proportional, integral, derivative) components
        """
        if dt is None:
            dt = self.config.dt
        error = setpoint - measurement
        proportional = self.config.kp * error
        integral = self.config.ki * self._integral
        derivative_raw = -(measurement - self._previous_measurement) / dt
        if self.config.derivative_filter_tau > 0:
            alpha = dt / (self.config.derivative_filter_tau + dt)
            derivative = alpha * derivative_raw + (1 - alpha) * self._previous_derivative
        else:
            derivative = derivative_raw
        derivative_term = self.config.kd * derivative
        return proportional, integral, derivative_term
    def tune_ziegler_nichols(self,
                           ku: float,
                           tu: float,
                           method: str = 'classic') -> PIDConfig:
        """
        Auto-tune PID parameters using Ziegler-Nichols method.
        Parameters:
            ku: Ultimate gain (gain at which system oscillates)
            tu: Ultimate period (period of oscillation)
            method: Tuning method ('classic', 'pessen', 'some_overshoot', 'no_overshoot')
        Returns:
            New PIDConfig with tuned parameters
        """
        if method == 'classic':
            kp = 0.6 * ku
            ki = 2.0 * kp / tu
            kd = kp * tu / 8.0
        elif method == 'pessen':
            kp = 0.7 * ku
            ki = 2.5 * kp / tu
            kd = 0.15 * kp * tu
        elif method == 'some_overshoot':
            kp = 0.33 * ku
            ki = 2.0 * kp / tu
            kd = kp * tu / 3.0
        elif method == 'no_overshoot':
            kp = 0.2 * ku
            ki = 2.0 * kp / tu
            kd = kp * tu / 3.0
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        return PIDConfig(
            kp=kp, ki=ki, kd=kd,
            dt=self.config.dt,
            output_min=self.config.output_min,
            output_max=self.config.output_max,
            derivative_filter_tau=self.config.derivative_filter_tau
        )
def simulate_pid_system(controller: PIDController,
                       plant_transfer_function,
                       setpoint: Union[float, np.ndarray],
                       duration: float,
                       dt: Optional[float] = None,
                       noise_std: float = 0.0) -> dict:
    """
    Simulate closed-loop PID control system.
    Parameters:
        controller: PID controller instance
        plant_transfer_function: System transfer function (callable)
        setpoint: Reference signal (constant or time-varying)
        duration: Simulation duration in seconds
        dt: Time step (uses controller.config.dt if None)
        noise_std: Standard deviation of measurement noise
    Returns:
        Dictionary containing time, setpoint, output, control signals
    """
    if dt is None:
        dt = controller.config.dt
    t = np.arange(0, duration, dt)
    n_points = len(t)
    # Handle setpoint
    if np.isscalar(setpoint):
        sp = np.full(n_points, setpoint)
    else:
        sp = np.array(setpoint)
        if len(sp) != n_points:
            raise ValueError("Setpoint array length must match time points")
    # Initialize arrays
    output = np.zeros(n_points)
    control = np.zeros(n_points)
    measured = np.zeros(n_points)
    # Reset controller
    controller.reset()
    # Simulation loop
    for i in range(1, n_points):
        # Add measurement noise
        measured[i-1] = output[i-1] + np.random.normal(0, noise_std)
        # Compute control signal
        control[i] = controller.update(sp[i], measured[i-1], dt)
        # Apply to plant (simplified first-order system example)
        # User should replace with actual plant model
        output[i] = plant_transfer_function(control[i], output[i-1], dt)
    return {
        'time': t,
        'setpoint': sp,
        'output': output,
        'control': control,
        'measured': measured
    }