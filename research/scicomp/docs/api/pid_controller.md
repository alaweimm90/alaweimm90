# pid_controller
**Module:** `Python/Control/core/pid_controller.py`
## Overview
PID Controller Implementation
A professional PID (Proportional-Integral-Derivative) controller implementation
with advanced features including windup protection, derivative filtering,
and adaptive tuning capabilities.
## Functions
### `simulate_pid_system(controller, plant_transfer_function, setpoint, duration, dt, noise_std)`
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
**Source:** [Line 208](Python/Control/core/pid_controller.py#L208)
## Classes
### `PIDConfig`
Configuration parameters for PID controller.
**Class Source:** [Line 16](Python/Control/core/pid_controller.py#L16)
### `PIDController`
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
#### Methods
##### `__init__(self, config)`
Initialize PID controller.
Parameters:
config: PID configuration parameters
**Source:** [Line 50](Python/Control/core/pid_controller.py#L50)
##### `reset(self)`
Reset controller internal state.
**Source:** [Line 60](Python/Control/core/pid_controller.py#L60)
##### `update(self, setpoint, measurement, dt)`
Compute PID control output.
Parameters:
setpoint: Desired value
measurement: Current measured value
dt: Time step (optional, uses config.dt if None)
Returns:
Control output
**Source:** [Line 68](Python/Control/core/pid_controller.py#L68)
##### `get_components(self, setpoint, measurement, dt)`
Get individual PID components (for analysis/tuning).
Parameters:
setpoint: Desired value
measurement: Current measured value
dt: Time step (optional)
Returns:
Tuple of (proportional, integral, derivative) components
**Source:** [Line 131](Python/Control/core/pid_controller.py#L131)
##### `tune_ziegler_nichols(self, ku, tu, method)`
Auto-tune PID parameters using Ziegler-Nichols method.
Parameters:
ku: Ultimate gain (gain at which system oscillates)
tu: Ultimate period (period of oscillation)
method: Tuning method ('classic', 'pessen', 'some_overshoot', 'no_overshoot')
Returns:
New PIDConfig with tuned parameters
**Source:** [Line 165](Python/Control/core/pid_controller.py#L165)
**Class Source:** [Line 33](Python/Control/core/pid_controller.py#L33)