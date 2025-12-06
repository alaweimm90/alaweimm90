# physics_informed_nn
**Module:** `Python/ml_physics/physics_informed_nn.py`
## Overview
Physics-Informed Neural Networks (PINNs) for SciComp.
This module implements physics-informed machine learning for solving PDEs,
discovering governing equations, and accelerating simulations.
Author: UC Berkeley SciComp Team
Copyright © 2025 Meshal Alawein — All rights reserved.
## Functions
### `create_pinn_for_pde(pde_type, config)`
Create a PINN for a specific PDE type.
Args:
pde_type: Type of PDE ('heat', 'wave', 'schrodinger')
config: PINN configuration
**kwargs: Additional PDE-specific parameters
Returns:
Configured PINN instance
**Source:** [Line 563](Python/ml_physics/physics_informed_nn.py#L563)
### `discover_physics_from_data(x, y, derivatives)`
Discover governing physics equations from observational data.
Args:
x: Spatial/temporal coordinates
y: Observed values
derivatives: Known derivatives (optional)
Returns:
Discovered equations and coefficients
**Source:** [Line 585](Python/ml_physics/physics_informed_nn.py#L585)
## Classes
### `PINNConfig`
Configuration for Physics-Informed Neural Networks.
**Class Source:** [Line 36](Python/ml_physics/physics_informed_nn.py#L36)
### `PhysicsInformedNN`
Base class for Physics-Informed Neural Networks.
#### Methods
##### `__init__(self, config)`
Initialize PINN.
Args:
config: PINN configuration
**Source:** [Line 51](Python/ml_physics/physics_informed_nn.py#L51)
##### `_build_tensorflow_model(self)`
Build TensorFlow/Keras model.
**Source:** [Line 70](Python/ml_physics/physics_informed_nn.py#L70)
##### `_build_pytorch_model(self)`
Build PyTorch model.
**Source:** [Line 86](Python/ml_physics/physics_informed_nn.py#L86)
##### `physics_loss(self, x, y_pred)`
Compute physics-based loss (to be overridden by specific PDEs).
Args:
x: Input coordinates
y_pred: Predicted values
Returns:
Physics loss value
**Source:** [Line 112](Python/ml_physics/physics_informed_nn.py#L112)
##### `train(self, x_data, y_data, x_physics, verbose)`
Train the PINN.
Args:
x_data: Input data points
y_data: Output data values
x_physics: Collocation points for physics loss
verbose: Print training progress
**Source:** [Line 125](Python/ml_physics/physics_informed_nn.py#L125)
##### `_train_tensorflow(self, x_data, y_data, x_physics, verbose)`
TensorFlow training loop.
**Source:** [Line 144](Python/ml_physics/physics_informed_nn.py#L144)
##### `predict(self, x)`
Make predictions.
Args:
x: Input points
Returns:
Predictions
**Source:** [Line 182](Python/ml_physics/physics_informed_nn.py#L182)
**Class Source:** [Line 48](Python/ml_physics/physics_informed_nn.py#L48)
### `HeatEquationPINN`
PINN for solving the heat equation.
#### Methods
##### `__init__(self, config, thermal_diffusivity)`
Initialize heat equation PINN.
Args:
config: PINN configuration
thermal_diffusivity: Thermal diffusivity coefficient
**Source:** [Line 209](Python/ml_physics/physics_informed_nn.py#L209)
##### `physics_loss(self, x, t)`
Compute physics loss for heat equation: ∂u/∂t - α∇²u = 0
Args:
x: Spatial coordinates
t: Time coordinates
Returns:
Physics residual loss
**Source:** [Line 220](Python/ml_physics/physics_informed_nn.py#L220)
##### `_physics_loss_tf(self, x, t)`
TensorFlow physics loss computation.
**Source:** [Line 237](Python/ml_physics/physics_informed_nn.py#L237)
**Class Source:** [Line 206](Python/ml_physics/physics_informed_nn.py#L206)
### `WaveEquationPINN`
PINN for solving the wave equation.
#### Methods
##### `__init__(self, config, wave_speed)`
Initialize wave equation PINN.
Args:
config: PINN configuration
wave_speed: Wave propagation speed
**Source:** [Line 263](Python/ml_physics/physics_informed_nn.py#L263)
##### `physics_loss(self, x, t)`
Compute physics loss for wave equation: ∂²u/∂t² - c²∇²u = 0
Args:
x: Spatial coordinates
t: Time coordinates
Returns:
Physics residual loss
**Source:** [Line 274](Python/ml_physics/physics_informed_nn.py#L274)
##### `_physics_loss_tf(self, x, t)`
TensorFlow physics loss computation.
**Source:** [Line 290](Python/ml_physics/physics_informed_nn.py#L290)
**Class Source:** [Line 260](Python/ml_physics/physics_informed_nn.py#L260)
### `SchrodingerPINN`
PINN for solving the Schrödinger equation.
#### Methods
##### `__init__(self, config, potential)`
Initialize Schrödinger equation PINN.
Args:
config: PINN configuration
potential: Potential energy function V(x)
**Source:** [Line 317](Python/ml_physics/physics_informed_nn.py#L317)
##### `physics_loss(self, x, E)`
Compute physics loss for time-independent Schrödinger equation:
-ℏ²/2m ∇²ψ + V(x)ψ = Eψ
Args:
x: Spatial coordinates
E: Energy eigenvalue
Returns:
Physics residual loss
**Source:** [Line 328](Python/ml_physics/physics_informed_nn.py#L328)
##### `_physics_loss_tf(self, x, E)`
TensorFlow physics loss computation.
**Source:** [Line 345](Python/ml_physics/physics_informed_nn.py#L345)
**Class Source:** [Line 314](Python/ml_physics/physics_informed_nn.py#L314)
### `EquationDiscovery`
Discover governing equations from data using sparse regression.
#### Methods
##### `__init__(self, library_functions)`
Initialize equation discovery.
Args:
library_functions: Candidate functions for the library
**Source:** [Line 371](Python/ml_physics/physics_informed_nn.py#L371)
##### `build_library(self, x)`
Build library matrix from candidate functions.
Args:
x: Input data
Returns:
Library matrix
**Source:** [Line 395](Python/ml_physics/physics_informed_nn.py#L395)
##### `sparse_regression(self, library, derivatives, lambda_reg)`
Perform sparse regression to identify governing equations.
Args:
library: Library matrix of candidate functions
derivatives: Time derivatives or target values
lambda_reg: Regularization parameter
Returns:
Sparse coefficients
**Source:** [Line 414](Python/ml_physics/physics_informed_nn.py#L414)
##### `discover(self, x, y, derivatives)`
Discover governing equations from data.
Args:
x: Input coordinates
y: Observed values
derivatives: Time derivatives (computed if not provided)
Returns:
Discovered equation coefficients and description
**Source:** [Line 450](Python/ml_physics/physics_informed_nn.py#L450)
**Class Source:** [Line 368](Python/ml_physics/physics_informed_nn.py#L368)
### `NeuralODE`
Neural Ordinary Differential Equations for continuous-time dynamics.
#### Methods
##### `__init__(self, hidden_dim)`
Initialize Neural ODE.
Args:
hidden_dim: Hidden layer dimension
**Source:** [Line 496](Python/ml_physics/physics_informed_nn.py#L496)
##### `_build_model(self)`
Build the neural network for ODE dynamics.
**Source:** [Line 509](Python/ml_physics/physics_informed_nn.py#L509)
##### `dynamics(self, t, state)`
Compute dynamics dx/dt = f(x, t).
Args:
t: Time
state: Current state
Returns:
State derivatives
**Source:** [Line 517](Python/ml_physics/physics_informed_nn.py#L517)
##### `solve(self, initial_state, t_span, t_eval)`
Solve the Neural ODE.
Args:
initial_state: Initial condition
t_span: Time span (t0, tf)
t_eval: Times to evaluate solution
Returns:
Solution trajectory
**Source:** [Line 537](Python/ml_physics/physics_informed_nn.py#L537)
**Class Source:** [Line 493](Python/ml_physics/physics_informed_nn.py#L493)
### `PINNModel`
*No documentation available.*
#### Methods
##### `__init__(self, layers, activation)`
*No documentation available.*
**Source:** [Line 89](Python/ml_physics/physics_informed_nn.py#L89)
##### `forward(self, x)`
*No documentation available.*
**Source:** [Line 103](Python/ml_physics/physics_informed_nn.py#L103)
**Class Source:** [Line 88](Python/ml_physics/physics_informed_nn.py#L88)