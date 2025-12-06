# physics_informed
**Module:** `Python/Machine_Learning/physics_informed.py`
## Overview
Physics-Informed Machine Learning for Scientific Computing
This module implements physics-informed machine learning methods that incorporate
physical laws, conservation principles, and domain knowledge into ML models.
Classes:
PINN: Physics-Informed Neural Networks
DeepONet: Deep Operator Networks
FNO: Fourier Neural Operator
PhysicsConstrainedNN: Neural networks with physics constraints
ConservationLawsNN: Neural networks enforcing conservation laws
SymmetryAwareNN: Neural networks respecting physical symmetries
## Functions
### `create_pde_test_data(equation_type)`
Create test data for PDE problems.
**Source:** [Line 593](Python/Machine_Learning/physics_informed.py#L593)
### `plot_pinn_training(results, title)`
Plot PINN training results with Berkeley styling.
**Source:** [Line 628](Python/Machine_Learning/physics_informed.py#L628)
### `plot_pde_solution(x_grid, t_grid, u_pred, u_true, title)`
Plot PDE solution with Berkeley styling.
**Source:** [Line 671](Python/Machine_Learning/physics_informed.py#L671)
## Classes
### `PINNResults`
Container for PINN training results.
**Class Source:** [Line 30](Python/Machine_Learning/physics_informed.py#L30)
### `PhysicsInformedModel`
Abstract base class for physics-informed models.
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 43](Python/Machine_Learning/physics_informed.py#L43)
##### `pde_residual(self, x, t, u)`
Compute PDE residual.
**Source:** [Line 47](Python/Machine_Learning/physics_informed.py#L47)
##### `boundary_conditions(self, x, t)`
Define boundary conditions.
**Source:** [Line 52](Python/Machine_Learning/physics_informed.py#L52)
##### `initial_conditions(self, x)`
Define initial conditions.
**Source:** [Line 57](Python/Machine_Learning/physics_informed.py#L57)
**Class Source:** [Line 40](Python/Machine_Learning/physics_informed.py#L40)
### `PINN`
Physics-Informed Neural Networks for solving PDEs.
Features:
- Automatic differentiation for PDE residuals
- Multiple loss weighting strategies
- Adaptive sampling
- Conservation law enforcement
#### Methods
##### `__init__(self, layers, activation, pde_weight, bc_weight, ic_weight, data_weight, learning_rate, adaptive_weights)`
*No documentation available.*
**Source:** [Line 73](Python/Machine_Learning/physics_informed.py#L73)
##### `_initialize_network(self)`
Initialize the neural network.
**Source:** [Line 99](Python/Machine_Learning/physics_informed.py#L99)
##### `forward(self, x, t)`
Forward pass through the network.
**Source:** [Line 110](Python/Machine_Learning/physics_informed.py#L110)
##### `compute_derivatives(self, x, t)`
Compute derivatives using finite differences.
**Source:** [Line 116](Python/Machine_Learning/physics_informed.py#L116)
##### `heat_equation_residual(self, x, t, diffusivity)`
Heat equation PDE residual: u_t - α * u_xx = 0.
**Source:** [Line 141](Python/Machine_Learning/physics_informed.py#L141)
##### `wave_equation_residual(self, x, t, wave_speed)`
Wave equation PDE residual: u_tt - c² * u_xx = 0.
**Source:** [Line 147](Python/Machine_Learning/physics_informed.py#L147)
##### `burgers_equation_residual(self, x, t, viscosity)`
Burgers equation PDE residual: u_t + u * u_x - ν * u_xx = 0.
**Source:** [Line 160](Python/Machine_Learning/physics_informed.py#L160)
##### `pde_residual(self, x, t, equation_type)`
Compute PDE residual based on equation type.
**Source:** [Line 167](Python/Machine_Learning/physics_informed.py#L167)
##### `boundary_conditions(self, x, t)`
Define boundary conditions (to be overridden).
**Source:** [Line 179](Python/Machine_Learning/physics_informed.py#L179)
##### `initial_conditions(self, x)`
Define initial conditions (to be overridden).
**Source:** [Line 187](Python/Machine_Learning/physics_informed.py#L187)
##### `compute_losses(self, x_pde, t_pde, x_bc, t_bc, x_ic, equation_type, x_data, t_data, u_data)`
Compute all loss components.
**Source:** [Line 192](Python/Machine_Learning/physics_informed.py#L192)
##### `total_loss(self, losses)`
Compute weighted total loss.
**Source:** [Line 229](Python/Machine_Learning/physics_informed.py#L229)
##### `train(self, x_domain, t_domain, n_pde, n_bc, n_ic, epochs, equation_type, x_data, t_data, u_data, verbose)`
Train the PINN.
Parameters:
x_domain: Spatial domain (x_min, x_max)
t_domain: Temporal domain (t_min, t_max)
n_pde: Number of PDE collocation points
n_bc: Number of boundary condition points
n_ic: Number of initial condition points
epochs: Number of training epochs
equation_type: Type of PDE ('heat', 'wave', 'burgers')
x_data, t_data, u_data: Optional measurement data
verbose: Whether to print training progress
**pde_kwargs: Additional parameters for PDE
Returns:
Training results
**Source:** [Line 236](Python/Machine_Learning/physics_informed.py#L236)
##### `predict(self, x, t)`
Make predictions using the trained PINN.
**Source:** [Line 332](Python/Machine_Learning/physics_informed.py#L332)
**Class Source:** [Line 62](Python/Machine_Learning/physics_informed.py#L62)
### `DeepONet`
Deep Operator Networks for learning operators between function spaces.
Features:
- Branch and trunk networks
- Multiple operator types
- Uncertainty quantification
- Physics-informed training
#### Methods
##### `__init__(self, branch_layers, trunk_layers, activation, learning_rate)`
*No documentation available.*
**Source:** [Line 351](Python/Machine_Learning/physics_informed.py#L351)
##### `_initialize_networks(self)`
Initialize branch and trunk networks.
**Source:** [Line 365](Python/Machine_Learning/physics_informed.py#L365)
##### `forward(self, u, y)`
Forward pass through DeepONet.
Parameters:
u: Input functions (n_samples, n_sensors)
y: Evaluation coordinates (n_points, n_dims)
Returns:
Output function values (n_samples, n_points)
**Source:** [Line 385](Python/Machine_Learning/physics_informed.py#L385)
##### `train(self, u_train, y_train, s_train, epochs, batch_size, verbose)`
Train the DeepONet.
Parameters:
u_train: Input functions (n_samples, n_sensors)
y_train: Evaluation coordinates (n_points, n_dims)
s_train: Target function values (n_samples, n_points)
epochs: Number of training epochs
batch_size: Batch size
verbose: Whether to print progress
**Source:** [Line 407](Python/Machine_Learning/physics_informed.py#L407)
**Class Source:** [Line 340](Python/Machine_Learning/physics_informed.py#L340)
### `ConservationLawsNN`
Neural network enforcing conservation laws.
Features:
- Built-in conservation constraints
- Mass, momentum, energy conservation
- Lagrangian mechanics integration
- Hamiltonian preservation
#### Methods
##### `__init__(self, layers, conservation_type, constraint_weight)`
*No documentation available.*
**Source:** [Line 468](Python/Machine_Learning/physics_informed.py#L468)
##### `conservation_constraint(self, u, x, t)`
Compute conservation law constraint.
**Source:** [Line 481](Python/Machine_Learning/physics_informed.py#L481)
##### `_energy_conservation_constraint(self, u, x, t)`
Energy conservation constraint.
**Source:** [Line 501](Python/Machine_Learning/physics_informed.py#L501)
##### `_compute_energy(self, x, t)`
Compute total energy.
**Source:** [Line 515](Python/Machine_Learning/physics_informed.py#L515)
**Class Source:** [Line 457](Python/Machine_Learning/physics_informed.py#L457)
### `SymmetryAwareNN`
Neural network respecting physical symmetries.
Features:
- Translation invariance
- Rotation invariance
- Scale invariance
- Time reversal symmetry
#### Methods
##### `__init__(self, layers, symmetry_type, symmetry_weight)`
*No documentation available.*
**Source:** [Line 532](Python/Machine_Learning/physics_informed.py#L532)
##### `apply_symmetry(self, x, transformation)`
Apply symmetry transformation to input.
**Source:** [Line 545](Python/Machine_Learning/physics_informed.py#L545)
##### `symmetry_loss(self, x)`
Compute symmetry violation loss.
**Source:** [Line 572](Python/Machine_Learning/physics_informed.py#L572)
**Class Source:** [Line 521](Python/Machine_Learning/physics_informed.py#L521)