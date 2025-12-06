# solvers
**Module:** `Python/Multiphysics/solvers.py`
## Overview
Multiphysics Coupled System Solvers.
This module provides various solution strategies for coupled
multiphysics systems including monolithic, partitioned,
and advanced iterative methods.
Classes:
MultiphysicsSolver: Base solver class
MonolithicSolver: Monolithic coupling solver
PartitionedSolver: Partitioned coupling solver
StaggeredSolver: Staggered solution approach
NewtonRaphson: Newton-Raphson for nonlinear coupling
FixedPointIteration: Fixed-point iteration solver
Functions:
solve_coupled_system: General coupled system solver
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `solve_coupled_system(coupled_system, solver_type, solver_parameters, initial_guess)`
Solve coupled multiphysics system.
Args:
coupled_system: Coupled system to solve
solver_type: Type of solver to use
solver_parameters: Solver parameters
initial_guess: Initial solution guess
**kwargs: Additional solver arguments
Returns:
Solution data
**Source:** [Line 881](Python/Multiphysics/solvers.py#L881)
## Classes
### `SolverParameters`
Parameters for coupled system solvers.
**Class Source:** [Line 34](Python/Multiphysics/solvers.py#L34)
### `SolutionData`
Solution data for coupled systems.
**Class Source:** [Line 45](Python/Multiphysics/solvers.py#L45)
### `MultiphysicsSolver`
Abstract base class for multiphysics solvers.
#### Methods
##### `__init__(self, coupled_system, parameters)`
Initialize multiphysics solver.
Args:
coupled_system: Coupled system to solve
parameters: Solver parameters
**Source:** [Line 59](Python/Multiphysics/solvers.py#L59)
##### `solve(self, initial_guess)`
Solve coupled system.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Solution data
**Source:** [Line 76](Python/Multiphysics/solvers.py#L76)
##### `compute_coupling_residual(self, solution)`
Compute coupling residual.
Args:
solution: Current solution state
Returns:
Coupling residual norm
**Source:** [Line 90](Python/Multiphysics/solvers.py#L90)
##### `apply_relaxation(self, new_solution, old_solution)`
Apply relaxation to solution update.
Args:
new_solution: New solution iterate
old_solution: Previous solution
Returns:
Relaxed solution
**Source:** [Line 119](Python/Multiphysics/solvers.py#L119)
**Class Source:** [Line 56](Python/Multiphysics/solvers.py#L56)
### `MonolithicSolver`
Monolithic coupling solver.
Solves all physics simultaneously in a single
large coupled system.
#### Methods
##### `solve(self, initial_guess)`
Solve monolithic coupled system.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Monolithic solution data
**Source:** [Line 151](Python/Multiphysics/solvers.py#L151)
##### `_assemble_global_system(self)`
Assemble global coupled system matrix.
**Source:** [Line 202](Python/Multiphysics/solvers.py#L202)
##### `_add_coupling_terms(self, global_matrix, domain_sizes)`
Add coupling terms to global matrix.
**Source:** [Line 245](Python/Multiphysics/solvers.py#L245)
##### `_apply_global_bc(self, matrix, rhs)`
Apply global boundary conditions.
**Source:** [Line 277](Python/Multiphysics/solvers.py#L277)
##### `_extract_domain_solutions(self, global_solution)`
Extract individual domain solutions from global solution.
**Source:** [Line 283](Python/Multiphysics/solvers.py#L283)
**Class Source:** [Line 144](Python/Multiphysics/solvers.py#L144)
### `PartitionedSolver`
Partitioned coupling solver.
Solves each physics separately with iterative
coupling through interface data exchange.
#### Methods
##### `solve(self, initial_guess)`
Solve partitioned coupled system.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Partitioned solution data
**Source:** [Line 307](Python/Multiphysics/solvers.py#L307)
##### `_initialize_solution(self)`
Initialize solution guess.
**Source:** [Line 384](Python/Multiphysics/solvers.py#L384)
##### `_get_coupling_data(self, target_domain)`
Get coupling data for target domain.
**Source:** [Line 394](Python/Multiphysics/solvers.py#L394)
##### `_solve_domain_fallback(self, domain_name, solver)`
Fallback domain solve method.
**Source:** [Line 414](Python/Multiphysics/solvers.py#L414)
**Class Source:** [Line 300](Python/Multiphysics/solvers.py#L300)
### `StaggeredSolver`
Staggered solution approach.
Solves physics in a predetermined sequence
with one-way coupling per time step.
#### Methods
##### `__init__(self, coupled_system, parameters, solve_order)`
Initialize staggered solver.
Args:
coupled_system: Coupled system
parameters: Solver parameters
solve_order: Order to solve physics domains
**Source:** [Line 430](Python/Multiphysics/solvers.py#L430)
##### `solve(self, initial_guess)`
Solve with staggered approach.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Staggered solution data
**Source:** [Line 444](Python/Multiphysics/solvers.py#L444)
##### `_initialize_solution(self)`
Initialize solution for staggered solve.
**Source:** [Line 497](Python/Multiphysics/solvers.py#L497)
##### `_get_coupling_data(self, target_domain)`
Get coupling data for staggered solve.
**Source:** [Line 504](Python/Multiphysics/solvers.py#L504)
##### `_solve_domain_fallback(self, domain_name, solver)`
Fallback solve for staggered approach.
**Source:** [Line 530](Python/Multiphysics/solvers.py#L530)
**Class Source:** [Line 423](Python/Multiphysics/solvers.py#L423)
### `NewtonRaphson`
Newton-Raphson solver for nonlinear coupling.
Uses Newton's method to solve coupled nonlinear systems.
#### Methods
##### `solve(self, initial_guess)`
Solve using Newton-Raphson method.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Newton-Raphson solution data
**Source:** [Line 544](Python/Multiphysics/solvers.py#L544)
##### `_compute_residual_vector(self)`
Compute residual vector for Newton method.
**Source:** [Line 616](Python/Multiphysics/solvers.py#L616)
##### `_compute_jacobian(self)`
Compute Jacobian matrix.
**Source:** [Line 629](Python/Multiphysics/solvers.py#L629)
##### `_line_search(self, direction)`
Perform line search to find optimal step size.
**Source:** [Line 639](Python/Multiphysics/solvers.py#L639)
##### `_update_solution(self, delta_x)`
Update solution with increment.
**Source:** [Line 663](Python/Multiphysics/solvers.py#L663)
##### `_initialize_solution(self)`
Initialize solution for Newton method.
**Source:** [Line 675](Python/Multiphysics/solvers.py#L675)
**Class Source:** [Line 538](Python/Multiphysics/solvers.py#L538)
### `FixedPointIteration`
Fixed-point iteration solver.
Uses fixed-point iteration with optional acceleration
for coupled systems.
#### Methods
##### `__init__(self, coupled_system, parameters, acceleration)`
Initialize fixed-point solver.
Args:
coupled_system: Coupled system
parameters: Solver parameters
acceleration: Acceleration method (none, aitken, anderson)
**Source:** [Line 690](Python/Multiphysics/solvers.py#L690)
##### `solve(self, initial_guess)`
Solve using fixed-point iteration.
Args:
initial_guess: Initial solution guess
**kwargs: Additional arguments
Returns:
Fixed-point solution data
**Source:** [Line 705](Python/Multiphysics/solvers.py#L705)
##### `_fixed_point_step(self)`
Perform one fixed-point iteration step.
**Source:** [Line 771](Python/Multiphysics/solvers.py#L771)
##### `_apply_acceleration(self, new_solution)`
Apply acceleration to fixed-point iteration.
**Source:** [Line 788](Python/Multiphysics/solvers.py#L788)
##### `_aitken_acceleration(self, new_solution)`
Apply Aitken's Δ² acceleration.
**Source:** [Line 797](Python/Multiphysics/solvers.py#L797)
##### `_anderson_acceleration(self, new_solution)`
Apply Anderson acceleration (simplified).
**Source:** [Line 827](Python/Multiphysics/solvers.py#L827)
##### `_compute_fixed_point_residual(self, x_new, x_old)`
Compute fixed-point residual ||x_new - x_old||.
**Source:** [Line 833](Python/Multiphysics/solvers.py#L833)
##### `_get_coupling_data(self, target_domain)`
Get coupling data for fixed-point iteration.
**Source:** [Line 845](Python/Multiphysics/solvers.py#L845)
##### `_solve_domain_fallback(self, domain_name, solver)`
Fallback domain solve.
**Source:** [Line 865](Python/Multiphysics/solvers.py#L865)
##### `_initialize_solution(self)`
Initialize solution for fixed-point iteration.
**Source:** [Line 872](Python/Multiphysics/solvers.py#L872)
**Class Source:** [Line 683](Python/Multiphysics/solvers.py#L683)