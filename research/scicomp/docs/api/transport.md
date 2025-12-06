# transport
**Module:** `Python/Multiphysics/transport.py`
## Overview
Multi-Physics Transport Phenomena Module.
This module implements coupled transport phenomena including
reactive transport, porous media flow, and convection-diffusion-reaction
systems for multiphysics applications.
Classes:
MultiphysicsTransport: Main transport solver
ReactiveTransport: Reactive transport in porous media
PorousMediaFlow: Flow in porous media
ConvectionDiffusionReaction: CDR equations
Functions:
species_transport: Multi-species transport
coupled_flow_transport: Coupled flow and transport
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `species_transport(mesh, species_properties, flow_field, boundary_conditions, reactions)`
Solve multi-species transport.
Args:
mesh: Finite element mesh
species_properties: Properties for each species
flow_field: Velocity field
boundary_conditions: Transport BC
reactions: Reaction data
Returns:
Multi-species concentration fields
**Source:** [Line 601](Python/Multiphysics/transport.py#L601)
### `coupled_flow_transport(mesh, transport_props, fluid_props, boundary_conditions, coupling_type)`
Solve coupled flow and transport.
Args:
mesh: Finite element mesh
transport_props: Transport properties
fluid_props: Fluid properties
boundary_conditions: Combined BC
coupling_type: Coupling strategy
Returns:
Coupled flow-transport solution
**Source:** [Line 638](Python/Multiphysics/transport.py#L638)
## Classes
### `TransportProperties`
Transport material properties.
#### Methods
##### `effective_diffusivity(self)`
Effective diffusivity in porous media.
**Source:** [Line 41](Python/Multiphysics/transport.py#L41)
**Class Source:** [Line 32](Python/Multiphysics/transport.py#L32)
### `FluidProperties`
Fluid properties for transport.
**Class Source:** [Line 47](Python/Multiphysics/transport.py#L47)
### `ReactionData`
Chemical reaction data.
**Class Source:** [Line 54](Python/Multiphysics/transport.py#L54)
### `ConvectionDiffusionReaction`
Convection-diffusion-reaction equation solver.
Solves: ∂c/∂t + ∇·(vc) - ∇·(D∇c) = R(c)
where c is concentration, v is velocity, D is diffusivity, R is reaction.
#### Methods
##### `__init__(self, mesh, transport_props, reaction_data)`
Initialize CDR solver.
Args:
mesh: Finite element mesh
transport_props: Transport properties
reaction_data: Reaction specifications
**Source:** [Line 69](Python/Multiphysics/transport.py#L69)
##### `_setup_system(self)`
Setup finite element system.
**Source:** [Line 99](Python/Multiphysics/transport.py#L99)
##### `solve_steady_state(self, velocity_field, boundary_conditions, source_term)`
Solve steady-state CDR equation.
Args:
velocity_field: Velocity field
boundary_conditions: Boundary conditions
source_term: Source/sink term
Returns:
Steady-state concentration
**Source:** [Line 115](Python/Multiphysics/transport.py#L115)
##### `solve_transient(self, time_span, dt, initial_concentration, velocity_field, boundary_conditions, source_term)`
Solve transient CDR equation.
Args:
time_span: Time interval
dt: Time step
initial_concentration: Initial conditions
velocity_field: Velocity field
boundary_conditions: Boundary conditions
source_term: Time-dependent source function
Returns:
Time history of solution
**Source:** [Line 158](Python/Multiphysics/transport.py#L158)
##### `_build_convection_matrix(self)`
Build convection matrix.
**Source:** [Line 227](Python/Multiphysics/transport.py#L227)
##### `_compute_reaction_rate(self, concentration)`
Compute reaction rate.
**Source:** [Line 245](Python/Multiphysics/transport.py#L245)
##### `_apply_transport_bc(self, bc, matrix, rhs)`
Apply transport boundary conditions.
**Source:** [Line 254](Python/Multiphysics/transport.py#L254)
**Class Source:** [Line 62](Python/Multiphysics/transport.py#L62)
### `PorousMediaFlow`
Flow in porous media solver.
Solves Darcy's law and continuity equation for flow in porous media.
#### Methods
##### `__init__(self, mesh, transport_props, fluid_props)`
Initialize porous media flow solver.
Args:
mesh: Finite element mesh
transport_props: Transport properties
fluid_props: Fluid properties
**Source:** [Line 274](Python/Multiphysics/transport.py#L274)
##### `_setup_system(self)`
Setup flow system.
**Source:** [Line 298](Python/Multiphysics/transport.py#L298)
##### `solve_flow(self, boundary_conditions, source_term)`
Solve porous media flow.
Args:
boundary_conditions: Flow boundary conditions
source_term: Source/sink term
Returns:
Flow solution (pressure, velocity)
**Source:** [Line 316](Python/Multiphysics/transport.py#L316)
##### `_compute_darcy_velocity(self)`
Compute velocity from pressure using Darcy's law.
**Source:** [Line 349](Python/Multiphysics/transport.py#L349)
##### `_apply_flow_bc(self, bc, matrix, rhs)`
Apply flow boundary conditions.
**Source:** [Line 372](Python/Multiphysics/transport.py#L372)
**Class Source:** [Line 268](Python/Multiphysics/transport.py#L268)
### `ReactiveTransport`
Reactive transport in porous media.
Couples flow, transport, and chemical reactions.
#### Methods
##### `__init__(self, mesh, transport_props, fluid_props, species_data, reaction_data)`
Initialize reactive transport solver.
Args:
mesh: Finite element mesh
transport_props: Transport properties
fluid_props: Fluid properties
species_data: Species information
reaction_data: Reaction data
**Source:** [Line 392](Python/Multiphysics/transport.py#L392)
##### `setup_domains(self)`
Setup flow and transport domains.
**Source:** [Line 438](Python/Multiphysics/transport.py#L438)
##### `setup_coupling(self)`
Setup reactive transport coupling.
**Source:** [Line 444](Python/Multiphysics/transport.py#L444)
##### `solve_reactive_transport(self, time_span, dt, initial_concentrations, boundary_conditions)`
Solve coupled reactive transport.
Args:
time_span: Time interval
dt: Time step
initial_concentrations: Initial concentrations for each species
boundary_conditions: Combined boundary conditions
Returns:
Reactive transport solution
**Source:** [Line 449](Python/Multiphysics/transport.py#L449)
##### `_compute_reaction_coupling(self, species_index, concentrations)`
Compute reaction source term for species coupling.
**Source:** [Line 526](Python/Multiphysics/transport.py#L526)
**Class Source:** [Line 386](Python/Multiphysics/transport.py#L386)
### `MultiphysicsTransport`
General multiphysics transport framework.
Integrates various transport phenomena with other physics.
#### Methods
##### `__init__(self, mesh, physics_list)`
Initialize multiphysics transport.
Args:
mesh: Finite element mesh
physics_list: List of physics to couple
**Source:** [Line 556](Python/Multiphysics/transport.py#L556)
##### `add_transport_physics(self, physics_name, solver)`
Add a transport physics solver.
Args:
physics_name: Name of physics domain
solver: Physics solver
**Source:** [Line 570](Python/Multiphysics/transport.py#L570)
##### `solve_multiphysics_transport(self, time_span, dt, initial_conditions, boundary_conditions)`
Solve multiphysics transport problem.
Args:
time_span: Time interval
dt: Time step
initial_conditions: Initial conditions for all physics
boundary_conditions: Boundary conditions for all physics
Returns:
Multiphysics solution
**Source:** [Line 580](Python/Multiphysics/transport.py#L580)
**Class Source:** [Line 550](Python/Multiphysics/transport.py#L550)