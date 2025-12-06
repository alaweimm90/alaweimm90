"""Multi-Physics Transport Phenomena Module.
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
"""
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import warnings
from .coupling import CoupledSystem, CouplingInterface, CouplingScheme
@dataclass
class TransportProperties:
    """Transport material properties."""
    diffusivity: float  # m²/s
    porosity: float = 1.0  # Dimensionless
    tortuosity: float = 1.0  # Dimensionless
    permeability: Optional[float] = None  # m² (for porous media)
    dispersivity: Optional[float] = None  # m (for dispersion)
    @property
    def effective_diffusivity(self) -> float:
        """Effective diffusivity in porous media."""
        return self.diffusivity * self.porosity / self.tortuosity
@dataclass
class FluidProperties:
    """Fluid properties for transport."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
@dataclass
class ReactionData:
    """Chemical reaction data."""
    stoichiometry: np.ndarray  # Stoichiometric matrix
    rate_constants: np.ndarray  # Reaction rate constants
    activation_energy: Optional[np.ndarray] = None  # J/mol
    reaction_orders: Optional[np.ndarray] = None  # Reaction orders
class ConvectionDiffusionReaction:
    """Convection-diffusion-reaction equation solver.
    Solves: ∂c/∂t + ∇·(vc) - ∇·(D∇c) = R(c)
    where c is concentration, v is velocity, D is diffusivity, R is reaction.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 transport_props: TransportProperties,
                 reaction_data: Optional[ReactionData] = None):
        """Initialize CDR solver.
        Args:
            mesh: Finite element mesh
            transport_props: Transport properties
            reaction_data: Reaction specifications
        """
        self.mesh = mesh
        self.transport_props = transport_props
        self.reaction_data = reaction_data
        # Solution variables
        self.concentration = None
        self.velocity = None
        # System matrices
        self.mass_matrix = None
        self.diffusion_matrix = None
        self.convection_matrix = None
        # Time integration
        self.dt = 0.01
        self.theta = 0.5  # Crank-Nicolson
        self._setup_system()
    def _setup_system(self):
        """Setup finite element system."""
        n_nodes = len(self.mesh['nodes'])
        # Initialize concentration field
        self.concentration = np.zeros(n_nodes)
        # Build system matrices (simplified)
        # Mass matrix
        self.mass_matrix = sparse.eye(n_nodes) * self.transport_props.porosity
        # Diffusion matrix: ∫ D ∇φ·∇ψ dΩ
        self.diffusion_matrix = sparse.eye(n_nodes) * self.transport_props.effective_diffusivity
        # Convection matrix will be built when velocity is known
    def solve_steady_state(self,
                         velocity_field: np.ndarray,
                         boundary_conditions: Dict[str, Any],
                         source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve steady-state CDR equation.
        Args:
            velocity_field: Velocity field
            boundary_conditions: Boundary conditions
            source_term: Source/sink term
        Returns:
            Steady-state concentration
        """
        self.velocity = velocity_field
        n_nodes = len(self.mesh['nodes'])
        # Build convection matrix
        self._build_convection_matrix()
        # System matrix: -D∇²c + v·∇c = S
        system_matrix = self.diffusion_matrix + self.convection_matrix
        # Right-hand side
        rhs = np.zeros(n_nodes)
        if source_term is not None:
            rhs += source_term
        # Add reaction terms
        if self.reaction_data is not None:
            # Linear reaction approximation for steady state
            # R = -k*c (first order decay)
            k = self.reaction_data.rate_constants[0] if len(self.reaction_data.rate_constants) > 0 else 0
            system_matrix += sparse.eye(n_nodes) * k
        # Apply boundary conditions
        self._apply_transport_bc(boundary_conditions, system_matrix, rhs)
        # Solve
        self.concentration = sparse_linalg.spsolve(system_matrix, rhs)
        return self.concentration
    def solve_transient(self,
                       time_span: Tuple[float, float],
                       dt: float,
                       initial_concentration: np.ndarray,
                       velocity_field: np.ndarray,
                       boundary_conditions: Dict[str, Any],
                       source_term: Optional[Callable] = None) -> Dict[str, List]:
        """Solve transient CDR equation.
        Args:
            time_span: Time interval
            dt: Time step
            initial_concentration: Initial conditions
            velocity_field: Velocity field
            boundary_conditions: Boundary conditions
            source_term: Time-dependent source function
        Returns:
            Time history of solution
        """
        self.dt = dt
        self.velocity = velocity_field
        t0, tf = time_span
        n_steps = int((tf - t0) / dt)
        # Initialize
        self.concentration = initial_concentration.copy()
        self._build_convection_matrix()
        # Storage
        history = {
            'time': [],
            'concentration': []
        }
        # Time stepping
        for step in range(n_steps):
            t = t0 + (step + 1) * dt
            # Source term
            source = np.zeros(len(self.concentration))
            if source_term is not None:
                source = source_term(t)
            # Reaction term
            reaction = self._compute_reaction_rate(self.concentration)
            # Time integration matrix
            # M/dt + θ*(K + C) = M/dt - (1-θ)*(K + C) + S + R
            lhs_matrix = (self.mass_matrix / dt +
                         self.theta * (self.diffusion_matrix + self.convection_matrix))
            rhs = (self.mass_matrix @ self.concentration / dt -
                   (1 - self.theta) * (self.diffusion_matrix + self.convection_matrix) @ self.concentration +
                   source + reaction)
            # Apply BC
            self._apply_transport_bc(boundary_conditions, lhs_matrix, rhs)
            # Solve
            self.concentration = sparse_linalg.spsolve(lhs_matrix, rhs)
            # Store
            history['time'].append(t)
            history['concentration'].append(self.concentration.copy())
        return history
    def _build_convection_matrix(self):
        """Build convection matrix."""
        n_nodes = len(self.mesh['nodes'])
        # Simplified convection matrix using upwinding
        # In practice, use proper finite element assembly
        # Approximate velocity gradient
        h = 0.01  # Grid spacing
        self.convection_matrix = sparse.diags(
            [-1, 1], [-1, 1], shape=(n_nodes, n_nodes)
        ) / (2 * h)
        # Scale by velocity magnitude
        if self.velocity is not None:
            v_magnitude = np.linalg.norm(self.velocity, axis=1) if self.velocity.ndim > 1 else np.abs(self.velocity)
            self.convection_matrix = sparse.diags(v_magnitude) @ self.convection_matrix
    def _compute_reaction_rate(self, concentration: np.ndarray) -> np.ndarray:
        """Compute reaction rate."""
        if self.reaction_data is None:
            return np.zeros_like(concentration)
        # Simple first-order reaction: R = -k*c
        k = self.reaction_data.rate_constants[0] if len(self.reaction_data.rate_constants) > 0 else 0
        return -k * concentration
    def _apply_transport_bc(self, bc: Dict[str, Any], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply transport boundary conditions."""
        # Dirichlet BC (concentration)
        if 'concentration' in bc:
            for node, conc in bc['concentration'].items():
                matrix[node, node] = 1.0
                rhs[node] = conc
        # Neumann BC (flux)
        if 'flux' in bc:
            for node, flux in bc['flux'].items():
                rhs[node] += flux
class PorousMediaFlow:
    """Flow in porous media solver.
    Solves Darcy's law and continuity equation for flow in porous media.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 transport_props: TransportProperties,
                 fluid_props: FluidProperties):
        """Initialize porous media flow solver.
        Args:
            mesh: Finite element mesh
            transport_props: Transport properties
            fluid_props: Fluid properties
        """
        self.mesh = mesh
        self.transport_props = transport_props
        self.fluid_props = fluid_props
        # Solution variables
        self.pressure = None
        self.velocity = None
        # System matrices
        self.permeability_matrix = None
        self._setup_system()
    def _setup_system(self):
        """Setup flow system."""
        n_nodes = len(self.mesh['nodes'])
        # Initialize fields
        self.pressure = np.zeros(n_nodes)
        self.velocity = np.zeros((n_nodes, 2))  # 2D
        # Permeability matrix for Darcy's law
        # v = -(k/μ)∇p
        if self.transport_props.permeability is not None:
            permeability_coeff = self.transport_props.permeability / self.fluid_props.viscosity
        else:
            permeability_coeff = 1e-12  # Default low permeability
        # Pressure equation: ∇·(k/μ ∇p) = S
        self.permeability_matrix = sparse.eye(n_nodes) * permeability_coeff
    def solve_flow(self,
                  boundary_conditions: Dict[str, Any],
                  source_term: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Solve porous media flow.
        Args:
            boundary_conditions: Flow boundary conditions
            source_term: Source/sink term
        Returns:
            Flow solution (pressure, velocity)
        """
        n_nodes = len(self.mesh['nodes'])
        # Right-hand side
        rhs = np.zeros(n_nodes)
        if source_term is not None:
            rhs += source_term
        # Apply boundary conditions
        self._apply_flow_bc(boundary_conditions, self.permeability_matrix, rhs)
        # Solve for pressure
        self.pressure = sparse_linalg.spsolve(self.permeability_matrix, rhs)
        # Compute velocity from Darcy's law
        self.velocity = self._compute_darcy_velocity()
        return {
            'pressure': self.pressure,
            'velocity': self.velocity
        }
    def _compute_darcy_velocity(self) -> np.ndarray:
        """Compute velocity from pressure using Darcy's law."""
        n_nodes = len(self.mesh['nodes'])
        velocity = np.zeros((n_nodes, 2))
        # v = -(k/μ)∇p
        # Simplified gradient computation
        h = 0.01  # Grid spacing
        for i in range(1, n_nodes-1):
            # Finite difference gradient
            pressure_grad_x = (self.pressure[i+1] - self.pressure[i-1]) / (2*h)
            # Darcy velocity
            if self.transport_props.permeability is not None:
                k_over_mu = self.transport_props.permeability / self.fluid_props.viscosity
            else:
                k_over_mu = 1e-12
            velocity[i, 0] = -k_over_mu * pressure_grad_x
        return velocity
    def _apply_flow_bc(self, bc: Dict[str, Any], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply flow boundary conditions."""
        # Pressure BC
        if 'pressure' in bc:
            for node, pressure in bc['pressure'].items():
                matrix[node, node] = 1.0
                rhs[node] = pressure
        # Flow rate BC
        if 'flow_rate' in bc:
            for node, rate in bc['flow_rate'].items():
                rhs[node] += rate
class ReactiveTransport(CoupledSystem):
    """Reactive transport in porous media.
    Couples flow, transport, and chemical reactions.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 transport_props: TransportProperties,
                 fluid_props: FluidProperties,
                 species_data: List[Dict[str, Any]],
                 reaction_data: ReactionData):
        """Initialize reactive transport solver.
        Args:
            mesh: Finite element mesh
            transport_props: Transport properties
            fluid_props: Fluid properties
            species_data: Species information
            reaction_data: Reaction data
        """
        super().__init__("ReactiveTransport", ["flow", "transport"], CouplingScheme.STAGGERED)
        self.mesh = mesh
        self.transport_props = transport_props
        self.fluid_props = fluid_props
        self.species_data = species_data
        self.reaction_data = reaction_data
        self.n_species = len(species_data)
        # Component solvers
        self.flow_solver = PorousMediaFlow(mesh, transport_props, fluid_props)
        self.transport_solvers = []
        for i, species in enumerate(species_data):
            # Create transport properties for each species
            species_props = TransportProperties(
                diffusivity=species.get('diffusivity', transport_props.diffusivity),
                porosity=transport_props.porosity,
                tortuosity=transport_props.tortuosity
            )
            solver = ConvectionDiffusionReaction(mesh, species_props, reaction_data)
            self.transport_solvers.append(solver)
        # Solution storage
        self.concentrations = []
        self.setup_domains()
        self.setup_coupling()
    def setup_domains(self):
        """Setup flow and transport domains."""
        self.add_domain_solver("flow", self.flow_solver)
        for i, solver in enumerate(self.transport_solvers):
            self.add_domain_solver(f"species_{i}", solver)
    def setup_coupling(self):
        """Setup reactive transport coupling."""
        # Coupling through velocity field from flow to transport
        pass
    def solve_reactive_transport(self,
                               time_span: Tuple[float, float],
                               dt: float,
                               initial_concentrations: List[np.ndarray],
                               boundary_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Solve coupled reactive transport.
        Args:
            time_span: Time interval
            dt: Time step
            initial_concentrations: Initial concentrations for each species
            boundary_conditions: Combined boundary conditions
        Returns:
            Reactive transport solution
        """
        t0, tf = time_span
        n_steps = int((tf - t0) / dt)
        # Initialize concentrations
        self.concentrations = [conc.copy() for conc in initial_concentrations]
        # Separate boundary conditions
        flow_bc = boundary_conditions.get('flow', {})
        transport_bc = boundary_conditions.get('transport', {})
        # Solve initial flow
        flow_result = self.flow_solver.solve_flow(flow_bc)
        velocity_field = flow_result['velocity']
        # Storage
        history = {
            'time': [],
            'concentrations': [[] for _ in range(self.n_species)],
            'velocity': [],
            'pressure': []
        }
        # Time stepping
        for step in range(n_steps):
            t = t0 + (step + 1) * dt
            # Update flow if needed (for variable properties)
            # flow_result = self.flow_solver.solve_flow(flow_bc)
            # velocity_field = flow_result['velocity']
            # Solve transport for each species
            new_concentrations = []
            for i, (solver, conc) in enumerate(zip(self.transport_solvers, self.concentrations)):
                # Species-specific BC
                species_bc = transport_bc.get(f'species_{i}', transport_bc)
                # Reaction coupling
                reaction_source = self._compute_reaction_coupling(i, self.concentrations)
                # Solve transport step
                transport_result = solver.solve_transient(
                    (t-dt, t), dt, conc, velocity_field, species_bc,
                    lambda t_val: reaction_source
                )
                new_conc = transport_result['concentration'][-1]
                new_concentrations.append(new_conc)
            # Update concentrations
            self.concentrations = new_concentrations
            # Store results
            history['time'].append(t)
            for i, conc in enumerate(self.concentrations):
                history['concentrations'][i].append(conc.copy())
            history['velocity'].append(velocity_field.copy())
            history['pressure'].append(flow_result['pressure'].copy())
        return history
    def _compute_reaction_coupling(self, species_index: int, concentrations: List[np.ndarray]) -> np.ndarray:
        """Compute reaction source term for species coupling."""
        n_nodes = len(self.mesh['nodes'])
        reaction_rate = np.zeros(n_nodes)
        if self.reaction_data is None:
            return reaction_rate
        # Multiple reactions
        for reaction_idx in range(len(self.reaction_data.rate_constants)):
            k = self.reaction_data.rate_constants[reaction_idx]
            stoich = self.reaction_data.stoichiometry[reaction_idx, species_index]
            # Rate law (simplified first-order)
            rate = k
            for j, conc in enumerate(concentrations):
                if self.reaction_data.stoichiometry[reaction_idx, j] < 0:  # Reactant
                    rate *= np.mean(conc)  # Simplified
            reaction_rate += stoich * rate
        return reaction_rate
class MultiphysicsTransport(CoupledSystem):
    """General multiphysics transport framework.
    Integrates various transport phenomena with other physics.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 physics_list: List[str]):
        """Initialize multiphysics transport.
        Args:
            mesh: Finite element mesh
            physics_list: List of physics to couple
        """
        super().__init__("MultiphysicsTransport", physics_list, CouplingScheme.PARTITIONED_IMPLICIT)
        self.mesh = mesh
        self.physics_solvers = {}
    def add_transport_physics(self, physics_name: str, solver: Any):
        """Add a transport physics solver.
        Args:
            physics_name: Name of physics domain
            solver: Physics solver
        """
        self.physics_solvers[physics_name] = solver
        self.add_domain_solver(physics_name, solver)
    def solve_multiphysics_transport(self,
                                   time_span: Tuple[float, float],
                                   dt: float,
                                   initial_conditions: Dict[str, Any],
                                   boundary_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Solve multiphysics transport problem.
        Args:
            time_span: Time interval
            dt: Time step
            initial_conditions: Initial conditions for all physics
            boundary_conditions: Boundary conditions for all physics
        Returns:
            Multiphysics solution
        """
        # Use parent class solve method
        return self.solve(time_span, dt, initial_conditions)
# Utility functions
def species_transport(mesh: Dict[str, np.ndarray],
                     species_properties: List[Dict[str, float]],
                     flow_field: np.ndarray,
                     boundary_conditions: Dict[str, Any],
                     reactions: Optional[ReactionData] = None) -> Dict[str, List[np.ndarray]]:
    """Solve multi-species transport.
    Args:
        mesh: Finite element mesh
        species_properties: Properties for each species
        flow_field: Velocity field
        boundary_conditions: Transport BC
        reactions: Reaction data
    Returns:
        Multi-species concentration fields
    """
    n_species = len(species_properties)
    results = {'concentrations': []}
    for i, props in enumerate(species_properties):
        transport_props = TransportProperties(
            diffusivity=props['diffusivity'],
            porosity=props.get('porosity', 1.0)
        )
        solver = ConvectionDiffusionReaction(mesh, transport_props, reactions)
        # Species-specific BC
        species_bc = boundary_conditions.get(f'species_{i}', boundary_conditions)
        concentration = solver.solve_steady_state(flow_field, species_bc)
        results['concentrations'].append(concentration)
    return results
def coupled_flow_transport(mesh: Dict[str, np.ndarray],
                         transport_props: TransportProperties,
                         fluid_props: FluidProperties,
                         boundary_conditions: Dict[str, Any],
                         coupling_type: str = "one_way") -> Dict[str, np.ndarray]:
    """Solve coupled flow and transport.
    Args:
        mesh: Finite element mesh
        transport_props: Transport properties
        fluid_props: Fluid properties
        boundary_conditions: Combined BC
        coupling_type: Coupling strategy
    Returns:
        Coupled flow-transport solution
    """
    # Flow solver
    flow_solver = PorousMediaFlow(mesh, transport_props, fluid_props)
    # Transport solver
    cdr_solver = ConvectionDiffusionReaction(mesh, transport_props)
    # Separate BC
    flow_bc = boundary_conditions.get('flow', {})
    transport_bc = boundary_conditions.get('transport', {})
    if coupling_type == "one_way":
        # Flow → Transport
        flow_result = flow_solver.solve_flow(flow_bc)
        concentration = cdr_solver.solve_steady_state(
            flow_result['velocity'], transport_bc
        )
        return {
            'pressure': flow_result['pressure'],
            'velocity': flow_result['velocity'],
            'concentration': concentration
        }
    elif coupling_type == "two_way":
        # Iterative coupling
        max_iterations = 10
        tolerance = 1e-6
        # Initial guess
        flow_result = flow_solver.solve_flow(flow_bc)
        for iteration in range(max_iterations):
            # Solve transport
            concentration = cdr_solver.solve_steady_state(
                flow_result['velocity'], transport_bc
            )
            # Update flow properties based on concentration
            # (e.g., density-dependent flow)
            # Solve flow with updated properties
            prev_pressure = flow_result['pressure'].copy()
            flow_result = flow_solver.solve_flow(flow_bc)
            # Check convergence
            change = np.linalg.norm(flow_result['pressure'] - prev_pressure)
            if change < tolerance:
                break
        return {
            'pressure': flow_result['pressure'],
            'velocity': flow_result['velocity'],
            'concentration': concentration,
            'iterations': iteration + 1
        }
    else:
        raise ValueError(f"Unknown coupling type: {coupling_type}")