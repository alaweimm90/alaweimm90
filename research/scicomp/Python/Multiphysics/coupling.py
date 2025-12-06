"""General Coupling Framework for Multiphysics Problems.
This module provides the core infrastructure for coupling different
physics domains, including coupling schemes, interface management,
and field transfer mechanisms.
Classes:
    CoupledSystem: Base class for coupled multiphysics systems
    CouplingInterface: Interface between coupled domains
    CouplingScheme: Enum for coupling strategies
    FieldTransfer: Field interpolation and projection
Functions:
    create_coupling_interface: Create interface between domains
    monolithic_coupling: Monolithic coupling approach
    partitioned_coupling: Partitioned coupling approach
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from scipy import interpolate, sparse
from scipy.sparse import linalg as sparse_linalg
class CouplingScheme(Enum):
    """Enumeration of coupling schemes."""
    MONOLITHIC = "monolithic"
    PARTITIONED_EXPLICIT = "partitioned_explicit"
    PARTITIONED_IMPLICIT = "partitioned_implicit"
    STAGGERED = "staggered"
    QUASI_NEWTON = "quasi_newton"
@dataclass
class CouplingData:
    """Data structure for coupling information."""
    field_name: str
    source_domain: str
    target_domain: str
    values: np.ndarray
    locations: Optional[np.ndarray] = None
    time: float = 0.0
    metadata: Dict[str, Any] = None
class CouplingInterface:
    """Interface between coupled physics domains.
    Manages data exchange, interpolation, and conservation
    properties at the interface between different physics.
    """
    def __init__(self,
                 name: str,
                 source_domain: str,
                 target_domain: str,
                 interface_type: str = "surface"):
        """Initialize coupling interface.
        Args:
            name: Interface identifier
            source_domain: Source physics domain
            target_domain: Target physics domain
            interface_type: Type of interface (surface, volume, point)
        """
        self.name = name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.interface_type = interface_type
        # Interface geometry
        self.source_nodes = None
        self.target_nodes = None
        self.mapping_matrix = None
        # Conservation tracking
        self.conservation_history = []
    def set_interface_geometry(self,
                             source_nodes: np.ndarray,
                             target_nodes: np.ndarray):
        """Set interface node locations.
        Args:
            source_nodes: Node coordinates in source domain
            target_nodes: Node coordinates in target domain
        """
        self.source_nodes = np.asarray(source_nodes)
        self.target_nodes = np.asarray(target_nodes)
        # Build mapping matrix
        self._build_mapping_matrix()
    def _build_mapping_matrix(self):
        """Build interpolation/projection matrix between domains."""
        if self.source_nodes is None or self.target_nodes is None:
            raise ValueError("Interface geometry not set")
        # Use nearest neighbor for simple mapping
        # In practice, use more sophisticated methods
        from scipy.spatial import cKDTree
        source_tree = cKDTree(self.source_nodes)
        distances, indices = source_tree.query(self.target_nodes)
        # Build sparse mapping matrix
        n_target = len(self.target_nodes)
        n_source = len(self.source_nodes)
        row_ind = np.arange(n_target)
        col_ind = indices
        data = np.ones(n_target)
        self.mapping_matrix = sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n_target, n_source)
        )
    def transfer_field(self,
                      field_data: np.ndarray,
                      direction: str = "forward",
                      conserve_integral: bool = False) -> np.ndarray:
        """Transfer field data across interface.
        Args:
            field_data: Field values to transfer
            direction: Transfer direction ('forward' or 'backward')
            conserve_integral: Whether to conserve integral quantities
        Returns:
            Transferred field values
        """
        if self.mapping_matrix is None:
            raise ValueError("Mapping matrix not built")
        if direction == "forward":
            # Source to target
            transferred = self.mapping_matrix @ field_data
            if conserve_integral:
                # Adjust to conserve integral
                source_integral = np.sum(field_data)
                target_integral = np.sum(transferred)
                if target_integral > 0:
                    transferred *= source_integral / target_integral
        else:
            # Target to source (transpose)
            transferred = self.mapping_matrix.T @ field_data
            if conserve_integral:
                source_integral = np.sum(field_data)
                target_integral = np.sum(transferred)
                if target_integral > 0:
                    transferred *= source_integral / target_integral
        return transferred
    def check_conservation(self,
                          source_field: np.ndarray,
                          target_field: np.ndarray,
                          quantity: str = "mass") -> float:
        """Check conservation of quantities across interface.
        Args:
            source_field: Source domain field
            target_field: Target domain field
            quantity: Quantity to check (mass, momentum, energy)
        Returns:
            Relative conservation error
        """
        source_total = np.sum(source_field)
        target_total = np.sum(target_field)
        if source_total > 0:
            relative_error = abs(target_total - source_total) / source_total
        else:
            relative_error = abs(target_total)
        # Store in history
        self.conservation_history.append({
            'quantity': quantity,
            'source_total': source_total,
            'target_total': target_total,
            'relative_error': relative_error
        })
        return relative_error
class FieldTransfer:
    """Advanced field transfer operations.
    Provides various interpolation and projection methods
    for transferring fields between non-matching meshes.
    """
    def __init__(self, method: str = "linear"):
        """Initialize field transfer.
        Args:
            method: Transfer method (linear, rbf, conservative)
        """
        self.method = method
        self.interpolator = None
    def setup_interpolation(self,
                          source_points: np.ndarray,
                          target_points: np.ndarray):
        """Setup interpolation between point sets.
        Args:
            source_points: Source mesh points
            target_points: Target mesh points
        """
        self.source_points = source_points
        self.target_points = target_points
        if self.method == "linear":
            # Use linear interpolation for structured grids
            pass
        elif self.method == "rbf":
            # Radial basis function interpolation
            from scipy.interpolate import RBFInterpolator
            self.interpolator = RBFInterpolator(
                source_points, np.zeros(len(source_points)),
                kernel="thin_plate_spline"
            )
        elif self.method == "conservative":
            # Conservative interpolation (preserves integrals)
            self._setup_conservative_transfer()
    def _setup_conservative_transfer(self):
        """Setup conservative field transfer."""
        # Implement conservative interpolation
        # This requires mesh connectivity information
        warnings.warn("Conservative transfer not fully implemented")
    def transfer(self, field_values: np.ndarray) -> np.ndarray:
        """Transfer field values.
        Args:
            field_values: Values at source points
        Returns:
            Interpolated values at target points
        """
        if self.method == "linear":
            # Use scipy griddata for unstructured interpolation
            from scipy.interpolate import griddata
            transferred = griddata(
                self.source_points, field_values, self.target_points,
                method='linear', fill_value=0.0
            )
        elif self.method == "rbf" and self.interpolator is not None:
            # Update RBF with new values
            self.interpolator.values = field_values
            transferred = self.interpolator(self.target_points)
        else:
            # Fallback to nearest neighbor
            from scipy.spatial import cKDTree
            tree = cKDTree(self.source_points)
            _, indices = tree.query(self.target_points)
            transferred = field_values[indices]
        return transferred
class CoupledSystem(ABC):
    """Abstract base class for coupled multiphysics systems.
    Provides framework for implementing various multiphysics
    coupling strategies and solution algorithms.
    """
    def __init__(self,
                 name: str,
                 physics_domains: List[str],
                 coupling_scheme: CouplingScheme = CouplingScheme.PARTITIONED_IMPLICIT):
        """Initialize coupled system.
        Args:
            name: System identifier
            physics_domains: List of physics domain names
            coupling_scheme: Coupling strategy to use
        """
        self.name = name
        self.physics_domains = physics_domains
        self.coupling_scheme = coupling_scheme
        # Domain solvers
        self.domain_solvers = {}
        # Coupling interfaces
        self.interfaces = {}
        # Solution state
        self.state = {}
        self.time = 0.0
        # Convergence parameters
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.relaxation_parameter = 1.0
    @abstractmethod
    def setup_domains(self):
        """Setup individual physics domains."""
        pass
    @abstractmethod
    def setup_coupling(self):
        """Setup coupling between domains."""
        pass
    def add_domain_solver(self, domain: str, solver: Any):
        """Add solver for physics domain.
        Args:
            domain: Domain name
            solver: Domain solver object
        """
        if domain not in self.physics_domains:
            raise ValueError(f"Unknown domain: {domain}")
        self.domain_solvers[domain] = solver
    def add_interface(self, interface: CouplingInterface):
        """Add coupling interface.
        Args:
            interface: Coupling interface object
        """
        self.interfaces[interface.name] = interface
    def solve(self,
              time_span: Tuple[float, float],
              dt: float,
              initial_conditions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Solve coupled system.
        Args:
            time_span: Time interval (t0, tf)
            dt: Time step size
            initial_conditions: Initial conditions for each domain
        Returns:
            Solution dictionary
        """
        t0, tf = time_span
        self.time = t0
        # Set initial conditions
        if initial_conditions:
            for domain, ic in initial_conditions.items():
                self.state[domain] = ic
        # Time stepping loop
        solution_history = {domain: [] for domain in self.physics_domains}
        while self.time < tf:
            # Advance time
            self.time += dt
            # Solve coupled system for current time step
            if self.coupling_scheme == CouplingScheme.MONOLITHIC:
                self._solve_monolithic(dt)
            elif self.coupling_scheme == CouplingScheme.PARTITIONED_EXPLICIT:
                self._solve_partitioned_explicit(dt)
            elif self.coupling_scheme == CouplingScheme.PARTITIONED_IMPLICIT:
                self._solve_partitioned_implicit(dt)
            else:
                raise NotImplementedError(f"Coupling scheme {self.coupling_scheme} not implemented")
            # Store solution
            for domain in self.physics_domains:
                solution_history[domain].append(self.state[domain].copy())
        return {
            'time': np.arange(t0 + dt, tf + dt, dt),
            'solution': solution_history,
            'final_state': self.state.copy()
        }
    def _solve_monolithic(self, dt: float):
        """Monolithic coupling solution."""
        # Solve all physics simultaneously
        # This requires assembling a global system
        raise NotImplementedError("Monolithic coupling requires problem-specific implementation")
    def _solve_partitioned_explicit(self, dt: float):
        """Explicit partitioned coupling."""
        # Solve each domain sequentially with explicit coupling
        for domain in self.physics_domains:
            if domain in self.domain_solvers:
                # Get coupling data from other domains
                coupling_data = self._get_coupling_data(domain)
                # Solve domain
                self.state[domain] = self.domain_solvers[domain].solve(
                    self.state.get(domain), dt, coupling_data
                )
    def _solve_partitioned_implicit(self, dt: float):
        """Implicit partitioned coupling with fixed-point iteration."""
        converged = False
        iteration = 0
        # Store previous states
        prev_states = {domain: self.state.get(domain, None)
                      for domain in self.physics_domains}
        while not converged and iteration < self.max_iterations:
            iteration += 1
            # Store current iteration states
            current_states = {}
            # Solve each domain
            for domain in self.physics_domains:
                if domain in self.domain_solvers:
                    # Get coupling data
                    coupling_data = self._get_coupling_data(domain)
                    # Solve domain
                    new_state = self.domain_solvers[domain].solve(
                        self.state.get(domain), dt, coupling_data
                    )
                    # Apply relaxation
                    if prev_states[domain] is not None and iteration > 1:
                        new_state = (self.relaxation_parameter * new_state +
                                   (1 - self.relaxation_parameter) * self.state[domain])
                    current_states[domain] = new_state
            # Check convergence
            max_change = 0.0
            for domain in self.physics_domains:
                if domain in current_states and domain in self.state:
                    change = np.linalg.norm(current_states[domain] - self.state[domain])
                    max_change = max(max_change, change)
            # Update states
            self.state.update(current_states)
            # Check convergence
            if max_change < self.tolerance:
                converged = True
        if not converged:
            warnings.warn(f"Partitioned implicit coupling did not converge after {iteration} iterations")
    def _get_coupling_data(self, target_domain: str) -> Dict[str, Any]:
        """Get coupling data for target domain from other domains."""
        coupling_data = {}
        for interface_name, interface in self.interfaces.items():
            if interface.target_domain == target_domain:
                # Get data from source domain
                source_domain = interface.source_domain
                if source_domain in self.state:
                    # Transfer field data
                    # This is problem-specific
                    coupling_data[interface_name] = {
                        'interface': interface,
                        'source_state': self.state[source_domain]
                    }
        return coupling_data
# Convenience functions
def create_coupling_interface(source_domain: str,
                            target_domain: str,
                            source_nodes: np.ndarray,
                            target_nodes: np.ndarray,
                            name: Optional[str] = None) -> CouplingInterface:
    """Create coupling interface between domains.
    Args:
        source_domain: Source domain name
        target_domain: Target domain name
        source_nodes: Source interface nodes
        target_nodes: Target interface nodes
        name: Interface name
    Returns:
        Configured coupling interface
    """
    if name is None:
        name = f"{source_domain}_to_{target_domain}"
    interface = CouplingInterface(name, source_domain, target_domain)
    interface.set_interface_geometry(source_nodes, target_nodes)
    return interface
def monolithic_coupling(physics_models: Dict[str, Any],
                       coupling_terms: Dict[str, Callable],
                       dt: float) -> Dict[str, np.ndarray]:
    """Monolithic coupling approach.
    Args:
        physics_models: Dictionary of physics models
        coupling_terms: Coupling term functions
        dt: Time step
    Returns:
        Updated states for all physics
    """
    # This is a placeholder for monolithic coupling
    # Actual implementation depends on specific physics
    warnings.warn("Monolithic coupling requires problem-specific implementation")
    return {}
def partitioned_coupling(physics_models: Dict[str, Any],
                        interfaces: List[CouplingInterface],
                        dt: float,
                        scheme: str = "implicit",
                        max_iterations: int = 50,
                        tolerance: float = 1e-6) -> Dict[str, np.ndarray]:
    """Partitioned coupling approach.
    Args:
        physics_models: Dictionary of physics models
        interfaces: List of coupling interfaces
        dt: Time step
        scheme: Coupling scheme (explicit/implicit)
        max_iterations: Maximum iterations for implicit
        tolerance: Convergence tolerance
    Returns:
        Updated states for all physics
    """
    states = {}
    if scheme == "explicit":
        # Explicit coupling - solve sequentially
        for name, model in physics_models.items():
            # Get interface data for this model
            interface_data = {}
            for interface in interfaces:
                if interface.target_domain == name:
                    # Get data from source
                    source_name = interface.source_domain
                    if source_name in states:
                        interface_data[interface.name] = states[source_name]
            # Solve model
            states[name] = model.solve(dt, interface_data)
    else:
        # Implicit coupling with iterations
        # Initialize states
        for name, model in physics_models.items():
            states[name] = model.get_state()
        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            iteration += 1
            prev_states = {name: state.copy() for name, state in states.items()}
            # Solve each model
            for name, model in physics_models.items():
                # Get interface data
                interface_data = {}
                for interface in interfaces:
                    if interface.target_domain == name:
                        source_name = interface.source_domain
                        if source_name in states:
                            # Transfer data through interface
                            source_data = states[source_name]
                            transferred = interface.transfer_field(source_data)
                            interface_data[interface.name] = transferred
                # Solve model
                states[name] = model.solve(dt, interface_data)
            # Check convergence
            max_change = 0.0
            for name in states:
                change = np.linalg.norm(states[name] - prev_states[name])
                max_change = max(max_change, change)
            if max_change < tolerance:
                converged = True
    return states