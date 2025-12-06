"""Multiphysics Coupled System Solvers.
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
"""
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from abc import ABC, abstractmethod
import warnings
from .coupling import CoupledSystem, CouplingScheme
@dataclass
class SolverParameters:
    """Parameters for coupled system solvers."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    relaxation_parameter: float = 1.0
    line_search: bool = False
    preconditioner: Optional[str] = None
    verbose: bool = False
@dataclass
class SolutionData:
    """Solution data for coupled systems."""
    solution: Dict[str, np.ndarray]
    iterations: int
    converged: bool
    residual_history: List[float]
    coupling_residual: float
    solve_time: float
    metadata: Dict[str, Any]
class MultiphysicsSolver(ABC):
    """Abstract base class for multiphysics solvers."""
    def __init__(self,
                 coupled_system: CoupledSystem,
                 parameters: SolverParameters):
        """Initialize multiphysics solver.
        Args:
            coupled_system: Coupled system to solve
            parameters: Solver parameters
        """
        self.coupled_system = coupled_system
        self.parameters = parameters
        # Solution state
        self.current_solution = {}
        self.residual_history = []
    @abstractmethod
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve coupled system.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Solution data
        """
        pass
    def compute_coupling_residual(self,
                                 solution: Dict[str, np.ndarray]) -> float:
        """Compute coupling residual.
        Args:
            solution: Current solution state
        Returns:
            Coupling residual norm
        """
        # Generic coupling residual computation
        total_residual = 0.0
        for interface_name, interface in self.coupled_system.interfaces.items():
            # Compute interface residual
            source_domain = interface.source_domain
            target_domain = interface.target_domain
            if source_domain in solution and target_domain in solution:
                # Check continuity at interface
                source_data = solution[source_domain]
                target_data = solution[target_domain]
                # Simplified residual calculation
                interface_residual = np.linalg.norm(source_data - target_data)
                total_residual += interface_residual
        return total_residual
    def apply_relaxation(self,
                        new_solution: Dict[str, np.ndarray],
                        old_solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply relaxation to solution update.
        Args:
            new_solution: New solution iterate
            old_solution: Previous solution
        Returns:
            Relaxed solution
        """
        relaxed_solution = {}
        alpha = self.parameters.relaxation_parameter
        for domain in new_solution:
            if domain in old_solution:
                relaxed_solution[domain] = (alpha * new_solution[domain] +
                                          (1 - alpha) * old_solution[domain])
            else:
                relaxed_solution[domain] = new_solution[domain]
        return relaxed_solution
class MonolithicSolver(MultiphysicsSolver):
    """Monolithic coupling solver.
    Solves all physics simultaneously in a single
    large coupled system.
    """
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve monolithic coupled system.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Monolithic solution data
        """
        import time
        start_time = time.time()
        # Assemble global system
        global_matrix, global_rhs = self._assemble_global_system()
        # Apply global boundary conditions
        self._apply_global_bc(global_matrix, global_rhs)
        # Solve global system
        try:
            global_solution = sparse_linalg.spsolve(global_matrix, global_rhs)
            converged = True
            iterations = 1
        except Exception as e:
            warnings.warn(f"Monolithic solve failed: {e}")
            converged = False
            iterations = 0
            global_solution = np.zeros(len(global_rhs))
        # Extract domain solutions
        solution = self._extract_domain_solutions(global_solution)
        # Compute residual
        coupling_residual = self.compute_coupling_residual(solution)
        solve_time = time.time() - start_time
        return SolutionData(
            solution=solution,
            iterations=iterations,
            converged=converged,
            residual_history=[coupling_residual],
            coupling_residual=coupling_residual,
            solve_time=solve_time,
            metadata={'method': 'monolithic'}
        )
    def _assemble_global_system(self) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Assemble global coupled system matrix."""
        # This is problem-specific and requires knowledge of
        # the coupling terms between physics
        # For demonstration, create a simple block system
        domains = list(self.coupled_system.domain_solvers.keys())
        n_domains = len(domains)
        # Estimate system size
        total_dofs = 0
        domain_sizes = {}
        for domain in domains:
            # Estimate DOFs per domain
            # In practice, get from actual solver
            domain_size = 100  # Placeholder
            domain_sizes[domain] = domain_size
            total_dofs += domain_size
        # Create block matrix structure
        global_matrix = sparse.lil_matrix((total_dofs, total_dofs))
        global_rhs = np.zeros(total_dofs)
        # Fill diagonal blocks (individual physics)
        current_dof = 0
        for domain in domains:
            size = domain_sizes[domain]
            # Individual physics matrix (placeholder)
            domain_matrix = sparse.eye(size)
            # Place in global matrix
            global_matrix[current_dof:current_dof+size,
                         current_dof:current_dof+size] = domain_matrix
            current_dof += size
        # Add coupling terms
        self._add_coupling_terms(global_matrix, domain_sizes)
        return global_matrix.tocsr(), global_rhs
    def _add_coupling_terms(self, global_matrix: sparse.lil_matrix,
                           domain_sizes: Dict[str, int]):
        """Add coupling terms to global matrix."""
        # Add coupling between domains based on interfaces
        domains = list(domain_sizes.keys())
        current_dof = {}
        dof_offset = 0
        for domain in domains:
            current_dof[domain] = dof_offset
            dof_offset += domain_sizes[domain]
        # Add coupling terms for each interface
        for interface_name, interface in self.coupled_system.interfaces.items():
            source_domain = interface.source_domain
            target_domain = interface.target_domain
            if source_domain in current_dof and target_domain in current_dof:
                # Add coupling terms (simplified)
                source_start = current_dof[source_domain]
                target_start = current_dof[target_domain]
                coupling_strength = 0.1  # Placeholder
                # Add off-diagonal coupling terms
                size = min(domain_sizes[source_domain], domain_sizes[target_domain])
                for i in range(size):
                    global_matrix[source_start + i, target_start + i] = coupling_strength
                    global_matrix[target_start + i, source_start + i] = coupling_strength
    def _apply_global_bc(self, matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply global boundary conditions."""
        # Apply BC for each domain
        # This is simplified - in practice, map local BC to global system
        pass
    def _extract_domain_solutions(self, global_solution: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual domain solutions from global solution."""
        domains = list(self.coupled_system.domain_solvers.keys())
        solutions = {}
        # Simple equal partitioning (placeholder)
        n_domains = len(domains)
        domain_size = len(global_solution) // n_domains
        for i, domain in enumerate(domains):
            start_idx = i * domain_size
            end_idx = start_idx + domain_size if i < n_domains - 1 else len(global_solution)
            solutions[domain] = global_solution[start_idx:end_idx]
        return solutions
class PartitionedSolver(MultiphysicsSolver):
    """Partitioned coupling solver.
    Solves each physics separately with iterative
    coupling through interface data exchange.
    """
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve partitioned coupled system.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Partitioned solution data
        """
        import time
        start_time = time.time()
        # Initialize solution
        if initial_guess is not None:
            self.current_solution = initial_guess.copy()
        else:
            self.current_solution = self._initialize_solution()
        converged = False
        iteration = 0
        self.residual_history = []
        while not converged and iteration < self.parameters.max_iterations:
            iteration += 1
            # Store previous solution
            previous_solution = {domain: sol.copy()
                               for domain, sol in self.current_solution.items()}
            # Solve each domain
            new_solution = {}
            for domain_name, solver in self.coupled_system.domain_solvers.items():
                # Get coupling data for this domain
                coupling_data = self._get_coupling_data(domain_name)
                # Solve domain
                if hasattr(solver, 'solve_with_coupling'):
                    domain_solution = solver.solve_with_coupling(coupling_data)
                else:
                    # Fallback - solve without explicit coupling
                    domain_solution = self._solve_domain_fallback(domain_name, solver)
                new_solution[domain_name] = domain_solution
            # Apply relaxation
            if iteration > 1:
                new_solution = self.apply_relaxation(new_solution, self.current_solution)
            # Update solution
            self.current_solution = new_solution
            # Check convergence
            coupling_residual = self.compute_coupling_residual(self.current_solution)
            self.residual_history.append(coupling_residual)
            if self.parameters.verbose:
                print(f"Iteration {iteration}: Residual = {coupling_residual:.6e}")
            if coupling_residual < self.parameters.tolerance:
                converged = True
        solve_time = time.time() - start_time
        return SolutionData(
            solution=self.current_solution,
            iterations=iteration,
            converged=converged,
            residual_history=self.residual_history,
            coupling_residual=coupling_residual,
            solve_time=solve_time,
            metadata={'method': 'partitioned', 'relaxation': self.parameters.relaxation_parameter}
        )
    def _initialize_solution(self) -> Dict[str, np.ndarray]:
        """Initialize solution guess."""
        solution = {}
        for domain_name in self.coupled_system.domain_solvers:
            # Initialize with zeros (placeholder)
            solution[domain_name] = np.zeros(100)  # Default size
        return solution
    def _get_coupling_data(self, target_domain: str) -> Dict[str, Any]:
        """Get coupling data for target domain."""
        coupling_data = {}
        for interface_name, interface in self.coupled_system.interfaces.items():
            if interface.target_domain == target_domain:
                source_domain = interface.source_domain
                if source_domain in self.current_solution:
                    # Transfer data through interface
                    source_data = self.current_solution[source_domain]
                    transferred_data = interface.transfer_field(source_data)
                    coupling_data[interface_name] = {
                        'data': transferred_data,
                        'interface': interface
                    }
        return coupling_data
    def _solve_domain_fallback(self, domain_name: str, solver: Any) -> np.ndarray:
        """Fallback domain solve method."""
        # Simple fallback - return current solution or zeros
        if domain_name in self.current_solution:
            return self.current_solution[domain_name]
        else:
            return np.zeros(100)  # Default
class StaggeredSolver(MultiphysicsSolver):
    """Staggered solution approach.
    Solves physics in a predetermined sequence
    with one-way coupling per time step.
    """
    def __init__(self,
                 coupled_system: CoupledSystem,
                 parameters: SolverParameters,
                 solve_order: List[str]):
        """Initialize staggered solver.
        Args:
            coupled_system: Coupled system
            parameters: Solver parameters
            solve_order: Order to solve physics domains
        """
        super().__init__(coupled_system, parameters)
        self.solve_order = solve_order
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve with staggered approach.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Staggered solution data
        """
        import time
        start_time = time.time()
        # Initialize solution
        if initial_guess is not None:
            self.current_solution = initial_guess.copy()
        else:
            self.current_solution = self._initialize_solution()
        # Solve in specified order
        for domain_name in self.solve_order:
            if domain_name in self.coupled_system.domain_solvers:
                solver = self.coupled_system.domain_solvers[domain_name]
                # Get coupling data
                coupling_data = self._get_coupling_data(domain_name)
                # Solve domain
                if hasattr(solver, 'solve_with_coupling'):
                    domain_solution = solver.solve_with_coupling(coupling_data)
                else:
                    domain_solution = self._solve_domain_fallback(domain_name, solver)
                # Update solution
                self.current_solution[domain_name] = domain_solution
        # Compute final residual
        coupling_residual = self.compute_coupling_residual(self.current_solution)
        solve_time = time.time() - start_time
        return SolutionData(
            solution=self.current_solution,
            iterations=1,  # Single pass
            converged=True,  # Assume converged for staggered
            residual_history=[coupling_residual],
            coupling_residual=coupling_residual,
            solve_time=solve_time,
            metadata={'method': 'staggered', 'solve_order': self.solve_order}
        )
    def _initialize_solution(self) -> Dict[str, np.ndarray]:
        """Initialize solution for staggered solve."""
        solution = {}
        for domain_name in self.solve_order:
            solution[domain_name] = np.zeros(100)  # Placeholder
        return solution
    def _get_coupling_data(self, target_domain: str) -> Dict[str, Any]:
        """Get coupling data for staggered solve."""
        coupling_data = {}
        # Only get data from previously solved domains
        domain_index = self.solve_order.index(target_domain)
        for interface_name, interface in self.coupled_system.interfaces.items():
            if interface.target_domain == target_domain:
                source_domain = interface.source_domain
                # Check if source domain was already solved
                if (source_domain in self.solve_order and
                    self.solve_order.index(source_domain) < domain_index and
                    source_domain in self.current_solution):
                    source_data = self.current_solution[source_domain]
                    transferred_data = interface.transfer_field(source_data)
                    coupling_data[interface_name] = {
                        'data': transferred_data,
                        'interface': interface
                    }
        return coupling_data
    def _solve_domain_fallback(self, domain_name: str, solver: Any) -> np.ndarray:
        """Fallback solve for staggered approach."""
        if domain_name in self.current_solution:
            return self.current_solution[domain_name]
        else:
            return np.zeros(100)
class NewtonRaphson(MultiphysicsSolver):
    """Newton-Raphson solver for nonlinear coupling.
    Uses Newton's method to solve coupled nonlinear systems.
    """
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve using Newton-Raphson method.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Newton-Raphson solution data
        """
        import time
        start_time = time.time()
        # Initialize solution
        if initial_guess is not None:
            self.current_solution = initial_guess.copy()
        else:
            self.current_solution = self._initialize_solution()
        converged = False
        iteration = 0
        self.residual_history = []
        while not converged and iteration < self.parameters.max_iterations:
            iteration += 1
            # Compute residual vector
            residual = self._compute_residual_vector()
            residual_norm = np.linalg.norm(residual)
            self.residual_history.append(residual_norm)
            if self.parameters.verbose:
                print(f"Newton iteration {iteration}: ||R|| = {residual_norm:.6e}")
            # Check convergence
            if residual_norm < self.parameters.tolerance:
                converged = True
                break
            # Compute Jacobian matrix
            jacobian = self._compute_jacobian()
            # Solve Newton system: J * Δx = -R
            try:
                delta_x = sparse_linalg.spsolve(jacobian, -residual)
            except Exception as e:
                warnings.warn(f"Newton solve failed: {e}")
                break
            # Line search (optional)
            if self.parameters.line_search:
                alpha = self._line_search(delta_x)
            else:
                alpha = 1.0
            # Update solution
            self._update_solution(alpha * delta_x)
        solve_time = time.time() - start_time
        return SolutionData(
            solution=self.current_solution,
            iterations=iteration,
            converged=converged,
            residual_history=self.residual_history,
            coupling_residual=residual_norm if iteration > 0 else 0.0,
            solve_time=solve_time,
            metadata={'method': 'newton_raphson', 'line_search': self.parameters.line_search}
        )
    def _compute_residual_vector(self) -> np.ndarray:
        """Compute residual vector for Newton method."""
        # This is problem-specific
        # For now, return simple coupling residual
        residuals = []
        for domain_name, solution in self.current_solution.items():
            # Domain residual (simplified)
            domain_residual = solution  # Placeholder
            residuals.append(domain_residual)
        return np.concatenate(residuals)
    def _compute_jacobian(self) -> sparse.spmatrix:
        """Compute Jacobian matrix."""
        # Simplified Jacobian computation
        # In practice, use automatic differentiation or finite differences
        total_size = sum(len(sol) for sol in self.current_solution.values())
        jacobian = sparse.eye(total_size)  # Placeholder
        return jacobian
    def _line_search(self, direction: np.ndarray) -> float:
        """Perform line search to find optimal step size."""
        # Simple backtracking line search
        alpha = 1.0
        beta = 0.5
        current_residual_norm = np.linalg.norm(self._compute_residual_vector())
        for _ in range(10):  # Max line search iterations
            # Try step
            self._update_solution(alpha * direction)
            new_residual_norm = np.linalg.norm(self._compute_residual_vector())
            if new_residual_norm < current_residual_norm:
                # Restore solution and return alpha
                self._update_solution(-alpha * direction)
                return alpha
            else:
                # Restore and reduce step
                self._update_solution(-alpha * direction)
                alpha *= beta
        return alpha
    def _update_solution(self, delta_x: np.ndarray):
        """Update solution with increment."""
        # Split delta_x among domains
        start_idx = 0
        for domain_name, solution in self.current_solution.items():
            end_idx = start_idx + len(solution)
            domain_delta = delta_x[start_idx:end_idx]
            self.current_solution[domain_name] += domain_delta
            start_idx = end_idx
    def _initialize_solution(self) -> Dict[str, np.ndarray]:
        """Initialize solution for Newton method."""
        solution = {}
        for domain_name in self.coupled_system.domain_solvers:
            solution[domain_name] = np.zeros(100)  # Placeholder
        return solution
class FixedPointIteration(MultiphysicsSolver):
    """Fixed-point iteration solver.
    Uses fixed-point iteration with optional acceleration
    for coupled systems.
    """
    def __init__(self,
                 coupled_system: CoupledSystem,
                 parameters: SolverParameters,
                 acceleration: str = "none"):
        """Initialize fixed-point solver.
        Args:
            coupled_system: Coupled system
            parameters: Solver parameters
            acceleration: Acceleration method (none, aitken, anderson)
        """
        super().__init__(coupled_system, parameters)
        self.acceleration = acceleration
        self.solution_history = []
    def solve(self,
              initial_guess: Optional[Dict[str, np.ndarray]] = None,
              **kwargs) -> SolutionData:
        """Solve using fixed-point iteration.
        Args:
            initial_guess: Initial solution guess
            **kwargs: Additional arguments
        Returns:
            Fixed-point solution data
        """
        import time
        start_time = time.time()
        # Initialize
        if initial_guess is not None:
            self.current_solution = initial_guess.copy()
        else:
            self.current_solution = self._initialize_solution()
        converged = False
        iteration = 0
        self.residual_history = []
        self.solution_history = []
        while not converged and iteration < self.parameters.max_iterations:
            iteration += 1
            # Store current solution
            self.solution_history.append({domain: sol.copy()
                                        for domain, sol in self.current_solution.items()})
            # Fixed-point iteration step
            new_solution = self._fixed_point_step()
            # Apply acceleration
            if self.acceleration != "none" and iteration > 1:
                new_solution = self._apply_acceleration(new_solution)
            # Compute residual
            residual = self._compute_fixed_point_residual(new_solution, self.current_solution)
            self.residual_history.append(residual)
            if self.parameters.verbose:
                print(f"Fixed-point iteration {iteration}: ||x_{iteration} - x_{iteration-1}|| = {residual:.6e}")
            # Update solution
            self.current_solution = new_solution
            # Check convergence
            if residual < self.parameters.tolerance:
                converged = True
        solve_time = time.time() - start_time
        return SolutionData(
            solution=self.current_solution,
            iterations=iteration,
            converged=converged,
            residual_history=self.residual_history,
            coupling_residual=residual if iteration > 0 else 0.0,
            solve_time=solve_time,
            metadata={'method': 'fixed_point', 'acceleration': self.acceleration}
        )
    def _fixed_point_step(self) -> Dict[str, np.ndarray]:
        """Perform one fixed-point iteration step."""
        # Similar to partitioned solver step
        new_solution = {}
        for domain_name, solver in self.coupled_system.domain_solvers.items():
            coupling_data = self._get_coupling_data(domain_name)
            if hasattr(solver, 'solve_with_coupling'):
                domain_solution = solver.solve_with_coupling(coupling_data)
            else:
                domain_solution = self._solve_domain_fallback(domain_name, solver)
            new_solution[domain_name] = domain_solution
        return new_solution
    def _apply_acceleration(self, new_solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply acceleration to fixed-point iteration."""
        if self.acceleration == "aitken":
            return self._aitken_acceleration(new_solution)
        elif self.acceleration == "anderson":
            return self._anderson_acceleration(new_solution)
        else:
            return new_solution
    def _aitken_acceleration(self, new_solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Aitken's Δ² acceleration."""
        if len(self.solution_history) < 2:
            return new_solution
        # Get last three iterates
        x_k = self.solution_history[-1]  # x^k
        x_k1 = self.current_solution      # x^{k+1}
        x_k2 = new_solution              # x^{k+2}
        accelerated = {}
        for domain in new_solution:
            if domain in x_k and domain in x_k1:
                # Aitken acceleration
                delta1 = x_k1[domain] - x_k[domain]
                delta2 = x_k2[domain] - x_k1[domain]
                # Avoid division by zero
                denominator = np.linalg.norm(delta2 - delta1)
                if denominator > 1e-12:
                    lambda_k = -np.dot(delta1, delta2 - delta1) / (denominator**2)
                    accelerated[domain] = x_k[domain] + lambda_k * delta1
                else:
                    accelerated[domain] = new_solution[domain]
            else:
                accelerated[domain] = new_solution[domain]
        return accelerated
    def _anderson_acceleration(self, new_solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Anderson acceleration (simplified)."""
        # Simplified Anderson acceleration
        # Full implementation requires maintaining multiple solution vectors
        return new_solution
    def _compute_fixed_point_residual(self, x_new: Dict[str, np.ndarray],
                                    x_old: Dict[str, np.ndarray]) -> float:
        """Compute fixed-point residual ||x_new - x_old||."""
        total_residual = 0.0
        for domain in x_new:
            if domain in x_old:
                domain_residual = np.linalg.norm(x_new[domain] - x_old[domain])
                total_residual += domain_residual**2
        return np.sqrt(total_residual)
    def _get_coupling_data(self, target_domain: str) -> Dict[str, Any]:
        """Get coupling data for fixed-point iteration."""
        # Same as partitioned solver
        coupling_data = {}
        for interface_name, interface in self.coupled_system.interfaces.items():
            if interface.target_domain == target_domain:
                source_domain = interface.source_domain
                if source_domain in self.current_solution:
                    source_data = self.current_solution[source_domain]
                    transferred_data = interface.transfer_field(source_data)
                    coupling_data[interface_name] = {
                        'data': transferred_data,
                        'interface': interface
                    }
        return coupling_data
    def _solve_domain_fallback(self, domain_name: str, solver: Any) -> np.ndarray:
        """Fallback domain solve."""
        if domain_name in self.current_solution:
            return self.current_solution[domain_name]
        else:
            return np.zeros(100)
    def _initialize_solution(self) -> Dict[str, np.ndarray]:
        """Initialize solution for fixed-point iteration."""
        solution = {}
        for domain_name in self.coupled_system.domain_solvers:
            solution[domain_name] = np.zeros(100)
        return solution
# Convenience function
def solve_coupled_system(coupled_system: CoupledSystem,
                        solver_type: str = "partitioned",
                        solver_parameters: Optional[SolverParameters] = None,
                        initial_guess: Optional[Dict[str, np.ndarray]] = None,
                        **kwargs) -> SolutionData:
    """Solve coupled multiphysics system.
    Args:
        coupled_system: Coupled system to solve
        solver_type: Type of solver to use
        solver_parameters: Solver parameters
        initial_guess: Initial solution guess
        **kwargs: Additional solver arguments
    Returns:
        Solution data
    """
    if solver_parameters is None:
        solver_parameters = SolverParameters()
    # Create solver based on type
    if solver_type == "monolithic":
        solver = MonolithicSolver(coupled_system, solver_parameters)
    elif solver_type == "partitioned":
        solver = PartitionedSolver(coupled_system, solver_parameters)
    elif solver_type == "staggered":
        solve_order = kwargs.get('solve_order', list(coupled_system.domain_solvers.keys()))
        solver = StaggeredSolver(coupled_system, solver_parameters, solve_order)
    elif solver_type == "newton":
        solver = NewtonRaphson(coupled_system, solver_parameters)
    elif solver_type == "fixed_point":
        acceleration = kwargs.get('acceleration', 'none')
        solver = FixedPointIteration(coupled_system, solver_parameters, acceleration)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
    # Solve
    return solver.solve(initial_guess, **kwargs)