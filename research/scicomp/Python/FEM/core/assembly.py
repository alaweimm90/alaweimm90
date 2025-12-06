"""
Global Assembly Procedures for Finite Element Analysis
Comprehensive assembly algorithms for finite element systems including
global matrix assembly, boundary condition application, and solution procedures.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import sparse
import warnings
from .finite_elements import FiniteElementBase, Node, Element, create_element_factory
from .mesh_generation import Mesh
from ..utils.material_properties import MaterialProperty
class GlobalAssembly:
    """
    Global assembly system for finite element analysis.
    Features:
    - Global stiffness and mass matrix assembly
    - Boundary condition application
    - Load vector assembly
    - Degree of freedom management
    """
    def __init__(self, mesh: Mesh, materials: Dict[int, MaterialProperty]):
        """
        Initialize global assembly system.
        Parameters:
            mesh: Finite element mesh
            materials: Dictionary of material properties by material ID
        """
        self.mesh = mesh
        self.materials = materials
        self.element_factory = create_element_factory()
        # DOF management
        self.num_dofs: Optional[int] = None
        self.dof_map: Dict[int, List[int]] = {}  # node_id -> list of DOF IDs
        self.global_dof_to_node: Dict[int, Tuple[int, int]] = {}  # global_dof -> (node_id, local_dof)
        # System matrices
        self.global_stiffness: Optional[sparse.csr_matrix] = None
        self.global_mass: Optional[sparse.csr_matrix] = None
        self.load_vector: Optional[np.ndarray] = None
        # Boundary conditions
        self.prescribed_dofs: List[int] = []
        self.prescribed_values: List[float] = []
        # Initialize DOF mapping
        self._initialize_dof_mapping()
    def _initialize_dof_mapping(self):
        """Initialize degree of freedom mapping."""
        dofs_per_node = self._get_dofs_per_node()
        # Assign global DOF numbers
        global_dof = 0
        for node_id in sorted(self.mesh.nodes.keys()):
            node_dofs = list(range(global_dof, global_dof + dofs_per_node))
            self.dof_map[node_id] = node_dofs
            # Create reverse mapping
            for local_dof, global_dof_id in enumerate(node_dofs):
                self.global_dof_to_node[global_dof_id] = (node_id, local_dof)
            global_dof += dofs_per_node
        self.num_dofs = global_dof
        print(f"Initialized {self.num_dofs} degrees of freedom")
    def _get_dofs_per_node(self) -> int:
        """Determine number of DOFs per node based on element types."""
        if self.mesh.dimension == 1:
            return 1  # Axial displacement only
        elif self.mesh.dimension == 2:
            return 2  # x and y displacements
        elif self.mesh.dimension == 3:
            return 3  # x, y, and z displacements
        else:
            raise ValueError(f"Unsupported dimension: {self.mesh.dimension}")
    def assemble_global_stiffness(self) -> sparse.csr_matrix:
        """
        Assemble global stiffness matrix.
        Returns:
            Global stiffness matrix (sparse)
        """
        print("Assembling global stiffness matrix...")
        # Initialize sparse matrix in COO format for efficient assembly
        row_indices = []
        col_indices = []
        data = []
        for element in self.mesh.elements.values():
            # Get element stiffness matrix
            element_stiffness = self._compute_element_stiffness(element)
            # Get global DOF indices for this element
            element_dofs = self._get_element_dofs(element)
            # Add element contributions to global matrix
            for i, global_i in enumerate(element_dofs):
                for j, global_j in enumerate(element_dofs):
                    row_indices.append(global_i)
                    col_indices.append(global_j)
                    data.append(element_stiffness[i, j])
        # Create sparse matrix
        self.global_stiffness = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.num_dofs, self.num_dofs)
        ).tocsr()
        print(f"Global stiffness matrix assembled: {self.global_stiffness.shape}")
        return self.global_stiffness
    def assemble_global_mass(self) -> sparse.csr_matrix:
        """
        Assemble global mass matrix.
        Returns:
            Global mass matrix (sparse)
        """
        print("Assembling global mass matrix...")
        # Initialize sparse matrix in COO format
        row_indices = []
        col_indices = []
        data = []
        for element in self.mesh.elements.values():
            # Get element mass matrix
            element_mass = self._compute_element_mass(element)
            # Get global DOF indices for this element
            element_dofs = self._get_element_dofs(element)
            # Add element contributions to global matrix
            for i, global_i in enumerate(element_dofs):
                for j, global_j in enumerate(element_dofs):
                    row_indices.append(global_i)
                    col_indices.append(global_j)
                    data.append(element_mass[i, j])
        # Create sparse matrix
        self.global_mass = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.num_dofs, self.num_dofs)
        ).tocsr()
        print(f"Global mass matrix assembled: {self.global_mass.shape}")
        return self.global_mass
    def _compute_element_stiffness(self, element: Element) -> np.ndarray:
        """Compute element stiffness matrix."""
        # Get nodes for this element
        element_nodes = [self.mesh.nodes[node_id] for node_id in element.node_ids]
        # Get material properties
        material = self.materials[element.material_id]
        # Create finite element object
        element_type = element.element_type
        if element_type not in self.element_factory:
            raise ValueError(f"Unknown element type: {element_type}")
        fe_class = self.element_factory[element_type]
        # Handle different element constructors
        if element_type == 'bar1d':
            area = element.properties.get('cross_section_area', 1.0)
            fe_element = fe_class(element.id, element_nodes, material, area)
        elif element_type in ['triangle2d', 'quad2d']:
            thickness = element.properties.get('thickness', element.thickness)
            plane_stress = element.properties.get('plane_stress', True)
            fe_element = fe_class(element.id, element_nodes, material, thickness, plane_stress)
        elif element_type == 'tetrahedron3d':
            fe_element = fe_class(element.id, element_nodes, material)
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return fe_element.stiffness_matrix()
    def _compute_element_mass(self, element: Element) -> np.ndarray:
        """Compute element mass matrix."""
        # Get nodes for this element
        element_nodes = [self.mesh.nodes[node_id] for node_id in element.node_ids]
        # Get material properties
        material = self.materials[element.material_id]
        # Create finite element object
        element_type = element.element_type
        fe_class = self.element_factory[element_type]
        # Handle different element constructors
        if element_type == 'bar1d':
            area = element.properties.get('cross_section_area', 1.0)
            fe_element = fe_class(element.id, element_nodes, material, area)
        elif element_type in ['triangle2d', 'quad2d']:
            thickness = element.properties.get('thickness', element.thickness)
            plane_stress = element.properties.get('plane_stress', True)
            fe_element = fe_class(element.id, element_nodes, material, thickness, plane_stress)
        elif element_type == 'tetrahedron3d':
            fe_element = fe_class(element.id, element_nodes, material)
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return fe_element.mass_matrix()
    def _get_element_dofs(self, element: Element) -> List[int]:
        """Get global DOF indices for an element."""
        element_dofs = []
        for node_id in element.node_ids:
            element_dofs.extend(self.dof_map[node_id])
        return element_dofs
    def apply_boundary_conditions(self, displacement_bcs: Dict[Tuple[int, int], float]):
        """
        Apply displacement boundary conditions.
        Parameters:
            displacement_bcs: Dictionary of {(node_id, dof): value} boundary conditions
                             where dof is 0=x, 1=y, 2=z
        """
        print(f"Applying {len(displacement_bcs)} boundary conditions...")
        self.prescribed_dofs = []
        self.prescribed_values = []
        for (node_id, local_dof), value in displacement_bcs.items():
            if node_id not in self.dof_map:
                warnings.warn(f"Node {node_id} not found in mesh")
                continue
            if local_dof >= len(self.dof_map[node_id]):
                warnings.warn(f"DOF {local_dof} exceeds node {node_id} DOFs")
                continue
            global_dof = self.dof_map[node_id][local_dof]
            self.prescribed_dofs.append(global_dof)
            self.prescribed_values.append(value)
    def apply_point_loads(self, point_loads: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Apply point loads to create load vector.
        Parameters:
            point_loads: Dictionary of {(node_id, dof): force} point loads
        Returns:
            Global load vector
        """
        print(f"Applying {len(point_loads)} point loads...")
        self.load_vector = np.zeros(self.num_dofs)
        for (node_id, local_dof), force in point_loads.items():
            if node_id not in self.dof_map:
                warnings.warn(f"Node {node_id} not found in mesh")
                continue
            if local_dof >= len(self.dof_map[node_id]):
                warnings.warn(f"DOF {local_dof} exceeds node {node_id} DOFs")
                continue
            global_dof = self.dof_map[node_id][local_dof]
            self.load_vector[global_dof] += force
        return self.load_vector
    def solve_static(self, tolerance: float = 1e-8) -> np.ndarray:
        """
        Solve static equilibrium: K * u = f
        Parameters:
            tolerance: Solver tolerance
        Returns:
            Global displacement vector
        """
        if self.global_stiffness is None:
            raise ValueError("Global stiffness matrix not assembled")
        if self.load_vector is None:
            raise ValueError("Load vector not defined")
        print("Solving static equilibrium...")
        # Create modified system for boundary conditions
        K_modified = self.global_stiffness.copy()
        f_modified = self.load_vector.copy()
        # Apply displacement boundary conditions using penalty method
        penalty = 1e12 * np.max(np.abs(K_modified.data))
        for i, (global_dof, prescribed_value) in enumerate(zip(self.prescribed_dofs, self.prescribed_values)):
            # Penalty method: add large value to diagonal
            K_modified[global_dof, global_dof] += penalty
            f_modified[global_dof] += penalty * prescribed_value
        # Solve system
        from scipy.sparse.linalg import spsolve
        try:
            displacement = spsolve(K_modified, f_modified)
        except Exception as e:
            raise RuntimeError(f"Failed to solve linear system: {e}")
        # Verify boundary conditions
        bc_error = 0.0
        for global_dof, prescribed_value in zip(self.prescribed_dofs, self.prescribed_values):
            error = abs(displacement[global_dof] - prescribed_value)
            bc_error = max(bc_error, error)
        if bc_error > tolerance:
            warnings.warn(f"Boundary condition error: {bc_error}")
        print(f"Static solution completed. Max displacement: {np.max(np.abs(displacement)):.6e}")
        return displacement
    def compute_element_stresses(self, displacement: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute stresses in all elements.
        Parameters:
            displacement: Global displacement vector
        Returns:
            Dictionary of element stresses {element_id: stress_array}
        """
        element_stresses = {}
        for element in self.mesh.elements.values():
            # Get element displacement vector
            element_dofs = self._get_element_dofs(element)
            element_displacement = displacement[element_dofs]
            # Compute element stress
            stress = self._compute_element_stress(element, element_displacement)
            element_stresses[element.id] = stress
        return element_stresses
    def _compute_element_stress(self, element: Element, element_displacement: np.ndarray) -> np.ndarray:
        """Compute stress for a single element."""
        # Get nodes for this element
        element_nodes = [self.mesh.nodes[node_id] for node_id in element.node_ids]
        # Get material properties
        material = self.materials[element.material_id]
        # Create finite element object
        element_type = element.element_type
        fe_class = self.element_factory[element_type]
        # Handle different element types
        if element_type == 'bar1d':
            area = element.properties.get('cross_section_area', 1.0)
            fe_element = fe_class(element.id, element_nodes, material, area)
            # For bar element: stress = E * strain = E * (du/dx)
            L = fe_element.length
            strain = (element_displacement[1] - element_displacement[0]) / L
            stress = material.youngs_modulus * strain
            return np.array([stress])
        elif element_type in ['triangle2d', 'quad2d']:
            thickness = element.properties.get('thickness', element.thickness)
            plane_stress = element.properties.get('plane_stress', True)
            fe_element = fe_class(element.id, element_nodes, material, thickness, plane_stress)
            if element_type == 'triangle2d':
                # Constant strain triangle
                B = fe_element.strain_displacement_matrix()
                strain = B @ element_displacement
                D = fe_element.constitutive_matrix()
                stress = D @ strain
                return stress  # [σxx, σyy, τxy]
            else:  # quad2d
                # Evaluate stress at element center
                xi = np.array([0.0, 0.0])
                B = fe_element.strain_displacement_matrix(xi)
                strain = B @ element_displacement
                D = fe_element.constitutive_matrix()
                stress = D @ strain
                return stress  # [σxx, σyy, τxy]
        elif element_type == 'tetrahedron3d':
            fe_element = fe_class(element.id, element_nodes, material)
            # Constant strain tetrahedron
            B = fe_element.strain_displacement_matrix()
            strain = B @ element_displacement
            D = fe_element.constitutive_matrix()
            stress = D @ strain
            return stress  # [σxx, σyy, σzz, τyz, τxz, τxy]
        else:
            raise ValueError(f"Stress computation not implemented for {element_type}")
    def compute_reaction_forces(self, displacement: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute reaction forces at constrained nodes.
        Parameters:
            displacement: Global displacement vector
        Returns:
            Dictionary of reaction forces {node_id: force_vector}
        """
        if self.global_stiffness is None:
            raise ValueError("Global stiffness matrix not assembled")
        # Compute internal forces: f_internal = K * u
        internal_forces = self.global_stiffness @ displacement
        # Reaction forces = internal forces - applied loads
        reaction_vector = internal_forces - self.load_vector
        # Extract reaction forces for constrained nodes
        reactions = {}
        constrained_nodes = set()
        for global_dof in self.prescribed_dofs:
            node_id, local_dof = self.global_dof_to_node[global_dof]
            constrained_nodes.add(node_id)
        dofs_per_node = self._get_dofs_per_node()
        for node_id in constrained_nodes:
            node_dofs = self.dof_map[node_id]
            node_reactions = reaction_vector[node_dofs]
            reactions[node_id] = node_reactions
        return reactions
    def get_displacement_at_node(self, displacement: np.ndarray, node_id: int) -> np.ndarray:
        """
        Get displacement vector at a specific node.
        Parameters:
            displacement: Global displacement vector
            node_id: Node ID
        Returns:
            Node displacement vector
        """
        if node_id not in self.dof_map:
            raise ValueError(f"Node {node_id} not found")
        node_dofs = self.dof_map[node_id]
        return displacement[node_dofs]
    def compute_system_energy(self, displacement: np.ndarray) -> Dict[str, float]:
        """
        Compute system energy quantities.
        Parameters:
            displacement: Global displacement vector
        Returns:
            Dictionary of energy quantities
        """
        if self.global_stiffness is None or self.global_mass is None:
            raise ValueError("Global matrices not assembled")
        # Strain energy: U = 0.5 * u^T * K * u
        strain_energy = 0.5 * displacement.T @ (self.global_stiffness @ displacement)
        # Potential energy: V = -u^T * f
        potential_energy = -displacement.T @ self.load_vector
        # Total potential energy
        total_potential = strain_energy + potential_energy
        return {
            'strain_energy': float(strain_energy),
            'potential_energy': float(potential_energy),
            'total_potential': float(total_potential)
        }