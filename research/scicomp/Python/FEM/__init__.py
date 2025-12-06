"""
Finite Element Method (FEM) Package
Comprehensive finite element analysis package for structural mechanics,
dynamics, and multi-physics simulations.
Features:
- Element formulations (1D, 2D, 3D)
- Mesh generation (structured and unstructured)
- Global assembly and solvers
- Static, dynamic, and nonlinear analysis
- Post-processing and visualization
- Material property library
Example Usage:
    >>> from FEM import *
    >>> from FEM.utils import MaterialLibrary
    >>> # Create mesh and materials
    >>> mesh = create_simple_mesh()
    >>> steel = MaterialLibrary.steel_mild()
    >>> # Setup and solve
    >>> assembly = GlobalAssembly(mesh, {0: steel})
    >>> solver = StaticSolver(assembly)
    >>> displacement = solver.solve_direct()
    >>> # Post-process
    >>> post_processor = FEMPostProcessor(assembly, displacement)
    >>> post_processor.plot_deformed_shape()
"""
# Core modules
from .core.finite_elements import (
    Node, Element, FiniteElementBase,
    LinearBar1D, LinearTriangle2D, LinearQuadrilateral2D, LinearTetrahedron3D,
    create_element_factory
)
from .core.mesh_generation import (
    Mesh, MeshParameters, Geometry,
    StructuredMeshGenerator, UnstructuredMeshGenerator
)
from .core.assembly import GlobalAssembly
from .core.solvers import (
    StaticSolver, DynamicSolver, NonlinearSolver
)
from .core.post_processing import FEMPostProcessor
# Utilities
from .utils.material_properties import (
    MaterialProperty, IsotropicMaterial, OrthotropicMaterial, ViscoelasticMaterial,
    MaterialLibrary, create_custom_material, scale_material_properties
)
# Version information
__version__ = "1.0.0"
__author__ = "SciComp"
__email__ = "scicomp@berkeley.edu"
# Package metadata
__title__ = "FEM"
__description__ = "Comprehensive Finite Element Method package"
__url__ = "https://github.com/berkeley/scicomp"
# Main exports
__all__ = [
    # Core classes
    'Node', 'Element', 'FiniteElementBase',
    'LinearBar1D', 'LinearTriangle2D', 'LinearQuadrilateral2D', 'LinearTetrahedron3D',
    'Mesh', 'MeshParameters', 'Geometry',
    'StructuredMeshGenerator', 'UnstructuredMeshGenerator',
    'GlobalAssembly',
    'StaticSolver', 'DynamicSolver', 'NonlinearSolver',
    'FEMPostProcessor',
    # Utilities
    'MaterialProperty', 'IsotropicMaterial', 'OrthotropicMaterial', 'ViscoelasticMaterial',
    'MaterialLibrary', 'create_custom_material', 'scale_material_properties',
    # Factory functions
    'create_element_factory',
    # Convenience functions
    'create_simple_truss_example',
    'create_beam_example',
    'run_modal_analysis_example'
]
def create_simple_truss_example():
    """
    Create a simple truss analysis example.
    Returns:
        Tuple of (mesh, materials, assembly) for immediate use
    """
    import numpy as np
    # Create simple 2-member truss
    mesh = Mesh()
    mesh.dimension = 2
    # Add nodes
    mesh.add_node(np.array([0.0, 0.0]))  # Support
    mesh.add_node(np.array([1.0, 1.0]))  # Top
    mesh.add_node(np.array([2.0, 0.0]))  # Support
    # Add elements
    mesh.add_element('bar1d', [0, 1], material_id=0, cross_section_area=0.01)
    mesh.add_element('bar1d', [1, 2], material_id=0, cross_section_area=0.01)
    # Material
    steel = MaterialLibrary.steel_mild()
    materials = {0: steel}
    # Assembly
    assembly = GlobalAssembly(mesh, materials)
    return mesh, materials, assembly
def create_beam_example(length=1.0, height=0.1, nx=20, ny=4):
    """
    Create a cantilever beam example.
    Parameters:
        length: Beam length
        height: Beam height
        nx: Number of elements in x-direction
        ny: Number of elements in y-direction
    Returns:
        Tuple of (mesh, materials, assembly)
    """
    # Create structured mesh
    mesh_generator = StructuredMeshGenerator()
    mesh = mesh_generator.generate_rectangle_mesh(length, height, nx, ny, 'quad2d')
    # Set element properties
    for element in mesh.elements.values():
        element.thickness = 0.01
        element.properties['plane_stress'] = True
    # Material
    aluminum = MaterialLibrary.aluminum_6061()
    materials = {0: aluminum}
    # Assembly
    assembly = GlobalAssembly(mesh, materials)
    return mesh, materials, assembly
def run_modal_analysis_example(num_modes=5):
    """
    Run a simple modal analysis example.
    Parameters:
        num_modes: Number of modes to extract
    Returns:
        Dictionary with modal results
    """
    # Create beam example
    mesh, materials, assembly = create_beam_example()
    # Apply boundary conditions (cantilever)
    boundary_conditions = {}
    for node_id, node in assembly.mesh.nodes.items():
        if abs(node.coordinates[0]) < 1e-10:  # Left end
            boundary_conditions[(node_id, 0)] = 0.0  # Fix x
            boundary_conditions[(node_id, 1)] = 0.0  # Fix y
    assembly.apply_boundary_conditions(boundary_conditions)
    # Assemble matrices
    assembly.assemble_global_stiffness()
    assembly.assemble_global_mass()
    # Modal analysis
    solver = DynamicSolver(assembly)
    eigenvalues, eigenvectors = solver.modal_analysis(num_modes=num_modes)
    # Convert to frequencies
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'frequencies': frequencies,
        'assembly': assembly
    }
# Package initialization
def _initialize_package():
    """Initialize package settings and configurations."""
    import warnings
    # Filter warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    # Set default numpy print options
    import numpy as np
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(f"FEM package v{__version__} initialized")
    print("Berkeley Scientific Computing Framework")
    print("Type help(FEM) for package documentation")
# Initialize when imported
_initialize_package()