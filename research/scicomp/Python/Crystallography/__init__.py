"""
Crystallography Module
Professional crystallographic analysis and structure determination tools.
Includes crystal structure analysis, diffraction pattern simulation,
space group operations, and structural refinement methods.
Modules:
    core: Core crystallographic algorithms and utilities
    examples: Educational examples at multiple difficulty levels
    tests: Comprehensive test suites
    benchmarks: Performance analysis tools
"""
from .core.crystal_structure import CrystalStructure, LatticeParameters
from .core.space_groups import SpaceGroup, SymmetryOperation
from .core.diffraction import DiffractionPattern, StructureFactor
from .core.structure_refinement import RietveldRefinement, LeastSquaresRefinement
__all__ = [
    'CrystalStructure',
    'LatticeParameters',
    'SpaceGroup',
    'SymmetryOperation',
    'DiffractionPattern',
    'StructureFactor',
    'RietveldRefinement',
    'LeastSquaresRefinement'
]
__version__ = "1.0.0"
__author__ = "SciComp Development Team"