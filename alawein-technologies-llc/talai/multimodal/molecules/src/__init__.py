"""
TalAI 3D Molecular Structure Analysis Module

Comprehensive system for analyzing and understanding molecular structures
from various file formats with property prediction and visualization.
"""

from .structure_parser import MolecularStructureParser
from .property_predictor import PropertyPredictor
from .binding_analyzer import BindingSiteAnalyzer
from .similarity_search import MolecularSimilarity
from .visualization import Molecular3DVisualizer
from .docking_interface import DockingSimulation
from .quantum_interface import QuantumChemistry
from .database_connector import ChemicalDatabaseConnector

__all__ = [
    'MolecularStructureParser',
    'PropertyPredictor',
    'BindingSiteAnalyzer',
    'MolecularSimilarity',
    'Molecular3DVisualizer',
    'DockingSimulation',
    'QuantumChemistry',
    'ChemicalDatabaseConnector'
]

__version__ = '1.0.0'