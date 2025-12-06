#!/usr/bin/env python3
"""
Materials Machine Learning Module
Advanced ML techniques for materials science including property prediction,
crystal graph neural networks, inverse design, and high-throughput screening.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .property_prediction import *
from .crystal_graph_nn import *
from .inverse_design import *
from .high_throughput import *
__all__ = [
    # Property Prediction
    'MaterialsPropertyPredictor', 'predict_bandgap', 'predict_formation_energy',
    'predict_elastic_modulus', 'multi_property_prediction',
    # Crystal Graph Networks
    'CrystalGraphNN', 'build_crystal_graph', 'atom_features', 'bond_features',
    'graph_convolution', 'pooling_layer',
    # Inverse Design
    'MaterialsInverseDesign', 'optimize_composition', 'target_property_search',
    'generative_materials_model', 'compositional_optimization',
    # High-Throughput Screening
    'HighThroughputScreening', 'materials_database_search', 'screening_pipeline',
    'property_space_exploration', 'pareto_optimization'
]