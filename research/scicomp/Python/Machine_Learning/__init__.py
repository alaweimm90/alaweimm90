"""
Machine Learning for Scientific Computing
This package provides comprehensive machine learning tools specifically designed
for scientific computing applications, including physics-informed neural networks,
materials discovery, and scientific data analysis.
Modules:
    supervised: Supervised learning algorithms (regression, classification)
    unsupervised: Unsupervised learning algorithms (clustering, dimensionality reduction)
    neural_networks: Neural network architectures for scientific computing
    physics_informed: Physics-informed machine learning methods
    optimization: ML optimization algorithms and techniques
    utils: Utility functions for data processing and model evaluation
"""
from .supervised import *
from .unsupervised import *
from .neural_networks import *
from .physics_informed import *
from .optimization import *
from .utils import *
__version__ = "1.0.0"
__author__ = "Berkeley SciComp Team"
__all__ = [
    # Supervised learning
    'LinearRegression', 'PolynomialRegression', 'RidgeRegression',
    'LogisticRegression', 'SVM', 'RandomForest', 'GradientBoosting',
    # Unsupervised learning
    'KMeans', 'HierarchicalClustering', 'DBSCAN', 'PCA', 'ICA',
    'tSNE', 'UMAP', 'GaussianMixture',
    # Neural networks
    'MLP', 'CNN', 'RNN', 'LSTM', 'Transformer', 'Autoencoder',
    'VAE', 'GAN',
    # Physics-informed ML
    'PINN', 'DeepONet', 'FNO', 'PhysicsConstrainedNN',
    'ConservationLawsNN', 'SymmetryAwareNN',
    # Optimization
    'GradientDescent', 'Adam', 'AdamW', 'LBFGS', 'GeneticAlgorithm',
    'ParticleSwarmOptimization', 'BayesianOptimization',
    # Utilities
    'DataProcessor', 'ModelEvaluator', 'CrossValidator',
    'FeatureSelector', 'Visualizer'
]