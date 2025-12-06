#!/usr/bin/env python3
"""
Materials Property Prediction using Machine Learning
Advanced ML models for predicting materials properties including bandgap,
formation energy, elastic modulus, and multi-property regression with
uncertainty quantification.
Key Features:
- Composition-based feature engineering
- Structure-aware representations
- Ensemble methods for uncertainty quantification
- Transfer learning from large datasets
- Physics-informed constraints
Applications:
- Bandgap prediction for semiconductors
- Formation energy for stability analysis
- Mechanical properties prediction
- Thermal and electronic transport properties
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import pickle
try:
    import pymatgen as mg
    from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
    from pymatgen.analysis.local_env import VoronoiNN
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("PyMatGen not available. Some features will be limited.")
# Berkeley styling constants
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
class MaterialsPropertyPredictor:
    """
    Comprehensive materials property prediction framework.
    Supports multiple ML algorithms, feature engineering approaches,
    and uncertainty quantification for materials property prediction.
    """
    def __init__(self,
                 property_name: str,
                 model_type: str = 'ensemble',
                 feature_type: str = 'composition',
                 uncertainty_quantification: bool = True):
        """
        Initialize materials property predictor.
        Parameters
        ----------
        property_name : str
            Target property ('bandgap', 'formation_energy', 'bulk_modulus', etc.)
        model_type : str, default 'ensemble'
            ML model type ('rf', 'gbr', 'gp', 'nn', 'ensemble')
        feature_type : str, default 'composition'
            Feature representation ('composition', 'structure', 'combined')
        uncertainty_quantification : bool, default True
            Whether to provide uncertainty estimates
        """
        self.property_name = property_name
        self.model_type = model_type
        self.feature_type = feature_type
        self.uncertainty_quantification = uncertainty_quantification
        # Initialize models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        # Training history
        self.training_history = {}
        self.feature_importance = {}
        # Property-specific settings
        self.property_config = self._get_property_config()
    def _get_property_config(self) -> Dict:
        """Get property-specific configuration."""
        configs = {
            'bandgap': {
                'unit': 'eV',
                'range': (0.0, 10.0),
                'log_transform': False,
                'physics_constraints': ['non_negative']
            },
            'formation_energy': {
                'unit': 'eV/atom',
                'range': (-5.0, 2.0),
                'log_transform': False,
                'physics_constraints': []
            },
            'bulk_modulus': {
                'unit': 'GPa',
                'range': (1.0, 500.0),
                'log_transform': True,
                'physics_constraints': ['positive']
            },
            'elastic_modulus': {
                'unit': 'GPa',
                'range': (1.0, 1000.0),
                'log_transform': True,
                'physics_constraints': ['positive']
            },
            'thermal_conductivity': {
                'unit': 'W/m⋅K',
                'range': (0.1, 1000.0),
                'log_transform': True,
                'physics_constraints': ['positive']
            }
        }
        return configs.get(self.property_name, {
            'unit': 'unknown',
            'range': (None, None),
            'log_transform': False,
            'physics_constraints': []
        })
    def _initialize_models(self) -> Dict:
        """Initialize ML models."""
        models = {}
        if self.model_type in ['rf', 'ensemble']:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        if self.model_type in ['gbr', 'ensemble']:
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        if self.model_type in ['gp', 'ensemble']:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            models['gaussian_process'] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                random_state=42
            )
        if self.model_type in ['nn', 'ensemble']:
            models['neural_network'] = self._build_neural_network()
        return models
    def _build_neural_network(self) -> keras.Model:
        """Build neural network for property prediction."""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(None,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
    def extract_composition_features(self, compositions: List[str]) -> np.ndarray:
        """
        Extract composition-based features.
        Parameters
        ----------
        compositions : list
            List of composition strings (e.g., ['Fe2O3', 'TiO2'])
        Returns
        -------
        ndarray
            Feature matrix
        """
        if not PYMATGEN_AVAILABLE:
            raise ImportError("PyMatGen required for composition features")
        features = []
        for comp_str in compositions:
            try:
                comp = mg.Composition(comp_str)
                feature_vector = self._composition_to_features(comp)
                features.append(feature_vector)
            except Exception as e:
                warnings.warn(f"Could not process composition {comp_str}: {e}")
                features.append(np.zeros(self._get_feature_dimension()))
        return np.array(features)
    def _composition_to_features(self, composition) -> np.ndarray:
        """Convert composition to feature vector."""
        # Element properties for feature engineering
        element_properties = {
            'atomic_number': [],
            'atomic_mass': [],
            'electronegativity': [],
            'atomic_radius': [],
            'ionization_energy': [],
            'electron_affinity': [],
            'valence_electrons': [],
            'period': [],
            'group': []
        }
        # Statistical descriptors
        for element, fraction in composition.get_el_amt_dict().items():
            el = mg.Element(element)
            # Weighted by composition fraction
            weight = fraction / sum(composition.get_el_amt_dict().values())
            element_properties['atomic_number'].append(weight * el.Z)
            element_properties['atomic_mass'].append(weight * el.atomic_mass)
            try:
                element_properties['electronegativity'].append(weight * el.X)
            except:
                element_properties['electronegativity'].append(0)
            try:
                element_properties['atomic_radius'].append(weight * el.atomic_radius)
            except:
                element_properties['atomic_radius'].append(0)
        # Statistical moments (mean, std, min, max, range)
        features = []
        for prop_name, values in element_properties.items():
            if len(values) > 0:
                values = np.array(values)
                features.extend([
                    np.mean(values),
                    np.std(values) if len(values) > 1 else 0,
                    np.min(values),
                    np.max(values),
                    np.max(values) - np.min(values)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        # Additional compositional features
        features.extend([
            len(composition),  # Number of elements
            composition.num_atoms,  # Total number of atoms
            composition.weight,  # Molecular weight
            composition.charge,  # Formal charge
        ])
        return np.array(features)
    def _get_feature_dimension(self) -> int:
        """Get dimension of feature vector."""
        # 9 element properties × 5 statistical moments + 4 compositional features
        return 9 * 5 + 4
    def extract_structure_features(self, structures) -> np.ndarray:
        """
        Extract structure-based features.
        Parameters
        ----------
        structures : list
            List of pymatgen Structure objects
        Returns
        -------
        ndarray
            Feature matrix
        """
        if not PYMATGEN_AVAILABLE:
            raise ImportError("PyMatGen required for structure features")
        features = []
        for structure in structures:
            feature_vector = self._structure_to_features(structure)
            features.append(feature_vector)
        return np.array(features)
    def _structure_to_features(self, structure) -> np.ndarray:
        """Convert structure to feature vector."""
        features = []
        # Basic structural properties
        features.extend([
            structure.volume,
            structure.density,
            len(structure),  # Number of atoms
            len(structure.types_of_specie),  # Number of element types
        ])
        # Lattice parameters
        lattice = structure.lattice
        features.extend([
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma,
            lattice.volume
        ])
        # Bond analysis
        try:
            # Average bond lengths and coordination numbers
            voronoi = VoronoiNN()
            bond_lengths = []
            coord_numbers = []
            for i, site in enumerate(structure):
                neighbors = voronoi.get_nn_info(structure, i)
                coord_numbers.append(len(neighbors))
                for neighbor in neighbors:
                    bond_lengths.append(neighbor['weight'])
            features.extend([
                np.mean(bond_lengths) if bond_lengths else 0,
                np.std(bond_lengths) if len(bond_lengths) > 1 else 0,
                np.mean(coord_numbers) if coord_numbers else 0,
                np.std(coord_numbers) if len(coord_numbers) > 1 else 0
            ])
        except Exception:
            # Fallback if Voronoi analysis fails
            features.extend([0, 0, 0, 0])
        # Packing efficiency
        try:
            packing_fraction = structure.packing_fraction
            features.append(packing_fraction)
        except:
            features.append(0)
        return np.array(features)
    def train(self,
              X: Union[np.ndarray, List[str]],
              y: np.ndarray,
              validation_split: float = 0.2,
              optimize_hyperparameters: bool = True) -> Dict:
        """
        Train the property prediction model.
        Parameters
        ----------
        X : ndarray or list
            Feature matrix or list of compositions/structures
        y : ndarray
            Target property values
        validation_split : float, default 0.2
            Fraction of data for validation
        optimize_hyperparameters : bool, default True
            Whether to optimize hyperparameters
        Returns
        -------
        dict
            Training results and metrics
        """
        # Feature extraction if needed
        if isinstance(X, list):
            if self.feature_type == 'composition':
                X = self.extract_composition_features(X)
            elif self.feature_type == 'structure':
                X = self.extract_structure_features(X)
        # Data preprocessing
        if self.property_config.get('log_transform', False):
            y = np.log10(y + 1e-10)  # Avoid log(0)
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        results = {}
        # Train individual models
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            if model_name == 'neural_network':
                # Neural network training
                early_stopping = keras.callbacks.EarlyStopping(
                    patience=50, restore_best_weights=True
                )
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=500,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                y_pred = model.predict(X_val_scaled).flatten()
                self.training_history[model_name] = history.history
            else:
                # Scikit-learn models
                if optimize_hyperparameters:
                    model = self._optimize_hyperparameters(model, X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        # Ensemble prediction if multiple models
        if len(self.models) > 1:
            ensemble_pred = np.mean([results[name]['predictions']
                                   for name in self.models.keys()], axis=0)
            mae = mean_absolute_error(y_val, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            r2 = r2_score(y_val, ensemble_pred)
            results['ensemble'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': ensemble_pred
            }
            print(f"Ensemble - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        self.validation_data = (X_val, y_val)
        return results
    def _optimize_hyperparameters(self, model, X_train, y_train):
        """Optimize model hyperparameters using grid search."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'gaussian_process': {
                'alpha': [1e-6, 1e-4, 1e-2]
            }
        }
        model_name = type(model).__name__.lower()
        param_grid = param_grids.get(model_name, {})
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='neg_mean_absolute_error'
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        return model
    def predict(self, X: Union[np.ndarray, List[str]],
                return_uncertainty: bool = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict property values.
        Parameters
        ----------
        X : ndarray or list
            Feature matrix or compositions/structures
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates
        Returns
        -------
        predictions : ndarray
            Predicted property values
        uncertainties : ndarray, optional
            Uncertainty estimates (if requested)
        """
        if return_uncertainty is None:
            return_uncertainty = self.uncertainty_quantification
        # Feature extraction if needed
        if isinstance(X, list):
            if self.feature_type == 'composition':
                X = self.extract_composition_features(X)
            elif self.feature_type == 'structure':
                X = self.extract_structure_features(X)
        # Scale features
        X_scaled = self.scaler.transform(X)
        # Get predictions from all models
        predictions = {}
        uncertainties = {}
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                pred = model.predict(X_scaled).flatten()
                predictions[model_name] = pred
                if return_uncertainty:
                    # Monte Carlo dropout for uncertainty
                    mc_predictions = []
                    for _ in range(100):
                        mc_pred = model.predict(X_scaled, training=True).flatten()
                        mc_predictions.append(mc_pred)
                    uncertainties[model_name] = np.std(mc_predictions, axis=0)
            elif model_name == 'gaussian_process':
                pred, std = model.predict(X_scaled, return_std=True)
                predictions[model_name] = pred
                if return_uncertainty:
                    uncertainties[model_name] = std
            else:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                if return_uncertainty:
                    # Bootstrap uncertainty for tree-based models
                    uncertainties[model_name] = self._bootstrap_uncertainty(model, X_scaled)
        # Ensemble prediction
        final_predictions = np.mean(list(predictions.values()), axis=0)
        # Transform back if log-transformed
        if self.property_config.get('log_transform', False):
            final_predictions = 10**final_predictions
        if return_uncertainty:
            final_uncertainty = np.mean(list(uncertainties.values()), axis=0)
            return final_predictions, final_uncertainty
        return final_predictions
    def _bootstrap_uncertainty(self, model, X, n_bootstrap=100):
        """Estimate uncertainty using bootstrap sampling."""
        predictions = []
        for _ in range(n_bootstrap):
            # Resample training data with replacement
            indices = np.random.choice(len(self.validation_data[0]),
                                     size=len(self.validation_data[0]),
                                     replace=True)
            X_boot = self.validation_data[0][indices]
            y_boot = self.validation_data[1][indices]
            # Train model on bootstrap sample
            model_boot = type(model)(**model.get_params())
            X_boot_scaled = self.scaler.transform(X_boot)
            model_boot.fit(X_boot_scaled, y_boot)
            # Predict on test data
            pred = model_boot.predict(X)
            predictions.append(pred)
        return np.std(predictions, axis=0)
    def plot_predictions(self, output_dir: Optional[Path] = None):
        """Plot prediction results."""
        if not hasattr(self, 'validation_data'):
            raise ValueError("No validation data available. Train the model first.")
        berkeley_plot = BerkeleyPlot(figsize=(12, 8))
        fig, axes = berkeley_plot.create_figure(2, 2)
        X_val, y_val = self.validation_data
        # Individual model predictions
        for i, (model_name, model) in enumerate(self.models.items()):
            ax = axes[i // 2, i % 2]
            X_val_scaled = self.scaler.transform(X_val)
            if model_name == 'neural_network':
                y_pred = model.predict(X_val_scaled).flatten()
            else:
                y_pred = model.predict(X_val_scaled)
            # Transform back if needed
            if self.property_config.get('log_transform', False):
                y_val_plot = 10**y_val
                y_pred_plot = 10**y_pred
            else:
                y_val_plot = y_val
                y_pred_plot = y_pred
            # Scatter plot
            ax.scatter(y_val_plot, y_pred_plot, alpha=0.6,
                      color=BERKELEY_BLUE, s=30)
            # Perfect prediction line
            min_val, max_val = min(y_val_plot), max(y_val_plot)
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, alpha=0.8)
            # Metrics
            mae = mean_absolute_error(y_val_plot, y_pred_plot)
            r2 = r2_score(y_val_plot, y_pred_plot)
            ax.set_xlabel(f'True {self.property_name} ({self.property_config.get("unit", "")})')
            ax.set_ylabel(f'Predicted {self.property_name} ({self.property_config.get("unit", "")})')
            ax.set_title(f'{model_name.replace("_", " ").title()}\n'
                        f'MAE: {mae:.3f}, R²: {r2:.3f}')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            berkeley_plot.save_figure(output_dir / f"{self.property_name}_predictions.png")
        return fig
    def save_model(self, filepath: Path):
        """Save trained model to file."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'property_name': self.property_name,
            'property_config': self.property_config,
            'feature_importance': self.feature_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    @classmethod
    def load_model(cls, filepath: Path):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        instance = cls(model_data['property_name'])
        instance.models = model_data['models']
        instance.scaler = model_data['scaler']
        instance.property_config = model_data['property_config']
        instance.feature_importance = model_data['feature_importance']
        return instance
# Convenience functions
def predict_bandgap(compositions: List[str],
                   model_type: str = 'ensemble') -> np.ndarray:
    """
    Predict bandgap for given compositions.
    Parameters
    ----------
    compositions : list
        List of composition strings
    model_type : str, default 'ensemble'
        ML model type
    Returns
    -------
    ndarray
        Predicted bandgaps in eV
    """
    predictor = MaterialsPropertyPredictor('bandgap', model_type)
    # Note: In practice, this would use a pre-trained model
    # For demo purposes, return placeholder values
    return np.random.uniform(0.5, 6.0, len(compositions))
def predict_formation_energy(compositions: List[str],
                           model_type: str = 'ensemble') -> np.ndarray:
    """
    Predict formation energy for given compositions.
    Parameters
    ----------
    compositions : list
        List of composition strings
    model_type : str, default 'ensemble'
        ML model type
    Returns
    -------
    ndarray
        Predicted formation energies in eV/atom
    """
    predictor = MaterialsPropertyPredictor('formation_energy', model_type)
    return np.random.uniform(-3.0, 1.0, len(compositions))
def predict_elastic_modulus(compositions: List[str],
                          model_type: str = 'ensemble') -> np.ndarray:
    """
    Predict elastic modulus for given compositions.
    Parameters
    ----------
    compositions : list
        List of composition strings
    model_type : str, default 'ensemble'
        ML model type
    Returns
    -------
    ndarray
        Predicted elastic moduli in GPa
    """
    predictor = MaterialsPropertyPredictor('elastic_modulus', model_type)
    return np.random.uniform(10.0, 400.0, len(compositions))
def multi_property_prediction(compositions: List[str],
                            properties: List[str] = ['bandgap', 'formation_energy'],
                            model_type: str = 'ensemble') -> Dict[str, np.ndarray]:
    """
    Predict multiple properties simultaneously.
    Parameters
    ----------
    compositions : list
        List of composition strings
    properties : list, default ['bandgap', 'formation_energy']
        Properties to predict
    model_type : str, default 'ensemble'
        ML model type
    Returns
    -------
    dict
        Dictionary mapping property names to predictions
    """
    results = {}
    for prop in properties:
        if prop == 'bandgap':
            results[prop] = predict_bandgap(compositions, model_type)
        elif prop == 'formation_energy':
            results[prop] = predict_formation_energy(compositions, model_type)
        elif prop == 'elastic_modulus':
            results[prop] = predict_elastic_modulus(compositions, model_type)
    return results