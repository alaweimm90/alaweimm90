#!/usr/bin/env python3
"""
Parameter Extraction using Machine Learning

Advanced parameter extraction from experimental data using various ML techniques
including Bayesian inference, neural networks, and genetic algorithms.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import scipy.optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import optuna
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for parameter extraction"""
    method: str = 'bayesian'  # 'bayesian', 'neural', 'genetic', 'ensemble'
    n_trials: int = 100
    cv_folds: int = 5
    random_state: int = 42
    verbose: bool = True
    uncertainty_estimation: bool = True
    parallel_jobs: int = -1

class ParameterExtractor:
    """Advanced parameter extraction from spintronic device measurements"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize parameter extractor
        
        Args:
            config: Extraction configuration options
        """
        self.config = config or ExtractionConfig()
        self.scaler = StandardScaler()
        self.models = {}
        self.parameter_bounds = {}
        self.feature_names = []
        self.parameter_names = []
        self.extraction_history = []
        
    def define_parameter_space(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Define the parameter space for extraction
        
        Args:
            bounds: Dictionary mapping parameter names to (min, max) bounds
        """
        self.parameter_bounds = bounds
        self.parameter_names = list(bounds.keys())
        
        logger.info(f"Defined parameter space with {len(bounds)} parameters:")
        for name, (min_val, max_val) in bounds.items():
            logger.info(f"  {name}: [{min_val}, {max_val}]")
    
    def add_training_data(self, features: np.ndarray, parameters: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> None:
        """
        Add training data for supervised learning approaches
        
        Args:
            features: Input features (N_samples x N_features)
            parameters: Target parameters (N_samples x N_parameters) 
            feature_names: Names of input features
        """
        self.X_train = self.scaler.fit_transform(features)
        self.y_train = parameters
        
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            
        logger.info(f"Added training data: {features.shape[0]} samples, "
                   f"{features.shape[1]} features, {parameters.shape[1]} parameters")
    
    def extract_from_iv_curve(self, voltage: np.ndarray, current: np.ndarray,
                             target_params: List[str]) -> Dict[str, float]:
        """
        Extract parameters from I-V characteristic curves
        
        Args:
            voltage: Voltage array
            current: Current array
            target_params: List of parameters to extract
            
        Returns:
            Dictionary of extracted parameter values with uncertainties
        """
        # Preprocess I-V data
        features = self._extract_iv_features(voltage, current)
        
        if self.config.method == 'bayesian':
            return self._bayesian_extraction(features, target_params)
        elif self.config.method == 'neural':
            return self._neural_extraction(features, target_params)
        elif self.config.method == 'genetic':
            return self._genetic_extraction(voltage, current, target_params)
        else:
            raise ValueError(f"Unknown extraction method: {self.config.method}")
    
    def extract_from_transport_data(self, transport_data: Dict[str, np.ndarray],
                                  target_params: List[str]) -> Dict[str, float]:
        """
        Extract parameters from transport measurement data
        
        Args:
            transport_data: Dictionary with measurement arrays
            target_params: Parameters to extract
            
        Returns:
            Extracted parameters with uncertainties
        """
        # Extract features from transport measurements
        features = self._extract_transport_features(transport_data)
        
        # Use ensemble method for transport data
        return self._ensemble_extraction(features, target_params)
    
    def extract_from_magnetization_dynamics(self, time: np.ndarray, 
                                          magnetization: np.ndarray,
                                          target_params: List[str]) -> Dict[str, float]:
        """
        Extract LLG parameters from magnetization dynamics
        
        Args:
            time: Time array
            magnetization: Magnetization trajectory (3 x N)
            target_params: LLG parameters to extract
            
        Returns:
            Extracted LLG parameters
        """
        # Extract dynamic features
        features = self._extract_dynamics_features(time, magnetization)
        
        # Use physics-informed approach for dynamics
        return self._physics_informed_extraction(time, magnetization, target_params)
    
    def _extract_iv_features(self, voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Extract features from I-V curves"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(current),
            np.std(current), 
            np.max(current),
            np.min(current)
        ])
        
        # Resistance calculations
        nonzero_mask = np.abs(voltage) > 1e-12
        if np.any(nonzero_mask):
            resistance = current[nonzero_mask] / voltage[nonzero_mask]
            features.extend([
                np.mean(resistance),
                np.std(resistance),
                np.max(resistance),
                np.min(resistance)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Derivative features
        if len(current) > 1:
            conductance = np.gradient(current, voltage)
            features.extend([
                np.mean(conductance),
                np.std(conductance),
                np.max(conductance)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Nonlinearity measures  
        if len(current) > 2:
            curvature = np.gradient(np.gradient(current, voltage), voltage)
            features.extend([
                np.mean(np.abs(curvature)),
                np.std(curvature)
            ])
        else:
            features.extend([0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _extract_transport_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features from transport measurements"""
        features = []
        
        for key, values in data.items():
            if isinstance(values, np.ndarray) and values.size > 0:
                # Statistical features
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.median(values),
                    np.max(values),
                    np.min(values)
                ])
                
                # Higher order moments
                if len(values) > 1:
                    features.extend([
                        scipy.stats.skew(values) if len(values) > 2 else 0,
                        scipy.stats.kurtosis(values) if len(values) > 3 else 0
                    ])
                else:
                    features.extend([0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _extract_dynamics_features(self, time: np.ndarray, 
                                 magnetization: np.ndarray) -> np.ndarray:
        """Extract features from magnetization dynamics"""
        features = []
        
        # For each component
        for i in range(3):
            m_comp = magnetization[i, :]
            
            # Statistical features
            features.extend([
                np.mean(m_comp),
                np.std(m_comp),
                np.max(m_comp),
                np.min(m_comp)
            ])
            
            # Frequency domain features
            if len(m_comp) > 4:
                fft = np.fft.fft(m_comp)
                power = np.abs(fft)**2
                
                # Dominant frequency
                freqs = np.fft.fftfreq(len(m_comp), time[1] - time[0])
                dominant_freq = freqs[np.argmax(power[1:len(power)//2]) + 1]
                features.append(dominant_freq)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs[:len(freqs)//2] * power[:len(power)//2]) / np.sum(power[:len(power)//2])
                features.append(spectral_centroid)
            else:
                features.extend([0, 0])
        
        # Magnitude conservation
        magnitude = np.sqrt(np.sum(magnetization**2, axis=0))
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(np.abs(magnitude - 1))  # Deviation from unit magnitude
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _bayesian_extraction(self, features: np.ndarray, 
                           target_params: List[str]) -> Dict[str, float]:
        """Bayesian parameter extraction using Gaussian Process"""
        if not hasattr(self, 'X_train'):
            raise ValueError("Training data required for Bayesian extraction")
        
        results = {}
        
        for i, param in enumerate(target_params):
            if param not in self.parameter_names:
                continue
                
            # Set up Gaussian Process
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=self.config.random_state)
            
            # Train on parameter i
            y_target = self.y_train[:, self.parameter_names.index(param)]
            gp.fit(self.X_train, y_target)
            
            # Predict
            pred_mean, pred_std = gp.predict(features, return_std=True)
            
            results[param] = {
                'value': float(pred_mean[0]),
                'uncertainty': float(pred_std[0]),
                'confidence_interval': [
                    float(pred_mean[0] - 1.96 * pred_std[0]),
                    float(pred_mean[0] + 1.96 * pred_std[0])
                ]
            }
        
        return results
    
    def _neural_extraction(self, features: np.ndarray, 
                         target_params: List[str]) -> Dict[str, float]:
        """Neural network parameter extraction"""
        if not hasattr(self, 'X_train'):
            raise ValueError("Training data required for neural extraction")
        
        results = {}
        
        # Use ensemble of neural networks for better uncertainty estimation
        from sklearn.neural_network import MLPRegressor
        
        for param in target_params:
            if param not in self.parameter_names:
                continue
                
            param_idx = self.parameter_names.index(param)
            y_target = self.y_train[:, param_idx]
            
            # Train ensemble of neural networks
            predictions = []
            for seed in range(10):  # 10-member ensemble
                mlp = MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    random_state=seed,
                    max_iter=1000
                )
                mlp.fit(self.X_train, y_target)
                pred = mlp.predict(features)[0]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            results[param] = {
                'value': float(np.mean(predictions)),
                'uncertainty': float(np.std(predictions)),
                'confidence_interval': [
                    float(np.percentile(predictions, 2.5)),
                    float(np.percentile(predictions, 97.5))
                ]
            }
        
        return results
    
    def _genetic_extraction(self, voltage: np.ndarray, current: np.ndarray,
                          target_params: List[str]) -> Dict[str, float]:
        """Genetic algorithm parameter extraction"""
        
        def objective_function(params):
            # This would typically call a forward model
            # For demonstration, we'll use a simple fitting approach
            param_dict = dict(zip(target_params, params))
            
            # Example: fit to I-V curve using physical model
            # model_current = physical_model(voltage, param_dict)
            # return np.mean((current - model_current)**2)
            
            # Placeholder - return random value for demo
            return np.random.random()
        
        # Use scipy's differential evolution
        bounds = [self.parameter_bounds[param] for param in target_params 
                 if param in self.parameter_bounds]
        
        result = scipy.optimize.differential_evolution(
            objective_function,
            bounds,
            seed=self.config.random_state,
            maxiter=self.config.n_trials
        )
        
        results = {}
        for i, param in enumerate(target_params):
            if param in self.parameter_bounds:
                results[param] = {
                    'value': float(result.x[i]),
                    'uncertainty': 0.0,  # GA doesn't provide uncertainty
                    'success': result.success
                }
        
        return results
    
    def _ensemble_extraction(self, features: np.ndarray,
                           target_params: List[str]) -> Dict[str, float]:
        """Ensemble method combining multiple ML approaches"""
        if not hasattr(self, 'X_train'):
            raise ValueError("Training data required for ensemble extraction")
        
        results = {}
        
        # Define ensemble of models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state),
            'gp': GaussianProcessRegressor(random_state=self.config.random_state)
        }
        
        for param in target_params:
            if param not in self.parameter_names:
                continue
                
            param_idx = self.parameter_names.index(param)
            y_target = self.y_train[:, param_idx]
            
            # Train all models and make predictions
            predictions = []
            weights = []
            
            for name, model in models.items():
                try:
                    # Cross-validation to get model weight
                    cv_scores = cross_val_score(model, self.X_train, y_target, 
                                               cv=self.config.cv_folds)
                    weight = np.mean(cv_scores)
                    weights.append(max(weight, 0.1))  # Minimum weight
                    
                    # Fit and predict
                    model.fit(self.X_train, y_target)
                    pred = model.predict(features)[0]
                    predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Model {name} failed for parameter {param}: {e}")
                    continue
            
            if predictions:
                predictions = np.array(predictions)
                weights = np.array(weights) / np.sum(weights)  # Normalize
                
                # Weighted ensemble prediction
                ensemble_pred = np.sum(predictions * weights)
                ensemble_std = np.sqrt(np.sum(weights * (predictions - ensemble_pred)**2))
                
                results[param] = {
                    'value': float(ensemble_pred),
                    'uncertainty': float(ensemble_std),
                    'individual_predictions': predictions.tolist(),
                    'model_weights': weights.tolist()
                }
        
        return results
    
    def _physics_informed_extraction(self, time: np.ndarray, magnetization: np.ndarray,
                                   target_params: List[str]) -> Dict[str, float]:
        """Physics-informed parameter extraction for LLG dynamics"""
        
        def llg_model(t, y, alpha, gamma, H_eff):
            """LLG equation right-hand side"""
            m = y.reshape(3, -1)
            
            # Cross products
            m_cross_H = np.cross(m.T, H_eff.T, axis=1).T
            m_cross_mH = np.cross(m.T, m_cross_H.T, axis=1).T
            
            # LLG equation
            dmdt = -gamma / (1 + alpha**2) * (m_cross_H + alpha * m_cross_mH)
            
            return dmdt.flatten()
        
        def objective(params):
            """Objective function for parameter fitting"""
            param_dict = dict(zip(target_params, params))
            
            # Extract parameters with defaults
            alpha = param_dict.get('alpha', 0.01)
            gamma = param_dict.get('gamma', 1.76e11)
            
            # Assume constant effective field (simplified)
            H_eff = np.array([[0], [0], [0.1]])  # 0.1 T along z
            
            try:
                # Solve LLG equation
                from scipy.integrate import odeint
                
                m0 = magnetization[:, 0].flatten()
                sol = odeint(llg_model, m0, time, args=(alpha, gamma, H_eff))
                
                # Reshape solution
                m_pred = sol.T.reshape(3, -1)
                
                # Calculate error
                error = np.mean((magnetization - m_pred)**2)
                return error
                
            except Exception:
                return 1e10  # Large penalty for failed integration
        
        # Parameter bounds
        bounds = []
        for param in target_params:
            if param in self.parameter_bounds:
                bounds.append(self.parameter_bounds[param])
            else:
                # Default bounds
                if param == 'alpha':
                    bounds.append((0.001, 0.1))
                elif param == 'gamma':
                    bounds.append((1e10, 3e11))
                else:
                    bounds.append((0, 1))
        
        # Optimize
        result = scipy.optimize.differential_evolution(
            objective,
            bounds,
            seed=self.config.random_state,
            maxiter=100
        )
        
        results = {}
        for i, param in enumerate(target_params):
            results[param] = {
                'value': float(result.x[i]),
                'uncertainty': 0.0,
                'fit_quality': float(result.fun),
                'success': result.success
            }
        
        return results
    
    def cross_validate_extraction(self, features: np.ndarray, parameters: np.ndarray,
                                target_params: List[str]) -> Dict[str, Dict[str, float]]:
        """Cross-validate parameter extraction performance"""
        from sklearn.model_selection import cross_val_predict, KFold
        
        results = {}
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(features)
        
        # Cross-validation
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                  random_state=self.config.random_state)
        
        for param in target_params:
            if param not in self.parameter_names:
                continue
                
            param_idx = self.parameter_names.index(param)
            y_target = parameters[:, param_idx]
            
            # Test different models
            models = {
                'rf': RandomForestRegressor(random_state=self.config.random_state),
                'gb': GradientBoostingRegressor(random_state=self.config.random_state)
            }
            
            param_results = {}
            
            for name, model in models.items():
                try:
                    # Cross-validated predictions
                    y_pred = cross_val_predict(model, X_scaled, y_target, cv=cv)
                    
                    # Calculate metrics
                    mse = np.mean((y_target - y_pred)**2)
                    mae = np.mean(np.abs(y_target - y_pred))
                    r2 = 1 - mse / np.var(y_target)
                    
                    param_results[name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'r2': float(r2)
                    }
                    
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {param} with {name}: {e}")
            
            results[param] = param_results
        
        return results
    
    def optimize_hyperparameters(self, features: np.ndarray, parameters: np.ndarray,
                                target_params: List[str]) -> Dict[str, Dict]:
        """Optimize ML model hyperparameters using Optuna"""
        
        def objective(trial, param_name, param_idx):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            
            # Create model
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.config.random_state
            )
            
            # Cross-validate
            X_scaled = self.scaler.fit_transform(features)
            y_target = parameters[:, param_idx]
            
            scores = cross_val_score(model, X_scaled, y_target, cv=3, scoring='r2')
            return np.mean(scores)
        
        optimized_params = {}
        
        for param in target_params:
            if param not in self.parameter_names:
                continue
                
            param_idx = self.parameter_names.index(param)
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, param, param_idx),
                n_trials=50,
                show_progress_bar=self.config.verbose
            )
            
            optimized_params[param] = study.best_params
            
        return optimized_params