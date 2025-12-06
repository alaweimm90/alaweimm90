#!/usr/bin/env python3
"""
Surrogate Models for Fast Spintronic Device Simulation

Fast approximate models using various ML techniques to replace expensive
physics simulations for optimization and parameter studies.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import optuna
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
import logging
import pickle
import joblib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SurrogateConfig:
    """Configuration for surrogate models"""
    model_type: str = 'gaussian_process'  # 'rf', 'xgb', 'nn', 'gaussian_process', 'ensemble'
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    scale_features: bool = True
    scale_targets: bool = False
    hyperparameter_tuning: bool = True
    n_hyperopt_trials: int = 100
    ensemble_models: List[str] = field(default_factory=lambda: ['rf', 'xgb', 'gp'])
    verbose: bool = True

class SurrogateModelBuilder:
    """Build and manage surrogate models for spintronic devices"""
    
    def __init__(self, config: Optional[SurrogateConfig] = None):
        """
        Initialize surrogate model builder
        
        Args:
            config: Surrogate model configuration
        """
        self.config = config or SurrogateConfig()
        
        # Scalers
        self.feature_scaler = StandardScaler() if self.config.scale_features else None
        self.target_scaler = StandardScaler() if self.config.scale_targets else None
        
        # Models
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.target_names = []
        
    def add_training_data(self, features: np.ndarray, targets: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         target_names: Optional[List[str]] = None):
        """
        Add training data for surrogate models
        
        Args:
            features: Input features (N_samples x N_features)
            targets: Target outputs (N_samples x N_targets)
            feature_names: Names of input features
            target_names: Names of target outputs
        """
        # Store raw data
        self.X_raw = features.copy()
        self.y_raw = targets.copy()
        
        # Feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        # Target names
        if target_names:
            self.target_names = target_names
        else:
            if targets.ndim == 1:
                self.target_names = ['target']
                targets = targets.reshape(-1, 1)
            else:
                self.target_names = [f'target_{i}' for i in range(targets.shape[1])]
        
        # Scale features if requested
        if self.feature_scaler:
            features_scaled = self.feature_scaler.fit_transform(features)
        else:
            features_scaled = features
        
        # Scale targets if requested
        if self.target_scaler:
            targets_scaled = self.target_scaler.fit_transform(targets)
        else:
            targets_scaled = targets
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_scaled, targets_scaled,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        logger.info(f"Added training data: {features.shape[0]} samples, "
                   f"{features.shape[1]} features, {targets.shape[1]} targets")
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
    
    def build_random_forest(self, hyperopt: bool = None) -> RandomForestRegressor:
        """Build Random Forest surrogate model"""
        
        hyperopt = hyperopt if hyperopt is not None else self.config.hyperparameter_tuning
        
        if hyperopt:
            # Hyperparameter optimization with Optuna
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                    'random_state': self.config.random_state
                }
                
                model = RandomForestRegressor(**params)
                scores = cross_val_score(model, self.X_train, self.y_train.ravel(), 
                                       cv=self.config.cv_folds, scoring='r2')
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_hyperopt_trials, 
                          show_progress_bar=self.config.verbose)
            
            best_params = study.best_params
            
        else:
            # Default parameters
            best_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': self.config.random_state
            }
        
        # Train final model
        model = RandomForestRegressor(**best_params)
        model.fit(self.X_train, self.y_train.ravel())
        
        self.models['rf'] = model
        logger.info(f"Random Forest trained with params: {best_params}")
        
        return model
    
    def build_xgboost(self, hyperopt: bool = None) -> xgb.XGBRegressor:
        """Build XGBoost surrogate model"""
        
        hyperopt = hyperopt if hyperopt is not None else self.config.hyperparameter_tuning
        
        if hyperopt:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.config.random_state
                }
                
                model = xgb.XGBRegressor(**params)
                scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                       cv=self.config.cv_folds, scoring='r2')
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_hyperopt_trials,
                          show_progress_bar=self.config.verbose)
            
            best_params = study.best_params
            
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.config.random_state
            }
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train, self.y_train.ravel())
        
        self.models['xgb'] = model
        logger.info(f"XGBoost trained with params: {best_params}")
        
        return model
    
    def build_gaussian_process(self, hyperopt: bool = None) -> GaussianProcessRegressor:
        """Build Gaussian Process surrogate model"""
        
        hyperopt = hyperopt if hyperopt is not None else self.config.hyperparameter_tuning
        
        if hyperopt:
            # Try different kernels
            kernels = [
                ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3),
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
            ]
            
            best_score = -np.inf
            best_kernel = None
            
            for kernel in kernels:
                model = GaussianProcessRegressor(kernel=kernel, random_state=self.config.random_state)
                scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                       cv=min(3, self.config.cv_folds), scoring='r2')  # Reduce CV for GP
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_kernel = kernel
            
            kernel = best_kernel
            
        else:
            # Default kernel
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        
        model = GaussianProcessRegressor(kernel=kernel, random_state=self.config.random_state)
        model.fit(self.X_train, self.y_train.ravel())
        
        self.models['gp'] = model
        logger.info(f"Gaussian Process trained with kernel: {kernel}")
        
        return model
    
    def build_neural_network(self, hyperopt: bool = None) -> MLPRegressor:
        """Build Neural Network surrogate model"""
        
        hyperopt = hyperopt if hyperopt is not None else self.config.hyperparameter_tuning
        
        if hyperopt:
            def objective(trial):
                # Suggest architecture
                n_layers = trial.suggest_int('n_layers', 1, 3)
                hidden_layer_sizes = []
                for i in range(n_layers):
                    size = trial.suggest_int(f'layer_{i}_size', 10, 200)
                    hidden_layer_sizes.append(size)
                
                params = {
                    'hidden_layer_sizes': tuple(hidden_layer_sizes),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                    'max_iter': 1000,
                    'random_state': self.config.random_state
                }
                
                model = MLPRegressor(**params)
                scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                       cv=self.config.cv_folds, scoring='r2')
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_hyperopt_trials,
                          show_progress_bar=self.config.verbose)
            
            best_params = study.best_params
            
            # Reconstruct hidden_layer_sizes
            n_layers = best_params.pop('n_layers')
            hidden_layer_sizes = []
            for i in range(n_layers):
                hidden_layer_sizes.append(best_params.pop(f'layer_{i}_size'))
            best_params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
            
        else:
            best_params = {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': self.config.random_state
            }
        
        model = MLPRegressor(**best_params)
        model.fit(self.X_train, self.y_train.ravel())
        
        self.models['nn'] = model
        logger.info(f"Neural Network trained with params: {best_params}")
        
        return model
    
    def build_ensemble(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build ensemble surrogate model"""
        
        models_to_use = models or self.config.ensemble_models
        
        # Build individual models
        ensemble_models = {}
        for model_name in models_to_use:
            if model_name == 'rf':
                ensemble_models['rf'] = self.build_random_forest()
            elif model_name == 'xgb':
                ensemble_models['xgb'] = self.build_xgboost()
            elif model_name == 'gp':
                ensemble_models['gp'] = self.build_gaussian_process()
            elif model_name == 'nn':
                ensemble_models['nn'] = self.build_neural_network()
        
        # Calculate ensemble weights based on validation performance
        weights = {}
        for name, model in ensemble_models.items():
            scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                   cv=self.config.cv_folds, scoring='r2')
            weights[name] = max(np.mean(scores), 0.1)  # Minimum weight of 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: weight/total_weight for name, weight in weights.items()}
        
        # Create ensemble wrapper
        class EnsembleModel:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict(X) * self.weights[name]
                    predictions.append(pred)
                return np.sum(predictions, axis=0)
            
            def predict_with_uncertainty(self, X):
                """Get prediction with uncertainty estimate"""
                individual_preds = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    individual_preds.append(pred)
                
                individual_preds = np.array(individual_preds)
                ensemble_pred = np.average(individual_preds, axis=0, weights=list(self.weights.values()))
                uncertainty = np.std(individual_preds, axis=0)
                
                return ensemble_pred, uncertainty
        
        ensemble_model = EnsembleModel(ensemble_models, weights)
        self.models['ensemble'] = ensemble_model
        
        logger.info(f"Ensemble model built with weights: {weights}")
        
        return {
            'model': ensemble_model,
            'individual_models': ensemble_models,
            'weights': weights
        }
    
    def build_all_models(self) -> Dict[str, Any]:
        """Build all available surrogate models"""
        
        logger.info("Building all surrogate models...")
        
        results = {}
        
        # Individual models
        try:
            results['rf'] = self.build_random_forest()
        except Exception as e:
            logger.warning(f"Failed to build Random Forest: {e}")
        
        try:
            results['xgb'] = self.build_xgboost()
        except Exception as e:
            logger.warning(f"Failed to build XGBoost: {e}")
        
        try:
            results['gp'] = self.build_gaussian_process()
        except Exception as e:
            logger.warning(f"Failed to build Gaussian Process: {e}")
        
        try:
            results['nn'] = self.build_neural_network()
        except Exception as e:
            logger.warning(f"Failed to build Neural Network: {e}")
        
        # Ensemble
        try:
            available_models = [name for name in ['rf', 'xgb', 'gp', 'nn'] 
                               if name in results]
            if len(available_models) >= 2:
                ensemble_result = self.build_ensemble(available_models)
                results['ensemble'] = ensemble_result['model']
        except Exception as e:
            logger.warning(f"Failed to build ensemble: {e}")
        
        return results
    
    def evaluate_models(self, models: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate surrogate models on test set"""
        
        models_to_eval = models or self.models
        results = {}
        
        for name, model in models_to_eval.items():
            try:
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test)
                else:
                    continue
                
                # Handle scaling
                if self.target_scaler:
                    y_test_orig = self.target_scaler.inverse_transform(self.y_test)
                    y_pred_orig = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1))
                else:
                    y_test_orig = self.y_test.ravel()
                    y_pred_orig = y_pred.ravel()
                
                # Calculate metrics
                mse = mean_squared_error(y_test_orig, y_pred_orig)
                mae = mean_absolute_error(y_test_orig, y_pred_orig)
                r2 = r2_score(y_test_orig, y_pred_orig)
                
                # Relative errors
                relative_error = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-8)))
                max_error = np.max(np.abs(y_test_orig - y_pred_orig))
                
                results[name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'relative_error': float(relative_error),
                    'max_error': float(max_error),
                    'rmse': float(np.sqrt(mse))
                }
                
                logger.info(f"{name.upper()} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate model {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.model_scores = results
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x].get('r2', -np.inf))
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
        
        return results
    
    def predict(self, features: np.ndarray, model_name: Optional[str] = None,
               return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with surrogate model"""
        
        # Select model
        if model_name:
            model = self.models[model_name]
        else:
            model = self.best_model
            if model is None:
                raise ValueError("No trained model available")
        
        # Scale features
        if self.feature_scaler:
            features_scaled = self.feature_scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        if hasattr(model, 'predict_with_uncertainty') and return_uncertainty:
            y_pred, uncertainty = model.predict_with_uncertainty(features_scaled)
        else:
            y_pred = model.predict(features_scaled)
            uncertainty = None
        
        # Scale back targets
        if self.target_scaler:
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            if uncertainty is not None:
                # Scale uncertainty (approximate)
                uncertainty = uncertainty * self.target_scaler.scale_[0]
        
        if return_uncertainty and uncertainty is not None:
            return y_pred, uncertainty
        else:
            return y_pred
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance from tree-based models"""
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        else:
            logger.warning(f"Model {model_name} does not provide feature importance")
            return None
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save trained surrogate model"""
        
        model = self.models[model_name] if model_name else self.best_model
        if model is None:
            raise ValueError("No model to save")
        
        # Save model and metadata
        model_data = {
            'model': model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config,
            'model_scores': self.model_scores
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained surrogate model"""
        
        model_data = joblib.load(filepath)
        
        # Restore model components
        if 'model' in model_data:
            model_name = 'loaded_model'
            self.models[model_name] = model_data['model']
            self.best_model = model_data['model']
        
        self.feature_scaler = model_data.get('feature_scaler')
        self.target_scaler = model_data.get('target_scaler')
        self.feature_names = model_data.get('feature_names', [])
        self.target_names = model_data.get('target_names', [])
        self.model_scores = model_data.get('model_scores', {})
        
        if 'config' in model_data:
            self.config = model_data['config']
        
        logger.info(f"Model loaded from {filepath}")
    
    def cross_validate_all(self) -> Dict[str, Dict[str, float]]:
        """Cross-validate all models"""
        
        cv_results = {}
        
        for name, model in self.models.items():
            if name == 'ensemble':
                continue  # Skip ensemble for CV
            
            try:
                # Different scoring metrics
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                
                scores = {}
                for score in scoring:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                               cv=self.config.cv_folds, scoring=score)
                    scores[score] = {
                        'mean': float(np.mean(cv_scores)),
                        'std': float(np.std(cv_scores)),
                        'scores': cv_scores.tolist()
                    }
                
                cv_results[name] = scores
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {e}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results