#!/usr/bin/env python3
"""
Uncertainty Quantification for Spintronic Device Models

Comprehensive UQ including parametric uncertainty, model uncertainty,
and sensitivity analysis using various statistical methods.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import saltelli, morris, fast_sampler
from SALib.analyze import sobol, morris as morris_analyze, fast
from SALib.plotting import morris as morris_plot
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
import logging
import warnings
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

@dataclass
class UQConfig:
    """Configuration for uncertainty quantification"""
    n_samples: int = 1000
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    sensitivity_method: str = 'sobol'  # 'sobol', 'morris', 'fast'
    monte_carlo_method: str = 'standard'  # 'standard', 'latin_hypercube', 'quasi_random'
    random_state: int = 42
    parallel: bool = True
    verbose: bool = True

class UncertaintyAnalyzer:
    """Comprehensive uncertainty quantification for spintronic devices"""
    
    def __init__(self, config: Optional[UQConfig] = None):
        """
        Initialize uncertainty analyzer
        
        Args:
            config: UQ configuration
        """
        self.config = config or UQConfig()
        
        # Set random seed
        np.random.seed(self.config.random_state)
        
        # Storage for results
        self.parameter_distributions = {}
        self.sensitivity_results = {}
        self.uncertainty_results = {}
        self.model_predictions = {}
        
    def define_parameter_uncertainty(self, param_distributions: Dict[str, Any]):
        """
        Define uncertainty in model parameters
        
        Args:
            param_distributions: Dictionary mapping parameter names to scipy distributions
        """
        self.parameter_distributions = param_distributions
        
        logger.info(f"Defined uncertainty for {len(param_distributions)} parameters:")
        for name, dist in param_distributions.items():
            if hasattr(dist, 'mean') and hasattr(dist, 'std'):
                logger.info(f"  {name}: {dist.dist.name}(μ={dist.mean():.3f}, σ={dist.std():.3f})")
            else:
                logger.info(f"  {name}: {dist}")
    
    def monte_carlo_analysis(self, model_func: Callable, 
                           param_distributions: Optional[Dict[str, Any]] = None,
                           n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Monte Carlo uncertainty propagation
        
        Args:
            model_func: Function that takes parameters dict and returns model outputs
            param_distributions: Parameter uncertainty distributions
            n_samples: Number of MC samples
            
        Returns:
            Monte Carlo results with statistics
        """
        param_dists = param_distributions or self.parameter_distributions
        n_samples = n_samples or self.config.n_samples
        
        if not param_dists:
            raise ValueError("Parameter distributions must be defined")
        
        logger.info(f"Starting Monte Carlo analysis with {n_samples} samples")
        
        # Generate samples
        param_names = list(param_dists.keys())
        samples = self._generate_samples(param_dists, n_samples)
        
        # Evaluate model
        outputs = []
        failed_evaluations = 0
        
        for i in range(n_samples):
            param_dict = dict(zip(param_names, samples[i, :]))
            
            try:
                result = model_func(param_dict)
                if isinstance(result, dict):
                    outputs.append(result)
                else:
                    outputs.append({'output': result})
            except Exception as e:
                logger.warning(f"Model evaluation failed for sample {i}: {e}")
                failed_evaluations += 1
                outputs.append(None)
        
        # Remove failed evaluations
        valid_outputs = [out for out in outputs if out is not None]
        valid_samples = samples[:len(valid_outputs), :]
        
        if len(valid_outputs) == 0:
            raise ValueError("All model evaluations failed")
        
        logger.info(f"Completed {len(valid_outputs)} valid evaluations "
                   f"({failed_evaluations} failed)")
        
        # Extract output arrays
        output_names = list(valid_outputs[0].keys())
        output_arrays = {}
        
        for name in output_names:
            values = [out[name] for out in valid_outputs if name in out]
            output_arrays[name] = np.array(values)
        
        # Calculate statistics
        statistics = {}
        for name, values in output_arrays.items():
            if len(values) > 0:
                statistics[name] = self._calculate_statistics(values)
        
        # Store results
        mc_results = {
            'samples': valid_samples,
            'parameter_names': param_names,
            'outputs': output_arrays,
            'statistics': statistics,
            'n_valid_samples': len(valid_outputs),
            'n_failed_samples': failed_evaluations
        }
        
        self.uncertainty_results['monte_carlo'] = mc_results
        
        return mc_results
    
    def sensitivity_analysis(self, model_func: Callable, 
                           parameter_bounds: Dict[str, Tuple[float, float]],
                           method: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform global sensitivity analysis
        
        Args:
            model_func: Model function
            parameter_bounds: Parameter bounds {name: (min, max)}
            method: Sensitivity analysis method ('sobol', 'morris', 'fast')
            
        Returns:
            Sensitivity analysis results
        """
        method = method or self.config.sensitivity_method
        
        logger.info(f"Starting {method.upper()} sensitivity analysis")
        
        # Define problem for SALib
        problem = {
            'num_vars': len(parameter_bounds),
            'names': list(parameter_bounds.keys()),
            'bounds': list(parameter_bounds.values())
        }
        
        # Generate samples
        if method == 'sobol':
            # Sobol indices require (2*D+2)*N samples
            n_samples = min(self.config.n_samples // (2 * len(parameter_bounds) + 2), 100)
            param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
        elif method == 'morris':
            param_values = morris.sample(problem, self.config.n_samples // 10)
        elif method == 'fast':
            param_values = fast_sampler.sample(problem, self.config.n_samples // len(parameter_bounds))
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")
        
        # Evaluate model
        logger.info(f"Evaluating model at {len(param_values)} points")
        
        outputs = []
        for i, params in enumerate(param_values):
            param_dict = dict(zip(problem['names'], params))
            
            try:
                result = model_func(param_dict)
                if isinstance(result, dict):
                    # Take first output for sensitivity analysis
                    output_value = list(result.values())[0]
                else:
                    output_value = result
                outputs.append(output_value)
            except Exception as e:
                logger.warning(f"Model evaluation failed: {e}")
                outputs.append(0.0)  # Use 0 for failed evaluations
        
        outputs = np.array(outputs)
        
        # Perform analysis
        if method == 'sobol':
            sa_results = sobol.analyze(problem, outputs, calc_second_order=True)
            
            sensitivity_results = {
                'method': 'sobol',
                'first_order': dict(zip(problem['names'], sa_results['S1'])),
                'first_order_conf': dict(zip(problem['names'], sa_results['S1_conf'])),
                'total_order': dict(zip(problem['names'], sa_results['ST'])),
                'total_order_conf': dict(zip(problem['names'], sa_results['ST_conf'])),
                'second_order': sa_results['S2'] if 'S2' in sa_results else None
            }
            
        elif method == 'morris':
            sa_results = morris_analyze.analyze(problem, param_values, outputs)
            
            sensitivity_results = {
                'method': 'morris',
                'mu': dict(zip(problem['names'], sa_results['mu'])),
                'mu_star': dict(zip(problem['names'], sa_results['mu_star'])),
                'sigma': dict(zip(problem['names'], sa_results['sigma'])),
                'mu_star_conf': dict(zip(problem['names'], sa_results['mu_star_conf']))
            }
            
        elif method == 'fast':
            sa_results = fast.analyze(problem, outputs)
            
            sensitivity_results = {
                'method': 'fast',
                'S1': dict(zip(problem['names'], sa_results['S1'])),
                'S1_conf': dict(zip(problem['names'], sa_results['S1_conf'])),
                'ST': dict(zip(problem['names'], sa_results['ST'])),
                'ST_conf': dict(zip(problem['names'], sa_results['ST_conf']))
            }
        
        sensitivity_results.update({
            'problem': problem,
            'outputs': outputs,
            'param_values': param_values
        })
        
        self.sensitivity_results[method] = sensitivity_results
        
        logger.info(f"{method.upper()} sensitivity analysis completed")
        
        return sensitivity_results
    
    def bayesian_parameter_estimation(self, observed_data: np.ndarray,
                                    model_func: Callable,
                                    prior_distributions: Dict[str, Any],
                                    likelihood_std: float = 0.1) -> Dict[str, Any]:
        """
        Bayesian parameter estimation using MCMC
        
        Args:
            observed_data: Observed experimental data
            model_func: Forward model function
            prior_distributions: Prior parameter distributions
            likelihood_std: Standard deviation for likelihood function
            
        Returns:
            Posterior parameter distributions
        """
        logger.info("Starting Bayesian parameter estimation")
        
        param_names = list(prior_distributions.keys())
        n_params = len(param_names)
        
        def log_prior(params):
            """Log prior probability"""
            log_p = 0.0
            for i, (name, dist) in enumerate(prior_distributions.items()):
                log_p += dist.logpdf(params[i])
            return log_p
        
        def log_likelihood(params, observed):
            """Log likelihood function"""
            param_dict = dict(zip(param_names, params))
            
            try:
                predicted = model_func(param_dict)
                if isinstance(predicted, dict):
                    predicted = list(predicted.values())[0]
                
                # Gaussian likelihood
                residuals = observed - predicted
                log_l = -0.5 * np.sum(residuals**2) / (likelihood_std**2)
                log_l -= len(observed) * np.log(likelihood_std * np.sqrt(2*np.pi))
                
                return log_l
            except:
                return -np.inf
        
        def log_posterior(params, observed):
            """Log posterior probability"""
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params, observed)
        
        # MCMC sampling (simplified Metropolis-Hastings)
        n_samples = self.config.n_samples
        n_burn = n_samples // 4
        
        # Initialize chain
        chain = np.zeros((n_samples, n_params))
        
        # Start from prior means
        current_params = np.array([dist.mean() for dist in prior_distributions.values()])
        current_log_post = log_posterior(current_params, observed_data)
        
        # Proposal covariance
        prop_cov = np.eye(n_params) * 0.1
        
        accepted = 0
        
        logger.info(f"Running MCMC with {n_samples} samples")
        
        for i in range(n_samples):
            # Propose new parameters
            proposed_params = multivariate_normal.rvs(current_params, prop_cov)
            proposed_log_post = log_posterior(proposed_params, observed_data)
            
            # Accept/reject
            if np.log(np.random.random()) < (proposed_log_post - current_log_post):
                current_params = proposed_params
                current_log_post = proposed_log_post
                accepted += 1
            
            chain[i, :] = current_params
        
        # Remove burn-in
        chain = chain[n_burn:, :]
        acceptance_rate = accepted / n_samples
        
        logger.info(f"MCMC completed. Acceptance rate: {acceptance_rate:.3f}")
        
        # Calculate posterior statistics
        posterior_stats = {}
        for i, name in enumerate(param_names):
            samples = chain[:, i]
            posterior_stats[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'q025': np.percentile(samples, 2.5),
                'q975': np.percentile(samples, 97.5),
                'samples': samples
            }
        
        bayesian_results = {
            'chain': chain,
            'parameter_names': param_names,
            'posterior_stats': posterior_stats,
            'acceptance_rate': acceptance_rate,
            'n_effective_samples': len(chain),
            'log_likelihood_values': [log_likelihood(params, observed_data) 
                                    for params in chain[::10]]  # Subsample for efficiency
        }
        
        self.uncertainty_results['bayesian'] = bayesian_results
        
        return bayesian_results
    
    def model_uncertainty_analysis(self, models: List[Callable],
                                  test_inputs: np.ndarray,
                                  model_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze uncertainty due to model form/structure
        
        Args:
            models: List of different model functions
            test_inputs: Input points for evaluation
            model_weights: Weights for model averaging
            
        Returns:
            Model uncertainty analysis results
        """
        logger.info(f"Analyzing model uncertainty with {len(models)} models")
        
        if model_weights is None:
            model_weights = np.ones(len(models)) / len(models)
        
        # Evaluate all models
        model_predictions = []
        
        for i, model in enumerate(models):
            predictions = []
            
            for input_point in test_inputs:
                if isinstance(input_point, np.ndarray):
                    param_dict = {'input': input_point}
                else:
                    param_dict = input_point
                
                try:
                    pred = model(param_dict)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Model {i} failed: {e}")
                    predictions.append(0.0)
            
            model_predictions.append(np.array(predictions))
        
        model_predictions = np.array(model_predictions)  # Shape: (n_models, n_points)
        
        # Calculate model ensemble statistics
        weighted_mean = np.average(model_predictions, axis=0, weights=model_weights)
        model_variance = np.average((model_predictions - weighted_mean)**2, axis=0, weights=model_weights)
        model_std = np.sqrt(model_variance)
        
        # Model diversity metrics
        pairwise_correlations = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                corr = np.corrcoef(model_predictions[i], model_predictions[j])[0, 1]
                pairwise_correlations.append(corr)
        
        mean_correlation = np.mean(pairwise_correlations)
        
        # Individual model performance metrics (if reference available)
        model_stats = {}
        for i in range(len(models)):
            model_stats[f'model_{i}'] = {
                'mean_prediction': np.mean(model_predictions[i]),
                'std_prediction': np.std(model_predictions[i]),
                'min_prediction': np.min(model_predictions[i]),
                'max_prediction': np.max(model_predictions[i])
            }
        
        model_uncertainty_results = {
            'individual_predictions': model_predictions,
            'ensemble_mean': weighted_mean,
            'ensemble_std': model_std,
            'model_weights': model_weights,
            'pairwise_correlations': pairwise_correlations,
            'mean_correlation': mean_correlation,
            'model_statistics': model_stats,
            'relative_model_uncertainty': model_std / (np.abs(weighted_mean) + 1e-8)
        }
        
        self.uncertainty_results['model_uncertainty'] = model_uncertainty_results
        
        return model_uncertainty_results
    
    def uncertainty_propagation_analysis(self, model_chain: List[Callable],
                                       input_uncertainties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze uncertainty propagation through a chain of models
        
        Args:
            model_chain: List of models in sequence
            input_uncertainties: Input parameter uncertainties
            
        Returns:
            Uncertainty propagation results
        """
        logger.info(f"Analyzing uncertainty propagation through {len(model_chain)} model stages")
        
        # Generate input samples
        param_names = list(input_uncertainties.keys())
        input_samples = self._generate_samples(input_uncertainties, self.config.n_samples)
        
        # Propagate through model chain
        current_samples = input_samples
        stage_outputs = []
        
        for stage, model in enumerate(model_chain):
            logger.info(f"Propagating through stage {stage+1}")
            
            stage_predictions = []
            
            for sample in current_samples:
                if len(param_names) == len(sample):
                    param_dict = dict(zip(param_names, sample))
                else:
                    param_dict = {'input': sample}
                
                try:
                    pred = model(param_dict)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    stage_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Stage {stage} failed: {e}")
                    stage_predictions.append(0.0)
            
            current_samples = np.array(stage_predictions).reshape(-1, 1)
            stage_outputs.append(current_samples.copy())
        
        # Calculate statistics for each stage
        stage_statistics = {}
        for i, outputs in enumerate(stage_outputs):
            stage_statistics[f'stage_{i+1}'] = self._calculate_statistics(outputs.ravel())
        
        # Uncertainty propagation metrics
        input_std = np.std(input_samples, axis=0)
        output_std = np.std(stage_outputs[-1])
        
        uncertainty_amplification = output_std / np.mean(input_std)
        
        propagation_results = {
            'stage_outputs': stage_outputs,
            'stage_statistics': stage_statistics,
            'input_std': input_std,
            'output_std': output_std,
            'uncertainty_amplification': uncertainty_amplification,
            'input_samples': input_samples
        }
        
        return propagation_results
    
    def _generate_samples(self, distributions: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Generate samples from parameter distributions"""
        
        param_names = list(distributions.keys())
        samples = np.zeros((n_samples, len(param_names)))
        
        for i, (name, dist) in enumerate(distributions.items()):
            if hasattr(dist, 'rvs'):
                # scipy distribution
                samples[:, i] = dist.rvs(n_samples)
            elif isinstance(dist, tuple) and len(dist) == 2:
                # Uniform distribution from tuple (min, max)
                samples[:, i] = np.random.uniform(dist[0], dist[1], n_samples)
            else:
                raise ValueError(f"Unknown distribution type for parameter {name}")
        
        return samples
    
    def _calculate_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics for array of values"""
        
        values = values[np.isfinite(values)]  # Remove inf/nan
        
        if len(values) == 0:
            return {'error': 'No valid values'}
        
        stats_dict = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'var': float(np.var(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'q95': float(np.percentile(values, 95)),
            'q05': float(np.percentile(values, 5)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
            'count': len(values)
        }
        
        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(values, 100 * alpha / 2)
        ci_upper = np.percentile(values, 100 * (1 - alpha / 2))
        
        stats_dict['confidence_interval'] = [float(ci_lower), float(ci_upper)]
        stats_dict['confidence_level'] = self.config.confidence_level
        
        return stats_dict
    
    def plot_sensitivity_results(self, method: str = 'sobol', save_path: Optional[str] = None):
        """Plot sensitivity analysis results"""
        
        if method not in self.sensitivity_results:
            raise ValueError(f"No sensitivity results for method {method}")
        
        results = self.sensitivity_results[method]
        param_names = results['problem']['names']
        
        plt.figure(figsize=(12, 8))
        
        if method == 'sobol':
            # Sobol indices plot
            first_order = [results['first_order'][name] for name in param_names]
            total_order = [results['total_order'][name] for name in param_names]
            
            x = np.arange(len(param_names))
            width = 0.35
            
            plt.subplot(2, 1, 1)
            plt.bar(x - width/2, first_order, width, label='First-order', alpha=0.8)
            plt.bar(x + width/2, total_order, width, label='Total-order', alpha=0.8)
            plt.xlabel('Parameters')
            plt.ylabel('Sensitivity Index')
            plt.title('Sobol Sensitivity Indices')
            plt.xticks(x, param_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Second-order interactions (if available)
            if results['second_order'] is not None:
                plt.subplot(2, 1, 2)
                S2 = results['second_order']
                
                # Create heatmap of second-order indices
                im = plt.imshow(S2, cmap='viridis', aspect='auto')
                plt.colorbar(im)
                plt.title('Second-order Sensitivity Indices')
                plt.xlabel('Parameter j')
                plt.ylabel('Parameter i')
                
        elif method == 'morris':
            # Morris plot
            mu_star = [results['mu_star'][name] for name in param_names]
            sigma = [results['sigma'][name] for name in param_names]
            
            plt.scatter(sigma, mu_star, s=100, alpha=0.7)
            
            for i, name in enumerate(param_names):
                plt.annotate(name, (sigma[i], mu_star[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('σ (Standard Deviation)')
            plt.ylabel('μ* (Mean of Absolute Elementary Effects)')
            plt.title('Morris Sensitivity Analysis')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_uncertainty_results(self, result_type: str = 'monte_carlo', 
                               output_name: Optional[str] = None,
                               save_path: Optional[str] = None):
        """Plot uncertainty analysis results"""
        
        if result_type not in self.uncertainty_results:
            raise ValueError(f"No uncertainty results for {result_type}")
        
        results = self.uncertainty_results[result_type]
        
        plt.figure(figsize=(15, 10))
        
        if result_type == 'monte_carlo':
            outputs = results['outputs']
            
            if output_name is None:
                output_name = list(outputs.keys())[0]
            
            values = outputs[output_name]
            
            # Histogram
            plt.subplot(2, 3, 1)
            plt.hist(values, bins=50, alpha=0.7, density=True)
            plt.xlabel(output_name)
            plt.ylabel('Density')
            plt.title('Output Distribution')
            plt.grid(True, alpha=0.3)
            
            # Q-Q plot for normality check
            plt.subplot(2, 3, 2)
            stats.probplot(values, dist="norm", plot=plt)
            plt.title('Q-Q Plot (Normality Check)')
            
            # Box plot
            plt.subplot(2, 3, 3)
            plt.boxplot(values)
            plt.ylabel(output_name)
            plt.title('Box Plot')
            plt.grid(True, alpha=0.3)
            
            # Cumulative distribution
            plt.subplot(2, 3, 4)
            sorted_values = np.sort(values)
            p = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            plt.plot(sorted_values, p, 'b-', linewidth=2)
            plt.xlabel(output_name)
            plt.ylabel('Cumulative Probability')
            plt.title('CDF')
            plt.grid(True, alpha=0.3)
            
            # Parameter correlation matrix (if multiple inputs)
            if results['samples'].shape[1] > 1:
                plt.subplot(2, 3, 5)
                param_names = results['parameter_names']
                corr_matrix = np.corrcoef(results['samples'].T)
                
                im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im)
                plt.xticks(range(len(param_names)), param_names, rotation=45)
                plt.yticks(range(len(param_names)), param_names)
                plt.title('Parameter Correlation Matrix')
            
            # Statistics summary (text)
            plt.subplot(2, 3, 6)
            plt.axis('off')
            stats_text = self._format_statistics_text(results['statistics'][output_name])
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
            plt.title('Statistics Summary')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _format_statistics_text(self, stats: Dict[str, float]) -> str:
        """Format statistics dictionary as text"""
        
        text_lines = []
        text_lines.append(f"Mean:     {stats['mean']:.4f}")
        text_lines.append(f"Std:      {stats['std']:.4f}")
        text_lines.append(f"Median:   {stats['median']:.4f}")
        text_lines.append(f"Min:      {stats['min']:.4f}")
        text_lines.append(f"Max:      {stats['max']:.4f}")
        text_lines.append(f"Q25:      {stats['q25']:.4f}")
        text_lines.append(f"Q75:      {stats['q75']:.4f}")
        text_lines.append(f"Skew:     {stats['skewness']:.4f}")
        text_lines.append(f"Kurt:     {stats['kurtosis']:.4f}")
        
        ci = stats['confidence_interval']
        text_lines.append(f"CI({stats['confidence_level']*100:.0f}%): [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        return '\n'.join(text_lines)
    
    def generate_uncertainty_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive uncertainty analysis report"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SPINCIRC UNCERTAINTY QUANTIFICATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Parameter uncertainties
        if self.parameter_distributions:
            report_lines.append("PARAMETER UNCERTAINTIES:")
            report_lines.append("-" * 25)
            for name, dist in self.parameter_distributions.items():
                if hasattr(dist, 'mean') and hasattr(dist, 'std'):
                    report_lines.append(f"{name:20}: {dist.dist.name}(μ={dist.mean():.3f}, σ={dist.std():.3f})")
                else:
                    report_lines.append(f"{name:20}: {dist}")
            report_lines.append("")
        
        # Monte Carlo results
        if 'monte_carlo' in self.uncertainty_results:
            mc_results = self.uncertainty_results['monte_carlo']
            report_lines.append("MONTE CARLO ANALYSIS:")
            report_lines.append("-" * 22)
            report_lines.append(f"Valid samples: {mc_results['n_valid_samples']}")
            report_lines.append(f"Failed samples: {mc_results['n_failed_samples']}")
            report_lines.append("")
            
            for output_name, stats in mc_results['statistics'].items():
                report_lines.append(f"Output: {output_name}")
                report_lines.append(f"  Mean ± Std:    {stats['mean']:.4f} ± {stats['std']:.4f}")
                report_lines.append(f"  Median:        {stats['median']:.4f}")
                report_lines.append(f"  Range:         [{stats['min']:.4f}, {stats['max']:.4f}]")
                ci = stats['confidence_interval']
                report_lines.append(f"  {stats['confidence_level']*100:.0f}% CI:        [{ci[0]:.4f}, {ci[1]:.4f}]")
                report_lines.append("")
        
        # Sensitivity analysis results
        for method, sa_results in self.sensitivity_results.items():
            report_lines.append(f"{method.upper()} SENSITIVITY ANALYSIS:")
            report_lines.append("-" * (len(method) + 22))
            
            if method == 'sobol':
                report_lines.append(f"{'Parameter':15} {'First-order':12} {'Total-order':12}")
                report_lines.append("-" * 40)
                for name in sa_results['first_order']:
                    s1 = sa_results['first_order'][name]
                    st = sa_results['total_order'][name]
                    report_lines.append(f"{name:15} {s1:12.4f} {st:12.4f}")
                    
            elif method == 'morris':
                report_lines.append(f"{'Parameter':15} {'μ*':12} {'σ':12}")
                report_lines.append("-" * 40)
                for name in sa_results['mu_star']:
                    mu_star = sa_results['mu_star'][name]
                    sigma = sa_results['sigma'][name]
                    report_lines.append(f"{name:15} {mu_star:12.4f} {sigma:12.4f}")
            
            report_lines.append("")
        
        # Model uncertainty
        if 'model_uncertainty' in self.uncertainty_results:
            mu_results = self.uncertainty_results['model_uncertainty']
            report_lines.append("MODEL UNCERTAINTY ANALYSIS:")
            report_lines.append("-" * 28)
            report_lines.append(f"Mean correlation between models: {mu_results['mean_correlation']:.4f}")
            report_lines.append(f"Relative model uncertainty: {np.mean(mu_results['relative_model_uncertainty']):.4f}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Uncertainty report saved to {output_file}")
        
        return report_text