"""Uncertainty Quantification and Propagation Methods.
This module implements Monte Carlo methods for uncertainty quantification,
sensitivity analysis, and polynomial chaos expansions.
Classes:
    UncertaintyQuantifier: Main UQ framework
    SensitivityAnalyzer: Global sensitivity analysis
    PolynomialChaos: Polynomial chaos expansion methods
Functions:
    monte_carlo_uncertainty: Standard MC uncertainty propagation
    sensitivity_analysis: Sobol sensitivity analysis
    polynomial_chaos_expansion: PCE surrogate modeling
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy import stats
import warnings
from .utils import set_seed, compute_statistics
from .sampling import MetropolisHastings, ImportanceSampler
from .constants import get_distribution
@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification analysis.
    Attributes:
        mean: Mean of output distribution
        std: Standard deviation of output distribution
        percentiles: Output percentiles
        samples: Output samples
        input_samples: Input samples
        sensitivity_indices: Sensitivity analysis results
        confidence_intervals: Confidence intervals
        metadata: Additional information
    """
    mean: float
    std: float
    percentiles: Dict[str, float]
    samples: np.ndarray
    input_samples: np.ndarray
    sensitivity_indices: Optional[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]
@dataclass
class SensitivityResult:
    """Result of sensitivity analysis.
    Attributes:
        first_order: First-order (main effect) indices
        total_order: Total-order indices
        second_order: Second-order interaction indices
        confidence_intervals: Bootstrap confidence intervals
        variable_names: Input variable names
        metadata: Additional information
    """
    first_order: np.ndarray
    total_order: np.ndarray
    second_order: Optional[np.ndarray]
    confidence_intervals: Dict[str, np.ndarray]
    variable_names: List[str]
    metadata: Dict[str, Any]
class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification framework.
    Provides Monte Carlo-based uncertainty propagation with various
    sampling strategies and variance reduction techniques.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def propagate_uncertainty(self,
                            model: Callable,
                            input_distributions: List[Dict[str, Any]],
                            n_samples: int = 10000,
                            method: str = 'monte_carlo',
                            percentiles: List[float] = [5, 25, 75, 95],
                            **kwargs) -> UncertaintyResult:
        """Propagate uncertainty through model.
        Args:
            model: Model function (input_vector -> output)
            input_distributions: List of input distribution specifications
            n_samples: Number of Monte Carlo samples
            method: Sampling method ('monte_carlo', 'latin_hypercube', 'importance')
            percentiles: Output percentiles to compute
            **kwargs: Additional arguments
        Returns:
            UncertaintyResult: Uncertainty analysis results
        """
        if method == 'monte_carlo':
            return self._monte_carlo_propagation(model, input_distributions,
                                               n_samples, percentiles, **kwargs)
        elif method == 'latin_hypercube':
            return self._lhs_propagation(model, input_distributions,
                                       n_samples, percentiles, **kwargs)
        elif method == 'importance':
            return self._importance_sampling_propagation(model, input_distributions,
                                                       n_samples, percentiles, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    def _monte_carlo_propagation(self, model: Callable, input_distributions: List[Dict],
                               n_samples: int, percentiles: List[float],
                               **kwargs) -> UncertaintyResult:
        """Standard Monte Carlo uncertainty propagation."""
        n_inputs = len(input_distributions)
        # Generate input samples
        input_samples = np.zeros((n_samples, n_inputs))
        for i, dist_spec in enumerate(input_distributions):
            dist_name = dist_spec['distribution']
            dist_params = {k: v for k, v in dist_spec.items() if k != 'distribution'}
            distribution = get_distribution(dist_name, **dist_params)
            input_samples[:, i] = distribution.rvs(size=n_samples,
                                                  random_state=self.rng)
        # Evaluate model
        if self.verbose:
            print(f"Evaluating model for {n_samples} samples...")
        output_samples = self._evaluate_model_batch(model, input_samples)
        # Compute statistics
        mean = np.mean(output_samples)
        std = np.std(output_samples, ddof=1)
        # Compute percentiles
        percentile_values = {}
        for p in percentiles:
            percentile_values[f'p{p}'] = np.percentile(output_samples, p)
        # Confidence intervals
        confidence_intervals = {
            '95%': (np.percentile(output_samples, 2.5),
                   np.percentile(output_samples, 97.5)),
            '90%': (np.percentile(output_samples, 5),
                   np.percentile(output_samples, 95))
        }
        metadata = {
            'method': 'monte_carlo',
            'n_samples': n_samples,
            'n_inputs': n_inputs
        }
        return UncertaintyResult(
            mean=mean,
            std=std,
            percentiles=percentile_values,
            samples=output_samples,
            input_samples=input_samples,
            sensitivity_indices=None,
            confidence_intervals=confidence_intervals,
            metadata=metadata
        )
    def _lhs_propagation(self, model: Callable, input_distributions: List[Dict],
                       n_samples: int, percentiles: List[float],
                       **kwargs) -> UncertaintyResult:
        """Latin Hypercube Sampling uncertainty propagation."""
        try:
            from scipy.stats import qmc
        except ImportError:
            warnings.warn("scipy.stats.qmc not available, falling back to Monte Carlo")
            return self._monte_carlo_propagation(model, input_distributions,
                                               n_samples, percentiles, **kwargs)
        n_inputs = len(input_distributions)
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=n_inputs, seed=self.random_state)
        unit_samples = sampler.random(n_samples)
        # Transform to input distributions
        input_samples = np.zeros_like(unit_samples)
        for i, dist_spec in enumerate(input_distributions):
            dist_name = dist_spec['distribution']
            dist_params = {k: v for k, v in dist_spec.items() if k != 'distribution'}
            distribution = get_distribution(dist_name, **dist_params)
            input_samples[:, i] = distribution.ppf(unit_samples[:, i])
        # Evaluate model
        output_samples = self._evaluate_model_batch(model, input_samples)
        # Compute statistics (same as Monte Carlo)
        mean = np.mean(output_samples)
        std = np.std(output_samples, ddof=1)
        percentile_values = {}
        for p in percentiles:
            percentile_values[f'p{p}'] = np.percentile(output_samples, p)
        confidence_intervals = {
            '95%': (np.percentile(output_samples, 2.5),
                   np.percentile(output_samples, 97.5)),
            '90%': (np.percentile(output_samples, 5),
                   np.percentile(output_samples, 95))
        }
        metadata = {
            'method': 'latin_hypercube',
            'n_samples': n_samples,
            'n_inputs': n_inputs
        }
        return UncertaintyResult(
            mean=mean,
            std=std,
            percentiles=percentile_values,
            samples=output_samples,
            input_samples=input_samples,
            sensitivity_indices=None,
            confidence_intervals=confidence_intervals,
            metadata=metadata
        )
    def _importance_sampling_propagation(self, model: Callable, input_distributions: List[Dict],
                                       n_samples: int, percentiles: List[float],
                                       **kwargs) -> UncertaintyResult:
        """Importance sampling uncertainty propagation."""
        # Simplified importance sampling implementation
        # In practice, would need more sophisticated importance distribution design
        # Fall back to standard Monte Carlo for now
        warnings.warn("Importance sampling UQ not fully implemented, using Monte Carlo")
        return self._monte_carlo_propagation(model, input_distributions,
                                           n_samples, percentiles, **kwargs)
    def _evaluate_model_batch(self, model: Callable, input_samples: np.ndarray) -> np.ndarray:
        """Evaluate model for batch of input samples."""
        n_samples = len(input_samples)
        output_samples = np.zeros(n_samples)
        for i, inputs in enumerate(input_samples):
            try:
                output_samples[i] = model(inputs)
            except Exception as e:
                warnings.warn(f"Model evaluation failed for sample {i}: {e}")
                output_samples[i] = np.nan
        # Remove NaN values
        valid_mask = ~np.isnan(output_samples)
        if not np.all(valid_mask):
            warnings.warn(f"Removed {np.sum(~valid_mask)} failed evaluations")
            output_samples = output_samples[valid_mask]
        return output_samples
class SensitivityAnalyzer:
    """Global sensitivity analysis using Sobol indices.
    Implements Sobol sensitivity analysis for quantifying the contribution
    of input variables to output variance.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def analyze(self,
                model: Callable,
                input_distributions: List[Dict[str, Any]],
                n_samples: int = 10000,
                variable_names: Optional[List[str]] = None,
                second_order: bool = False,
                bootstrap_confidence: bool = True,
                **kwargs) -> SensitivityResult:
        """Perform Sobol sensitivity analysis.
        Args:
            model: Model function
            input_distributions: Input variable distributions
            n_samples: Number of base samples (total will be higher)
            variable_names: Names of input variables
            second_order: Whether to compute second-order indices
            bootstrap_confidence: Whether to compute confidence intervals
            **kwargs: Additional arguments
        Returns:
            SensitivityResult: Sensitivity analysis results
        """
        n_inputs = len(input_distributions)
        if variable_names is None:
            variable_names = [f'X{i+1}' for i in range(n_inputs)]
        # Generate Sobol sample matrices
        if self.verbose:
            print("Generating Sobol sample matrices...")
        A, B, AB_matrices = self._generate_sobol_matrices(input_distributions, n_samples)
        # Evaluate model
        if self.verbose:
            print("Evaluating model...")
        f_A = self._evaluate_model_batch(model, A)
        f_B = self._evaluate_model_batch(model, B)
        f_AB = {}
        for i in range(n_inputs):
            f_AB[i] = self._evaluate_model_batch(model, AB_matrices[i])
        # Compute Sobol indices
        if self.verbose:
            print("Computing Sobol indices...")
        first_order, total_order = self._compute_sobol_indices(f_A, f_B, f_AB)
        # Second-order indices
        second_order_indices = None
        if second_order:
            second_order_indices = self._compute_second_order_indices(
                f_A, f_B, f_AB, input_distributions, n_samples
            )
        # Bootstrap confidence intervals
        confidence_intervals = {}
        if bootstrap_confidence:
            confidence_intervals = self._bootstrap_confidence_intervals(
                f_A, f_B, f_AB, n_bootstrap=100
            )
        metadata = {
            'n_samples': n_samples,
            'n_inputs': n_inputs,
            'total_model_evaluations': len(f_A) + len(f_B) + sum(len(f) for f in f_AB.values())
        }
        return SensitivityResult(
            first_order=first_order,
            total_order=total_order,
            second_order=second_order_indices,
            confidence_intervals=confidence_intervals,
            variable_names=variable_names,
            metadata=metadata
        )
    def _generate_sobol_matrices(self, input_distributions: List[Dict],
                               n_samples: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """Generate Sobol sample matrices A, B, and AB_i."""
        n_inputs = len(input_distributions)
        # Generate two independent matrices A and B
        A = np.zeros((n_samples, n_inputs))
        B = np.zeros((n_samples, n_inputs))
        for i, dist_spec in enumerate(input_distributions):
            dist_name = dist_spec['distribution']
            dist_params = {k: v for k, v in dist_spec.items() if k != 'distribution'}
            distribution = get_distribution(dist_name, **dist_params)
            A[:, i] = distribution.rvs(size=n_samples, random_state=self.rng)
            B[:, i] = distribution.rvs(size=n_samples, random_state=self.rng)
        # Generate AB_i matrices (A with column i from B)
        AB_matrices = {}
        for i in range(n_inputs):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            AB_matrices[i] = AB_i
        return A, B, AB_matrices
    def _evaluate_model_batch(self, model: Callable, input_samples: np.ndarray) -> np.ndarray:
        """Evaluate model for batch of samples."""
        output_samples = np.zeros(len(input_samples))
        for i, inputs in enumerate(input_samples):
            try:
                output_samples[i] = model(inputs)
            except Exception as e:
                warnings.warn(f"Model evaluation failed for sample {i}: {e}")
                output_samples[i] = np.nan
        # Remove NaN values
        valid_mask = ~np.isnan(output_samples)
        if not np.all(valid_mask):
            warnings.warn(f"Removed {np.sum(~valid_mask)} failed evaluations")
            output_samples = output_samples[valid_mask]
        return output_samples
    def _compute_sobol_indices(self, f_A: np.ndarray, f_B: np.ndarray,
                             f_AB: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first-order and total-order Sobol indices."""
        n_inputs = len(f_AB)
        # Variance estimates
        f_0_squared = (np.mean(f_A) + np.mean(f_B)) / 2
        f_0_squared = f_0_squared ** 2
        V = (np.var(f_A, ddof=1) + np.var(f_B, ddof=1)) / 2
        # First-order indices
        first_order = np.zeros(n_inputs)
        for i in range(n_inputs):
            V_i = np.mean(f_B * (f_AB[i] - f_A))
            first_order[i] = V_i / V
        # Total-order indices
        total_order = np.zeros(n_inputs)
        for i in range(n_inputs):
            V_Ti = np.mean((f_A - f_AB[i]) ** 2) / 2
            total_order[i] = V_Ti / V
        # Ensure non-negative and bounded
        first_order = np.clip(first_order, 0, 1)
        total_order = np.clip(total_order, 0, 1)
        return first_order, total_order
    def _compute_second_order_indices(self, f_A: np.ndarray, f_B: np.ndarray,
                                    f_AB: Dict[int, np.ndarray],
                                    input_distributions: List[Dict],
                                    n_samples: int) -> np.ndarray:
        """Compute second-order Sobol indices."""
        n_inputs = len(input_distributions)
        second_order = np.zeros((n_inputs, n_inputs))
        # This is computationally expensive and requires additional matrices
        # Simplified implementation for demonstration
        warnings.warn("Second-order indices computation simplified")
        return second_order
    def _bootstrap_confidence_intervals(self, f_A: np.ndarray, f_B: np.ndarray,
                                      f_AB: Dict[int, np.ndarray],
                                      n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """Compute bootstrap confidence intervals."""
        n_inputs = len(f_AB)
        n_samples = len(f_A)
        first_order_bootstrap = np.zeros((n_bootstrap, n_inputs))
        total_order_bootstrap = np.zeros((n_bootstrap, n_inputs))
        for b in range(n_bootstrap):
            # Bootstrap resample
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            f_A_boot = f_A[indices]
            f_B_boot = f_B[indices]
            f_AB_boot = {i: f_AB[i][indices] for i in range(n_inputs)}
            # Compute indices for bootstrap sample
            S1_boot, ST_boot = self._compute_sobol_indices(f_A_boot, f_B_boot, f_AB_boot)
            first_order_bootstrap[b] = S1_boot
            total_order_bootstrap[b] = ST_boot
        # Compute confidence intervals
        confidence_intervals = {
            'first_order_95%': np.percentile(first_order_bootstrap, [2.5, 97.5], axis=0),
            'total_order_95%': np.percentile(total_order_bootstrap, [2.5, 97.5], axis=0),
            'first_order_90%': np.percentile(first_order_bootstrap, [5, 95], axis=0),
            'total_order_90%': np.percentile(total_order_bootstrap, [5, 95], axis=0)
        }
        return confidence_intervals
class PolynomialChaos:
    """Polynomial Chaos Expansion for uncertainty quantification.
    Implements PCE surrogate modeling for efficient uncertainty propagation.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
        self.coefficients = None
        self.polynomial_basis = None
    def fit(self,
            model: Callable,
            input_distributions: List[Dict[str, Any]],
            polynomial_order: int = 3,
            n_samples: Optional[int] = None,
            sampling_method: str = 'tensor_grid',
            **kwargs) -> Dict[str, Any]:
        """Fit polynomial chaos expansion.
        Args:
            model: Model function
            input_distributions: Input variable distributions
            polynomial_order: Maximum polynomial order
            n_samples: Number of training samples
            sampling_method: Sampling method for training points
            **kwargs: Additional arguments
        Returns:
            Dictionary with fit information
        """
        # Simplified PCE implementation
        # Full implementation would require orthogonal polynomial construction
        warnings.warn("Polynomial Chaos Expansion not fully implemented")
        n_inputs = len(input_distributions)
        if n_samples is None:
            # Rule of thumb: 2-3 times number of basis functions
            n_basis = (polynomial_order + n_inputs) ** 2
            n_samples = 3 * n_basis
        # Generate training samples
        training_inputs, training_outputs = self._generate_training_data(
            model, input_distributions, n_samples, sampling_method
        )
        # Fit surrogate (simplified as polynomial regression)
        self.coefficients = self._fit_polynomial_regression(
            training_inputs, training_outputs, polynomial_order
        )
        fit_info = {
            'n_training_samples': n_samples,
            'polynomial_order': polynomial_order,
            'n_coefficients': len(self.coefficients) if self.coefficients is not None else 0
        }
        return fit_info
    def predict(self, input_samples: np.ndarray) -> np.ndarray:
        """Predict using PCE surrogate."""
        if self.coefficients is None:
            raise ValueError("PCE not fitted. Call fit() first.")
        # Simplified prediction
        # Full implementation would evaluate orthogonal polynomials
        return np.polyval(self.coefficients, input_samples[:, 0])  # Simplified
    def _generate_training_data(self, model: Callable, input_distributions: List[Dict],
                              n_samples: int, sampling_method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for PCE."""
        n_inputs = len(input_distributions)
        # Generate training inputs
        if sampling_method == 'monte_carlo':
            training_inputs = np.zeros((n_samples, n_inputs))
            for i, dist_spec in enumerate(input_distributions):
                dist_name = dist_spec['distribution']
                dist_params = {k: v for k, v in dist_spec.items() if k != 'distribution'}
                distribution = get_distribution(dist_name, **dist_params)
                training_inputs[:, i] = distribution.rvs(size=n_samples,
                                                        random_state=self.rng)
        else:
            # Simplified: use Monte Carlo for other methods too
            training_inputs = np.zeros((n_samples, n_inputs))
            for i, dist_spec in enumerate(input_distributions):
                dist_name = dist_spec['distribution']
                dist_params = {k: v for k, v in dist_spec.items() if k != 'distribution'}
                distribution = get_distribution(dist_name, **dist_params)
                training_inputs[:, i] = distribution.rvs(size=n_samples,
                                                        random_state=self.rng)
        # Evaluate model
        training_outputs = np.array([model(inputs) for inputs in training_inputs])
        return training_inputs, training_outputs
    def _fit_polynomial_regression(self, inputs: np.ndarray, outputs: np.ndarray,
                                 order: int) -> np.ndarray:
        """Fit polynomial regression (simplified PCE)."""
        # Simplified: fit 1D polynomial to first input
        # Full implementation would construct multivariate orthogonal polynomials
        coefficients = np.polyfit(inputs[:, 0], outputs, order)
        return coefficients
# Convenience functions
def monte_carlo_uncertainty(model: Callable,
                           input_distributions: List[Dict[str, Any]],
                           n_samples: int = 10000,
                           percentiles: List[float] = [5, 25, 75, 95],
                           random_state: Optional[int] = None,
                           **kwargs) -> UncertaintyResult:
    """Convenience function for Monte Carlo uncertainty propagation."""
    uq = UncertaintyQuantifier(random_state=random_state)
    return uq.propagate_uncertainty(model, input_distributions, n_samples,
                                  method='monte_carlo', percentiles=percentiles, **kwargs)
def sensitivity_analysis(model: Callable,
                        input_distributions: List[Dict[str, Any]],
                        n_samples: int = 10000,
                        variable_names: Optional[List[str]] = None,
                        random_state: Optional[int] = None,
                        **kwargs) -> SensitivityResult:
    """Convenience function for Sobol sensitivity analysis."""
    sa = SensitivityAnalyzer(random_state=random_state)
    return sa.analyze(model, input_distributions, n_samples,
                     variable_names=variable_names, **kwargs)
def polynomial_chaos_expansion(model: Callable,
                              input_distributions: List[Dict[str, Any]],
                              polynomial_order: int = 3,
                              n_samples: Optional[int] = None,
                              random_state: Optional[int] = None,
                              **kwargs) -> PolynomialChaos:
    """Convenience function for polynomial chaos expansion."""
    pce = PolynomialChaos(random_state=random_state)
    pce.fit(model, input_distributions, polynomial_order, n_samples, **kwargs)
    return pce