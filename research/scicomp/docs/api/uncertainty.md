# uncertainty
**Module:** `Python/Monte_Carlo/uncertainty.py`
## Overview
Uncertainty Quantification and Propagation Methods.
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
## Functions
### `monte_carlo_uncertainty(model, input_distributions, n_samples, percentiles, random_state)`
Convenience function for Monte Carlo uncertainty propagation.
**Source:** [Line 596](Python/Monte_Carlo/uncertainty.py#L596)
### `sensitivity_analysis(model, input_distributions, n_samples, variable_names, random_state)`
Convenience function for Sobol sensitivity analysis.
**Source:** [Line 608](Python/Monte_Carlo/uncertainty.py#L608)
### `polynomial_chaos_expansion(model, input_distributions, polynomial_order, n_samples, random_state)`
Convenience function for polynomial chaos expansion.
**Source:** [Line 620](Python/Monte_Carlo/uncertainty.py#L620)
## Classes
### `UncertaintyResult`
Result of uncertainty quantification analysis.
Attributes:
mean: Mean of output distribution
std: Standard deviation of output distribution
percentiles: Output percentiles
samples: Output samples
input_samples: Input samples
sensitivity_indices: Sensitivity analysis results
confidence_intervals: Confidence intervals
metadata: Additional information
**Class Source:** [Line 32](Python/Monte_Carlo/uncertainty.py#L32)
### `SensitivityResult`
Result of sensitivity analysis.
Attributes:
first_order: First-order (main effect) indices
total_order: Total-order indices
second_order: Second-order interaction indices
confidence_intervals: Bootstrap confidence intervals
variable_names: Input variable names
metadata: Additional information
**Class Source:** [Line 56](Python/Monte_Carlo/uncertainty.py#L56)
### `UncertaintyQuantifier`
Comprehensive uncertainty quantification framework.
Provides Monte Carlo-based uncertainty propagation with various
sampling strategies and variance reduction techniques.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 82](Python/Monte_Carlo/uncertainty.py#L82)
##### `propagate_uncertainty(self, model, input_distributions, n_samples, method, percentiles)`
Propagate uncertainty through model.
Args:
model: Model function (input_vector -> output)
input_distributions: List of input distribution specifications
n_samples: Number of Monte Carlo samples
method: Sampling method ('monte_carlo', 'latin_hypercube', 'importance')
percentiles: Output percentiles to compute
**kwargs: Additional arguments
Returns:
UncertaintyResult: Uncertainty analysis results
**Source:** [Line 87](Python/Monte_Carlo/uncertainty.py#L87)
##### `_monte_carlo_propagation(self, model, input_distributions, n_samples, percentiles)`
Standard Monte Carlo uncertainty propagation.
**Source:** [Line 119](Python/Monte_Carlo/uncertainty.py#L119)
##### `_lhs_propagation(self, model, input_distributions, n_samples, percentiles)`
Latin Hypercube Sampling uncertainty propagation.
**Source:** [Line 176](Python/Monte_Carlo/uncertainty.py#L176)
##### `_importance_sampling_propagation(self, model, input_distributions, n_samples, percentiles)`
Importance sampling uncertainty propagation.
**Source:** [Line 238](Python/Monte_Carlo/uncertainty.py#L238)
##### `_evaluate_model_batch(self, model, input_samples)`
Evaluate model for batch of input samples.
**Source:** [Line 250](Python/Monte_Carlo/uncertainty.py#L250)
**Class Source:** [Line 75](Python/Monte_Carlo/uncertainty.py#L75)
### `SensitivityAnalyzer`
Global sensitivity analysis using Sobol indices.
Implements Sobol sensitivity analysis for quantifying the contribution
of input variables to output variance.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 278](Python/Monte_Carlo/uncertainty.py#L278)
##### `analyze(self, model, input_distributions, n_samples, variable_names, second_order, bootstrap_confidence)`
Perform Sobol sensitivity analysis.
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
**Source:** [Line 283](Python/Monte_Carlo/uncertainty.py#L283)
##### `_generate_sobol_matrices(self, input_distributions, n_samples)`
Generate Sobol sample matrices A, B, and AB_i.
**Source:** [Line 362](Python/Monte_Carlo/uncertainty.py#L362)
##### `_evaluate_model_batch(self, model, input_samples)`
Evaluate model for batch of samples.
**Source:** [Line 388](Python/Monte_Carlo/uncertainty.py#L388)
##### `_compute_sobol_indices(self, f_A, f_B, f_AB)`
Compute first-order and total-order Sobol indices.
**Source:** [Line 407](Python/Monte_Carlo/uncertainty.py#L407)
##### `_compute_second_order_indices(self, f_A, f_B, f_AB, input_distributions, n_samples)`
Compute second-order Sobol indices.
**Source:** [Line 436](Python/Monte_Carlo/uncertainty.py#L436)
##### `_bootstrap_confidence_intervals(self, f_A, f_B, f_AB, n_bootstrap)`
Compute bootstrap confidence intervals.
**Source:** [Line 450](Python/Monte_Carlo/uncertainty.py#L450)
**Class Source:** [Line 271](Python/Monte_Carlo/uncertainty.py#L271)
### `PolynomialChaos`
Polynomial Chaos Expansion for uncertainty quantification.
Implements PCE surrogate modeling for efficient uncertainty propagation.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 491](Python/Monte_Carlo/uncertainty.py#L491)
##### `fit(self, model, input_distributions, polynomial_order, n_samples, sampling_method)`
Fit polynomial chaos expansion.
Args:
model: Model function
input_distributions: Input variable distributions
polynomial_order: Maximum polynomial order
n_samples: Number of training samples
sampling_method: Sampling method for training points
**kwargs: Additional arguments
Returns:
Dictionary with fit information
**Source:** [Line 498](Python/Monte_Carlo/uncertainty.py#L498)
##### `predict(self, input_samples)`
Predict using PCE surrogate.
**Source:** [Line 548](Python/Monte_Carlo/uncertainty.py#L548)
##### `_generate_training_data(self, model, input_distributions, n_samples, sampling_method)`
Generate training data for PCE.
**Source:** [Line 557](Python/Monte_Carlo/uncertainty.py#L557)
##### `_fit_polynomial_regression(self, inputs, outputs, order)`
Fit polynomial regression (simplified PCE).
**Source:** [Line 586](Python/Monte_Carlo/uncertainty.py#L586)
**Class Source:** [Line 485](Python/Monte_Carlo/uncertainty.py#L485)