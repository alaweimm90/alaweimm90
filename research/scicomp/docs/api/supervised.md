# supervised
**Module:** `Python/Machine_Learning/supervised.py`
## Overview
Supervised Learning Algorithms for Scientific Computing
This module implements supervised learning algorithms specifically designed for
scientific computing applications, with emphasis on interpretability and
physics-aware modeling.
Classes:
LinearRegression: Linear regression with scientific extensions
PolynomialRegression: Polynomial regression for nonlinear relationships
RidgeRegression: Regularized regression for high-dimensional data
LogisticRegression: Classification with probabilistic outputs
SVM: Support Vector Machines for complex decision boundaries
RandomForest: Ensemble method for robust predictions
GradientBoosting: Gradient boosting for high-performance modeling
## Functions
### `create_test_datasets()`
Create test datasets for supervised learning validation.
**Source:** [Line 629](Python/Machine_Learning/supervised.py#L629)
### `plot_regression_results(model, X, y, X_test, title)`
Plot regression results with Berkeley styling.
**Source:** [Line 655](Python/Machine_Learning/supervised.py#L655)
## Classes
### `ModelResults`
Container for model results and diagnostics.
**Class Source:** [Line 33](Python/Machine_Learning/supervised.py#L33)
### `SupervisedModel`
Abstract base class for supervised learning models.
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 50](Python/Machine_Learning/supervised.py#L50)
##### `fit(self, X, y)`
Fit the model to training data.
**Source:** [Line 56](Python/Machine_Learning/supervised.py#L56)
##### `predict(self, X)`
Make predictions on new data.
**Source:** [Line 61](Python/Machine_Learning/supervised.py#L61)
##### `score(self, X, y)`
Calculate R² score.
**Source:** [Line 65](Python/Machine_Learning/supervised.py#L65)
##### `_validate_input(self, X, y)`
Validate input data.
**Source:** [Line 72](Python/Machine_Learning/supervised.py#L72)
**Class Source:** [Line 47](Python/Machine_Learning/supervised.py#L47)
### `LinearRegression`
Linear regression with advanced scientific computing features.
Features:
- Multiple solvers (normal equation, SVD, iterative)
- Uncertainty quantification
- Statistical inference
- Physics-aware constraints
#### Methods
##### `__init__(self, fit_intercept, solver, regularization, uncertainty_estimation)`
*No documentation available.*
**Source:** [Line 97](Python/Machine_Learning/supervised.py#L97)
##### `fit(self, X, y)`
Fit linear regression model.
Parameters:
X: Feature matrix (n_samples, n_features)
y: Target values (n_samples,)
Returns:
self: Fitted model
**Source:** [Line 111](Python/Machine_Learning/supervised.py#L111)
##### `_fit_normal_equation(self, X, y)`
Fit using normal equation.
**Source:** [Line 148](Python/Machine_Learning/supervised.py#L148)
##### `_fit_svd(self, X, y)`
Fit using SVD (most stable).
**Source:** [Line 157](Python/Machine_Learning/supervised.py#L157)
##### `_fit_iterative(self, X, y)`
Fit using iterative solver.
**Source:** [Line 169](Python/Machine_Learning/supervised.py#L169)
##### `_extract_coefficients(self, coeffs)`
Extract intercept and coefficients.
**Source:** [Line 183](Python/Machine_Learning/supervised.py#L183)
##### `_compute_uncertainty(self, X, y)`
Compute uncertainty estimates.
**Source:** [Line 192](Python/Machine_Learning/supervised.py#L192)
##### `predict(self, X, return_uncertainty)`
Make predictions.
Parameters:
X: Feature matrix
return_uncertainty: Whether to return prediction uncertainties
Returns:
predictions: Predicted values
uncertainties: Prediction uncertainties (if requested)
**Source:** [Line 207](Python/Machine_Learning/supervised.py#L207)
##### `confidence_intervals(self, X, confidence_level)`
Compute confidence intervals for predictions.
**Source:** [Line 239](Python/Machine_Learning/supervised.py#L239)
##### `summary(self)`
Return model summary statistics.
**Source:** [Line 251](Python/Machine_Learning/supervised.py#L251)
**Class Source:** [Line 86](Python/Machine_Learning/supervised.py#L86)
### `PolynomialRegression`
Polynomial regression for nonlinear relationships.
Features:
- Automatic feature generation
- Cross-validation for degree selection
- Regularization options
- Extrapolation warnings
#### Methods
##### `__init__(self, degree, include_bias, interaction_only, regularization)`
*No documentation available.*
**Source:** [Line 286](Python/Machine_Learning/supervised.py#L286)
##### `_generate_polynomial_features(self, X)`
Generate polynomial features.
**Source:** [Line 301](Python/Machine_Learning/supervised.py#L301)
##### `fit(self, X, y)`
Fit polynomial regression model.
**Source:** [Line 332](Python/Machine_Learning/supervised.py#L332)
##### `predict(self, X)`
Make predictions using polynomial model.
**Source:** [Line 355](Python/Machine_Learning/supervised.py#L355)
**Class Source:** [Line 275](Python/Machine_Learning/supervised.py#L275)
### `RidgeRegression`
Ridge regression with L2 regularization.
Inherits from LinearRegression with automatic regularization parameter selection.
#### Methods
##### `__init__(self, alpha, fit_intercept, solver, cv_folds)`
*No documentation available.*
**Source:** [Line 383](Python/Machine_Learning/supervised.py#L383)
##### `fit_with_cv(self, X, y, alpha_range)`
Fit with cross-validation for optimal alpha.
**Source:** [Line 394](Python/Machine_Learning/supervised.py#L394)
**Class Source:** [Line 376](Python/Machine_Learning/supervised.py#L376)
### `LogisticRegression`
Logistic regression for binary and multiclass classification.
Features:
- Multiple solvers
- Regularization options
- Probability predictions
- Feature importance
#### Methods
##### `__init__(self, penalty, C, solver, max_iter, multi_class)`
*No documentation available.*
**Source:** [Line 447](Python/Machine_Learning/supervised.py#L447)
##### `_sigmoid(self, z)`
Sigmoid activation function.
**Source:** [Line 463](Python/Machine_Learning/supervised.py#L463)
##### `_cost_function(self, params, X, y)`
Logistic regression cost function.
**Source:** [Line 469](Python/Machine_Learning/supervised.py#L469)
##### `_gradient(self, params, X, y)`
Gradient of cost function.
**Source:** [Line 495](Python/Machine_Learning/supervised.py#L495)
##### `fit(self, X, y)`
Fit logistic regression model.
**Source:** [Line 530](Python/Machine_Learning/supervised.py#L530)
##### `_fit_binary(self, X, y)`
Fit binary logistic regression.
**Source:** [Line 546](Python/Machine_Learning/supervised.py#L546)
##### `_fit_multiclass(self, X, y)`
Fit multiclass logistic regression using one-vs-rest.
**Source:** [Line 576](Python/Machine_Learning/supervised.py#L576)
##### `predict_proba(self, X)`
Predict class probabilities.
**Source:** [Line 600](Python/Machine_Learning/supervised.py#L600)
##### `predict(self, X)`
Make class predictions.
**Source:** [Line 618](Python/Machine_Learning/supervised.py#L618)
**Class Source:** [Line 436](Python/Machine_Learning/supervised.py#L436)