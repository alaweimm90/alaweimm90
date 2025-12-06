"""
Portfolio Optimization Objective Functions

This module implements various portfolio optimization objectives including:
- Mean-Variance (Markowitz)
- Conditional Value at Risk (CVaR)
- Risk Parity
- Black-Litterman
- Maximum Sharpe Ratio
- Minimum Variance
- Factor Model Based
- Robust Optimization

All objectives follow the convention of minimization.
For maximization objectives (e.g., Sharpe ratio), we minimize the negative.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from Librex.domains.portfolio.portfolio_problem import PortfolioProblem


def mean_variance_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Mean-Variance (Markowitz) objective function

    Minimize: -μ'w + λ/2 * w'Σw + transaction_costs

    Where:
    - μ is expected returns vector
    - w is portfolio weights
    - λ is risk aversion parameter
    - Σ is covariance matrix

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Objective value (to minimize)
    """
    expected_returns = problem.get_expected_returns()
    cov_matrix = problem.get_risk_matrix()

    # Portfolio return
    portfolio_return = np.dot(weights, expected_returns)

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights

    # Mean-variance objective
    objective = -portfolio_return + (problem.risk_aversion / 2) * portfolio_variance

    # Add transaction costs if applicable
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        objective += transaction_cost

    return objective


def risk_parity_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Risk Parity objective function

    Minimize the squared difference between marginal risk contributions
    and target equal contributions.

    Risk contribution of asset i:
    RC_i = w_i * (Σw)_i / σ_p

    Target: RC_i = 1/n for all i

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Objective value (sum of squared deviations from equal risk contribution)
    """
    cov_matrix = problem.get_risk_matrix()
    n = len(weights)

    # Avoid division by zero
    if np.sum(np.abs(weights)) < 1e-10:
        return 1e10

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights

    if portfolio_variance <= 1e-10:
        return 1e10

    portfolio_vol = np.sqrt(portfolio_variance)

    # Marginal contributions to risk
    marginal_contrib = cov_matrix @ weights

    # Risk contributions
    risk_contrib = weights * marginal_contrib / portfolio_vol

    # Target contribution (equal for all assets)
    target_contrib = 1.0 / n

    # Objective: minimize squared deviations from target
    objective = np.sum((risk_contrib - target_contrib) ** 2)

    # Add soft constraint to encourage full investment
    weight_sum_penalty = 100 * (np.sum(weights) - 1.0) ** 2

    return objective + weight_sum_penalty


def cvar_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Conditional Value at Risk (CVaR) objective function

    Minimize CVaR at confidence level α
    CVaR_α = E[L | L >= VaR_α]

    Where L is portfolio loss (-return)

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        CVaR value (expected shortfall)
    """
    if problem.returns_scenarios is None:
        warnings.warn("No return scenarios provided, using mean-variance approximation")
        return mean_variance_objective(weights, problem)

    # Portfolio returns for each scenario
    portfolio_returns = problem.returns_scenarios @ weights

    # Convert to losses (negative returns)
    losses = -portfolio_returns

    # Calculate VaR at confidence level
    alpha = problem.confidence_level
    var_index = int(np.ceil(len(losses) * (1 - alpha)))
    sorted_losses = np.sort(losses)

    # VaR
    var = sorted_losses[var_index] if var_index < len(sorted_losses) else sorted_losses[-1]

    # CVaR: expected value of losses exceeding VaR
    tail_losses = sorted_losses[var_index:]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        cvar += transaction_cost

    return cvar


def maximum_sharpe_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Maximum Sharpe Ratio objective (minimize negative Sharpe)

    Sharpe Ratio = (E[R] - r_f) / σ

    Where:
    - E[R] is expected portfolio return
    - r_f is risk-free rate
    - σ is portfolio standard deviation

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Negative Sharpe ratio (for minimization)
    """
    expected_returns = problem.get_expected_returns()
    cov_matrix = problem.get_risk_matrix()

    # Portfolio return
    portfolio_return = np.dot(weights, expected_returns)

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights

    # Avoid division by zero
    if portfolio_variance <= 1e-10:
        return 1e10

    portfolio_std = np.sqrt(portfolio_variance)

    # Excess return
    excess_return = portfolio_return - problem.risk_free_rate

    # Sharpe ratio (negative for minimization)
    sharpe = excess_return / portfolio_std

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        # Adjust Sharpe for transaction costs
        sharpe = (excess_return - transaction_cost) / portfolio_std

    return -sharpe  # Minimize negative Sharpe = maximize Sharpe


def minimum_variance_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Minimum Variance Portfolio objective

    Minimize: w'Σw

    Subject to typical portfolio constraints

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Portfolio variance
    """
    cov_matrix = problem.get_risk_matrix()

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        portfolio_variance += transaction_cost

    return portfolio_variance


def black_litterman_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Black-Litterman objective function

    Uses posterior expected returns from Black-Litterman model
    combined with mean-variance optimization

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Mean-variance objective with Black-Litterman returns
    """
    # Get Black-Litterman posterior returns
    bl_returns = problem.get_expected_returns()  # This uses BL if problem type is BLACK_LITTERMAN

    # Get posterior covariance (can be adjusted with uncertainty)
    cov_matrix = problem.get_risk_matrix()

    # Portfolio return with BL estimates
    portfolio_return = np.dot(weights, bl_returns)

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights

    # Mean-variance objective with BL parameters
    objective = -portfolio_return + (problem.risk_aversion / 2) * portfolio_variance

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        objective += transaction_cost

    return objective


def factor_model_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Factor Model based portfolio optimization

    Uses factor decomposition of returns and risk

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Objective value based on factor model
    """
    if problem.factor_model is None:
        warnings.warn("No factor model provided, using standard mean-variance")
        return mean_variance_objective(weights, problem)

    # Get returns and covariance from factor model
    expected_returns = problem.factor_model.get_expected_returns()
    cov_matrix = problem.factor_model.get_covariance_matrix()

    # Portfolio factor exposures
    factor_exposures = problem.factor_model.factor_loadings.T @ weights

    # Check factor exposure constraints if any
    if problem.constraints.factor_exposures:
        exposure_penalty = 0
        for factor_name, (min_exp, max_exp) in problem.constraints.factor_exposures.items():
            # Assume factor_name maps to an index
            # In practice, you'd have a mapping
            factor_idx = 0  # Placeholder
            exposure = factor_exposures[factor_idx]

            if exposure < min_exp:
                exposure_penalty += 100 * (min_exp - exposure) ** 2
            elif exposure > max_exp:
                exposure_penalty += 100 * (exposure - max_exp) ** 2
    else:
        exposure_penalty = 0

    # Standard mean-variance with factor model
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = weights @ cov_matrix @ weights

    objective = -portfolio_return + (problem.risk_aversion / 2) * portfolio_variance + exposure_penalty

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        objective += transaction_cost

    return objective


def robust_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Robust portfolio optimization objective

    Considers uncertainty in expected returns and covariance estimates
    Uses worst-case optimization over an uncertainty set

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Worst-case objective value
    """
    expected_returns = problem.get_expected_returns()
    cov_matrix = problem.get_risk_matrix()

    # Uncertainty set size
    epsilon = problem.uncertainty_set_size

    # Worst-case returns (subtract uncertainty)
    returns_std = np.sqrt(np.diag(cov_matrix))
    worst_case_returns = expected_returns - epsilon * returns_std

    # Robust variance (add uncertainty to covariance)
    robust_cov = cov_matrix * (1 + epsilon)

    # Portfolio metrics with robust estimates
    portfolio_return = np.dot(weights, worst_case_returns)
    portfolio_variance = weights @ robust_cov @ weights

    # Robust objective
    objective = -portfolio_return + (problem.risk_aversion / 2) * portfolio_variance

    # Add regularization to encourage diversification
    if problem.regularization > 0:
        # L2 regularization on weights
        regularization = problem.regularization * np.sum(weights ** 2)
        objective += regularization

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        objective += transaction_cost

    return objective


def tracking_error_objective(weights: np.ndarray,
                            problem: 'PortfolioProblem',
                            lambda_te: float = 1.0) -> float:
    """
    Tracking Error minimization objective

    Minimize: (w - w_b)'Σ(w - w_b) + λ * other_objectives

    Where w_b is the benchmark portfolio

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance
        lambda_te: Weight for tracking error vs return

    Returns:
        Tracking error plus return objective
    """
    if problem.benchmark_weights is None:
        warnings.warn("No benchmark provided, using minimum variance")
        return minimum_variance_objective(weights, problem)

    cov_matrix = problem.get_risk_matrix()

    # Active weights (deviation from benchmark)
    active_weights = weights - problem.benchmark_weights

    # Tracking error (variance of active weights)
    tracking_variance = active_weights @ cov_matrix @ active_weights

    # Expected active return
    expected_returns = problem.get_expected_returns()
    active_return = np.dot(active_weights, expected_returns)

    # Combine tracking error with return objective
    objective = lambda_te * tracking_variance - active_return

    # Add transaction costs
    if problem.has_transaction_costs and problem.current_weights is not None:
        trades = np.abs(weights - problem.current_weights)
        transaction_cost = np.dot(trades, problem.transaction_costs)
        objective += transaction_cost

    return objective


def max_diversification_objective(weights: np.ndarray, problem: 'PortfolioProblem') -> float:
    """
    Maximum Diversification Ratio objective

    DR = (w'σ) / σ_p

    Where σ is vector of asset volatilities and σ_p is portfolio volatility

    Maximize DR = minimize -DR

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance

    Returns:
        Negative diversification ratio
    """
    cov_matrix = problem.get_risk_matrix()

    # Individual asset volatilities
    asset_vols = np.sqrt(np.diag(cov_matrix))

    # Weighted average of asset volatilities
    avg_vol = np.dot(weights, asset_vols)

    # Portfolio volatility
    portfolio_variance = weights @ cov_matrix @ weights

    if portfolio_variance <= 1e-10:
        return 1e10

    portfolio_vol = np.sqrt(portfolio_variance)

    # Diversification ratio (negative for minimization)
    div_ratio = avg_vol / portfolio_vol

    return -div_ratio


def entropy_regularized_objective(weights: np.ndarray,
                                 problem: 'PortfolioProblem',
                                 gamma: float = 0.1) -> float:
    """
    Entropy-regularized portfolio objective

    Adds entropy term to encourage diversification:
    H(w) = -Σ w_i log(w_i)

    Args:
        weights: Portfolio weight vector
        problem: PortfolioProblem instance
        gamma: Entropy regularization parameter

    Returns:
        Mean-variance objective with entropy regularization
    """
    # Base mean-variance objective
    base_objective = mean_variance_objective(weights, problem)

    # Entropy regularization
    # Avoid log(0) by adding small constant
    weights_positive = np.maximum(weights, 1e-10)
    entropy = -np.sum(weights_positive * np.log(weights_positive))

    # Maximize entropy = minimize negative entropy
    return base_objective - gamma * entropy