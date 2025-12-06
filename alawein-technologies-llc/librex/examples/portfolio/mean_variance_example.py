#!/usr/bin/env python3
"""
Mean-Variance Portfolio Optimization Example

This example demonstrates how to use Librex's portfolio optimization
framework to solve the classic Markowitz mean-variance optimization problem.
"""

import numpy as np
import pandas as pd
from Librex.domains.portfolio import (
    PortfolioProblem,
    ProblemType,
    PortfolioConstraints,
    PortfolioAdapter,
    PortfolioMetrics,
    DataLoader,
    RiskEstimator
)
from Librex import optimize


def main():
    """Run mean-variance portfolio optimization example"""

    print("=" * 70)
    print("MEAN-VARIANCE PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 70)

    # Step 1: Load and prepare data
    print("\n1. Loading data...")
    loader = DataLoader()

    # Generate synthetic data for demonstration
    # In practice, replace with: loader.load_from_yahoo(['AAPL', 'GOOGL', 'MSFT', ...])
    prices = loader.generate_synthetic_prices(
        n_assets=8,
        n_periods=252 * 2,  # 2 years of daily data
        annual_return=0.10,
        annual_volatility=0.15,
        correlation=0.3,
        asset_names=['Tech_1', 'Tech_2', 'Finance_1', 'Finance_2',
                    'Healthcare_1', 'Healthcare_2', 'Energy_1', 'Energy_2']
    )

    returns = loader.prices_to_returns(prices)
    print(f"  Loaded {len(prices)} days of data for {prices.shape[1]} assets")

    # Step 2: Estimate risk parameters
    print("\n2. Estimating risk parameters...")
    estimator = RiskEstimator(returns.values)

    # Use Ledoit-Wolf shrinkage for more stable covariance
    expected_returns, cov_matrix = estimator.estimate_risk(
        method='ledoit_wolf',
        return_method='exponential'  # Give more weight to recent returns
    )

    print(f"  Expected annual returns: {expected_returns * 252:.1%} (average)")
    print(f"  Average correlation: {np.mean(np.corrcoef(returns.T)[np.triu_indices(8, k=1)]):.2f}")

    # Step 3: Define optimization problem
    print("\n3. Setting up optimization problem...")

    # Define constraints
    constraints = PortfolioConstraints(
        long_only=True,        # No short selling
        sum_to_one=True,       # Fully invested
        max_weight=0.3,        # Maximum 30% in any asset
        min_weight=0.02,       # Minimum 2% if invested
        cardinality=6          # Invest in at most 6 assets
    )

    # Create portfolio problem
    problem = PortfolioProblem(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        problem_type=ProblemType.MEAN_VARIANCE,
        constraints=constraints,
        risk_aversion=2.0,      # Risk aversion parameter
        transaction_costs=0.001, # 10 basis points
        risk_free_rate=0.02     # 2% risk-free rate
    )

    print(f"  Problem type: Mean-Variance Optimization")
    print(f"  Risk aversion: {problem.risk_aversion}")
    print(f"  Constraints: Long-only, Max weight 30%, Max 6 positions")

    # Step 4: Optimize portfolio
    print("\n4. Optimizing portfolio...")

    adapter = PortfolioAdapter()
    standardized = adapter.encode_problem(problem)

    # Use Librex's optimization engine
    result = optimize(
        standardized.objective_function,
        bounds=adapter.get_bounds(),
        initial_point=adapter.get_initial_solution(),
        method='scipy',  # Can also use 'genetic_algorithm' for global optimization
        max_iter=1000
    )

    # Decode solution
    portfolio = adapter.decode_solution(
        StandardizedSolution(
            vector=result['solution'],
            objective_value=result['objective_value'],
            is_valid=True
        )
    )

    # Step 5: Analyze results
    print("\n5. OPTIMIZATION RESULTS:")
    print("-" * 40)

    weights = portfolio['weights']
    non_zero = weights > 0.01

    print("\nPortfolio Allocation:")
    for i, asset in enumerate(prices.columns):
        if weights[i] > 0.01:
            print(f"  {asset:15s}: {weights[i]:6.2%}")

    print(f"\nNumber of positions: {np.sum(non_zero)}")
    print(f"Total weight: {np.sum(weights):.2%}")

    # Step 6: Calculate portfolio metrics
    print("\n6. PORTFOLIO METRICS:")
    print("-" * 40)

    metrics = PortfolioMetrics(
        weights=weights,
        returns=returns.values,
        risk_free_rate=problem.risk_free_rate
    )

    summary = metrics.summary()

    print(f"Expected Return (Annual): {summary['expected_return']:7.2%}")
    print(f"Volatility (Annual):      {summary['portfolio_volatility']:7.2%}")
    print(f"Sharpe Ratio:            {summary['sharpe_ratio']:7.3f}")
    print(f"Sortino Ratio:           {summary['sortino_ratio']:7.3f}")
    print(f"Maximum Drawdown:        {summary['max_drawdown']:7.2%}")
    print(f"Value at Risk (95%):     {summary['var_95']:7.2%}")
    print(f"CVaR (95%):              {summary['cvar_95']:7.2%}")

    # Step 7: Compare with equal-weight portfolio
    print("\n7. COMPARISON WITH EQUAL-WEIGHT PORTFOLIO:")
    print("-" * 40)

    equal_weights = np.ones(len(prices.columns)) / len(prices.columns)
    equal_metrics = PortfolioMetrics(
        weights=equal_weights,
        returns=returns.values,
        risk_free_rate=problem.risk_free_rate
    )

    equal_summary = equal_metrics.summary()

    print(f"                     Optimized    Equal-Weight")
    print(f"Expected Return:     {summary['expected_return']:7.2%}      {equal_summary['expected_return']:7.2%}")
    print(f"Volatility:          {summary['portfolio_volatility']:7.2%}      {equal_summary['portfolio_volatility']:7.2%}")
    print(f"Sharpe Ratio:        {summary['sharpe_ratio']:7.3f}      {equal_summary['sharpe_ratio']:7.3f}")

    # Step 8: Risk decomposition
    print("\n8. RISK DECOMPOSITION:")
    print("-" * 40)

    risk_contributions = metrics.risk_contributions()
    print("\nRisk Contributions:")
    for i, asset in enumerate(prices.columns):
        if weights[i] > 0.01:
            print(f"  {asset:15s}: {risk_contributions[i]:6.2%}")

    print("\n" + "=" * 70)
    print("Optimization complete!")


if __name__ == "__main__":
    # For compatibility with StandardizedSolution import
    from Librex.core.interfaces import StandardizedSolution
    main()