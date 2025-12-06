#!/usr/bin/env python3
"""
Risk Parity Portfolio Optimization Example

This example demonstrates risk parity portfolio construction where each
asset contributes equally to the overall portfolio risk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from Librex.core.interfaces import StandardizedSolution


def main():
    """Run risk parity portfolio optimization example"""

    print("=" * 70)
    print("RISK PARITY PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 70)

    # Step 1: Create diversified asset universe
    print("\n1. Creating diversified asset universe...")

    loader = DataLoader()

    # Asset classes with different risk/return profiles
    asset_classes = {
        'US_Equity': {'return': 0.10, 'vol': 0.16},
        'Int_Equity': {'return': 0.08, 'vol': 0.18},
        'Bonds': {'return': 0.04, 'vol': 0.05},
        'Real_Estate': {'return': 0.07, 'vol': 0.12},
        'Commodities': {'return': 0.05, 'vol': 0.20},
        'Gold': {'return': 0.06, 'vol': 0.15}
    }

    # Generate returns with realistic correlations
    n_periods = 252 * 3  # 3 years of daily data
    n_assets = len(asset_classes)

    # Create correlation matrix (lower correlations between asset classes)
    correlation_matrix = np.array([
        [1.00, 0.70, 0.10, 0.50, 0.20, 0.05],  # US_Equity
        [0.70, 1.00, 0.05, 0.45, 0.25, 0.10],  # Int_Equity
        [0.10, 0.05, 1.00, 0.30, -0.10, 0.20], # Bonds
        [0.50, 0.45, 0.30, 1.00, 0.35, 0.25],  # Real_Estate
        [0.20, 0.25, -0.10, 0.35, 1.00, 0.50], # Commodities
        [0.05, 0.10, 0.20, 0.25, 0.50, 1.00]   # Gold
    ])

    # Generate correlated returns
    asset_names = list(asset_classes.keys())
    returns_data = []

    for i, (name, params) in enumerate(asset_classes.items()):
        daily_return = params['return'] / 252
        daily_vol = params['vol'] / np.sqrt(252)
        returns_data.append(np.random.normal(daily_return, daily_vol, n_periods))

    # Apply correlations
    from scipy.linalg import cholesky
    L = cholesky(correlation_matrix, lower=True)
    returns_array = np.array(returns_data).T @ L.T

    returns_df = pd.DataFrame(
        returns_array,
        columns=asset_names,
        index=pd.date_range(end=pd.Timestamp.now(), periods=n_periods, freq='D')
    )

    print(f"  Created portfolio with {n_assets} asset classes")
    print(f"  Data period: {n_periods} days")

    # Step 2: Estimate risk parameters
    print("\n2. Estimating risk parameters...")

    estimator = RiskEstimator(returns_df.values)
    expected_returns, cov_matrix = estimator.estimate_risk(
        method='exponential',  # Recent data more important
        cov_halflife=120,      # 4-month halflife
        return_method='exponential',
        return_halflife=60     # 2-month halflife
    )

    # Display correlation matrix
    corr_matrix = np.corrcoef(returns_df.T)
    print("\nAsset Correlations:")
    print(pd.DataFrame(corr_matrix, columns=asset_names, index=asset_names).round(2))

    # Step 3: Risk Parity Optimization
    print("\n3. Optimizing Risk Parity portfolio...")

    # Risk parity doesn't need many constraints
    constraints = PortfolioConstraints(
        long_only=True,
        sum_to_one=True
    )

    problem = PortfolioProblem(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        problem_type=ProblemType.RISK_PARITY,
        constraints=constraints
    )

    adapter = PortfolioAdapter()
    standardized = adapter.encode_problem(problem)

    # Optimize
    result = optimize(
        standardized.objective_function,
        bounds=adapter.get_bounds(),
        initial_point=np.ones(n_assets) / n_assets,  # Start with equal weights
        method='scipy',
        max_iter=2000
    )

    portfolio = adapter.decode_solution(
        StandardizedSolution(
            vector=result['solution'],
            objective_value=result['objective_value'],
            is_valid=True
        )
    )

    # Step 4: Analyze Risk Parity Results
    print("\n4. RISK PARITY RESULTS:")
    print("-" * 40)

    rp_weights = portfolio['weights']

    print("\nAsset Allocation:")
    for i, asset in enumerate(asset_names):
        print(f"  {asset:15s}: {rp_weights[i]:6.2%}")

    # Calculate risk contributions
    metrics = PortfolioMetrics(
        weights=rp_weights,
        returns=returns_df.values,
        risk_free_rate=0.02
    )

    risk_contributions = metrics.risk_contributions()

    print("\nRisk Contributions (should be ~equal):")
    for i, asset in enumerate(asset_names):
        print(f"  {asset:15s}: {risk_contributions[i]:6.2%}")

    # Verify equal risk contribution
    risk_parity_score = np.std(risk_contributions)
    print(f"\nRisk Parity Score (lower is better): {risk_parity_score:.4f}")

    # Step 5: Compare with other strategies
    print("\n5. STRATEGY COMPARISON:")
    print("-" * 40)

    strategies = {}

    # 1. Risk Parity
    strategies['Risk Parity'] = rp_weights

    # 2. Equal Weight
    strategies['Equal Weight'] = np.ones(n_assets) / n_assets

    # 3. Minimum Variance
    min_var_problem = PortfolioProblem(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        problem_type=ProblemType.MIN_VARIANCE,
        constraints=constraints
    )

    min_var_standardized = adapter.encode_problem(min_var_problem)
    min_var_result = optimize(
        min_var_standardized.objective_function,
        bounds=adapter.get_bounds(),
        initial_point=np.ones(n_assets) / n_assets,
        method='scipy'
    )

    strategies['Min Variance'] = adapter._normalize_weights(min_var_result['solution'])

    # 4. Maximum Sharpe
    max_sharpe_problem = PortfolioProblem(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        problem_type=ProblemType.MAX_SHARPE,
        constraints=constraints,
        risk_free_rate=0.02
    )

    max_sharpe_standardized = adapter.encode_problem(max_sharpe_problem)
    max_sharpe_result = optimize(
        max_sharpe_standardized.objective_function,
        bounds=adapter.get_bounds(),
        initial_point=np.ones(n_assets) / n_assets,
        method='scipy'
    )

    strategies['Max Sharpe'] = adapter._normalize_weights(max_sharpe_result['solution'])

    # Compare strategies
    print("\nStrategy Performance Comparison:")
    print(f"{'Strategy':15s} {'Return':>10s} {'Vol':>10s} {'Sharpe':>10s} {'Max RC':>10s}")
    print("-" * 55)

    for name, weights in strategies.items():
        strategy_metrics = PortfolioMetrics(
            weights=weights,
            returns=returns_df.values,
            risk_free_rate=0.02
        )
        summary = strategy_metrics.summary()
        risk_contrib = strategy_metrics.risk_contributions()

        print(f"{name:15s} {summary['expected_return']:10.2%} "
              f"{summary['portfolio_volatility']:10.2%} "
              f"{summary['sharpe_ratio']:10.3f} "
              f"{np.max(risk_contrib):10.2%}")

    # Step 6: Visualize allocations
    print("\n6. Creating visualization...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Asset Allocations
        ax = axes[0, 0]
        x = np.arange(n_assets)
        width = 0.2

        for i, (name, weights) in enumerate(strategies.items()):
            ax.bar(x + i * width, weights, width, label=name)

        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Allocations by Strategy')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(asset_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Risk Contributions for Risk Parity
        ax = axes[0, 1]
        rp_risk_contrib = metrics.risk_contributions()
        ax.bar(asset_names, rp_risk_contrib)
        ax.axhline(y=1/n_assets, color='r', linestyle='--', label='Equal Risk Target')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Risk Parity - Risk Contributions')
        ax.set_xticklabels(asset_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Efficient Frontier
        ax = axes[1, 0]

        # Generate efficient frontier points
        n_points = 50
        target_returns = np.linspace(
            expected_returns.min() * 252,
            expected_returns.max() * 252,
            n_points
        )

        frontier_vols = []
        frontier_rets = []

        for target_ret in target_returns:
            # Simple mean-variance with target return
            try:
                mv_problem = PortfolioProblem(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    problem_type=ProblemType.MEAN_VARIANCE,
                    constraints=constraints,
                    risk_aversion=0.1  # Low risk aversion to get frontier
                )

                # Add return constraint (simplified)
                frontier_vols.append(np.sqrt(252) * np.sqrt(np.diagonal(cov_matrix).mean()))
                frontier_rets.append(target_ret)
            except:
                pass

        # Plot strategies on efficient frontier
        for name, weights in strategies.items():
            strategy_metrics = PortfolioMetrics(
                weights=weights,
                returns=returns_df.values
            )
            summary = strategy_metrics.summary()
            ax.scatter(summary['portfolio_volatility'], summary['expected_return'],
                      s=100, label=name, zorder=5)

        ax.set_xlabel('Volatility')
        ax.set_ylabel('Expected Return')
        ax.set_title('Risk-Return Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Correlation heatmap
        ax = axes[1, 1]
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(n_assets))
        ax.set_yticks(np.arange(n_assets))
        ax.set_xticklabels(asset_names, rotation=45)
        ax.set_yticklabels(asset_names)
        ax.set_title('Asset Correlation Matrix')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations
        for i in range(n_assets):
            for j in range(n_assets):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

        plt.tight_layout()
        plt.savefig('risk_parity_analysis.png', dpi=150, bbox_inches='tight')
        print("  Saved visualization to 'risk_parity_analysis.png'")

    except ImportError:
        print("  Matplotlib not available for visualization")

    print("\n" + "=" * 70)
    print("Risk Parity optimization complete!")
    print("\nKey Insights:")
    print("- Risk Parity allocates more to low-risk assets (Bonds)")
    print("- Each asset contributes equally to portfolio risk")
    print("- Often provides better risk-adjusted returns than equal weight")
    print("- Particularly effective in multi-asset class portfolios")


if __name__ == "__main__":
    main()