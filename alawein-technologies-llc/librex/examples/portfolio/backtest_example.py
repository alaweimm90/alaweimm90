#!/usr/bin/env python3
"""
Portfolio Backtesting Example

This example demonstrates how to backtest portfolio optimization strategies
using historical data with realistic transaction costs and rebalancing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from Librex.domains.portfolio import (
    PortfolioProblem,
    ProblemType,
    PortfolioConstraints,
    PortfolioBacktester,
    BacktestConfig,
    DataLoader
)


def main():
    """Run portfolio backtesting example"""

    print("=" * 70)
    print("PORTFOLIO BACKTESTING EXAMPLE")
    print("=" * 70)

    # Step 1: Generate realistic market data
    print("\n1. Generating market data...")

    loader = DataLoader()

    # Create a realistic market with different regimes
    n_periods = 252 * 5  # 5 years of daily data
    n_assets = 6

    # Asset characteristics
    assets = {
        'Large_Cap_Growth': {'mu': 0.12, 'sigma': 0.18},
        'Large_Cap_Value': {'mu': 0.10, 'sigma': 0.15},
        'Small_Cap': {'mu': 0.14, 'sigma': 0.25},
        'International': {'mu': 0.09, 'sigma': 0.20},
        'Bonds': {'mu': 0.04, 'sigma': 0.05},
        'REIT': {'mu': 0.08, 'sigma': 0.22}
    }

    # Generate returns with regime changes
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='B')  # Business days
    returns_list = []

    # Market regimes
    regime_changes = [0, n_periods//4, n_periods//2, 3*n_periods//4, n_periods]
    regimes = ['bull', 'volatile', 'bear', 'recovery']

    for i in range(len(regimes)):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i+1]
        regime = regimes[i]
        regime_length = end_idx - start_idx

        # Adjust returns based on regime
        if regime == 'bull':
            mu_adj = 1.5
            sigma_adj = 0.8
            corr = 0.4
        elif regime == 'bear':
            mu_adj = -0.5
            sigma_adj = 1.5
            corr = 0.7
        elif regime == 'volatile':
            mu_adj = 0.8
            sigma_adj = 2.0
            corr = 0.6
        else:  # recovery
            mu_adj = 1.2
            sigma_adj = 1.0
            corr = 0.5

        # Generate correlated returns for this regime
        regime_returns = []
        for name, params in assets.items():
            daily_mu = (params['mu'] * mu_adj) / 252
            daily_sigma = (params['sigma'] * sigma_adj) / np.sqrt(252)
            regime_returns.append(np.random.normal(daily_mu, daily_sigma, regime_length))

        # Apply correlation
        corr_matrix = np.full((n_assets, n_assets), corr)
        np.fill_diagonal(corr_matrix, 1.0)

        from scipy.linalg import cholesky
        try:
            L = cholesky(corr_matrix, lower=True)
            regime_array = np.array(regime_returns).T @ L.T
        except:
            regime_array = np.array(regime_returns).T

        returns_list.append(regime_array)

    # Combine all regimes
    all_returns = np.vstack(returns_list)
    returns_df = pd.DataFrame(
        all_returns,
        index=dates,
        columns=list(assets.keys())
    )

    print(f"  Generated {n_periods} days of returns for {n_assets} assets")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Market regimes: {regimes}")

    # Step 2: Configure backtesting
    print("\n2. Configuring backtest parameters...")

    # Different strategies to test
    strategies = {
        'Conservative': {
            'problem_type': ProblemType.MIN_VARIANCE,
            'rebalance': 'quarterly',
            'risk_aversion': 5.0,
            'constraints': PortfolioConstraints(
                long_only=True,
                max_weight=0.4,
                cardinality=5
            )
        },
        'Balanced': {
            'problem_type': ProblemType.MEAN_VARIANCE,
            'rebalance': 'monthly',
            'risk_aversion': 2.0,
            'constraints': PortfolioConstraints(
                long_only=True,
                max_weight=0.35
            )
        },
        'Aggressive': {
            'problem_type': ProblemType.MAX_SHARPE,
            'rebalance': 'monthly',
            'risk_aversion': 0.5,
            'constraints': PortfolioConstraints(
                long_only=True,
                max_weight=0.5,
                min_weight=0.05
            )
        },
        'Risk_Parity': {
            'problem_type': ProblemType.RISK_PARITY,
            'rebalance': 'quarterly',
            'risk_aversion': 1.0,
            'constraints': PortfolioConstraints(long_only=True)
        }
    }

    # Benchmark: 60/40 portfolio
    benchmark_weights = np.array([0.15, 0.15, 0.10, 0.20, 0.40, 0.0])  # 60% equity, 40% bonds

    # Step 3: Run backtests for each strategy
    print("\n3. Running backtests...")
    results = {}

    for strategy_name, strategy_params in strategies.items():
        print(f"\n  Testing {strategy_name} strategy...")

        config = BacktestConfig(
            lookback_window=252,  # 1 year lookback
            rebalance_frequency=strategy_params['rebalance'],
            min_history=60,  # Minimum 60 days before first trade
            transaction_cost=0.001,  # 10 basis points
            slippage=0.0005,  # 5 basis points slippage
            initial_capital=1000000,
            risk_free_rate=0.02,
            benchmark_weights=benchmark_weights
        )

        backtester = PortfolioBacktester(returns_df, config)

        backtest_result = backtester.run_backtest(
            problem_type=strategy_params['problem_type'],
            constraints=strategy_params['constraints']
        )

        results[strategy_name] = backtest_result

        # Print summary statistics
        print(f"    Total Return: {backtest_result.total_return:.2%}")
        print(f"    Sharpe Ratio: {backtest_result.sharpe_ratio:.3f}")
        print(f"    Max Drawdown: {backtest_result.maximum_drawdown:.2%}")
        print(f"    Total Trades: {backtest_result.total_trades}")

    # Step 4: Compare strategies
    print("\n4. STRATEGY COMPARISON:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'Trades':<10}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{name:<15} "
              f"{result.annualized_return:<10.2%} "
              f"{result.annualized_volatility:<10.2%} "
              f"{result.sharpe_ratio:<10.3f} "
              f"{result.sortino_ratio:<10.3f} "
              f"{result.maximum_drawdown:<10.2%} "
              f"{result.total_trades:<10d}")

    # Step 5: Analyze transaction costs impact
    print("\n5. TRANSACTION COST ANALYSIS:")
    print("-" * 50)

    for name, result in results.items():
        cost_impact = result.total_transaction_cost
        avg_turnover = result.average_turnover

        print(f"{name:<15}: "
              f"Total Cost: {cost_impact:.2%}, "
              f"Avg Turnover: {avg_turnover:.1%}")

    # Step 6: Create visualizations
    print("\n6. Creating performance visualizations...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cumulative returns
        ax = axes[0, 0]
        for name, result in results.items():
            cumulative = (1 + pd.Series(result.portfolio_returns)).cumprod()
            ax.plot(result.dates, cumulative, label=name, linewidth=1.5)

        # Add benchmark
        if results['Conservative'].benchmark_returns is not None:
            benchmark_cumulative = (1 + pd.Series(results['Conservative'].benchmark_returns)).cumprod()
            ax.plot(results['Conservative'].dates, benchmark_cumulative,
                   label='60/40 Benchmark', linestyle='--', color='black', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Strategy Performance Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 2: Rolling Sharpe Ratio (252-day)
        ax = axes[0, 1]
        window = 252

        for name, result in results.items():
            returns_series = pd.Series(result.portfolio_returns, index=result.dates)
            rolling_mean = returns_series.rolling(window).mean() * 252
            rolling_std = returns_series.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_mean - 0.02) / rolling_std
            ax.plot(result.dates, rolling_sharpe, label=name, linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Sharpe Ratio (1Y)')
        ax.set_title('Rolling Risk-Adjusted Performance')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 3: Drawdown visualization
        ax = axes[1, 0]

        for name, result in results.items():
            cumulative = (1 + pd.Series(result.portfolio_returns)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            ax.fill_between(result.dates, drawdown, 0, label=name, alpha=0.3)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Underwater Plot (Drawdowns)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 4: Portfolio weights over time (for Balanced strategy)
        ax = axes[1, 1]
        balanced_result = results['Balanced']
        weights_df = pd.DataFrame(
            balanced_result.portfolio_weights,
            index=balanced_result.dates,
            columns=list(assets.keys())
        )

        # Plot stacked area chart
        ax.stackplot(balanced_result.dates,
                    weights_df.T,
                    labels=weights_df.columns,
                    alpha=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Composition Over Time (Balanced Strategy)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print("  Saved visualization to 'backtest_results.png'")

    except ImportError:
        print("  Matplotlib not available for visualization")

    # Step 7: Generate detailed report for best strategy
    print("\n7. DETAILED REPORT - BEST PERFORMING STRATEGY:")
    print("-" * 70)

    # Find best strategy by Sharpe ratio
    best_strategy = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    best_result = results[best_strategy]

    print(f"\nBest Strategy: {best_strategy}")
    print(best_result.summary_report())

    # Additional analysis
    print("\n8. KEY INSIGHTS:")
    print("-" * 50)
    print("• Transaction costs significantly impact returns")
    print("• More frequent rebalancing doesn't always improve performance")
    print("• Risk Parity provides stable risk-adjusted returns")
    print("• Strategy performance varies across market regimes")
    print("• Diversification across strategies may be beneficial")

    print("\n" + "=" * 70)
    print("Backtesting complete!")


if __name__ == "__main__":
    main()