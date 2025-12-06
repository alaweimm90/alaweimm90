"""
Portfolio Backtesting Framework

Comprehensive backtesting system for portfolio optimization strategies including:
- Rolling window optimization
- Multiple rebalancing frequencies
- Transaction cost modeling
- Performance attribution
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

from Librex.domains.portfolio.portfolio_problem import PortfolioProblem, ProblemType, PortfolioConstraints
from Librex.domains.portfolio.portfolio_adapter import PortfolioAdapter
from Librex.domains.portfolio.metrics import PortfolioMetrics
from Librex.domains.portfolio.risk_models import RiskEstimator


@dataclass
class BacktestConfig:
    """
    Configuration for portfolio backtesting

    Attributes:
        lookback_window: Number of periods for estimation window
        rebalance_frequency: Frequency of rebalancing ('daily', 'weekly', 'monthly', 'quarterly')
        min_history: Minimum periods required before first optimization
        transaction_cost: Transaction cost per trade (fractional)
        slippage: Market impact / slippage (fractional)
        initial_capital: Starting capital amount
        max_leverage: Maximum leverage allowed
        risk_free_rate: Risk-free rate for Sharpe calculation
        benchmark_weights: Benchmark portfolio for comparison
    """
    lookback_window: int = 252  # 1 year for daily data
    rebalance_frequency: str = 'monthly'
    min_history: int = 60
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    initial_capital: float = 1000000.0
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02  # 2% annual
    benchmark_weights: Optional[np.ndarray] = None


@dataclass
class BacktestResults:
    """
    Results from portfolio backtesting

    Contains time series of portfolio values, weights, and performance metrics
    """
    dates: pd.DatetimeIndex
    portfolio_values: np.ndarray
    portfolio_returns: np.ndarray
    portfolio_weights: np.ndarray
    turnover: np.ndarray
    transaction_costs: np.ndarray

    # Performance metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float

    # vs Benchmark (if provided)
    benchmark_values: Optional[np.ndarray] = None
    benchmark_returns: Optional[np.ndarray] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

    # Additional statistics
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    best_period: float = 0.0
    worst_period: float = 0.0
    total_trades: int = 0
    average_turnover: float = 0.0
    total_transaction_cost: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis"""
        df = pd.DataFrame(index=self.dates)
        df['portfolio_value'] = self.portfolio_values
        df['portfolio_return'] = self.portfolio_returns
        df['turnover'] = self.turnover
        df['transaction_cost'] = self.transaction_costs

        if self.benchmark_values is not None:
            df['benchmark_value'] = self.benchmark_values
            df['benchmark_return'] = self.benchmark_returns
            df['active_return'] = self.portfolio_returns - self.benchmark_returns

        # Add weights as separate columns
        for i in range(self.portfolio_weights.shape[1]):
            df[f'weight_{i}'] = self.portfolio_weights[:, i]

        return df

    def summary_report(self) -> str:
        """Generate formatted summary report"""
        report = []
        report.append("=" * 70)
        report.append("PORTFOLIO BACKTEST RESULTS")
        report.append("=" * 70)

        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  Total Return:          {self.total_return:.2%}")
        report.append(f"  Annualized Return:     {self.annualized_return:.2%}")
        report.append(f"  Annualized Volatility: {self.annualized_volatility:.2%}")
        report.append(f"  Sharpe Ratio:          {self.sharpe_ratio:.3f}")
        report.append(f"  Sortino Ratio:         {self.sortino_ratio:.3f}")
        report.append(f"  Maximum Drawdown:      {self.maximum_drawdown:.2%}")
        report.append(f"  Calmar Ratio:          {self.calmar_ratio:.3f}")

        if self.benchmark_values is not None:
            report.append("\nVS BENCHMARK:")
            report.append(f"  Tracking Error:        {self.tracking_error:.2%}")
            report.append(f"  Information Ratio:     {self.information_ratio:.3f}")

        report.append("\nTRADING STATISTICS:")
        report.append(f"  Win Rate:              {self.win_rate:.2%}")
        report.append(f"  Average Win:           {self.average_win:.2%}")
        report.append(f"  Average Loss:          {self.average_loss:.2%}")
        report.append(f"  Best Period:           {self.best_period:.2%}")
        report.append(f"  Worst Period:          {self.worst_period:.2%}")

        report.append("\nTRANSACTION ANALYSIS:")
        report.append(f"  Total Trades:          {self.total_trades}")
        report.append(f"  Average Turnover:      {self.average_turnover:.2%}")
        report.append(f"  Total Trans. Cost:     {self.total_transaction_cost:.2%}")

        report.append("=" * 70)

        return "\n".join(report)


class PortfolioBacktester:
    """
    Main backtesting engine for portfolio optimization strategies
    """

    def __init__(self,
                 returns_data: pd.DataFrame,
                 config: BacktestConfig,
                 optimizer: Optional[Callable] = None):
        """
        Initialize backtester

        Args:
            returns_data: DataFrame of asset returns (dates as index, assets as columns)
            config: Backtest configuration
            optimizer: Optional custom optimizer function
        """
        self.returns_data = returns_data
        self.dates = returns_data.index
        self.n_periods = len(self.dates)
        self.n_assets = returns_data.shape[1]
        self.asset_names = returns_data.columns.tolist()
        self.config = config
        self.optimizer = optimizer

        # Set rebalance dates
        self.rebalance_dates = self._get_rebalance_dates()

    def _get_rebalance_dates(self) -> List[datetime]:
        """Get dates for portfolio rebalancing"""
        rebalance_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'yearly': 252
        }

        frequency = rebalance_map.get(self.config.rebalance_frequency, 21)

        # Start after minimum history
        start_idx = self.config.min_history
        rebalance_indices = list(range(start_idx, self.n_periods, frequency))

        return [self.dates[i] for i in rebalance_indices]

    def run_backtest(self,
                    problem_type: ProblemType = ProblemType.MEAN_VARIANCE,
                    constraints: Optional[PortfolioConstraints] = None,
                    **optimizer_kwargs) -> BacktestResults:
        """
        Run portfolio backtest

        Args:
            problem_type: Type of portfolio optimization
            constraints: Portfolio constraints
            **optimizer_kwargs: Additional arguments for optimizer

        Returns:
            BacktestResults object
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        # Initialize tracking arrays
        portfolio_values = np.zeros(self.n_periods)
        portfolio_returns = np.zeros(self.n_periods)
        portfolio_weights = np.zeros((self.n_periods, self.n_assets))
        turnover = np.zeros(self.n_periods)
        transaction_costs = np.zeros(self.n_periods)

        # Initialize portfolio
        portfolio_values[0] = self.config.initial_capital
        current_weights = np.ones(self.n_assets) / self.n_assets  # Equal weight start
        portfolio_weights[0] = current_weights

        # Benchmark tracking if provided
        if self.config.benchmark_weights is not None:
            benchmark_values = np.zeros(self.n_periods)
            benchmark_returns = np.zeros(self.n_periods)
            benchmark_values[0] = self.config.initial_capital
        else:
            benchmark_values = None
            benchmark_returns = None

        # Main backtest loop
        for t in range(1, self.n_periods):
            current_date = self.dates[t]
            period_return = self.returns_data.iloc[t].values

            # Check if rebalancing needed
            if current_date in self.rebalance_dates:
                # Get historical data for optimization
                lookback_start = max(0, t - self.config.lookback_window)
                historical_returns = self.returns_data.iloc[lookback_start:t].values

                # Estimate risk parameters
                risk_estimator = RiskEstimator(historical_returns)
                expected_returns, cov_matrix = risk_estimator.estimate_risk(
                    method='ledoit_wolf',
                    return_method='exponential'
                )

                # Create optimization problem
                problem = PortfolioProblem(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    problem_type=problem_type,
                    constraints=constraints,
                    current_weights=current_weights,
                    transaction_costs=self.config.transaction_cost,
                    risk_free_rate=self.config.risk_free_rate
                )

                # Optimize portfolio
                new_weights = self._optimize_portfolio(problem, **optimizer_kwargs)

                # Calculate turnover and transaction costs
                trades = np.abs(new_weights - current_weights)
                period_turnover = np.sum(trades)
                period_cost = period_turnover * self.config.transaction_cost

                # Apply slippage
                slippage_cost = period_turnover * self.config.slippage
                period_cost += slippage_cost

                turnover[t] = period_turnover
                transaction_costs[t] = period_cost

                # Update weights
                current_weights = new_weights

            # Calculate portfolio return for the period
            gross_return = np.dot(current_weights, period_return)
            net_return = gross_return - transaction_costs[t]

            portfolio_returns[t] = net_return
            portfolio_values[t] = portfolio_values[t-1] * (1 + net_return)
            portfolio_weights[t] = current_weights

            # Update benchmark if provided
            if self.config.benchmark_weights is not None:
                benchmark_return = np.dot(self.config.benchmark_weights, period_return)
                benchmark_returns[t] = benchmark_return
                benchmark_values[t] = benchmark_values[t-1] * (1 + benchmark_return)

        # Calculate performance metrics
        results = self._calculate_metrics(
            portfolio_values,
            portfolio_returns,
            portfolio_weights,
            turnover,
            transaction_costs,
            benchmark_values,
            benchmark_returns
        )

        return results

    def _optimize_portfolio(self,
                           problem: PortfolioProblem,
                           **kwargs) -> np.ndarray:
        """
        Optimize portfolio weights

        Args:
            problem: Portfolio optimization problem
            **kwargs: Additional optimizer arguments

        Returns:
            Optimized weight vector
        """
        if self.optimizer is not None:
            # Use custom optimizer
            return self.optimizer(problem, **kwargs)

        # Default: Use Librex's optimization
        from Librex import optimize

        adapter = PortfolioAdapter()
        standardized = adapter.encode_problem(problem)

        # Simple optimization with default method
        result = optimize(
            standardized.objective_function,
            bounds=adapter.get_bounds(),
            method=kwargs.get('method', 'scipy'),
            max_iter=kwargs.get('max_iter', 1000)
        )

        # Decode and normalize solution
        weights = adapter._normalize_weights(result['solution'])

        return weights

    def _calculate_metrics(self,
                          portfolio_values: np.ndarray,
                          portfolio_returns: np.ndarray,
                          portfolio_weights: np.ndarray,
                          turnover: np.ndarray,
                          transaction_costs: np.ndarray,
                          benchmark_values: Optional[np.ndarray],
                          benchmark_returns: Optional[np.ndarray]) -> BacktestResults:
        """Calculate performance metrics from backtest results"""

        # Basic performance
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Remove initial zero return
        returns_clean = portfolio_returns[1:]

        # Annualized metrics
        years = self.n_periods / 252  # Assuming daily data
        annualized_return = (1 + total_return) ** (1/years) - 1
        annualized_volatility = np.std(returns_clean) * np.sqrt(252)

        # Risk-adjusted returns
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns_clean[returns_clean < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cum_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = 1 - cum_returns / running_max
        maximum_drawdown = np.max(drawdowns)

        # Calmar ratio
        calmar_ratio = annualized_return / maximum_drawdown if maximum_drawdown > 0 else 0

        # Win/loss statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]

        win_rate = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        average_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        average_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        best_period = np.max(returns_clean) if len(returns_clean) > 0 else 0
        worst_period = np.min(returns_clean) if len(returns_clean) > 0 else 0

        # Transaction statistics
        total_trades = np.sum(turnover > 0)
        average_turnover = np.mean(turnover[turnover > 0]) if total_trades > 0 else 0
        total_transaction_cost = np.sum(transaction_costs) / portfolio_values[0]

        # Benchmark comparison
        tracking_error = None
        information_ratio = None

        if benchmark_returns is not None:
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = np.std(active_returns[1:]) * np.sqrt(252)

            if tracking_error > 0:
                active_return = annualized_return - ((benchmark_values[-1] / benchmark_values[0]) ** (1/years) - 1)
                information_ratio = active_return / tracking_error

        return BacktestResults(
            dates=self.dates,
            portfolio_values=portfolio_values,
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            turnover=turnover,
            transaction_costs=transaction_costs,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            maximum_drawdown=maximum_drawdown,
            calmar_ratio=calmar_ratio,
            benchmark_values=benchmark_values,
            benchmark_returns=benchmark_returns,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            best_period=best_period,
            worst_period=worst_period,
            total_trades=int(total_trades),
            average_turnover=average_turnover,
            total_transaction_cost=total_transaction_cost
        )


def walk_forward_analysis(returns_data: pd.DataFrame,
                         config: BacktestConfig,
                         problem_type: ProblemType,
                         n_splits: int = 5) -> List[BacktestResults]:
    """
    Perform walk-forward analysis

    Split data into multiple training/testing periods

    Args:
        returns_data: Asset returns DataFrame
        config: Backtest configuration
        problem_type: Portfolio optimization type
        n_splits: Number of walk-forward splits

    Returns:
        List of backtest results for each split
    """
    results = []
    n_periods = len(returns_data)
    test_size = n_periods // (n_splits + 1)

    for i in range(n_splits):
        # Define train/test split
        train_end = (i + 1) * test_size + config.lookback_window
        test_end = min(train_end + test_size, n_periods)

        # Run backtest on test period
        test_data = returns_data.iloc[:test_end]

        backtester = PortfolioBacktester(test_data, config)
        result = backtester.run_backtest(problem_type)
        results.append(result)

    return results