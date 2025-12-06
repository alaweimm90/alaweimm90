"""
Comprehensive tests for Portfolio Optimization Domain

Tests all components including:
- Portfolio problems
- Adapter interface
- Objectives
- Metrics
- Risk models
- Backtesting
- Data loaders
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from Librex.domains.portfolio import (
    PortfolioProblem,
    ProblemType,
    PortfolioConstraints,
    PortfolioAdapter,
    PortfolioMetrics,
    RiskEstimator,
    PortfolioBacktester,
    BacktestConfig,
    DataLoader
)
try:
    from Librex.domains.portfolio.objectives import (
        mean_variance_objective,
        risk_parity_objective,
        cvar_objective,
        maximum_sharpe_objective,
        minimum_variance_objective
    )
except ImportError:
    # If scikit-learn not installed, skip these imports
    mean_variance_objective = None
    risk_parity_objective = None
    cvar_objective = None
    maximum_sharpe_objective = None
    minimum_variance_objective = None

from Librex.core.interfaces import StandardizedSolution


class TestPortfolioProblem:
    """Test PortfolioProblem class"""

    def test_problem_initialization(self):
        """Test basic problem initialization"""
        n_assets = 5
        expected_returns = np.array([0.05, 0.07, 0.06, 0.08, 0.04])
        cov_matrix = np.eye(n_assets) * 0.01

        problem = PortfolioProblem(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            problem_type=ProblemType.MEAN_VARIANCE
        )

        assert problem.n_assets == n_assets
        assert np.array_equal(problem.expected_returns, expected_returns)
        assert np.array_equal(problem.covariance_matrix, cov_matrix)

    def test_constraint_validation(self):
        """Test constraint validation"""
        constraints = PortfolioConstraints(
            long_only=True,
            sum_to_one=True,
            max_weight=0.3,
            cardinality=3
        )

        assert constraints.long_only
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 0.3

    def test_weight_validation(self):
        """Test weight validation method"""
        n_assets = 3
        problem = PortfolioProblem(
            expected_returns=np.ones(n_assets) * 0.05,
            covariance_matrix=np.eye(n_assets) * 0.01,
            constraints=PortfolioConstraints(long_only=True, sum_to_one=True)
        )

        # Valid weights
        valid_weights = np.array([0.3, 0.5, 0.2])
        is_valid, violations = problem.validate_weights(valid_weights)
        assert is_valid
        assert len(violations) == 0

        # Invalid weights (don't sum to 1)
        invalid_weights = np.array([0.3, 0.3, 0.3])
        is_valid, violations = problem.validate_weights(invalid_weights)
        assert not is_valid
        assert len(violations) > 0

        # Invalid weights (negative)
        invalid_weights = np.array([0.5, 0.7, -0.2])
        is_valid, violations = problem.validate_weights(invalid_weights)
        assert not is_valid
        assert any("Short positions" in v for v in violations)


class TestPortfolioAdapter:
    """Test PortfolioAdapter class"""

    def test_adapter_encoding(self):
        """Test problem encoding"""
        n_assets = 4
        problem = PortfolioProblem(
            expected_returns=np.ones(n_assets) * 0.05,
            covariance_matrix=np.eye(n_assets) * 0.01
        )

        adapter = PortfolioAdapter()
        standardized = adapter.encode_problem(problem)

        assert standardized.dimension == n_assets
        assert standardized.objective_function is not None
        assert callable(standardized.objective_function)

    def test_solution_decoding(self):
        """Test solution decoding"""
        n_assets = 4
        problem = PortfolioProblem(
            expected_returns=np.array([0.05, 0.07, 0.06, 0.08]),
            covariance_matrix=np.eye(n_assets) * 0.01
        )

        adapter = PortfolioAdapter()
        adapter.encode_problem(problem)

        # Create solution
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        solution = StandardizedSolution(
            vector=weights,
            objective_value=0.01,
            is_valid=True
        )

        decoded = adapter.decode_solution(solution)

        assert 'weights' in decoded
        assert 'expected_return' in decoded
        assert 'sharpe_ratio' in decoded
        assert np.allclose(decoded['weights'].sum(), 1.0)

    def test_objective_computation(self):
        """Test objective function computation"""
        n_assets = 3
        problem = PortfolioProblem(
            expected_returns=np.array([0.05, 0.07, 0.06]),
            covariance_matrix=np.eye(n_assets) * 0.01,
            risk_aversion=2.0
        )

        adapter = PortfolioAdapter()
        adapter.encode_problem(problem)

        weights = np.array([0.3, 0.5, 0.2])
        obj_value = adapter.compute_objective(weights)

        assert isinstance(obj_value, float)
        assert not np.isnan(obj_value)


class TestObjectiveFunctions:
    """Test portfolio objective functions"""

    def setup_method(self):
        """Setup test data"""
        self.n_assets = 4
        self.problem = PortfolioProblem(
            expected_returns=np.array([0.05, 0.07, 0.06, 0.08]),
            covariance_matrix=np.eye(self.n_assets) * 0.01,
            risk_aversion=2.0,
            risk_free_rate=0.02
        )
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def test_mean_variance_objective(self):
        """Test mean-variance objective"""
        obj = mean_variance_objective(self.weights, self.problem)
        assert isinstance(obj, float)
        assert not np.isnan(obj)

    def test_risk_parity_objective(self):
        """Test risk parity objective"""
        obj = risk_parity_objective(self.weights, self.problem)
        assert isinstance(obj, float)
        assert not np.isnan(obj)
        assert obj >= 0  # Should be positive (squared deviations)

    def test_maximum_sharpe_objective(self):
        """Test maximum Sharpe ratio objective"""
        obj = maximum_sharpe_objective(self.weights, self.problem)
        assert isinstance(obj, float)
        assert not np.isnan(obj)

    def test_minimum_variance_objective(self):
        """Test minimum variance objective"""
        obj = minimum_variance_objective(self.weights, self.problem)
        assert isinstance(obj, float)
        assert not np.isnan(obj)
        assert obj >= 0  # Variance should be non-negative

    def test_cvar_objective(self):
        """Test CVaR objective"""
        # Add return scenarios
        n_scenarios = 100
        self.problem.returns_scenarios = np.random.randn(n_scenarios, self.n_assets) * 0.1

        obj = cvar_objective(self.weights, self.problem)
        assert isinstance(obj, float)
        assert not np.isnan(obj)


class TestPortfolioMetrics:
    """Test portfolio metrics calculations"""

    def setup_method(self):
        """Setup test data"""
        self.n_assets = 4
        self.n_periods = 252
        self.weights = np.array([0.3, 0.3, 0.2, 0.2])

        # Generate synthetic returns
        self.returns = np.random.randn(self.n_periods, self.n_assets) * 0.01 + 0.0003
        self.cov_matrix = np.cov(self.returns.T)

    def test_basic_metrics(self):
        """Test basic portfolio metrics"""
        metrics = PortfolioMetrics(
            weights=self.weights,
            returns=self.returns,
            covariance=self.cov_matrix
        )

        # Test expected return
        exp_return = metrics.expected_return()
        assert isinstance(exp_return, float)
        assert not np.isnan(exp_return)

        # Test volatility
        vol = metrics.portfolio_volatility()
        assert isinstance(vol, float)
        assert vol > 0

        # Test Sharpe ratio
        sharpe = metrics.sharpe_ratio()
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_risk_metrics(self):
        """Test risk metrics"""
        metrics = PortfolioMetrics(
            weights=self.weights,
            returns=self.returns,
            covariance=self.cov_matrix
        )

        # Test VaR
        var_95 = metrics.value_at_risk(0.95)
        assert isinstance(var_95, float)
        assert var_95 >= 0  # VaR is positive loss

        # Test CVaR
        cvar_95 = metrics.conditional_value_at_risk(0.95)
        assert isinstance(cvar_95, float)
        assert cvar_95 >= var_95  # CVaR >= VaR

        # Test maximum drawdown
        max_dd = metrics.maximum_drawdown()
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1

    def test_diversification_metrics(self):
        """Test diversification metrics"""
        metrics = PortfolioMetrics(
            weights=self.weights,
            returns=self.returns,
            covariance=self.cov_matrix
        )

        # Test effective number of assets
        eff_n = metrics.effective_number_of_assets()
        assert 1 <= eff_n <= self.n_assets

        # Test concentration ratio
        conc = metrics.concentration_ratio(2)
        assert 0 <= conc <= 1

        # Test diversification ratio
        div_ratio = metrics.diversification_ratio()
        assert div_ratio >= 1


class TestRiskEstimator:
    """Test risk estimation methods"""

    def setup_method(self):
        """Setup test data"""
        self.n_periods = 252
        self.n_assets = 5
        self.returns = np.random.randn(self.n_periods, self.n_assets) * 0.01 + 0.0002

    def test_sample_covariance(self):
        """Test sample covariance estimation"""
        estimator = RiskEstimator(self.returns)
        cov = estimator.sample_covariance()

        assert cov.shape == (self.n_assets, self.n_assets)
        assert np.allclose(cov, cov.T)  # Symmetric
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Positive semi-definite

    def test_ledoit_wolf_shrinkage(self):
        """Test Ledoit-Wolf shrinkage"""
        estimator = RiskEstimator(self.returns)
        cov, shrinkage = estimator.ledoit_wolf_shrinkage()

        assert cov.shape == (self.n_assets, self.n_assets)
        assert 0 <= shrinkage <= 1
        assert np.allclose(cov, cov.T)

    def test_exponentially_weighted_covariance(self):
        """Test exponentially weighted covariance"""
        estimator = RiskEstimator(self.returns)
        cov = estimator.exponentially_weighted_covariance(halflife=60)

        assert cov.shape == (self.n_assets, self.n_assets)
        assert np.allclose(cov, cov.T)

    def test_return_estimation(self):
        """Test return estimation methods"""
        estimator = RiskEstimator(self.returns)

        # Historical mean
        hist_mean = estimator.historical_mean()
        assert hist_mean.shape == (self.n_assets,)

        # James-Stein
        js_mean = estimator.james_stein_mean()
        assert js_mean.shape == (self.n_assets,)

        # Exponentially weighted
        ew_mean = estimator.exponentially_weighted_mean()
        assert ew_mean.shape == (self.n_assets,)


class TestBacktesting:
    """Test backtesting framework"""

    def setup_method(self):
        """Setup test data"""
        # Generate synthetic returns data
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        self.returns_data = pd.DataFrame(
            np.random.randn(1000, 5) * 0.01 + 0.0002,
            index=dates,
            columns=[f'Asset_{i}' for i in range(5)]
        )

    def test_backtest_config(self):
        """Test backtest configuration"""
        config = BacktestConfig(
            lookback_window=252,
            rebalance_frequency='monthly',
            transaction_cost=0.001
        )

        assert config.lookback_window == 252
        assert config.rebalance_frequency == 'monthly'
        assert config.transaction_cost == 0.001

    def test_basic_backtest(self):
        """Test basic backtesting functionality"""
        config = BacktestConfig(
            lookback_window=100,
            rebalance_frequency='quarterly',
            min_history=50
        )

        backtester = PortfolioBacktester(self.returns_data, config)
        results = backtester.run_backtest(
            problem_type=ProblemType.MEAN_VARIANCE,
            constraints=PortfolioConstraints(long_only=True)
        )

        assert results is not None
        assert len(results.portfolio_values) == len(self.returns_data)
        assert results.total_return is not None
        assert results.sharpe_ratio is not None
        assert results.maximum_drawdown >= 0

    def test_backtest_results(self):
        """Test backtest results structure"""
        config = BacktestConfig()
        backtester = PortfolioBacktester(self.returns_data, config)
        results = backtester.run_backtest()

        # Check DataFrame conversion
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'portfolio_value' in df.columns
        assert 'portfolio_return' in df.columns

        # Check summary report
        report = results.summary_report()
        assert isinstance(report, str)
        assert 'PORTFOLIO BACKTEST RESULTS' in report


class TestDataLoader:
    """Test data loading functionality"""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        loader = DataLoader()
        prices = loader.generate_synthetic_prices(
            n_assets=5,
            n_periods=252,
            annual_return=0.08,
            annual_volatility=0.15
        )

        assert prices.shape == (252, 5)
        assert isinstance(prices, pd.DataFrame)
        assert not prices.isnull().any().any()

    def test_price_to_returns_conversion(self):
        """Test price to returns conversion"""
        loader = DataLoader()
        prices = loader.generate_synthetic_prices(n_assets=3, n_periods=100)

        # Simple returns
        simple_returns = loader.prices_to_returns(prices, method='simple')
        assert simple_returns.shape[0] == prices.shape[0] - 1

        # Log returns
        log_returns = loader.prices_to_returns(prices, method='log')
        assert log_returns.shape[0] == prices.shape[0] - 1

    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        loader = DataLoader()

        # Create data with missing values
        prices = pd.DataFrame(
            np.random.randn(100, 3) * 10 + 100,
            columns=['A', 'B', 'C']
        )
        prices.iloc[10:15, 0] = np.nan
        prices.iloc[20:22, 1] = np.nan

        cleaned = loader.clean_price_data(prices, method='forward_fill')
        assert not cleaned.isnull().any().any()

    def test_market_scenarios(self):
        """Test market scenario generation"""
        loader = DataLoader()
        returns = pd.DataFrame(np.random.randn(252, 4) * 0.01)

        scenarios = loader.create_market_scenarios(
            base_returns=returns,
            n_scenarios=300,
            scenario_types=['normal', 'crisis', 'boom']
        )

        assert scenarios.shape == (300, 4)
        assert not np.isnan(scenarios).any()


# Integration test
class TestIntegration:
    """Integration tests for complete workflow"""

    def test_complete_optimization_workflow(self):
        """Test complete portfolio optimization workflow"""
        # 1. Generate data
        loader = DataLoader()
        prices = loader.generate_synthetic_prices(n_assets=5, n_periods=500)
        returns = loader.prices_to_returns(prices)

        # 2. Estimate risk
        risk_estimator = RiskEstimator(returns.values)
        expected_returns, cov_matrix = risk_estimator.estimate_risk()

        # 3. Create problem
        problem = PortfolioProblem(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            problem_type=ProblemType.MEAN_VARIANCE,
            constraints=PortfolioConstraints(
                long_only=True,
                max_weight=0.4
            )
        )

        # 4. Create adapter
        adapter = PortfolioAdapter()
        standardized = adapter.encode_problem(problem)

        # 5. Optimize (simulate)
        weights = np.ones(5) / 5  # Equal weights
        obj_value = adapter.compute_objective(weights)

        # 6. Decode solution
        solution = StandardizedSolution(
            vector=weights,
            objective_value=obj_value,
            is_valid=True
        )
        decoded = adapter.decode_solution(solution)

        # 7. Calculate metrics
        metrics = PortfolioMetrics(
            weights=decoded['weights'],
            returns=expected_returns,
            covariance=cov_matrix
        )
        sharpe = metrics.sharpe_ratio()

        # Verify everything worked
        assert decoded['weights'].sum() == pytest.approx(1.0)
        assert decoded['expected_return'] is not None
        assert sharpe is not None

    def test_backtest_with_real_optimization(self):
        """Test backtesting with actual optimization"""
        # Generate larger dataset
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        returns_data = pd.DataFrame(
            np.random.randn(500, 4) * 0.01 + 0.0003,
            index=dates,
            columns=['A', 'B', 'C', 'D']
        )

        # Configure backtest
        config = BacktestConfig(
            lookback_window=100,
            rebalance_frequency='monthly',
            min_history=50,
            transaction_cost=0.001
        )

        # Run backtest
        backtester = PortfolioBacktester(returns_data, config)
        results = backtester.run_backtest(
            problem_type=ProblemType.MIN_VARIANCE,
            constraints=PortfolioConstraints(long_only=True, max_weight=0.5)
        )

        # Verify results
        assert results.total_return is not None
        assert results.sharpe_ratio is not None
        assert results.portfolio_weights.shape == (500, 4)
        assert np.all(results.portfolio_weights >= -1e-6)  # Long only
        assert np.all(results.portfolio_weights <= 0.5 + 1e-6)  # Max weight