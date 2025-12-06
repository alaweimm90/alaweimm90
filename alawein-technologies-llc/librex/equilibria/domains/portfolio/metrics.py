"""
Portfolio Performance Metrics

Comprehensive metrics for evaluating portfolio performance including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Downside risk measures (Maximum Drawdown, VaR, CVaR)
- Tracking and information metrics
- Portfolio characteristics (diversification, concentration)
"""

import numpy as np
from typing import Optional, Dict, Tuple, Union
import warnings
from scipy import stats


class PortfolioMetrics:
    """
    Comprehensive portfolio performance metrics calculator

    Provides both ex-ante (forward-looking) and ex-post (historical) metrics
    """

    def __init__(self,
                 weights: np.ndarray,
                 returns: np.ndarray,
                 covariance: Optional[np.ndarray] = None,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252):
        """
        Initialize portfolio metrics calculator

        Args:
            weights: Portfolio weights vector
            returns: Either expected returns (1D) or historical returns (2D)
            covariance: Covariance matrix (optional, computed if returns is 2D)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        """
        self.weights = weights
        self.n_assets = len(weights)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Determine if returns are historical (2D) or expected (1D)
        if returns.ndim == 2:
            self.historical_returns = returns
            self.expected_returns = np.mean(returns, axis=0)
            if covariance is None:
                self.covariance = np.cov(returns.T)
            else:
                self.covariance = covariance
        else:
            self.historical_returns = None
            self.expected_returns = returns
            if covariance is None:
                raise ValueError("Covariance matrix required when using expected returns")
            self.covariance = covariance

        # Compute portfolio returns if historical data available
        if self.historical_returns is not None:
            self.portfolio_returns = self.historical_returns @ self.weights
        else:
            self.portfolio_returns = None

    # ===== Basic Metrics =====

    def expected_return(self, annualized: bool = True) -> float:
        """
        Calculate expected portfolio return

        Args:
            annualized: If True, annualize the return

        Returns:
            Expected portfolio return
        """
        ret = np.dot(self.weights, self.expected_returns)
        if annualized:
            ret *= self.periods_per_year
        return ret

    def portfolio_variance(self) -> float:
        """Calculate portfolio variance"""
        return self.weights @ self.covariance @ self.weights

    def portfolio_volatility(self, annualized: bool = True) -> float:
        """
        Calculate portfolio standard deviation

        Args:
            annualized: If True, annualize the volatility

        Returns:
            Portfolio volatility
        """
        vol = np.sqrt(self.portfolio_variance())
        if annualized:
            vol *= np.sqrt(self.periods_per_year)
        return vol

    # ===== Risk-Adjusted Returns =====

    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio

        SR = (E[R] - R_f) / σ

        Returns:
            Annualized Sharpe ratio
        """
        excess_return = self.expected_return(annualized=True) - self.risk_free_rate
        volatility = self.portfolio_volatility(annualized=True)

        if volatility < 1e-10:
            return 0.0

        return excess_return / volatility

    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation)

        Sortino = (E[R] - Target) / DownsideDeviation

        Args:
            target_return: Minimum acceptable return (annualized)

        Returns:
            Sortino ratio
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for Sortino ratio, using Sharpe instead")
            return self.sharpe_ratio()

        annualized_return = self.expected_return(annualized=True)
        excess_return = annualized_return - target_return

        # Downside returns (below target)
        daily_target = target_return / self.periods_per_year
        downside_returns = self.portfolio_returns[self.portfolio_returns < daily_target] - daily_target

        if len(downside_returns) == 0:
            return np.inf  # No downside risk

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.periods_per_year)

        if downside_deviation < 1e-10:
            return np.inf

        return excess_return / downside_deviation

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio

        Calmar = Annualized Return / Maximum Drawdown

        Returns:
            Calmar ratio
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for Calmar ratio")
            return 0.0

        annualized_return = self.expected_return(annualized=True)
        max_dd = self.maximum_drawdown()

        if max_dd < 1e-10:
            return np.inf

        return annualized_return / max_dd

    def information_ratio(self, benchmark_weights: np.ndarray) -> float:
        """
        Calculate Information Ratio vs benchmark

        IR = Active Return / Tracking Error

        Args:
            benchmark_weights: Benchmark portfolio weights

        Returns:
            Information ratio
        """
        # Active weights
        active_weights = self.weights - benchmark_weights

        # Active return
        active_return = np.dot(active_weights, self.expected_returns) * self.periods_per_year

        # Tracking error (annualized)
        tracking_var = active_weights @ self.covariance @ active_weights
        tracking_error = np.sqrt(tracking_var) * np.sqrt(self.periods_per_year)

        if tracking_error < 1e-10:
            return 0.0

        return active_return / tracking_error

    # ===== Downside Risk Measures =====

    def maximum_drawdown(self) -> float:
        """
        Calculate Maximum Drawdown

        MDD = max(1 - P_t / max(P_s for s <= t))

        Returns:
            Maximum drawdown (as positive fraction)
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for maximum drawdown")
            return 0.0

        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + self.portfolio_returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Calculate drawdowns
        drawdowns = 1 - cum_returns / running_max

        return np.max(drawdowns)

    def value_at_risk(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        Calculate Value at Risk

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish-fisher'

        Returns:
            VaR (as positive loss)
        """
        if method == 'historical':
            if self.portfolio_returns is None:
                warnings.warn("Historical returns needed, using parametric VaR")
                method = 'parametric'
            else:
                return -np.percentile(self.portfolio_returns, (1 - confidence) * 100)

        if method == 'parametric':
            # Assume normal distribution
            mean = self.expected_return(annualized=False)
            std = self.portfolio_volatility(annualized=False)
            z_score = stats.norm.ppf(1 - confidence)
            return -(mean + z_score * std)

        elif method == 'cornish-fisher':
            # Cornish-Fisher expansion for non-normal distributions
            if self.portfolio_returns is None:
                return self.value_at_risk(confidence, 'parametric')

            mean = np.mean(self.portfolio_returns)
            std = np.std(self.portfolio_returns)
            skew = stats.skew(self.portfolio_returns)
            kurt = stats.kurtosis(self.portfolio_returns)

            z = stats.norm.ppf(1 - confidence)
            cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36

            return -(mean + cf_z * std)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def conditional_value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        CVaR = E[L | L > VaR]

        Args:
            confidence: Confidence level

        Returns:
            CVaR (as positive loss)
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for CVaR")
            # Parametric approximation
            var = self.value_at_risk(confidence, 'parametric')
            std = self.portfolio_volatility(annualized=False)
            # For normal distribution: CVaR = μ - σ * φ(z) / (1-Φ(z))
            z = stats.norm.ppf(confidence)
            return var + std * stats.norm.pdf(z) / (1 - confidence)

        var = self.value_at_risk(confidence, 'historical')
        # Returns worse than VaR
        tail_returns = self.portfolio_returns[self.portfolio_returns <= -var]

        if len(tail_returns) == 0:
            return var

        return -np.mean(tail_returns)

    # ===== Portfolio Characteristics =====

    def effective_number_of_assets(self) -> float:
        """
        Calculate Effective Number of Assets (ENB)

        ENB = 1 / Σw_i^2

        Measures portfolio diversification
        """
        return 1 / np.sum(self.weights ** 2)

    def concentration_ratio(self, top_n: int = 5) -> float:
        """
        Calculate concentration in top N assets

        Args:
            top_n: Number of top positions to consider

        Returns:
            Sum of top N weights
        """
        sorted_weights = np.sort(np.abs(self.weights))[::-1]
        return np.sum(sorted_weights[:top_n])

    def diversification_ratio(self) -> float:
        """
        Calculate Diversification Ratio

        DR = Weighted average volatility / Portfolio volatility

        Higher values indicate better diversification
        """
        individual_vols = np.sqrt(np.diag(self.covariance))
        weighted_avg_vol = np.dot(np.abs(self.weights), individual_vols)
        portfolio_vol = self.portfolio_volatility(annualized=False)

        if portfolio_vol < 1e-10:
            return 1.0

        return weighted_avg_vol / portfolio_vol

    def risk_contributions(self) -> np.ndarray:
        """
        Calculate marginal risk contributions for each asset

        RC_i = w_i * (∂σ/∂w_i) = w_i * (Σw)_i / σ

        Returns:
            Risk contribution vector
        """
        portfolio_vol = self.portfolio_volatility(annualized=False)

        if portfolio_vol < 1e-10:
            return self.weights / self.n_assets

        marginal_contrib = self.covariance @ self.weights / portfolio_vol
        risk_contrib = self.weights * marginal_contrib

        return risk_contrib

    def portfolio_beta(self, market_weights: np.ndarray) -> float:
        """
        Calculate portfolio beta vs market

        β = Cov(R_p, R_m) / Var(R_m)

        Args:
            market_weights: Market portfolio weights

        Returns:
            Portfolio beta
        """
        # Market variance
        market_var = market_weights @ self.covariance @ market_weights

        if market_var < 1e-10:
            return 1.0

        # Covariance with market
        portfolio_market_cov = self.weights @ self.covariance @ market_weights

        return portfolio_market_cov / market_var

    def tracking_error(self, benchmark_weights: np.ndarray, annualized: bool = True) -> float:
        """
        Calculate tracking error vs benchmark

        TE = σ(R_p - R_b)

        Args:
            benchmark_weights: Benchmark portfolio weights
            annualized: If True, annualize the tracking error

        Returns:
            Tracking error
        """
        active_weights = self.weights - benchmark_weights
        tracking_var = active_weights @ self.covariance @ active_weights
        te = np.sqrt(tracking_var)

        if annualized:
            te *= np.sqrt(self.periods_per_year)

        return te

    # ===== Statistical Measures =====

    def skewness(self) -> float:
        """Calculate portfolio return skewness"""
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for skewness")
            return 0.0

        return stats.skew(self.portfolio_returns)

    def kurtosis(self, excess: bool = True) -> float:
        """
        Calculate portfolio return kurtosis

        Args:
            excess: If True, return excess kurtosis (kurtosis - 3)

        Returns:
            Portfolio kurtosis
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for kurtosis")
            return 0.0 if excess else 3.0

        kurt = stats.kurtosis(self.portfolio_returns, fisher=excess)
        return kurt

    def tail_ratio(self, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio (right tail / left tail)

        Args:
            percentile: Percentile for tail definition

        Returns:
            Ratio of right tail to left tail magnitude
        """
        if self.portfolio_returns is None:
            warnings.warn("Historical returns needed for tail ratio")
            return 1.0

        right_tail = np.percentile(self.portfolio_returns, 100 * (1 - percentile))
        left_tail = np.percentile(self.portfolio_returns, 100 * percentile)

        if abs(left_tail) < 1e-10:
            return np.inf

        return abs(right_tail / left_tail)

    # ===== Comprehensive Summary =====

    def summary(self, benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Generate comprehensive metrics summary

        Args:
            benchmark_weights: Optional benchmark for relative metrics

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            # Returns
            'expected_return': self.expected_return(),
            'portfolio_volatility': self.portfolio_volatility(),

            # Risk-adjusted
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),

            # Downside risk
            'max_drawdown': self.maximum_drawdown(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_value_at_risk(0.95),

            # Diversification
            'effective_n_assets': self.effective_number_of_assets(),
            'concentration_top5': self.concentration_ratio(5),
            'diversification_ratio': self.diversification_ratio(),

            # Statistical
            'skewness': self.skewness(),
            'excess_kurtosis': self.kurtosis(excess=True),
        }

        # Add Calmar if we have historical returns
        if self.portfolio_returns is not None:
            metrics['calmar_ratio'] = self.calmar_ratio()

        # Add benchmark-relative metrics if provided
        if benchmark_weights is not None:
            metrics['tracking_error'] = self.tracking_error(benchmark_weights)
            metrics['information_ratio'] = self.information_ratio(benchmark_weights)
            metrics['portfolio_beta'] = self.portfolio_beta(benchmark_weights)

        return metrics

    def risk_report(self) -> str:
        """
        Generate formatted risk report

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO RISK METRICS REPORT")
        report.append("=" * 60)

        summary = self.summary()

        report.append("\nRETURN METRICS:")
        report.append(f"  Expected Return (Annual):  {summary['expected_return']:.2%}")
        report.append(f"  Volatility (Annual):       {summary['portfolio_volatility']:.2%}")

        report.append("\nRISK-ADJUSTED RETURNS:")
        report.append(f"  Sharpe Ratio:             {summary['sharpe_ratio']:.3f}")
        report.append(f"  Sortino Ratio:            {summary['sortino_ratio']:.3f}")

        if 'calmar_ratio' in summary:
            report.append(f"  Calmar Ratio:             {summary['calmar_ratio']:.3f}")

        report.append("\nDOWNSIDE RISK:")
        report.append(f"  Maximum Drawdown:         {summary['max_drawdown']:.2%}")
        report.append(f"  Value at Risk (95%):      {summary['var_95']:.2%}")
        report.append(f"  CVaR (95%):               {summary['cvar_95']:.2%}")

        report.append("\nPORTFOLIO CHARACTERISTICS:")
        report.append(f"  Effective # Assets:       {summary['effective_n_assets']:.1f}")
        report.append(f"  Top 5 Concentration:      {summary['concentration_top5']:.2%}")
        report.append(f"  Diversification Ratio:    {summary['diversification_ratio']:.3f}")

        report.append("\nDISTRIBUTION:")
        report.append(f"  Skewness:                 {summary['skewness']:.3f}")
        report.append(f"  Excess Kurtosis:          {summary['excess_kurtosis']:.3f}")

        report.append("=" * 60)

        return "\n".join(report)