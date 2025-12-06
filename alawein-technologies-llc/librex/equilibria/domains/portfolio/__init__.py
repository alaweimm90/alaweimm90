"""
Portfolio Optimization Domain Adapter for Librex

Comprehensive financial portfolio optimization framework supporting:
- Mean-Variance (Markowitz) Optimization
- Conditional Value at Risk (CVaR) Optimization
- Risk Parity Portfolio Construction
- Black-Litterman Model
- Maximum Sharpe Ratio Portfolio
- Minimum Variance Portfolio
- Factor-based Portfolio Optimization

Author: Librex Team
License: Apache 2.0
"""

from Librex.domains.portfolio.portfolio_adapter import PortfolioAdapter
from Librex.domains.portfolio.portfolio_problem import (
    PortfolioProblem,
    ProblemType,
    PortfolioConstraints
)
from Librex.domains.portfolio.metrics import PortfolioMetrics
from Librex.domains.portfolio.risk_models import RiskEstimator
from Librex.domains.portfolio.objectives import (
    mean_variance_objective,
    risk_parity_objective,
    cvar_objective,
    maximum_sharpe_objective,
    minimum_variance_objective
)
from Librex.domains.portfolio.backtesting import PortfolioBacktester, BacktestConfig, BacktestResults
from Librex.domains.portfolio.data_loaders import DataLoader

__all__ = [
    "PortfolioAdapter",
    "PortfolioProblem",
    "ProblemType",
    "PortfolioConstraints",
    "PortfolioMetrics",
    "RiskEstimator",
    "mean_variance_objective",
    "risk_parity_objective",
    "cvar_objective",
    "maximum_sharpe_objective",
    "minimum_variance_objective",
    "PortfolioBacktester",
    "BacktestConfig",
    "BacktestResults",
    "DataLoader"
]