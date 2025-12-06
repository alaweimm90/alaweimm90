"""
Portfolio Optimization Problem Definitions

This module defines various portfolio optimization problems including:
- Mean-Variance (Markowitz) Optimization
- Conditional Value at Risk (CVaR) Optimization
- Risk Parity Portfolio
- Black-Litterman Model
- Maximum Sharpe Ratio
- Minimum Variance Portfolio
- Factor-based Portfolio Optimization

Mathematical formulations follow industry-standard conventions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import warnings


class ProblemType(Enum):
    """Portfolio optimization problem types"""
    MEAN_VARIANCE = "mean_variance"
    CVAR = "conditional_value_at_risk"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MAX_SHARPE = "maximum_sharpe"
    MIN_VARIANCE = "minimum_variance"
    FACTOR_MODEL = "factor_model"
    ROBUST = "robust_optimization"


@dataclass
class PortfolioConstraints:
    """
    Portfolio constraints for optimization

    Attributes:
        long_only: If True, only allow positive weights (no short selling)
        sum_to_one: If True, weights must sum to 1 (fully invested)
        max_weight: Maximum weight for any single asset
        min_weight: Minimum weight for any single asset (can be negative for shorts)
        cardinality: Maximum number of non-zero positions
        turnover_limit: Maximum portfolio turnover from initial weights
        sector_limits: Dict mapping sector names to (min, max) exposure limits
        factor_exposures: Constraints on factor exposures
        leverage_limit: Maximum leverage (sum of absolute weights)
        holding_constraints: Asset-specific min/max weights
        group_constraints: Constraints on groups of assets
    """
    long_only: bool = True
    sum_to_one: bool = True
    max_weight: Optional[float] = 1.0
    min_weight: Optional[float] = 0.0
    cardinality: Optional[int] = None
    turnover_limit: Optional[float] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    factor_exposures: Optional[Dict[str, Tuple[float, float]]] = None
    leverage_limit: Optional[float] = None
    holding_constraints: Optional[Dict[int, Tuple[float, float]]] = None
    group_constraints: Optional[List[Dict]] = None

    def __post_init__(self):
        """Validate constraints"""
        if self.long_only:
            if self.min_weight is None:
                self.min_weight = 0.0
            elif self.min_weight < 0:
                warnings.warn("long_only=True but min_weight<0, setting min_weight=0")
                self.min_weight = 0.0

        if self.max_weight is not None and self.min_weight is not None:
            if self.max_weight < self.min_weight:
                raise ValueError(f"max_weight ({self.max_weight}) < min_weight ({self.min_weight})")


@dataclass
class BlackLittermanViews:
    """
    Investor views for Black-Litterman model

    Attributes:
        P: View matrix (k x n) where k is number of views, n is number of assets
        Q: Expected returns for each view (k x 1)
        omega: Uncertainty/confidence in views (k x k diagonal matrix)
    """
    P: np.ndarray
    Q: np.ndarray
    omega: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize omega if not provided"""
        if self.omega is None:
            # Default: proportional to view variance
            k = len(self.Q)
            self.omega = np.eye(k) * 0.01


@dataclass
class FactorModel:
    """
    Factor model specification for portfolio optimization

    Attributes:
        factor_loadings: Asset exposures to factors (n x k matrix)
        factor_returns: Expected factor returns (k x 1)
        factor_covariance: Factor covariance matrix (k x k)
        idiosyncratic_risk: Asset-specific risk (n x 1)
    """
    factor_loadings: np.ndarray
    factor_returns: np.ndarray
    factor_covariance: np.ndarray
    idiosyncratic_risk: np.ndarray

    def get_expected_returns(self) -> np.ndarray:
        """Calculate expected returns from factor model"""
        return self.factor_loadings @ self.factor_returns

    def get_covariance_matrix(self) -> np.ndarray:
        """Calculate asset covariance from factor model"""
        factor_cov = self.factor_loadings @ self.factor_covariance @ self.factor_loadings.T
        idio_cov = np.diag(self.idiosyncratic_risk ** 2)
        return factor_cov + idio_cov


@dataclass
class PortfolioProblem:
    """
    Complete specification of a portfolio optimization problem

    This class encapsulates all data and parameters needed to solve
    various types of portfolio optimization problems.

    Attributes:
        expected_returns: Expected returns vector (n x 1)
        covariance_matrix: Asset covariance matrix (n x n)
        problem_type: Type of optimization problem
        constraints: Portfolio constraints
        risk_aversion: Risk aversion parameter (for mean-variance)
        confidence_level: Confidence level for CVaR (e.g., 0.95)
        returns_scenarios: Historical or simulated returns for scenario-based optimization
        current_weights: Current portfolio weights (for turnover constraints)
        transaction_costs: Transaction cost per unit traded
        market_weights: Market capitalization weights (for Black-Litterman)
        views: Investor views (for Black-Litterman)
        factor_model: Factor model specification
        benchmark_weights: Benchmark portfolio for tracking error
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        regularization: Regularization parameter for robust optimization
        time_horizon: Investment time horizon in periods
        asset_names: Optional asset names/tickers
        sector_mapping: Dict mapping asset indices to sectors
        metadata: Additional problem-specific metadata
    """
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    problem_type: ProblemType = ProblemType.MEAN_VARIANCE
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Problem-specific parameters
    risk_aversion: float = 1.0
    confidence_level: float = 0.95
    returns_scenarios: Optional[np.ndarray] = None
    current_weights: Optional[np.ndarray] = None
    transaction_costs: Union[float, np.ndarray] = 0.0

    # Black-Litterman specific
    market_weights: Optional[np.ndarray] = None
    views: Optional[BlackLittermanViews] = None
    tau: float = 0.05  # Scaling factor for Black-Litterman

    # Factor model
    factor_model: Optional[FactorModel] = None

    # Benchmarking
    benchmark_weights: Optional[np.ndarray] = None
    risk_free_rate: float = 0.0

    # Robust optimization
    regularization: float = 0.0
    uncertainty_set_size: float = 0.1

    # General
    time_horizon: int = 1
    asset_names: Optional[List[str]] = None
    sector_mapping: Optional[Dict[int, str]] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate problem specification"""
        self._validate_inputs()
        self._setup_problem_specific_data()

    def _validate_inputs(self):
        """Validate input data consistency"""
        n_assets = len(self.expected_returns)

        # Check covariance matrix
        if self.covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance matrix shape {self.covariance_matrix.shape} doesn't match "
                           f"number of assets {n_assets}")

        # Check symmetry
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
            warnings.warn("Covariance matrix is not symmetric, symmetrizing...")
            self.covariance_matrix = (self.covariance_matrix + self.covariance_matrix.T) / 2

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.covariance_matrix)
        if eigenvalues.min() < -1e-8:
            warnings.warn(f"Covariance matrix is not positive semi-definite (min eigenvalue: {eigenvalues.min()})")
            # Fix by adding small regularization
            self.covariance_matrix += np.eye(n_assets) * max(-eigenvalues.min() * 1.1, 1e-8)

        # Validate current weights if provided
        if self.current_weights is not None:
            if len(self.current_weights) != n_assets:
                raise ValueError(f"Current weights length {len(self.current_weights)} doesn't match "
                               f"number of assets {n_assets}")

        # Validate scenarios if provided
        if self.returns_scenarios is not None:
            if self.returns_scenarios.shape[1] != n_assets:
                raise ValueError(f"Returns scenarios has {self.returns_scenarios.shape[1]} assets, "
                               f"expected {n_assets}")

        # Validate Black-Litterman inputs
        if self.problem_type == ProblemType.BLACK_LITTERMAN:
            if self.market_weights is None:
                raise ValueError("Black-Litterman requires market_weights")
            if self.views is None:
                warnings.warn("Black-Litterman without views is equivalent to market equilibrium")

    def _setup_problem_specific_data(self):
        """Setup any problem-specific derived data"""
        n_assets = len(self.expected_returns)

        # Initialize current weights if not provided
        if self.current_weights is None:
            self.current_weights = np.zeros(n_assets)

        # Setup transaction costs as vector if scalar
        if isinstance(self.transaction_costs, (int, float)):
            self.transaction_costs = np.ones(n_assets) * self.transaction_costs

        # Generate scenarios from normal distribution if not provided
        if self.returns_scenarios is None and self.problem_type in [ProblemType.CVAR, ProblemType.ROBUST]:
            warnings.warn("Generating synthetic return scenarios from normal distribution")
            n_scenarios = 1000
            self.returns_scenarios = np.random.multivariate_normal(
                self.expected_returns,
                self.covariance_matrix,
                size=n_scenarios
            )

    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio"""
        return len(self.expected_returns)

    @property
    def has_transaction_costs(self) -> bool:
        """Check if transaction costs are specified"""
        return np.any(self.transaction_costs > 0)

    @property
    def is_long_short(self) -> bool:
        """Check if short selling is allowed"""
        return not self.constraints.long_only

    def get_risk_matrix(self) -> np.ndarray:
        """
        Get the risk matrix for optimization

        Returns covariance matrix or factor-based risk if available
        """
        if self.factor_model is not None:
            return self.factor_model.get_covariance_matrix()
        return self.covariance_matrix

    def get_expected_returns(self) -> np.ndarray:
        """
        Get expected returns vector

        Uses factor model or Black-Litterman if applicable
        """
        if self.problem_type == ProblemType.BLACK_LITTERMAN:
            return self._compute_black_litterman_returns()
        elif self.factor_model is not None:
            return self.factor_model.get_expected_returns()
        return self.expected_returns

    def _compute_black_litterman_returns(self) -> np.ndarray:
        """Compute Black-Litterman posterior expected returns"""
        if self.market_weights is None:
            return self.expected_returns

        # Equilibrium returns
        delta = self.risk_aversion
        pi = delta * self.covariance_matrix @ self.market_weights

        if self.views is None:
            return pi

        # Posterior returns with views
        tau_sigma = self.tau * self.covariance_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)

        P = self.views.P
        Q = self.views.Q
        omega_inv = np.linalg.inv(self.views.omega)

        # Black-Litterman formula
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_mean = np.linalg.inv(posterior_precision) @ (
            tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        )

        return posterior_mean

    def validate_weights(self, weights: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate that portfolio weights satisfy all constraints

        Args:
            weights: Portfolio weights vector

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check dimensions
        if len(weights) != self.n_assets:
            violations.append(f"Wrong dimension: {len(weights)} != {self.n_assets}")
            return False, violations

        # Sum to one constraint
        if self.constraints.sum_to_one:
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                violations.append(f"Weights sum to {weight_sum:.6f}, not 1.0")

        # Long-only constraint
        if self.constraints.long_only:
            negative_weights = weights[weights < -1e-8]
            if len(negative_weights) > 0:
                violations.append(f"Short positions not allowed: {negative_weights}")

        # Position limits
        if self.constraints.max_weight is not None:
            over_limit = weights[weights > self.constraints.max_weight + 1e-8]
            if len(over_limit) > 0:
                violations.append(f"Weights exceed max_weight {self.constraints.max_weight}: {over_limit}")

        if self.constraints.min_weight is not None:
            under_limit = weights[weights < self.constraints.min_weight - 1e-8]
            if len(under_limit) > 0:
                violations.append(f"Weights below min_weight {self.constraints.min_weight}: {under_limit}")

        # Cardinality constraint
        if self.constraints.cardinality is not None:
            n_positions = np.sum(np.abs(weights) > 1e-8)
            if n_positions > self.constraints.cardinality:
                violations.append(f"Too many positions: {n_positions} > {self.constraints.cardinality}")

        # Turnover constraint
        if self.constraints.turnover_limit is not None and self.current_weights is not None:
            turnover = np.sum(np.abs(weights - self.current_weights))
            if turnover > self.constraints.turnover_limit + 1e-8:
                violations.append(f"Turnover {turnover:.4f} exceeds limit {self.constraints.turnover_limit}")

        # Leverage constraint
        if self.constraints.leverage_limit is not None:
            leverage = np.sum(np.abs(weights))
            if leverage > self.constraints.leverage_limit + 1e-8:
                violations.append(f"Leverage {leverage:.4f} exceeds limit {self.constraints.leverage_limit}")

        return len(violations) == 0, violations

    def to_dict(self) -> Dict:
        """Convert problem to dictionary for serialization"""
        return {
            'expected_returns': self.expected_returns.tolist(),
            'covariance_matrix': self.covariance_matrix.tolist(),
            'problem_type': self.problem_type.value,
            'risk_aversion': self.risk_aversion,
            'n_assets': self.n_assets,
            'constraints': {
                'long_only': self.constraints.long_only,
                'sum_to_one': self.constraints.sum_to_one,
                'max_weight': self.constraints.max_weight,
                'min_weight': self.constraints.min_weight,
                'cardinality': self.constraints.cardinality
            },
            'metadata': self.metadata
        }