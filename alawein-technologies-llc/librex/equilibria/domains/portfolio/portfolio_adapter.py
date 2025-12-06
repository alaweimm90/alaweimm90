"""
Portfolio Optimization Domain Adapter

Implements the UniversalOptimizationInterface for portfolio optimization problems,
enabling integration with Librex's optimization algorithms.
"""

from typing import Any, Dict, Optional, Union, Callable
import numpy as np
import warnings

from Librex.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
    ValidationResult
)
from Librex.domains.portfolio.portfolio_problem import (
    PortfolioProblem,
    ProblemType,
    PortfolioConstraints
)
from Librex.domains.portfolio.objectives import (
    mean_variance_objective,
    risk_parity_objective,
    cvar_objective,
    maximum_sharpe_objective,
    minimum_variance_objective,
    black_litterman_objective,
    factor_model_objective,
    robust_objective
)


class PortfolioAdapter(UniversalOptimizationInterface):
    """
    Adapter for converting portfolio optimization problems to Librex's standardized format

    This adapter handles the conversion between portfolio-specific problem representations
    and Librex's universal optimization interface, enabling portfolio problems to be
    solved using any of Librex's optimization algorithms.
    """

    def __init__(self):
        """Initialize portfolio adapter"""
        super().__init__({
            'name': 'portfolio',
            'type': 'continuous',
            'constraints': ['linear', 'bound', 'cardinality'],
            'objectives': ['minimize', 'maximize']
        })
        self.problem: Optional[PortfolioProblem] = None
        self.objective_function: Optional[Callable] = None
        self.constraint_matrix: Optional[np.ndarray] = None
        self.constraint_bounds: Optional[np.ndarray] = None

    def encode_problem(self, problem: PortfolioProblem) -> StandardizedProblem:
        """
        Convert portfolio problem to standardized optimization format

        Args:
            problem: PortfolioProblem instance

        Returns:
            StandardizedProblem for use with optimization algorithms
        """
        self.problem = problem

        # Select objective function based on problem type
        self.objective_function = self._get_objective_function(problem.problem_type)

        # Build constraint matrices
        self._build_constraints()

        # Create standardized problem
        standardized = StandardizedProblem(
            dimension=problem.n_assets,
            objective_matrix=None,  # Portfolio objectives are nonlinear
            objective_function=self._wrapped_objective,
            constraint_matrix=self.constraint_matrix,
            problem_metadata={
                'type': 'portfolio_optimization',
                'problem_subtype': problem.problem_type.value,
                'n_assets': problem.n_assets,
                'has_transaction_costs': problem.has_transaction_costs,
                'is_long_short': problem.is_long_short,
                'constraints': self._get_constraint_summary()
            }
        )

        return standardized

    def decode_solution(self, solution: StandardizedSolution) -> Dict[str, Any]:
        """
        Convert standardized solution to portfolio-specific format

        Args:
            solution: StandardizedSolution from optimizer

        Returns:
            Dictionary containing portfolio weights and metrics
        """
        if self.problem is None:
            raise ValueError("No problem has been encoded yet")

        weights = self._normalize_weights(solution.vector)

        # Compute portfolio metrics
        expected_return = np.dot(weights, self.problem.get_expected_returns())
        portfolio_variance = weights @ self.problem.get_risk_matrix() @ weights
        portfolio_std = np.sqrt(portfolio_variance)

        # Sharpe ratio
        sharpe = (expected_return - self.problem.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        # Compute turnover if current weights provided
        turnover = 0.0
        if self.problem.current_weights is not None:
            turnover = np.sum(np.abs(weights - self.problem.current_weights))

        # Transaction costs
        transaction_cost = 0.0
        if self.problem.has_transaction_costs and self.problem.current_weights is not None:
            trades = np.abs(weights - self.problem.current_weights)
            transaction_cost = np.dot(trades, self.problem.transaction_costs)

        # Number of positions
        n_positions = np.sum(np.abs(weights) > 1e-8)

        # Compute risk contributions
        marginal_risk = self.problem.get_risk_matrix() @ weights
        portfolio_risk = np.sqrt(portfolio_variance)
        risk_contributions = weights * marginal_risk / portfolio_risk if portfolio_risk > 0 else weights * 0

        result = {
            'weights': weights,
            'expected_return': expected_return,
            'portfolio_variance': portfolio_variance,
            'portfolio_std': portfolio_std,
            'sharpe_ratio': sharpe,
            'objective_value': solution.objective_value,
            'turnover': turnover,
            'transaction_cost': transaction_cost,
            'n_positions': n_positions,
            'risk_contributions': risk_contributions,
            'is_valid': solution.is_valid
        }

        # Add asset names if available
        if self.problem.asset_names is not None:
            result['asset_allocation'] = {
                name: weight for name, weight in zip(self.problem.asset_names, weights)
            }

        return result

    def validate_solution(self, weights: np.ndarray) -> ValidationResult:
        """
        Validate portfolio weights against all constraints

        Args:
            weights: Portfolio weight vector

        Returns:
            ValidationResult with constraint violation details
        """
        if self.problem is None:
            raise ValueError("No problem has been encoded yet")

        is_valid, violations = self.problem.validate_weights(weights)

        # Calculate violation magnitudes
        magnitudes = []
        for violation in violations:
            if "sum to" in violation:
                magnitudes.append(abs(np.sum(weights) - 1.0))
            elif "Short positions" in violation:
                magnitudes.append(float(np.sum(np.maximum(-weights, 0))))
            elif "exceed max_weight" in violation:
                magnitudes.append(float(np.sum(np.maximum(weights - self.problem.constraints.max_weight, 0))))
            elif "below min_weight" in violation:
                magnitudes.append(float(np.sum(np.maximum(self.problem.constraints.min_weight - weights, 0))))
            elif "Too many positions" in violation:
                n_positions = np.sum(np.abs(weights) > 1e-8)
                magnitudes.append(float(n_positions - self.problem.constraints.cardinality))
            elif "Turnover" in violation:
                turnover = np.sum(np.abs(weights - self.problem.current_weights))
                magnitudes.append(float(turnover - self.problem.constraints.turnover_limit))
            elif "Leverage" in violation:
                leverage = np.sum(np.abs(weights))
                magnitudes.append(float(leverage - self.problem.constraints.leverage_limit))
            else:
                magnitudes.append(1.0)

        return ValidationResult(
            is_valid=is_valid,
            constraint_violations=violations,
            violation_magnitudes=magnitudes
        )

    def compute_objective(self, weights: np.ndarray) -> float:
        """
        Compute objective value for portfolio weights

        Args:
            weights: Portfolio weight vector

        Returns:
            Objective function value
        """
        if self.objective_function is None or self.problem is None:
            raise ValueError("No problem has been encoded yet")

        return self.objective_function(weights, self.problem)

    def _wrapped_objective(self, x: np.ndarray) -> float:
        """
        Wrapped objective function with constraint handling

        Args:
            x: Decision variable vector (may be unnormalized)

        Returns:
            Objective value with penalty for constraint violations
        """
        # Normalize weights if needed
        weights = self._normalize_weights(x)

        # Compute base objective
        obj_value = self.compute_objective(weights)

        # Add penalty for constraint violations
        validation = self.validate_solution(weights)
        if not validation.is_valid:
            penalty = sum(validation.violation_magnitudes) * 1e6
            obj_value += penalty

        return obj_value

    def _normalize_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize weight vector to satisfy sum-to-one constraint

        Args:
            x: Raw weight vector

        Returns:
            Normalized weights
        """
        if self.problem is None:
            return x

        weights = x.copy()

        # Apply bounds
        if self.problem.constraints.long_only:
            weights = np.maximum(weights, 0)

        if self.problem.constraints.max_weight is not None:
            weights = np.minimum(weights, self.problem.constraints.max_weight)

        # Normalize to sum to one if required
        if self.problem.constraints.sum_to_one:
            weight_sum = np.sum(weights)
            if weight_sum > 1e-8:
                weights = weights / weight_sum
            else:
                # Fallback to equal weights if all zeros
                weights = np.ones_like(weights) / len(weights)

        return weights

    def _get_objective_function(self, problem_type: ProblemType) -> Callable:
        """
        Get the appropriate objective function for the problem type

        Args:
            problem_type: Type of portfolio optimization problem

        Returns:
            Objective function callable
        """
        objective_map = {
            ProblemType.MEAN_VARIANCE: mean_variance_objective,
            ProblemType.CVAR: cvar_objective,
            ProblemType.RISK_PARITY: risk_parity_objective,
            ProblemType.BLACK_LITTERMAN: black_litterman_objective,
            ProblemType.MAX_SHARPE: maximum_sharpe_objective,
            ProblemType.MIN_VARIANCE: minimum_variance_objective,
            ProblemType.FACTOR_MODEL: factor_model_objective,
            ProblemType.ROBUST: robust_objective
        }

        if problem_type not in objective_map:
            raise ValueError(f"Unknown problem type: {problem_type}")

        return objective_map[problem_type]

    def _build_constraints(self):
        """Build constraint matrices for the optimization problem"""
        if self.problem is None:
            return

        n = self.problem.n_assets
        constraints = []
        bounds = []

        # Sum to one constraint: sum(w) = 1
        if self.problem.constraints.sum_to_one:
            constraints.append(np.ones((1, n)))
            bounds.append([1.0, 1.0])

        # Individual asset bounds are handled separately by optimizers
        # Here we only build linear constraints

        # Sector constraints
        if self.problem.constraints.sector_limits and self.problem.sector_mapping:
            for sector, (min_exp, max_exp) in self.problem.constraints.sector_limits.items():
                # Create sector exposure vector
                sector_vec = np.zeros(n)
                for i, s in self.problem.sector_mapping.items():
                    if s == sector:
                        sector_vec[i] = 1.0

                # Add min constraint: sector_weights >= min_exp
                if min_exp > -np.inf:
                    constraints.append(sector_vec.reshape(1, -1))
                    bounds.append([min_exp, np.inf])

                # Add max constraint: sector_weights <= max_exp
                if max_exp < np.inf:
                    constraints.append(sector_vec.reshape(1, -1))
                    bounds.append([-np.inf, max_exp])

        # Group constraints
        if self.problem.constraints.group_constraints:
            for group in self.problem.constraints.group_constraints:
                group_vec = np.zeros(n)
                for idx in group['assets']:
                    group_vec[idx] = 1.0

                if 'min_weight' in group:
                    constraints.append(group_vec.reshape(1, -1))
                    bounds.append([group['min_weight'], np.inf])

                if 'max_weight' in group:
                    constraints.append(group_vec.reshape(1, -1))
                    bounds.append([-np.inf, group['max_weight']])

        # Build final constraint matrix
        if constraints:
            self.constraint_matrix = np.vstack(constraints)
            self.constraint_bounds = np.array(bounds)
        else:
            self.constraint_matrix = None
            self.constraint_bounds = None

    def _get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of active constraints"""
        if self.problem is None:
            return {}

        summary = {
            'long_only': self.problem.constraints.long_only,
            'sum_to_one': self.problem.constraints.sum_to_one
        }

        if self.problem.constraints.cardinality is not None:
            summary['cardinality'] = self.problem.constraints.cardinality

        if self.problem.constraints.turnover_limit is not None:
            summary['turnover_limit'] = self.problem.constraints.turnover_limit

        if self.problem.constraints.leverage_limit is not None:
            summary['leverage_limit'] = self.problem.constraints.leverage_limit

        if self.problem.constraints.sector_limits:
            summary['n_sector_constraints'] = len(self.problem.constraints.sector_limits)

        return summary

    def get_initial_solution(self) -> np.ndarray:
        """
        Generate a feasible initial solution for the portfolio problem

        Returns:
            Initial weight vector
        """
        if self.problem is None:
            raise ValueError("No problem has been encoded yet")

        n = self.problem.n_assets

        # Start with equal weights
        weights = np.ones(n) / n

        # Adjust for cardinality if needed
        if self.problem.constraints.cardinality is not None:
            k = min(self.problem.constraints.cardinality, n)
            # Select top k assets by expected return
            top_k = np.argsort(self.problem.get_expected_returns())[-k:]
            weights = np.zeros(n)
            weights[top_k] = 1.0 / k

        # Use current weights if available and valid
        if self.problem.current_weights is not None:
            current_valid, _ = self.problem.validate_weights(self.problem.current_weights)
            if current_valid:
                weights = self.problem.current_weights.copy()

        return weights

    def get_bounds(self) -> list:
        """
        Get variable bounds for the optimization problem

        Returns:
            List of (min, max) tuples for each asset
        """
        if self.problem is None:
            return []

        n = self.problem.n_assets
        bounds = []

        for i in range(n):
            min_w = self.problem.constraints.min_weight or 0.0
            max_w = self.problem.constraints.max_weight or 1.0

            # Check for asset-specific constraints
            if self.problem.constraints.holding_constraints and i in self.problem.constraints.holding_constraints:
                min_w, max_w = self.problem.constraints.holding_constraints[i]

            bounds.append((min_w, max_w))

        return bounds