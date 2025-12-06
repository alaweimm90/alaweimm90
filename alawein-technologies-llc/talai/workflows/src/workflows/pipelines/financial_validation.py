"""
Financial Validation Pipeline - Validation for financial models

Specialized pipeline for validating financial models, risk assessments, and trading strategies.
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
import math
import statistics

from .validation_pipeline import ValidationPipeline, ValidationStep, ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class FinancialValidationPipeline(ValidationPipeline):
    """Pipeline for financial model validation"""

    def __init__(self):
        super().__init__(
            name="financial_validation",
            description="Validates financial models and trading strategies"
        )
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize financial validation steps"""

        # Data quality validation
        self.add_step(ValidationStep(
            name="data_quality",
            validator=self._validate_data_quality,
            required=True,
            timeout=30
        ))

        # Risk metrics validation
        self.add_step(ValidationStep(
            name="risk_metrics",
            validator=self._validate_risk_metrics,
            required=True,
            timeout=60
        ))

        # Backtesting validation
        self.add_step(ValidationStep(
            name="backtesting",
            validator=self._validate_backtesting,
            required=True,
            timeout=120
        ))

        # Regulatory compliance
        self.add_step(ValidationStep(
            name="regulatory_compliance",
            validator=self._validate_compliance,
            required=True,
            timeout=45
        ))

        # Model stability
        self.add_step(ValidationStep(
            name="model_stability",
            validator=self._validate_model_stability,
            required=False,
            timeout=90
        ))

        # Stress testing
        self.add_step(ValidationStep(
            name="stress_testing",
            validator=self._validate_stress_test,
            required=False,
            timeout=180
        ))

    async def _validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial data quality"""
        time_series = data.get("time_series", [])
        missing_data_ratio = data.get("missing_data_ratio", 0)

        if not time_series:
            return {
                "status": "failed",
                "message": "No time series data provided"
            }

        # Check for missing data
        if missing_data_ratio > 0.05:
            return {
                "status": "warning",
                "score": 1 - missing_data_ratio,
                "message": f"High missing data ratio: {missing_data_ratio:.2%}"
            }

        # Check for outliers
        if len(time_series) > 3:
            mean = statistics.mean(time_series)
            std = statistics.stdev(time_series)
            outliers = [x for x in time_series if abs(x - mean) > 3 * std]
            outlier_ratio = len(outliers) / len(time_series)
        else:
            outlier_ratio = 0

        await asyncio.sleep(0.5)

        score = 1 - (missing_data_ratio + outlier_ratio) / 2

        return {
            "status": "passed" if score > 0.8 else "warning",
            "score": score,
            "message": "Data quality acceptable",
            "details": {
                "missing_data_ratio": missing_data_ratio,
                "outlier_ratio": outlier_ratio,
                "data_points": len(time_series)
            }
        }

    async def _validate_risk_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk metrics"""
        returns = data.get("returns", [])
        var_95 = data.get("value_at_risk_95", 0)
        sharpe_ratio = data.get("sharpe_ratio", 0)
        max_drawdown = data.get("max_drawdown", 0)

        if not returns:
            return {
                "status": "failed",
                "message": "No returns data provided"
            }

        # Check Sharpe ratio
        if sharpe_ratio < 0:
            status = "failed"
            message = "Negative Sharpe ratio"
            score = 0.2
        elif sharpe_ratio < 1:
            status = "warning"
            message = "Low Sharpe ratio"
            score = 0.5
        else:
            status = "passed"
            message = "Good risk-adjusted returns"
            score = min(1.0, sharpe_ratio / 3)

        # Check max drawdown
        if abs(max_drawdown) > 0.3:
            status = "warning"
            score *= 0.7
            message += f", High drawdown: {max_drawdown:.1%}"

        await asyncio.sleep(0.5)

        return {
            "status": status,
            "score": score,
            "message": message,
            "details": {
                "sharpe_ratio": sharpe_ratio,
                "var_95": var_95,
                "max_drawdown": max_drawdown,
                "volatility": statistics.stdev(returns) if len(returns) > 1 else 0
            }
        }

    async def _validate_backtesting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtesting results"""
        backtest_returns = data.get("backtest_returns", [])
        win_rate = data.get("win_rate", 0)
        profit_factor = data.get("profit_factor", 1)
        num_trades = data.get("num_trades", 0)

        if not backtest_returns:
            return {
                "status": "failed",
                "message": "No backtesting data"
            }

        # Check sample size
        if num_trades < 30:
            return {
                "status": "warning",
                "score": 0.5,
                "message": f"Insufficient sample size: {num_trades} trades"
            }

        # Check win rate
        if win_rate < 0.4:
            status = "warning"
            score = win_rate
        else:
            status = "passed"
            score = win_rate

        # Check profit factor
        if profit_factor < 1.2:
            score *= 0.8
            status = "warning"

        await asyncio.sleep(1.0)

        return {
            "status": status,
            "score": score,
            "message": f"Backtest win rate: {win_rate:.1%}",
            "details": {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "num_trades": num_trades,
                "avg_return": statistics.mean(backtest_returns) if backtest_returns else 0
            }
        }

    async def _validate_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance"""
        leverage = data.get("leverage", 1)
        position_limits = data.get("position_limits_ok", True)
        regulatory_framework = data.get("regulatory_framework", "")

        violations = []

        # Check leverage limits
        if leverage > 2:
            violations.append("Excessive leverage")

        # Check position limits
        if not position_limits:
            violations.append("Position limits exceeded")

        # Check regulatory framework
        if not regulatory_framework:
            violations.append("No regulatory framework specified")

        await asyncio.sleep(0.5)

        if violations:
            return {
                "status": "failed" if len(violations) > 1 else "warning",
                "score": 1 - len(violations) * 0.3,
                "message": f"Compliance issues: {', '.join(violations)}",
                "details": {"violations": violations}
            }

        return {
            "status": "passed",
            "score": 1.0,
            "message": "Compliant with regulations",
            "details": {
                "framework": regulatory_framework,
                "leverage": leverage
            }
        }

    async def _validate_model_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model stability over time"""
        performance_over_time = data.get("performance_over_time", [])
        parameter_sensitivity = data.get("parameter_sensitivity", {})

        if not performance_over_time:
            return {
                "status": "skipped",
                "message": "No temporal performance data"
            }

        # Calculate stability metrics
        if len(performance_over_time) > 1:
            performance_std = statistics.stdev(performance_over_time)
            performance_mean = statistics.mean(performance_over_time)
            cv = performance_std / abs(performance_mean) if performance_mean != 0 else float('inf')
        else:
            cv = 0

        # Check coefficient of variation
        if cv > 0.5:
            status = "warning"
            score = 0.5
            message = "High performance variability"
        else:
            status = "passed"
            score = 1 - cv
            message = "Model shows good stability"

        await asyncio.sleep(0.5)

        return {
            "status": status,
            "score": score,
            "message": message,
            "details": {
                "coefficient_variation": cv,
                "performance_periods": len(performance_over_time),
                "parameter_sensitivity": parameter_sensitivity
            }
        }

    async def _validate_stress_test(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stress testing results"""
        scenarios = data.get("stress_scenarios", {})

        if not scenarios:
            return {
                "status": "skipped",
                "message": "No stress test scenarios provided"
            }

        failed_scenarios = []
        scenario_scores = []

        for scenario_name, result in scenarios.items():
            loss = result.get("loss", 0)
            recovery_time = result.get("recovery_time_days", 0)

            # Check if loss exceeds threshold
            if abs(loss) > 0.5:
                failed_scenarios.append(scenario_name)
                scenario_scores.append(0.2)
            elif abs(loss) > 0.3:
                scenario_scores.append(0.5)
            else:
                scenario_scores.append(0.9)

        avg_score = statistics.mean(scenario_scores) if scenario_scores else 0

        await asyncio.sleep(1.0)

        if failed_scenarios:
            return {
                "status": "warning",
                "score": avg_score,
                "message": f"Failed scenarios: {', '.join(failed_scenarios)}",
                "details": {
                    "scenarios_tested": len(scenarios),
                    "failed_scenarios": failed_scenarios
                }
            }

        return {
            "status": "passed",
            "score": avg_score,
            "message": f"Passed {len(scenarios)} stress scenarios",
            "details": {"scenarios_tested": len(scenarios)}
        }