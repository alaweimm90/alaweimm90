#!/usr/bin/env python3
"""
Shadow Evaluation System for Regression Detection
Priority [060] implementation with parallel evaluation and comparison.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalMode(Enum):
    """Evaluation execution modes."""

    PRODUCTION = "production"
    SHADOW = "shadow"
    COMPARISON = "comparison"


@dataclass
class ShadowResult:
    """Shadow evaluation result."""

    eval_id: str
    timestamp: str
    mode: EvalMode
    version: str
    input_hash: str
    output: Dict[str, Any]
    metrics: Dict[str, float]
    duration_ms: float
    errors: List[str]


class ShadowEvaluator:
    """Shadow evaluation system for regression detection."""

    def __init__(self, config_path: Optional[Path] = None):
        self.prod_version = "v1.0.0"
        self.shadow_versions = ["v1.1.0-rc1", "v1.1.0-rc2"]
        self.results = []
        self.regression_thresholds = {
            "score_delta": 5.0,
            "latency_increase": 1.5,
            "error_rate": 0.01,
        }

    def run_shadow_eval(self, input_data: Dict[str, Any], feature: str) -> Dict[str, Any]:
        """Run shadow evaluation alongside production."""
        input_hash = self._hash_input(input_data)
        results = {}

        # Run production
        prod_result = self._run_evaluation(
            input_data, feature, self.prod_version, EvalMode.PRODUCTION
        )
        results["production"] = prod_result

        # Run shadow versions in parallel (simulated)
        shadow_results = []
        for version in self.shadow_versions:
            shadow_result = self._run_evaluation(input_data, feature, version, EvalMode.SHADOW)
            shadow_results.append(shadow_result)
        results["shadow"] = shadow_results

        # Compare results
        comparison = self._compare_results(prod_result, shadow_results)
        results["comparison"] = comparison

        # Store for analysis
        self.results.extend([prod_result] + shadow_results)

        return results

    def _run_evaluation(
        self, input_data: Dict[str, Any], feature: str, version: str, mode: EvalMode
    ) -> ShadowResult:
        """Run single evaluation."""
        start_time = time.time()

        # Simulate evaluation based on version
        if version == self.prod_version:
            output = self._simulate_production(input_data, feature)
        else:
            output = self._simulate_shadow(input_data, feature, version)

        duration_ms = (time.time() - start_time) * 1000

        return ShadowResult(
            eval_id=f"{feature}-{version}-{int(time.time())}",
            timestamp=self._get_timestamp(),
            mode=mode,
            version=version,
            input_hash=self._hash_input(input_data),
            output=output,
            metrics=self._extract_metrics(output),
            duration_ms=duration_ms,
            errors=[],
        )

    def _simulate_production(self, input_data: Dict[str, Any], feature: str) -> Dict[str, Any]:
        """Simulate production evaluation."""
        if feature == "nightmare":
            return {"survival_score": 67.5, "attacks_generated": 185, "confidence": 0.85}
        elif feature == "chaos":
            return {"novelty_score": 0.87, "collisions": 25, "feasibility": 0.45}
        else:
            return {"score": 75.0, "confidence": 0.8}

    def _simulate_shadow(
        self, input_data: Dict[str, Any], feature: str, version: str
    ) -> Dict[str, Any]:
        """Simulate shadow evaluation with variations."""
        base = self._simulate_production(input_data, feature)

        # Add version-specific variations
        if "rc1" in version:
            # RC1: slightly better scores, higher latency
            for key in base:
                if "score" in key:
                    base[key] *= 1.02
        elif "rc2" in version:
            # RC2: slightly worse scores, lower latency
            for key in base:
                if "score" in key:
                    base[key] *= 0.98

        return base

    def _compare_results(self, prod: ShadowResult, shadows: List[ShadowResult]) -> Dict[str, Any]:
        """Compare production and shadow results."""
        comparison = {"regressions": [], "improvements": [], "metrics_delta": {}}

        for shadow in shadows:
            # Compare metrics
            for metric, prod_value in prod.metrics.items():
                if metric in shadow.metrics:
                    shadow_value = shadow.metrics[metric]
                    delta = shadow_value - prod_value
                    delta_pct = (delta / prod_value * 100) if prod_value else 0

                    metric_info = {
                        "metric": metric,
                        "version": shadow.version,
                        "prod_value": prod_value,
                        "shadow_value": shadow_value,
                        "delta": delta,
                        "delta_pct": delta_pct,
                    }

                    # Check for regression
                    if abs(delta_pct) > 5:
                        if delta < 0:
                            comparison["regressions"].append(metric_info)
                        else:
                            comparison["improvements"].append(metric_info)

            # Compare latency
            latency_ratio = shadow.duration_ms / prod.duration_ms
            if latency_ratio > self.regression_thresholds["latency_increase"]:
                comparison["regressions"].append(
                    {
                        "metric": "latency",
                        "version": shadow.version,
                        "prod_value": prod.duration_ms,
                        "shadow_value": shadow.duration_ms,
                        "ratio": latency_ratio,
                    }
                )

        return comparison

    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """Create hash of input for correlation."""
        return hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()[:16]

    def _extract_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from output."""
        metrics = {}
        for key, value in output.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def generate_regression_report(self) -> Dict[str, Any]:
        """Generate comprehensive regression report."""
        report = {
            "summary": {
                "total_evaluations": len(self.results),
                "production_runs": sum(1 for r in self.results if r.mode == EvalMode.PRODUCTION),
                "shadow_runs": sum(1 for r in self.results if r.mode == EvalMode.SHADOW),
                "versions_tested": list(set(r.version for r in self.results)),
            },
            "regressions_by_version": {},
            "performance_comparison": {},
            "recommendations": [],
        }

        # Analyze by version
        for version in self.shadow_versions:
            version_results = [r for r in self.results if r.version == version]
            if version_results:
                avg_latency = sum(r.duration_ms for r in version_results) / len(version_results)
                report["performance_comparison"][version] = {
                    "avg_latency_ms": avg_latency,
                    "total_runs": len(version_results),
                }

        # Generate recommendations
        if report["performance_comparison"]:
            best_version = min(
                report["performance_comparison"].items(), key=lambda x: x[1]["avg_latency_ms"]
            )[0]
            report["recommendations"].append(
                f"Consider promoting {best_version} based on performance"
            )

        return report


# Priority [062] - Dry-run defaults
class DryRunWrapper:
    """Dry-run wrapper for safe script execution."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.operations = []

    def execute(self, operation: str, actual_fn=None, *args, **kwargs):
        """Execute or simulate operation."""
        if self.dry_run:
            self.operations.append(
                {"operation": operation, "args": args, "kwargs": kwargs, "timestamp": time.time()}
            )
            print(f"[DRY-RUN] Would execute: {operation}")
            return {"simulated": True}
        else:
            if actual_fn:
                return actual_fn(*args, **kwargs)
            else:
                raise ValueError("No actual function provided for non-dry-run mode")

    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Get log of operations that would be executed."""
        return self.operations


# Priority [071] - Budget threshold alerts
class BudgetMonitor:
    """Budget monitoring with threshold alerts."""

    def __init__(self, budget_limit: float):
        self.budget_limit = budget_limit
        self.current_spend = 0.0
        self.alerts = []
        self.thresholds = [0.5, 0.7, 0.9, 1.0]

    def track_spend(self, amount: float, operation: str) -> Dict[str, Any]:
        """Track spending and generate alerts."""
        previous_spend = self.current_spend
        self.current_spend += amount

        result = {
            "operation": operation,
            "amount": amount,
            "total_spend": self.current_spend,
            "budget_remaining": self.budget_limit - self.current_spend,
            "budget_used_pct": (self.current_spend / self.budget_limit) * 100,
            "alerts": [],
        }

        # Check thresholds
        for threshold in self.thresholds:
            threshold_value = self.budget_limit * threshold
            if previous_spend < threshold_value <= self.current_spend:
                alert = {
                    "level": self._get_alert_level(threshold),
                    "threshold": threshold,
                    "message": f"Budget {int(threshold*100)}% consumed",
                    "action": self._get_alert_action(threshold),
                }
                self.alerts.append(alert)
                result["alerts"].append(alert)

        return result

    def _get_alert_level(self, threshold: float) -> str:
        """Get alert severity level."""
        if threshold >= 1.0:
            return "CRITICAL"
        elif threshold >= 0.9:
            return "HIGH"
        elif threshold >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_alert_action(self, threshold: float) -> str:
        """Get recommended action for threshold."""
        if threshold >= 1.0:
            return "ABORT_EXECUTION"
        elif threshold >= 0.9:
            return "REQUIRE_APPROVAL"
        elif threshold >= 0.7:
            return "SWITCH_TO_CHEAPER_MODEL"
        else:
            return "NOTIFY"


if __name__ == "__main__":
    # Test shadow evaluation
    shadow = ShadowEvaluator()
    test_input = {"title": "Test", "hypothesis": "Shadow evaluation test", "feature": "nightmare"}

    results = shadow.run_shadow_eval(test_input, "nightmare")
    print("Shadow Evaluation Results:", json.dumps(results, indent=2))

    # Test dry-run
    dry_run = DryRunWrapper(dry_run=True)
    dry_run.execute("delete_file", None, path="/important/file.txt")
    print("Dry-run operations:", dry_run.get_operations_log())

    # Test budget monitoring
    monitor = BudgetMonitor(budget_limit=10.0)
    monitor.track_spend(3.0, "initial_processing")
    monitor.track_spend(4.0, "ensemble_execution")
    result = monitor.track_spend(2.5, "output_generation")
    print("Budget status:", json.dumps(result, indent=2))
