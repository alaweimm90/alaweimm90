#!/usr/bin/env python3
"""
Advanced Systems for ORCHEX: Multiple Priority Implementations
Includes: [082] Meta-evaluation, [122] Bootstrap CIs, [149] Comparative Analysis,
[151] Adaptive Scheduling, [221] ML Routing, [230] Tenant Quotas
"""

import hashlib
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# Priority [082] - Evaluator Meta-Evaluation Suite
class MetaEvaluator:
    """Meta-evaluation system to evaluate the evaluators."""

    def __init__(self):
        self.evaluator_metrics = {}
        self.ground_truth = {}
        self.calibration_data = []

    def evaluate_evaluator(
        self, evaluator_name: str, predictions: List[float], actuals: List[float]
    ) -> Dict[str, Any]:
        """Evaluate an evaluator's performance."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        metrics = {
            "evaluator": evaluator_name,
            "n_samples": len(predictions),
            "accuracy_metrics": {},
            "calibration_metrics": {},
            "consistency_metrics": {},
            "bias_metrics": {},
        }

        # Accuracy metrics
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        rmse = np.sqrt(mse)

        metrics["accuracy_metrics"] = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r_squared": float(self._calculate_r_squared(predictions, actuals)),
        }

        # Calibration metrics
        ece = self._calculate_ece(predictions, actuals)
        metrics["calibration_metrics"] = {
            "ece": float(ece),
            "overconfidence": float(self._calculate_overconfidence(predictions, actuals)),
            "underconfidence": float(self._calculate_underconfidence(predictions, actuals)),
        }

        # Consistency metrics
        metrics["consistency_metrics"] = {
            "variance": float(np.var(predictions)),
            "stability": float(self._calculate_stability(predictions)),
            "monotonicity": float(self._check_monotonicity(predictions, actuals)),
        }

        # Bias detection
        metrics["bias_metrics"] = {
            "mean_bias": float(np.mean(np.array(predictions) - np.array(actuals))),
            "systematic_bias": self._detect_systematic_bias(predictions, actuals),
            "outlier_sensitivity": float(self._calculate_outlier_sensitivity(predictions, actuals)),
        }

        # Store for tracking
        self.evaluator_metrics[evaluator_name] = metrics

        return metrics

    def _calculate_r_squared(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate R-squared value."""
        ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
        ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _calculate_ece(
        self, predictions: List[float], actuals: List[float], n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_confidence = np.mean(np.array(predictions)[mask])
                bin_accuracy = np.mean(np.array(actuals)[mask])
                bin_weight = np.sum(mask) / len(predictions)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return ece

    def _calculate_overconfidence(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate overconfidence score."""
        overconf = [p - a for p, a in zip(predictions, actuals) if p > a]
        return np.mean(overconf) if overconf else 0

    def _calculate_underconfidence(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate underconfidence score."""
        underconf = [a - p for p, a in zip(predictions, actuals) if a > p]
        return np.mean(underconf) if underconf else 0

    def _calculate_stability(self, predictions: List[float]) -> float:
        """Calculate prediction stability."""
        if len(predictions) < 2:
            return 1.0
        diffs = np.diff(predictions)
        return 1.0 / (1.0 + np.std(diffs))

    def _check_monotonicity(self, predictions: List[float], actuals: List[float]) -> float:
        """Check if predictions maintain monotonic relationship with actuals."""
        if len(predictions) < 2:
            return 1.0

        pred_order = np.argsort(predictions)
        actual_order = np.argsort(actuals)

        concordant = sum(1 for i, j in zip(pred_order, actual_order) if i == j)
        return concordant / len(predictions)

    def _detect_systematic_bias(
        self, predictions: List[float], actuals: List[float]
    ) -> Dict[str, Any]:
        """Detect systematic bias patterns."""
        residuals = np.array(predictions) - np.array(actuals)

        return {
            "has_bias": abs(np.mean(residuals)) > 0.05,
            "bias_direction": "overestimate" if np.mean(residuals) > 0 else "underestimate",
            "bias_magnitude": float(abs(np.mean(residuals))),
        }

    def _calculate_outlier_sensitivity(
        self, predictions: List[float], actuals: List[float]
    ) -> float:
        """Calculate sensitivity to outliers."""
        if len(predictions) < 10:
            return 0

        # Calculate error with and without outliers
        errors = np.abs(np.array(predictions) - np.array(actuals))
        q75, q25 = np.percentile(errors, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr

        non_outlier_mask = errors <= outlier_threshold
        if np.sum(non_outlier_mask) == 0:
            return 1.0

        mae_all = np.mean(errors)
        mae_no_outliers = np.mean(errors[non_outlier_mask])

        return (mae_all - mae_no_outliers) / mae_all if mae_all > 0 else 0


# Priority [122] - Bootstrap Confidence Intervals
class BootstrapCI:
    """Bootstrap confidence interval calculator."""

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def calculate_ci(self, data: List[float], statistic_fn=np.mean) -> Dict[str, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        bootstrap_statistics = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_fn(sample))

        # Calculate percentile CI
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)

        return {
            "point_estimate": float(statistic_fn(data)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "std_error": float(np.std(bootstrap_statistics)),
        }

    def calculate_metrics_ci(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate CIs for multiple metrics."""
        results = {}

        for metric_name, values in metrics.items():
            results[metric_name] = self.calculate_ci(values)

        return results


# Priority [149] - Automated Comparative Analysis
class ComparativeAnalyzer:
    """Automated comparative analysis system."""

    def __init__(self):
        self.comparisons = []
        self.statistical_tests = {}

    def compare_systems(
        self, system_a: Dict[str, Any], system_b: Dict[str, Any], metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare two systems across metrics."""
        comparison = {
            "system_a": system_a["name"],
            "system_b": system_b["name"],
            "timestamp": time.time(),
            "metrics_comparison": {},
            "statistical_tests": {},
            "winner": None,
            "summary": [],
        }

        wins = {"a": 0, "b": 0, "tie": 0}

        for metric in metrics:
            if metric in system_a and metric in system_b:
                a_values = system_a[metric]
                b_values = system_b[metric]

                # Basic comparison
                metric_comp = {
                    "mean_a": np.mean(a_values),
                    "mean_b": np.mean(b_values),
                    "median_a": np.median(a_values),
                    "median_b": np.median(b_values),
                    "std_a": np.std(a_values),
                    "std_b": np.std(b_values),
                }

                # Statistical test (t-test)
                t_stat, p_value = stats.ttest_ind(a_values, b_values)
                metric_comp["t_statistic"] = float(t_stat)
                metric_comp["p_value"] = float(p_value)
                metric_comp["significant"] = p_value < 0.05

                # Determine winner for this metric
                if metric_comp["significant"]:
                    if metric_comp["mean_a"] > metric_comp["mean_b"]:
                        metric_comp["winner"] = "a"
                        wins["a"] += 1
                    else:
                        metric_comp["winner"] = "b"
                        wins["b"] += 1
                else:
                    metric_comp["winner"] = "tie"
                    wins["tie"] += 1

                comparison["metrics_comparison"][metric] = metric_comp

        # Overall winner
        if wins["a"] > wins["b"]:
            comparison["winner"] = system_a["name"]
        elif wins["b"] > wins["a"]:
            comparison["winner"] = system_b["name"]
        else:
            comparison["winner"] = "tie"

        comparison["win_counts"] = wins

        # Generate summary
        comparison["summary"] = self._generate_summary(comparison)

        self.comparisons.append(comparison)
        return comparison

    def _generate_summary(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate textual summary of comparison."""
        summary = []

        if comparison["winner"] != "tie":
            summary.append(f"{comparison['winner']} performs better overall")

        for metric, comp in comparison["metrics_comparison"].items():
            if comp["significant"]:
                winner = comparison["system_a"] if comp["winner"] == "a" else comparison["system_b"]
                improvement = (
                    abs(comp["mean_a"] - comp["mean_b"]) / min(comp["mean_a"], comp["mean_b"]) * 100
                )
                summary.append(f"{winner} shows {improvement:.1f}% improvement in {metric}")

        return summary


# Priority [151] - Adaptive Scheduling by Information Gain
class AdaptiveScheduler:
    """Adaptive scheduling system based on information gain."""

    def __init__(self):
        self.task_history = []
        self.information_gains = defaultdict(list)
        self.task_queue = []

    def schedule_task(self, available_tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Schedule next task based on expected information gain."""
        if not available_tasks:
            return None

        # Calculate expected information gain for each task
        task_scores = []

        for task in available_tasks:
            score = self._calculate_information_gain(task)
            task_scores.append((task, score))

        # Sort by information gain (descending)
        task_scores.sort(key=lambda x: x[1], reverse=True)

        # Select highest information gain task
        selected_task = task_scores[0][0]

        # Record selection
        self.task_history.append(
            {"task": selected_task, "expected_gain": task_scores[0][1], "timestamp": time.time()}
        )

        return selected_task

    def _calculate_information_gain(self, task: Dict[str, Any]) -> float:
        """Calculate expected information gain for a task."""
        base_score = 1.0

        # Factor 1: Uncertainty (higher uncertainty = higher gain)
        if "uncertainty" in task:
            base_score *= 1 + task["uncertainty"]

        # Factor 2: Coverage (unexplored areas = higher gain)
        if "coverage" in task:
            base_score *= 2 - task["coverage"]

        # Factor 3: Historical performance
        task_type = task.get("type", "unknown")
        if task_type in self.information_gains:
            historical_gains = self.information_gains[task_type]
            avg_gain = np.mean(historical_gains[-10:])  # Last 10 runs
            base_score *= 1 + avg_gain

        # Factor 4: Priority
        priority_multipliers = {"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.5}
        priority = task.get("priority", "medium")
        base_score *= priority_multipliers.get(priority, 1.0)

        # Factor 5: Cost efficiency
        if "estimated_cost" in task:
            cost_efficiency = 1.0 / (1 + task["estimated_cost"])
            base_score *= cost_efficiency

        return base_score

    def update_information_gain(self, task_type: str, actual_gain: float):
        """Update information gain history."""
        self.information_gains[task_type].append(actual_gain)


# Priority [221] - ML-based Task Routing
class MLTaskRouter:
    """Machine learning based task routing system."""

    def __init__(self):
        self.routing_history = []
        self.feature_weights = {
            "hypothesis_length": 0.2,
            "has_quantitative": 0.3,
            "has_comparison": 0.25,
            "domain_keywords": 0.25,
        }
        self.feature_overrides = {}

    def route_task(
        self, input_data: Dict[str, Any], available_features: List[str]
    ) -> Tuple[str, float]:
        """Route task to appropriate feature using ML."""
        # Extract features
        features = self._extract_features(input_data)

        # Check for manual overrides
        if "feature_override" in input_data:
            return input_data["feature_override"], 1.0

        # Calculate scores for each feature
        scores = {}
        for feature in available_features:
            score = self._calculate_feature_score(features, feature)
            scores[feature] = score

        # Select best feature
        best_feature = max(scores.items(), key=lambda x: x[1])

        # Record routing decision
        self.routing_history.append(
            {
                "input_hash": hashlib.md5(str(input_data).encode()).hexdigest()[:8],
                "features": features,
                "scores": scores,
                "selected": best_feature[0],
                "confidence": best_feature[1],
                "timestamp": time.time(),
            }
        )

        return best_feature

    def _extract_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from input."""
        hypothesis = input_data.get("hypothesis", "")

        features = {
            "hypothesis_length": len(hypothesis.split()),
            "has_quantitative": any(char.isdigit() for char in hypothesis),
            "has_comparison": any(
                word in hypothesis.lower()
                for word in ["versus", "vs", "compared", "better", "worse"]
            ),
            "domain_keywords": self._extract_domain_keywords(hypothesis),
        }

        return features

    def _extract_domain_keywords(self, text: str) -> Dict[str, int]:
        """Extract domain-specific keywords."""
        domains = {
            "ai_safety": ["safety", "alignment", "interpretability", "risk"],
            "innovation": ["novel", "breakthrough", "innovation", "creative"],
            "optimization": ["optimize", "improve", "enhance", "efficiency"],
            "research": ["hypothesis", "experiment", "analysis", "study"],
            "market": ["prediction", "forecast", "market", "trading"],
        }

        keyword_counts = {}
        text_lower = text.lower()

        for domain, keywords in domains.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_counts[domain] = count

        return keyword_counts

    def _calculate_feature_score(self, features: Dict[str, Any], target_feature: str) -> float:
        """Calculate score for routing to target feature."""
        score = 0.5  # Base score

        # Feature-specific scoring
        if target_feature == "nightmare":
            # Nightmare prefers longer, research-oriented hypotheses
            score += features["hypothesis_length"] * 0.01
            score += features["domain_keywords"].get("ai_safety", 0) * 0.1
            score += features["domain_keywords"].get("research", 0) * 0.1

        elif target_feature == "chaos":
            # Chaos prefers innovative, creative hypotheses
            score += features["domain_keywords"].get("innovation", 0) * 0.15
            score += 0.1 if features["has_comparison"] else 0

        elif target_feature == "evolution":
            # Evolution prefers optimization-focused hypotheses
            score += features["domain_keywords"].get("optimization", 0) * 0.15
            score += 0.05 if features["has_quantitative"] else 0

        elif target_feature == "multiverse":
            # Multiverse prefers comparative hypotheses
            score += 0.2 if features["has_comparison"] else 0
            score += features["hypothesis_length"] * 0.005

        elif target_feature == "market":
            # Market prefers prediction-oriented hypotheses
            score += features["domain_keywords"].get("market", 0) * 0.2
            score += 0.1 if features["has_quantitative"] else 0

        return min(score, 1.0)  # Cap at 1.0

    def add_override(self, pattern: str, feature: str):
        """Add routing override rule."""
        self.feature_overrides[pattern] = feature


# Priority [230] - Per-tenant Quotas
class TenantQuotaManager:
    """Per-tenant quota management system."""

    def __init__(self):
        self.quotas = {}
        self.usage = defaultdict(lambda: defaultdict(float))
        self.queue = defaultdict(list)

    def set_quota(self, tenant_id: str, quota_type: str, limit: float, period: str = "daily"):
        """Set quota for tenant."""
        if tenant_id not in self.quotas:
            self.quotas[tenant_id] = {}

        self.quotas[tenant_id][quota_type] = {
            "limit": limit,
            "period": period,
            "reset_time": self._get_next_reset(period),
        }

    def check_quota(self, tenant_id: str, quota_type: str, amount: float = 1.0) -> Dict[str, Any]:
        """Check if tenant has available quota."""
        if tenant_id not in self.quotas or quota_type not in self.quotas[tenant_id]:
            return {"allowed": True, "unlimited": True}

        quota = self.quotas[tenant_id][quota_type]
        current_usage = self.usage[tenant_id][quota_type]

        # Check if reset needed
        if time.time() > quota["reset_time"]:
            self.usage[tenant_id][quota_type] = 0
            quota["reset_time"] = self._get_next_reset(quota["period"])
            current_usage = 0

        available = quota["limit"] - current_usage

        return {
            "allowed": available >= amount,
            "unlimited": False,
            "current_usage": current_usage,
            "limit": quota["limit"],
            "available": available,
            "usage_pct": (current_usage / quota["limit"]) * 100,
        }

    def consume_quota(self, tenant_id: str, quota_type: str, amount: float = 1.0) -> bool:
        """Consume quota if available."""
        check = self.check_quota(tenant_id, quota_type, amount)

        if check["allowed"]:
            self.usage[tenant_id][quota_type] += amount
            return True
        else:
            # Add to queue if quota exceeded
            self.queue[tenant_id].append(
                {"quota_type": quota_type, "amount": amount, "timestamp": time.time()}
            )
            return False

    def _get_next_reset(self, period: str) -> float:
        """Calculate next reset time based on period."""
        current = time.time()

        if period == "minutely":
            return current + 60
        elif period == "hourly":
            return current + 3600
        elif period == "daily":
            return current + 86400
        elif period == "weekly":
            return current + 604800
        else:
            return current + 86400  # Default to daily

    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get complete quota status for tenant."""
        if tenant_id not in self.quotas:
            return {"tenant_id": tenant_id, "quotas": {}}

        status = {
            "tenant_id": tenant_id,
            "quotas": {},
            "queued_requests": len(self.queue[tenant_id]),
        }

        for quota_type in self.quotas[tenant_id]:
            check = self.check_quota(tenant_id, quota_type, 0)
            status["quotas"][quota_type] = check

        return status


if __name__ == "__main__":
    # Test meta-evaluator
    meta_eval = MetaEvaluator()
    predictions = [0.7, 0.8, 0.6, 0.9, 0.75]
    actuals = [0.72, 0.78, 0.65, 0.85, 0.73]
    result = meta_eval.evaluate_evaluator("test_evaluator", predictions, actuals)
    print("Meta-evaluation:", json.dumps(result, indent=2))

    # Test bootstrap CI
    bootstrap = BootstrapCI()
    data = [67.5, 68.2, 66.9, 69.1, 67.8, 68.5, 67.0]
    ci = bootstrap.calculate_ci(data)
    print("Bootstrap CI:", ci)

    # Test ML routing
    router = MLTaskRouter()
    test_input = {
        "hypothesis": "AI safety through interpretability can ensure safe deployment of large language models",
        "title": "AI Safety Research",
    }
    feature, confidence = router.route_task(test_input, ["nightmare", "chaos", "evolution"])
    print(f"Routed to: {feature} (confidence: {confidence:.2f})")

    # Test tenant quotas
    quota_mgr = TenantQuotaManager()
    quota_mgr.set_quota("tenant_001", "requests", 100, "daily")
    quota_mgr.set_quota("tenant_001", "tokens", 100000, "hourly")

    for i in range(5):
        quota_mgr.consume_quota("tenant_001", "requests", 1)

    status = quota_mgr.get_tenant_status("tenant_001")
    print("Tenant status:", json.dumps(status, indent=2))
