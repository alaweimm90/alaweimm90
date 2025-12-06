"""Cross-Validation Framework for MEZAN.

This module provides comprehensive cross-validation capabilities for comparing
MEZAN against baselines, solver selection validation, benchmark replay, and
performance regression detection.

Author: Meshal Alawein
Date: 2025-11-18
"""

import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
)

from .validation import StatisticalValidator, ValidationResult


class ComparisonMethod(Enum):
    """Methods for comparing systems."""
    HEAD_TO_HEAD = "head_to_head"
    RELATIVE_IMPROVEMENT = "relative_improvement"
    STATISTICAL_TEST = "statistical_test"
    PARETO_DOMINANCE = "pareto_dominance"
    NORMALIZED_SCORE = "normalized_score"


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    COST = "cost"
    QUALITY = "quality"


class FoldStrategy(Enum):
    """Cross-validation folding strategies."""
    K_FOLD = "k_fold"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    LEAVE_ONE_OUT = "leave_one_out"
    MONTE_CARLO = "monte_carlo"


@dataclass
class BaselineSystem:
    """Represents a baseline system for comparison."""
    name: str
    version: str
    executor: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCase:
    """Represents a benchmark test case."""
    id: str
    name: str
    category: str
    input_data: Any
    expected_output: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a system run."""
    execution_time: float
    accuracy: Optional[float] = None
    throughput: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'accuracy': self.accuracy,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            **self.custom_metrics
        }


@dataclass
class CrossValidationResult:
    """Result from cross-validation."""
    method: FoldStrategy
    n_folds: int
    scores: List[float]
    mean_score: float
    std_score: float
    fold_details: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComparisonResult:
    """Result from system comparison."""
    mezan_metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics
    comparison_method: ComparisonMethod
    is_better: bool
    improvement_percentage: float
    statistical_significance: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class CrossValidator:
    """Main cross-validation engine for MEZAN."""

    def __init__(self,
                 mezan_executor: Callable,
                 storage_path: Optional[Path] = None):
        """
        Initialize the cross-validator.

        Args:
            mezan_executor: Function to execute MEZAN
            storage_path: Path for storing results
        """
        self.mezan_executor = mezan_executor
        self.baselines: Dict[str, BaselineSystem] = {}
        self.benchmarks: List[BenchmarkCase] = []
        self.results_history: List[Dict[str, Any]] = []
        self.storage_path = storage_path or Path("cross_validation_results")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.validator = StatisticalValidator()

    def add_baseline(self, baseline: BaselineSystem):
        """Add a baseline system for comparison."""
        self.baselines[baseline.name] = baseline

    def add_benchmark(self, benchmark: BenchmarkCase):
        """Add a benchmark test case."""
        self.benchmarks.append(benchmark)

    def load_benchmarks(self, file_path: Path):
        """Load benchmarks from file."""
        with open(file_path) as f:
            data = json.load(f)

        for bench_data in data['benchmarks']:
            benchmark = BenchmarkCase(
                id=bench_data['id'],
                name=bench_data['name'],
                category=bench_data.get('category', 'general'),
                input_data=bench_data['input'],
                expected_output=bench_data.get('expected'),
                constraints=bench_data.get('constraints', {}),
                metadata=bench_data.get('metadata', {})
            )
            self.benchmarks.append(benchmark)

    def cross_validate_solver_selection(self,
                                       solver_selector: Callable,
                                       problems: List[Dict],
                                       labels: List[str],
                                       strategy: FoldStrategy = FoldStrategy.K_FOLD,
                                       n_folds: int = 5) -> CrossValidationResult:
        """
        Cross-validate solver selection strategy.

        Args:
            solver_selector: Function that selects a solver for a problem
            problems: List of problem instances
            labels: Ground truth best solvers for each problem
            strategy: Folding strategy to use
            n_folds: Number of folds

        Returns:
            Cross-validation results
        """
        # Convert problems and labels to arrays
        X = np.array(problems)
        y = np.array(labels)

        # Select fold strategy
        if strategy == FoldStrategy.K_FOLD:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == FoldStrategy.STRATIFIED:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == FoldStrategy.TIME_SERIES:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        scores = []
        fold_details = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train solver selector (if it has a train method)
            if hasattr(solver_selector, 'fit'):
                solver_selector.fit(X_train, y_train)

            # Predict on test set
            predictions = []
            for problem in X_test:
                selected_solver = solver_selector(problem)
                predictions.append(selected_solver)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)
            scores.append(accuracy)

            # Store fold details
            fold_details.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': accuracy,
                'predictions': predictions,
                'ground_truth': y_test.tolist()
            })

        result = CrossValidationResult(
            method=strategy,
            n_folds=n_folds,
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details
        )

        self._save_result('solver_selection_cv', result)
        return result

    def compare_against_baseline(self,
                                baseline_name: str,
                                benchmark_ids: Optional[List[str]] = None,
                                comparison_method: ComparisonMethod = ComparisonMethod.HEAD_TO_HEAD
                                ) -> List[ComparisonResult]:
        """
        Compare MEZAN against a baseline system.

        Args:
            baseline_name: Name of the baseline system
            benchmark_ids: Specific benchmarks to run (None = all)
            comparison_method: Method for comparison

        Returns:
            List of comparison results
        """
        if baseline_name not in self.baselines:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        baseline = self.baselines[baseline_name]
        results = []

        # Select benchmarks
        benchmarks_to_run = self.benchmarks
        if benchmark_ids:
            benchmarks_to_run = [b for b in self.benchmarks if b.id in benchmark_ids]

        for benchmark in benchmarks_to_run:
            # Run MEZAN
            mezan_metrics = self._run_system(self.mezan_executor, benchmark)

            # Run baseline
            baseline_metrics = self._run_system(baseline.executor, benchmark)

            # Compare results
            comparison = self._compare_metrics(
                mezan_metrics,
                baseline_metrics,
                comparison_method
            )

            results.append(comparison)

        # Save results
        self._save_comparison_results(baseline_name, results)

        return results

    def _run_system(self,
                   executor: Callable,
                   benchmark: BenchmarkCase) -> PerformanceMetrics:
        """Run a system on a benchmark and collect metrics."""
        import psutil
        import resource

        # Setup monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()

        # Run system
        start_time = time.time()
        try:
            output = executor(benchmark.input_data)
            execution_time = time.time() - start_time

            # Collect resource metrics
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu = process.cpu_percent()

            # Calculate accuracy if expected output provided
            accuracy = None
            if benchmark.expected_output is not None:
                accuracy = self._calculate_accuracy(output, benchmark.expected_output)

            # Calculate throughput
            throughput = 1.0 / execution_time if execution_time > 0 else 0

            return PerformanceMetrics(
                execution_time=execution_time,
                accuracy=accuracy,
                throughput=throughput,
                memory_usage=end_memory - start_memory,
                cpu_usage=(end_cpu + start_cpu) / 2
            )

        except Exception as e:
            # Handle execution failure
            return PerformanceMetrics(
                execution_time=float('inf'),
                accuracy=0.0,
                throughput=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                custom_metrics={'error': str(e)}
            )

    def _calculate_accuracy(self, output: Any, expected: Any) -> float:
        """Calculate accuracy between output and expected result."""
        if isinstance(output, (list, np.ndarray)) and isinstance(expected, (list, np.ndarray)):
            # For arrays, calculate element-wise accuracy
            output = np.array(output)
            expected = np.array(expected)
            return np.mean(output == expected)
        elif isinstance(output, (int, float)) and isinstance(expected, (int, float)):
            # For numerical values, calculate relative error
            if expected != 0:
                return 1.0 - abs(output - expected) / abs(expected)
            else:
                return 1.0 if output == expected else 0.0
        else:
            # Direct comparison
            return 1.0 if output == expected else 0.0

    def _compare_metrics(self,
                        mezan: PerformanceMetrics,
                        baseline: PerformanceMetrics,
                        method: ComparisonMethod) -> ComparisonResult:
        """Compare performance metrics between systems."""
        if method == ComparisonMethod.HEAD_TO_HEAD:
            # Simple comparison of execution time
            is_better = mezan.execution_time < baseline.execution_time
            improvement = ((baseline.execution_time - mezan.execution_time) /
                         baseline.execution_time * 100 if baseline.execution_time > 0 else 0)

        elif method == ComparisonMethod.RELATIVE_IMPROVEMENT:
            # Compare multiple metrics with weights
            mezan_score = self._calculate_composite_score(mezan)
            baseline_score = self._calculate_composite_score(baseline)
            is_better = mezan_score > baseline_score
            improvement = ((mezan_score - baseline_score) /
                         baseline_score * 100 if baseline_score > 0 else 0)

        elif method == ComparisonMethod.STATISTICAL_TEST:
            # Requires multiple runs for statistical comparison
            # This is simplified - would need multiple samples
            is_better = mezan.execution_time < baseline.execution_time
            improvement = ((baseline.execution_time - mezan.execution_time) /
                         baseline.execution_time * 100 if baseline.execution_time > 0 else 0)
            # Would calculate p-value with multiple samples

        elif method == ComparisonMethod.PARETO_DOMINANCE:
            # Check Pareto dominance across multiple objectives
            dominates = True
            strictly_better = False

            metrics_to_compare = ['execution_time', 'memory_usage', 'cpu_usage']
            for metric in metrics_to_compare:
                mezan_val = getattr(mezan, metric, float('inf'))
                baseline_val = getattr(baseline, metric, float('inf'))

                if mezan_val > baseline_val:
                    dominates = False
                    break
                elif mezan_val < baseline_val:
                    strictly_better = True

            is_better = dominates and strictly_better
            improvement = 0  # Pareto comparison doesn't have single improvement value

        else:
            raise ValueError(f"Unknown comparison method: {method}")

        return ComparisonResult(
            mezan_metrics=mezan,
            baseline_metrics=baseline,
            comparison_method=method,
            is_better=is_better,
            improvement_percentage=improvement
        )

    def _calculate_composite_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate composite score from multiple metrics."""
        # Weighted combination of metrics (lower is better for time/resource)
        score = 0
        weights = {
            'execution_time': -0.4,  # Negative because lower is better
            'accuracy': 0.3,
            'throughput': 0.2,
            'memory_usage': -0.05,
            'cpu_usage': -0.05
        }

        for metric, weight in weights.items():
            value = getattr(metrics, metric, 0)
            if value is not None:
                score += weight * value

        return score

    def benchmark_replay(self,
                        recorded_results_path: Path,
                        validate_reproducibility: bool = True) -> Dict[str, Any]:
        """
        Replay recorded benchmark results and validate.

        Args:
            recorded_results_path: Path to recorded results
            validate_reproducibility: Whether to check reproducibility

        Returns:
            Replay analysis results
        """
        # Load recorded results
        with open(recorded_results_path) as f:
            recorded = json.load(f)

        replay_results = []
        reproducibility_scores = []

        for record in recorded['benchmarks']:
            benchmark_id = record['benchmark_id']
            benchmark = next((b for b in self.benchmarks if b.id == benchmark_id), None)

            if not benchmark:
                continue

            # Re-run benchmark
            current_metrics = self._run_system(self.mezan_executor, benchmark)

            # Compare with recorded
            recorded_metrics = PerformanceMetrics(
                execution_time=record['execution_time'],
                accuracy=record.get('accuracy'),
                throughput=record.get('throughput'),
                memory_usage=record.get('memory_usage'),
                cpu_usage=record.get('cpu_usage')
            )

            # Calculate reproducibility
            if validate_reproducibility:
                repro_score = self._calculate_reproducibility(current_metrics, recorded_metrics)
                reproducibility_scores.append(repro_score)

            replay_results.append({
                'benchmark_id': benchmark_id,
                'recorded': recorded_metrics.to_dict(),
                'current': current_metrics.to_dict(),
                'reproducibility': repro_score if validate_reproducibility else None
            })

        analysis = {
            'total_benchmarks': len(replay_results),
            'mean_reproducibility': np.mean(reproducibility_scores) if reproducibility_scores else None,
            'std_reproducibility': np.std(reproducibility_scores) if reproducibility_scores else None,
            'results': replay_results
        }

        return analysis

    def _calculate_reproducibility(self,
                                  current: PerformanceMetrics,
                                  recorded: PerformanceMetrics) -> float:
        """Calculate reproducibility score between runs."""
        scores = []

        # Compare execution time (allow 10% variance)
        if recorded.execution_time > 0:
            time_diff = abs(current.execution_time - recorded.execution_time)
            time_score = max(0, 1 - time_diff / recorded.execution_time)
            scores.append(time_score)

        # Compare accuracy (should be nearly identical)
        if recorded.accuracy is not None and current.accuracy is not None:
            acc_score = 1 - abs(current.accuracy - recorded.accuracy)
            scores.append(acc_score)

        # Compare resource usage (allow 20% variance)
        if recorded.memory_usage is not None and current.memory_usage is not None:
            if recorded.memory_usage > 0:
                mem_diff = abs(current.memory_usage - recorded.memory_usage)
                mem_score = max(0, 1 - mem_diff / recorded.memory_usage * 0.5)
                scores.append(mem_score)

        return np.mean(scores) if scores else 0.0

    def detect_performance_regression(self,
                                     historical_results: List[Dict],
                                     current_results: Dict,
                                     threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect performance regressions.

        Args:
            historical_results: Historical benchmark results
            current_results: Current benchmark results
            threshold: Regression threshold (e.g., 0.1 = 10% worse)

        Returns:
            Regression analysis results
        """
        regressions = []
        improvements = []

        # Group historical results by benchmark
        historical_by_benchmark = defaultdict(list)
        for result in historical_results:
            historical_by_benchmark[result['benchmark_id']].append(result)

        # Compare current with historical
        for benchmark_id, current_metric in current_results.items():
            if benchmark_id not in historical_by_benchmark:
                continue

            historical = historical_by_benchmark[benchmark_id]

            # Calculate historical statistics
            hist_times = [h['execution_time'] for h in historical]
            hist_mean = np.mean(hist_times)
            hist_std = np.std(hist_times)

            # Check for regression
            current_time = current_metric['execution_time']
            z_score = (current_time - hist_mean) / hist_std if hist_std > 0 else 0

            if current_time > hist_mean * (1 + threshold):
                regressions.append({
                    'benchmark_id': benchmark_id,
                    'historical_mean': hist_mean,
                    'current': current_time,
                    'degradation_percentage': (current_time - hist_mean) / hist_mean * 100,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'low'
                })
            elif current_time < hist_mean * (1 - threshold):
                improvements.append({
                    'benchmark_id': benchmark_id,
                    'historical_mean': hist_mean,
                    'current': current_time,
                    'improvement_percentage': (hist_mean - current_time) / hist_mean * 100,
                    'z_score': z_score
                })

        # Statistical analysis
        regression_detected = len(regressions) > 0
        overall_trend = 'regression' if regression_detected else 'stable'

        if len(improvements) > len(regressions):
            overall_trend = 'improvement'

        return {
            'overall_trend': overall_trend,
            'regressions': regressions,
            'improvements': improvements,
            'total_benchmarks': len(current_results),
            'regression_count': len(regressions),
            'improvement_count': len(improvements),
            'threshold_used': threshold
        }

    def generate_comparison_report(self,
                                  baseline_name: str,
                                  output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.

        Args:
            baseline_name: Name of baseline to compare against
            output_path: Path to save report

        Returns:
            Comparison report dictionary
        """
        # Run full comparison
        comparison_results = self.compare_against_baseline(baseline_name)

        # Aggregate statistics
        mezan_wins = sum(1 for r in comparison_results if r.is_better)
        baseline_wins = len(comparison_results) - mezan_wins

        improvements = [r.improvement_percentage for r in comparison_results]

        report = {
            'summary': {
                'total_benchmarks': len(comparison_results),
                'mezan_wins': mezan_wins,
                'baseline_wins': baseline_wins,
                'win_rate': mezan_wins / len(comparison_results) * 100 if comparison_results else 0,
                'mean_improvement': np.mean(improvements),
                'median_improvement': np.median(improvements),
                'std_improvement': np.std(improvements)
            },
            'detailed_results': [
                {
                    'mezan': r.mezan_metrics.to_dict(),
                    'baseline': r.baseline_metrics.to_dict(),
                    'is_better': r.is_better,
                    'improvement': r.improvement_percentage
                }
                for r in comparison_results
            ],
            'statistical_analysis': self._perform_statistical_analysis(comparison_results)
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _perform_statistical_analysis(self,
                                     results: List[ComparisonResult]) -> Dict[str, Any]:
        """Perform statistical analysis on comparison results."""
        mezan_times = [r.mezan_metrics.execution_time for r in results]
        baseline_times = [r.baseline_metrics.execution_time for r in results]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(mezan_times, baseline_times)

        # Effect size (Cohen's d)
        diff = np.array(mezan_times) - np.array(baseline_times)
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(mezan_times, baseline_times)

        return {
            'paired_t_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': effect_size,
            'effect_size_interpretation': (
                'large' if abs(effect_size) > 0.8 else
                'medium' if abs(effect_size) > 0.5 else
                'small' if abs(effect_size) > 0.2 else
                'negligible'
            ),
            'wilcoxon_test': {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'significant': wilcoxon_p < 0.05
            }
        }

    def visualize_comparison(self,
                            comparison_results: List[ComparisonResult],
                            save_path: Optional[Path] = None):
        """
        Create visualizations of comparison results.

        Args:
            comparison_results: Results to visualize
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Execution time comparison
        mezan_times = [r.mezan_metrics.execution_time for r in comparison_results]
        baseline_times = [r.baseline_metrics.execution_time for r in comparison_results]

        axes[0, 0].scatter(baseline_times, mezan_times, alpha=0.6)
        axes[0, 0].plot([0, max(baseline_times)], [0, max(baseline_times)],
                       'r--', label='Equal performance')
        axes[0, 0].set_xlabel('Baseline Execution Time (s)')
        axes[0, 0].set_ylabel('MEZAN Execution Time (s)')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].legend()

        # Improvement distribution
        improvements = [r.improvement_percentage for r in comparison_results]
        axes[0, 1].hist(improvements, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', label='No improvement')
        axes[0, 1].set_xlabel('Improvement (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Improvement Distribution')
        axes[0, 1].legend()

        # Win rate by category (if available)
        if hasattr(comparison_results[0], 'category'):
            categories = defaultdict(lambda: {'wins': 0, 'total': 0})
            for r in comparison_results:
                categories[r.category]['total'] += 1
                if r.is_better:
                    categories[r.category]['wins'] += 1

            cat_names = list(categories.keys())
            win_rates = [categories[c]['wins'] / categories[c]['total'] * 100
                        for c in cat_names]

            axes[1, 0].bar(cat_names, win_rates)
            axes[1, 0].axhline(y=50, color='r', linestyle='--', label='50% win rate')
            axes[1, 0].set_xlabel('Category')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].set_title('Win Rate by Category')
            axes[1, 0].legend()

        # Resource usage comparison
        if all(r.mezan_metrics.memory_usage is not None for r in comparison_results):
            mezan_memory = [r.mezan_metrics.memory_usage for r in comparison_results]
            baseline_memory = [r.baseline_metrics.memory_usage for r in comparison_results]

            axes[1, 1].boxplot([baseline_memory, mezan_memory],
                              labels=['Baseline', 'MEZAN'])
            axes[1, 1].set_ylabel('Memory Usage (MB)')
            axes[1, 1].set_title('Memory Usage Comparison')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def _save_result(self, name: str, result: Any):
        """Save result to storage."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.storage_path / f"{name}_{timestamp}.pkl"

        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    def _save_comparison_results(self, baseline_name: str, results: List[ComparisonResult]):
        """Save comparison results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.storage_path / f"comparison_{baseline_name}_{timestamp}.json"

        data = {
            'baseline': baseline_name,
            'timestamp': timestamp,
            'results': [
                {
                    'mezan': r.mezan_metrics.to_dict(),
                    'baseline': r.baseline_metrics.to_dict(),
                    'is_better': r.is_better,
                    'improvement': r.improvement_percentage,
                    'method': r.comparison_method.value
                }
                for r in results
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions
def quick_cross_validation(data: List[Dict],
                          labels: List[Any],
                          predictor: Callable,
                          n_folds: int = 5) -> Dict[str, Any]:
    """
    Quick cross-validation helper.

    Args:
        data: Input data
        labels: Ground truth labels
        predictor: Prediction function
        n_folds: Number of folds

    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.base import BaseEstimator

    class PredictorWrapper(BaseEstimator):
        def __init__(self, predictor):
            self.predictor = predictor

        def fit(self, X, y):
            return self

        def predict(self, X):
            return [self.predictor(x) for x in X]

    wrapper = PredictorWrapper(predictor)
    scores = cross_val_score(wrapper, data, labels, cv=n_folds)

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores.tolist(),
        'n_folds': n_folds
    }


def compare_systems_quick(system1_results: List[float],
                         system2_results: List[float],
                         system1_name: str = "System1",
                         system2_name: str = "System2") -> Dict[str, Any]:
    """
    Quick comparison between two systems.

    Args:
        system1_results: Performance metrics for system 1
        system2_results: Performance metrics for system 2
        system1_name: Name of system 1
        system2_name: Name of system 2

    Returns:
        Comparison results
    """
    validator = StatisticalValidator()
    result = validator.t_test(np.array(system1_results), np.array(system2_results))

    return {
        f'{system1_name}_mean': np.mean(system1_results),
        f'{system2_name}_mean': np.mean(system2_results),
        'p_value': result.p_value,
        'is_significant': result.is_significant,
        'effect_size': result.effect_size,
        'better_system': system1_name if np.mean(system1_results) < np.mean(system2_results) else system2_name
    }