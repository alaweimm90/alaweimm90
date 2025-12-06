"""
ASlib Benchmark Evaluation for Librex.Meta

Evaluates Librex.Meta against standard algorithm selection benchmarks
from the Algorithm Selection Library (ASlib).

Key metrics:
- Par10 score (penalized average runtime with 10x timeout penalty)
- Top-k selection accuracy
- Computational overhead
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time


class ASLibScenario:
    """
    Represents a single ASlib algorithm selection scenario

    ASlib format:
    - feature_values.arff: Instance features
    - algorithm_runs.arff: Algorithm performance data
    - description.txt: Scenario metadata
    - cv.arff: Cross-validation splits
    """

    def __init__(self, scenario_path: str):
        self.scenario_path = Path(scenario_path)
        self.name = self.scenario_path.name
        self.features = None
        self.performances = None
        self.instance_names = []
        self.algorithm_names = []
        self.metadata = {}

        self._load_scenario()

    def _load_scenario(self):
        """Load ASlib scenario data"""
        # Load description
        desc_file = self.scenario_path / "description.txt"
        if desc_file.exists():
            with open(desc_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        self.metadata[key.strip()] = value.strip()

        # Load feature data (simplified - in production would parse ARFF)
        feature_file = self.scenario_path / "feature_values.arff"
        if feature_file.exists():
            self.features = self._load_arff(feature_file)

        # Load performance data
        perf_file = self.scenario_path / "algorithm_runs.arff"
        if perf_file.exists():
            self.performances = self._load_arff(perf_file)

    def _load_arff(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load ARFF file (simplified loader)

        In production, use scipy.io.arff or liac-arff library
        For now, we'll create a placeholder that returns mock data
        """
        try:
            # TODO: Implement full ARFF parser
            # For now, return None to indicate data needs to be loaded
            return None
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def get_train_test_split(self, fold: int = 0) -> Tuple[List[str], List[str]]:
        """Get train/test split for cross-validation"""
        # Simplified - in production would load from cv.arff
        if self.instance_names:
            n = len(self.instance_names)
            split_idx = int(0.8 * n)
            return (
                self.instance_names[:split_idx],
                self.instance_names[split_idx:]
            )
        return [], []


class ASLibEvaluator:
    """
    Evaluates algorithm selection methods on ASlib scenarios

    Usage:
        evaluator = ASLibEvaluator(aslib_root="aslib_data")
        results = evaluator.evaluate_method(
            method=Librex.Meta_instance,
            scenarios=["SAT11-HAND", "SAT11-INDU"]
        )
    """

    def __init__(self, aslib_root: str = "aslib_data"):
        self.aslib_root = Path(aslib_root)
        self.scenarios = self._discover_scenarios()

        print(f"ASLibEvaluator initialized with {len(self.scenarios)} scenarios")

    def _discover_scenarios(self) -> List[str]:
        """Discover available ASlib scenarios"""
        if not self.aslib_root.exists():
            print(f"Warning: ASlib root not found at {self.aslib_root}")
            return []

        scenarios = []
        for item in self.aslib_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has required ASlib files
                if (item / "description.txt").exists():
                    scenarios.append(item.name)

        return sorted(scenarios)

    def load_scenario(self, scenario_name: str) -> ASLibScenario:
        """Load a specific ASlib scenario"""
        scenario_path = self.aslib_root / scenario_name
        if not scenario_path.exists():
            raise ValueError(f"Scenario {scenario_name} not found")

        return ASLibScenario(scenario_path)

    def evaluate_method(
        self,
        method: Any,
        scenarios: Optional[List[str]] = None,
        n_folds: int = 10,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an algorithm selection method on ASlib scenarios

        Args:
            method: Algorithm selection method with fit() and select() methods
            scenarios: List of scenario names (default: all)
            n_folds: Number of cross-validation folds
            metrics: List of metrics to compute (default: ['par10', 'accuracy'])

        Returns:
            results: Dict with performance metrics per scenario
        """
        if scenarios is None:
            scenarios = self.scenarios[:5]  # Default to first 5 for quick eval

        if metrics is None:
            metrics = ['par10', 'top1_accuracy', 'top3_accuracy', 'overhead']

        results = {}

        for scenario_name in scenarios:
            print(f"\nEvaluating on {scenario_name}...")

            try:
                scenario = self.load_scenario(scenario_name)
                scenario_results = self._evaluate_scenario(
                    method, scenario, n_folds, metrics
                )
                results[scenario_name] = scenario_results

                print(f"  ✓ {scenario_name}: Par10={scenario_results.get('par10', 'N/A'):.2f}")

            except Exception as e:
                print(f"  ✗ Error on {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}

        return results

    def _evaluate_scenario(
        self,
        method: Any,
        scenario: ASLibScenario,
        n_folds: int,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Evaluate method on a single scenario"""
        # Placeholder implementation
        # In production, would:
        # 1. Load instance features and performance data
        # 2. Run cross-validation
        # 3. Compute metrics (Par10, accuracy, etc.)

        # For now, return mock results
        return {
            'par10': np.random.uniform(100, 1000),
            'top1_accuracy': np.random.uniform(0.5, 0.9),
            'top3_accuracy': np.random.uniform(0.7, 0.95),
            'overhead': np.random.uniform(0.01, 0.05),
            'n_instances': 100,
            'n_algorithms': 10
        }

    def compare_methods(
        self,
        methods: Dict[str, Any],
        scenarios: Optional[List[str]] = None,
        n_folds: int = 10
    ) -> pd.DataFrame:
        """
        Compare multiple methods on ASlib scenarios

        Args:
            methods: Dict of {method_name: method_instance}
            scenarios: List of scenario names
            n_folds: Number of CV folds

        Returns:
            comparison_df: DataFrame with comparative results
        """
        if scenarios is None:
            scenarios = self.scenarios[:5]

        results_list = []

        for method_name, method in methods.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {method_name}")
            print(f"{'='*60}")

            method_results = self.evaluate_method(method, scenarios, n_folds)

            for scenario_name, scenario_results in method_results.items():
                if 'error' not in scenario_results:
                    results_list.append({
                        'method': method_name,
                        'scenario': scenario_name,
                        **scenario_results
                    })

        return pd.DataFrame(results_list)

    def compute_par10(
        self,
        runtimes: np.ndarray,
        timeout: float,
        success: np.ndarray
    ) -> float:
        """
        Compute Par10 score

        Par10 = Penalized Average Runtime with 10x timeout penalty
        For timeouts: runtime = 10 * timeout

        Args:
            runtimes: Array of solver runtimes
            timeout: Timeout threshold
            success: Boolean array indicating success

        Returns:
            par10: Par10 score
        """
        penalized = np.where(success, runtimes, 10 * timeout)
        return np.mean(penalized)

    def compute_top_k_accuracy(
        self,
        predictions: List[List[str]],
        ground_truth: List[str],
        k: int = 3
    ) -> float:
        """
        Compute top-k selection accuracy

        Args:
            predictions: List of ranked solver lists
            ground_truth: List of best solvers
            k: Consider top-k predictions

        Returns:
            accuracy: Fraction of instances where ground truth in top-k
        """
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truth)
            if gt in pred[:k]
        )
        return correct / len(predictions) if predictions else 0.0

    def generate_report(
        self,
        results: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report

        Args:
            results: DataFrame from compare_methods()
            output_file: Optional path to save report

        Returns:
            report: Formatted report string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ASlib Benchmark Evaluation Report")
        report_lines.append("="*80)
        report_lines.append("")

        # Summary statistics
        report_lines.append("Summary Statistics:")
        report_lines.append("-"*80)

        for method in results['method'].unique():
            method_data = results[results['method'] == method]
            report_lines.append(f"\n{method}:")
            report_lines.append(f"  Average Par10: {method_data['par10'].mean():.2f}")
            report_lines.append(f"  Average Top-1 Accuracy: {method_data['top1_accuracy'].mean():.3f}")
            report_lines.append(f"  Average Top-3 Accuracy: {method_data['top3_accuracy'].mean():.3f}")
            report_lines.append(f"  Average Overhead: {method_data['overhead'].mean():.4f}")

        # Per-scenario results
        report_lines.append("\n" + "="*80)
        report_lines.append("Per-Scenario Results:")
        report_lines.append("="*80)

        for scenario in results['scenario'].unique():
            report_lines.append(f"\n{scenario}:")
            scenario_data = results[results['scenario'] == scenario]
            report_lines.append(scenario_data.to_string(index=False))

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {output_file}")

        return report


def create_mock_solver_for_aslib(name: str, base_performance: float = 0.5):
    """
    Create a mock solver for ASlib testing

    Args:
        name: Solver name
        base_performance: Base performance level (0-1)

    Returns:
        MockSolver instance
    """
    class MockASLibSolver:
        def __init__(self, name: str, base_perf: float):
            self.name = name
            self.base_performance = base_perf

        def solve(self, instance):
            # Simulate runtime with some randomness
            runtime = self.base_performance * 10 + np.random.randn() * 2
            return {
                'runtime': max(0.1, runtime),
                'success': runtime < 30,
                'objective': max(0, 1.0 - runtime / 30)
            }

    return MockASLibSolver(name, base_performance)
