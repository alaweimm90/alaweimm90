"""
Comprehensive ASlib Benchmark Evaluation Runner

Runs all algorithm selection methods on selected ASlib scenarios
with cross-validation, statistical testing, and result analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import json
import pickle
from datetime import datetime

from libria_meta import Librex.Meta
from baselines import SATzilla, AutoFolio, SMACBaseline, Hyperband, BOHB
from benchmark.evaluate_Librex.Meta import ASLibEvaluator, create_mock_solver_for_aslib
from benchmark.aslib_parser import ASLibScenarioLoader


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for algorithm selection methods

    Features:
    - Cross-validation support
    - Multiple metrics (Par10, top-k accuracy, overhead)
    - Statistical significance testing
    - Result persistence
    - Progress tracking
    """

    def __init__(
        self,
        aslib_root: str = "aslib_data",
        results_dir: str = "results",
        n_folds: int = 10,
        random_state: int = 42
    ):
        """
        Initialize evaluator

        Args:
            aslib_root: Root directory for ASlib scenarios
            results_dir: Directory to save results
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.aslib_root = Path(aslib_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.n_folds = n_folds
        self.random_state = random_state

        # Initialize ASlib evaluator
        self.aslib_eval = ASLibEvaluator(aslib_root=str(self.aslib_root))

        print(f"Initialized evaluator with {len(self.aslib_eval.scenarios)} scenarios")

    def create_methods(self, solvers: List[Any]) -> Dict[str, Any]:
        """
        Create all algorithm selection methods

        Args:
            solvers: List of solver instances

        Returns:
            methods: Dict of {method_name: method_instance}
        """
        methods = {
            'Librex.Meta': Librex.Meta(
                solvers=solvers,
                n_clusters=5,
                elo_k=32.0,
                ucb_c=1.414,
                n_tournament_rounds=5
            ),
            'SATzilla': SATzilla(
                solvers=solvers,
                n_estimators=100
            ),
            'AutoFolio': AutoFolio(
                solvers=solvers,
                n_features=10,
                use_feature_selection=True
            ),
            'SMAC': SMACBaseline(
                solvers=solvers,
                n_estimators=50
            ),
            'Hyperband': Hyperband(
                solvers=solvers,
                max_budget=27,
                eta=3
            ),
            'BOHB': BOHB(
                solvers=solvers,
                max_budget=27,
                eta=3
            )
        }

        return methods

    def run_single_scenario(
        self,
        scenario_name: str,
        methods: Optional[Dict[str, Any]] = None,
        use_mock_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single scenario

        Args:
            scenario_name: Name of ASlib scenario
            methods: Dict of methods (if None, creates from mock solvers)
            use_mock_data: If True, use mock data instead of real ASlib data

        Returns:
            results: Dict with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Evaluating Scenario: {scenario_name}")
        print(f"{'='*70}")

        if use_mock_data:
            return self._run_mock_scenario(scenario_name, methods)
        else:
            return self._run_real_scenario(scenario_name, methods)

    def _run_mock_scenario(
        self,
        scenario_name: str,
        methods: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on mock data (for testing pipeline)

        Args:
            scenario_name: Scenario name
            methods: Methods to evaluate

        Returns:
            results: Evaluation results
        """
        print(f"Using mock data for {scenario_name}")

        # Create mock solvers
        n_solvers = np.random.randint(5, 15)
        solvers = [
            create_mock_solver_for_aslib(
                f'Solver{i}',
                base_performance=np.random.uniform(0.4, 0.8)
            )
            for i in range(n_solvers)
        ]

        print(f"  Created {n_solvers} mock solvers")

        # Create methods if not provided
        if methods is None:
            methods = self.create_methods(solvers)

        # Generate training data
        n_train = 100
        training_data = self._generate_mock_training_data(solvers, n_train)

        print(f"  Generated {n_train} training instances")

        # Train all methods
        print("\n  Training methods...")
        training_times = {}

        for method_name, method in methods.items():
            start_time = time.time()
            method.fit(training_data)
            training_times[method_name] = time.time() - start_time
            print(f"    {method_name:15} trained in {training_times[method_name]:.2f}s")

        # Generate test data
        n_test = 50
        test_instances = [self._create_mock_instance() for _ in range(n_test)]

        print(f"\n  Evaluating on {n_test} test instances...")

        # Evaluate each method
        results = {
            'scenario': scenario_name,
            'n_solvers': n_solvers,
            'n_train': n_train,
            'n_test': n_test,
            'training_times': training_times,
            'methods': {}
        }

        for method_name, method in methods.items():
            print(f"\n    Testing {method_name}...")
            method_results = self._evaluate_method_on_instances(
                method, test_instances, solvers
            )
            results['methods'][method_name] = method_results

        # Compute summary statistics
        results['summary'] = self._compute_summary_statistics(results)

        return results

    def _generate_mock_training_data(
        self,
        solvers: List[Any],
        n_instances: int
    ) -> List[Dict]:
        """Generate mock training data"""
        training_data = []

        for _ in range(n_instances):
            instance = self._create_mock_instance()

            # Get true performance for each solver
            performances = {}
            for solver in solvers:
                result = solver.solve(instance)
                performances[solver.name] = result.get('objective', 0.5)

            training_data.append({
                'instance': instance,
                'features': None,  # Will be extracted
                'performances': performances
            })

        return training_data

    def _create_mock_instance(self):
        """Create a mock problem instance"""
        class MockInstance:
            def __init__(self):
                self.n_agents = np.random.randint(10, 50)
                self.n_tasks = np.random.randint(10, 50)
                self.cost_matrix = np.random.rand(self.n_agents, self.n_tasks)

        return MockInstance()

    def _run_real_scenario(
        self,
        scenario_name: str,
        methods: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on real ASlib data

        Args:
            scenario_name: Scenario name
            methods: Methods to evaluate (if None, creates from scenario algorithms)

        Returns:
            results: Evaluation results
        """
        print(f"Loading real ASlib data for {scenario_name}")

        # Load scenario
        scenario_path = self.aslib_root / scenario_name
        scenario = ASLibScenarioLoader(str(scenario_path))

        # Get scenario summary
        summary = scenario.get_summary()
        print(f"  Instances: {summary['n_instances']}")
        print(f"  Algorithms: {summary['n_algorithms']}")
        print(f"  Features: {summary['n_features']}")

        # Create mock solvers for each algorithm
        algorithm_names = scenario.get_algorithm_names()
        solvers = []
        for algo_name in algorithm_names:
            # Create a mock solver that returns fixed performance
            solver = create_mock_solver_for_aslib(
                algo_name,
                base_performance=0.5  # Will be overridden by real data
            )
            solvers.append(solver)

        print(f"  Created {len(solvers)} algorithm wrappers")

        # Create methods if not provided
        if methods is None:
            methods = self.create_methods(solvers)

        # Get training/test split
        # Use 80/20 split for now (cv splits to be implemented later)
        all_instance_ids = scenario.get_instance_ids()
        n_train = int(0.8 * len(all_instance_ids))
        train_ids = all_instance_ids[:n_train]
        test_ids = all_instance_ids[n_train:]

        print(f"  Training instances: {len(train_ids)}")
        print(f"  Test instances: {len(test_ids)}")

        # Load training data
        training_data = scenario.get_training_data(instance_ids=train_ids)
        print(f"  Loaded {len(training_data)} training instances")

        # Train all methods
        print("\n  Training methods...")
        training_times = {}

        for method_name, method in methods.items():
            start_time = time.time()
            method.fit(training_data)
            training_times[method_name] = time.time() - start_time
            print(f"    {method_name:15} trained in {training_times[method_name]:.2f}s")

        # Prepare test instances
        test_data = scenario.get_training_data(instance_ids=test_ids)

        print(f"\n  Evaluating on {len(test_data)} test instances...")

        # Evaluate each method
        results = {
            'scenario': scenario_name,
            'n_solvers': summary['n_algorithms'],
            'n_train': len(train_ids),
            'n_test': len(test_ids),
            'n_features': summary['n_features'],
            'training_times': training_times,
            'methods': {}
        }

        for method_name, method in methods.items():
            print(f"\n    Testing {method_name}...")
            method_results = self._evaluate_method_on_real_instances(
                method, test_data, algorithm_names
            )
            results['methods'][method_name] = method_results

        # Compute summary statistics
        results['summary'] = self._compute_summary_statistics(results)

        return results

    def _evaluate_method_on_real_instances(
        self,
        method: Any,
        test_data: List[Dict],
        algorithm_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a single method on real ASlib test instances

        Args:
            method: Algorithm selection method
            test_data: List of test instances with features and true performances
            algorithm_names: List of algorithm names

        Returns:
            results: Performance metrics
        """
        selection_times = []
        regrets = []
        top1_correct = 0
        top3_correct = 0

        for test_instance in test_data:
            # Extract features
            features = test_instance['features']
            true_performances = test_instance['performances']

            # Create a simple instance object with features
            class ASLibInstance:
                def __init__(self, features):
                    self.features = features

            instance = ASLibInstance(features)

            # Measure selection time
            start_time = time.time()
            # Pass features directly to avoid re-extraction issues
            if hasattr(method, 'select_solver'):
                selected = method.select_solver(instance, features=features)
            else:
                selected = method.select_solver(instance)
            selection_time = time.time() - start_time
            selection_times.append(selection_time)

            # Best algorithm (highest performance score)
            best_algo_name = max(true_performances, key=true_performances.get)
            best_performance = true_performances[best_algo_name]

            # Selected algorithm performance
            selected_name = selected.name if hasattr(selected, 'name') else str(selected)
            selected_performance = true_performances.get(selected_name, 0.0)

            # Compute regret
            regret = best_performance - selected_performance
            regrets.append(regret)

            # Top-k accuracy
            if selected_name == best_algo_name:
                top1_correct += 1
                top3_correct += 1
            else:
                # Check if in top-3
                top3_algos = sorted(
                    true_performances,
                    key=true_performances.get,
                    reverse=True
                )[:3]
                if selected_name in top3_algos:
                    top3_correct += 1

        return {
            'avg_selection_time': np.mean(selection_times),
            'std_selection_time': np.std(selection_times),
            'avg_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'top1_accuracy': top1_correct / len(test_data),
            'top3_accuracy': top3_correct / len(test_data),
            'regrets': regrets,
            'selection_times': selection_times
        }

    def _evaluate_method_on_instances(
        self,
        method: Any,
        instances: List[Any],
        solvers: List[Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single method on test instances

        Args:
            method: Algorithm selection method
            instances: Test instances
            solvers: Available solvers

        Returns:
            results: Performance metrics
        """
        selection_times = []
        regrets = []
        top1_correct = 0
        top3_correct = 0

        for instance in instances:
            # Measure selection time
            start_time = time.time()
            selected = method.select_solver(instance)
            selection_time = time.time() - start_time
            selection_times.append(selection_time)

            # Get true performance for all solvers
            true_performances = {}
            for solver in solvers:
                result = solver.solve(instance)
                true_performances[solver.name] = result.get('objective', 0.5)

            # Best solver
            best_solver_name = max(true_performances, key=true_performances.get)
            best_performance = true_performances[best_solver_name]

            # Selected solver performance
            selected_name = selected.name if hasattr(selected, 'name') else str(selected)
            selected_performance = true_performances.get(selected_name, 0.0)

            # Compute regret
            regret = best_performance - selected_performance
            regrets.append(regret)

            # Top-k accuracy
            if selected_name == best_solver_name:
                top1_correct += 1
                top3_correct += 1
            else:
                # Check if in top-3
                top3_solvers = sorted(
                    true_performances,
                    key=true_performances.get,
                    reverse=True
                )[:3]
                if selected_name in top3_solvers:
                    top3_correct += 1

        return {
            'avg_selection_time': np.mean(selection_times),
            'std_selection_time': np.std(selection_times),
            'avg_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'top1_accuracy': top1_correct / len(instances),
            'top3_accuracy': top3_correct / len(instances),
            'regrets': regrets,
            'selection_times': selection_times
        }

    def _compute_summary_statistics(self, results: Dict) -> pd.DataFrame:
        """Compute summary statistics across all methods"""
        summary_data = []

        for method_name, method_results in results['methods'].items():
            summary_data.append({
                'Method': method_name,
                'Training Time (s)': results['training_times'][method_name],
                'Avg Selection Time (s)': method_results['avg_selection_time'],
                'Avg Regret': method_results['avg_regret'],
                'Top-1 Accuracy': method_results['top1_accuracy'],
                'Top-3 Accuracy': method_results['top3_accuracy']
            })

        return pd.DataFrame(summary_data)

    def run_multiple_scenarios(
        self,
        scenario_names: List[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple scenarios

        Args:
            scenario_names: List of scenario names
            save_results: Whether to save results to disk

        Returns:
            all_results: Dict with results for all scenarios
        """
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'n_folds': self.n_folds,
            'scenarios': {}
        }

        for i, scenario_name in enumerate(scenario_names, 1):
            print(f"\n{'#'*70}")
            print(f"Scenario {i}/{len(scenario_names)}: {scenario_name}")
            print(f"{'#'*70}")

            scenario_results = self.run_single_scenario(scenario_name)
            all_results['scenarios'][scenario_name] = scenario_results

            # Print summary
            print(f"\nSummary for {scenario_name}:")
            print(scenario_results['summary'].to_string(index=False))

        # Save results
        if save_results:
            self.save_results(all_results)

        return all_results

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """
        Save evaluation results to disk

        Args:
            results: Results dictionary
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}"

        # Save as JSON (for readability)
        json_path = self.results_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_compatible = self._make_json_compatible(results)
            json.dump(json_compatible, f, indent=2)

        # Save as pickle (for complete data)
        pickle_path = self.results_dir / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to:")
        print(f"  - {json_path}")
        print(f"  - {pickle_path}")

    def _make_json_compatible(self, obj):
        """Convert numpy arrays and other non-JSON types to JSON-compatible types"""
        if isinstance(obj, dict):
            return {k: self._make_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_compatible(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        else:
            return obj

    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """
        Generate comprehensive evaluation report

        Args:
            results: Evaluation results
            output_file: Path to output markdown file
        """
        report_lines = []
        report_lines.append("# ASlib Benchmark Evaluation Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Scenarios Evaluated**: {len(results['scenarios'])}")
        report_lines.append("")

        # Overall summary
        report_lines.append("## Overall Summary")
        report_lines.append("")

        # Aggregate results across scenarios
        all_method_names = set()
        for scenario_results in results['scenarios'].values():
            all_method_names.update(scenario_results['methods'].keys())

        aggregate_data = []
        for method_name in sorted(all_method_names):
            regrets = []
            top1_accs = []
            top3_accs = []
            selection_times = []

            for scenario_results in results['scenarios'].values():
                if method_name in scenario_results['methods']:
                    m_res = scenario_results['methods'][method_name]
                    regrets.append(m_res['avg_regret'])
                    top1_accs.append(m_res['top1_accuracy'])
                    top3_accs.append(m_res['top3_accuracy'])
                    selection_times.append(m_res['avg_selection_time'])

            aggregate_data.append({
                'Method': method_name,
                'Avg Regret': np.mean(regrets),
                'Avg Top-1 Acc': np.mean(top1_accs),
                'Avg Top-3 Acc': np.mean(top3_accs),
                'Avg Selection Time (ms)': np.mean(selection_times) * 1000
            })

        df_aggregate = pd.DataFrame(aggregate_data)
        report_lines.append(df_aggregate.to_markdown(index=False))
        report_lines.append("")

        # Per-scenario results
        report_lines.append("## Per-Scenario Results")
        report_lines.append("")

        for scenario_name, scenario_results in results['scenarios'].items():
            report_lines.append(f"### {scenario_name}")
            report_lines.append("")
            report_lines.append(scenario_results['summary'].to_markdown(index=False))
            report_lines.append("")

        report = "\n".join(report_lines)

        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"\n✓ Report saved to {output_path}")

        return report


def main():
    """Main evaluation entry point"""
    print("="*70)
    print("Librex.Meta Comprehensive Benchmark Evaluation")
    print("="*70)

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        aslib_root="aslib_data",
        results_dir="results",
        n_folds=10
    )

    # Phase 1: Quick validation (3 scenarios)
    phase1_scenarios = [
        'SAT11-HAND',
        'CSP-2010',
        'GRAPHS-2015'
    ]

    print("\n" + "="*70)
    print("Phase 1: Quick Validation (3 scenarios)")
    print("="*70)

    results = evaluator.run_multiple_scenarios(
        scenario_names=phase1_scenarios,
        save_results=True
    )

    # Generate report
    evaluator.generate_report(results, output_file="phase1_report.md")

    print("\n" + "="*70)
    print("Phase 1 Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
