"""
Phase 2 Evaluation: Multi-Scenario Benchmark

Tests Librex.Meta (default vs optimal config) against all baselines
on 5 ASlib scenarios to identify strengths and weaknesses.

Scenarios:
1. SAT11-HAND - Boolean satisfiability (hard)
2. CSP-2010 - Constraint satisfaction (easy)
3. GRAPHS-2015 - Graph problems
4. MAXSAT12-PMS - Maximum satisfiability
5. ASP-POTASSCO - Answer set programming
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import time
import json

from benchmark.run_evaluation import ComprehensiveEvaluator
from libria_meta import Librex.Meta
from baselines import SATzilla, AutoFolio, SMACBaseline, Hyperband, BOHB
from benchmark.aslib_parser import ASLibScenarioLoader
from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib


class Phase2Evaluator:
    """
    Phase 2 multi-scenario evaluation framework
    """

    def __init__(
        self,
        aslib_root: str = "aslib_data",
        results_dir: str = "results/phase2"
    ):
        self.aslib_root = Path(aslib_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_scenario(
        self,
        scenario_name: str,
        test_optimal_config: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate all methods on a single scenario

        Args:
            scenario_name: Name of ASlib scenario
            test_optimal_config: If True, test both default and optimal Librex.Meta

        Returns:
            results: Dict with results for all methods
        """
        print(f"\n{'='*70}")
        print(f"Evaluating Scenario: {scenario_name}")
        print(f"{'='*70}")

        # Load scenario
        scenario_path = self.aslib_root / scenario_name
        scenario = ASLibScenarioLoader(str(scenario_path))

        summary = scenario.get_summary()
        print(f"  Instances: {summary['n_instances']}")
        print(f"  Algorithms: {summary['n_algorithms']}")
        print(f"  Features: {summary['n_features']}")

        # Create mock solvers
        algorithm_names = scenario.get_algorithm_names()
        solvers = [
            create_mock_solver_for_aslib(name, base_performance=0.5)
            for name in algorithm_names
        ]

        # Get train/test split
        all_instance_ids = scenario.get_instance_ids()
        n_train = int(0.8 * len(all_instance_ids))
        train_ids = all_instance_ids[:n_train]
        test_ids = all_instance_ids[n_train:]

        print(f"  Training: {len(train_ids)} instances")
        print(f"  Testing: {len(test_ids)} instances")

        # Load training data
        training_data = scenario.get_training_data(instance_ids=train_ids)

        # Create all methods
        methods = {}

        # Librex.Meta - Default configuration
        methods['Librex.Meta (default)'] = Librex.Meta(
            solvers=solvers,
            n_clusters=5,
            elo_k=32.0,
            ucb_c=1.414,
            n_tournament_rounds=5
        )

        # Librex.Meta - Optimal configuration (if requested)
        if test_optimal_config:
            methods['Librex.Meta (optimal)'] = Librex.Meta(
                solvers=solvers,
                n_clusters=3,      # Week 6 optimal
                elo_k=32.0,
                ucb_c=1.414,
                n_tournament_rounds=5
            )

        # Baselines
        methods['SATzilla'] = SATzilla(solvers=solvers, n_estimators=100)
        methods['AutoFolio'] = AutoFolio(solvers=solvers, n_features=10, use_feature_selection=True)
        methods['SMAC'] = SMACBaseline(solvers=solvers, n_estimators=50)
        methods['Hyperband'] = Hyperband(solvers=solvers, max_budget=27, eta=3)
        methods['BOHB'] = BOHB(solvers=solvers, max_budget=27, eta=3)

        # Train all methods
        print("\n  Training methods...")
        training_times = {}

        for method_name, method in methods.items():
            start_time = time.time()
            try:
                method.fit(training_data)
                training_times[method_name] = time.time() - start_time
                print(f"    {method_name:25} trained in {training_times[method_name]:.2f}s")
            except Exception as e:
                print(f"    {method_name:25} FAILED: {e}")
                training_times[method_name] = -1

        # Test all methods
        test_data = scenario.get_training_data(instance_ids=test_ids)

        print(f"\n  Evaluating on {len(test_data)} test instances...")

        results = {
            'scenario': scenario_name,
            'n_instances': summary['n_instances'],
            'n_algorithms': summary['n_algorithms'],
            'n_features': summary['n_features'],
            'n_train': len(train_ids),
            'n_test': len(test_ids),
            'training_times': training_times,
            'methods': {}
        }

        for method_name, method in methods.items():
            if training_times[method_name] < 0:
                continue  # Skip failed methods

            print(f"    Testing {method_name}...")

            try:
                method_results = self._evaluate_method(method, test_data)
                results['methods'][method_name] = method_results

                print(f"      Regret: {method_results['avg_regret']:.4f}, "
                      f"Top-1: {method_results['top1_accuracy']:.1%}, "
                      f"Time: {method_results['avg_selection_time']*1000:.2f}ms")
            except Exception as e:
                print(f"      FAILED: {e}")

        return results

    def _evaluate_method(
        self,
        method: Any,
        test_data: List[Dict]
    ) -> Dict[str, Any]:
        """Evaluate a single method on test data"""
        selection_times = []
        regrets = []
        top1_correct = 0
        top3_correct = 0

        for test_instance in test_data:
            features = test_instance['features']
            true_performances = test_instance['performances']

            # Create instance
            class ASLibInstance:
                def __init__(self, features):
                    self.features = features

            instance = ASLibInstance(features)

            # Select solver
            start_time = time.time()
            selected = method.select_solver(instance, features=features)
            selection_time = time.time() - start_time
            selection_times.append(selection_time)

            # Get best algorithm
            best_algo = max(true_performances, key=true_performances.get)
            best_perf = true_performances[best_algo]

            # Get selected performance
            selected_name = selected.name
            selected_perf = true_performances.get(selected_name, 0.0)

            # Compute regret
            regret = best_perf - selected_perf
            regrets.append(regret)

            # Top-k accuracy
            if selected_name == best_algo:
                top1_correct += 1
                top3_correct += 1
            else:
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
            'n_test': len(test_data)
        }

    def run_phase2(
        self,
        scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Run Phase 2 evaluation on all scenarios

        Args:
            scenarios: List of scenario names

        Returns:
            all_results: Complete results for all scenarios
        """
        all_results = {
            'scenarios': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'#'*70}")
            print(f"Scenario {i}/{len(scenarios)}: {scenario}")
            print(f"{'#'*70}")

            try:
                results = self.evaluate_scenario(scenario, test_optimal_config=True)
                all_results['scenarios'][scenario] = results
            except Exception as e:
                print(f"\nERROR evaluating {scenario}: {e}")
                import traceback
                traceback.print_exc()

        return all_results

    def generate_summary_table(
        self,
        all_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate summary table across all scenarios"""
        summary_data = []

        for scenario, results in all_results['scenarios'].items():
            for method_name, method_results in results['methods'].items():
                summary_data.append({
                    'Scenario': scenario,
                    'Method': method_name,
                    'Avg Regret': method_results['avg_regret'],
                    'Top-1 Acc': method_results['top1_accuracy'],
                    'Top-3 Acc': method_results['top3_accuracy'],
                    'Sel Time (ms)': method_results['avg_selection_time'] * 1000,
                    'Train Time (s)': results['training_times'][method_name]
                })

        return pd.DataFrame(summary_data)

    def save_results(self, all_results: Dict[str, Any], filename: str = "phase2_results"):
        """Save results to disk"""
        # Save full results as JSON
        json_path = self.results_dir / f"{filename}.json"

        # Convert to JSON-serializable format
        json_compatible = self._make_json_compatible(all_results)

        with open(json_path, 'w') as f:
            json.dump(json_compatible, f, indent=2)

        print(f"\n✓ Results saved to {json_path}")

        # Save summary table as CSV
        summary_df = self.generate_summary_table(all_results)
        csv_path = self.results_dir / f"{filename}_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        print(f"✓ Summary saved to {csv_path}")

    def _make_json_compatible(self, obj):
        """Convert to JSON-compatible format"""
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


def main():
    """Run Phase 2 evaluation"""
    print("="*70)
    print("Phase 2 Evaluation: Multi-Scenario Benchmark")
    print("="*70)

    evaluator = Phase2Evaluator(
        aslib_root="aslib_data",
        results_dir="results/phase2"
    )

    # Phase 2 scenarios
    scenarios = [
        'SAT11-HAND',     # Boolean SAT (tested in Week 5)
        'CSP-2010',       # Constraint satisfaction (tested in Week 6)
        'GRAPHS-2015',    # Graph problems (new)
        'MAXSAT12-PMS',   # MaxSAT (new)
        'ASP-POTASSCO'    # Answer set programming (new)
    ]

    # Run evaluation
    results = evaluator.run_phase2(scenarios)

    # Save results
    evaluator.save_results(results, filename="phase2_results")

    # Print summary
    print(f"\n{'='*70}")
    print("Phase 2 Summary")
    print(f"{'='*70}\n")

    summary_df = evaluator.generate_summary_table(results)
    print(summary_df.to_string(index=False))

    # Aggregate by method (average across scenarios)
    print(f"\n{'='*70}")
    print("Average Performance Across All Scenarios")
    print(f"{'='*70}\n")

    agg_df = summary_df.groupby('Method').agg({
        'Avg Regret': 'mean',
        'Top-1 Acc': 'mean',
        'Top-3 Acc': 'mean',
        'Sel Time (ms)': 'mean'
    }).round(4)

    agg_df = agg_df.sort_values('Avg Regret')
    print(agg_df.to_string())

    print(f"\n{'='*70}")
    print("✓ Phase 2 Evaluation Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
