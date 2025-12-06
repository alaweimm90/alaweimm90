"""
Ablation Studies for Librex.Meta

Tests the impact of key hyperparameters:
1. Number of clusters (n_clusters)
2. UCB exploration constant (ucb_c)
3. Number of tournament rounds (n_tournament_rounds)
4. Elo K-factor (elo_k)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time

from libria_meta import Librex.Meta
from benchmark.run_evaluation import ComprehensiveEvaluator


class AblationStudy:
    """
    Framework for running ablation studies on Librex.Meta
    """

    def __init__(
        self,
        aslib_root: str = "aslib_data",
        results_dir: str = "results/ablation"
    ):
        """
        Initialize ablation study framework

        Args:
            aslib_root: Root directory for ASlib data
            results_dir: Directory to save ablation results
        """
        self.aslib_root = Path(aslib_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = ComprehensiveEvaluator(
            aslib_root=str(self.aslib_root),
            results_dir=str(self.results_dir)
        )

    def study_n_clusters(
        self,
        scenarios: List[str],
        cluster_values: List[int] = [1, 3, 5, 10, 20],
        n_train: int = 100,
        n_test: int = 50
    ) -> pd.DataFrame:
        """
        Study impact of number of clusters

        Args:
            scenarios: List of scenario names
            cluster_values: Values of n_clusters to test
            n_train: Number of training instances
            n_test: Number of test instances

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Ablation Study: Number of Clusters")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            for n_clusters in cluster_values:
                print(f"  Testing n_clusters={n_clusters}...")

                # Create solvers (mock)
                n_solvers = np.random.randint(5, 12)
                solvers = [
                    self.evaluator._create_mock_instance.__self__.__class__()  # Mock solver
                    for _ in range(n_solvers)
                ]

                # Actually, let me use the proper mock solver creation
                from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib
                solvers = [
                    create_mock_solver_for_aslib(
                        f'Solver{i}',
                        base_performance=np.random.uniform(0.4, 0.8)
                    )
                    for i in range(n_solvers)
                ]

                # Create Librex.Meta with specific n_clusters
                meta = Librex.Meta(
                    solvers=solvers,
                    n_clusters=n_clusters,
                    elo_k=32.0,
                    ucb_c=1.414,
                    n_tournament_rounds=5
                )

                # Generate training data
                training_data = self.evaluator._generate_mock_training_data(
                    solvers, n_train
                )

                # Train
                start_time = time.time()
                meta.fit(training_data)
                train_time = time.time() - start_time

                # Test
                test_instances = [
                    self.evaluator._create_mock_instance()
                    for _ in range(n_test)
                ]

                test_results = self.evaluator._evaluate_method_on_instances(
                    meta, test_instances, solvers
                )

                # Store results
                results.append({
                    'scenario': scenario,
                    'n_clusters': n_clusters,
                    'train_time': train_time,
                    'avg_regret': test_results['avg_regret'],
                    'top1_accuracy': test_results['top1_accuracy'],
                    'top3_accuracy': test_results['top3_accuracy'],
                    'avg_selection_time': test_results['avg_selection_time']
                })

                print(f"    Regret: {test_results['avg_regret']:.4f}, "
                      f"Top-1: {test_results['top1_accuracy']:.2%}")

        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(
            self.results_dir / "ablation_n_clusters.csv",
            index=False
        )

        print(f"\n✓ Results saved to {self.results_dir / 'ablation_n_clusters.csv'}")

        return results_df

    def study_ucb_constant(
        self,
        scenarios: List[str],
        ucb_values: List[float] = [0.5, 1.0, 1.414, 2.0],
        n_train: int = 100,
        n_test: int = 50
    ) -> pd.DataFrame:
        """
        Study impact of UCB exploration constant

        Args:
            scenarios: List of scenario names
            ucb_values: Values of ucb_c to test
            n_train: Number of training instances
            n_test: Number of test instances

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Ablation Study: UCB Exploration Constant")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            for ucb_c in ucb_values:
                print(f"  Testing ucb_c={ucb_c:.3f}...")

                # Create solvers
                from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib
                n_solvers = np.random.randint(5, 12)
                solvers = [
                    create_mock_solver_for_aslib(
                        f'Solver{i}',
                        base_performance=np.random.uniform(0.4, 0.8)
                    )
                    for i in range(n_solvers)
                ]

                # Create Librex.Meta with specific ucb_c
                meta = Librex.Meta(
                    solvers=solvers,
                    n_clusters=5,
                    elo_k=32.0,
                    ucb_c=ucb_c,
                    n_tournament_rounds=5
                )

                # Train
                training_data = self.evaluator._generate_mock_training_data(
                    solvers, n_train
                )
                meta.fit(training_data)

                # Test
                test_instances = [
                    self.evaluator._create_mock_instance()
                    for _ in range(n_test)
                ]
                test_results = self.evaluator._evaluate_method_on_instances(
                    meta, test_instances, solvers
                )

                results.append({
                    'scenario': scenario,
                    'ucb_c': ucb_c,
                    'avg_regret': test_results['avg_regret'],
                    'top1_accuracy': test_results['top1_accuracy'],
                    'top3_accuracy': test_results['top3_accuracy']
                })

                print(f"    Regret: {test_results['avg_regret']:.4f}, "
                      f"Top-1: {test_results['top1_accuracy']:.2%}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            self.results_dir / "ablation_ucb_constant.csv",
            index=False
        )

        print(f"\n✓ Results saved to {self.results_dir / 'ablation_ucb_constant.csv'}")

        return results_df

    def study_tournament_rounds(
        self,
        scenarios: List[str],
        rounds_values: List[int] = [1, 3, 5, 10],
        n_train: int = 100,
        n_test: int = 50
    ) -> pd.DataFrame:
        """
        Study impact of number of tournament rounds

        Args:
            scenarios: List of scenario names
            rounds_values: Values of n_tournament_rounds to test
            n_train: Number of training instances
            n_test: Number of test instances

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Ablation Study: Number of Tournament Rounds")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            for n_rounds in rounds_values:
                print(f"  Testing n_tournament_rounds={n_rounds}...")

                # Create solvers
                from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib
                n_solvers = np.random.randint(5, 12)
                solvers = [
                    create_mock_solver_for_aslib(
                        f'Solver{i}',
                        base_performance=np.random.uniform(0.4, 0.8)
                    )
                    for i in range(n_solvers)
                ]

                # Create Librex.Meta with specific n_rounds
                meta = Librex.Meta(
                    solvers=solvers,
                    n_clusters=5,
                    elo_k=32.0,
                    ucb_c=1.414,
                    n_tournament_rounds=n_rounds
                )

                # Train
                training_data = self.evaluator._generate_mock_training_data(
                    solvers, n_train
                )
                meta.fit(training_data)

                # Test
                test_instances = [
                    self.evaluator._create_mock_instance()
                    for _ in range(n_test)
                ]
                test_results = self.evaluator._evaluate_method_on_instances(
                    meta, test_instances, solvers
                )

                results.append({
                    'scenario': scenario,
                    'n_tournament_rounds': n_rounds,
                    'avg_regret': test_results['avg_regret'],
                    'top1_accuracy': test_results['top1_accuracy'],
                    'top3_accuracy': test_results['top3_accuracy']
                })

                print(f"    Regret: {test_results['avg_regret']:.4f}, "
                      f"Top-1: {test_results['top1_accuracy']:.2%}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            self.results_dir / "ablation_tournament_rounds.csv",
            index=False
        )

        print(f"\n✓ Results saved to {self.results_dir / 'ablation_tournament_rounds.csv'}")

        return results_df

    def study_elo_k_factor(
        self,
        scenarios: List[str],
        k_values: List[float] = [16, 32, 64, 128],
        n_train: int = 100,
        n_test: int = 50
    ) -> pd.DataFrame:
        """
        Study impact of Elo K-factor

        Args:
            scenarios: List of scenario names
            k_values: Values of elo_k to test
            n_train: Number of training instances
            n_test: Number of test instances

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Ablation Study: Elo K-Factor")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            for elo_k in k_values:
                print(f"  Testing elo_k={elo_k}...")

                # Create solvers
                from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib
                n_solvers = np.random.randint(5, 12)
                solvers = [
                    create_mock_solver_for_aslib(
                        f'Solver{i}',
                        base_performance=np.random.uniform(0.4, 0.8)
                    )
                    for i in range(n_solvers)
                ]

                # Create Librex.Meta with specific elo_k
                meta = Librex.Meta(
                    solvers=solvers,
                    n_clusters=5,
                    elo_k=elo_k,
                    ucb_c=1.414,
                    n_tournament_rounds=5
                )

                # Train
                training_data = self.evaluator._generate_mock_training_data(
                    solvers, n_train
                )
                meta.fit(training_data)

                # Test
                test_instances = [
                    self.evaluator._create_mock_instance()
                    for _ in range(n_test)
                ]
                test_results = self.evaluator._evaluate_method_on_instances(
                    meta, test_instances, solvers
                )

                results.append({
                    'scenario': scenario,
                    'elo_k': elo_k,
                    'avg_regret': test_results['avg_regret'],
                    'top1_accuracy': test_results['top1_accuracy'],
                    'top3_accuracy': test_results['top3_accuracy']
                })

                print(f"    Regret: {test_results['avg_regret']:.4f}, "
                      f"Top-1: {test_results['top1_accuracy']:.2%}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            self.results_dir / "ablation_elo_k.csv",
            index=False
        )

        print(f"\n✓ Results saved to {self.results_dir / 'ablation_elo_k.csv'}")

        return results_df

    def run_all_studies(
        self,
        scenarios: List[str] = ['SAT11-HAND', 'CSP-2010', 'GRAPHS-2015']
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all ablation studies

        Args:
            scenarios: List of scenarios to test on

        Returns:
            all_results: Dict of {study_name: results_df}
        """
        print(f"\n{'#'*70}")
        print("Running All Ablation Studies")
        print(f"Scenarios: {', '.join(scenarios)}")
        print(f"{'#'*70}")

        all_results = {}

        # Study 1: Number of clusters
        all_results['n_clusters'] = self.study_n_clusters(scenarios)

        # Study 2: UCB constant
        all_results['ucb_constant'] = self.study_ucb_constant(scenarios)

        # Study 3: Tournament rounds
        all_results['tournament_rounds'] = self.study_tournament_rounds(scenarios)

        # Study 4: Elo K-factor
        all_results['elo_k'] = self.study_elo_k_factor(scenarios)

        print(f"\n{'#'*70}")
        print("All Ablation Studies Complete!")
        print(f"{'#'*70}")

        # Generate summary report
        self._generate_summary_report(all_results)

        return all_results

    def _generate_summary_report(self, all_results: Dict[str, pd.DataFrame]):
        """Generate summary report for all ablation studies"""
        report_lines = []
        report_lines.append("# Ablation Studies Summary Report")
        report_lines.append("")
        report_lines.append("## Overview")
        report_lines.append("")

        for study_name, results_df in all_results.items():
            report_lines.append(f"### {study_name.replace('_', ' ').title()}")
            report_lines.append("")

            # Group by parameter and compute average
            param_col = [col for col in results_df.columns if col not in ['scenario', 'avg_regret', 'top1_accuracy', 'top3_accuracy', 'train_time', 'avg_selection_time']][0]

            summary = results_df.groupby(param_col).agg({
                'avg_regret': 'mean',
                'top1_accuracy': 'mean',
                'top3_accuracy': 'mean'
            }).round(4)

            report_lines.append(summary.to_markdown())
            report_lines.append("")

            # Find best parameter value
            best_idx = summary['avg_regret'].idxmin()
            report_lines.append(f"**Best {param_col}**: {best_idx} (avg regret: {summary.loc[best_idx, 'avg_regret']:.4f})")
            report_lines.append("")

        report = "\n".join(report_lines)

        # Save report
        report_path = self.results_dir / "ablation_summary.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n✓ Summary report saved to {report_path}")


def main():
    """Run ablation studies"""
    ablation = AblationStudy(
        aslib_root="aslib_data",
        results_dir="results/ablation"
    )

    # Run all studies on 3 scenarios
    scenarios = ['SAT11-HAND', 'CSP-2010', 'GRAPHS-2015']

    results = ablation.run_all_studies(scenarios)

    print("\n" + "="*70)
    print("Ablation Studies Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
