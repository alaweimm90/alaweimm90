"""
Ablation Studies for Librex.Meta on Real ASlib Data

Re-runs Week 4 ablation studies on real ASlib scenarios to find
hyperparameters that actually generalize to real data.

Tests the impact of:
1. Number of clusters (n_clusters)
2. UCB exploration constant (ucb_c)
3. Number of tournament rounds (n_tournament_rounds)
4. Elo K-factor (elo_k)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

from libria_meta import Librex.Meta
from benchmark.aslib_parser import ASLibScenarioLoader
from benchmark.evaluate_Librex.Meta import create_mock_solver_for_aslib


class RealDataAblationStudy:
    """
    Framework for running ablation studies on real ASlib data
    """

    def __init__(
        self,
        aslib_root: str = "aslib_data",
        results_dir: str = "results/ablation_real"
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

    def _evaluate_config(
        self,
        scenario_name: str,
        n_clusters: int,
        elo_k: float,
        ucb_c: float,
        n_tournament_rounds: int
    ) -> Dict[str, Any]:
        """
        Evaluate a specific configuration on a scenario

        Args:
            scenario_name: Name of ASlib scenario
            n_clusters: Number of clusters
            elo_k: Elo K-factor
            ucb_c: UCB exploration constant
            n_tournament_rounds: Number of tournament rounds

        Returns:
            results: Dict with performance metrics
        """
        # Load scenario
        scenario_path = self.aslib_root / scenario_name
        scenario = ASLibScenarioLoader(str(scenario_path))

        # Create mock solvers for algorithms
        algorithm_names = scenario.get_algorithm_names()
        solvers = [
            create_mock_solver_for_aslib(name, base_performance=0.5)
            for name in algorithm_names
        ]

        # Create Librex.Meta with specific config
        meta = Librex.Meta(
            solvers=solvers,
            n_clusters=n_clusters,
            elo_k=elo_k,
            ucb_c=ucb_c,
            n_tournament_rounds=n_tournament_rounds
        )

        # Get train/test split (80/20)
        all_instance_ids = scenario.get_instance_ids()
        n_train = int(0.8 * len(all_instance_ids))
        train_ids = all_instance_ids[:n_train]
        test_ids = all_instance_ids[n_train:]

        # Load training data
        training_data = scenario.get_training_data(instance_ids=train_ids)

        # Train
        start_time = time.time()
        meta.fit(training_data)
        train_time = time.time() - start_time

        # Test
        test_data = scenario.get_training_data(instance_ids=test_ids)

        selection_times = []
        regrets = []
        top1_correct = 0
        top3_correct = 0

        for test_instance in test_data:
            features = test_instance['features']
            true_performances = test_instance['performances']

            # Create instance object
            class ASLibInstance:
                def __init__(self, features):
                    self.features = features

            instance = ASLibInstance(features)

            # Select solver
            start_time = time.time()
            selected = meta.select_solver(instance, features=features)
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
            'train_time': train_time,
            'avg_selection_time': np.mean(selection_times),
            'avg_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'top1_accuracy': top1_correct / len(test_data),
            'top3_accuracy': top3_correct / len(test_data),
            'n_train': len(train_ids),
            'n_test': len(test_ids)
        }

    def study_n_clusters(
        self,
        scenarios: List[str],
        cluster_values: List[int] = [1, 3, 5, 10, 20, 30]
    ) -> pd.DataFrame:
        """
        Study impact of number of clusters on real data

        Args:
            scenarios: List of ASlib scenario names
            cluster_values: Values of n_clusters to test

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Real Data Ablation Study: Number of Clusters")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\n{'#'*70}")
            print(f"Scenario: {scenario}")
            print(f"{'#'*70}\n")

            for n_clusters in cluster_values:
                print(f"  Testing n_clusters={n_clusters}...")

                try:
                    result = self._evaluate_config(
                        scenario_name=scenario,
                        n_clusters=n_clusters,
                        elo_k=32.0,
                        ucb_c=1.414,
                        n_tournament_rounds=5
                    )

                    results.append({
                        'scenario': scenario,
                        'n_clusters': n_clusters,
                        **result
                    })

                    print(f"    Regret: {result['avg_regret']:.4f}, "
                          f"Top-1: {result['top1_accuracy']:.2%}, "
                          f"Top-3: {result['top3_accuracy']:.2%}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        results_df = pd.DataFrame(results)

        # Save results
        output_file = self.results_dir / "ablation_n_clusters_real.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n✓ Results saved to {output_file}")

        return results_df

    def study_ucb_constant(
        self,
        scenarios: List[str],
        ucb_values: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0, 1.414, 2.0]
    ) -> pd.DataFrame:
        """
        Study impact of UCB exploration constant on real data

        Args:
            scenarios: List of ASlib scenario names
            ucb_values: Values of ucb_c to test

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Real Data Ablation Study: UCB Exploration Constant")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\n{'#'*70}")
            print(f"Scenario: {scenario}")
            print(f"{'#'*70}\n")

            for ucb_c in ucb_values:
                print(f"  Testing ucb_c={ucb_c:.3f}...")

                try:
                    result = self._evaluate_config(
                        scenario_name=scenario,
                        n_clusters=5,
                        elo_k=32.0,
                        ucb_c=ucb_c,
                        n_tournament_rounds=5
                    )

                    results.append({
                        'scenario': scenario,
                        'ucb_c': ucb_c,
                        **result
                    })

                    print(f"    Regret: {result['avg_regret']:.4f}, "
                          f"Top-1: {result['top1_accuracy']:.2%}, "
                          f"Top-3: {result['top3_accuracy']:.2%}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        results_df = pd.DataFrame(results)

        # Save results
        output_file = self.results_dir / "ablation_ucb_constant_real.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n✓ Results saved to {output_file}")

        return results_df

    def study_tournament_rounds(
        self,
        scenarios: List[str],
        rounds_values: List[int] = [1, 3, 5, 7, 10, 15]
    ) -> pd.DataFrame:
        """
        Study impact of tournament rounds on real data

        Args:
            scenarios: List of ASlib scenario names
            rounds_values: Values of n_tournament_rounds to test

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Real Data Ablation Study: Tournament Rounds")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\n{'#'*70}")
            print(f"Scenario: {scenario}")
            print(f"{'#'*70}\n")

            for n_rounds in rounds_values:
                print(f"  Testing n_tournament_rounds={n_rounds}...")

                try:
                    result = self._evaluate_config(
                        scenario_name=scenario,
                        n_clusters=5,
                        elo_k=32.0,
                        ucb_c=1.414,
                        n_tournament_rounds=n_rounds
                    )

                    results.append({
                        'scenario': scenario,
                        'n_tournament_rounds': n_rounds,
                        **result
                    })

                    print(f"    Regret: {result['avg_regret']:.4f}, "
                          f"Top-1: {result['top1_accuracy']:.2%}, "
                          f"Top-3: {result['top3_accuracy']:.2%}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        results_df = pd.DataFrame(results)

        # Save results
        output_file = self.results_dir / "ablation_tournament_rounds_real.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n✓ Results saved to {output_file}")

        return results_df

    def study_elo_k(
        self,
        scenarios: List[str],
        k_values: List[float] = [8.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    ) -> pd.DataFrame:
        """
        Study impact of Elo K-factor on real data

        Args:
            scenarios: List of ASlib scenario names
            k_values: Values of elo_k to test

        Returns:
            results_df: DataFrame with results
        """
        print(f"\n{'='*70}")
        print("Real Data Ablation Study: Elo K-Factor")
        print(f"{'='*70}\n")

        results = []

        for scenario in scenarios:
            print(f"\n{'#'*70}")
            print(f"Scenario: {scenario}")
            print(f"{'#'*70}\n")

            for elo_k in k_values:
                print(f"  Testing elo_k={elo_k:.1f}...")

                try:
                    result = self._evaluate_config(
                        scenario_name=scenario,
                        n_clusters=5,
                        elo_k=elo_k,
                        ucb_c=1.414,
                        n_tournament_rounds=5
                    )

                    results.append({
                        'scenario': scenario,
                        'elo_k': elo_k,
                        **result
                    })

                    print(f"    Regret: {result['avg_regret']:.4f}, "
                          f"Top-1: {result['top1_accuracy']:.2%}, "
                          f"Top-3: {result['top3_accuracy']:.2%}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        results_df = pd.DataFrame(results)

        # Save results
        output_file = self.results_dir / "ablation_elo_k_real.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n✓ Results saved to {output_file}")

        return results_df

    def generate_summary(self, results_dfs: Dict[str, pd.DataFrame]) -> str:
        """
        Generate summary report across all ablation studies

        Args:
            results_dfs: Dict of {study_name: results_df}

        Returns:
            summary: Markdown formatted summary
        """
        lines = []
        lines.append("# Real Data Ablation Studies Summary\n")
        lines.append("## Overview\n")

        for study_name, df in results_dfs.items():
            lines.append(f"### {study_name}\n")

            # Group by parameter and compute average across scenarios
            param_col = [c for c in df.columns if c not in [
                'scenario', 'train_time', 'avg_selection_time',
                'avg_regret', 'std_regret', 'top1_accuracy', 'top3_accuracy',
                'n_train', 'n_test'
            ]][0]

            summary = df.groupby(param_col).agg({
                'avg_regret': 'mean',
                'top1_accuracy': 'mean',
                'top3_accuracy': 'mean'
            }).reset_index()

            lines.append(summary.to_markdown(index=False))
            lines.append("")

            # Find best parameter value
            best_idx = summary['avg_regret'].idxmin()
            best_value = summary.loc[best_idx, param_col]
            best_regret = summary.loc[best_idx, 'avg_regret']

            lines.append(f"**Best {param_col}**: {best_value} (avg regret: {best_regret:.4f})\n")

        summary_text = "\n".join(lines)

        # Save summary
        output_file = self.results_dir / "ablation_summary_real.md"
        with open(output_file, 'w') as f:
            f.write(summary_text)

        print(f"\n✓ Summary saved to {output_file}")

        return summary_text


def main():
    """Run all ablation studies on real data"""
    print("="*70)
    print("Real Data Ablation Studies - Week 6")
    print("="*70)

    ablation = RealDataAblationStudy(
        aslib_root="aslib_data",
        results_dir="results/ablation_real"
    )

    # Test scenarios (start with 2, expand if time permits)
    scenarios = [
        'SAT11-HAND',
        'CSP-2010'
    ]

    # Run all 4 studies
    results_dfs = {}

    # Study 1: n_clusters
    print("\n" + "="*70)
    print("STUDY 1/4: Number of Clusters")
    print("="*70)
    results_dfs['N Clusters'] = ablation.study_n_clusters(scenarios)

    # Study 2: ucb_c
    print("\n" + "="*70)
    print("STUDY 2/4: UCB Constant")
    print("="*70)
    results_dfs['UCB Constant'] = ablation.study_ucb_constant(scenarios)

    # Study 3: n_tournament_rounds
    print("\n" + "="*70)
    print("STUDY 3/4: Tournament Rounds")
    print("="*70)
    results_dfs['Tournament Rounds'] = ablation.study_tournament_rounds(scenarios)

    # Study 4: elo_k
    print("\n" + "="*70)
    print("STUDY 4/4: Elo K-Factor")
    print("="*70)
    results_dfs['Elo K'] = ablation.study_elo_k(scenarios)

    # Generate summary
    print("\n" + "="*70)
    print("Generating Summary...")
    print("="*70)
    summary = ablation.generate_summary(results_dfs)
    print("\n" + summary)

    print("\n" + "="*70)
    print("✓ All Real Data Ablation Studies Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
