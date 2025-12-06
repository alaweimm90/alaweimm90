#!/usr/bin/env python3
"""
Nightly Benchmark Runner for Librex

This is the main entry point for automated performance benchmarking.
It runs configurable benchmark suites and tracks performance over time.

Usage:
    python nightly_benchmark.py --suite smoke
    python nightly_benchmark.py --suite standard
    python nightly_benchmark.py --suite comprehensive
    python nightly_benchmark.py --suite method_comparison
    python nightly_benchmark.py --all
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import yaml

# Optional import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.adapters.tsp import TSPAdapter
from Librex.benchmarks.qaplib import load_qaplib_instance, list_qaplib_instances
from Librex.core.interfaces import StandardizedProblem


class BenchmarkRunner:
    """Main benchmark orchestrator"""

    def __init__(self, config_file: str = "benchmark_config.yaml"):
        """Initialize the benchmark runner

        Args:
            config_file: Path to the configuration file
        """
        self.config_file = Path(config_file)
        if not self.config_file.is_absolute():
            self.config_file = Path(__file__).parent / config_file

        self.config = self._load_config()
        self.results_dir = Path(__file__).parent / self.config['output']['results_dir']
        self.results_dir.mkdir(exist_ok=True)

        # Track current run metadata
        self.run_metadata = {
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'python_version': sys.version,
            'platform': sys.platform,
        }

        if PSUTIL_AVAILABLE:
            self.run_metadata['cpu_count'] = psutil.cpu_count()
            self.run_metadata['memory_total'] = psutil.virtual_memory().total
        else:
            import multiprocessing
            self.run_metadata['cpu_count'] = multiprocessing.cpu_count()
            self.run_metadata['memory_total'] = None

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except:
            pass
        return None

    def run_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific benchmark suite

        Args:
            suite_name: Name of the suite to run

        Returns:
            Dictionary containing all results
        """
        if suite_name not in self.config['benchmark_suites']:
            raise ValueError(f"Unknown suite: {suite_name}")

        suite_config = self.config['benchmark_suites'][suite_name]

        if not suite_config.get('enabled', True):
            print(f"Suite '{suite_name}' is disabled in configuration")
            return {}

        print("=" * 80)
        print(f"Running {suite_config['name']}")
        print(f"Description: {suite_config['description']}")
        print("=" * 80)
        print()

        suite_start_time = time.time()
        suite_results = {
            'suite_name': suite_name,
            'suite_config': suite_config,
            'start_time': datetime.now().isoformat(),
            'results': []
        }

        # Run benchmarks for each problem type and size
        for problem_config in suite_config['problems']:
            problem_type = problem_config['type']

            if problem_type == 'qap' and problem_config.get('use_qaplib', False):
                # Use QAPLIB instances
                for instance_name in problem_config.get('qaplib_instances', []):
                    results = self._run_qaplib_benchmark(
                        instance_name,
                        suite_config['methods'],
                        suite_config['runs_per_method']
                    )
                    suite_results['results'].extend(results)
            else:
                # Use generated problems
                for size in problem_config.get('sizes', []):
                    for instance_idx in range(problem_config.get('instances_per_size', 1)):
                        results = self._run_generated_benchmark(
                            problem_type,
                            size,
                            instance_idx,
                            suite_config['methods'],
                            suite_config['runs_per_method']
                        )
                        suite_results['results'].extend(results)

            # Check time limit
            elapsed = time.time() - suite_start_time
            if elapsed > suite_config.get('max_duration', float('inf')):
                print(f"Time limit reached ({elapsed:.1f}s), stopping suite")
                break

        suite_results['end_time'] = datetime.now().isoformat()
        suite_results['total_duration'] = time.time() - suite_start_time

        return suite_results

    def _run_qaplib_benchmark(
        self,
        instance_name: str,
        methods: List[Dict],
        runs_per_method: int
    ) -> List[Dict]:
        """Run benchmark on a QAPLIB instance"""
        print(f"\nBenchmarking QAPLIB instance: {instance_name}")
        print("-" * 50)

        # Load the instance
        try:
            instance_data = load_qaplib_instance(instance_name)
            problem = {
                'flow_matrix': instance_data['flow_matrix'],
                'distance_matrix': instance_data['distance_matrix']
            }
            optimal = instance_data.get('optimal_value')
            adapter = QAPAdapter()
        except Exception as e:
            print(f"  Error loading instance: {e}")
            return []

        results = []
        for method_config in methods:
            method_name = method_config['name']

            if 'parameter_sweep' in method_config:
                # Run parameter sweep
                for params in self._generate_parameter_combinations(method_config['parameter_sweep']):
                    result = self._benchmark_single_method(
                        problem,
                        adapter,
                        method_name,
                        params,
                        runs_per_method,
                        problem_info={
                            'type': 'qap',
                            'instance': instance_name,
                            'size': instance_data['flow_matrix'].shape[0],
                            'optimal': optimal
                        }
                    )
                    results.append(result)
            else:
                # Run with fixed config
                result = self._benchmark_single_method(
                    problem,
                    adapter,
                    method_name,
                    method_config.get('config', {}),
                    runs_per_method,
                    problem_info={
                        'type': 'qap',
                        'instance': instance_name,
                        'size': instance_data['flow_matrix'].shape[0],
                        'optimal': optimal
                    }
                )
                results.append(result)

        return results

    def _run_generated_benchmark(
        self,
        problem_type: str,
        size: int,
        instance_idx: int,
        methods: List[Dict],
        runs_per_method: int
    ) -> List[Dict]:
        """Run benchmark on a generated problem instance"""
        print(f"\nBenchmarking {problem_type.upper()} - Size: {size}, Instance: {instance_idx + 1}")
        print("-" * 50)

        # Generate problem
        problem, adapter = self._generate_problem(problem_type, size, seed=42 + instance_idx)

        results = []
        for method_config in methods:
            method_name = method_config['name']

            if 'parameter_sweep' in method_config:
                # Run parameter sweep
                for params in self._generate_parameter_combinations(method_config['parameter_sweep']):
                    result = self._benchmark_single_method(
                        problem,
                        adapter,
                        method_name,
                        params,
                        runs_per_method,
                        problem_info={
                            'type': problem_type,
                            'size': size,
                            'instance_idx': instance_idx
                        }
                    )
                    results.append(result)
            else:
                # Run with fixed config
                result = self._benchmark_single_method(
                    problem,
                    adapter,
                    method_name,
                    method_config.get('config', {}),
                    runs_per_method,
                    problem_info={
                        'type': problem_type,
                        'size': size,
                        'instance_idx': instance_idx
                    }
                )
                results.append(result)

        return results

    def _generate_problem(self, problem_type: str, size: int, seed: int) -> Tuple[Dict, Any]:
        """Generate a problem instance"""
        np.random.seed(seed)

        if problem_type == 'qap':
            flow = np.random.randint(1, 100, (size, size))
            distance = np.random.randint(1, 100, (size, size))

            # Make symmetric
            flow = (flow + flow.T) // 2
            distance = (distance + distance.T) // 2

            # Zero diagonal
            np.fill_diagonal(flow, 0)
            np.fill_diagonal(distance, 0)

            return {'flow_matrix': flow, 'distance_matrix': distance}, QAPAdapter()

        elif problem_type == 'tsp':
            # Generate random points
            points = np.random.rand(size, 2) * 100
            distances = np.zeros((size, size))

            for i in range(size):
                for j in range(size):
                    if i != j:
                        distances[i, j] = np.linalg.norm(points[i] - points[j])

            return {'distance_matrix': distances}, TSPAdapter()

        elif problem_type == 'knapsack':
            # Generate knapsack problem
            weights = np.random.randint(1, 50, size)
            values = np.random.randint(1, 100, size)
            capacity = int(sum(weights) * 0.5)  # 50% of total weight

            # For now, we'll use a simple adapter (would need KnapsackAdapter)
            # This is a placeholder
            return {
                'weights': weights,
                'values': values,
                'capacity': capacity
            }, None

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def _generate_parameter_combinations(self, param_sweep: Dict) -> List[Dict]:
        """Generate all combinations of parameters for parameter sweep"""
        import itertools

        keys = list(param_sweep.keys())
        values = [param_sweep[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _benchmark_single_method(
        self,
        problem: Dict,
        adapter: Any,
        method: str,
        config: Dict,
        runs: int,
        problem_info: Dict
    ) -> Dict:
        """Benchmark a single method on a problem"""
        print(f"  {method} with config {config}...", end=" ")

        objectives = []
        runtimes = []
        memory_usage = []
        convergence_data = []

        for run_idx in range(runs):
            # Set seed for reproducibility
            run_config = config.copy()
            run_config['seed'] = 42 + run_idx

            # Track memory before
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
            else:
                mem_before = 0

            # Run optimization
            start_time = time.time()
            try:
                result = optimize(problem, adapter, method=method, config=run_config)
                runtime = time.time() - start_time

                # Track memory after
                if PSUTIL_AVAILABLE:
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    mem_used = max(0, mem_after - mem_before)
                else:
                    mem_used = 0

                objectives.append(result['objective'])
                runtimes.append(runtime)
                memory_usage.append(mem_used)

                # Extract convergence data if available
                if 'history' in result and result['history']:
                    convergence_data.append(result['history'])

            except Exception as e:
                print(f"\n    Error in run {run_idx + 1}: {e}")
                continue

        if not objectives:
            print("FAILED")
            return {
                'problem': problem_info,
                'method': method,
                'config': config,
                'status': 'failed'
            }

        # Calculate statistics
        result = {
            'problem': problem_info,
            'method': method,
            'config': config,
            'runs': runs,
            'objectives': objectives,
            'runtimes': runtimes,
            'memory_usage': memory_usage,
            'statistics': {
                'mean_objective': np.mean(objectives),
                'std_objective': np.std(objectives),
                'best_objective': np.min(objectives),
                'worst_objective': np.max(objectives),
                'mean_runtime': np.mean(runtimes),
                'std_runtime': np.std(runtimes),
                'mean_memory': np.mean(memory_usage) if memory_usage else 0,
                'success_rate': len(objectives) / runs * 100
            }
        }

        # Calculate optimality gap if known
        if 'optimal' in problem_info and problem_info['optimal'] is not None:
            optimal = problem_info['optimal']
            result['statistics']['optimality_gap'] = (
                (result['statistics']['best_objective'] - optimal) / optimal * 100
            )

        # Analyze convergence if data available
        if convergence_data:
            result['convergence'] = self._analyze_convergence(convergence_data)

        print(f"Best: {result['statistics']['best_objective']:.2f}, "
              f"Mean: {result['statistics']['mean_objective']:.2f}±"
              f"{result['statistics']['std_objective']:.2f}, "
              f"Time: {result['statistics']['mean_runtime']:.2f}s")

        return result

    def _analyze_convergence(self, convergence_data: List) -> Dict:
        """Analyze convergence characteristics"""
        # This is a simplified analysis
        # In practice, you'd want more sophisticated metrics

        convergence_stats = {}

        for history in convergence_data:
            if not history:
                continue

            # Find iterations to reach 90% of final quality
            final_quality = history[-1] if isinstance(history[-1], (int, float)) else history[-1].get('objective', 0)
            target_quality = final_quality * 0.9

            iterations_to_90 = len(history)
            for i, point in enumerate(history):
                value = point if isinstance(point, (int, float)) else point.get('objective', 0)
                if value <= target_quality:
                    iterations_to_90 = i + 1
                    break

            if 'iterations_to_90' not in convergence_stats:
                convergence_stats['iterations_to_90'] = []
            convergence_stats['iterations_to_90'].append(iterations_to_90)

        # Calculate average convergence speed
        if 'iterations_to_90' in convergence_stats:
            convergence_stats['mean_iterations_to_90'] = np.mean(convergence_stats['iterations_to_90'])

        return convergence_stats

    def save_results(self, results: Dict, suite_name: str):
        """Save benchmark results to various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{suite_name}_{timestamp}"

        # Add metadata
        results['metadata'] = self.run_metadata

        # Save JSON (primary format)
        json_path = self.results_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")

        # Generate other formats based on config
        formats = self.config['output']['formats']

        if 'html' in formats:
            from report_generator import HTMLReportGenerator
            html_gen = HTMLReportGenerator()
            html_path = self.results_dir / f"{base_filename}.html"
            html_gen.generate(results, html_path)
            print(f"HTML report saved to: {html_path}")

        if 'markdown' in formats:
            from report_generator import MarkdownReportGenerator
            md_gen = MarkdownReportGenerator()
            md_path = self.results_dir / f"{base_filename}.md"
            md_gen.generate(results, md_path)
            print(f"Markdown report saved to: {md_path}")

        if 'csv' in formats:
            from report_generator import CSVReportGenerator
            csv_gen = CSVReportGenerator()
            csv_path = self.results_dir / f"{base_filename}.csv"
            csv_gen.generate(results, csv_path)
            print(f"CSV export saved to: {csv_path}")

        # Update latest symlink
        latest_link = self.results_dir / f"{suite_name}_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(json_path.name)

        return json_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Librex Nightly Benchmark Runner"
    )
    parser.add_argument(
        '--suite',
        choices=['smoke', 'standard', 'comprehensive', 'method_comparison'],
        help='Benchmark suite to run'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all enabled benchmark suites'
    )
    parser.add_argument(
        '--config',
        default='benchmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save results to file (default: True)'
    )

    args = parser.parse_args()

    if not args.suite and not args.all:
        parser.error("Either --suite or --all must be specified")

    # Initialize runner
    runner = BenchmarkRunner(args.config)

    # Determine which suites to run
    if args.all:
        suites_to_run = [
            name for name, config in runner.config['benchmark_suites'].items()
            if config.get('enabled', True)
        ]
    else:
        suites_to_run = [args.suite]

    # Run benchmarks
    all_results = {}
    for suite_name in suites_to_run:
        try:
            results = runner.run_suite(suite_name)
            all_results[suite_name] = results

            # Save results
            if args.save and results:
                runner.save_results(results, suite_name)

        except Exception as e:
            print(f"\nError running suite '{suite_name}': {e}")
            traceback.print_exc()
            continue

    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK RUN COMPLETE")
    print("=" * 80)

    for suite_name, results in all_results.items():
        if results:
            print(f"\n{suite_name}:")
            print(f"  Total duration: {results.get('total_duration', 0):.1f}s")
            print(f"  Problems tested: {len(set(r['problem'].get('type', '') for r in results.get('results', [])))}")
            print(f"  Methods tested: {len(set(r['method'] for r in results.get('results', [])))}")
            print(f"  Total runs: {sum(r.get('runs', 0) for r in results.get('results', []))}")

    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())