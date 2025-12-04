"""
MetaHub Benchmarking Core

Universal benchmark runner and result management.

Originally from: organizations/AlaweinOS/Benchmarks/run_benchmarks.py
Refactored: 2025-01-29
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import statistics


def run_benchmark(
    target: Callable,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Universal benchmark runner for any callable.

    Args:
        target: Function or callable to benchmark
        iterations: Number of iterations to run
        warmup: Warmup iterations (not counted)
        **kwargs: Arguments to pass to target

    Returns:
        Dict with timing statistics:
            - iterations: Number of runs
            - mean: Average time
            - median: Median time
            - stdev: Standard deviation
            - min: Minimum time
            - max: Maximum time
            - total: Total time

    Example:
        >>> results = run_benchmark(my_function, iterations=1000)
        >>> print(f"Mean: {results['mean']:.4f}s")
    """
    # Warmup phase
    for _ in range(warmup):
        target(**kwargs)

    # Benchmark phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = target(**kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'iterations': iterations,
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'total': sum(times),
        'result_sample': result  # Last result (for validation)
    }


def format_results(results: Dict[str, Any]) -> str:
    """
    Format benchmark results for display.

    Args:
        results: Results dict from run_benchmark()

    Returns:
        Formatted string for console output
    """
    return f"""
Benchmark Results ({results['iterations']} iterations):
  Mean:   {results['mean']*1000:.4f} ms
  Median: {results['median']*1000:.4f} ms
  StdDev: {results['stdev']*1000:.4f} ms
  Min:    {results['min']*1000:.4f} ms
  Max:    {results['max']*1000:.4f} ms
  Total:  {results['total']:.4f} s
"""


class BenchmarkRunner:
    """
    Universal benchmark suite runner.

    Manages running multiple benchmarks, collecting results,
    and generating reports.

    Originally from: AlaweinOS/Benchmarks/run_benchmarks.py
    Generalized for use across all projects.

    Usage:
        runner = BenchmarkRunner(output_dir="./results")
        runner.add_benchmark("test1", my_function, iterations=100)
        runner.run_all()
        runner.save_results()
    """

    def __init__(self, output_dir: Path):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def add_benchmark(
        self,
        name: str,
        target: Callable,
        iterations: int = 100,
        warmup: int = 10,
        metadata: Optional[Dict] = None,
        **kwargs
    ):
        """
        Add a benchmark to the suite.

        Args:
            name: Benchmark name
            target: Function to benchmark
            iterations: Number of iterations
            warmup: Warmup iterations
            metadata: Additional metadata (problem_type, size, etc.)
            **kwargs: Arguments for target function
        """
        self.benchmarks.append({
            'name': name,
            'target': target,
            'iterations': iterations,
            'warmup': warmup,
            'metadata': metadata or {},
            'kwargs': kwargs
        })

    def run_all(self) -> List[Dict[str, Any]]:
        """
        Run all registered benchmarks.

        Returns:
            List of result dictionaries
        """
        print("=" * 70)
        print("🚀 Running Benchmark Suite")
        print("=" * 70)

        all_results = []

        for bench in self.benchmarks:
            print(f"\n🔬 Running: {bench['name']}")

            try:
                result = run_benchmark(
                    bench['target'],
                    iterations=bench['iterations'],
                    warmup=bench['warmup'],
                    **bench['kwargs']
                )

                # Add metadata
                result['name'] = bench['name']
                result['status'] = 'success'
                result.update(bench['metadata'])

                all_results.append(result)

                print(f"  ✅ Completed in {result['mean']:.4f}s avg")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                all_results.append({
                    'name': bench['name'],
                    'status': 'failed',
                    'error': str(e),
                    **bench['metadata']
                })

        self.results = all_results
        return all_results

    def generate_summary(self, results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics from results.

        Args:
            results: Results list (uses self.results if None)

        Returns:
            Summary dictionary
        """
        results = results or self.results

        if not results:
            return {}

        # Filter successful results
        successful = [r for r in results if r.get('status') == 'success']

        summary = {
            'total_benchmarks': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'by_type': {},
            'overall_stats': {}
        }

        if successful:
            all_times = [r['mean'] for r in successful]
            summary['overall_stats'] = {
                'mean': statistics.mean(all_times),
                'median': statistics.median(all_times),
                'stdev': statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
                'min': min(all_times),
                'max': max(all_times)
            }

            # Group by problem_type if available
            by_type = {}
            for r in successful:
                ptype = r.get('problem_type', r.get('type', 'default'))
                if ptype not in by_type:
                    by_type[ptype] = []
                by_type[ptype].append(r['mean'])

            for ptype, times in by_type.items():
                summary['by_type'][ptype] = {
                    'count': len(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
                    'min': min(times),
                    'max': max(times)
                }

        return summary

    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save benchmark results to JSON file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"benchmark_results_{int(time.time())}.json"

        output_file = self.output_dir / filename

        output_data = {
            'timestamp': time.time(),
            'results': self.results,
            'summary': self.generate_summary(self.results),
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        return output_file

    def print_summary(self):
        """Print benchmark summary to console."""
        if not self.results:
            print("No results to summarize")
            return

        summary = self.generate_summary(self.results)

        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nTotal Benchmarks: {summary['total_benchmarks']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")

        if summary.get('overall_stats'):
            stats = summary['overall_stats']
            print(f"\nOverall Performance:")
            print(f"  Mean:   {stats['mean']:.6f}s")
            print(f"  Median: {stats['median']:.6f}s")
            print(f"  StdDev: {stats['stdev']:.6f}s")
            print(f"  Range:  [{stats['min']:.6f}s, {stats['max']:.6f}s]")

        if summary.get('by_type'):
            print(f"\nBy Type:")
            for ptype, stats in summary['by_type'].items():
                print(f"  {ptype}: {stats['count']} runs, avg {stats['mean']:.6f}s")

        print("\n" + "=" * 70)


def load_results(results_file: Path) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file.

    Args:
        results_file: Path to results JSON

    Returns:
        Results dictionary
    """
    with open(results_file) as f:
        return json.load(f)


def load_latest_results(results_dir: Path) -> Dict[str, Any]:
    """
    Load the most recent benchmark results from a directory.

    Args:
        results_dir: Directory containing result files

    Returns:
        Most recent results dictionary

    Raises:
        FileNotFoundError: If no results found
    """
    result_files = list(Path(results_dir).glob("benchmark_results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No benchmark results found in {results_dir}")

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    return load_results(latest)
