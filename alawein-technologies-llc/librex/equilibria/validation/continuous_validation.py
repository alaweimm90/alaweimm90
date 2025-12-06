#!/usr/bin/env python3
"""
Continuous validation suite for CI/CD integration.

This script runs comprehensive validation tests and generates reports
for continuous integration pipelines.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import warnings

# Suppress warnings for cleaner CI output
warnings.filterwarnings('ignore')

from .validator import OptimizationValidator
from .test_problems import get_test_problems, get_problem_categories


class ContinuousValidator:
    """
    Continuous validation runner for CI/CD pipelines.

    Provides automated testing, threshold checking, and report generation.
    """

    def __init__(self,
                 output_dir: str = "reports",
                 success_threshold: float = 0.7,
                 verbose: bool = True):
        """
        Initialize continuous validator.

        Args:
            output_dir: Directory for output reports
            success_threshold: Minimum success rate required (0.7 = 70%)
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.success_threshold = success_threshold
        self.verbose = verbose

        # Initialize validator
        self.validator = OptimizationValidator(
            objective_tolerance=1e-3,
            solution_tolerance=1e-2,
            relative_tolerance=0.01,
            verbose=verbose
        )

        # Test configurations
        self.test_configs = self._get_test_configurations()

    def _get_test_configurations(self) -> Dict[str, Any]:
        """Get test configurations for different validation levels."""
        return {
            'quick': {
                'methods': ['random_search', 'simulated_annealing', 'genetic_algorithm'],
                'dimensions': [2, 5],
                'n_runs': 1,
                'max_problems': 10,
                'description': 'Quick validation for rapid feedback'
            },
            'standard': {
                'methods': [
                    'random_search', 'simulated_annealing', 'genetic_algorithm',
                    'local_search', 'tabu_search'
                ],
                'dimensions': [2, 5, 10],
                'n_runs': 3,
                'max_problems': 20,
                'description': 'Standard validation for regular CI'
            },
            'comprehensive': {
                'methods': [
                    'random_search', 'simulated_annealing', 'genetic_algorithm',
                    'local_search', 'tabu_search', 'pso', 'aco', 'ils', 'vns', 'grasp'
                ],
                'dimensions': [2, 5, 10, 20],
                'n_runs': 5,
                'max_problems': None,  # Test all problems
                'description': 'Comprehensive validation for release'
            },
            'nightly': {
                'methods': None,  # Test all available methods
                'dimensions': [2, 5, 10, 20, 50],
                'n_runs': 10,
                'max_problems': None,
                'description': 'Nightly exhaustive validation'
            }
        }

    def run_validation_suite(self,
                           level: str = 'standard',
                           methods_override: List[str] = None) -> Dict[str, Any]:
        """
        Run validation suite at specified level.

        Args:
            level: Validation level ('quick', 'standard', 'comprehensive', 'nightly')
            methods_override: Override methods to test

        Returns:
            Dictionary with validation results and pass/fail status
        """
        if level not in self.test_configs:
            raise ValueError(f"Invalid validation level: {level}")

        config = self.test_configs[level]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running {level.upper()} Validation Suite")
            print(f"Description: {config['description']}")
            print(f"{'='*60}\n")

        # Get methods to test
        methods = methods_override or config['methods']
        if methods is None:
            methods = self.validator.available_methods

        # Get test problems
        test_problems = get_test_problems(config['dimensions'])

        # Limit problems if specified
        if config['max_problems']:
            test_problems = test_problems[:config['max_problems']]

        # Extract problem names
        problem_names = list(set([p.name.split('_')[0] for p in test_problems]))

        # Run validation
        results_df = self.validator.compare_methods(
            methods=methods,
            dimensions=config['dimensions'],
            problem_names=problem_names[:config['max_problems']] if config['max_problems'] else None,
            n_runs=config['n_runs'],
            seed=42
        )

        # Generate reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"validation_{level}_{timestamp}"

        # Generate HTML report
        html_path = self.output_dir / f"{report_name}.html"
        self.validator.generate_report(
            results_df,
            str(html_path),
            title=f"Librex {level.capitalize()} Validation Report"
        )

        # Export raw data
        csv_path = self.output_dir / f"{report_name}.csv"
        self.validator.export_results(results_df, str(csv_path))

        # Compute pass/fail status
        validation_results = self._check_thresholds(results_df, methods)

        # Save summary JSON
        json_path = self.output_dir / f"{report_name}_summary.json"
        self._save_summary_json(validation_results, json_path)

        # Print summary
        self._print_summary(validation_results)

        return validation_results

    def _check_thresholds(self,
                         results_df: pd.DataFrame,
                         methods: List[str]) -> Dict[str, Any]:
        """
        Check if results meet success thresholds.

        Args:
            results_df: DataFrame with results
            methods: List of methods tested

        Returns:
            Dictionary with pass/fail status and details
        """
        validation_results = {
            'overall_passed': True,
            'timestamp': datetime.now().isoformat(),
            'methods': {},
            'summary': {
                'total_tests': len(results_df),
                'total_passed': results_df['passed'].sum(),
                'overall_success_rate': results_df['passed'].mean(),
            }
        }

        # Check each method
        for method in methods:
            method_results = results_df[results_df['method_name'] == method]

            if len(method_results) == 0:
                continue

            success_rate = method_results['passed'].mean()
            passed = success_rate >= self.success_threshold

            validation_results['methods'][method] = {
                'success_rate': success_rate,
                'passed': passed,
                'total_tests': len(method_results),
                'tests_passed': method_results['passed'].sum(),
                'mean_error': method_results['objective_error'].mean(),
                'mean_runtime': method_results['runtime'].mean(),
            }

            if not passed:
                validation_results['overall_passed'] = False

        # Problem-specific analysis
        problem_stats = results_df.groupby('problem_name').agg({
            'passed': 'mean',
            'objective_error': 'mean',
        }).sort_values('passed')

        validation_results['hardest_problems'] = problem_stats.head(5).to_dict()
        validation_results['easiest_problems'] = problem_stats.tail(5).to_dict()

        return validation_results

    def _save_summary_json(self, results: Dict[str, Any], path: Path):
        """Save summary results to JSON file."""
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        clean_results = convert_numpy(results)

        with open(path, 'w') as f:
            json.dump(clean_results, f, indent=2)

    def _print_summary(self, results: Dict[str, Any]):
        """Print validation summary to console."""
        if not self.verbose:
            return

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        # Overall status
        status = "✅ PASSED" if results['overall_passed'] else "❌ FAILED"
        print(f"\nOverall Status: {status}")
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Tests Passed: {results['summary']['total_passed']}")
        print(f"Success Rate: {results['summary']['overall_success_rate']:.1%}")

        # Method-specific results
        print("\n" + "-"*40)
        print("Method Performance:")
        print("-"*40)

        for method, stats in results['methods'].items():
            status_icon = "✅" if stats['passed'] else "❌"
            print(f"{status_icon} {method:20s} - Success: {stats['success_rate']:.1%} "
                  f"({stats['tests_passed']}/{stats['total_tests']})")

        # Hardest problems
        print("\n" + "-"*40)
        print("Hardest Problems (lowest success rate):")
        print("-"*40)

        for problem in list(results['hardest_problems']['passed'].keys())[:3]:
            success_rate = results['hardest_problems']['passed'][problem]
            print(f"  {problem}: {success_rate:.1%} success rate")

        print("\n" + "="*60)

    def run_regression_test(self,
                          baseline_file: str,
                          methods: List[str] = None) -> bool:
        """
        Run regression test against baseline results.

        Args:
            baseline_file: Path to baseline results JSON
            methods: Methods to test

        Returns:
            True if no regression detected
        """
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        # Run current validation
        current = self.run_validation_suite(level='standard', methods_override=methods)

        # Compare results
        regression_detected = False

        print("\n" + "="*60)
        print("REGRESSION TEST RESULTS")
        print("="*60)

        for method in current['methods']:
            if method not in baseline['methods']:
                print(f"⚠️  {method}: New method (no baseline)")
                continue

            current_rate = current['methods'][method]['success_rate']
            baseline_rate = baseline['methods'][method]['success_rate']

            # Allow 5% degradation tolerance
            if current_rate < baseline_rate - 0.05:
                regression_detected = True
                print(f"❌ {method}: REGRESSION - {baseline_rate:.1%} → {current_rate:.1%}")
            elif current_rate > baseline_rate + 0.05:
                print(f"✅ {method}: IMPROVED - {baseline_rate:.1%} → {current_rate:.1%}")
            else:
                print(f"➡️  {method}: No change - {current_rate:.1%}")

        return not regression_detected

    def generate_benchmark_report(self,
                                 methods: List[str] = None,
                                 output_file: str = None):
        """
        Generate detailed benchmark report.

        Args:
            methods: Methods to benchmark
            output_file: Output file path
        """
        if methods is None:
            methods = ['simulated_annealing', 'genetic_algorithm', 'tabu_search']

        # Test on different problem categories
        categories = get_problem_categories()

        results = []

        for category, problem_names in categories.items():
            if self.verbose:
                print(f"\nBenchmarking {category} problems...")

            # Get problems in this category
            cat_problems = [p for p in get_test_problems([2, 5, 10])
                          if any(name in p.name.lower() for name in problem_names)]

            if not cat_problems:
                continue

            # Run validation
            for method in methods:
                for problem in cat_problems[:3]:  # Test first 3 problems per category
                    result = self.validator._run_single_validation(
                        method, problem, seed=42
                    )

                    results.append({
                        'category': category,
                        'method': method,
                        'problem': problem.name,
                        'dimension': problem.dimension,
                        'success': result.passed,
                        'error': result.objective_error,
                        'runtime': result.runtime,
                        'evaluations': result.evaluations,
                    })

        # Create DataFrame
        df = pd.DataFrame(results)

        # Generate report
        if output_file is None:
            output_file = self.output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.html"

        self._generate_benchmark_html(df, output_file)

        if self.verbose:
            print(f"\nBenchmark report saved to {output_file}")

    def _generate_benchmark_html(self, df: pd.DataFrame, output_file: str):
        """Generate HTML benchmark report."""
        # Aggregate statistics
        by_category = df.groupby(['category', 'method']).agg({
            'success': 'mean',
            'error': 'mean',
            'runtime': 'mean',
            'evaluations': 'mean',
        }).round(3)

        by_method = df.groupby('method').agg({
            'success': 'mean',
            'error': 'mean',
            'runtime': 'mean',
            'evaluations': 'mean',
        }).round(3)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Librex Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .category {{ background-color: #f0f0f0; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Librex Benchmark Report</h1>
            <p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>

            <h2>Overall Performance by Method</h2>
            {by_method.to_html()}

            <h2>Performance by Problem Category</h2>
            {by_category.to_html()}

            <h2>Detailed Results</h2>
            {df.to_html()}
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html)


def main():
    """Main entry point for CI/CD validation."""
    parser = argparse.ArgumentParser(
        description='Run Librex validation suite for CI/CD'
    )
    parser.add_argument(
        '--level',
        choices=['quick', 'standard', 'comprehensive', 'nightly'],
        default='standard',
        help='Validation level'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        help='Specific methods to test'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Success rate threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--output-dir',
        default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--regression-baseline',
        help='Baseline file for regression testing'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark suite'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Initialize validator
    validator = ContinuousValidator(
        output_dir=args.output_dir,
        success_threshold=args.threshold,
        verbose=not args.quiet
    )

    # Run appropriate validation
    if args.regression_baseline:
        # Run regression test
        success = validator.run_regression_test(
            args.regression_baseline,
            args.methods
        )
        sys.exit(0 if success else 1)

    elif args.benchmark:
        # Run benchmark suite
        validator.generate_benchmark_report(args.methods)
        sys.exit(0)

    else:
        # Run standard validation
        results = validator.run_validation_suite(
            level=args.level,
            methods_override=args.methods
        )

        # Exit with appropriate code
        sys.exit(0 if results['overall_passed'] else 1)


if __name__ == '__main__':
    main()