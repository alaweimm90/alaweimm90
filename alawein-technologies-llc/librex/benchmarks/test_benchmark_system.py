#!/usr/bin/env python3
"""
Test script for the Librex benchmarking system

This script tests the core functionality of the benchmarking infrastructure.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from benchmarks.nightly_benchmark import BenchmarkRunner
        print("✓ nightly_benchmark imported successfully")

        from benchmarks.benchmark_history import BenchmarkHistory
        print("✓ benchmark_history imported successfully")

        from benchmarks.report_generator import (
            HTMLReportGenerator,
            MarkdownReportGenerator,
            CSVReportGenerator
        )
        print("✓ report_generator imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config_loading():
    """Test configuration file loading"""
    print("\nTesting configuration loading...")
    try:
        from benchmarks.nightly_benchmark import BenchmarkRunner

        runner = BenchmarkRunner()
        config = runner.config

        # Check required sections exist
        assert 'benchmark_suites' in config, "Missing benchmark_suites section"
        assert 'metrics' in config, "Missing metrics section"
        assert 'output' in config, "Missing output section"

        # Check suites
        suites = config['benchmark_suites']
        assert 'smoke' in suites, "Missing smoke suite"
        assert 'standard' in suites, "Missing standard suite"

        print("✓ Configuration loaded successfully")
        print(f"  Found {len(suites)} benchmark suites")
        print(f"  Found {len(config['metrics'])} metrics to track")

        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_mini_benchmark():
    """Run a minimal benchmark test"""
    print("\nRunning mini benchmark test...")
    try:
        from benchmarks.nightly_benchmark import BenchmarkRunner
        from Librex.adapters.qap import QAPAdapter
        import numpy as np

        runner = BenchmarkRunner()

        # Create a tiny test problem
        n = 5
        np.random.seed(42)
        flow = np.random.randint(1, 10, (n, n))
        distance = np.random.randint(1, 10, (n, n))

        # Make symmetric
        flow = (flow + flow.T) // 2
        distance = (distance + distance.T) // 2
        np.fill_diagonal(flow, 0)
        np.fill_diagonal(distance, 0)

        problem = {
            'flow_matrix': flow,
            'distance_matrix': distance
        }
        adapter = QAPAdapter()

        # Run a quick benchmark
        result = runner._benchmark_single_method(
            problem=problem,
            adapter=adapter,
            method='random_search',
            config={'n_iterations': 10},
            runs=2,
            problem_info={'type': 'qap', 'size': n, 'instance': 'test'}
        )

        # Check if benchmark succeeded (status only exists when failed)
        assert result.get('status') != 'failed', "Benchmark failed"
        assert 'statistics' in result, "Missing statistics"
        assert result['statistics']['mean_objective'] > 0, "Invalid objective value"

        print("✓ Mini benchmark completed successfully")
        print(f"  Method: {result['method']}")
        print(f"  Best objective: {result['statistics']['best_objective']:.2f}")
        print(f"  Mean runtime: {result['statistics']['mean_runtime']:.4f}s")

        return True
    except Exception as e:
        print(f"✗ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_history_tracking():
    """Test performance history tracking"""
    print("\nTesting history tracking...")
    try:
        from benchmarks.benchmark_history import BenchmarkHistory
        import tempfile
        import sqlite3

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        history = BenchmarkHistory(db_path)

        # Create mock results
        mock_results = {
            'suite_name': 'test_suite',
            'start_time': '2024-01-01T00:00:00',
            'total_duration': 10.5,
            'metadata': {'git_commit': 'abc123'},
            'results': [
                {
                    'problem': {'type': 'qap', 'size': 10},
                    'method': 'test_method',
                    'config': {},
                    'runs': 3,
                    'statistics': {
                        'mean_objective': 100.0,
                        'std_objective': 5.0,
                        'best_objective': 95.0,
                        'worst_objective': 105.0,
                        'mean_runtime': 0.5,
                        'mean_memory': 10.0,
                        'success_rate': 100.0
                    }
                }
            ]
        }

        # Record the run
        run_id = history.record_run(mock_results)
        assert run_id > 0, "Failed to record run"

        print(f"✓ History tracking working (run_id={run_id})")

        # Clean up
        history.close()
        Path(db_path).unlink()

        return True
    except Exception as e:
        print(f"✗ History tracking error: {e}")
        return False


def test_report_generation():
    """Test report generation"""
    print("\nTesting report generation...")
    try:
        from benchmarks.report_generator import (
            MarkdownReportGenerator,
            CSVReportGenerator
        )
        import tempfile

        # Create mock results
        mock_results = {
            'suite_name': 'test_suite',
            'suite_config': {'description': 'Test suite'},
            'start_time': '2024-01-01T00:00:00',
            'total_duration': 10.5,
            'metadata': {'git_commit': 'abc123'},
            'results': [
                {
                    'problem': {'type': 'qap', 'size': 10, 'instance': 'test1'},
                    'method': 'method1',
                    'config': {},
                    'runs': 3,
                    'statistics': {
                        'mean_objective': 100.0,
                        'std_objective': 5.0,
                        'best_objective': 95.0,
                        'worst_objective': 105.0,
                        'mean_runtime': 0.5,
                        'success_rate': 100.0
                    }
                },
                {
                    'problem': {'type': 'qap', 'size': 10, 'instance': 'test1'},
                    'method': 'method2',
                    'config': {},
                    'runs': 3,
                    'statistics': {
                        'mean_objective': 110.0,
                        'std_objective': 3.0,
                        'best_objective': 107.0,
                        'worst_objective': 113.0,
                        'mean_runtime': 0.3,
                        'success_rate': 100.0
                    }
                }
            ]
        }

        # Test Markdown generation
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            md_path = Path(tmp.name)

        md_gen = MarkdownReportGenerator()
        md_gen.generate(mock_results, md_path)

        assert md_path.exists(), "Markdown file not created"
        content = md_path.read_text()
        assert "test_suite" in content, "Suite name not in report"
        assert "method1" in content, "Method not in report"

        print("✓ Markdown report generated successfully")
        md_path.unlink()

        # Test CSV generation
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            csv_path = Path(tmp.name)

        csv_gen = CSVReportGenerator()
        csv_gen.generate(mock_results, csv_path)

        assert csv_path.exists(), "CSV file not created"
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert 'method' in df.columns, "Missing method column"

        print("✓ CSV export generated successfully")
        csv_path.unlink()

        return True
    except Exception as e:
        print(f"✗ Report generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Librex Benchmark System")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_loading),
        ("Mini Benchmark", test_mini_benchmark),
        ("History Tracking", test_history_tracking),
        ("Report Generation", test_report_generation)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The benchmarking system is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())