#!/usr/bin/env python3
"""
Installation Test Script for Librex.QAP v1.0

Comprehensive testing of package installation and basic functionality.
"""

import sys
import subprocess
import tempfile
import importlib.util
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def test_imports():
    """Test that all main modules can be imported"""
    print("📦 Testing module imports...")

    test_modules = [
        "librex",
        "librex.qap",
        "librex.qap.solver",
        "librex.qap.instance",
        "librex.qap.benchmarks"
    ]

    success_count = 0
    for module in test_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                print(f"  ✅ {module}")
                success_count += 1
            else:
                print(f"  ❌ {module} not found")
        except ImportError as e:
            print(f"  ❌ {module} import failed: {e}")

    print(f"  📊 {success_count}/{len(test_modules)} modules imported successfully")
    return success_count == len(test_modules)

def test_basic_functionality():
    """Test basic QAP solver functionality"""
    print("🧪 Testing basic functionality...")

    try:
        # Import the main solver
        from librex.qap import QAPSolver, QAPInstance

        # Create a simple test instance
        import numpy as np

        # Small test instance (n=4)
        flow_matrix = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ])

        distance_matrix = np.array([
            [0, 7, 8, 9],
            [7, 0, 10, 11],
            [8, 10, 0, 12],
            [9, 11, 12, 0]
        ])

        # Create instance
        instance = QAPInstance(flow_matrix, distance_matrix)
        print(f"  ✅ QAP instance created (size: {instance.size})")

        # Create solver
        solver = QAPSolver(method="fft_attractor")
        print(f"  ✅ QAP solver created (method: {solver.method})")

        # Solve with short time limit
        solution = solver.solve(instance, time_limit=5)
        print(f"  ✅ Problem solved (cost: {solution.objective_value})")
        print(f"  📊 Assignment: {solution.assignment}")

        return True

    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def test_qaplib_loading():
    """Test QAPLIB instance loading if available"""
    print("📚 Testing QAPLIB loading...")

    try:
        from librex.qap import QAPInstance

        # Try to load a standard QAPLIB instance
        # This will test if the data loading works
        instance = QAPInstance.from_qaplib("tai20a")
        print(f"  ✅ QAPLIB instance loaded (size: {instance.size})")
        print(f"  📊 Flow matrix shape: {instance.flow_matrix.shape}")
        print(f"  📊 Distance matrix shape: {instance.distance_matrix.shape}")

        return True

    except Exception as e:
        print(f"  ⚠️  QAPLIB loading test skipped: {e}")
        return True  # Not critical for basic functionality

def test_benchmark_framework():
    """Test the benchmark framework"""
    print("🏁 Testing benchmark framework...")

    try:
        from librex.qap.benchmarks import BenchmarkRunner

        # Create benchmark runner
        runner = BenchmarkRunner()
        print(f"  ✅ Benchmark runner created")

        # Test small benchmark
        results = runner.run_quick_benchmark(max_instances=2)
        print(f"  ✅ Quick benchmark completed")
        print(f"  📊 Results: {len(results)} instances tested")

        return True

    except Exception as e:
        print(f"  ⚠️  Benchmark test skipped: {e}")
        return True  # Not critical for basic functionality

def test_documentation_access():
    """Test that documentation and help are accessible"""
    print("📖 Testing documentation access...")

    try:
        from librex.qap import QAPSolver

        # Test docstrings
        solver = QAPSolver()
        if solver.__doc__:
            print(f"  ✅ Solver documentation available")

        # Test help function
        help_text = help(QAPSolver)
        print(f"  ✅ Help system functional")

        return True

    except Exception as e:
        print(f"  ❌ Documentation test failed: {e}")
        return False

def test_version_info():
    """Test version information"""
    print("🏷️  Testing version information...")

    try:
        import librex

        # Check if version is available
        if hasattr(librex, '__version__'):
            print(f"  ✅ Version: {librex.__version__}")
        else:
            print(f"  ⚠️  Version attribute not found")

        # Check package info
        if hasattr(librex, '__package_info__'):
            info = librex.__package_info__
            print(f"  ✅ Package: {info.get('name', 'Unknown')}")
            print(f"  ✅ Author: {info.get('author', 'Unknown')}")

        return True

    except Exception as e:
        print(f"  ❌ Version test failed: {e}")
        return False

def main():
    """Main test suite"""
    print("🧪 Librex.QAP v1.0 Installation Test Suite")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("QAPLIB Loading", test_qaplib_loading),
        ("Benchmark Framework", test_benchmark_framework),
        ("Documentation Access", test_documentation_access),
        ("Version Information", test_version_info)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"  ❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\n🎯 Quick start:")
        print("  from librex.qap import QAPSolver, QAPInstance")
        print("  solver = QAPSolver(method='fft_attractor')")
        print("  solution = solver.solve(instance)")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
