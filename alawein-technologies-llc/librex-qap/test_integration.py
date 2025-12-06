"""
Integration tests for Librex.QAP-new API and Dashboard
Tests actual server functionality
"""

import requests
import json
import time
from typing import Dict, Any, List
import sys

# Configuration
API_URL = "http://localhost:8000"
TIMEOUT = 10

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name: str):
    """Print test name."""
    print(f"\n{Colors.BLUE}▶ {name}{Colors.END}")

def print_pass(msg: str = "PASS"):
    """Print pass message."""
    print(f"  {Colors.GREEN}✓ {msg}{Colors.END}")

def print_fail(msg: str = "FAIL"):
    """Print fail message."""
    print(f"  {Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg: str):
    """Print info message."""
    print(f"  {Colors.YELLOW}ℹ {msg}{Colors.END}")

# ============================================================================
# TEST SUITE 1: HEALTH & STATUS
# ============================================================================

def test_health_check():
    """Test /health endpoint."""
    print_test("Health Check")

    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data["status"] == "healthy", f"Status is {data['status']}"
        assert "timestamp" in data, "Missing timestamp"

        print_pass(f"Server healthy: {data['timestamp']}")
        return True
    except Exception as e:
        print_fail(f"Health check failed: {str(e)}")
        return False

def test_readiness_check():
    """Test /ready endpoint."""
    print_test("Readiness Check")

    try:
        response = requests.get(f"{API_URL}/ready", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data["status"] == "ready", f"Status is {data['status']}"

        print_pass(f"Service ready with components: {list(data['components'].keys())}")
        return True
    except Exception as e:
        print_fail(f"Readiness check failed: {str(e)}")
        return False

# ============================================================================
# TEST SUITE 2: SOLVE ENDPOINTS
# ============================================================================

def test_solve_basic():
    """Test basic /solve endpoint."""
    print_test("Basic Solve Endpoint")

    try:
        # Create simple 5x5 problem
        n = 5
        problem_matrix = [[float(i+j) for j in range(n)] for i in range(n)]

        payload = {
            "problem_size": n,
            "problem_matrix": problem_matrix,
            "method": "fft_laplace",
            "iterations": 100
        }

        response = requests.post(
            f"{API_URL}/solve",
            json=payload,
            timeout=TIMEOUT
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        assert result["status"] == "completed", f"Status is {result['status']}"
        assert result["method"] == "fft_laplace", f"Method is {result['method']}"
        assert result["problem_size"] == n, f"Size is {result['problem_size']}"
        assert len(result["best_solution"]) == n, f"Solution size is {len(result['best_solution'])}"
        assert result["objective_value"] > 0, "Objective should be positive"
        assert result["runtime_seconds"] > 0, "Runtime should be positive"

        print_pass(f"Solution found: obj={result['objective_value']:.2f}, "
                   f"time={result['runtime_seconds']:.3f}s")
        return True
    except Exception as e:
        print_fail(f"Basic solve failed: {str(e)}")
        return False

def test_solve_all_methods():
    """Test /solve with all methods."""
    print_test("Solve with All Methods")

    methods = [
        "fft_laplace",
        "reverse_time",
        "genetic_algorithm",
        "simulated_annealing",
        "tabu_search"
    ]

    try:
        n = 10
        problem_matrix = [[float(i+j) for j in range(n)] for i in range(n)]

        for method in methods:
            payload = {
                "problem_size": n,
                "problem_matrix": problem_matrix,
                "method": method,
                "iterations": 50
            }

            response = requests.post(
                f"{API_URL}/solve",
                json=payload,
                timeout=TIMEOUT
            )

            assert response.status_code == 200, f"Failed for {method}"
            result = response.json()
            assert result["method"] == method, f"Method mismatch for {method}"

            print_pass(f"{method}: obj={result['objective_value']:.2f}")

        return True
    except Exception as e:
        print_fail(f"Method testing failed: {str(e)}")
        return False

def test_solve_invalid_input():
    """Test /solve with invalid inputs."""
    print_test("Invalid Input Handling")

    try:
        # Test 1: Mismatched sizes
        payload = {
            "problem_size": 5,
            "problem_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3x3, not 5x5
            "method": "fft_laplace",
            "iterations": 100
        }

        response = requests.post(
            f"{API_URL}/solve",
            json=payload,
            timeout=TIMEOUT
        )

        assert response.status_code == 422, "Should reject mismatched sizes"
        print_pass("Correctly rejected mismatched matrix size")

        return True
    except Exception as e:
        print_fail(f"Invalid input test failed: {str(e)}")
        return False

# ============================================================================
# TEST SUITE 3: METHODS ENDPOINTS
# ============================================================================

def test_list_methods():
    """Test /methods endpoint."""
    print_test("List Methods")

    try:
        response = requests.get(f"{API_URL}/methods", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        methods = response.json()
        assert len(methods) > 0, "No methods returned"
        assert len(methods) == 8, f"Expected 8 methods, got {len(methods)}"

        for method in methods:
            assert "name" in method, f"Missing 'name' in {method}"
            assert "description" in method, f"Missing 'description' in {method}"
            assert "avg_quality" in method, f"Missing 'avg_quality' in {method}"

        print_pass(f"Found {len(methods)} methods")
        for m in methods:
            print_info(f"{m['name']}: {m['description'][:50]}...")
        return True
    except Exception as e:
        print_fail(f"List methods failed: {str(e)}")
        return False

def test_get_method():
    """Test /methods/{method_name} endpoint."""
    print_test("Get Specific Method")

    try:
        response = requests.get(
            f"{API_URL}/methods/fft_laplace",
            timeout=TIMEOUT
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        method = response.json()
        assert method["name"] == "fft_laplace", "Wrong method returned"
        assert "complexity_time" in method, "Missing complexity_time"
        assert "parameters" in method, "Missing parameters"

        print_pass(f"Method: {method['name']}")
        print_info(f"Complexity: {method['complexity_time']}")
        return True
    except Exception as e:
        print_fail(f"Get method failed: {str(e)}")
        return False

# ============================================================================
# TEST SUITE 4: METRICS & ANALYTICS
# ============================================================================

def test_metrics():
    """Test /metrics endpoint."""
    print_test("Metrics Endpoint")

    try:
        response = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        metrics = response.json()
        assert "total_optimizations" in metrics, "Missing total_optimizations"
        assert "methods_available" in metrics, "Missing methods_available"

        print_pass(f"Total optimizations: {metrics['total_optimizations']}")
        print_info(f"Methods available: {metrics['methods_available']}")
        return True
    except Exception as e:
        print_fail(f"Metrics test failed: {str(e)}")
        return False

def test_stats():
    """Test /stats endpoint."""
    print_test("Statistics Endpoint")

    try:
        response = requests.get(f"{API_URL}/stats", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        stats = response.json()
        assert "total_requests" in stats, "Missing total_requests"
        assert "average_quality" in stats, "Missing average_quality"
        assert "average_runtime_seconds" in stats, "Missing average_runtime_seconds"

        print_pass(f"Total requests: {stats['total_requests']}")
        print_info(f"Avg quality: {stats['average_quality']:.2%}")
        print_info(f"Avg runtime: {stats['average_runtime_seconds']:.3f}s")
        return True
    except Exception as e:
        print_fail(f"Stats test failed: {str(e)}")
        return False

# ============================================================================
# TEST SUITE 5: BENCHMARK
# ============================================================================

def test_benchmark():
    """Test /benchmark endpoint."""
    print_test("Benchmark Endpoint")

    try:
        payload = {
            "instance_name": "nug20",
            "methods": ["fft_laplace", "reverse_time"],
            "num_runs": 2,
            "iterations_per_run": 100
        }

        response = requests.post(
            f"{API_URL}/benchmark",
            json=payload,
            timeout=TIMEOUT
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        assert "results" in result, "Missing results"
        assert len(result["results"]) > 0, "No results returned"
        assert result["status"] == "completed", f"Status is {result['status']}"

        print_pass(f"Benchmark completed: {len(result['results'])} runs")
        print_info(f"Methods: {', '.join([r['method'] for r in result['results'][:2]])}")
        return True
    except Exception as e:
        print_fail(f"Benchmark test failed: {str(e)}")
        return False

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_response_time():
    """Test response times."""
    print_test("Response Time Performance")

    try:
        endpoints = [
            "/health",
            "/methods",
            "/metrics"
        ]

        times = {}
        for endpoint in endpoints:
            start = time.time()
            response = requests.get(f"{API_URL}{endpoint}", timeout=TIMEOUT)
            elapsed = time.time() - start

            assert response.status_code == 200, f"Failed for {endpoint}"
            times[endpoint] = elapsed * 1000  # Convert to ms

        for endpoint, duration in times.items():
            status = "✓" if duration < 100 else "!" if duration < 500 else "✗"
            print_info(f"{endpoint}: {duration:.1f}ms {status}")

        avg = sum(times.values()) / len(times)
        assert avg < 500, f"Average response time too high: {avg:.1f}ms"

        print_pass(f"Average response time: {avg:.1f}ms")
        return True
    except Exception as e:
        print_fail(f"Response time test failed: {str(e)}")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Librex.QAP-new Integration Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    tests = [
        # Health & Status
        test_health_check,
        test_readiness_check,

        # Solve Endpoints
        test_solve_basic,
        test_solve_all_methods,
        test_solve_invalid_input,

        # Methods
        test_list_methods,
        test_get_method,

        # Metrics
        test_metrics,
        test_stats,

        # Benchmarks
        test_benchmark,

        # Performance
        test_response_time,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_fail(f"Unexpected error: {str(e)}")
            failed += 1

    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"Test Results: {Colors.GREEN}{passed} passed{Colors.END}, "
          f"{Colors.RED}{failed} failed{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    return failed == 0

if __name__ == "__main__":
    print_info(f"Testing API at {API_URL}")
    print_info("Make sure to run 'python server.py' first")

    success = run_all_tests()
    sys.exit(0 if success else 1)
