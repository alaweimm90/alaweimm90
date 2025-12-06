"""
Test Script for Enhanced Server Features
=========================================
Demonstrates all new production-grade features of server_enhanced.py
"""

import requests
import time
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_response(response: requests.Response):
    """Pretty print response."""
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    try:
        print(f"Body: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Body: {response.text}")


# ============================================================================
# 1. BASIC HEALTH CHECKS
# ============================================================================

def test_health_checks():
    """Test health and readiness endpoints."""
    print_section("1. Health Checks")

    # Basic health
    print("1.1. Basic Health Check:")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)

    # Readiness check
    print("\n1.2. Readiness Check:")
    response = requests.get(f"{BASE_URL}/ready")
    print_response(response)

    # Root endpoint
    print("\n1.3. Root Endpoint:")
    response = requests.get(f"{BASE_URL}/")
    print_response(response)


# ============================================================================
# 2. ERROR HANDLING
# ============================================================================

def test_error_handling():
    """Test advanced error handling."""
    print_section("2. Error Handling")

    # Validation error - invalid matrix size
    print("2.1. Validation Error (Matrix Size Mismatch):")
    response = requests.post(
        f"{BASE_URL}/solve",
        json={
            "problem_size": 10,
            "problem_matrix": [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]
            ],
            "method": "fft_laplace"
        }
    )
    print_response(response)

    # Validation error - problem size too large
    print("\n2.2. Validation Error (Problem Size Exceeds Maximum):")
    response = requests.post(
        f"{BASE_URL}/solve",
        json={
            "problem_size": 2000,
            "problem_matrix": [[0]],
            "method": "fft_laplace"
        }
    )
    print_response(response)

    # Validation error - problem size too small
    print("\n2.3. Validation Error (Problem Size Too Small):")
    response = requests.post(
        f"{BASE_URL}/solve",
        json={
            "problem_size": 1,
            "problem_matrix": [[0]],
            "method": "fft_laplace"
        }
    )
    print_response(response)


# ============================================================================
# 3. BASIC SOLVE
# ============================================================================

def test_basic_solve():
    """Test basic solve functionality."""
    print_section("3. Basic Solve")

    print("3.1. Solve with FFT-Laplace:")
    response = requests.post(
        f"{BASE_URL}/solve",
        json={
            "problem_size": 5,
            "problem_matrix": [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]
            ],
            "method": "fft_laplace",
            "iterations": 500,
            "timeout_seconds": 60
        }
    )
    print_response(response)


# ============================================================================
# 4. CACHING & DEDUPLICATION
# ============================================================================

def test_caching():
    """Test response caching and deduplication."""
    print_section("4. Caching & Deduplication")

    problem_data = {
        "problem_size": 5,
        "problem_matrix": [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ],
        "method": "genetic_algorithm",
        "iterations": 1000
    }

    # First request (cache miss)
    print("4.1. First Request (Cache Miss):")
    start = time.time()
    response1 = requests.post(f"{BASE_URL}/solve", json=problem_data)
    time1 = time.time() - start
    result1 = response1.json()
    print(f"Time: {time1:.3f}s")
    print(f"Cached: {result1.get('cached', False)}")
    print(f"Request ID: {result1.get('request_id')}")

    # Second request (cache hit)
    print("\n4.2. Second Request (Cache Hit):")
    start = time.time()
    response2 = requests.post(f"{BASE_URL}/solve", json=problem_data)
    time2 = time.time() - start
    result2 = response2.json()
    print(f"Time: {time2:.3f}s")
    print(f"Cached: {result2.get('cached', False)}")
    print(f"Request ID: {result2.get('request_id')}")
    print(f"Speed improvement: {time1/time2:.2f}x faster")


# ============================================================================
# 5. BATCH SOLVING
# ============================================================================

def test_batch_solving():
    """Test batch solving functionality."""
    print_section("5. Batch Solving")

    print("5.1. Batch Solve (Parallel):")
    response = requests.post(
        f"{BASE_URL}/solve/batch",
        json={
            "parallel": True,
            "problems": [
                {
                    "problem_size": 5,
                    "problem_matrix": [
                        [0, 1, 2, 3, 4],
                        [1, 0, 1, 2, 3],
                        [2, 1, 0, 1, 2],
                        [3, 2, 1, 0, 1],
                        [4, 3, 2, 1, 0]
                    ],
                    "method": "fft_laplace"
                },
                {
                    "problem_size": 5,
                    "problem_matrix": [
                        [0, 2, 4, 6, 8],
                        [2, 0, 2, 4, 6],
                        [4, 2, 0, 2, 4],
                        [6, 4, 2, 0, 2],
                        [8, 6, 4, 2, 0]
                    ],
                    "method": "genetic_algorithm"
                },
                {
                    "problem_size": 5,
                    "problem_matrix": [
                        [0, 3, 6, 9, 12],
                        [3, 0, 3, 6, 9],
                        [6, 3, 0, 3, 6],
                        [9, 6, 3, 0, 3],
                        [12, 9, 6, 3, 0]
                    ],
                    "method": "simulated_annealing"
                }
            ]
        }
    )
    print_response(response)


# ============================================================================
# 6. ASYNC SOLVING
# ============================================================================

def test_async_solving():
    """Test asynchronous solving with polling."""
    print_section("6. Async Solving")

    # Submit async request
    print("6.1. Submit Async Request:")
    response = requests.post(
        f"{BASE_URL}/solve-async",
        json={
            "problem_size": 5,
            "problem_matrix": [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]
            ],
            "method": "reverse_time",
            "iterations": 1000
        }
    )
    result = response.json()
    print_response(response)

    request_id = result.get("request_id")
    if not request_id:
        print("Failed to get request ID")
        return

    # Poll for status
    print(f"\n6.2. Polling for Status (Request ID: {request_id}):")
    max_polls = 10
    for i in range(max_polls):
        time.sleep(0.5)
        status_response = requests.get(f"{BASE_URL}/solve/{request_id}/status")
        status = status_response.json()
        print(f"Poll {i+1}: Status = {status.get('status')}")

        if status.get("status") == "completed":
            print("\nFinal Result:")
            print_response(status_response)
            break
    else:
        print(f"Request still not completed after {max_polls} polls")


# ============================================================================
# 7. METRICS & MONITORING
# ============================================================================

def test_metrics():
    """Test metrics and monitoring endpoints."""
    print_section("7. Metrics & Monitoring")

    # General metrics
    print("7.1. General Metrics:")
    response = requests.get(f"{BASE_URL}/metrics")
    print_response(response)

    # Detailed stats
    print("\n7.2. Detailed Statistics:")
    response = requests.get(f"{BASE_URL}/stats")
    print_response(response)

    # Prometheus metrics
    print("\n7.3. Prometheus Metrics:")
    response = requests.get(f"{BASE_URL}/metrics/prometheus")
    print(f"Status: {response.status_code}")
    print("Metrics Preview:")
    lines = response.text.split('\n')[:20]
    for line in lines:
        if line and not line.startswith('#'):
            print(f"  {line}")


# ============================================================================
# 8. ANALYTICS & HISTORY
# ============================================================================

def test_analytics():
    """Test analytics and history endpoints."""
    print_section("8. Analytics & History")

    # Get request history
    print("8.1. Request History:")
    response = requests.get(f"{BASE_URL}/analytics/history?limit=10")
    print_response(response)

    # Export as JSON
    print("\n8.2. Export History (JSON):")
    response = requests.get(f"{BASE_URL}/analytics/export/json")
    result = response.json()
    print(f"Total Records: {result.get('total_records')}")
    print(f"First 3 records:")
    for record in result.get('data', [])[:3]:
        print(f"  - {record}")


# ============================================================================
# 9. ADMIN OPERATIONS
# ============================================================================

def test_admin_operations():
    """Test admin operations."""
    print_section("9. Admin Operations")

    # Admin status
    print("9.1. Admin Status:")
    response = requests.get(f"{BASE_URL}/admin/status")
    print_response(response)

    # Circuit breaker status
    print("\n9.2. Circuit Breaker Status:")
    status = requests.get(f"{BASE_URL}/admin/status").json()
    print(f"Circuit Breaker State: {status.get('circuit_breaker', {})}")

    # Note: Skipping clear and reset operations to preserve test data


# ============================================================================
# 10. METHODS INFORMATION
# ============================================================================

def test_methods_info():
    """Test methods information endpoints."""
    print_section("10. Methods Information")

    # List all methods
    print("10.1. List All Methods:")
    response = requests.get(f"{BASE_URL}/methods")
    methods = response.json()
    print(f"Total Methods: {len(methods)}")
    for method in methods[:3]:  # Show first 3
        print(f"\n  {method['name']}:")
        print(f"    Description: {method['description']}")
        print(f"    Complexity: {method['complexity_time']}")
        print(f"    Avg Quality: {method['avg_quality']}")
        print(f"    Avg Runtime: {method['avg_runtime_ms']}ms")

    # Get specific method
    print("\n10.2. Get Specific Method (fft_laplace):")
    response = requests.get(f"{BASE_URL}/methods/fft_laplace")
    print_response(response)


# ============================================================================
# 11. RATE LIMITING
# ============================================================================

def test_rate_limiting():
    """Test rate limiting (be careful not to trigger actual limits)."""
    print_section("11. Rate Limiting Headers")

    print("11.1. Check Rate Limit Headers:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"X-RateLimit-Limit: {response.headers.get('X-RateLimit-Limit', 'N/A')}")
    print(f"X-RateLimit-Reset: {response.headers.get('X-RateLimit-Reset', 'N/A')}")
    print(f"X-Request-ID: {response.headers.get('X-Request-ID', 'N/A')}")
    print(f"X-Response-Time: {response.headers.get('X-Response-Time', 'N/A')}")


# ============================================================================
# 12. BENCHMARK
# ============================================================================

def test_benchmark():
    """Test benchmark functionality."""
    print_section("12. Benchmark")

    print("12.1. Run Benchmark:")
    response = requests.post(
        f"{BASE_URL}/benchmark",
        json={
            "instance_name": "test_instance_5x5",
            "methods": ["fft_laplace", "genetic_algorithm", "simulated_annealing"],
            "num_runs": 3,
            "iterations_per_run": 500
        }
    )
    result = response.json()
    print(f"Benchmark ID: {result.get('benchmark_id')}")
    print(f"Total Results: {len(result.get('results', []))}")
    print(f"\nStatistics:")
    for method, stats in result.get('statistics', {}).items():
        print(f"  {method}:")
        print(f"    Avg Quality: {stats.get('avg_quality', 0):.4f}")
        print(f"    Avg Runtime: {stats.get('avg_runtime', 0):.3f}s")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print("  Librex.QAP-new Enhanced Server - Feature Test Suite")
    print("=" * 70)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.RequestException:
        print(f"\n⚠️  ERROR: Server not running at {BASE_URL}")
        print("Please start the server first:")
        print("  python server_enhanced.py")
        return

    test_suites = [
        ("Health Checks", test_health_checks),
        ("Error Handling", test_error_handling),
        ("Basic Solve", test_basic_solve),
        ("Caching & Deduplication", test_caching),
        ("Batch Solving", test_batch_solving),
        ("Async Solving", test_async_solving),
        ("Metrics & Monitoring", test_metrics),
        ("Analytics & History", test_analytics),
        ("Admin Operations", test_admin_operations),
        ("Methods Information", test_methods_info),
        ("Rate Limiting", test_rate_limiting),
        ("Benchmark", test_benchmark),
    ]

    for name, test_func in test_suites:
        try:
            test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  Test Suite Completed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run specific test or all tests
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            "health": test_health_checks,
            "errors": test_error_handling,
            "solve": test_basic_solve,
            "cache": test_caching,
            "batch": test_batch_solving,
            "async": test_async_solving,
            "metrics": test_metrics,
            "analytics": test_analytics,
            "admin": test_admin_operations,
            "methods": test_methods_info,
            "ratelimit": test_rate_limiting,
            "benchmark": test_benchmark,
        }

        if test_name in test_map:
            test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
    else:
        run_all_tests()
