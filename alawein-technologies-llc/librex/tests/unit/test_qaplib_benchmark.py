"""
Unit tests for QAPLIB benchmark suite
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np

from Librex.benchmarks.qaplib import (
    load_qaplib_instance,
    list_qaplib_instances,
    get_qaplib_metadata,
    get_instance_by_size,
    get_instance_by_class,
    get_small_instances,
    get_all_instance_names,
    get_problem_classes,
    QAPLIBLoader,
    QAPLIBInstance,
    QAPLIB_REGISTRY,
    QAPLIBBenchmark,
    run_qaplib_benchmark,
)
from Librex.benchmarks.qaplib.embedded_data import EMBEDDED_INSTANCES


class TestRegistry:
    """Test QAPLIB registry"""

    def test_registry_size(self):
        """Test that registry contains all 138 instances"""
        assert len(QAPLIB_REGISTRY) == 138

    def test_registry_instances(self):
        """Test that key instances are present"""
        # Check some well-known instances
        assert "nug12" in QAPLIB_REGISTRY
        assert "chr12a" in QAPLIB_REGISTRY
        assert "tai20a" in QAPLIB_REGISTRY
        assert "sko100a" in QAPLIB_REGISTRY
        assert "tai256c" in QAPLIB_REGISTRY

    def test_instance_properties(self):
        """Test QAPLIBInstance properties"""
        instance = QAPLIB_REGISTRY["nug12"]
        assert instance.name == "nug12"
        assert instance.size == 12
        assert instance.optimal_value == 578
        assert instance.is_optimal_known()
        assert instance.get_value() == 578

    def test_get_instance_by_size(self):
        """Test filtering instances by size"""
        small = get_instance_by_size(0, 20)
        assert all(QAPLIB_REGISTRY[name].size <= 20 for name in small)

        medium = get_instance_by_size(30, 50)
        assert all(30 <= QAPLIB_REGISTRY[name].size <= 50 for name in medium)

    def test_get_instance_by_class(self):
        """Test filtering instances by problem class"""
        real_world = get_instance_by_class("real-world")
        assert all(QAPLIB_REGISTRY[name].problem_class == "real-world"
                  for name in real_world)

        grid = get_instance_by_class("grid")
        assert all(QAPLIB_REGISTRY[name].problem_class == "grid"
                  for name in grid)

    def test_get_small_instances(self):
        """Test getting small instances"""
        small = get_small_instances()
        assert all(QAPLIB_REGISTRY[name].size <= 20 for name in small)
        assert len(small) > 0

    def test_get_all_instance_names(self):
        """Test getting all instance names"""
        names = get_all_instance_names()
        assert len(names) == 138
        assert "nug12" in names
        assert "tai256c" in names

    def test_get_problem_classes(self):
        """Test getting problem classes"""
        classes = get_problem_classes()
        assert "real-world" in classes
        assert "random" in classes
        assert "grid" in classes
        assert "Hadamard" in classes

    def test_instance_metadata(self):
        """Test getting instance metadata"""
        meta = get_qaplib_metadata("chr12a")
        assert meta is not None
        assert meta["name"] == "chr12a"
        assert meta["size"] == 12
        assert meta["optimal_value"] == 9552
        assert meta["problem_class"] == "grid"


class TestEmbeddedData:
    """Test embedded instances"""

    def test_embedded_instances_count(self):
        """Test that we have enough embedded instances"""
        assert len(EMBEDDED_INSTANCES) >= 20

    def test_embedded_instance_structure(self):
        """Test structure of embedded instances"""
        for name, data in EMBEDDED_INSTANCES.items():
            assert "flow_matrix" in data
            assert "distance_matrix" in data

            flow = data["flow_matrix"]
            dist = data["distance_matrix"]

            # Check that matrices are square and same size
            assert len(flow) == len(flow[0])
            assert len(dist) == len(dist[0])
            assert len(flow) == len(dist)

            # Check size matches registry
            if name in QAPLIB_REGISTRY:
                assert len(flow) == QAPLIB_REGISTRY[name].size

    def test_embedded_nug12(self):
        """Test specific embedded instance nug12"""
        data = EMBEDDED_INSTANCES["nug12"]
        flow = np.array(data["flow_matrix"])
        dist = np.array(data["distance_matrix"])

        assert flow.shape == (12, 12)
        assert dist.shape == (12, 12)

        # Test known optimal solution
        # Identity permutation for testing
        perm = np.arange(12)
        obj = 0
        for i in range(12):
            for j in range(12):
                obj += flow[i, j] * dist[perm[i], perm[j]]

        assert obj > 0  # Should have non-zero objective


class TestLoader:
    """Test QAPLIB loader"""

    def test_loader_creation(self):
        """Test creating loader"""
        loader = QAPLIBLoader()
        assert loader is not None
        assert loader.cache_dir.exists()

    def test_load_embedded_instance(self):
        """Test loading embedded instance"""
        data = load_qaplib_instance("nug12")
        assert "flow_matrix" in data
        assert "distance_matrix" in data
        assert data["flow_matrix"].shape == (12, 12)
        assert data["distance_matrix"].shape == (12, 12)

    def test_load_multiple_embedded(self):
        """Test loading multiple embedded instances"""
        for name in ["chr12a", "chr15a", "had12", "esc16a"]:
            data = load_qaplib_instance(name)
            size = QAPLIB_REGISTRY[name].size
            assert data["flow_matrix"].shape == (size, size)
            assert data["distance_matrix"].shape == (size, size)

    def test_parse_dat_format(self):
        """Test parsing .dat format"""
        loader = QAPLIBLoader()

        # Create test .dat content
        content = """4

        0 5 2 4
        5 0 3 0
        2 3 0 0
        4 0 0 0

        0 3 8 15
        3 0 13 8
        8 13 0 15
        15 8 15 0
        """

        data = loader._parse_dat_format(content)
        assert data["flow_matrix"].shape == (4, 4)
        assert data["distance_matrix"].shape == (4, 4)
        assert data["flow_matrix"][0, 1] == 5
        assert data["distance_matrix"][0, 3] == 15

    def test_load_from_file(self):
        """Test loading from file"""
        loader = QAPLIBLoader()

        # Create temporary .dat file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("3\n")
            f.write("0 1 2\n")
            f.write("1 0 3\n")
            f.write("2 3 0\n")
            f.write("0 4 5\n")
            f.write("4 0 6\n")
            f.write("5 6 0\n")
            temp_path = f.name

        try:
            data = loader.load_from_file(temp_path)
            assert data["flow_matrix"].shape == (3, 3)
            assert data["distance_matrix"].shape == (3, 3)
            assert data["flow_matrix"][0, 2] == 2
            assert data["distance_matrix"][1, 2] == 6
        finally:
            os.unlink(temp_path)

    def test_verify_instance(self):
        """Test instance verification"""
        loader = QAPLIBLoader()

        # Verify nug12
        results = loader.verify_instance("nug12")
        assert results["valid"]
        assert results["size_match"]
        assert results["square_matrices"]
        assert results["non_negative"]
        assert results["finite_values"]

        # Verify with solution
        solution = np.arange(12)  # Identity permutation
        results = loader.verify_instance("nug12", solution)
        assert results["solution_valid"]
        assert "objective_value" in results

    def test_compute_objective(self):
        """Test objective computation"""
        loader = QAPLIBLoader()
        flow = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        dist = np.array([[0, 4, 5], [4, 0, 6], [5, 6, 0]])
        perm = np.array([0, 1, 2])  # Identity

        obj = loader._compute_objective(flow, dist, perm)
        # obj = sum(flow[i,j] * dist[perm[i], perm[j]])
        #     = 1*4 + 2*5 + 1*4 + 3*6 + 2*5 + 3*6 = 4+10+4+18+10+18 = 64
        assert obj == 64

    def test_list_instances(self):
        """Test listing instances with filters"""
        # List all
        all_instances = list_qaplib_instances()
        assert len(all_instances) == 138

        # Filter by size
        small = list_qaplib_instances(filter_by_size=(10, 20))
        assert all(10 <= QAPLIB_REGISTRY[name].size <= 20 for name in small)

        # Filter by class
        real_world = list_qaplib_instances(filter_by_class="real-world")
        assert all(QAPLIB_REGISTRY[name].problem_class == "real-world"
                  for name in real_world)


class TestBenchmarkRunner:
    """Test benchmark runner"""

    def test_benchmark_result(self):
        """Test BenchmarkResult dataclass"""
        from Librex.benchmarks.qaplib.benchmark_runner import BenchmarkResult

        result = BenchmarkResult(
            instance_name="nug12",
            method_name="test_method",
            objective_value=600,
            runtime_seconds=1.5,
            iterations=100,
            optimal_value=578
        )

        assert result.instance_name == "nug12"
        assert result.gap_percent == pytest.approx(100 * (600 - 578) / 578, 0.01)

    def test_benchmark_summary(self):
        """Test BenchmarkSummary dataclass"""
        from Librex.benchmarks.qaplib.benchmark_runner import BenchmarkResult, BenchmarkSummary

        results = [
            BenchmarkResult("nug12", "test", 578, 1.0, 100, optimal_value=578),
            BenchmarkResult("chr12a", "test", 10000, 2.0, 200, optimal_value=9552),
        ]

        summary = BenchmarkSummary(
            method_name="test_method",
            results=results,
            total_runtime=3.0
        )

        assert summary.method_name == "test_method"
        assert summary.solved_optimally == 1  # Only nug12 is optimal
        assert summary.avg_gap is not None

    def test_qaplib_benchmark(self):
        """Test QAPLIBBenchmark class"""
        benchmark = QAPLIBBenchmark(verbose=False)

        # Define simple test method
        def random_method(instance_data):
            n = len(instance_data["flow_matrix"])
            return {
                "solution": np.random.permutation(n),
                "objective": 1000,
                "iterations": 10
            }

        # Run on single instance
        result = benchmark.run_instance("nug12", random_method, "random")
        assert result.instance_name == "nug12"
        assert result.method_name == "random"
        assert result.objective_value == 1000

    def test_run_multiple(self):
        """Test running on multiple instances"""
        benchmark = QAPLIBBenchmark(verbose=False)

        def dummy_method(instance_data):
            n = len(instance_data["flow_matrix"])
            return {
                "solution": np.arange(n),
                "objective": 1000,
                "iterations": 1
            }

        instances = ["nug12", "chr12a"]
        summary = benchmark.run_multiple(instances, dummy_method, "dummy")

        assert len(summary.results) == 2
        assert summary.method_name == "dummy"

    def test_compare_methods(self):
        """Test comparing multiple methods"""
        benchmark = QAPLIBBenchmark(verbose=False)

        def method1(data):
            n = len(data["flow_matrix"])
            return {"solution": np.arange(n), "objective": 1000, "iterations": 1}

        def method2(data):
            n = len(data["flow_matrix"])
            return {"solution": np.arange(n), "objective": 2000, "iterations": 2}

        methods = {"method1": method1, "method2": method2}
        instances = ["nug12"]

        summaries = benchmark.compare_methods(methods, instances)
        assert "method1" in summaries
        assert "method2" in summaries
        assert summaries["method1"].results[0].objective_value == 1000
        assert summaries["method2"].results[0].objective_value == 2000

    def test_run_qaplib_benchmark(self):
        """Test convenience function"""

        def simple_method(data):
            n = len(data["flow_matrix"])
            return {
                "solution": np.arange(n),
                "objective": 1000,
                "iterations": 10
            }

        # Test with string instance specification
        summary = run_qaplib_benchmark(
            method=simple_method,
            instances=["nug12"],
            method_name="simple",
            verbose=False
        )

        assert summary.method_name == "simple"
        assert len(summary.results) == 1

    def test_objective_computation_in_benchmark(self):
        """Test that benchmark correctly computes objectives"""
        benchmark = QAPLIBBenchmark(verbose=False)

        def identity_method(data):
            n = len(data["flow_matrix"])
            return np.arange(n)  # Return just the solution

        result = benchmark.run_instance("nug12", identity_method, "identity")

        # Should compute objective from solution
        assert result.objective_value > 0
        assert result.solution is not None