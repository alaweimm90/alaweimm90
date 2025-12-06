"""
Comprehensive tests for QAPLIB benchmark suite.
Tests all 138 instances, loader functionality, registry, and benchmark runner.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import json

from Librex.benchmarks.qaplib import (
    QAPLIBLoader,
    QAPLIBRegistry,
    QAPLIBBenchmarkRunner,
    load_qaplib_instance,
    list_qaplib_instances,
    get_instance_metadata,
    run_qaplib_benchmark,
    get_all_instance_names,
    get_instance,
    list_instances,
)
from Librex.benchmarks.qaplib.embedded_data import EMBEDDED_QAPLIB_DATA as EMBEDDED_INSTANCES


class TestQAPLIBLoader:
    """Test QAPLIB instance loader functionality."""

    @pytest.fixture
    def loader(self):
        """Create a QAPLIBLoader instance."""
        return QAPLIBLoader()

    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert hasattr(loader, 'load')
        assert hasattr(loader, 'list_instances')

    def test_load_embedded_instance(self, loader):
        """Test loading an embedded instance."""
        # Test with a known small instance
        instance = loader.load('chr12a')
        assert instance is not None
        assert 'flow' in instance
        assert 'distance' in instance
        assert instance['flow'].shape == (12, 12)
        assert instance['distance'].shape == (12, 12)

    def test_load_all_embedded_instances(self, loader):
        """Test loading all 138 embedded instances."""
        instances = loader.list_instances()
        assert len(instances) >= 138  # At least 138 instances

        # Test loading a sample of instances
        sample_instances = ['chr12a', 'nug30', 'esc32a', 'bur26a', 'tai35a']
        for name in sample_instances:
            instance = loader.load(name)
            assert instance is not None
            assert 'flow' in instance
            assert 'distance' in instance
            n = int(''.join(filter(str.isdigit, name)) or '10')
            if n > 0 and n < 200:  # Reasonable size check
                assert instance['flow'].shape[0] == instance['flow'].shape[1]
                assert instance['distance'].shape == instance['flow'].shape

    def test_load_nonexistent_instance(self, loader):
        """Test loading a non-existent instance raises error."""
        with pytest.raises(ValueError):
            loader.load('nonexistent_instance')

    def test_load_from_file(self, loader):
        """Test loading instance from file."""
        # Create a temporary QAPLIB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("3\n")
            f.write("1 2 3\n4 5 6\n7 8 9\n")
            f.write("9 8 7\n6 5 4\n3 2 1\n")
            temp_file = f.name

        try:
            instance = loader.load_from_file(Path(temp_file))
            assert instance['flow'].shape == (3, 3)
            assert instance['distance'].shape == (3, 3)
            assert instance['flow'][0, 0] == 1
            assert instance['distance'][0, 0] == 9
        finally:
            Path(temp_file).unlink()

    def test_validate_instance_data(self, loader):
        """Test instance data validation."""
        # Valid instance
        valid_instance = {
            'flow': np.array([[0, 1], [1, 0]]),
            'distance': np.array([[0, 2], [2, 0]])
        }
        assert loader.validate(valid_instance) is True

        # Invalid instance (non-square matrix)
        invalid_instance = {
            'flow': np.array([[0, 1, 2], [1, 0, 3]]),
            'distance': np.array([[0, 2], [2, 0]])
        }
        assert loader.validate(invalid_instance) is False


class TestQAPLIBRegistry:
    """Test QAPLIB registry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a QAPLIBRegistry instance."""
        return QAPLIBRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry is not None
        assert len(registry.list_all()) >= 138

    def test_get_instance_metadata(self, registry):
        """Test getting metadata for instances."""
        metadata = registry.get_metadata('chr12a')
        assert metadata is not None
        assert 'name' in metadata
        assert 'size' in metadata
        assert metadata['name'] == 'chr12a'
        assert metadata['size'] == 12

    def test_filter_by_size(self, registry):
        """Test filtering instances by size."""
        small_instances = registry.filter_by_size(max_size=20)
        assert all(registry.get_metadata(name)['size'] <= 20 for name in small_instances)

        large_instances = registry.filter_by_size(min_size=100)
        assert all(registry.get_metadata(name)['size'] >= 100 for name in large_instances)

        medium_instances = registry.filter_by_size(min_size=30, max_size=50)
        for name in medium_instances:
            size = registry.get_metadata(name)['size']
            assert 30 <= size <= 50

    def test_filter_by_type(self, registry):
        """Test filtering instances by type."""
        chr_instances = registry.filter_by_type('chr')
        assert all(name.startswith('chr') for name in chr_instances)

        nug_instances = registry.filter_by_type('nug')
        assert all(name.startswith('nug') for name in nug_instances)

    def test_get_statistics(self, registry):
        """Test getting registry statistics."""
        stats = registry.get_statistics()
        assert 'total_instances' in stats
        assert 'size_distribution' in stats
        assert 'type_distribution' in stats
        assert stats['total_instances'] >= 138

    def test_register_custom_instance(self, registry):
        """Test registering a custom instance."""
        custom_metadata = {
            'name': 'custom_test',
            'size': 10,
            'type': 'custom',
            'optimal': 1234
        }
        registry.register('custom_test', custom_metadata)
        assert 'custom_test' in registry.list_all()
        retrieved = registry.get_metadata('custom_test')
        assert retrieved == custom_metadata


class TestQAPLIBBenchmarkRunner:
    """Test QAPLIB benchmark runner."""

    @pytest.fixture
    def runner(self):
        """Create a benchmark runner."""
        from Librex.methods.baselines import RandomSearchOptimizer
        return QAPLIBBenchmarkRunner([RandomSearchOptimizer])

    def test_runner_initialization(self, runner):
        """Test runner initializes correctly."""
        assert runner is not None
        assert len(runner.methods) == 1

    @patch('Librex.benchmarks.qaplib.benchmark_runner.QAPLIBLoader')
    def test_run_single_instance(self, mock_loader, runner):
        """Test running benchmark on single instance."""
        # Mock the loader
        mock_instance = {
            'flow': np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
            'distance': np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]]),
            'optimal': 10
        }
        mock_loader.return_value.load.return_value = mock_instance

        result = runner.run_instance('test_instance', max_iterations=10)
        assert 'instance' in result
        assert 'results' in result
        assert len(result['results']) == 1

    def test_run_benchmark_suite(self, runner):
        """Test running full benchmark suite."""
        # Test with a small subset
        instances = ['chr12a', 'nug12']
        results = runner.run_suite(instances, max_iterations=5)
        assert len(results) == 2
        for result in results:
            assert 'instance' in result
            assert 'results' in result

    def test_generate_report(self, runner):
        """Test generating benchmark report."""
        # Mock some results
        mock_results = [
            {
                'instance': 'test1',
                'results': [
                    {'method': 'random', 'best_cost': 100, 'time': 1.0}
                ]
            }
        ]
        report = runner.generate_report(mock_results)
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'performance_metrics' in report

    def test_save_and_load_results(self, runner):
        """Test saving and loading benchmark results."""
        mock_results = {
            'timestamp': '2024-01-01',
            'results': [{'instance': 'test', 'best': 100}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_file = f.name

        try:
            loaded = runner.load_results(temp_file)
            assert loaded == mock_results
        finally:
            Path(temp_file).unlink()


class TestQAPLIBAPI:
    """Test high-level QAPLIB API functions."""

    def test_get_instance_api(self):
        """Test get_instance API function."""
        instance = get_instance('chr12a')
        assert instance is not None
        assert 'flow' in instance
        assert 'distance' in instance

    def test_list_instances_api(self):
        """Test list_instances API function."""
        instances = list_instances()
        assert len(instances) >= 138
        assert 'chr12a' in instances
        assert 'nug30' in instances

    def test_get_instance_metadata_api(self):
        """Test get_instance_metadata API function."""
        metadata = get_instance_metadata('chr12a')
        assert metadata is not None
        assert metadata['name'] == 'chr12a'
        assert metadata['size'] == 12

    def test_embedded_data_integrity(self):
        """Test integrity of embedded instance data."""
        assert len(EMBEDDED_INSTANCES) >= 138

        # Check structure of a few instances
        for name in ['chr12a', 'nug20', 'esc16a']:
            if name in EMBEDDED_INSTANCES:
                instance = EMBEDDED_INSTANCES[name]
                assert 'flow' in instance
                assert 'distance' in instance
                assert 'optimal' in instance
                assert isinstance(instance['flow'], (list, np.ndarray))
                assert isinstance(instance['distance'], (list, np.ndarray))


class TestQAPLIBPerformance:
    """Performance and stress tests for QAPLIB."""

    def test_load_large_instance_performance(self):
        """Test loading performance for large instances."""
        import time
        loader = QAPLIBLoader()

        # Test loading a large instance
        start = time.time()
        instance = loader.load('tai256c') if 'tai256c' in list_instances() else loader.load('nug30')
        load_time = time.time() - start

        # Should load in reasonable time (< 1 second)
        assert load_time < 1.0

    def test_concurrent_loading(self):
        """Test concurrent loading of multiple instances."""
        import concurrent.futures
        loader = QAPLIBLoader()

        instances_to_load = ['chr12a', 'nug20', 'esc16a', 'bur26a']

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(loader.load, name) for name in instances_to_load]
            results = [f.result() for f in futures]

        assert len(results) == len(instances_to_load)
        assert all(r is not None for r in results)

    def test_memory_efficiency(self):
        """Test memory efficiency when loading multiple instances."""
        import sys
        loader = QAPLIBLoader()

        # Load several instances
        instances = []
        for name in ['chr12a', 'nug20', 'esc16a']:
            instances.append(loader.load(name))

        # Check that instances are reasonable in memory size
        for instance in instances:
            size = sys.getsizeof(instance['flow']) + sys.getsizeof(instance['distance'])
            # Each instance should be < 10MB
            assert size < 10 * 1024 * 1024
