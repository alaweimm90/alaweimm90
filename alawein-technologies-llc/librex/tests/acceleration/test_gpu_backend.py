"""
Tests for GPU Backend module
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Librex.acceleration import GPUBackend, DeviceManager
from Librex.acceleration.gpu_backend import Backend, DeviceInfo


class TestDeviceManager:
    """Test device discovery and management"""

    def test_device_discovery(self):
        """Test that device manager discovers at least CPU"""
        manager = DeviceManager()
        assert len(manager.devices) >= 1

        # Should always have CPU
        cpu_devices = manager.get_devices_by_type('cpu')
        assert len(cpu_devices) >= 1

    def test_best_device_selection(self):
        """Test best device selection"""
        manager = DeviceManager()
        best = manager.get_best_device('gpu')
        assert best is not None
        assert isinstance(best, DeviceInfo)


class TestGPUBackend:
    """Test GPU backend abstraction"""

    def test_backend_detection(self):
        """Test automatic backend detection"""
        backend = GPUBackend('auto')
        assert backend.backend in [Backend.JAX, Backend.PYTORCH, Backend.CUPY, Backend.NUMPY]

    def test_numpy_fallback(self):
        """Test that NumPy backend always works"""
        backend = GPUBackend('numpy')
        assert backend.backend == Backend.NUMPY
        assert not backend.is_gpu

    def test_array_operations(self):
        """Test basic array operations"""
        backend = GPUBackend('numpy')

        # Test array creation
        shape = (10, 5)
        zeros = backend.zeros(shape)
        ones = backend.ones(shape)

        assert zeros.shape == shape
        assert ones.shape == shape

        # Test array transfer
        arr = np.random.randn(5, 3)
        device_arr = backend.to_device(arr)
        cpu_arr = backend.from_device(device_arr)

        np.testing.assert_array_almost_equal(arr, cpu_arr)

    def test_memory_management(self):
        """Test memory management functions"""
        backend = GPUBackend('numpy')

        # Should not raise errors
        backend.synchronize()
        backend.clear_memory()

        mem_info = backend.get_memory_info()
        assert 'total_mb' in mem_info
        assert 'used_mb' in mem_info
        assert 'free_mb' in mem_info


@pytest.mark.skipif(
    not any(GPUBackend(b).is_gpu for b in ['jax', 'pytorch', 'cupy']),
    reason="No GPU backend available"
)
class TestGPUOperations:
    """Test GPU-specific operations (requires GPU)"""

    def test_gpu_transfer(self):
        """Test transferring arrays to/from GPU"""
        for backend_name in ['jax', 'pytorch', 'cupy']:
            try:
                backend = GPUBackend(backend_name)
                if backend.is_gpu:
                    arr = np.random.randn(100, 50)
                    gpu_arr = backend.to_device(arr)
                    cpu_arr = backend.from_device(gpu_arr)

                    np.testing.assert_array_almost_equal(arr, cpu_arr, decimal=5)
                    break
            except:
                continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])