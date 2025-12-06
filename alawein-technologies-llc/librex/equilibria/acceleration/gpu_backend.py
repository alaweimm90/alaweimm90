"""
GPU Backend Abstraction Layer

Provides a unified interface for multiple GPU computation frameworks.
Supports JAX, PyTorch, CuPy, and CPU fallback.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import numpy as np

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Supported computation backends"""
    JAX = "jax"
    PYTORCH = "pytorch"
    CUPY = "cupy"
    NUMPY = "numpy"  # CPU fallback
    AUTO = "auto"


@dataclass
class DeviceInfo:
    """Information about a compute device"""
    name: str
    type: str  # 'gpu', 'tpu', 'cpu'
    memory_mb: int
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True


class DeviceManager:
    """Manage and query available compute devices"""

    def __init__(self):
        self.devices = self._discover_devices()

    def _discover_devices(self) -> List[DeviceInfo]:
        """Discover all available compute devices"""
        devices = []

        # Check for CUDA GPUs
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_mb = memory_info.total // (1024 * 1024)

                devices.append(DeviceInfo(
                    name=name,
                    type='gpu',
                    memory_mb=memory_mb,
                    is_available=True
                ))
        except Exception:
            pass

        # Check for TPUs (JAX)
        try:
            import jax
            if len(jax.devices('tpu')) > 0:
                for device in jax.devices('tpu'):
                    devices.append(DeviceInfo(
                        name=str(device),
                        type='tpu',
                        memory_mb=16384,  # Standard TPU memory
                        is_available=True
                    ))
        except Exception:
            pass

        # Always add CPU
        import psutil
        cpu_mem_mb = psutil.virtual_memory().total // (1024 * 1024)
        devices.append(DeviceInfo(
            name='CPU',
            type='cpu',
            memory_mb=cpu_mem_mb,
            is_available=True
        ))

        return devices

    def get_best_device(self, prefer_type: str = 'gpu') -> DeviceInfo:
        """Get the best available device based on preferences"""
        # Sort devices by preference and memory
        sorted_devices = sorted(
            self.devices,
            key=lambda d: (
                d.type != prefer_type,  # Preferred type first
                -d.memory_mb  # Then by memory
            )
        )
        return sorted_devices[0]

    def get_devices_by_type(self, device_type: str) -> List[DeviceInfo]:
        """Get all devices of a specific type"""
        return [d for d in self.devices if d.type == device_type]


class GPUBackend:
    """
    Unified interface for GPU computation across multiple frameworks.

    Supports:
    - JAX (preferred for automatic differentiation and JIT compilation)
    - PyTorch (for deep learning integration)
    - CuPy (for NumPy-like GPU arrays)
    - NumPy (CPU fallback)
    """

    def __init__(self, backend: Union[str, Backend] = Backend.AUTO, device_id: int = 0):
        """
        Initialize GPU backend.

        Args:
            backend: Backend to use ('jax', 'pytorch', 'cupy', 'numpy', 'auto')
            device_id: GPU device ID to use (default 0)
        """
        self.device_manager = DeviceManager()

        if isinstance(backend, str):
            backend = Backend(backend.lower())

        self.backend = self._detect_backend(backend)
        self.device_id = device_id
        self._module = None
        self._device = None
        self._initialize_backend()

        logger.info(f"Initialized {self.backend.value} backend on device {device_id}")

    def _detect_backend(self, backend: Backend) -> Backend:
        """Auto-detect the best available GPU backend"""
        if backend != Backend.AUTO:
            if self._is_available(backend):
                return backend
            else:
                logger.warning(f"Requested backend {backend.value} not available, falling back to auto-detection")

        # Try backends in order of preference
        preference_order = [Backend.JAX, Backend.PYTORCH, Backend.CUPY, Backend.NUMPY]

        for b in preference_order:
            if self._is_available(b):
                logger.info(f"Auto-selected backend: {b.value}")
                return b

        # Default to NumPy (always available)
        return Backend.NUMPY

    def _is_available(self, backend: Backend) -> bool:
        """Check if a backend is available"""
        if backend == Backend.NUMPY:
            return True  # NumPy is always available

        try:
            if backend == Backend.JAX:
                import jax
                # Check for GPU/TPU availability
                devices = jax.devices()
                return any(d.platform != 'cpu' for d in devices)

            elif backend == Backend.PYTORCH:
                import torch
                return torch.cuda.is_available()

            elif backend == Backend.CUPY:
                import cupy
                # Try to create a small array to verify CUDA is working
                test_array = cupy.array([1.0])
                del test_array
                return True

        except ImportError:
            return False
        except Exception:
            return False

        return False

    def _initialize_backend(self):
        """Initialize the selected backend"""
        try:
            if self.backend == Backend.JAX:
                import jax
                import jax.numpy as jnp
                self._module = jax
                self._array_module = jnp

                # Set device
                devices = jax.devices()
                gpu_devices = [d for d in devices if d.platform != 'cpu']
                if gpu_devices and self.device_id < len(gpu_devices):
                    self._device = gpu_devices[self.device_id]
                else:
                    self._device = devices[0]

                # Configure JAX
                jax.config.update('jax_platform_name', self._device.platform)

            elif self.backend == Backend.PYTORCH:
                import torch
                self._module = torch
                self._device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')

            elif self.backend == Backend.CUPY:
                import cupy
                self._module = cupy
                self._array_module = cupy
                cupy.cuda.Device(self.device_id).use()

            else:  # NumPy fallback
                self._module = np
                self._array_module = np
                self._device = 'cpu'

        except Exception as e:
            logger.error(f"Failed to initialize {self.backend.value}: {e}")
            # Fall back to NumPy
            self.backend = Backend.NUMPY
            self._module = np
            self._array_module = np
            self._device = 'cpu'

    def to_device(self, array: np.ndarray, dtype: Optional[np.dtype] = None) -> Any:
        """
        Move array to GPU/device.

        Args:
            array: NumPy array to move to device
            dtype: Optional dtype to cast to

        Returns:
            Array on the target device
        """
        if dtype is not None:
            array = array.astype(dtype)

        if self.backend == Backend.JAX:
            import jax
            return jax.device_put(array, self._device)

        elif self.backend == Backend.PYTORCH:
            import torch
            return torch.tensor(array, device=self._device, dtype=self._get_torch_dtype(array.dtype))

        elif self.backend == Backend.CUPY:
            import cupy
            with cupy.cuda.Device(self.device_id):
                return cupy.asarray(array)

        else:  # NumPy
            return array

    def from_device(self, array: Any) -> np.ndarray:
        """
        Move array from GPU/device back to CPU NumPy array.

        Args:
            array: Device array

        Returns:
            NumPy array on CPU
        """
        if self.backend == Backend.JAX:
            return np.array(array)

        elif self.backend == Backend.PYTORCH:
            if hasattr(array, 'cpu'):
                return array.cpu().numpy()
            return array.numpy()

        elif self.backend == Backend.CUPY:
            import cupy
            return cupy.asnumpy(array)

        else:  # NumPy
            return np.asarray(array)

    def zeros(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Any:
        """Create a zero array on device"""
        if self.backend == Backend.JAX:
            import jax.numpy as jnp
            return jnp.zeros(shape, dtype=dtype)

        elif self.backend == Backend.PYTORCH:
            import torch
            return torch.zeros(shape, device=self._device, dtype=self._get_torch_dtype(dtype))

        elif self.backend == Backend.CUPY:
            import cupy
            with cupy.cuda.Device(self.device_id):
                return cupy.zeros(shape, dtype=dtype)

        else:
            return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Any:
        """Create a ones array on device"""
        if self.backend == Backend.JAX:
            import jax.numpy as jnp
            return jnp.ones(shape, dtype=dtype)

        elif self.backend == Backend.PYTORCH:
            import torch
            return torch.ones(shape, device=self._device, dtype=self._get_torch_dtype(dtype))

        elif self.backend == Backend.CUPY:
            import cupy
            with cupy.cuda.Device(self.device_id):
                return cupy.ones(shape, dtype=dtype)

        else:
            return np.ones(shape, dtype=dtype)

    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0,
                       seed: Optional[int] = None) -> Any:
        """Generate random uniform array on device"""
        if self.backend == Backend.JAX:
            import jax
            import jax.numpy as jnp
            key = jax.random.PRNGKey(seed if seed is not None else 0)
            return jax.random.uniform(key, shape, minval=low, maxval=high)

        elif self.backend == Backend.PYTORCH:
            import torch
            if seed is not None:
                torch.manual_seed(seed)
            return torch.rand(shape, device=self._device) * (high - low) + low

        elif self.backend == Backend.CUPY:
            import cupy
            if seed is not None:
                cupy.random.seed(seed)
            with cupy.cuda.Device(self.device_id):
                return cupy.random.uniform(low, high, shape)

        else:
            if seed is not None:
                np.random.seed(seed)
            return np.random.uniform(low, high, shape)

    def synchronize(self):
        """Synchronize device operations (wait for completion)"""
        if self.backend == Backend.JAX:
            # JAX operations are asynchronous
            import jax
            jax.block_until_ready

        elif self.backend == Backend.PYTORCH:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(self._device)

        elif self.backend == Backend.CUPY:
            import cupy
            cupy.cuda.Stream.null.synchronize()

    def clear_memory(self):
        """Clear GPU memory cache"""
        if self.backend == Backend.JAX:
            import jax
            # JAX doesn't have explicit memory clearing, but we can trigger GC
            import gc
            gc.collect()

        elif self.backend == Backend.PYTORCH:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif self.backend == Backend.CUPY:
            import cupy
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

    def get_memory_info(self) -> Dict[str, int]:
        """Get memory usage information"""
        info = {'total_mb': 0, 'used_mb': 0, 'free_mb': 0}

        if self.backend == Backend.PYTORCH:
            import torch
            if torch.cuda.is_available():
                info['total_mb'] = torch.cuda.get_device_properties(self.device_id).total_memory // (1024 * 1024)
                info['used_mb'] = torch.cuda.memory_allocated(self.device_id) // (1024 * 1024)
                info['free_mb'] = info['total_mb'] - info['used_mb']

        elif self.backend == Backend.CUPY:
            import cupy
            mempool = cupy.get_default_memory_pool()
            info['used_mb'] = mempool.used_bytes() // (1024 * 1024)
            info['total_mb'] = mempool.total_bytes() // (1024 * 1024)

        elif self.backend == Backend.JAX:
            # JAX doesn't provide direct memory querying
            # We can use nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info['total_mb'] = mem_info.total // (1024 * 1024)
                info['used_mb'] = mem_info.used // (1024 * 1024)
                info['free_mb'] = mem_info.free // (1024 * 1024)
            except Exception:
                pass

        return info

    def _get_torch_dtype(self, numpy_dtype: np.dtype):
        """Convert NumPy dtype to PyTorch dtype"""
        import torch

        dtype_map = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.int16: torch.int16,
            np.int8: torch.int8,
            np.uint8: torch.uint8,
            np.bool_: torch.bool,
        }

        return dtype_map.get(numpy_dtype.type, torch.float32)

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU acceleration"""
        return self.backend != Backend.NUMPY

    @property
    def backend_name(self) -> str:
        """Get backend name string"""
        return self.backend.value

    @property
    def array_module(self):
        """Get the array module for this backend"""
        if self.backend == Backend.JAX:
            import jax.numpy as jnp
            return jnp
        elif self.backend == Backend.PYTORCH:
            import torch
            return torch
        elif self.backend == Backend.CUPY:
            import cupy
            return cupy
        else:
            return np

    def __repr__(self) -> str:
        return f"GPUBackend(backend={self.backend.value}, device_id={self.device_id}, device={self._device})"