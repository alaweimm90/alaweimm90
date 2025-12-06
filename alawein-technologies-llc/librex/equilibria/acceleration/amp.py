"""
Automatic Mixed Precision Module

Implements mixed precision computation for faster GPU performance.
"""

import logging
import numpy as np
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    MIXED = "mixed"
    DYNAMIC = "dynamic"


class MixedPrecisionConfig:
    """Configuration for mixed precision"""

    def __init__(self,
                 compute_dtype: str = "float16",
                 accumulate_dtype: str = "float32",
                 enable_autocast: bool = True,
                 loss_scale: float = 128.0,
                 dynamic_loss_scaling: bool = True,
                 grad_scaling_factor: float = 2.0):
        """
        Initialize mixed precision config.

        Args:
            compute_dtype: Dtype for computation (float16, bfloat16)
            accumulate_dtype: Dtype for accumulation (float32)
            enable_autocast: Enable automatic casting
            loss_scale: Initial loss scale for gradient scaling
            dynamic_loss_scaling: Enable dynamic loss scaling
            grad_scaling_factor: Factor for gradient scaling updates
        """
        self.compute_dtype = compute_dtype
        self.accumulate_dtype = accumulate_dtype
        self.enable_autocast = enable_autocast
        self.loss_scale = loss_scale
        self.dynamic_loss_scaling = dynamic_loss_scaling
        self.grad_scaling_factor = grad_scaling_factor


_global_amp_config = MixedPrecisionConfig()
_amp_enabled = False


def enable_mixed_precision(backend: Optional[str] = None,
                          config: Optional[MixedPrecisionConfig] = None):
    """
    Enable mixed precision for faster GPU computation.

    Args:
        backend: Optional backend specification ('jax', 'pytorch', 'auto')
        config: Optional mixed precision configuration
    """
    global _amp_enabled, _global_amp_config

    if config:
        _global_amp_config = config

    # Auto-detect backend if not specified
    if backend is None or backend == 'auto':
        backend = _detect_backend()

    if backend == 'jax':
        _enable_jax_mixed_precision()
    elif backend == 'pytorch':
        _enable_pytorch_mixed_precision()
    elif backend == 'tensorflow':
        _enable_tensorflow_mixed_precision()
    else:
        logger.warning(f"Mixed precision not supported for backend: {backend}")
        return

    _amp_enabled = True
    logger.info(f"Mixed precision enabled for {backend} backend")


def disable_mixed_precision():
    """Disable mixed precision computation"""
    global _amp_enabled

    # Restore default precision for each backend
    try:
        _disable_jax_mixed_precision()
    except ImportError:
        pass

    try:
        _disable_pytorch_mixed_precision()
    except ImportError:
        pass

    try:
        _disable_tensorflow_mixed_precision()
    except ImportError:
        pass

    _amp_enabled = False
    logger.info("Mixed precision disabled")


def _detect_backend() -> str:
    """Auto-detect available backend"""
    try:
        import jax
        return 'jax'
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return 'pytorch'
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            return 'tensorflow'
    except ImportError:
        pass

    return 'none'


def _enable_jax_mixed_precision():
    """Enable mixed precision for JAX"""
    import jax

    # Set default dtype
    if _global_amp_config.compute_dtype == 'float16':
        jax.config.update('jax_default_dtype_bits', '16')
    elif _global_amp_config.compute_dtype == 'bfloat16':
        # Enable bfloat16 if available
        try:
            import jax.numpy as jnp
            jax.config.update('jax_default_matmul_precision', 'bfloat16')
        except Exception:
            jax.config.update('jax_default_dtype_bits', '16')

    # Disable x64 for memory efficiency
    jax.config.update('jax_enable_x64', False)

    logger.debug("JAX mixed precision enabled")


def _disable_jax_mixed_precision():
    """Disable mixed precision for JAX"""
    import jax

    # Restore default precision
    jax.config.update('jax_default_dtype_bits', '32')
    jax.config.update('jax_default_matmul_precision', 'float32')
    jax.config.update('jax_enable_x64', True)

    logger.debug("JAX mixed precision disabled")


def _enable_pytorch_mixed_precision():
    """Enable mixed precision for PyTorch"""
    import torch

    # Set default dtype
    if _global_amp_config.compute_dtype == 'float16':
        torch.set_default_dtype(torch.float16)
    elif _global_amp_config.compute_dtype == 'bfloat16':
        if torch.cuda.is_bf16_supported():
            torch.set_default_dtype(torch.bfloat16)
        else:
            torch.set_default_dtype(torch.float16)

    # Set matmul precision
    torch.set_float32_matmul_precision('medium')

    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.debug("PyTorch mixed precision enabled")


def _disable_pytorch_mixed_precision():
    """Disable mixed precision for PyTorch"""
    import torch

    # Restore default precision
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    logger.debug("PyTorch mixed precision disabled")


def _enable_tensorflow_mixed_precision():
    """Enable mixed precision for TensorFlow"""
    import tensorflow as tf

    # Set mixed precision policy
    if _global_amp_config.compute_dtype == 'float16':
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
    elif _global_amp_config.compute_dtype == 'bfloat16':
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.Policy('float32')

    tf.keras.mixed_precision.set_global_policy(policy)

    logger.debug(f"TensorFlow mixed precision enabled with policy: {policy.name}")


def _disable_tensorflow_mixed_precision():
    """Disable mixed precision for TensorFlow"""
    import tensorflow as tf

    # Restore float32 policy
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    logger.debug("TensorFlow mixed precision disabled")


def mixed_precision_context(backend: str = 'auto', config: Optional[MixedPrecisionConfig] = None):
    """
    Context manager for mixed precision computation.

    Usage:
        with mixed_precision_context():
            # Your optimization code here
            result = optimizer.optimize(problem)
    """
    class MixedPrecisionContext:
        def __init__(self, backend, config):
            self.backend = backend
            self.config = config
            self.was_enabled = _amp_enabled

        def __enter__(self):
            if not self.was_enabled:
                enable_mixed_precision(self.backend, self.config)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.was_enabled:
                disable_mixed_precision()

    return MixedPrecisionContext(backend, config)


def autocast_function(backend: str = 'auto'):
    """
    Decorator for automatic mixed precision casting.

    Usage:
        @autocast_function()
        def objective_function(x):
            # Computation automatically uses mixed precision
            return np.sum(x ** 2)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if backend == 'jax' or (backend == 'auto' and _detect_backend() == 'jax'):
                return _jax_autocast_wrapper(func, *args, **kwargs)
            elif backend == 'pytorch' or (backend == 'auto' and _detect_backend() == 'pytorch'):
                return _pytorch_autocast_wrapper(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _jax_autocast_wrapper(func: Callable, *args, **kwargs):
    """JAX autocast wrapper"""
    import jax.numpy as jnp

    # Convert inputs to lower precision
    def to_compute_dtype(x):
        if hasattr(x, 'dtype') and x.dtype == jnp.float32:
            if _global_amp_config.compute_dtype == 'float16':
                return x.astype(jnp.float16)
            elif _global_amp_config.compute_dtype == 'bfloat16':
                return x.astype(jnp.bfloat16)
        return x

    args = [to_compute_dtype(arg) if hasattr(arg, 'dtype') else arg for arg in args]

    # Run computation
    result = func(*args, **kwargs)

    # Convert output back to float32 for numerical stability
    if hasattr(result, 'dtype') and result.dtype in [jnp.float16, jnp.bfloat16]:
        result = result.astype(jnp.float32)

    return result


def _pytorch_autocast_wrapper(func: Callable, *args, **kwargs):
    """PyTorch autocast wrapper"""
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if _global_amp_config.compute_dtype == 'float16' else torch.bfloat16

    with torch.autocast(device_type=device, dtype=dtype):
        return func(*args, **kwargs)


class GradientScaler:
    """
    Gradient scaler for mixed precision training.

    Prevents gradient underflow in float16 computation.
    """

    def __init__(self, backend: str = 'auto', init_scale: float = 128.0):
        """
        Initialize gradient scaler.

        Args:
            backend: Computation backend
            init_scale: Initial loss scale
        """
        self.backend = backend if backend != 'auto' else _detect_backend()
        self.scale = init_scale
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 100
        self._growth_tracker = 0

    def scale_loss(self, loss: Any) -> Any:
        """Scale loss for backward pass"""
        if self.backend == 'pytorch':
            import torch
            if isinstance(loss, torch.Tensor):
                return loss * self.scale
        elif self.backend == 'jax':
            import jax.numpy as jnp
            if hasattr(loss, 'dtype'):
                return loss * self.scale

        return loss

    def unscale_gradients(self, gradients: Any) -> Any:
        """Unscale gradients after backward pass"""
        if self.backend == 'pytorch':
            import torch
            if isinstance(gradients, torch.Tensor):
                return gradients / self.scale
        elif self.backend == 'jax':
            import jax.numpy as jnp
            if hasattr(gradients, 'dtype'):
                return gradients / self.scale

        return gradients

    def update_scale(self, overflow: bool = False):
        """Update loss scale based on gradient overflow"""
        if overflow:
            # Gradient overflow detected, reduce scale
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            logger.debug(f"Gradient overflow detected, reducing scale to {self.scale}")
        else:
            # No overflow, potentially increase scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
                logger.debug(f"Increasing gradient scale to {self.scale}")


def benchmark_mixed_precision(objective_fn: Callable, test_size: int = 1000) -> Dict[str, float]:
    """
    Benchmark performance improvement from mixed precision.

    Args:
        objective_fn: Function to benchmark
        test_size: Size of test problem

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Generate test data
    test_data = np.random.randn(test_size, 10).astype(np.float32)

    # Benchmark without mixed precision
    disable_mixed_precision()
    start = time.time()
    for _ in range(100):
        _ = objective_fn(test_data)
    fp32_time = time.time() - start

    # Benchmark with mixed precision
    enable_mixed_precision()
    start = time.time()
    for _ in range(100):
        _ = objective_fn(test_data)
    amp_time = time.time() - start

    # Restore original state
    disable_mixed_precision()

    speedup = fp32_time / amp_time if amp_time > 0 else 1.0

    return {
        'fp32_time': fp32_time,
        'amp_time': amp_time,
        'speedup': speedup,
        'improvement_pct': (speedup - 1) * 100,
    }


# Utility functions for precision management
def get_optimal_dtype(backend: str = 'auto', prefer_bfloat16: bool = True) -> str:
    """
    Get optimal dtype for current hardware.

    Args:
        backend: Computation backend
        prefer_bfloat16: Prefer bfloat16 over float16 if available

    Returns:
        Optimal dtype string
    """
    if backend == 'auto':
        backend = _detect_backend()

    if backend == 'pytorch':
        import torch
        if torch.cuda.is_available():
            if prefer_bfloat16 and torch.cuda.is_bf16_supported():
                return 'bfloat16'
            else:
                return 'float16'

    elif backend == 'jax':
        # JAX supports both, prefer bfloat16 for better range
        if prefer_bfloat16:
            return 'bfloat16'
        else:
            return 'float16'

    return 'float32'


def is_mixed_precision_enabled() -> bool:
    """Check if mixed precision is currently enabled"""
    return _amp_enabled


def get_current_precision_config() -> MixedPrecisionConfig:
    """Get current mixed precision configuration"""
    return _global_amp_config