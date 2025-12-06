"""
GPU acceleration support for SciComp.
This module provides CUDA/GPU acceleration for computational physics using CuPy.
Includes optimized kernels for quantum simulations, matrix operations, and FFTs.
Author: UC Berkeley SciComp Team
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import warnings
from typing import Union, Optional, Tuple, List
import os
# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupy import cuda
    from cupyx.scipy import linalg as cp_linalg
    from cupyx.scipy import fft as cp_fft
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    warnings.warn("CuPy not installed. GPU acceleration unavailable.")
class GPUAccelerator:
    """GPU acceleration manager for scientific computing."""
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU accelerator.
        Args:
            device_id: CUDA device ID to use
        """
        self.gpu_available = GPU_AVAILABLE
        self.device_id = device_id
        self.memory_pool = None
        if self.gpu_available:
            try:
                cuda.Device(device_id).use()
                self.device = cp.cuda.Device(device_id)
                self.memory_pool = cp.get_default_memory_pool()
                # Get device properties
                self.device_name = self.device.name
                self.compute_capability = self.device.compute_capability
                self.memory_size = self.device.mem_info[1] / 1e9  # GB
                print(f"🚀 GPU Acceleration Enabled")
                print(f"   Device: {self.device_name}")
                print(f"   Compute Capability: {self.compute_capability}")
                print(f"   Memory: {self.memory_size:.1f} GB")
            except Exception as e:
                self.gpu_available = False
                warnings.warn(f"GPU initialization failed: {e}")
        else:
            print("⚠️  GPU not available, using CPU fallback")
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU memory."""
        if self.gpu_available and cp is not None:
            return cp.asarray(array)
        return array
    def to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer array from GPU to CPU memory."""
        if self.gpu_available and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    def synchronize(self):
        """Synchronize GPU operations."""
        if self.gpu_available:
            cp.cuda.Stream.null.synchronize()
    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.gpu_available and self.memory_pool:
            self.memory_pool.free_all_blocks()
class QuantumGPU:
    """GPU-accelerated quantum computing operations."""
    def __init__(self, accelerator: GPUAccelerator):
        """
        Initialize quantum GPU operations.
        Args:
            accelerator: GPU accelerator instance
        """
        self.acc = accelerator
        self.xp = cp if self.acc.gpu_available else np
    def tensor_product_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated tensor product (Kronecker product).
        Args:
            A, B: Input matrices
        Returns:
            Tensor product A ⊗ B
        """
        A_gpu = self.acc.to_gpu(A)
        B_gpu = self.acc.to_gpu(B)
        result = self.xp.kron(A_gpu, B_gpu)
        return self.acc.to_cpu(result)
    def evolve_state_gpu(self, state: np.ndarray, hamiltonian: np.ndarray,
                        time: float, steps: int = 100) -> np.ndarray:
        """
        GPU-accelerated quantum state time evolution.
        Uses matrix exponentiation: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        Args:
            state: Initial quantum state
            hamiltonian: System Hamiltonian
            time: Evolution time
            steps: Number of time steps
        Returns:
            Time-evolved state
        """
        state_gpu = self.acc.to_gpu(state)
        H_gpu = self.acc.to_gpu(hamiltonian)
        dt = time / steps
        U_gpu = self.xp.eye(len(state), dtype=complex)
        if self.acc.gpu_available:
            # Use GPU-accelerated matrix exponential
            U_step = cp_linalg.expm(-1j * H_gpu * dt)
            for _ in range(steps):
                state_gpu = U_step @ state_gpu
        else:
            # CPU fallback
            from scipy.linalg import expm
            U_step = expm(-1j * hamiltonian * dt)
            for _ in range(steps):
                state_gpu = U_step @ state_gpu
        return self.acc.to_cpu(state_gpu)
    def quantum_fft_gpu(self, state: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated Quantum Fourier Transform.
        Args:
            state: Input quantum state
        Returns:
            QFT of input state
        """
        state_gpu = self.acc.to_gpu(state)
        if self.acc.gpu_available:
            # Use CuPy's FFT
            result = cp_fft.fft(state_gpu) / self.xp.sqrt(len(state_gpu))
        else:
            # CPU fallback
            result = np.fft.fft(state_gpu) / np.sqrt(len(state_gpu))
        return self.acc.to_cpu(result)
    def entanglement_entropy_gpu(self, state: np.ndarray, partition: int) -> float:
        """
        GPU-accelerated entanglement entropy calculation.
        Args:
            state: Quantum state vector
            partition: Partition size for partial trace
        Returns:
            Von Neumann entropy
        """
        state_gpu = self.acc.to_gpu(state)
        n = len(state_gpu)
        dim_a = partition
        dim_b = n // dim_a
        # Reshape state into matrix form
        psi_matrix = state_gpu.reshape((dim_a, dim_b))
        # Compute reduced density matrix
        if self.acc.gpu_available:
            rho = psi_matrix @ psi_matrix.conj().T
            eigenvalues = cp_linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -self.xp.sum(eigenvalues * self.xp.log2(eigenvalues))
        else:
            rho = psi_matrix @ psi_matrix.conj().T
            eigenvalues = np.linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return float(self.acc.to_cpu(entropy))
class PhysicsGPU:
    """GPU-accelerated physics simulations."""
    def __init__(self, accelerator: GPUAccelerator):
        """
        Initialize physics GPU operations.
        Args:
            accelerator: GPU accelerator instance
        """
        self.acc = accelerator
        self.xp = cp if self.acc.gpu_available else np
    def solve_heat_equation_gpu(self, initial: np.ndarray,
                               alpha: float, dx: float, dt: float,
                               steps: int) -> np.ndarray:
        """
        GPU-accelerated 1D heat equation solver using finite differences.
        Args:
            initial: Initial temperature distribution
            alpha: Thermal diffusivity
            dx: Spatial step size
            dt: Time step size
            steps: Number of time steps
        Returns:
            Temperature field evolution
        """
        u = self.acc.to_gpu(initial)
        n = len(u)
        r = alpha * dt / (dx ** 2)
        # Stability check
        if r > 0.5:
            warnings.warn(f"Unstable parameters: r={r:.3f} > 0.5")
        result = self.xp.zeros((steps + 1, n), dtype=float)
        result[0] = u
        for t in range(steps):
            u_new = self.xp.zeros_like(u)
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            u_new[0] = u[0]  # Boundary condition
            u_new[-1] = u[-1]  # Boundary condition
            u = u_new
            result[t + 1] = u
        return self.acc.to_cpu(result)
    def solve_wave_equation_gpu(self, initial_u: np.ndarray, initial_v: np.ndarray,
                               c: float, dx: float, dt: float,
                               steps: int) -> np.ndarray:
        """
        GPU-accelerated 1D wave equation solver.
        Args:
            initial_u: Initial displacement
            initial_v: Initial velocity
            c: Wave speed
            dx: Spatial step size
            dt: Time step size
            steps: Number of time steps
        Returns:
            Wave field evolution
        """
        u_prev = self.acc.to_gpu(initial_u)
        u_curr = u_prev + dt * self.acc.to_gpu(initial_v)
        n = len(u_prev)
        r = (c * dt / dx) ** 2
        # CFL condition check
        if r > 1:
            warnings.warn(f"CFL condition violated: r={r:.3f} > 1")
        result = self.xp.zeros((steps + 1, n), dtype=float)
        result[0] = u_prev
        result[1] = u_curr
        for t in range(2, steps + 1):
            u_next = self.xp.zeros_like(u_curr)
            u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] +
                           r * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
            u_next[0] = 0  # Fixed boundary
            u_next[-1] = 0  # Fixed boundary
            u_prev = u_curr
            u_curr = u_next
            result[t] = u_curr
        return self.acc.to_cpu(result)
    def fft_convolution_gpu(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated convolution using FFT.
        Args:
            signal: Input signal
            kernel: Convolution kernel
        Returns:
            Convolved signal
        """
        signal_gpu = self.acc.to_gpu(signal)
        kernel_gpu = self.acc.to_gpu(kernel)
        # Pad kernel to signal length
        if len(kernel_gpu) < len(signal_gpu):
            kernel_padded = self.xp.zeros_like(signal_gpu)
            kernel_padded[:len(kernel_gpu)] = kernel_gpu
            kernel_gpu = kernel_padded
        if self.acc.gpu_available:
            # GPU FFT convolution
            signal_fft = cp_fft.fft(signal_gpu)
            kernel_fft = cp_fft.fft(kernel_gpu)
            result = cp_fft.ifft(signal_fft * kernel_fft).real
        else:
            # CPU fallback
            signal_fft = np.fft.fft(signal_gpu)
            kernel_fft = np.fft.fft(kernel_gpu)
            result = np.fft.ifft(signal_fft * kernel_fft).real
        return self.acc.to_cpu(result)
class MatrixGPU:
    """GPU-accelerated matrix operations."""
    def __init__(self, accelerator: GPUAccelerator):
        """
        Initialize matrix GPU operations.
        Args:
            accelerator: GPU accelerator instance
        """
        self.acc = accelerator
        self.xp = cp if self.acc.gpu_available else np
    def large_matrix_multiply_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated large matrix multiplication.
        Args:
            A, B: Input matrices
        Returns:
            Matrix product A @ B
        """
        A_gpu = self.acc.to_gpu(A)
        B_gpu = self.acc.to_gpu(B)
        result = A_gpu @ B_gpu
        return self.acc.to_cpu(result)
    def eigendecomposition_gpu(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated eigenvalue decomposition.
        Args:
            matrix: Input matrix
        Returns:
            Eigenvalues and eigenvectors
        """
        matrix_gpu = self.acc.to_gpu(matrix)
        if self.acc.gpu_available:
            eigenvalues, eigenvectors = cp_linalg.eigh(matrix_gpu)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix_gpu)
        return self.acc.to_cpu(eigenvalues), self.acc.to_cpu(eigenvectors)
    def svd_gpu(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated singular value decomposition.
        Args:
            matrix: Input matrix
        Returns:
            U, S, Vt matrices from SVD
        """
        matrix_gpu = self.acc.to_gpu(matrix)
        if self.acc.gpu_available:
            U, S, Vt = cp_linalg.svd(matrix_gpu)
        else:
            U, S, Vt = np.linalg.svd(matrix_gpu)
        return self.acc.to_cpu(U), self.acc.to_cpu(S), self.acc.to_cpu(Vt)
# Convenience functions
def gpu_accelerate_computation(func):
    """Decorator to automatically GPU-accelerate a function."""
    def wrapper(*args, **kwargs):
        acc = GPUAccelerator()
        # Convert numpy arrays to GPU
        gpu_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                gpu_args.append(acc.to_gpu(arg))
            else:
                gpu_args.append(arg)
        gpu_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                gpu_kwargs[key] = acc.to_gpu(val)
            else:
                gpu_kwargs[key] = val
        # Execute function
        result = func(*gpu_args, **gpu_kwargs)
        # Convert result back to CPU
        if isinstance(result, (cp.ndarray if cp else np.ndarray)):
            result = acc.to_cpu(result)
        elif isinstance(result, (list, tuple)):
            result = type(result)(acc.to_cpu(r) if isinstance(r, (cp.ndarray if cp else np.ndarray)) else r for r in result)
        acc.clear_memory()
        return result
    return wrapper
def benchmark_gpu_vs_cpu(func_gpu, func_cpu, test_data, n_runs=10):
    """
    Benchmark GPU vs CPU performance.
    Args:
        func_gpu: GPU-accelerated function
        func_cpu: CPU function
        test_data: Test data for benchmarking
        n_runs: Number of benchmark runs
    Returns:
        Performance comparison dict
    """
    import time
    # CPU benchmark
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func_cpu(*test_data)
        cpu_times.append(time.perf_counter() - start)
    # GPU benchmark (if available)
    if GPU_AVAILABLE:
        gpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = func_gpu(*test_data)
            gpu_times.append(time.perf_counter() - start)
        speedup = np.mean(cpu_times) / np.mean(gpu_times)
    else:
        gpu_times = [0]
        speedup = 1.0
    return {
        'cpu_time': np.mean(cpu_times),
        'gpu_time': np.mean(gpu_times) if GPU_AVAILABLE else None,
        'speedup': speedup,
        'gpu_available': GPU_AVAILABLE
    }