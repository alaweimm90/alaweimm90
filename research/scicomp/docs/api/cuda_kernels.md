# cuda_kernels
**Module:** `Python/gpu_acceleration/cuda_kernels.py`
## Overview
GPU acceleration support for SciComp.
This module provides CUDA/GPU acceleration for computational physics using CuPy.
Includes optimized kernels for quantum simulations, matrix operations, and FFTs.
Author: UC Berkeley SciComp Team
Copyright © 2025 Meshal Alawein — All rights reserved.
## Functions
### `gpu_accelerate_computation(func)`
Decorator to automatically GPU-accelerate a function.
**Source:** [Line 407](Python/gpu_acceleration/cuda_kernels.py#L407)
### `benchmark_gpu_vs_cpu(func_gpu, func_cpu, test_data, n_runs)`
Benchmark GPU vs CPU performance.
Args:
func_gpu: GPU-accelerated function
func_cpu: CPU function
test_data: Test data for benchmarking
n_runs: Number of benchmark runs
Returns:
Performance comparison dict
**Source:** [Line 442](Python/gpu_acceleration/cuda_kernels.py#L442)
## Classes
### `GPUAccelerator`
GPU acceleration manager for scientific computing.
#### Methods
##### `__init__(self, device_id)`
Initialize GPU accelerator.
Args:
device_id: CUDA device ID to use
**Source:** [Line 32](Python/gpu_acceleration/cuda_kernels.py#L32)
##### `to_gpu(self, array)`
Transfer array to GPU memory.
**Source:** [Line 65](Python/gpu_acceleration/cuda_kernels.py#L65)
##### `to_cpu(self, array)`
Transfer array from GPU to CPU memory.
**Source:** [Line 71](Python/gpu_acceleration/cuda_kernels.py#L71)
##### `synchronize(self)`
Synchronize GPU operations.
**Source:** [Line 77](Python/gpu_acceleration/cuda_kernels.py#L77)
##### `clear_memory(self)`
Clear GPU memory cache.
**Source:** [Line 82](Python/gpu_acceleration/cuda_kernels.py#L82)
**Class Source:** [Line 29](Python/gpu_acceleration/cuda_kernels.py#L29)
### `QuantumGPU`
GPU-accelerated quantum computing operations.
#### Methods
##### `__init__(self, accelerator)`
Initialize quantum GPU operations.
Args:
accelerator: GPU accelerator instance
**Source:** [Line 91](Python/gpu_acceleration/cuda_kernels.py#L91)
##### `tensor_product_gpu(self, A, B)`
GPU-accelerated tensor product (Kronecker product).
Args:
A, B: Input matrices
Returns:
Tensor product A ⊗ B
**Source:** [Line 101](Python/gpu_acceleration/cuda_kernels.py#L101)
##### `evolve_state_gpu(self, state, hamiltonian, time, steps)`
GPU-accelerated quantum state time evolution.
Uses matrix exponentiation: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
Args:
state: Initial quantum state
hamiltonian: System Hamiltonian
time: Evolution time
steps: Number of time steps
Returns:
Time-evolved state
**Source:** [Line 116](Python/gpu_acceleration/cuda_kernels.py#L116)
##### `quantum_fft_gpu(self, state)`
GPU-accelerated Quantum Fourier Transform.
Args:
state: Input quantum state
Returns:
QFT of input state
**Source:** [Line 154](Python/gpu_acceleration/cuda_kernels.py#L154)
##### `entanglement_entropy_gpu(self, state, partition)`
GPU-accelerated entanglement entropy calculation.
Args:
state: Quantum state vector
partition: Partition size for partial trace
Returns:
Von Neumann entropy
**Source:** [Line 175](Python/gpu_acceleration/cuda_kernels.py#L175)
**Class Source:** [Line 88](Python/gpu_acceleration/cuda_kernels.py#L88)
### `PhysicsGPU`
GPU-accelerated physics simulations.
#### Methods
##### `__init__(self, accelerator)`
Initialize physics GPU operations.
Args:
accelerator: GPU accelerator instance
**Source:** [Line 212](Python/gpu_acceleration/cuda_kernels.py#L212)
##### `solve_heat_equation_gpu(self, initial, alpha, dx, dt, steps)`
GPU-accelerated 1D heat equation solver using finite differences.
Args:
initial: Initial temperature distribution
alpha: Thermal diffusivity
dx: Spatial step size
dt: Time step size
steps: Number of time steps
Returns:
Temperature field evolution
**Source:** [Line 222](Python/gpu_acceleration/cuda_kernels.py#L222)
##### `solve_wave_equation_gpu(self, initial_u, initial_v, c, dx, dt, steps)`
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
**Source:** [Line 259](Python/gpu_acceleration/cuda_kernels.py#L259)
##### `fft_convolution_gpu(self, signal, kernel)`
GPU-accelerated convolution using FFT.
Args:
signal: Input signal
kernel: Convolution kernel
Returns:
Convolved signal
**Source:** [Line 303](Python/gpu_acceleration/cuda_kernels.py#L303)
**Class Source:** [Line 209](Python/gpu_acceleration/cuda_kernels.py#L209)
### `MatrixGPU`
GPU-accelerated matrix operations.
#### Methods
##### `__init__(self, accelerator)`
Initialize matrix GPU operations.
Args:
accelerator: GPU accelerator instance
**Source:** [Line 340](Python/gpu_acceleration/cuda_kernels.py#L340)
##### `large_matrix_multiply_gpu(self, A, B)`
GPU-accelerated large matrix multiplication.
Args:
A, B: Input matrices
Returns:
Matrix product A @ B
**Source:** [Line 350](Python/gpu_acceleration/cuda_kernels.py#L350)
##### `eigendecomposition_gpu(self, matrix)`
GPU-accelerated eigenvalue decomposition.
Args:
matrix: Input matrix
Returns:
Eigenvalues and eigenvectors
**Source:** [Line 367](Python/gpu_acceleration/cuda_kernels.py#L367)
##### `svd_gpu(self, matrix)`
GPU-accelerated singular value decomposition.
Args:
matrix: Input matrix
Returns:
U, S, Vt matrices from SVD
**Source:** [Line 386](Python/gpu_acceleration/cuda_kernels.py#L386)
**Class Source:** [Line 337](Python/gpu_acceleration/cuda_kernels.py#L337)