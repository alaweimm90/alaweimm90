#!/usr/bin/env python3
"""
SciComp - Performance Benchmarks
===================================================
Comprehensive performance benchmarking suite for the SciComp.
Tests computational efficiency across all major modules and provides detailed
performance metrics for optimization and deployment planning.
Author: UC Berkeley SciComp Team
Date: 2025
License: MIT
"""
import sys
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
# Add Python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))
# Berkeley colors for output
class Colors:
    BERKELEY_BLUE = '\033[94m'
    CALIFORNIA_GOLD = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
def print_berkeley_header():
    """Print Berkeley performance benchmark header."""
    print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}🐻 SciComp - Performance Benchmarks 🐻{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}")
    print(f"{Colors.CALIFORNIA_GOLD}Scientific Computing Performance Excellence{Colors.RESET}")
    print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}\n")
class PerformanceBenchmark:
    """Performance benchmarking suite."""
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
        }
    def print_status(self, message: str):
        """Print status message."""
        print(f"{Colors.BERKELEY_BLUE}[BENCHMARK]{Colors.RESET} {message}")
    def benchmark_matrix_operations(self):
        """Benchmark matrix operations."""
        self.print_status("Benchmarking matrix operations...")
        sizes = [100, 500, 1000, 2000, 5000]
        results = {}
        for n in sizes:
            self.print_status(f"  Matrix size: {n}x{n}")
            # Generate random matrices
            A = np.random.random((n, n)).astype(np.float64)
            B = np.random.random((n, n)).astype(np.float64)
            # Matrix multiplication
            start = time.time()
            C = A @ B
            mm_time = time.time() - start
            mm_gflops = (2 * n**3) / (mm_time * 1e9)
            # Matrix inversion
            start = time.time()
            try:
                A_inv = np.linalg.inv(A)
                inv_time = time.time() - start
                inv_gflops = (n**3) / (inv_time * 1e9)  # Rough estimate
            except:
                inv_time = float('inf')
                inv_gflops = 0
            # Eigendecomposition
            start = time.time()
            try:
                eigenvals, eigenvecs = np.linalg.eig(A)
                eig_time = time.time() - start
                eig_gflops = (4 * n**3) / (eig_time * 1e9)  # Rough estimate
            except:
                eig_time = float('inf')
                eig_gflops = 0
            results[n] = {
                'multiplication': {'time': mm_time, 'gflops': mm_gflops},
                'inversion': {'time': inv_time, 'gflops': inv_gflops},
                'eigendecomposition': {'time': eig_time, 'gflops': eig_gflops}
            }
        self.results['matrix_operations'] = results
    def benchmark_quantum_physics(self):
        """Benchmark quantum physics operations."""
        self.print_status("Benchmarking quantum physics...")
        try:
            from Quantum.core.quantum_states import QuantumState, BellStates, EntanglementMeasures
            # Bell state generation and concurrence calculation
            sizes = [2, 4, 8, 16, 32]  # Number of qubits (system size = 2^n)
            results = {}
            for n_qubits in sizes:
                if n_qubits <= 16:  # Avoid memory issues
                    self.print_status(f"  Quantum system: {n_qubits} qubits")
                    # Create random quantum state
                    dim = 2**n_qubits
                    start = time.time()
                    # Random state vector
                    state_vec = np.random.random(dim) + 1j * np.random.random(dim)
                    state = QuantumState(state_vec)
                    creation_time = time.time() - start
                    # Normalization check
                    start = time.time()
                    is_normalized = state.is_normalized()
                    norm_time = time.time() - start
                    results[n_qubits] = {
                        'state_creation_time': creation_time,
                        'normalization_check_time': norm_time,
                        'dimension': dim
                    }
            self.results['quantum_physics'] = results
        except Exception as e:
            self.results['quantum_physics'] = {'error': str(e)}
    def benchmark_optimization(self):
        """Benchmark optimization algorithms."""
        self.print_status("Benchmarking optimization algorithms...")
        try:
            from Optimization.unconstrained import BFGS
            from Optimization.linear_programming import SimplexMethod, LinearProgram
            results = {}
            # Test BFGS on different problem sizes
            dimensions = [2, 5, 10, 20, 50]
            for dim in dimensions:
                if dim <= 20:  # Reasonable limit for benchmarking
                    self.print_status(f"  BFGS optimization: {dim}D problem")
                    # Quadratic function: f(x) = x^T * A * x + b^T * x
                    A = np.random.random((dim, dim))
                    A = A.T @ A  # Make positive definite
                    b = np.random.random(dim)
                    def objective(x):
                        return 0.5 * x.T @ A @ x + b.T @ x
                    def gradient(x):
                        return A @ x + b
                    optimizer = BFGS(tolerance=1e-6, max_iterations=100)
                    x0 = np.random.random(dim)
                    start = time.time()
                    result = optimizer.minimize(objective, x0, gradient=gradient)
                    opt_time = time.time() - start
                    results[f'bfgs_{dim}d'] = {
                        'time': opt_time,
                        'iterations': result.nit,
                        'function_evaluations': result.nfev,
                        'success': result.success
                    }
            # Test Linear Programming
            self.print_status("  Linear programming benchmark")
            problem_sizes = [10, 50, 100]
            for n in problem_sizes:
                if n <= 100:
                    # Random LP problem
                    c = np.random.random(n)
                    A = np.random.random((n//2, n))
                    b = np.random.random(n//2) + 1  # Ensure feasibility
                    lp = LinearProgram(c=c, A=A, b=b, sense='min')
                    solver = SimplexMethod(verbose=False)
                    start = time.time()
                    try:
                        result = solver.solve(lp)
                        lp_time = time.time() - start
                        results[f'lp_{n}vars'] = {
                            'time': lp_time,
                            'success': result.success,
                            'variables': n,
                            'constraints': n//2
                        }
                    except:
                        results[f'lp_{n}vars'] = {'error': 'failed'}
            self.results['optimization'] = results
        except Exception as e:
            self.results['optimization'] = {'error': str(e)}
    def benchmark_signal_processing(self):
        """Benchmark signal processing operations."""
        self.print_status("Benchmarking signal processing...")
        try:
            from Signal_Processing.core.fourier_transforms import FFT, SpectralAnalysis
            fft = FFT()
            spectral = SpectralAnalysis()
            # Test FFT on different signal lengths
            signal_lengths = [1024, 4096, 16384, 65536, 262144]
            results = {}
            for N in signal_lengths:
                if N <= 262144:  # Reasonable limit
                    self.print_status(f"  FFT: {N} samples")
                    # Generate test signal
                    t = np.linspace(0, 1, N)
                    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
                    signal += 0.1 * np.random.randn(N)
                    # FFT benchmark
                    start = time.time()
                    frequencies, spectrum = fft.compute_fft(signal, sample_rate=N)
                    fft_time = time.time() - start
                    # Power spectrum benchmark
                    start = time.time()
                    freq_ps, power_spectrum = spectral.power_spectrum(signal, sample_rate=N)
                    ps_time = time.time() - start
                    results[N] = {
                        'fft_time': fft_time,
                        'power_spectrum_time': ps_time,
                        'samples_per_second': N / fft_time if fft_time > 0 else 0
                    }
            self.results['signal_processing'] = results
        except Exception as e:
            self.results['signal_processing'] = {'error': str(e)}
    def benchmark_thermal_transport(self):
        """Benchmark thermal transport simulations."""
        self.print_status("Benchmarking thermal transport...")
        try:
            from Thermal_Transport.core.heat_equation import HeatEquationSolver1D
            # Test different grid sizes
            grid_sizes = [50, 100, 200, 500, 1000]
            results = {}
            for nx in grid_sizes:
                self.print_status(f"  Heat equation 1D: {nx} grid points")
                solver = HeatEquationSolver1D(L=1.0, nx=nx, alpha=0.01)
                # Initial condition
                x = solver.grid
                T_initial = 100 * np.exp(-((x - 0.5) / 0.1)**2)
                # Solve
                start = time.time()
                T_final = solver.solve(T_initial, t_final=0.1, dt=0.0001)
                solve_time = time.time() - start
                results[nx] = {
                    'solve_time': solve_time,
                    'grid_points': nx,
                    'time_steps': int(0.1 / 0.0001),
                    'points_per_second': nx / solve_time if solve_time > 0 else 0
                }
            self.results['thermal_transport'] = results
        except Exception as e:
            self.results['thermal_transport'] = {'error': str(e)}
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print_berkeley_header()
        print(f"{Colors.BOLD}System Information:{Colors.RESET}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"CPU Frequency: {self.system_info['cpu_freq']} MHz")
        print(f"Memory: {self.system_info['memory_gb']:.2f} GB")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print()
        # Run benchmarks
        benchmarks = [
            ("Matrix Operations", self.benchmark_matrix_operations),
            ("Quantum Physics", self.benchmark_quantum_physics),
            ("Optimization", self.benchmark_optimization),
            ("Signal Processing", self.benchmark_signal_processing),
            ("Thermal Transport", self.benchmark_thermal_transport),
        ]
        for name, benchmark_func in benchmarks:
            try:
                benchmark_func()
                print(f"{Colors.GREEN}✓{Colors.RESET} {name} benchmark completed")
            except Exception as e:
                print(f"{Colors.RED}✗{Colors.RESET} {name} benchmark failed: {e}")
        # Print summary
        self._print_summary()
    def _print_summary(self):
        """Print benchmark summary."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}PERFORMANCE BENCHMARK SUMMARY{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        # Matrix operations summary
        if 'matrix_operations' in self.results:
            print(f"\n{Colors.BOLD}Matrix Operations:{Colors.RESET}")
            for size, ops in self.results['matrix_operations'].items():
                print(f"  {size}x{size} matrix multiplication: {ops['multiplication']['gflops']:.2f} GFLOPS")
        # Quantum physics summary
        if 'quantum_physics' in self.results:
            print(f"\n{Colors.BOLD}Quantum Physics:{Colors.RESET}")
            for n_qubits, metrics in self.results['quantum_physics'].items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    print(f"  {n_qubits} qubits: state creation {metrics['state_creation_time']:.4f}s")
        # Optimization summary
        if 'optimization' in self.results:
            print(f"\n{Colors.BOLD}Optimization:{Colors.RESET}")
            for prob, metrics in self.results['optimization'].items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    if 'bfgs' in prob:
                        print(f"  {prob}: {metrics['time']:.4f}s ({metrics['iterations']} iterations)")
                    elif 'lp' in prob:
                        print(f"  {prob}: {metrics['time']:.4f}s")
        # Signal processing summary
        if 'signal_processing' in self.results:
            print(f"\n{Colors.BOLD}Signal Processing:{Colors.RESET}")
            for N, metrics in self.results['signal_processing'].items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    print(f"  FFT {N} samples: {metrics['samples_per_second']:.0f} samples/sec")
        # Thermal transport summary
        if 'thermal_transport' in self.results:
            print(f"\n{Colors.BOLD}Thermal Transport:{Colors.RESET}")
            for nx, metrics in self.results['thermal_transport'].items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    print(f"  1D heat equation {nx} points: {metrics['points_per_second']:.0f} points/sec")
        print(f"\n{Colors.CALIFORNIA_GOLD}🐻💙💛 Berkeley SciComp Performance Benchmarking Complete! 💙💛🐻{Colors.RESET}")
def main():
    """Main benchmarking function."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
if __name__ == "__main__":
    main()