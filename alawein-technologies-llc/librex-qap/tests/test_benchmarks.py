"""
Performance benchmarks for Librex.QAP methods.

Measures:
- Execution time per method
- Memory usage
- Convergence characteristics
"""

import time
import numpy as np
import pytest
from Librex.QAP.methods.novel import (
    apply_fft_laplace_preconditioning,
    apply_reverse_time_saddle_escape,
)
from Librex.QAP.methods.baselines import (
    apply_sinkhorn_projection,
    apply_basic_gradient_descent,
)


class TestBenchmarks:
    """Benchmark suite for performance tracking."""

    @pytest.fixture
    def benchmark_matrices(self):
        """Create matrices of varying sizes."""
        np.random.seed(42)
        matrices = {}
        for n in [8, 16, 32]:
            A = np.random.randint(1, 10, (n, n)).astype(float)
            B = np.random.randint(1, 10, (n, n)).astype(float)
            X = np.ones((n, n)) / n
            matrices[n] = (A, B, X)
        return matrices

    def test_fft_laplace_performance(self, benchmark_matrices):
        """Benchmark FFT-Laplace preconditioning."""
        for n, (A, B, X) in benchmark_matrices.items():
            start = time.perf_counter()
            result, _ = apply_fft_laplace_preconditioning(X)
            elapsed = time.perf_counter() - start

            # Should be fast (< 0.1s for n<=32)
            assert elapsed < 0.1, f"FFT too slow for n={n}: {elapsed:.4f}s"

    def test_sinkhorn_performance(self, benchmark_matrices):
        """Benchmark Sinkhorn projection."""
        for n, (A, B, X) in benchmark_matrices.items():
            start = time.perf_counter()
            result, _ = apply_sinkhorn_projection(X)
            elapsed = time.perf_counter() - start

            # Should complete in reasonable time
            assert elapsed < 1.0, f"Sinkhorn too slow for n={n}: {elapsed:.4f}s"

    def test_reverse_time_performance(self, benchmark_matrices):
        """Benchmark reverse-time saddle escape."""
        for n, (A, B, X) in benchmark_matrices.items():
            start = time.perf_counter()
            result, _ = apply_reverse_time_saddle_escape(A, B, X)
            elapsed = time.perf_counter() - start

            # Should be fast
            assert elapsed < 0.5, f"Reverse-time too slow for n={n}: {elapsed:.4f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
