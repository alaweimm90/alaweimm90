"""
Unit tests for FFT-Laplace deprecation warning

Tests verify that the FFT-Laplace method properly warns users
about mathematical issues and raises NotImplementedError.
"""

import pytest
import warnings

from Librex.methods.novel.fft_laplace import fft_laplace_optimize


def test_fft_laplace_raises_deprecation_warning():
    """Test that FFT-Laplace raises DeprecationWarning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            fft_laplace_optimize(problem=None, config={})
        except NotImplementedError:
            pass  # Expected

        # Check that a deprecation warning was issued
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "fundamental mathematical errors" in str(w[0].message).lower()


def test_fft_laplace_raises_not_implemented():
    """Test that FFT-Laplace raises NotImplementedError"""
    with pytest.raises(NotImplementedError, match="under mathematical review"):
        fft_laplace_optimize(problem=None, config={})


def test_fft_laplace_warning_mentions_alternatives():
    """Test that warning suggests alternative methods"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            fft_laplace_optimize(problem=None, config={})
        except NotImplementedError:
            pass

        # Check warning suggests alternatives
        warning_text = str(w[0].message).lower()
        assert any(method in warning_text for method in [
            'simulated_annealing',
            'genetic_algorithm',
            'tabu_search'
        ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
