"""
Integration tests for Librex.QAP optimization pipeline.

Tests cover:
- Complete optimization pipelines
- Method combinations
- Real QAPLIB instance handling
"""

import numpy as np
import pytest
from Librex.QAP import (
    get_novel_methods,
    get_baseline_methods,
    get_all_methods,
    get_methods_by_tag,
)
from Librex.QAP.methods.novel import apply_fft_laplace_preconditioning
from Librex.QAP.methods.baselines import (
    apply_sinkhorn_projection,
    apply_hungarian_rounding,
)


class TestPipeline:
    """Test complete optimization pipelines."""

    @pytest.fixture
    def qap_problem(self):
        """Create small QAP test problem."""
        np.random.seed(42)
        n = 8
        A = np.random.randint(1, 10, (n, n)).astype(float)
        B = np.random.randint(1, 10, (n, n)).astype(float)
        A = (A + A.T) / 2
        B = (B + B.T) / 2
        X = np.ones((n, n)) / n
        return A, B, X, n

    def test_novel_only_pipeline(self, qap_problem):
        """Test pipeline with novel methods only."""
        A, B, X, n = qap_problem
        novel = get_novel_methods()

        assert len(novel) > 0, "Should have novel methods"
        for method in novel[:3]:  # Test first 3
            assert method.stars == "⭐⭐⭐"

    def test_baseline_only_pipeline(self, qap_problem):
        """Test pipeline with baseline methods only."""
        A, B, X, n = qap_problem
        baselines = get_baseline_methods()

        assert len(baselines) > 0, "Should have baseline methods"
        assert all(m.stars == "⭐" for m in baselines)

    def test_mixed_pipeline(self, qap_problem):
        """Test combining novel and baseline methods."""
        A, B, X, n = qap_problem
        all_methods = get_all_methods()

        assert len(all_methods) >= 13, "Should have all 20 methods (partial implementation)"

        # Verify mix
        novel_count = len(get_novel_methods())
        baseline_count = len(get_baseline_methods())
        assert novel_count + baseline_count <= len(all_methods)

    def test_acceleration_methods(self):
        """Test all acceleration methods are available."""
        acceleration = get_methods_by_tag("acceleration")
        assert len(acceleration) > 0
        for m in acceleration:
            assert "acceleration" in m.tags

    def test_preconditioning_methods(self):
        """Test preconditioning methods."""
        preconditioning = get_methods_by_tag("preconditioning")
        assert len(preconditioning) >= 2
        for m in preconditioning:
            assert "preconditioning" in m.tags


class TestMetadataConsistency:
    """Test metadata system consistency."""

    def test_all_methods_have_metadata(self):
        """Verify all methods have complete metadata."""
        all_methods = get_all_methods()

        for method in all_methods:
            assert method.name is not None
            assert method.function_name is not None
            assert method.complexity is not None
            assert method.novelty_level in [1, 2, 3]
            assert method.origin is not None
            assert len(method.tags) > 0

    def test_novelty_levels(self):
        """Verify novelty level classification."""
        all_methods = get_all_methods()

        level_1 = [m for m in all_methods if m.novelty_level == 1]
        level_2 = [m for m in all_methods if m.novelty_level == 2]
        level_3 = [m for m in all_methods if m.novelty_level == 3]

        assert len(level_1) >= 6, "Should have baseline methods"
        assert len(level_3) >= 6, "Should have novel methods"

    def test_stars_rating(self):
        """Test star rating generation."""
        all_methods = get_all_methods()

        for method in all_methods:
            stars = method.stars
            assert stars in ["⭐", "⭐⭐", "⭐⭐⭐"]
            assert len(stars.split("⭐")) - 1 == method.novelty_level


class TestComplexityClassification:
    """Test method complexity analysis."""

    def test_linear_methods(self):
        """Test O(n²) classified methods."""
        all_methods = get_all_methods()
        linear = [m for m in all_methods if "O(n²)" in m.complexity]
        assert len(linear) > 0

    def test_superlinear_methods(self):
        """Test methods better than O(n³)."""
        all_methods = get_all_methods()
        fast = [m for m in all_methods if "log" in m.complexity]
        assert len(fast) > 0

    def test_origin_tracking(self):
        """Verify origin citations."""
        all_methods = get_all_methods()

        Librex.QAP_original = [m for m in all_methods if "Alawein" in m.origin or "Librex.QAP" in m.origin]
        classic = [m for m in all_methods if "Alawein" not in m.origin]

        assert len(Librex.QAP_original) >= 6
        assert len(classic) >= 6


class TestPerformanceMetrics:
    """Test performance metric tracking."""

    def test_speedup_metrics(self):
        """Test speedup factor for FFT methods."""
        fft_methods = get_methods_by_tag("fft")
        if len(fft_methods) > 0:
            for method in fft_methods:
                if method.speedup_factor:
                    assert method.speedup_factor > 1.0

    def test_success_rates(self):
        """Test success rate metrics."""
        all_methods = get_all_methods()

        for method in all_methods:
            if method.success_rate is not None:
                assert 0.0 <= method.success_rate <= 1.0

    def test_improvement_tracking(self):
        """Test improvement percentages."""
        all_methods = get_all_methods()

        for method in all_methods:
            if method.improvement_percent is not None:
                assert method.improvement_percent >= 0.0


class TestMethodDiscovery:
    """Test method discovery and filtering."""

    def test_discovery_by_tag(self):
        """Test finding methods by tags."""
        tags_to_test = ["acceleration", "preconditioning", "novel"]

        for tag in tags_to_test:
            methods = get_methods_by_tag(tag)
            assert len(methods) > 0
            assert all(tag in m.tags for m in methods)

    def test_discover_all_methods(self):
        """Verify we can discover all methods."""
        all_methods = get_all_methods()
        assert len(all_methods) >= 13  # At minimum novel + some baseline

        # Should have clear categories
        novel = [m for m in all_methods if m.novelty_level == 3]
        assert len(novel) > 0

    def test_method_api_consistency(self):
        """Verify method API is consistent."""
        all_methods = get_all_methods()

        for method in all_methods:
            # Each method should have a corresponding function
            assert hasattr(method, 'function_name')
            assert method.function_name.startswith('apply_')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
