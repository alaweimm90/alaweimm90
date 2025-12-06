"""
Unit tests for HELIOS Discovery Module.

Tests for:
- HypothesisGenerator
- BrainstormEngine
- Literature search integration
"""

import pytest


class TestHypothesisGenerator:
    """Test HypothesisGenerator class."""

    @pytest.mark.unit
    def test_hypothesis_structure(self, mock_hypothesis):
        """Test that hypothesis has required fields."""
        required_fields = ['id', 'text', 'domain']
        for field in required_fields:
            assert field in mock_hypothesis

    @pytest.mark.unit
    def test_mock_hypotheses_count(self, mock_hypotheses):
        """Test that mock hypotheses are created correctly."""
        assert len(mock_hypotheses) == 3
        assert all('id' in h for h in mock_hypotheses)
        assert all('text' in h for h in mock_hypotheses)

    @pytest.mark.unit
    def test_hypothesis_domains(self, mock_hypotheses):
        """Test that hypotheses have valid domains."""
        valid_domains = {
            'quantum', 'materials', 'optimization',
            'ml', 'nas', 'synthesis', 'graph'
        }
        for h in mock_hypotheses:
            assert h['domain'] in valid_domains

    @pytest.mark.unit
    def test_hypothesis_novelty_score(self, mock_hypothesis):
        """Test that novelty scores are in valid range."""
        score = mock_hypothesis.get('novelty_score', 0)
        assert 0 <= score <= 100, f"Score {score} out of range [0, 100]"

    @pytest.mark.unit
    @pytest.mark.requires_api
    def test_generate_basic(self):
        """Test basic hypothesis generation (requires API)."""
        try:
            from helios.core.discovery import HypothesisGenerator

            generator = HypothesisGenerator(num_hypotheses=2)
            hypotheses = generator.generate("test topic")

            assert isinstance(hypotheses, list)
            assert len(hypotheses) > 0
        except ImportError:
            pytest.skip("HypothesisGenerator not available")

    @pytest.mark.unit
    def test_generate_empty_topic_fails(self):
        """Test that empty topic raises ValueError."""
        try:
            from helios.core.discovery import HypothesisGenerator

            generator = HypothesisGenerator()
            with pytest.raises(ValueError):
                generator.generate("")
        except ImportError:
            pytest.skip("HypothesisGenerator not available")

    @pytest.mark.unit
    def test_hypothesis_metadata(self, mock_hypothesis):
        """Test that hypothesis metadata is present."""
        assert 'metadata' in mock_hypothesis
        assert isinstance(mock_hypothesis['metadata'], dict)

    @pytest.mark.unit
    def test_hypothesis_sources(self, mock_hypothesis):
        """Test that hypothesis includes source papers."""
        assert 'source_papers' in mock_hypothesis
        assert isinstance(mock_hypothesis['source_papers'], list)


class TestBrainstormEngine:
    """Test BrainstormEngine class."""

    @pytest.mark.unit
    def test_brainstorm_engine_exists(self):
        """Test that BrainstormEngine can be imported."""
        try:
            from helios.core.discovery import BrainstormEngine
            assert BrainstormEngine is not None
        except ImportError:
            pytest.skip("BrainstormEngine not available")

    @pytest.mark.unit
    @pytest.mark.requires_api
    def test_brainstorm_generates_ideas(self):
        """Test that brainstorm engine generates ideas (requires API)."""
        try:
            from helios.core.discovery import BrainstormEngine

            engine = BrainstormEngine()
            ideas = engine.brainstorm("quantum computing")

            assert isinstance(ideas, list)
            assert len(ideas) > 0
        except ImportError:
            pytest.skip("BrainstormEngine not available")


# Integration between modules would go here
@pytest.mark.integration
def test_discovery_validation_pipeline(mock_hypotheses, mock_validation_results):
    """Test workflow from discovery to validation."""
    assert len(mock_hypotheses) == len(mock_validation_results)

    for h, r in zip(mock_hypotheses, mock_validation_results):
        assert h['id'] == r['hypothesis_id']
        assert 0 <= r['overall_score'] <= 100
