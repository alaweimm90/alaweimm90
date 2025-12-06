"""
Tests for Consultation dataclass
"""
import pytest
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ghost_researcher.main import Consultation


class TestConsultationDataclass:
    """Test Consultation dataclass"""

    def test_consultation_creation(self, sample_consultation_data):
        """Test creating a Consultation instance"""
        c = Consultation(**sample_consultation_data)
        assert c.consultation_id == 1
        assert c.scientist == 'einstein'

    def test_consultation_to_dict(self, sample_consultation_data):
        """Test converting Consultation to dict"""
        c = Consultation(**sample_consultation_data)
        d = asdict(c)

        assert isinstance(d, dict)
        assert d['scientist'] == 'einstein'
        assert d['modern_problem'] == 'quantum computing'

    def test_consultation_required_fields(self):
        """Test that required fields must be provided"""
        with pytest.raises(TypeError):
            # Missing required fields
            Consultation(consultation_id=1)

    def test_consultation_field_types(self, sample_consultation_data):
        """Test field types are correct"""
        c = Consultation(**sample_consultation_data)

        assert isinstance(c.consultation_id, int)
        assert isinstance(c.scientist, str)
        assert isinstance(c.modern_problem, str)
        assert isinstance(c.domain, str)
        assert isinstance(c.initial_reaction, str)
        assert isinstance(c.analogies_to_their_time, list)
        assert isinstance(c.how_they_would_approach, str)
        assert isinstance(c.predicted_obstacles, list)
        assert isinstance(c.key_insights, list)
        assert isinstance(c.experimental_suggestions, list)
        assert isinstance(c.theoretical_framework, str)
        assert isinstance(c.characteristic_quotes, list)
        assert isinstance(c.thought_experiments, list)
        assert isinstance(c.confidence_in_opinion, float)
        assert isinstance(c.limitations_of_perspective, list)
        assert isinstance(c.created_at, str)
