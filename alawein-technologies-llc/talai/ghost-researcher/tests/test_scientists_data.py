"""
Tests for SCIENTISTS data structure and content
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ghost_researcher.main import SCIENTISTS


class TestScientistsDataStructure:
    """Test the SCIENTISTS database structure"""

    def test_scientists_dict_not_empty(self):
        """Test SCIENTISTS dict has entries"""
        assert len(SCIENTISTS) > 0

    def test_all_scientists_have_required_fields(self):
        """Test each scientist has all required fields"""
        required_fields = ['name', 'life', 'field', 'known_for', 'personality', 'quotes']

        for scientist_id, data in SCIENTISTS.items():
            for field in required_fields:
                assert field in data, f"{scientist_id} missing {field}"

    def test_all_scientists_have_lists(self):
        """Test list fields are actually lists"""
        for scientist_id, data in SCIENTISTS.items():
            assert isinstance(data['known_for'], list)
            assert isinstance(data['personality'], list)
            assert isinstance(data['quotes'], list)

    def test_all_scientists_have_sufficient_data(self):
        """Test each scientist has enough data in lists"""
        for scientist_id, data in SCIENTISTS.items():
            assert len(data['known_for']) >= 3, f"{scientist_id} needs more known_for items"
            assert len(data['personality']) >= 3, f"{scientist_id} needs more personality traits"
            assert len(data['quotes']) >= 3, f"{scientist_id} needs more quotes"


class TestIndividualScientists:
    """Test specific scientists are present with correct data"""

    def test_einstein_present(self):
        """Test Einstein is in database"""
        assert 'einstein' in SCIENTISTS
        assert SCIENTISTS['einstein']['name'] == 'Albert Einstein'

    def test_feynman_present(self):
        """Test Feynman is in database"""
        assert 'feynman' in SCIENTISTS
        assert SCIENTISTS['feynman']['name'] == 'Richard Feynman'

    def test_curie_present(self):
        """Test Marie Curie is in database"""
        assert 'curie' in SCIENTISTS
        assert SCIENTISTS['curie']['name'] == 'Marie Curie'

    def test_darwin_present(self):
        """Test Darwin is in database"""
        assert 'darwin' in SCIENTISTS
        assert SCIENTISTS['darwin']['name'] == 'Charles Darwin'

    def test_turing_present(self):
        """Test Turing is in database"""
        assert 'turing' in SCIENTISTS
        assert SCIENTISTS['turing']['name'] == 'Alan Turing'

    def test_lovelace_present(self):
        """Test Ada Lovelace is in database"""
        assert 'lovelace' in SCIENTISTS
        assert SCIENTISTS['lovelace']['name'] == 'Ada Lovelace'

    def test_newton_present(self):
        """Test Newton is in database"""
        assert 'newton' in SCIENTISTS
        assert SCIENTISTS['newton']['name'] == 'Isaac Newton'

    def test_franklin_present(self):
        """Test Rosalind Franklin is in database"""
        assert 'franklin' in SCIENTISTS
        assert SCIENTISTS['franklin']['name'] == 'Rosalind Franklin'


class TestLifeDates:
    """Test life date formatting"""

    def test_life_dates_format(self):
        """Test life dates are in correct format YYYY-YYYY"""
        import re
        pattern = re.compile(r'^\d{4}-\d{4}$')

        for scientist_id, data in SCIENTISTS.items():
            assert pattern.match(data['life']), f"{scientist_id} has invalid life format: {data['life']}"

    def test_birth_before_death(self):
        """Test birth year is before death year"""
        for scientist_id, data in SCIENTISTS.items():
            birth, death = data['life'].split('-')
            assert int(birth) < int(death), f"{scientist_id} has invalid date range"

    def test_historical_figures(self):
        """Test all scientists are historical (died before 2024)"""
        for scientist_id, data in SCIENTISTS.items():
            death_year = int(data['life'].split('-')[1])
            assert death_year < 2024, f"{scientist_id} should be historical"


class TestFieldsValidity:
    """Test fields are valid"""

    def test_fields_are_strings(self):
        """Test field values are strings"""
        for scientist_id, data in SCIENTISTS.items():
            assert isinstance(data['field'], str)
            assert len(data['field']) > 0

    def test_fields_are_reasonable(self):
        """Test fields contain reasonable values"""
        valid_field_keywords = [
            'Physics', 'Biology', 'Chemistry', 'Mathematics',
            'Computer Science', 'Computing', 'Science'
        ]

        for scientist_id, data in SCIENTISTS.items():
            field = data['field']
            assert any(keyword in field for keyword in valid_field_keywords), \
                f"{scientist_id} has unexpected field: {field}"


class TestQuotesQuality:
    """Test quotes are proper"""

    def test_quotes_are_strings(self):
        """Test all quotes are strings"""
        for scientist_id, data in SCIENTISTS.items():
            for quote in data['quotes']:
                assert isinstance(quote, str)
                assert len(quote) > 0

    def test_quotes_are_substantial(self):
        """Test quotes are substantial (not too short)"""
        for scientist_id, data in SCIENTISTS.items():
            for quote in data['quotes']:
                assert len(quote) > 10, f"{scientist_id} has very short quote: {quote}"
