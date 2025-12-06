"""
Test configuration and fixtures for GhostResearcher
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime


@pytest.fixture
def temp_data_file():
    """Create a temporary data file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    # Delete the file so it doesn't exist (tests will create it)
    temp_path.unlink()
    yield temp_path
    # Clean up after test
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_consultation_data():
    """Sample consultation data for testing"""
    return {
        "consultation_id": 1,
        "scientist": "einstein",
        "modern_problem": "quantum computing",
        "domain": "physics",
        "initial_reaction": "Fascinating problem!",
        "analogies_to_their_time": [
            "Similar to early quantum mechanics",
            "Like the photoelectric effect"
        ],
        "how_they_would_approach": "Through thought experiments",
        "predicted_obstacles": [
            "Insufficient precision",
            "Mathematical complexity"
        ],
        "key_insights": [
            "Energy quantization is key",
            "Superposition matters"
        ],
        "experimental_suggestions": [
            "Test quantum states",
            "Measure coherence"
        ],
        "theoretical_framework": "Based on quantum principles",
        "characteristic_quotes": [
            "Imagination is more important than knowledge",
            "Make things simple"
        ],
        "thought_experiments": [
            "Imagine a quantum coin flip",
            "Consider superposition states"
        ],
        "confidence_in_opinion": 0.75,
        "limitations_of_perspective": [
            "Lacks modern computational knowledge",
            "Limited by 1955 physics"
        ],
        "created_at": datetime.now().isoformat()
    }


@pytest.fixture
def mock_scientist_profile():
    """Mock scientist profile for testing"""
    return {
        'name': 'Test Scientist',
        'life': '1900-1980',
        'field': 'Physics',
        'known_for': [
            'Test Theory',
            'Test Experiment',
            'Test Discovery'
        ],
        'personality': [
            'Methodical and systematic',
            'Creative thinker',
            'Rigorous experimentalist'
        ],
        'quotes': [
            'Science is wonderful',
            'Experiment carefully',
            'Think deeply'
        ]
    }


@pytest.fixture
def prepopulated_data_file(temp_data_file, sample_consultation_data):
    """Create a data file with existing consultation data"""
    data = {
        'consultations': {
            '1': sample_consultation_data
        }
    }
    with open(temp_data_file, 'w') as f:
        json.dump(data, f)
    return temp_data_file


@pytest.fixture
def all_scientist_names():
    """List of all available scientists"""
    return [
        'einstein', 'feynman', 'curie', 'darwin',
        'turing', 'lovelace', 'newton', 'franklin'
    ]


@pytest.fixture
def valid_domains():
    """List of valid problem domains"""
    return [
        'physics', 'biology', 'computer_science',
        'medicine', 'mathematics', 'chemistry'
    ]


@pytest.fixture
def sample_problems():
    """Sample modern problems for testing"""
    return {
        'physics': 'dark matter detection',
        'biology': 'CRISPR gene editing',
        'computer_science': 'artificial general intelligence',
        'medicine': 'personalized cancer therapy',
        'mathematics': 'P vs NP problem'
    }
