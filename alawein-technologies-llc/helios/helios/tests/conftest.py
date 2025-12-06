"""
Pytest configuration and fixtures for HELIOS tests.

This file provides:
- Fixtures for common test objects
- Mocking utilities
- Test configuration
"""

import pytest


@pytest.fixture
def mock_hypothesis():
    """Create a mock hypothesis for testing."""
    return {
        'id': 'test_h1',
        'text': 'Novel neural network architecture improves accuracy',
        'domain': 'ml',
        'source_papers': ['paper1', 'paper2'],
        'novelty_score': 75.5,
        'metadata': {'category': 'architecture'},
    }


@pytest.fixture
def mock_hypotheses(mock_hypothesis):
    """Create multiple mock hypotheses."""
    return [
        mock_hypothesis,
        {
            'id': 'test_h2',
            'text': 'Quantum error correction achieves threshold',
            'domain': 'quantum',
            'source_papers': ['paper3'],
            'novelty_score': 82.0,
            'metadata': {'category': 'validation'},
        },
        {
            'id': 'test_h3',
            'text': 'Combinatorial optimization solved with novel method',
            'domain': 'optimization',
            'source_papers': [],
            'novelty_score': 68.5,
            'metadata': {'category': 'algorithm'},
        },
    ]


@pytest.fixture
def mock_validation_result():
    """Create a mock validation result."""
    return {
        'hypothesis_id': 'test_h1',
        'logical_score': 82.0,
        'empirical_score': 75.5,
        'analogical_score': 78.0,
        'boundary_score': 80.5,
        'mechanism_score': 76.0,
        'overall_score': 78.4,
        'weaknesses': [
            'Limited empirical validation',
            'Assumes normal distribution',
        ],
        'interrogation_results': {
            'num_questions': 200,
            'num_weaknesses_found': 3,
        },
        'validation_timestamp': '2025-11-19T03:00:00Z',
    }


@pytest.fixture
def mock_validation_results(mock_validation_result):
    """Create multiple mock validation results."""
    result1 = mock_validation_result.copy()
    result1['hypothesis_id'] = 'test_h1'
    result1['overall_score'] = 78.4

    result2 = mock_validation_result.copy()
    result2['hypothesis_id'] = 'test_h2'
    result2['overall_score'] = 85.2

    result3 = mock_validation_result.copy()
    result3['hypothesis_id'] = 'test_h3'
    result3['overall_score'] = 65.1

    return [result1, result2, result3]


@pytest.fixture
def mock_agent_performance():
    """Create mock agent performance data."""
    return {
        'agent_name': 'conservative',
        'domain': 'quantum',
        'validation_count': 10,
        'avg_score': 76.5,
        'success_rate': 0.7,
        'performance_history': [70.0, 75.5, 76.0, 77.5, 78.0],
    }


@pytest.fixture
def mock_failure():
    """Create a mock failure for Hall of Failures."""
    return {
        'hypothesis_text': 'Failed hypothesis example',
        'domain': 'ml',
        'reason_for_failure': 'Empirical validation failed against benchmark',
        'lessons_learned': [
            'Assumption of independence violated',
            'Model overfits to training data',
        ],
        'similarity': 0.85,
    }


@pytest.fixture
def available_domains():
    """Get list of available domains."""
    return [
        'quantum',
        'materials',
        'optimization',
        'ml',
        'nas',
        'synthesis',
        'graph',
    ]


# Test markers for organizing tests
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API keys"
    )
    # Advanced testing markers
    config.addinivalue_line(
        "markers", "property_based: mark test as property-based test (Hypothesis)"
    )
    config.addinivalue_line(
        "markers", "mutation: mark test as mutation detection test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance regression test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security vulnerability test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as load/stress test"
    )
    config.addinivalue_line(
        "markers", "advanced: mark test as part of advanced test suite"
    )
