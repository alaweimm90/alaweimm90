"""
Shared fixtures for Swarm Voting tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(return_value="Yes")
    return orchestrator


@pytest.fixture
def simple_binary_question():
    """Simple yes/no question."""
    return "Should we proceed with this approach?"


@pytest.fixture
def complex_question():
    """Complex multi-option question."""
    return "Which optimization strategy should we prioritize?"


@pytest.fixture
def research_context():
    """Research context for informed voting."""
    return """
    Experimental results show:
    - 15% performance improvement
    - Statistical significance: p < 0.01
    - Sample size: n = 500
    - Validated across 3 different environments
    """
