"""
Shared fixtures for agent tournament tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List

from agent_tournaments.core.models import Agent, Problem


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Agent A provides a superior solution with better optimization."
    )
    return orchestrator


@pytest.fixture
def sample_agent():
    """Create a single sample agent."""
    return Agent(
        agent_id="test_agent",
        agent_name="Test Agent",
        expertise=["optimization", "machine_learning"],
    )


@pytest.fixture
def sample_agents_4() -> List[Agent]:
    """Create 4 sample agents for small tournaments."""
    return [
        Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}", expertise=["optimization"])
        for i in range(4)
    ]


@pytest.fixture
def sample_agents_8() -> List[Agent]:
    """Create 8 sample agents for medium tournaments."""
    return [
        Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}", expertise=["optimization"])
        for i in range(8)
    ]


@pytest.fixture
def sample_agents_16() -> List[Agent]:
    """Create 16 sample agents for large tournaments."""
    return [
        Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}", expertise=["optimization"])
        for i in range(16)
    ]


@pytest.fixture
def sample_problem() -> Problem:
    """Create a sample optimization problem."""
    return Problem(
        description="Minimize the quadratic function f(x) = x^2 + 2x + 1",
        domain="optimization",
        difficulty="medium",
    )


@pytest.fixture
def complex_problem() -> Problem:
    """Create a complex problem for testing."""
    return Problem(
        description="Design an optimal quantum circuit for solving the traveling salesman problem",
        domain="quantum_optimization",
        difficulty="hard",
    )
