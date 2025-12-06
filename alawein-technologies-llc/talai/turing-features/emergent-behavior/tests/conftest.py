"""
Shared fixtures for Emergent Behavior tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Beneficial emergence: Agents coordinating efficiently"
    )
    return orchestrator


@pytest.fixture
def mock_agent():
    """Create single mock agent."""
    return Mock(agent_id="test_agent", name="Test Agent")


@pytest.fixture
def coordinated_agents():
    """Create agents exhibiting coordinated behavior."""
    agents = []
    for i in range(10):
        agent = Mock(agent_id=f"agent_{i}", name=f"Coordinated Agent {i}")
        agent.behavior_pattern = "coordinated"
        agents.append(agent)
    return agents


@pytest.fixture
def diverse_agents():
    """Create agents with diverse behaviors."""
    agents = []
    behaviors = ["explorer", "exploiter", "coordinator", "specialist"]
    for i in range(12):
        agent = Mock(agent_id=f"agent_{i}", name=f"Diverse Agent {i}")
        agent.behavior_pattern = behaviors[i % len(behaviors)]
        agents.append(agent)
    return agents
