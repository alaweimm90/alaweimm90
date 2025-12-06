"""
Test suite for ORCHEX Engine
"""

import pytest
from atlas_core.engine import ATLASEngine
from atlas_core.agent import AgentConfig
from atlas_core.agents import create_agent


def test_engine_initialization():
    """Test ORCHEX Engine initialization"""
    # Use mock Libria router
    engine = ATLASEngine(libria_enabled=False)
    assert engine is not None
    assert len(engine.agents) == 0
    assert not engine.libria_enabled


def test_agent_registration():
    """Test agent registration with ORCHEX Engine"""
    engine = ATLASEngine(libria_enabled=False)

    # Create and register agent
    agent = create_agent(
        agent_type="synthesis",
        agent_id="test_agent_1",
        specialization="machine_learning",
        skill_level=0.85,
    )

    engine.register_agent(agent)

    assert len(engine.agents) == 1
    assert "test_agent_1" in engine.agents


def test_multiple_agents():
    """Test registering multiple agents"""
    engine = ATLASEngine(libria_enabled=False)

    # Create various agent types
    agent_types = [
        "synthesis",
        "literature_review",
        "hypothesis_generation",
        "critical_analysis",
        "validation",
    ]

    for i, agent_type in enumerate(agent_types):
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"agent_{i}",
            specialization="general",
        )
        engine.register_agent(agent)

    assert len(engine.agents) == 5


def test_task_assignment():
    """Test task assignment to agents"""
    engine = ATLASEngine(libria_enabled=False)

    # Register agents
    for i in range(3):
        agent = create_agent(
            agent_type="synthesis",
            agent_id=f"agent_{i}",
            max_tasks=5,
        )
        engine.register_agent(agent)

    # Create task
    task = {
        "task_id": "task_001",
        "task_type": "synthesis",
        "complexity": 0.7,
    }

    # Assign task
    assigned_agent_id = engine.assign_task(task)

    assert assigned_agent_id is not None
    assert assigned_agent_id in engine.agents


def test_dialectical_workflow():
    """Test thesis-antithesis-synthesis workflow"""
    engine = ATLASEngine(libria_enabled=False)

    # Register required agents
    agent_types_needed = [
        "hypothesis_generation",
        "critical_analysis",
        "synthesis",
    ]

    for agent_type in agent_types_needed:
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"{agent_type}_agent",
        )
        engine.register_agent(agent)

    # Execute workflow
    result = engine.execute_workflow(
        workflow_type="thesis_antithesis_synthesis",
        inputs={"topic": "neural_architecture_search"},
    )

    assert "thesis" in result
    assert "antithesis" in result
    assert "synthesis" in result


def test_engine_stats():
    """Test engine statistics"""
    engine = ATLASEngine(libria_enabled=False)

    # Register agents
    for i in range(10):
        agent_type = ["synthesis", "validation"][i % 2]
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"agent_{i}",
        )
        engine.register_agent(agent)

    stats = engine.get_stats()

    assert stats["total_agents"] == 10
    assert "synthesis" in stats["agents_by_type"]
    assert "validation" in stats["agents_by_type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
