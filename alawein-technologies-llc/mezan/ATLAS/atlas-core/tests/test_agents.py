"""
Test suite for ORCHEX Research Agents
"""

import pytest
from atlas_core.agent import AgentConfig, ResearchAgent
from atlas_core.agents import (
    SynthesisAgent,
    HypothesisGenerationAgent,
    CriticalAnalysisAgent,
    create_agent,
)


def test_agent_config():
    """Test AgentConfig dataclass"""
    config = AgentConfig(
        agent_id="test_1",
        agent_type="synthesis",
        specialization="machine_learning",
        skill_level=0.85,
        max_tasks=5,
        model="claude-3-opus",
    )

    assert config.agent_id == "test_1"
    assert config.agent_type == "synthesis"
    assert config.skill_level == 0.85


def test_research_agent_base():
    """Test ResearchAgent base class"""
    config = AgentConfig(
        agent_id="test_2",
        agent_type="synthesis",
        specialization="general",
        skill_level=0.8,
        max_tasks=3,
        model="claude-3-opus",
    )

    agent = SynthesisAgent(config)

    assert agent.config.agent_id == "test_2"
    assert agent.current_workload == 0
    assert len(agent.execution_history) == 0


def test_agent_can_accept_task():
    """Test agent task acceptance logic"""
    config = AgentConfig(
        agent_id="test_3",
        agent_type="synthesis",
        specialization="general",
        skill_level=0.8,
        max_tasks=2,
        model="claude-3-opus",
    )

    agent = SynthesisAgent(config)

    assert agent.can_accept_task({})

    # Simulate workload
    agent.current_workload = 2
    assert not agent.can_accept_task({})


def test_synthesis_agent():
    """Test SynthesisAgent execution"""
    agent = create_agent(
        agent_type="synthesis",
        agent_id="synth_1",
    )

    task = {
        "task_id": "task_001",
        "task_type": "synthesize",
        "thesis": {"conclusion": "A is true"},
        "antithesis": {"challenge": "But B contradicts A"},
    }

    result = agent.execute(task)

    assert "synthesis" in result
    assert "quality" in result
    assert result["quality"] > 0


def test_hypothesis_generation_agent():
    """Test HypothesisGenerationAgent execution"""
    agent = create_agent(
        agent_type="hypothesis_generation",
        agent_id="hyp_1",
    )

    task = {
        "task_id": "task_002",
        "task_type": "generate_hypothesis",
        "input": {"topic": "machine_learning"},
    }

    result = agent.execute(task)

    assert "hypothesis" in result
    assert "testable" in result
    assert result["testable"] is True


def test_critical_analysis_agent():
    """Test CriticalAnalysisAgent execution"""
    agent = create_agent(
        agent_type="critical_analysis",
        agent_id="crit_1",
    )

    task = {
        "task_id": "task_003",
        "task_type": "challenge_hypothesis",
        "thesis": {"hypothesis": "X causes Y"},
    }

    result = agent.execute(task)

    assert "challenges" in result
    assert "counterarguments" in result
    assert len(result["challenges"]) > 0


def test_create_agent_factory():
    """Test agent factory function"""
    agent = create_agent(
        agent_type="literature_review",
        agent_id="lit_1",
        specialization="NLP",
        skill_level=0.9,
    )

    assert agent.config.agent_type == "literature_review"
    assert agent.config.specialization == "NLP"
    assert agent.config.skill_level == 0.9


def test_agent_to_features():
    """Test agent feature extraction"""
    agent = create_agent(
        agent_type="validation",
        agent_id="val_1",
        skill_level=0.85,
        max_tasks=5,
    )

    features = agent.to_features()

    assert len(features) == 4  # skill, workload ratio, history length, specialization hash
    assert 0 <= features[0] <= 1  # skill level
    assert 0 <= features[1] <= 1  # workload ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
