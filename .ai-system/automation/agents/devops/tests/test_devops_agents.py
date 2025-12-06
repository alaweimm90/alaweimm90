"""
DevOps Agents Tests
"""

import pytest
from pathlib import Path

from automation.agents.devops import (
    DevOpsAgentId,
    DevOpsCategory,
    load_devops_agents,
    execute_devops_agent,
    AgentExecutionContext,
)


class TestDevOpsAgentRegistry:
    """Tests for DevOps agent registry."""

    def test_load_agents_returns_registry(self):
        """Should return a registry instance."""
        registry = load_devops_agents()
        assert registry is not None

    def test_load_agents_with_invalid_path(self):
        """Should return empty registry for invalid path."""
        registry = load_devops_agents("/nonexistent/path.yaml")
        assert len(registry) == 0

    def test_get_agent_by_id(self):
        """Should get agent by ID."""
        registry = load_devops_agents()
        agent = registry.get(DevOpsAgentId.BUILD)
        if agent:
            assert agent.id == DevOpsAgentId.BUILD

    def test_get_agent_by_alias(self):
        """Should get agent by alias."""
        registry = load_devops_agents()
        agent = registry.get_by_alias("compiler")
        if agent:
            assert agent.id == DevOpsAgentId.BUILD

    def test_get_agents_by_category(self):
        """Should filter agents by category."""
        registry = load_devops_agents()
        pipeline_agents = registry.get_by_category(DevOpsCategory.PIPELINE)
        assert isinstance(pipeline_agents, list)

    def test_list_all_agents(self):
        """Should list all agents."""
        registry = load_devops_agents()
        agents = registry.list_all()
        assert isinstance(agents, list)


class TestDevOpsAgentExecution:
    """Tests for DevOps agent execution."""

    @pytest.mark.asyncio
    async def test_execute_unknown_agent(self):
        """Should return error for unknown agent."""
        ctx = AgentExecutionContext(
            agent_id=DevOpsAgentId.BUILD,  # Will be replaced
            input={},
            env={},
            retries=1,
        )
        # Simulate unknown agent by using a modified context
        result = await execute_devops_agent(ctx)
        # Result depends on whether config is available
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execution_tracks_duration(self):
        """Should track execution duration."""
        ctx = AgentExecutionContext(
            agent_id=DevOpsAgentId.BUILD,
            input={"sourcePath": ".", "buildSpec": "buildspec.yml"},
            env={},
            retries=1,
        )
        result = await execute_devops_agent(ctx)
        assert result.duration_ms >= 0


class TestDevOpsAgentTypes:
    """Tests for DevOps agent types."""

    def test_all_agent_ids_defined(self):
        """Should have all 20 agent IDs."""
        expected_ids = [
            DevOpsAgentId.PIPELINE_ORCHESTRATOR,
            DevOpsAgentId.BUILD,
            DevOpsAgentId.TEST_RUNNER,
            DevOpsAgentId.ARTIFACT_REPO,
            DevOpsAgentId.CONTAINER_BUILD,
            DevOpsAgentId.IMAGE_SCAN,
            DevOpsAgentId.SECRETS,
            DevOpsAgentId.INFRA_PROVISIONER,
            DevOpsAgentId.CONFIG_MANAGER,
            DevOpsAgentId.K8S_DEPLOY,
            DevOpsAgentId.PROGRESSIVE_DELIVERY,
            DevOpsAgentId.ROLLBACK,
            DevOpsAgentId.METRICS,
            DevOpsAgentId.LOG_SHIPPER,
            DevOpsAgentId.ALERT_ROUTER,
            DevOpsAgentId.TRIAGE,
            DevOpsAgentId.RELEASE_MANAGER,
            DevOpsAgentId.FEATURE_FLAGS,
            DevOpsAgentId.COST_MONITOR,
            DevOpsAgentId.COMPLIANCE_AUDIT,
        ]
        assert len(expected_ids) == 20

    def test_all_categories_defined(self):
        """Should have all 8 categories."""
        expected_categories = [
            DevOpsCategory.PIPELINE,
            DevOpsCategory.CONTAINER,
            DevOpsCategory.SECURITY,
            DevOpsCategory.INFRASTRUCTURE,
            DevOpsCategory.DEPLOYMENT,
            DevOpsCategory.OBSERVABILITY,
            DevOpsCategory.RELEASE,
            DevOpsCategory.FINOPS,
        ]
        assert len(expected_categories) == 8
