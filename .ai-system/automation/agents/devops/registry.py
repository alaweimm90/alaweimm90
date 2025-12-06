"""
DevOps Agent Registry
Load and manage DevOps agents from YAML configuration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .types import (
    DevOpsAgent,
    DevOpsAgentId,
    DevOpsCategory,
    LLMConfig,
)


class DevOpsAgentRegistry:
    """Registry for DevOps agents."""

    def __init__(self):
        self._agents: Dict[DevOpsAgentId, DevOpsAgent] = {}
        self._aliases: Dict[str, DevOpsAgentId] = {}

    def register(self, agent: DevOpsAgent) -> None:
        """Register an agent."""
        self._agents[agent.id] = agent
        for alias in agent.aliases:
            self._aliases[alias] = agent.id

    def get(self, agent_id: DevOpsAgentId) -> Optional[DevOpsAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_by_alias(self, alias: str) -> Optional[DevOpsAgent]:
        """Get an agent by alias."""
        agent_id = self._aliases.get(alias)
        if agent_id:
            return self._agents.get(agent_id)
        return None

    def get_by_category(self, category: DevOpsCategory) -> List[DevOpsAgent]:
        """Get all agents in a category."""
        return [a for a in self._agents.values() if a.category == category]

    def list_all(self) -> List[DevOpsAgent]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_ids(self) -> List[DevOpsAgentId]:
        """List all agent IDs."""
        return list(self._agents.keys())

    def list_aliases(self) -> Dict[str, DevOpsAgentId]:
        """List all aliases."""
        return dict(self._aliases)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: DevOpsAgentId) -> bool:
        return agent_id in self._agents


def load_devops_agents(config_path: Optional[str] = None) -> DevOpsAgentRegistry:
    """
    Load DevOps agents from YAML configuration.

    Args:
        config_path: Path to the YAML config file. If not provided,
                    uses the default path.

    Returns:
        DevOpsAgentRegistry with loaded agents.
    """
    registry = DevOpsAgentRegistry()

    # Default path
    if config_path is None:
        base_path = Path(__file__).parent.parent / "config" / "devops-agents.yaml"
        config_path = str(base_path)

    if not os.path.exists(config_path):
        print(f"Warning: DevOps agents config not found at {config_path}")
        return registry

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        return registry

    # Build category lookup
    category_lookup: Dict[str, DevOpsCategory] = {}
    categories = config.get("categories", {})
    for cat_name, cat_data in categories.items():
        try:
            category = DevOpsCategory(cat_name)
            for agent_name in cat_data.get("agents", []):
                category_lookup[agent_name] = category
        except ValueError:
            continue

    # Load agents
    agents_config = config.get("agents", {})
    for name, agent_data in agents_config.items():
        try:
            agent_id_str = agent_data.get("id", name)
            agent_id = DevOpsAgentId(agent_id_str)

            # Parse LLM config
            llm_data = agent_data.get("llm_config", {})
            llm_config = LLMConfig(
                model=llm_data.get("model", "claude-3-sonnet"),
                temperature=llm_data.get("temperature", 0.2),
                max_tokens=llm_data.get("max_tokens"),
            )

            # Get category
            category = category_lookup.get(name, DevOpsCategory.PIPELINE)

            agent = DevOpsAgent(
                id=agent_id,
                role=agent_data.get("role", name),
                goal=agent_data.get("goal", ""),
                backstory=agent_data.get("backstory", ""),
                aliases=agent_data.get("aliases", []),
                tools=agent_data.get("tools", []),
                inputs=agent_data.get("inputs", []),
                outputs=agent_data.get("outputs", []),
                category=category,
                llm_config=llm_config,
                version_compat=agent_data.get("version_compat", {}),
                security=agent_data.get("security", []),
            )

            registry.register(agent)

        except (ValueError, KeyError) as e:
            print(f"Warning: Failed to load agent {name}: {e}")
            continue

    return registry


# Global registry instance
_global_registry: Optional[DevOpsAgentRegistry] = None


def get_registry() -> DevOpsAgentRegistry:
    """Get or create the global registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = load_devops_agents()
    return _global_registry
