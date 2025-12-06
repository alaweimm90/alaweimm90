"""
Meta-Learning Core

Self-improving research system with personality-based agents.

Makes research fun with agents like:
- Grumpy Refuter 😠 - "Everything is flawed!"
- Skeptical Steve 🤨 - "Show me the data!"
- Failure Frank 🤦 - "I've seen this before..."
- Optimistic Oliver 😄 - "Brilliant idea!"
"""

__version__ = "0.1.0"

from meta_learning.protocol import MetaLearningProtocol
from meta_learning.agent_personality import (
    AgentPersonality,
    AgentMood,
    get_agent,
    get_agent_by_role,
    list_all_agents,
    create_custom_agent,
    AGENT_ROSTER,
)
from meta_learning.trajectory_recorder import (
    TrajectoryRecorder,
    ResearchTrajectory,
    ActionRecord,
)
from meta_learning.bandit import UCB1Bandit, ContextualBandit

__all__ = [
    # Main API
    "MetaLearningProtocol",

    # Agents
    "AgentPersonality",
    "AgentMood",
    "get_agent",
    "get_agent_by_role",
    "list_all_agents",
    "create_custom_agent",
    "AGENT_ROSTER",

    # Trajectory
    "TrajectoryRecorder",
    "ResearchTrajectory",
    "ActionRecord",

    # Bandit
    "UCB1Bandit",
    "ContextualBandit",
]
