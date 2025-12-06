"""
Meta-Learning Protocol

Self-improving research system that learns from experience.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from meta_learning.agent_personality import (
    AgentPersonality,
    get_agent,
    get_agent_by_role,
    list_all_agents,
)
from meta_learning.trajectory_recorder import TrajectoryRecorder
from meta_learning.bandit import ContextualBandit


class MetaLearningProtocol:
    """
    Meta-Learning system for autonomous research

    Features:
    - Agent personality system (funny names!)
    - Trajectory recording (learn from every project)
    - UCB1 bandit (optimal agent selection)
    - Performance tracking (continuous improvement)
    """

    def __init__(
        self,
        trajectory_path: str = "trajectories.jsonl",
        exploration_param: float = 2.0,
    ):
        """
        Initialize meta-learning

        Args:
            trajectory_path: Where to store trajectories
            exploration_param: UCB1 exploration parameter (higher = more exploration)
        """
        self.trajectory_recorder = TrajectoryRecorder(trajectory_path)

        # Get all available agents
        all_agents = list_all_agents()
        agent_ids = list(all_agents.keys())

        # Common research domains (Cycle 15: Added chemistry, biology)
        contexts = [
            "optimization",
            "machine_learning",
            "computer_science",
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "general",
        ]

        # Initialize bandit
        self.bandit = ContextualBandit(
            agent_ids=agent_ids, contexts=contexts, c=exploration_param
        )

        # Load past performance
        self._warm_start()

    def _warm_start(self):
        """Initialize bandit from past trajectories"""
        trajectories = self.trajectory_recorder.load_trajectories()

        for traj in trajectories:
            for action in traj.actions:
                # Normalize score to 0-1 for bandit
                reward = action.score / 100.0 if action.score else (1.0 if action.success else 0.0)

                # Update bandit
                self.bandit.update(
                    agent_id=action.agent_id,
                    reward=reward,
                    context=traj.domain,
                )

        print(f"✓ Loaded {len(trajectories)} past trajectories for warm start")

    def start_research(self, topic: str, domain: str) -> str:
        """Start a new research project"""
        trajectory_id = self.trajectory_recorder.start_trajectory(topic, domain)
        print(f"\n{'='*80}")
        print(f"🧠 Meta-Learning Enabled!")
        print(f"{'='*80}")
        print(f"Trajectory ID: {trajectory_id}")
        print(f"Past projects: {self.bandit.total_pulls}")
        print(f"{'='*80}\n")
        return trajectory_id

    def select_agent(
        self,
        role: str,
        domain: str,
        force_agent: Optional[str] = None
    ) -> AgentPersonality:
        """
        Select best agent for a task

        Args:
            role: Task role (e.g., "self_refutation")
            domain: Research domain (e.g., "optimization")
            force_agent: Force specific agent (bypasses bandit)

        Returns:
            AgentPersonality to use
        """
        if force_agent:
            agent = get_agent(force_agent)
            if agent:
                print(f"  🎭 Using forced agent: {agent.name} {agent.emoji}")
                return agent

        # Get candidates for this role
        all_agents = list_all_agents()
        role_agents = [
            agent_id for agent_id, agent in all_agents.items()
            if agent.role == role
        ]

        if not role_agents:
            # Fallback to default
            agent = get_agent_by_role(role)
            if agent:
                return agent
            # Ultimate fallback
            return list(all_agents.values())[0]

        # Use bandit to select best agent
        if len(role_agents) == 1:
            selected_id = role_agents[0]
        else:
            # Multiple agents for this role - use bandit
            selected_id = self.bandit.select_agent(context=domain)

            # Ensure selected agent has correct role
            if selected_id not in role_agents:
                selected_id = role_agents[0]

        agent = get_agent(selected_id)

        # Show selection
        stats = self.bandit.get_context_stats(domain).get(selected_id, {})
        pulls = stats.get("pulls", 0)
        avg_reward = stats.get("avg_reward", 0.0)

        print(f"  🎭 Selected: {agent.name} {agent.emoji}")
        print(f"     {agent.catchphrase}")
        if pulls > 0:
            print(f"     Performance: {pulls} uses, {avg_reward:.2f} avg score")

        return agent

    def record_agent_action(
        self,
        agent: AgentPersonality,
        action_type: str,
        input_data: Dict,
        output_data: Dict,
        success: bool,
        score: Optional[float] = None,
        duration: float = 0.0,
        cost: float = 0.0,
    ):
        """Record an agent's action"""
        self.trajectory_recorder.record_action(
            agent_id=agent.role,  # Use role as ID for now
            agent_name=agent.name,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            success=success,
            score=score,
            duration=duration,
            cost=cost,
        )

    def agent_react(self, agent: AgentPersonality, score: float) -> str:
        """Get agent's reaction to a score"""
        reaction = agent.react_to_score(score)
        print(f"     {reaction}")
        return reaction

    def complete_research(
        self,
        success: bool,
        final_score: float,
        domain: str
    ):
        """Complete research and update bandit"""
        # Record trajectory
        self.trajectory_recorder.complete_trajectory(success, final_score)

        # Update bandit based on final outcome
        if self.trajectory_recorder.current_trajectory:
            # This shouldn't happen, but handle it
            return

        # Get last trajectory
        trajectories = self.trajectory_recorder.load_trajectories(limit=1)
        if not trajectories:
            return

        last_traj = trajectories[0]

        # Update bandit for each agent used
        for action in last_traj.actions:
            if action.score is not None:
                reward = action.score / 100.0
            else:
                reward = 1.0 if success else 0.0

            self.bandit.update(
                agent_id=action.agent_id,
                reward=reward,
                context=domain,
            )

        print(f"\n📊 Meta-Learning Updated!")
        print(f"   Total projects: {self.bandit.total_pulls}")

    def get_best_agents(self, domain: str, top_k: int = 3) -> List[AgentPersonality]:
        """Get top performing agents for a domain"""
        stats = self.bandit.get_context_stats(domain)

        # Sort by avg reward
        sorted_agents = sorted(
            stats.items(),
            key=lambda x: x[1].get("avg_reward", 0.0),
            reverse=True
        )

        best_ids = [agent_id for agent_id, _ in sorted_agents[:top_k]]
        return [get_agent(agent_id) for agent_id in best_ids if get_agent(agent_id)]

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the system has learned"""
        return {
            "total_projects": self.bandit.total_pulls,
            "total_agent_pulls": sum(self.bandit.agent_pulls.values()),
            "agent_performance": self.bandit.get_stats(),
        }

    def export_learning(self, output_path: str):
        """Export learned knowledge"""
        import json

        summary = self.get_learning_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Exported learning to {output_path}")
