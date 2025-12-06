"""
Trajectory Recorder

Records every decision and outcome for meta-learning.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel


class ActionRecord(BaseModel):
    """A single action taken by an agent"""
    timestamp: datetime
    agent_id: str
    agent_name: str
    action_type: str  # "refute", "interrogate", "assess_risk", etc.
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    score: Optional[float] = None
    duration_seconds: float = 0.0
    cost_dollars: float = 0.0
    metadata: Dict[str, Any] = {}


class ResearchTrajectory(BaseModel):
    """Complete trajectory of a research project"""
    trajectory_id: str
    topic: str
    domain: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    actions: List[ActionRecord] = []

    # Outcomes
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    experiments_run: int = 0
    papers_written: int = 0

    # Success metrics
    final_success: bool = False
    final_score: float = 0.0
    total_cost: float = 0.0
    total_duration: float = 0.0

    def add_action(self, action: ActionRecord):
        """Add an action to trajectory"""
        self.actions.append(action)
        self.total_cost += action.cost_dollars
        self.total_duration += action.duration_seconds

    def complete(self, success: bool, score: float):
        """Mark trajectory as complete"""
        self.completed_at = datetime.now()
        self.final_success = success
        self.final_score = score


class TrajectoryRecorder:
    """
    Records research trajectories for meta-learning

    Learns from:
    - Which agents work best for which tasks
    - Which strategies succeed/fail
    - What patterns lead to good research
    """

    def __init__(self, storage_path: str = "trajectories.jsonl"):
        """
        Initialize recorder

        Args:
            storage_path: Path to store trajectories (JSONL format)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Current trajectory
        self.current_trajectory: Optional[ResearchTrajectory] = None

    def start_trajectory(self, topic: str, domain: str) -> str:
        """Start recording a new trajectory"""
        trajectory_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_trajectory = ResearchTrajectory(
            trajectory_id=trajectory_id,
            topic=topic,
            domain=domain,
            started_at=datetime.now(),
        )

        return trajectory_id

    def record_action(
        self,
        agent_id: str,
        agent_name: str,
        action_type: str,
        input_data: Dict,
        output_data: Dict,
        success: bool,
        score: Optional[float] = None,
        duration: float = 0.0,
        cost: float = 0.0,
        **metadata
    ):
        """Record an agent action"""
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory. Call start_trajectory() first.")

        action = ActionRecord(
            timestamp=datetime.now(),
            agent_id=agent_id,
            agent_name=agent_name,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            success=success,
            score=score,
            duration_seconds=duration,
            cost_dollars=cost,
            metadata=metadata,
        )

        self.current_trajectory.add_action(action)

    def complete_trajectory(self, success: bool, score: float):
        """Complete current trajectory and save"""
        if not self.current_trajectory:
            return

        self.current_trajectory.complete(success, score)

        # Save to JSONL
        with open(self.storage_path, 'a') as f:
            f.write(self.current_trajectory.model_dump_json() + '\n')

        self.current_trajectory = None

    def load_trajectories(
        self,
        domain: Optional[str] = None,
        success_only: bool = False,
        limit: Optional[int] = None
    ) -> List[ResearchTrajectory]:
        """Load past trajectories"""
        trajectories = []

        if not self.storage_path.exists():
            return trajectories

        with open(self.storage_path, 'r') as f:
            for line in f:
                try:
                    traj_dict = json.loads(line)
                    traj = ResearchTrajectory(**traj_dict)

                    # Filter
                    if domain and traj.domain != domain:
                        continue
                    if success_only and not traj.final_success:
                        continue

                    trajectories.append(traj)

                    if limit and len(trajectories) >= limit:
                        break
                except Exception:
                    continue

        return trajectories

    def analyze_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Analyze how well an agent performs"""
        trajectories = self.load_trajectories()

        # Find all actions by this agent
        agent_actions = []
        for traj in trajectories:
            for action in traj.actions:
                if action.agent_id == agent_id:
                    agent_actions.append(action)

        if not agent_actions:
            return {
                "agent_id": agent_id,
                "total_uses": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "avg_duration": 0.0,
                "total_cost": 0.0,
            }

        # Compute statistics
        successful = [a for a in agent_actions if a.success]
        scores = [a.score for a in agent_actions if a.score is not None]

        return {
            "agent_id": agent_id,
            "total_uses": len(agent_actions),
            "success_rate": len(successful) / len(agent_actions),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_duration": sum(a.duration_seconds for a in agent_actions) / len(agent_actions),
            "total_cost": sum(a.cost_dollars for a in agent_actions),
        }

    def get_best_agents(self, action_type: str, top_k: int = 3) -> List[str]:
        """Get best performing agents for an action type"""
        trajectories = self.load_trajectories()

        # Collect agent performance
        agent_scores = {}

        for traj in trajectories:
            for action in traj.actions:
                if action.action_type == action_type and action.score is not None:
                    if action.agent_id not in agent_scores:
                        agent_scores[action.agent_id] = []
                    agent_scores[action.agent_id].append(action.score)

        # Average scores
        avg_scores = {
            agent_id: sum(scores) / len(scores)
            for agent_id, scores in agent_scores.items()
        }

        # Sort and return top k
        sorted_agents = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, score in sorted_agents[:top_k]]
