"""
Base Agent class for all Turingo agents
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str
    content: Dict[str, Any]
    priority: int = 5  # 1-10, higher is more urgent
    timestamp: str = None


class TuringoAgent:
    """
    Base class for all Turingo agents

    All agents share:
    - Access to knowledge blackboard
    - Async message passing
    - Standard logging
    - Performance tracking
    """

    def __init__(self, name: str, role: str, specialty: str):
        self.name = name
        self.role = role  # executive | specialist | consultant
        self.specialty = specialty
        self.message_queue: List[AgentMessage] = []
        self.active = True
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "avg_time_seconds": 0.0
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally return response"""
        raise NotImplementedError(f"{self.name} must implement process_message")

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task"""
        raise NotImplementedError(f"{self.name} must implement execute_task")

    def log(self, message: str, level: str = "INFO"):
        """Log agent activity"""
        print(f"[{self.name}] {level}: {message}")

    def update_metrics(self, success: bool, duration: float):
        """Update performance metrics"""
        self.performance_metrics["tasks_completed"] += 1
        old_rate = self.performance_metrics["success_rate"]
        old_time = self.performance_metrics["avg_time_seconds"]
        n = self.performance_metrics["tasks_completed"]

        # Running average
        self.performance_metrics["success_rate"] = (
            old_rate * (n-1) + (1.0 if success else 0.0)
        ) / n
        self.performance_metrics["avg_time_seconds"] = (
            old_time * (n-1) + duration
        ) / n

    async def send_message(self, recipient: str, content: Dict[str, Any], priority: int = 5):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            priority=priority
        )
        # In production, this would go through a message broker
        # For now, we'll simulate async messaging
        await asyncio.sleep(0.01)
        return message

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "name": self.name,
            "role": self.role,
            "specialty": self.specialty,
            "active": self.active,
            "metrics": self.performance_metrics
        }


class ExecutiveAgent(TuringoAgent):
    """Base class for executive agents (always active)"""

    def __init__(self, name: str, specialty: str):
        super().__init__(name, role="executive", specialty=specialty)
        self.priority_level = 10  # Highest priority


class SpecialistAgent(TuringoAgent):
    """Base class for specialist agents (on-demand)"""

    def __init__(self, name: str, specialty: str):
        super().__init__(name, role="specialist", specialty=specialty)
        self.priority_level = 5


class ConsultantAgent(TuringoAgent):
    """Base class for consultant agents (rare, specialized tasks)"""

    def __init__(self, name: str, specialty: str):
        super().__init__(name, role="consultant", specialty=specialty)
        self.priority_level = 3
