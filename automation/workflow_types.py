"""
Shared types for parallel workflow system.
Moved here to avoid circular imports between services and executor.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task or workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for parallel tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelTask:
    """A task that can be executed in parallel."""
    task_id: str
    stage: Dict[str, Any]
    context: Any  # WorkflowContext - avoiding circular import
    priority: TaskPriority = TaskPriority.MEDIUM
    resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkflowContext:
    """Context passed through workflow execution."""
    workflow_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, TaskResult] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def checkpoint(self, stage_name: str):
        """Add a checkpoint for completed stage."""
        self.checkpoints.append({
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "completed_stages": len([r for r in self.stage_results.values() if r.status == TaskStatus.COMPLETED])
        })
