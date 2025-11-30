"""
Data models for the ATLAS SDK.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class TaskStatus(str, Enum):
    """Task execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Available task types."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"


class TaskContext(BaseModel):
    """Context information for tasks."""
    repository: Optional[str] = None
    files: Optional[List[str]] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None


class TaskRequirements(BaseModel):
    """Requirements for task execution."""
    required_capabilities: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None
    priority: Optional[str] = None
    max_cost_usd: Optional[float] = None
    preferred_providers: Optional[List[str]] = None


class TaskResult(BaseModel):
    """Result of a completed task."""
    code: Optional[str] = None
    explanation: Optional[str] = None
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Task:
    """
    Represents a task submitted to ATLAS.

    Attributes:
        task_id: Unique task identifier
        type: Type of task
        status: Current execution status
        description: Task description
        result: Task result (when completed)
        agent_id: ID of agent that executed the task
        created_at: Task creation timestamp
        completed_at: Task completion timestamp
        cost_usd: Cost incurred for task execution
        error: Error message (if failed)
    """

    def __init__(
        self,
        task_id: str,
        type: Optional[str] = None,
        status: TaskStatus = TaskStatus.QUEUED,
        description: Optional[str] = None,
        client: Optional['AtlasClient'] = None
    ):
        self.task_id = task_id
        self.type = type
        self.status = status
        self.description = description
        self.client = client

        # Private attributes populated by API responses
        self._result: Optional[TaskResult] = None
        self._agent_id: Optional[str] = None
        self._created_at: Optional[str] = None
        self._completed_at: Optional[str] = None
        self._cost_usd: Optional[float] = None
        self._error: Optional[str] = None

    @property
    def result(self) -> Optional[TaskResult]:
        """Get the task result."""
        return self._result

    @property
    def agent_id(self) -> Optional[str]:
        """Get the agent that executed this task."""
        return self._agent_id

    @property
    def created_at(self) -> Optional[str]:
        """Get task creation timestamp."""
        return self._created_at

    @property
    def completed_at(self) -> Optional[str]:
        """Get task completion timestamp."""
        return self._completed_at

    @property
    def cost_usd(self) -> Optional[float]:
        """Get task execution cost."""
        return self._cost_usd

    @property
    def error(self) -> Optional[str]:
        """Get error message if task failed."""
        return self._error

    def wait(self, timeout: Optional[float] = None, poll_interval: float = 2.0) -> 'Task':
        """
        Wait for task completion.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds

        Returns:
            Updated Task object

        Raises:
            TimeoutError: If timeout is exceeded
        """
        if not self.client:
            raise RuntimeError("No client associated with this task")

        import time
        start_time = time.time()

        while self.status in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {self.task_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)
            updated_task = self.client.get_task(self.task_id)

            # Update this task's attributes
            self.status = updated_task.status
            self._result = updated_task._result
            self._agent_id = updated_task._agent_id
            self._completed_at = updated_task._completed_at
            self._cost_usd = updated_task._cost_usd
            self._error = updated_task._error

        return self

    def cancel(self) -> bool:
        """
        Cancel the task if it's still running.

        Returns:
            True if cancellation was successful
        """
        if not self.client:
            raise RuntimeError("No client associated with this task")

        if self.status not in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
            return False

        # Implementation would call client.cancel_task()
        # For now, return False as placeholder
        return False

    def __repr__(self) -> str:
        return f"Task(task_id='{self.task_id}', status={self.status}, type='{self.type}')"


class Agent(BaseModel):
    """Represents an AI agent available in ATLAS."""
    agent_id: str
    name: str
    provider: str
    model: str
    status: str
    capabilities: List[str]

    def __repr__(self) -> str:
        return f"Agent(agent_id='{self.agent_id}', name='{self.name}', provider='{self.provider}', status='{self.status}')"


class Analysis:
    """
    Represents a repository analysis job.

    Attributes:
        analysis_id: Unique analysis identifier
        status: Current analysis status
        repository: Repository being analyzed
        summary: Analysis summary (when completed)
        opportunities: Refactoring opportunities found
    """

    def __init__(self, analysis_id: str, status: str, client: Optional['AtlasClient'] = None):
        self.analysis_id = analysis_id
        self.status = status
        self.client = client
        self.repository: Optional[str] = None
        self.summary: Optional[Dict[str, Any]] = None
        self.opportunities: Optional[List[Dict[str, Any]]] = None

    def wait(self, timeout: Optional[float] = None, poll_interval: float = 5.0) -> 'Analysis':
        """
        Wait for analysis completion.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds

        Returns:
            Updated Analysis object
        """
        if not self.client:
            raise RuntimeError("No client associated with this analysis")

        import time
        start_time = time.time()

        while self.status not in ['completed', 'failed']:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Analysis {self.analysis_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)
            updated_analysis = self.client._make_request('GET', f'/analyze/{self.analysis_id}')
            data = updated_analysis['data']

            self.status = data['status']
            if 'summary' in data:
                self.summary = data['summary']
            if 'opportunities' in data:
                self.opportunities = data['opportunities']

        return self

    def __repr__(self) -> str:
        return f"Analysis(analysis_id='{self.analysis_id}', status='{self.status}')"


class Metrics(BaseModel):
    """System metrics and statistics."""
    period: str
    total_tasks: int
    success_rate: float
    avg_duration_ms: float
    total_cost_usd: float
    agents: Dict[str, Dict[str, Any]]

    def __repr__(self) -> str:
        return f"Metrics(period='{self.period}', total_tasks={self.total_tasks}, success_rate={self.success_rate:.2%})"