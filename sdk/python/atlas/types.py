"""
ATLAS Python SDK Types

Type definitions for the ATLAS API.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    """Task types supported by ATLAS."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    EXPLANATION = "explanation"
    CHAT = "chat"


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """Agent availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class TaskContext:
    """Context for task execution."""

    files: Optional[List[str]] = None
    code_snippet: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    additional_context: Optional[str] = None


@dataclass
class Task:
    """Task definition for ATLAS execution."""

    type: TaskType
    description: str
    id: Optional[str] = None
    context: Optional[TaskContext] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    preferred_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "type": self.type.value,
            "description": self.description,
            "context": {
                "files": self.context.files if self.context else None,
                "codeSnippet": self.context.code_snippet if self.context else None,
                "language": self.context.language if self.context else None,
                "framework": self.context.framework if self.context else None,
                "additionalContext": self.context.additional_context if self.context else None,
            }
            if self.context
            else {},
            "priority": self.priority.value,
            "preferredAgent": self.preferred_agent,
        }


@dataclass
class ExecutionResult:
    """Result of task execution."""

    success: bool
    task_id: Optional[str] = None
    agent_used: Optional[str] = None
    attempts: int = 0
    output: Optional[str] = None
    tokens_used: Optional[int] = None
    latency: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from API response."""
        result = data.get("result", {})
        return cls(
            success=data.get("success", False),
            task_id=data.get("taskId"),
            agent_used=data.get("agentUsed"),
            attempts=data.get("attempts", 0),
            output=result.get("output"),
            tokens_used=result.get("tokensUsed"),
            latency=result.get("latency"),
            errors=data.get("errors", []),
        )


@dataclass
class AgentMetrics:
    """Agent performance metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    total_tokens: int = 0


@dataclass
class Agent:
    """Agent information."""

    id: str
    name: str
    provider: str
    model: str
    capabilities: List[str]
    status: AgentStatus
    configured: bool
    metrics: Optional[AgentMetrics] = None
    registered_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "Agent":
        """Create from API response."""
        metrics_data = data.get("metrics", {})
        return cls(
            id=data["id"],
            name=data["name"],
            provider=data["provider"],
            model=data["model"],
            capabilities=data.get("capabilities", []),
            status=AgentStatus(data.get("status", "unavailable")),
            configured=data.get("configured", False),
            metrics=AgentMetrics(
                total_requests=metrics_data.get("totalRequests", 0),
                successful_requests=metrics_data.get("successfulRequests", 0),
                failed_requests=metrics_data.get("failedRequests", 0),
                avg_latency=metrics_data.get("avgLatency", 0.0),
                total_tokens=metrics_data.get("totalTokens", 0),
            )
            if metrics_data
            else None,
        )


@dataclass
class HealthStatus:
    """API health status."""

    status: str
    timestamp: datetime
    agents_total: int
    agents_available: int
    agents_configured: int
    circuits_healthy: int
    circuits_degraded: int
    circuits_unhealthy: int

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "HealthStatus":
        """Create from API response."""
        agents = data.get("agents", {})
        circuits = data.get("circuits", {})
        return cls(
            status=data.get("status", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp", "")),
            agents_total=agents.get("total", 0),
            agents_available=agents.get("available", 0),
            agents_configured=agents.get("configured", 0),
            circuits_healthy=circuits.get("healthy", 0),
            circuits_degraded=circuits.get("degraded", 0),
            circuits_unhealthy=circuits.get("unhealthy", 0),
        )
