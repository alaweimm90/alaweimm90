"""
ATLAS Python SDK

A Python client for the ATLAS Multi-Agent Orchestration Platform.

Example usage:
    from atlas import AtlasClient

    client = AtlasClient(
        base_url="http://localhost:3000",
        api_key="your-api-key"
    )

    # Execute a task
    result = client.execute(
        task_type="code_generation",
        description="Generate a hello world function"
    )

    # Generate code
    code = client.generate(
        description="Write a Python function to calculate fibonacci",
        language="python"
    )

    # Review code
    review = client.review(code="def foo(): pass", language="python")

    # Chat
    response = client.chat("What is dependency injection?")
"""

__version__ = "0.1.0"
__author__ = "ATLAS Team"

from .client import AtlasClient
from .types import (
    Task,
    TaskType,
    TaskPriority,
    ExecutionResult,
    Agent,
    AgentStatus,
)
from .exceptions import (
    AtlasError,
    AuthenticationError,
    RateLimitError,
    TaskExecutionError,
)

__all__ = [
    "AtlasClient",
    "Task",
    "TaskType",
    "TaskPriority",
    "ExecutionResult",
    "Agent",
    "AgentStatus",
    "AtlasError",
    "AuthenticationError",
    "RateLimitError",
    "TaskExecutionError",
]
