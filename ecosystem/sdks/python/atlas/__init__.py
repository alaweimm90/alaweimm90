"""
ATLAS Python SDK

A comprehensive Python SDK for the ATLAS (Autonomous Technical Leadership & Adaptive System)
multiagent LLM orchestration platform.

Example:
    >>> from atlas import AtlasClient
    >>>
    >>> client = AtlasClient(api_key="your-api-key")
    >>> task = client.submit_task(
    ...     type="code_generation",
    ...     description="Create a REST API endpoint for user authentication"
    ... )
    >>> result = task.wait()
    >>> print(result.code)
"""

__version__ = "1.0.0"
__author__ = "ATLAS Platform"
__email__ = "support@atlas-platform.com"

from .client import AtlasClient
from .models import (
    Task,
    TaskStatus,
    TaskType,
    Agent,
    Analysis,
    Metrics,
    TaskContext,
    TaskRequirements
)
from .exceptions import (
    AtlasError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServerError
)

__all__ = [
    "AtlasClient",
    "Task",
    "TaskStatus",
    "TaskType",
    "Agent",
    "Analysis",
    "Metrics",
    "TaskContext",
    "TaskRequirements",
    "AtlasError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
]