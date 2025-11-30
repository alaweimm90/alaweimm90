"""
ATLAS Python SDK Exceptions

Custom exceptions for the ATLAS client.
"""

from typing import Any, Dict, Optional


class AtlasError(Exception):
    """Base exception for ATLAS SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(AtlasError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class RateLimitError(AtlasError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class TaskExecutionError(AtlasError):
    """Raised when task execution fails."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        agent_used: Optional[str] = None,
        errors: Optional[list] = None,
    ):
        super().__init__(message, status_code=500)
        self.task_id = task_id
        self.agent_used = agent_used
        self.errors = errors or []


class AgentNotFoundError(AtlasError):
    """Raised when specified agent is not found."""

    def __init__(self, agent_id: str):
        super().__init__(f"Agent '{agent_id}' not found", status_code=404)
        self.agent_id = agent_id


class ValidationError(AtlasError):
    """Raised when request validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, status_code=400)
        self.field = field


class ConnectionError(AtlasError):
    """Raised when connection to ATLAS server fails."""

    def __init__(self, message: str = "Failed to connect to ATLAS server"):
        super().__init__(message)


class TimeoutError(AtlasError):
    """Raised when request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)
