"""
Base adapter interface for AI models
"""

from abc import ABC, abstractmethod
from typing import Optional
from atlas_orchestrator.core.task import Task, TaskResult


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters

    All model adapters must implement this interface
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def execute(self, task: Task, model_id: str) -> TaskResult:
        """
        Execute a task using the specified model

        Args:
            task: The task to execute
            model_id: Specific model to use

        Returns:
            TaskResult with the response

        Raises:
            ModelError: If execution fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model_id: str) -> int:
        """
        Count tokens in text for a specific model

        Args:
            text: Text to count tokens for
            model_id: Model to use for counting

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the adapter is available (API key configured, etc.)

        Returns:
            True if available, False otherwise
        """
        pass


class ModelError(Exception):
    """Base exception for model errors"""
    pass


class ModelUnavailableError(ModelError):
    """Raised when model is not available"""
    pass


class ModelRateLimitError(ModelError):
    """Raised when rate limit is hit"""
    pass


class ModelAPIError(ModelError):
    """Raised for API errors"""
    pass
