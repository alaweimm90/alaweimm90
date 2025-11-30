"""
ATLAS Python SDK Client

Main client for interacting with the ATLAS API.
"""

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .types import (
    Agent,
    ExecutionResult,
    HealthStatus,
    Task,
    TaskContext,
    TaskPriority,
    TaskType,
)
from .exceptions import (
    AtlasError,
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    TaskExecutionError,
    ValidationError,
)


class AtlasClient:
    """
    Client for the ATLAS Multi-Agent Orchestration Platform.

    Args:
        base_url: Base URL of the ATLAS API server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.

    Example:
        >>> client = AtlasClient("http://localhost:3000", "your-api-key")
        >>> result = client.execute("code_generation", "Write a hello world")
        >>> print(result.output)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to ATLAS API."""
        url = f"{self.base_url}{path}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        body = json.dumps(data).encode("utf-8") if data else None

        request = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            body = json.loads(e.read().decode("utf-8")) if e.fp else {}
            error_msg = body.get("error", str(e))

            if e.code == 401:
                raise AuthenticationError(error_msg)
            elif e.code == 429:
                raise RateLimitError(error_msg)
            elif e.code == 400:
                raise ValidationError(error_msg)
            else:
                raise AtlasError(error_msg, status_code=e.code, response=body)
        except URLError as e:
            raise ConnectionError(str(e.reason))
        except Exception as e:
            raise AtlasError(str(e))

    def health(self) -> HealthStatus:
        """
        Check API health status.

        Returns:
            HealthStatus object with server status information.
        """
        response = self._request("GET", "/health")
        return HealthStatus.from_response(response)

    def list_agents(self) -> List[Agent]:
        """
        List all available agents.

        Returns:
            List of Agent objects.
        """
        response = self._request("GET", "/agents")
        agents_data = response.get("agents", [])
        return [Agent.from_response(a) for a in agents_data]

    def get_agent(self, agent_id: str) -> Agent:
        """
        Get details of a specific agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            Agent object with full details.
        """
        response = self._request("GET", f"/agents/{agent_id}")
        return Agent.from_response(response)

    def execute(
        self,
        task_type: TaskType | str,
        description: str,
        context: Optional[TaskContext] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        preferred_agent: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a task using ATLAS orchestration.

        Args:
            task_type: Type of task (e.g., "code_generation").
            description: Task description.
            context: Optional context for the task.
            priority: Task priority level.
            preferred_agent: Optional preferred agent ID.

        Returns:
            ExecutionResult with task output.

        Example:
            >>> result = client.execute(
            ...     "code_generation",
            ...     "Write a function to calculate factorial"
            ... )
            >>> print(result.output)
        """
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        task = Task(
            type=task_type,
            description=description,
            context=context,
            priority=priority,
            preferred_agent=preferred_agent,
        )

        response = self._request("POST", "/execute", task.to_dict())
        result = ExecutionResult.from_response(response)

        if not result.success:
            raise TaskExecutionError(
                "Task execution failed",
                task_id=result.task_id,
                agent_used=result.agent_used,
                errors=result.errors,
            )

        return result

    def generate(
        self,
        description: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate code using ATLAS.

        Args:
            description: What code to generate.
            language: Target programming language.
            context: Additional context.

        Returns:
            Generated code as string.

        Example:
            >>> code = client.generate(
            ...     "Write a Python function to reverse a string",
            ...     language="python"
            ... )
            >>> print(code)
        """
        response = self._request(
            "POST",
            "/generate",
            {
                "description": description,
                "language": language,
                "context": context,
            },
        )

        if not response.get("success"):
            raise TaskExecutionError(
                "Code generation failed",
                errors=response.get("errors", []),
            )

        return response.get("code", "")

    def review(
        self,
        code: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Review code using ATLAS.

        Args:
            code: Code to review.
            language: Programming language.
            context: Additional context.

        Returns:
            Code review feedback.

        Example:
            >>> review = client.review(
            ...     "def foo(): pass",
            ...     language="python"
            ... )
            >>> print(review)
        """
        response = self._request(
            "POST",
            "/review",
            {
                "code": code,
                "language": language,
                "context": context,
            },
        )

        if not response.get("success"):
            raise TaskExecutionError(
                "Code review failed",
                errors=response.get("errors", []),
            )

        return response.get("review", "")

    def explain(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Explain code using ATLAS.

        Args:
            code: Code to explain.
            language: Programming language.

        Returns:
            Code explanation.

        Example:
            >>> explanation = client.explain(
            ...     "lambda x: x * 2",
            ...     language="python"
            ... )
            >>> print(explanation)
        """
        response = self._request(
            "POST",
            "/explain",
            {
                "code": code,
                "language": language,
            },
        )

        if not response.get("success"):
            raise TaskExecutionError(
                "Code explanation failed",
                errors=response.get("errors", []),
            )

        return response.get("explanation", "")

    def chat(self, message: str) -> str:
        """
        Chat with an ATLAS agent.

        Args:
            message: Chat message.

        Returns:
            Agent response.

        Example:
            >>> response = client.chat("What is dependency injection?")
            >>> print(response)
        """
        response = self._request(
            "POST",
            "/chat",
            {"message": message},
        )

        if not response.get("success"):
            raise TaskExecutionError(
                "Chat failed",
                errors=response.get("errors", []),
            )

        return response.get("response", "")

    def reset_circuit(self, agent_id: str) -> bool:
        """
        Reset circuit breaker for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            True if reset was successful.
        """
        response = self._request("POST", f"/agents/{agent_id}/reset-circuit")
        return "message" in response
