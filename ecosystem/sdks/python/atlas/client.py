"""
ATLAS Python SDK Client

The main client class for interacting with the ATLAS platform.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


class AtlasClient:
    """
    Main client for interacting with the ATLAS platform.

    Args:
        api_key: Your ATLAS API key
        api_url: ATLAS API base URL (default: https://api.atlas-platform.com/v1)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retries for failed requests (default: 3)

    Example:
        >>> client = AtlasClient(api_key="your-api-key")
        >>> task = client.submit_task(
        ...     type=TaskType.CODE_GENERATION,
        ...     description="Create a user authentication endpoint"
        ... )
        >>> result = task.wait()
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.atlas-platform.com/v1",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'atlas-sdk-python/{__import__("atlas").__version__}'
        })

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    def submit_task(
        self,
        type: Union[TaskType, str],
        description: str,
        context: Optional[TaskContext] = None,
        requirements: Optional[TaskRequirements] = None,
        **kwargs
    ) -> Task:
        """
        Submit a new task for execution.

        Args:
            type: Type of task (code_generation, code_review, refactoring, etc.)
            description: Detailed description of what the task should accomplish
            context: Additional context information
            requirements: Specific requirements for the task
            **kwargs: Additional task parameters

        Returns:
            Task object representing the submitted task

        Raises:
            ValidationError: If task parameters are invalid
            AuthenticationError: If API key is invalid
            ServerError: If server returns an error
        """
        task_type = type.value if isinstance(type, TaskType) else type

        payload = {
            'type': task_type,
            'description': description,
            **kwargs
        }

        if context:
            payload['context'] = context.dict() if hasattr(context, 'dict') else context
        if requirements:
            payload['requirements'] = requirements.dict() if hasattr(requirements, 'dict') else requirements

        response = self._make_request('POST', '/tasks', json=payload)
        data = response['data']

        return Task(
            task_id=data['task_id'],
            type=task_type,
            status=TaskStatus(data['status']),
            description=description,
            client=self
        )

    def get_task(self, task_id: str) -> Task:
        """
        Retrieve a task by ID.

        Args:
            task_id: The task ID to retrieve

        Returns:
            Task object with current status and results
        """
        response = self._make_request('GET', f'/tasks/{task_id}')
        data = response['data']

        task = Task(
            task_id=task_id,
            type=data.get('type'),
            status=TaskStatus(data['status']),
            description=data.get('description'),
            client=self
        )

        # Populate additional fields if available
        if 'result' in data:
            task._result = data['result']
        if 'agent_id' in data:
            task._agent_id = data['agent_id']
        if 'created_at' in data:
            task._created_at = data['created_at']
        if 'completed_at' in data:
            task._completed_at = data['completed_at']
        if 'cost_usd' in data:
            task._cost_usd = data['cost_usd']
        if 'error' in data:
            task._error = data['error']

        return task

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        type: Optional[TaskType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Task]:
        """
        List tasks with optional filtering.

        Args:
            status: Filter by task status
            type: Filter by task type
            limit: Maximum number of tasks to return
            offset: Pagination offset

        Returns:
            List of Task objects
        """
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status.value
        if type:
            params['type'] = type.value

        response = self._make_request('GET', '/tasks', params=params)
        data = response['data']

        tasks = []
        for task_data in data['tasks']:
            task = Task(
                task_id=task_data['task_id'],
                type=task_data.get('type'),
                status=TaskStatus(task_data['status']),
                description=task_data.get('description'),
                client=self
            )
            # Populate additional fields
            if 'agent_id' in task_data:
                task._agent_id = task_data['agent_id']
            if 'created_at' in task_data:
                task._created_at = task_data['created_at']
            if 'completed_at' in task_data:
                task._completed_at = task_data['completed_at']
            if 'cost_usd' in task_data:
                task._cost_usd = task_data['cost_usd']
            tasks.append(task)

        return tasks

    def list_agents(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Agent]:
        """
        List available agents with optional filtering.

        Args:
            provider: Filter by provider (anthropic, openai, google)
            capability: Filter by capability
            status: Filter by status (active, inactive)

        Returns:
            List of Agent objects
        """
        params = {}
        if provider:
            params['provider'] = provider
        if capability:
            params['capability'] = capability
        if status:
            params['status'] = status

        response = self._make_request('GET', '/agents', params=params)
        data = response['data']

        agents = []
        for agent_data in data['agents']:
            agent = Agent(
                agent_id=agent_data['agent_id'],
                name=agent_data['name'],
                provider=agent_data['provider'],
                model=agent_data['model'],
                status=agent_data['status'],
                capabilities=agent_data['capabilities']
            )
            agents.append(agent)

        return agents

    def start_analysis(
        self,
        repository_path: Optional[str] = None,
        repository_url: Optional[str] = None,
        branch: str = 'main',
        analysis_type: str = 'full',
        **options
    ) -> Analysis:
        """
        Start a repository analysis.

        Args:
            repository_path: Local path to repository
            repository_url: Remote repository URL
            branch: Branch to analyze
            analysis_type: Type of analysis (full, quick, etc.)
            **options: Additional analysis options

        Returns:
            Analysis object representing the analysis job
        """
        payload = {
            'branch': branch,
            'analysis_type': analysis_type,
            **options
        }

        if repository_path:
            payload['repository_path'] = repository_path
        if repository_url:
            payload['repository_url'] = repository_url

        response = self._make_request('POST', '/analyze', json=payload)
        data = response['data']

        return Analysis(
            analysis_id=data['analysis_id'],
            status=data['status'],
            client=self
        )

    def get_metrics(self, period: str = '24h') -> Metrics:
        """
        Get system metrics for the specified period.

        Args:
            period: Time period (1h, 24h, 7d, 30d)

        Returns:
            Metrics object with system statistics
        """
        response = self._make_request('GET', '/metrics', params={'period': period})
        data = response['data']

        return Metrics(
            period=data['period'],
            total_tasks=data['total_tasks'],
            success_rate=data['success_rate'],
            avg_duration_ms=data['avg_duration_ms'],
            total_cost_usd=data['total_cost_usd'],
            agents=data['agents']
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the ATLAS API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json: JSON payload

        Returns:
            Response data

        Raises:
            AuthenticationError: If authentication fails
            ValidationError: If request is invalid
            RateLimitError: If rate limit is exceeded
            ServerError: For server errors
        """
        url = f"{self.api_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout
            )

            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 400:
                raise ValidationError("Invalid request parameters")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 500:
                raise ServerError(f"Server error: {response.status_code}")

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise AtlasError(f"Request failed: {str(e)}") from e

    def close(self):
        """Close the client and cleanup resources."""
        self.session.close()
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()