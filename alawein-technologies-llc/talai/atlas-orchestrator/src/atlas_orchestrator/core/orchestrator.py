"""
Main Orchestrator class - the heart of ORCHEX
"""

import asyncio
from typing import List, Optional
from atlas_orchestrator.core.task import Task, TaskResult
from atlas_orchestrator.core.config import OrchestratorConfig
from atlas_orchestrator.core.router import Router, RoutingStrategy
from atlas_orchestrator.core.cost_tracker import CostTracker, BudgetExceededError, CostReport
from atlas_orchestrator.models.claude_adapter import ClaudeAdapter
from atlas_orchestrator.models.openai_adapter import OpenAIAdapter
from atlas_orchestrator.models.gemini_adapter import GeminiAdapter
from atlas_orchestrator.models.base import ModelAdapter, ModelError


class Orchestrator:
    """
    Multi-model AI Orchestrator

    Features:
    - Intelligent routing across multiple models
    - Cost optimization
    - Automatic fallback chains
    - Parallel execution
    - Real-time cost tracking
    - Budget enforcement

    Example:
        >>> orchestrator = Orchestrator()
        >>> task = Task(prompt="Explain quantum physics", task_type="explanation")
        >>> result = await orchestrator.execute(task)
        >>> print(result.content)
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[OrchestratorConfig] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    ):
        """
        Initialize orchestrator

        Args:
            anthropic_api_key: Anthropic API key
            openai_api_key: OpenAI API key
            google_api_key: Google API key
            config: Configuration object
            routing_strategy: Strategy for model selection
        """
        # Load config
        if config is None:
            config = OrchestratorConfig(
                anthropic_api_key=anthropic_api_key,
                openai_api_key=openai_api_key,
                google_api_key=google_api_key
            )
        self.config = config

        # Initialize components
        self.router = Router(config, routing_strategy)
        self.cost_tracker = CostTracker(
            max_daily_cost=config.max_daily_cost,
            max_per_request=config.max_per_request,
            alert_threshold=config.alert_threshold
        )

        # Initialize model adapters
        self.adapters: dict[str, ModelAdapter] = {
            "anthropic": ClaudeAdapter(config.anthropic_api_key),
            "openai": OpenAIAdapter(config.openai_api_key),
            "google": GeminiAdapter(config.google_api_key),
        }

    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a single task

        Args:
            task: Task to execute

        Returns:
            TaskResult

        Raises:
            BudgetExceededError: If budget would be exceeded
            ModelError: If all models fail
        """
        # Select model
        model_id = self.router.select_model(task)

        # Get fallback chain
        fallback_chain = self.router.get_fallback_chain(task, model_id)

        # Estimate cost and check budget
        estimated_cost = self.router.estimate_cost(task, model_id)
        try:
            self.cost_tracker.check_budget(estimated_cost)
        except BudgetExceededError as e:
            return TaskResult(
                task_id=task.task_id or "unknown",
                model=model_id,
                success=False,
                error=str(e)
            )

        # Try execution with fallback
        last_error = None
        for attempt, model_id in enumerate(fallback_chain):
            try:
                result = await self._execute_with_model(task, model_id)

                if result.success:
                    # Record cost
                    self.cost_tracker.record_cost(
                        model=model_id,
                        task_type=task.task_type.value,
                        cost=result.cost,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        task_id=result.task_id
                    )
                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                if attempt < len(fallback_chain) - 1:
                    # Try next in chain
                    continue
                else:
                    # All attempts failed
                    break

        # All models failed
        return TaskResult(
            task_id=task.task_id or "unknown",
            model=fallback_chain[0] if fallback_chain else "unknown",
            success=False,
            error=f"All models failed. Last error: {last_error}"
        )

    async def execute_batch(self, tasks: List[Task]) -> List[TaskResult]:
        """
        Execute multiple tasks sequentially

        Args:
            tasks: List of tasks to execute

        Returns:
            List of TaskResults
        """
        results = []
        for task in tasks:
            result = await self.execute(task)
            results.append(result)
        return results

    async def execute_parallel(self, tasks: List[Task]) -> List[TaskResult]:
        """
        Execute multiple tasks in parallel

        Args:
            tasks: List of tasks to execute

        Returns:
            List of TaskResults (same order as input)
        """
        coroutines = [self.execute(task) for task in tasks]
        results = await asyncio.gather(*coroutines)
        return list(results)

    async def _execute_with_model(self, task: Task, model_id: str) -> TaskResult:
        """Execute task with a specific model"""
        # Determine which adapter to use
        adapter = self._get_adapter_for_model(model_id)

        if not adapter:
            return TaskResult(
                task_id=task.task_id or "unknown",
                model=model_id,
                success=False,
                error=f"No adapter available for model {model_id}"
            )

        if not adapter.is_available():
            return TaskResult(
                task_id=task.task_id or "unknown",
                model=model_id,
                success=False,
                error=f"Adapter for {model_id} not available (API key not configured)"
            )

        # Execute
        return await adapter.execute(task, model_id)

    def _get_adapter_for_model(self, model_id: str) -> Optional[ModelAdapter]:
        """Get the appropriate adapter for a model"""
        if "claude" in model_id or "anthropic" in model_id:
            return self.adapters["anthropic"]
        elif "gpt" in model_id or "openai" in model_id:
            return self.adapters["openai"]
        elif "gemini" in model_id or "google" in model_id:
            return self.adapters["google"]
        return None

    # Cost tracking methods

    def get_cost_report(
        self,
        start_date=None,
        end_date=None
    ) -> CostReport:
        """Get cost report for a period"""
        return self.cost_tracker.get_report(start_date, end_date)

    def get_today_cost(self) -> float:
        """Get today's total cost"""
        return self.cost_tracker.get_today_cost()

    def get_savings_report(self) -> dict:
        """Get cost savings report"""
        return self.cost_tracker.get_savings_report()

    def set_budget(self, max_daily_cost: Optional[float] = None, max_per_request: Optional[float] = None):
        """Update budget limits"""
        if max_daily_cost is not None:
            self.cost_tracker.max_daily_cost = max_daily_cost
        if max_per_request is not None:
            self.cost_tracker.max_per_request = max_per_request

    # Model information

    def list_available_models(self) -> List[str]:
        """List all available models"""
        available = []
        for model_id, config in self.config.model_configs.items():
            adapter = self._get_adapter_for_model(model_id)
            if adapter and adapter.is_available():
                available.append(model_id)
        return available

    def get_model_info(self, model_id: str):
        """Get information about a model"""
        return self.router.get_model_info(model_id)
