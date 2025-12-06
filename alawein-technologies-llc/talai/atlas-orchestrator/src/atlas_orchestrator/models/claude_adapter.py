"""
Anthropic Claude adapter
"""

import time
import uuid
from typing import Optional
from anthropic import Anthropic, AsyncAnthropic
from atlas_orchestrator.models.base import ModelAdapter, ModelError, ModelAPIError
from atlas_orchestrator.core.task import Task, TaskResult


class ClaudeAdapter(ModelAdapter):
    """
    Adapter for Anthropic Claude models

    Supports:
    - Claude 3 Opus
    - Claude Sonnet 4
    - Claude 3.5 Sonnet
    - Claude 3 Haiku
    """

    # Model ID mapping
    MODEL_MAP = {
        "claude-opus": "claude-3-opus-20240229",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
        "claude-haiku": "claude-3-haiku-20240307",
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if api_key:
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = None

    async def execute(self, task: Task, model_id: str) -> TaskResult:
        """Execute task with Claude"""
        if not self.is_available():
            raise ModelAPIError("Claude API key not configured")

        # Map model ID
        claude_model = self.MODEL_MAP.get(model_id, model_id)

        # Build prompt
        messages = [{"role": "user", "content": task.prompt}]

        # Add context if provided
        if task.context:
            messages[0]["content"] = f"{task.context}\n\n{task.prompt}"

        # Execute
        start_time = time.time()
        task_id = task.task_id or str(uuid.uuid4())

        try:
            response = await self.client.messages.create(
                model=claude_model,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                messages=messages
            )

            latency = time.time() - start_time

            # Extract content
            content = response.content[0].text if response.content else ""

            # Calculate cost (rough approximation)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Cost calculation based on model
            cost = self._calculate_cost(model_id, input_tokens, output_tokens)

            return TaskResult(
                task_id=task_id,
                content=content,
                model=model_id,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                success=True,
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                }
            )

        except Exception as e:
            return TaskResult(
                task_id=task_id,
                model=model_id,
                success=False,
                error=str(e),
                latency=time.time() - start_time
            )

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def is_available(self) -> bool:
        """Check if Claude is available"""
        return self.client is not None

    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and token counts"""
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "claude-opus": {"input": 0.015, "output": 0.075},
            "claude-sonnet-4": {"input": 0.003, "output": 0.015},
            "claude-sonnet-3.5": {"input": 0.003, "output": 0.015},
            "claude-haiku": {"input": 0.00025, "output": 0.00125},
        }

        if model_id not in pricing:
            # Default to Sonnet pricing
            model_id = "claude-sonnet-4"

        prices = pricing[model_id]
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]

        return input_cost + output_cost
