"""
OpenAI GPT adapter
"""

import time
import uuid
from typing import Optional
from openai import AsyncOpenAI
import tiktoken
from atlas_orchestrator.models.base import ModelAdapter, ModelError, ModelAPIError
from atlas_orchestrator.core.task import Task, TaskResult


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI models

    Supports:
    - GPT-4
    - GPT-4 Turbo
    - GPT-3.5 Turbo
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = None

    async def execute(self, task: Task, model_id: str) -> TaskResult:
        """Execute task with OpenAI"""
        if not self.is_available():
            raise ModelAPIError("OpenAI API key not configured")

        # Build messages
        messages = []
        if task.context:
            messages.append({"role": "system", "content": task.context})
        messages.append({"role": "user", "content": task.prompt})

        # Execute
        start_time = time.time()
        task_id = task.task_id or str(uuid.uuid4())

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=task.max_tokens,
                temperature=task.temperature
            )

            latency = time.time() - start_time

            # Extract content
            content = response.choices[0].message.content or ""

            # Token counts and cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
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
                    "finish_reason": response.choices[0].finish_reason,
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
        """Count tokens using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model_id)
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimate
            return len(text) // 4

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.client is not None

    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and token counts"""
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

        # Match model
        prices = None
        for key in pricing:
            if key in model_id:
                prices = pricing[key]
                break

        if not prices:
            # Default to GPT-4 pricing
            prices = pricing["gpt-4"]

        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]

        return input_cost + output_cost
