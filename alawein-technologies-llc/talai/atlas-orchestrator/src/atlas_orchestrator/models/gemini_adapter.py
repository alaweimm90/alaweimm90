"""
Google Gemini adapter
"""

import asyncio
import time
import uuid
from typing import Optional

import google.generativeai as genai

from atlas_orchestrator.core.task import Task, TaskResult
from atlas_orchestrator.models.base import ModelAdapter, ModelAPIError, ModelError


class GeminiAdapter(ModelAdapter):
    """
    Adapter for Google Gemini models

    Supports:
    - Gemini Pro
    - Gemini Pro Vision (future)
    """

    def __init__(self, api_key: Optional[str] = None, request_timeout: int = 60):
        super().__init__(api_key)
        if api_key:
            genai.configure(api_key=api_key)
            self._configured = True
        else:
            self._configured = False
        self._request_timeout = request_timeout

    async def execute(self, task: Task, model_id: str) -> TaskResult:
        """Execute task with Gemini"""
        if not self.is_available():
            raise ModelAPIError("Gemini API key not configured")

        # Build prompt
        prompt = task.prompt
        if task.context:
            prompt = f"{task.context}\n\n{prompt}"

        # Execute
        start_time = time.time()
        task_id = task.task_id or str(uuid.uuid4())

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_content,
                    model_id,
                    prompt,
                    task.max_tokens,
                    task.temperature
                ),
                timeout=self._request_timeout
            )

            latency = time.time() - start_time

            # Extract content
            content = response.text if response.text else ""

            # Token counts (approximate)
            input_tokens = self.count_tokens(prompt, model_id)
            output_tokens = self.count_tokens(content, model_id)
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
                    "safety_ratings": [
                        {"category": r.category, "probability": r.probability}
                        for r in response.safety_ratings
                    ] if response.safety_ratings else []
                }
            )

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_id,
                model=model_id,
                success=False,
                error="Gemini request timed out",
                latency=time.time() - start_time
            )
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                model=model_id,
                success=False,
                error=str(e),
                latency=time.time() - start_time
            )

    def _generate_content(self, model_id: str, prompt: str, max_tokens: int, temperature: float):
        """Blocking Gemini call executed in a worker thread"""
        model = genai.GenerativeModel(model_id)
        return model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens (rough approximation)"""
        # Gemini: roughly 1 token ≈ 4 characters
        return len(text) // 4

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self._configured

    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and token counts"""
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
        }

        prices = pricing.get("gemini-pro", {"input": 0.00025, "output": 0.0005})

        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]

        return input_cost + output_cost
