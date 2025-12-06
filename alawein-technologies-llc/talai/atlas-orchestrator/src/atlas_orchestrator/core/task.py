"""
Task definitions and types for ORCHEX Orchestrator
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of tasks that can be executed"""
    CODE = "code"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    SIMPLE = "simple"
    LONG_CONTEXT = "long_context"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    MATH = "math"
    TRANSLATION = "translation"


class Task(BaseModel):
    """
    A task to be executed by the orchestrator

    Attributes:
        prompt: The main prompt/question for the AI
        task_type: Type of task (affects model selection)
        context: Optional additional context
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-1)
        fallback_chain: List of model IDs to try if primary fails
        metadata: Additional task metadata
    """
    prompt: str = Field(..., description="The prompt to send to the AI model")
    task_type: TaskType = Field(TaskType.SIMPLE, description="Type of task")
    context: Optional[str] = Field(None, description="Additional context")
    max_tokens: int = Field(1000, ge=1, le=200000, description="Max response tokens")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    fallback_chain: Optional[List[str]] = Field(None, description="Fallback model chain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Internal fields
    task_id: Optional[str] = Field(None, description="Unique task identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class TaskResult(BaseModel):
    """
    Result of task execution

    Attributes:
        task_id: ID of the task
        content: The AI's response
        model: Model that generated the response
        cost: Cost in USD
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency: Time taken in seconds
        success: Whether task succeeded
        error: Error message if failed
        metadata: Additional result metadata
    """
    task_id: str = Field(..., description="Task identifier")
    content: str = Field("", description="AI response content")
    model: str = Field(..., description="Model that generated response")
    cost: float = Field(0.0, ge=0.0, description="Cost in USD")
    input_tokens: int = Field(0, ge=0, description="Input token count")
    output_tokens: int = Field(0, ge=0, description="Output token count")
    latency: float = Field(0.0, ge=0.0, description="Latency in seconds")
    success: bool = Field(True, description="Success status")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    completed_at: datetime = Field(default_factory=datetime.now, description="Completion timestamp")

    @property
    def total_tokens(self) -> int:
        """Total tokens used"""
        return self.input_tokens + self.output_tokens

    @property
    def cost_per_token(self) -> float:
        """Cost per token"""
        if self.total_tokens == 0:
            return 0.0
        return self.cost / self.total_tokens


class ModelCapabilities(BaseModel):
    """
    Capabilities and pricing for a model

    Attributes:
        model_id: Unique model identifier
        name: Human-readable name
        provider: Provider (anthropic, openai, google)
        input_cost_per_1k: Cost per 1K input tokens (USD)
        output_cost_per_1k: Cost per 1K output tokens (USD)
        max_tokens: Maximum context window
        strengths: List of task types this model excels at
        available: Whether model is currently available
    """
    model_id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Human-readable name")
    provider: str = Field(..., description="Model provider")
    input_cost_per_1k: float = Field(..., ge=0.0, description="Input cost per 1K tokens (USD)")
    output_cost_per_1k: float = Field(..., ge=0.0, description="Output cost per 1K tokens (USD)")
    max_tokens: int = Field(..., ge=1, description="Maximum context window")
    strengths: List[TaskType] = Field(default_factory=list, description="Task types this model excels at")
    available: bool = Field(True, description="Is model available")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts"""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
