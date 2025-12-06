"""
Configuration for ORCHEX Orchestrator
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from atlas_orchestrator.core.task import TaskType, ModelCapabilities


class OrchestratorConfig(BaseSettings):
    """
    Configuration for the orchestrator

    Can be loaded from environment variables or .env file
    """
    # API Keys
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    # Default settings
    default_model: str = Field("claude-sonnet-4", description="Default model to use")
    enable_fallback: bool = Field(True, description="Enable fallback to alternative models")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    timeout: int = Field(60, ge=1, description="Request timeout in seconds")

    # Cost limits
    max_daily_cost: float = Field(50.0, ge=0.0, description="Max daily cost in USD")
    max_per_request: float = Field(1.0, ge=0.0, description="Max cost per request in USD")
    alert_threshold: float = Field(40.0, ge=0.0, description="Cost alert threshold in USD")

    # Model configurations
    model_configs: Dict[str, ModelCapabilities] = Field(
        default_factory=lambda: DEFAULT_MODEL_CONFIGS
    )

    # Routing rules
    routing_rules: Dict[TaskType, List[str]] = Field(
        default_factory=lambda: DEFAULT_ROUTING_RULES
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "claude-opus": ModelCapabilities(
        model_id="claude-opus",
        name="Claude 3 Opus",
        provider="anthropic",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        max_tokens=200000,
        strengths=[TaskType.ANALYSIS, TaskType.REASONING, TaskType.LONG_CONTEXT],
    ),
    "claude-sonnet-4": ModelCapabilities(
        model_id="claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="anthropic",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=200000,
        strengths=[TaskType.CODE, TaskType.ANALYSIS, TaskType.REASONING],
    ),
    "claude-sonnet-3.5": ModelCapabilities(
        model_id="claude-sonnet-3.5",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=200000,
        strengths=[TaskType.CODE, TaskType.ANALYSIS],
    ),
    "claude-haiku": ModelCapabilities(
        model_id="claude-haiku",
        name="Claude 3 Haiku",
        provider="anthropic",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        max_tokens=200000,
        strengths=[TaskType.SIMPLE, TaskType.EXPLANATION],
    ),
    "gpt-4": ModelCapabilities(
        model_id="gpt-4",
        name="GPT-4",
        provider="openai",
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        max_tokens=128000,
        strengths=[TaskType.RESEARCH, TaskType.CREATIVE, TaskType.REASONING],
    ),
    "gpt-4-turbo": ModelCapabilities(
        model_id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        max_tokens=128000,
        strengths=[TaskType.RESEARCH, TaskType.ANALYSIS],
    ),
    "gpt-3.5-turbo": ModelCapabilities(
        model_id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="openai",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_tokens=16000,
        strengths=[TaskType.SIMPLE, TaskType.EXPLANATION],
    ),
    "gemini-pro": ModelCapabilities(
        model_id="gemini-pro",
        name="Gemini Pro",
        provider="google",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.0005,
        max_tokens=32000,
        strengths=[TaskType.RESEARCH, TaskType.EXPLANATION],
    ),
}

# Default routing rules: Task type -> [primary, fallback1, fallback2, ...]
DEFAULT_ROUTING_RULES = {
    TaskType.CODE: ["claude-sonnet-4", "gpt-4", "claude-opus"],
    TaskType.RESEARCH: ["gpt-4", "claude-opus", "claude-sonnet-4"],
    TaskType.ANALYSIS: ["claude-sonnet-4", "claude-opus", "gpt-4"],
    TaskType.CREATIVE: ["gpt-4", "claude-opus"],
    TaskType.SIMPLE: ["gpt-3.5-turbo", "claude-haiku", "gemini-pro"],
    TaskType.LONG_CONTEXT: ["claude-opus", "claude-sonnet-4"],
    TaskType.EXPLANATION: ["gpt-3.5-turbo", "claude-haiku", "gemini-pro"],
    TaskType.REASONING: ["claude-opus", "claude-sonnet-4", "gpt-4"],
    TaskType.MATH: ["gpt-4", "claude-sonnet-4"],
    TaskType.TRANSLATION: ["gpt-4", "claude-sonnet-4"],
}
