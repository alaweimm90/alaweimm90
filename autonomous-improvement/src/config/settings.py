"""
Configuration settings for REPZ Autonomous Improvement Workflow System
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Database settings
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/repz_workflow",
        env="DATABASE_URL"
    )

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # AI/ML settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_ai_api_key: Optional[str] = Field(default=None, env="GOOGLE_AI_API_KEY")

    # REPZ system endpoints
    repz_portal_url: str = Field(default="https://app.repzcoach.com", env="REPZ_PORTAL_URL")
    repz_api_url: str = Field(default="https://api.repzcoach.com", env="REPZ_API_URL")
    repz_mobile_api_url: str = Field(default="https://mobile-api.repzcoach.com", env="REPZ_MOBILE_API_URL")

    # Workflow settings
    max_concurrent_steps: int = Field(default=5, env="MAX_CONCURRENT_STEPS")
    step_timeout_seconds: int = Field(default=3600, env="STEP_TIMEOUT_SECONDS")  # 1 hour
    workflow_retry_attempts: int = Field(default=3, env="WORKFLOW_RETRY_ATTEMPTS")

    # Monitoring settings
    metrics_collection_interval: int = Field(default=60, env="METRICS_COLLECTION_INTERVAL")  # seconds
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")

    # Security settings
    jwt_secret_key: str = Field(default="change-this-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # External integrations
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    discord_webhook_url: Optional[str] = Field(default=None, env="DISCORD_WEBHOOK_URL")
    email_smtp_server: Optional[str] = Field(default=None, env="EMAIL_SMTP_SERVER")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")

    # Cloud settings
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    azure_subscription_id: Optional[str] = Field(default=None, env="AZURE_SUBSCRIPTION_ID")

    # Docker settings
    docker_registry: str = Field(default="docker.io", env="DOCKER_REGISTRY")
    docker_username: Optional[str] = Field(default=None, env="DOCKER_USERNAME")
    docker_password: Optional[str] = Field(default=None, env="DOCKER_PASSWORD")

    # Kubernetes settings
    k8s_namespace: str = Field(default="repz-workflow", env="K8S_NAMESPACE")
    k8s_config_path: Optional[str] = Field(default=None, env="K8S_CONFIG_PATH")

    # YOLO mode settings
    enable_yolo_mode: bool = Field(default=True, env="ENABLE_YOLO_MODE")
    auto_rollback_on_failure: bool = Field(default=True, env="AUTO_ROLLBACK_ON_FAILURE")
    risk_tolerance_level: str = Field(default="high", env="RISK_TOLERANCE_LEVEL")  # low, medium, high

    # Ethical AI settings
    enable_bias_detection: bool = Field(default=True, env="ENABLE_BIAS_DETECTION")
    enable_privacy_protection: bool = Field(default=True, env="ENABLE_PRIVACY_PROTECTION")
    data_retention_days: int = Field(default=90, env="DATA_RETENTION_DAYS")

    # Performance thresholds
    max_response_time_ms: int = Field(default=5000, env="MAX_RESPONSE_TIME_MS")
    min_uptime_percentage: float = Field(default=99.9, env="MIN_UPTIME_PERCENTAGE")
    max_error_rate_percentage: float = Field(default=0.1, env="MAX_ERROR_RATE_PERCENTAGE")

    # Workflow phases
    assessment_steps: List[int] = Field(default=list(range(1, 101)), env="ASSESSMENT_STEPS")
    ideation_steps: List[int] = Field(default=list(range(101, 201)), env="IDEATION_STEPS")
    testing_steps: List[int] = Field(default=list(range(201, 301)), env="TESTING_STEPS")
    feedback_steps: List[int] = Field(default=list(range(301, 401)), env="FEEDBACK_STEPS")
    scalability_steps: List[int] = Field(default=list(range(401, 451)), env="SCALABILITY_STEPS")
    output_steps: List[int] = Field(default=list(range(451, 501)), env="OUTPUT_STEPS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings