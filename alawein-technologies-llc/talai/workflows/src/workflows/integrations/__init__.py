"""
Integration Hub - External service integrations and webhooks

Provides connectors for GitHub, Slack, Email, and custom integrations.
"""

from .integration_hub import IntegrationHub, IntegrationConfig
from .github_integration import GitHubIntegration
from .slack_integration import SlackIntegration
from .email_integration import EmailIntegration
from .webhook_manager import WebhookManager, Webhook
from .custom_integration import CustomIntegration, IntegrationAdapter

__all__ = [
    "IntegrationHub",
    "IntegrationConfig",
    "GitHubIntegration",
    "SlackIntegration",
    "EmailIntegration",
    "WebhookManager",
    "Webhook",
    "CustomIntegration",
    "IntegrationAdapter",
]