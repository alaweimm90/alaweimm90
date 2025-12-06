"""
Integration Hub - Central hub for managing external integrations

Coordinates all external service integrations and provides a unified interface.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import aiohttp
import json


logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of integrations"""
    GITHUB = "github"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CUSTOM = "custom"
    ZAPIER = "zapier"
    IFTTT = "ifttt"


@dataclass
class IntegrationConfig:
    """Configuration for an integration"""
    name: str
    type: IntegrationType
    enabled: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegrationHub:
    """Central hub for managing integrations"""

    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._metrics: Dict[str, Any] = {
            "requests_sent": 0,
            "requests_failed": 0,
            "events_processed": 0,
            "by_integration": {}
        }

    async def initialize(self):
        """Initialize integration hub"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        logger.info("Integration hub initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

        for integration in self.integrations.values():
            if hasattr(integration, 'cleanup'):
                await integration.cleanup()

        logger.info("Integration hub cleaned up")

    def register_integration(self, name: str, integration: Any,
                           config: IntegrationConfig):
        """Register an integration"""
        self.integrations[name] = integration
        self.configs[name] = config

        # Initialize metrics
        self._metrics["by_integration"][name] = {
            "requests": 0,
            "failures": 0,
            "events": 0
        }

        logger.info(f"Registered integration: {name} ({config.type.value})")

    async def configure_integration(self, name: str,
                                  config: Dict[str, Any]) -> bool:
        """Configure an integration"""
        if name not in self.integrations:
            return False

        integration = self.integrations[name]
        integration_config = self.configs[name]

        # Update configuration
        integration_config.settings.update(config.get("settings", {}))
        integration_config.credentials.update(config.get("credentials", {}))
        integration_config.rate_limits.update(config.get("rate_limits", {}))

        # Apply configuration to integration
        if hasattr(integration, 'configure'):
            await integration.configure(integration_config)

        logger.info(f"Configured integration: {name}")
        return True

    async def send_event(self, event_type: str, data: Dict[str, Any],
                        integration: Optional[str] = None):
        """Send event to integrations"""
        self._metrics["events_processed"] += 1

        # Send to specific integration
        if integration:
            if integration in self.integrations:
                await self._send_to_integration(integration, event_type, data)
        else:
            # Send to all enabled integrations
            tasks = []
            for name, config in self.configs.items():
                if config.enabled:
                    task = self._send_to_integration(name, event_type, data)
                    tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

        # Trigger event handlers
        await self._trigger_handlers(event_type, data)

    async def _send_to_integration(self, name: str, event_type: str,
                                  data: Dict[str, Any]):
        """Send event to specific integration"""
        try:
            integration = self.integrations[name]
            self._metrics["by_integration"][name]["events"] += 1

            # Check if integration handles this event type
            if hasattr(integration, 'handle_event'):
                await integration.handle_event(event_type, data)
            elif hasattr(integration, 'send'):
                await integration.send(data)

            logger.debug(f"Sent event {event_type} to {name}")

        except Exception as e:
            self._metrics["by_integration"][name]["failures"] += 1
            logger.error(f"Failed to send event to {name}: {e}")

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event: {event_type}")

    async def _trigger_handlers(self, event_type: str, data: Dict[str, Any]):
        """Trigger registered event handlers"""
        handlers = self.event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

    async def create_workflow_trigger(self, integration: str,
                                     trigger_config: Dict[str, Any]) -> str:
        """Create workflow trigger from integration"""
        if integration not in self.integrations:
            raise ValueError(f"Integration {integration} not found")

        integration_obj = self.integrations[integration]

        # Create trigger based on integration type
        if hasattr(integration_obj, 'create_trigger'):
            trigger_id = await integration_obj.create_trigger(trigger_config)
            logger.info(f"Created trigger {trigger_id} for {integration}")
            return trigger_id

        raise NotImplementedError(f"Integration {integration} doesn't support triggers")

    async def test_integration(self, name: str) -> Dict[str, Any]:
        """Test integration connection"""
        if name not in self.integrations:
            return {"success": False, "error": "Integration not found"}

        integration = self.integrations[name]

        try:
            if hasattr(integration, 'test_connection'):
                result = await integration.test_connection()
            else:
                # Default test: try to send a test event
                await self._send_to_integration(name, "test", {"test": True})
                result = {"success": True}

            logger.info(f"Integration test for {name}: {result}")
            return result

        except Exception as e:
            logger.error(f"Integration test failed for {name}: {e}")
            return {"success": False, "error": str(e)}

    def get_integration(self, name: str) -> Optional[Any]:
        """Get integration by name"""
        return self.integrations.get(name)

    def list_integrations(self, enabled_only: bool = False) -> List[str]:
        """List registered integrations"""
        if enabled_only:
            return [
                name for name, config in self.configs.items()
                if config.enabled
            ]
        return list(self.integrations.keys())

    def enable_integration(self, name: str):
        """Enable an integration"""
        if name in self.configs:
            self.configs[name].enabled = True
            logger.info(f"Enabled integration: {name}")

    def disable_integration(self, name: str):
        """Disable an integration"""
        if name in self.configs:
            self.configs[name].enabled = False
            logger.info(f"Disabled integration: {name}")

    async def batch_send(self, events: List[Dict[str, Any]],
                        integration: str):
        """Send batch of events to integration"""
        if integration not in self.integrations:
            return

        integration_obj = self.integrations[integration]

        if hasattr(integration_obj, 'send_batch'):
            await integration_obj.send_batch(events)
            self._metrics["by_integration"][integration]["events"] += len(events)
        else:
            # Send individually
            for event in events:
                await self._send_to_integration(
                    integration,
                    event.get("type", "default"),
                    event
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        return {
            **self._metrics,
            "active_integrations": len([
                c for c in self.configs.values() if c.enabled
            ]),
            "total_integrations": len(self.integrations)
        }

    def get_integration_status(self, name: str) -> Dict[str, Any]:
        """Get status of specific integration"""
        if name not in self.integrations:
            return {"error": "Integration not found"}

        config = self.configs[name]
        metrics = self._metrics["by_integration"].get(name, {})

        status = {
            "name": name,
            "type": config.type.value,
            "enabled": config.enabled,
            "metrics": metrics
        }

        # Get integration-specific status
        integration = self.integrations[name]
        if hasattr(integration, 'get_status'):
            status["details"] = integration.get_status()

        return status

    async def sync_integrations(self):
        """Synchronize state across integrations"""
        tasks = []

        for name, integration in self.integrations.items():
            if self.configs[name].enabled:
                if hasattr(integration, 'sync'):
                    task = integration.sync()
                    tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Synchronized integrations")

    def export_config(self) -> Dict[str, Any]:
        """Export integration configurations"""
        return {
            name: {
                "type": config.type.value,
                "enabled": config.enabled,
                "settings": config.settings,
                "rate_limits": config.rate_limits,
                "metadata": config.metadata
            }
            for name, config in self.configs.items()
        }

    def import_config(self, config_data: Dict[str, Any]):
        """Import integration configurations"""
        for name, config in config_data.items():
            if name in self.configs:
                self.configs[name].enabled = config.get("enabled", True)
                self.configs[name].settings.update(config.get("settings", {}))
                self.configs[name].rate_limits.update(config.get("rate_limits", {}))
                self.configs[name].metadata.update(config.get("metadata", {}))

        logger.info(f"Imported configurations for {len(config_data)} integrations")