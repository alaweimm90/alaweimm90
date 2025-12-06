"""
Custom Integration - Framework for custom integrations

Provides base classes and utilities for creating custom integrations.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import logging
import aiohttp


logger = logging.getLogger(__name__)


class IntegrationAdapter(ABC):
    """Abstract base class for integration adapters"""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to service"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from service"""
        pass

    @abstractmethod
    async def send(self, data: Dict[str, Any]) -> bool:
        """Send data to service"""
        pass

    @abstractmethod
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive data from service"""
        pass

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test service connection"""
        pass


class CustomIntegration:
    """Framework for custom integrations"""

    def __init__(self, name: str, adapter: Optional[IntegrationAdapter] = None):
        self.name = name
        self.adapter = adapter
        self.transformers: Dict[str, Callable] = {}
        self.validators: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.connected = False

    async def initialize(self):
        """Initialize custom integration"""
        if self.adapter:
            self.connected = await self.adapter.connect()
            logger.info(f"Initialized custom integration: {self.name}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.adapter and self.connected:
            await self.adapter.disconnect()
            self.connected = False

    def register_transformer(self, name: str, transformer: Callable):
        """Register data transformer"""
        self.transformers[name] = transformer
        logger.info(f"Registered transformer {name} for {self.name}")

    def register_validator(self, name: str, validator: Callable):
        """Register data validator"""
        self.validators[name] = validator
        logger.info(f"Registered validator {name} for {self.name}")

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def send_data(self, data: Dict[str, Any],
                       transform: Optional[str] = None,
                       validate: Optional[str] = None) -> bool:
        """Send data with optional transformation and validation"""
        # Validate data
        if validate and validate in self.validators:
            if not self.validators[validate](data):
                logger.error(f"Validation {validate} failed")
                return False

        # Transform data
        if transform and transform in self.transformers:
            data = self.transformers[transform](data)

        # Send via adapter
        if self.adapter and self.connected:
            return await self.adapter.send(data)

        return False

    async def receive_data(self, transform: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Receive data with optional transformation"""
        if not self.adapter or not self.connected:
            return None

        data = await self.adapter.receive()

        if data and transform and transform in self.transformers:
            data = self.transformers[transform](data)

        return data

    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle integration events"""
        handlers = self.event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler failed in {self.name}: {e}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test integration connection"""
        if self.adapter:
            return await self.adapter.test_connection()
        return {"success": False, "error": "No adapter configured"}

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "name": self.name,
            "connected": self.connected,
            "has_adapter": self.adapter is not None,
            "transformers": list(self.transformers.keys()),
            "validators": list(self.validators.keys()),
            "event_handlers": list(self.event_handlers.keys())
        }


class HTTPAdapter(IntegrationAdapter):
    """HTTP/REST API adapter"""

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Establish HTTP session"""
        try:
            self.session = aiohttp.ClientSession(headers=self.headers)
            return True
        except Exception as e:
            logger.error(f"Failed to create HTTP session: {e}")
            return False

    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def send(self, data: Dict[str, Any]) -> bool:
        """Send data via HTTP POST"""
        if not self.session:
            return False

        try:
            url = f"{self.base_url}/webhook"
            async with self.session.post(url, json=data) as response:
                return response.status < 300
        except Exception as e:
            logger.error(f"HTTP send failed: {e}")
            return False

    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive data via HTTP GET"""
        if not self.session:
            return None

        try:
            url = f"{self.base_url}/events"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"HTTP receive failed: {e}")

        return None

    async def test_connection(self) -> Dict[str, Any]:
        """Test HTTP connection"""
        if not self.session:
            return {"success": False, "error": "Session not established"}

        try:
            async with self.session.get(self.base_url) as response:
                return {
                    "success": response.status < 300,
                    "status_code": response.status,
                    "url": self.base_url
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ZapierAdapter(IntegrationAdapter):
    """Zapier webhook adapter"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Initialize Zapier connection"""
        self.session = aiohttp.ClientSession()
        return True

    async def disconnect(self):
        """Disconnect from Zapier"""
        if self.session:
            await self.session.close()
            self.session = None

    async def send(self, data: Dict[str, Any]) -> bool:
        """Send data to Zapier webhook"""
        if not self.session:
            return False

        try:
            async with self.session.post(self.webhook_url, json=data) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Zapier send failed: {e}")
            return False

    async def receive(self) -> Optional[Dict[str, Any]]:
        """Zapier is push-only, no receive"""
        return None

    async def test_connection(self) -> Dict[str, Any]:
        """Test Zapier webhook"""
        test_data = {"test": True, "timestamp": datetime.now().isoformat()}
        success = await self.send(test_data)
        return {"success": success, "webhook_url": self.webhook_url}


from datetime import datetime