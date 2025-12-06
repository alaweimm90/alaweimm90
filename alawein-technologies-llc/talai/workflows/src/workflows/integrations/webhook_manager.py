"""
Webhook Manager - Manages webhooks for event-driven workflows

Handles webhook registration, validation, and event processing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
import hmac
import hashlib
import logging
import json


logger = logging.getLogger(__name__)


@dataclass
class Webhook:
    """Webhook definition"""
    id: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    active: bool = True
    retry_count: int = 3
    timeout: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebhookManager:
    """Manages webhooks and event delivery"""

    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.delivery_stats: Dict[str, Any] = {
            "total_sent": 0,
            "total_failed": 0,
            "by_webhook": {}
        }
        self._worker_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize webhook manager"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Start delivery worker
        self._worker_task = asyncio.create_task(self._delivery_worker())
        logger.info("Webhook manager initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)

        if self.session:
            await self.session.close()
            self.session = None

    def register_webhook(self, webhook: Webhook) -> str:
        """Register a new webhook"""
        self.webhooks[webhook.id] = webhook

        # Initialize stats
        self.delivery_stats["by_webhook"][webhook.id] = {
            "sent": 0,
            "failed": 0,
            "last_success": None,
            "last_failure": None
        }

        logger.info(f"Registered webhook {webhook.id} for events: {webhook.events}")
        return webhook.id

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False

    async def trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger webhook for event"""
        # Find matching webhooks
        matching_webhooks = [
            w for w in self.webhooks.values()
            if w.active and (event_type in w.events or "*" in w.events)
        ]

        if not matching_webhooks:
            logger.debug(f"No webhooks registered for event {event_type}")
            return

        # Queue event for delivery
        for webhook in matching_webhooks:
            await self.event_queue.put({
                "webhook": webhook,
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })

        logger.info(f"Queued event {event_type} for {len(matching_webhooks)} webhooks")

    async def _delivery_worker(self):
        """Worker to deliver webhook events"""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()

                # Deliver webhook
                await self._deliver_webhook(
                    event["webhook"],
                    event["event_type"],
                    event["data"],
                    event["timestamp"]
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Webhook delivery worker error: {e}")

    async def _deliver_webhook(self, webhook: Webhook, event_type: str,
                              data: Dict[str, Any], timestamp: str):
        """Deliver event to webhook"""
        payload = {
            "event": event_type,
            "data": data,
            "timestamp": timestamp,
            "webhook_id": webhook.id
        }

        # Add signature if secret configured
        headers = webhook.headers.copy()
        if webhook.secret:
            signature = self._generate_signature(payload, webhook.secret)
            headers["X-Webhook-Signature"] = signature

        headers["Content-Type"] = "application/json"
        headers["X-Webhook-Event"] = event_type

        # Attempt delivery with retries
        for attempt in range(webhook.retry_count):
            try:
                async with self.session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout)
                ) as response:
                    if response.status < 300:
                        # Success
                        self._record_success(webhook.id)
                        webhook.last_triggered = datetime.now()
                        logger.info(f"Delivered webhook {webhook.id} successfully")
                        return
                    else:
                        logger.warning(f"Webhook {webhook.id} returned {response.status}")

            except asyncio.TimeoutError:
                logger.warning(f"Webhook {webhook.id} timed out (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Webhook {webhook.id} delivery failed: {e}")

            # Wait before retry
            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        self._record_failure(webhook.id)
        logger.error(f"Failed to deliver webhook {webhook.id} after {webhook.retry_count} attempts")

    def _generate_signature(self, payload: Dict[str, Any],
                           secret: str) -> str:
        """Generate HMAC signature for webhook"""
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def verify_signature(self, payload: str, signature: str,
                        secret: str) -> bool:
        """Verify webhook signature"""
        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        provided = signature.replace("sha256=", "") if signature else ""
        return hmac.compare_digest(expected, provided)

    def _record_success(self, webhook_id: str):
        """Record successful delivery"""
        self.delivery_stats["total_sent"] += 1
        stats = self.delivery_stats["by_webhook"].get(webhook_id, {})
        stats["sent"] = stats.get("sent", 0) + 1
        stats["last_success"] = datetime.now().isoformat()
        self.delivery_stats["by_webhook"][webhook_id] = stats

    def _record_failure(self, webhook_id: str):
        """Record failed delivery"""
        self.delivery_stats["total_failed"] += 1
        stats = self.delivery_stats["by_webhook"].get(webhook_id, {})
        stats["failed"] = stats.get("failed", 0) + 1
        stats["last_failure"] = datetime.now().isoformat()
        self.delivery_stats["by_webhook"][webhook_id] = stats

    async def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test webhook connectivity"""
        if webhook_id not in self.webhooks:
            return {"success": False, "error": "Webhook not found"}

        webhook = self.webhooks[webhook_id]

        # Send test event
        test_payload = {
            "event": "test",
            "data": {"message": "Test webhook delivery"},
            "timestamp": datetime.now().isoformat(),
            "webhook_id": webhook.id
        }

        try:
            async with self.session.post(
                webhook.url,
                json=test_payload,
                headers=webhook.headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return {
                    "success": response.status < 300,
                    "status_code": response.status,
                    "response": await response.text()
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_webhooks(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List registered webhooks"""
        webhooks = []

        for webhook in self.webhooks.values():
            if active_only and not webhook.active:
                continue

            webhooks.append({
                "id": webhook.id,
                "url": webhook.url,
                "events": webhook.events,
                "active": webhook.active,
                "created_at": webhook.created_at.isoformat(),
                "last_triggered": webhook.last_triggered.isoformat() if webhook.last_triggered else None,
                "stats": self.delivery_stats["by_webhook"].get(webhook.id, {})
            })

        return webhooks

    def enable_webhook(self, webhook_id: str) -> bool:
        """Enable a webhook"""
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id].active = True
            logger.info(f"Enabled webhook {webhook_id}")
            return True
        return False

    def disable_webhook(self, webhook_id: str) -> bool:
        """Disable a webhook"""
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id].active = False
            logger.info(f"Disabled webhook {webhook_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        return {
            **self.delivery_stats,
            "total_webhooks": len(self.webhooks),
            "active_webhooks": sum(1 for w in self.webhooks.values() if w.active),
            "queue_size": self.event_queue.qsize()
        }