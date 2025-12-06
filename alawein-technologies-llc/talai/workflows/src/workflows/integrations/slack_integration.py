"""
Slack Integration - Slack API integration for notifications

Supports notifications, interactive messages, and workflow triggers.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import logging
import json


logger = logging.getLogger(__name__)


@dataclass
class SlackConfig:
    """Slack integration configuration"""
    token: str
    channel: str
    webhook_url: Optional[str] = None
    bot_name: str = "TalAI Workflow Bot"
    icon_emoji: str = ":robot_face:"


class SlackIntegration:
    """Slack API integration"""

    def __init__(self, config: Optional[SlackConfig] = None):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize Slack client"""
        if not self.session:
            headers = {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(headers=headers)

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    async def configure(self, config: SlackConfig):
        """Update configuration"""
        self.config = config
        if self.session:
            await self.cleanup()
            await self.initialize()

    async def test_connection(self) -> Dict[str, Any]:
        """Test Slack connection"""
        try:
            url = "https://slack.com/api/auth.test"
            async with self.session.get(url) as response:
                data = await response.json()
                if data.get("ok"):
                    return {
                        "success": True,
                        "team": data.get("team"),
                        "user": data.get("user")
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", "Unknown error")
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_message(self, text: str, channel: Optional[str] = None,
                          attachments: Optional[List[Dict]] = None) -> bool:
        """Send message to Slack channel"""
        channel = channel or self.config.channel

        # Use webhook if available
        if self.config.webhook_url:
            return await self._send_via_webhook(text, attachments)

        # Use API
        url = "https://slack.com/api/chat.postMessage"

        payload = {
            "channel": channel,
            "text": text,
            "username": self.config.bot_name,
            "icon_emoji": self.config.icon_emoji
        }

        if attachments:
            payload["attachments"] = attachments

        try:
            async with self.session.post(url, json=payload) as response:
                data = await response.json()
                if data.get("ok"):
                    logger.info(f"Sent Slack message to {channel}")
                    return True
                else:
                    logger.error(f"Failed to send Slack message: {data.get('error')}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False

    async def _send_via_webhook(self, text: str,
                               attachments: Optional[List[Dict]] = None) -> bool:
        """Send message via webhook URL"""
        payload = {
            "text": text,
            "username": self.config.bot_name,
            "icon_emoji": self.config.icon_emoji
        }

        if attachments:
            payload["attachments"] = attachments

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Sent Slack webhook message")
                        return True
                    else:
                        logger.error(f"Webhook failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False

    async def send_workflow_notification(self, workflow_id: str,
                                        status: str, details: Dict[str, Any]) -> bool:
        """Send workflow status notification"""
        # Format status emoji
        status_emoji = {
            "started": "🚀",
            "completed": "✅",
            "failed": "❌",
            "warning": "⚠️"
        }.get(status, "ℹ️")

        text = f"{status_emoji} Workflow {workflow_id} {status}"

        # Create attachment with details
        attachment = {
            "color": self._get_status_color(status),
            "fields": [
                {"title": k, "value": str(v), "short": True}
                for k, v in details.items()
            ],
            "footer": "TalAI Workflow Engine",
            "ts": int(datetime.now().timestamp())
        }

        return await self.send_message(text, attachments=[attachment])

    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors = {
            "completed": "good",
            "failed": "danger",
            "warning": "warning",
            "started": "#36a64f"
        }
        return colors.get(status, "#808080")

    async def send_alert(self, title: str, message: str,
                        severity: str = "info") -> bool:
        """Send alert message"""
        emoji = {
            "critical": "🚨",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(severity, "📢")

        text = f"{emoji} *{title}*\n{message}"

        attachment = {
            "color": self._get_severity_color(severity),
            "text": message,
            "title": title
        }

        return await self.send_message(text, attachments=[attachment])

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        colors = {
            "critical": "#FF0000",
            "error": "danger",
            "warning": "warning",
            "info": "#36a64f"
        }
        return colors.get(severity, "#808080")

    async def create_interactive_message(self, text: str,
                                        actions: List[Dict[str, Any]]) -> bool:
        """Create interactive message with buttons"""
        attachment = {
            "text": text,
            "fallback": text,
            "callback_id": f"workflow_{datetime.now().timestamp()}",
            "color": "#36a64f",
            "attachment_type": "default",
            "actions": actions
        }

        return await self.send_message("Interactive Workflow Action",
                                      attachments=[attachment])

    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle Slack events"""
        handlers = {
            "workflow_started": self._handle_workflow_started,
            "workflow_completed": self._handle_workflow_completed,
            "workflow_failed": self._handle_workflow_failed,
            "job_completed": self._handle_job_completed
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(data)

    async def _handle_workflow_started(self, data: Dict[str, Any]):
        """Handle workflow started event"""
        await self.send_workflow_notification(
            data.get("workflow_id"),
            "started",
            {"Time": datetime.now().isoformat()}
        )

    async def _handle_workflow_completed(self, data: Dict[str, Any]):
        """Handle workflow completed event"""
        await self.send_workflow_notification(
            data.get("workflow_id"),
            "completed",
            {
                "Duration": data.get("duration", "N/A"),
                "Output Count": len(data.get("outputs", {}))
            }
        )

    async def _handle_workflow_failed(self, data: Dict[str, Any]):
        """Handle workflow failed event"""
        await self.send_workflow_notification(
            data.get("workflow_id"),
            "failed",
            {"Error": data.get("error", "Unknown error")}
        )

    async def _handle_job_completed(self, data: Dict[str, Any]):
        """Handle job completed event"""
        # Only notify for important jobs
        if data.get("priority") in ["CRITICAL", "HIGH"]:
            await self.send_message(
                f"Job {data.get('job_id')} completed successfully"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "configured": self.config is not None,
            "connected": self.session is not None,
            "channel": self.config.channel if self.config else None,
            "webhook_configured": bool(self.config.webhook_url) if self.config else False
        }


from datetime import datetime