"""
GitHub Integration - GitHub API integration for workflow automation

Supports PR validation, issue management, and repository automation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import logging
import base64


logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """GitHub integration configuration"""
    token: str
    owner: str
    repo: str
    base_url: str = "https://api.github.com"
    webhook_secret: Optional[str] = None


class GitHubIntegration:
    """GitHub API integration"""

    def __init__(self, config: Optional[GitHubConfig] = None):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.webhooks: Dict[str, str] = {}

    async def initialize(self):
        """Initialize GitHub client"""
        if not self.session:
            headers = {
                "Authorization": f"token {self.config.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            self.session = aiohttp.ClientSession(headers=headers)

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    async def configure(self, config: GitHubConfig):
        """Update configuration"""
        self.config = config
        if self.session:
            await self.cleanup()
            await self.initialize()

    async def test_connection(self) -> Dict[str, Any]:
        """Test GitHub connection"""
        try:
            url = f"{self.config.base_url}/user"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "user": data.get("login"),
                        "rate_limit": response.headers.get("X-RateLimit-Remaining")
                    }
                else:
                    return {
                        "success": False,
                        "status": response.status,
                        "error": await response.text()
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def validate_pr(self, pr_number: int) -> Dict[str, Any]:
        """Validate a pull request"""
        # Get PR details
        pr = await self.get_pr(pr_number)
        if not pr:
            return {"valid": False, "error": "PR not found"}

        validation_results = {
            "pr_number": pr_number,
            "title": pr["title"],
            "checks": [],
            "valid": True
        }

        # Check PR status
        checks = await self.get_pr_checks(pr_number)
        validation_results["checks"] = checks

        # Check for required approvals
        reviews = await self.get_pr_reviews(pr_number)
        approved = any(r["state"] == "APPROVED" for r in reviews)
        validation_results["approved"] = approved

        # Check merge conflicts
        validation_results["mergeable"] = pr.get("mergeable", False)

        # Overall validation
        validation_results["valid"] = (
            approved and
            validation_results["mergeable"] and
            all(check["conclusion"] == "success" for check in checks)
        )

        return validation_results

    async def get_pr(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get pull request details"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get PR {pr_number}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting PR {pr_number}: {e}")
            return None

    async def get_pr_checks(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get PR check runs"""
        pr = await self.get_pr(pr_number)
        if not pr:
            return []

        sha = pr["head"]["sha"]
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/commits/{sha}/check-runs"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("check_runs", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting PR checks: {e}")
            return []

    async def get_pr_reviews(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get PR reviews"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/reviews"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting PR reviews: {e}")
            return []

    async def create_issue(self, title: str, body: str,
                          labels: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Create GitHub issue"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/issues"

        data = {
            "title": title,
            "body": body
        }
        if labels:
            data["labels"] = labels

        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 201:
                    issue = await response.json()
                    logger.info(f"Created issue #{issue['number']}: {title}")
                    return issue
                else:
                    logger.error(f"Failed to create issue: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error creating issue: {e}")
            return None

    async def comment_on_issue(self, issue_number: int, comment: str) -> bool:
        """Add comment to issue or PR"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/issues/{issue_number}/comments"

        try:
            async with self.session.post(url, json={"body": comment}) as response:
                if response.status == 201:
                    logger.info(f"Added comment to issue #{issue_number}")
                    return True
                else:
                    logger.error(f"Failed to comment: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error commenting: {e}")
            return False

    async def create_pr_status(self, sha: str, state: str,
                              context: str, description: str) -> bool:
        """Create status for commit"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/statuses/{sha}"

        data = {
            "state": state,  # pending, success, error, failure
            "context": context,
            "description": description
        }

        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 201:
                    logger.info(f"Created status for {sha}: {state}")
                    return True
                else:
                    logger.error(f"Failed to create status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error creating status: {e}")
            return False

    async def create_webhook(self, events: List[str],
                           url: str, secret: Optional[str] = None) -> Optional[str]:
        """Create webhook for repository"""
        webhook_url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/hooks"

        config = {
            "url": url,
            "content_type": "json"
        }
        if secret:
            config["secret"] = secret

        data = {
            "config": config,
            "events": events,
            "active": True
        }

        try:
            async with self.session.post(webhook_url, json=data) as response:
                if response.status == 201:
                    webhook = await response.json()
                    webhook_id = str(webhook["id"])
                    self.webhooks[webhook_id] = url
                    logger.info(f"Created webhook {webhook_id}")
                    return webhook_id
                else:
                    logger.error(f"Failed to create webhook: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            return None

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete webhook"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/hooks/{webhook_id}"

        try:
            async with self.session.delete(url) as response:
                if response.status == 204:
                    self.webhooks.pop(webhook_id, None)
                    logger.info(f"Deleted webhook {webhook_id}")
                    return True
                else:
                    logger.error(f"Failed to delete webhook: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return False

    async def get_file_content(self, path: str, ref: str = "main") -> Optional[str]:
        """Get file content from repository"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/contents/{path}"
        params = {"ref": ref}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return content
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return None

    async def create_or_update_file(self, path: str, content: str,
                                   message: str, branch: str = "main") -> bool:
        """Create or update file in repository"""
        url = f"{self.config.base_url}/repos/{self.config.owner}/{self.config.repo}/contents/{path}"

        # Check if file exists
        existing = await self.get_file_content(path, branch)
        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }

        # If updating, need the SHA
        if existing is not None:
            params = {"ref": branch}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    file_data = await response.json()
                    data["sha"] = file_data["sha"]

        try:
            async with self.session.put(url, json=data) as response:
                if response.status in [200, 201]:
                    logger.info(f"Updated file {path}")
                    return True
                else:
                    logger.error(f"Failed to update file: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error updating file: {e}")
            return False

    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle GitHub events"""
        handlers = {
            "pull_request": self._handle_pr_event,
            "issue": self._handle_issue_event,
            "push": self._handle_push_event,
            "workflow_run": self._handle_workflow_event
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(data)
        else:
            logger.debug(f"No handler for event type: {event_type}")

    async def _handle_pr_event(self, data: Dict[str, Any]):
        """Handle pull request events"""
        action = data.get("action")
        pr = data.get("pull_request", {})
        pr_number = pr.get("number")

        if action == "opened":
            # Auto-validate new PR
            validation = await self.validate_pr(pr_number)
            if not validation["valid"]:
                await self.comment_on_issue(
                    pr_number,
                    f"⚠️ Validation failed: {validation.get('error', 'Unknown error')}"
                )

    async def _handle_issue_event(self, data: Dict[str, Any]):
        """Handle issue events"""
        pass  # Implement as needed

    async def _handle_push_event(self, data: Dict[str, Any]):
        """Handle push events"""
        pass  # Implement as needed

    async def _handle_workflow_event(self, data: Dict[str, Any]):
        """Handle workflow run events"""
        pass  # Implement as needed

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "configured": self.config is not None,
            "connected": self.session is not None,
            "webhooks": len(self.webhooks),
            "repo": f"{self.config.owner}/{self.config.repo}" if self.config else None
        }