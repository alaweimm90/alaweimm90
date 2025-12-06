"""
Dead letter queue for failed task handling and recovery.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """
    Dead letter queue for handling permanently failed tasks
    with recovery and analysis capabilities.
    """

    def __init__(self, redis_client, namespace: str = "talai:dlq"):
        """Initialize dead letter queue."""
        self.redis = redis_client
        self.namespace = namespace
        self.dlq_key = f"{namespace}:tasks"
        self.metadata_key = f"{namespace}:metadata"

    async def add_failed_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        error: str,
        retry_count: int
    ) -> bool:
        """Add failed task to DLQ."""
        try:
            # Store task data
            dlq_entry = {
                "task_id": task_id,
                "task_data": task_data,
                "error": error,
                "retry_count": retry_count,
                "failed_at": datetime.utcnow().isoformat(),
                "recovered": False
            }

            await self.redis.hset(self.dlq_key, task_id, dlq_entry)

            # Update metadata
            await self._update_metadata(task_data.get("name", "unknown"))

            logger.info(f"Added task {task_id} to dead letter queue")
            return True

        except Exception as e:
            logger.error(f"Failed to add task to DLQ: {e}")
            return False

    async def recover_task(self, task_id: str, requeue_func) -> bool:
        """Recover and requeue a task from DLQ."""
        try:
            # Get task from DLQ
            task_data = await self.redis.hget(self.dlq_key, task_id)

            if not task_data:
                logger.warning(f"Task {task_id} not found in DLQ")
                return False

            # Requeue task
            success = await requeue_func(task_data["task_data"])

            if success:
                # Mark as recovered
                task_data["recovered"] = True
                task_data["recovered_at"] = datetime.utcnow().isoformat()
                await self.redis.hset(self.dlq_key, task_id, task_data)

                logger.info(f"Recovered task {task_id} from DLQ")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to recover task {task_id}: {e}")
            return False

    async def bulk_recover(
        self,
        pattern: Optional[str] = None,
        limit: int = 100,
        requeue_func = None
    ) -> int:
        """Bulk recover tasks from DLQ."""
        count = 0

        # Get tasks matching pattern
        tasks = await self.get_failed_tasks(pattern, limit)

        for task in tasks:
            if await self.recover_task(task["task_id"], requeue_func):
                count += 1

        logger.info(f"Bulk recovered {count} tasks from DLQ")
        return count

    async def get_failed_tasks(
        self,
        pattern: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of failed tasks."""
        tasks = []

        # Get all tasks from DLQ
        cursor = 0
        while len(tasks) < limit:
            cursor, task_ids = await self.redis.hscan(
                self.dlq_key,
                cursor,
                match=pattern or "*",
                count=min(100, limit - len(tasks))
            )

            for task_id, task_data in task_ids.items():
                tasks.append(task_data)

            if cursor == 0:
                break

        return tasks[:limit]

    async def analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns in DLQ."""
        analysis = {
            "total_failed": 0,
            "by_error_type": {},
            "by_task_name": {},
            "by_retry_count": {},
            "failure_timeline": []
        }

        # Scan all DLQ entries
        cursor = 0
        while True:
            cursor, entries = await self.redis.hscan(self.dlq_key, cursor, count=100)

            for task_id, task_data in entries.items():
                analysis["total_failed"] += 1

                # Group by error type
                error = task_data.get("error", "unknown")
                error_type = error.split(":")[0] if ":" in error else error
                analysis["by_error_type"][error_type] = analysis["by_error_type"].get(error_type, 0) + 1

                # Group by task name
                task_name = task_data.get("task_data", {}).get("name", "unknown")
                analysis["by_task_name"][task_name] = analysis["by_task_name"].get(task_name, 0) + 1

                # Group by retry count
                retry_count = str(task_data.get("retry_count", 0))
                analysis["by_retry_count"][retry_count] = analysis["by_retry_count"].get(retry_count, 0) + 1

            if cursor == 0:
                break

        return analysis

    async def purge_old_tasks(self, days: int = 30) -> int:
        """Purge tasks older than specified days."""
        count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Scan and delete old entries
        cursor = 0
        while True:
            cursor, entries = await self.redis.hscan(self.dlq_key, cursor, count=100)

            for task_id, task_data in entries.items():
                failed_at = datetime.fromisoformat(task_data.get("failed_at"))

                if failed_at < cutoff_date:
                    await self.redis.hdel(self.dlq_key, task_id)
                    count += 1

            if cursor == 0:
                break

        logger.info(f"Purged {count} old tasks from DLQ")
        return count

    async def _update_metadata(self, task_name: str) -> None:
        """Update DLQ metadata."""
        meta_key = f"{self.metadata_key}:{task_name}"

        # Increment failure count
        await self.redis.hincrby(meta_key, "failure_count", 1)

        # Update last failure time
        await self.redis.hset(meta_key, "last_failure", datetime.utcnow().isoformat())

    async def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        total = await self.redis.hlen(self.dlq_key)

        return {
            "total_tasks": total,
            "analysis": await self.analyze_failures()
        }