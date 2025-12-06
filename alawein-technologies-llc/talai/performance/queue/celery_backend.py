"""
Celery backend integration for distributed task processing.
"""

import logging
from typing import Any, Dict, Optional

from celery import Celery, Task
from celery.result import AsyncResult
from kombu import Queue

logger = logging.getLogger(__name__)


class CeleryBackend:
    """
    Celery integration for distributed task processing
    with advanced features and monitoring.
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        backend_url: str = "redis://localhost:6379/1",
        app_name: str = "talai_tasks"
    ):
        """Initialize Celery backend."""
        self.app = Celery(
            app_name,
            broker=broker_url,
            backend=backend_url
        )

        # Configure Celery
        self.app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=3600,
            task_soft_time_limit=3300,
            task_acks_late=True,
            worker_prefetch_multiplier=4,
            worker_max_tasks_per_child=1000,
            broker_pool_limit=100,
            result_expires=86400,
            task_default_queue='default',
            task_queues=(
                Queue('critical', routing_key='critical'),
                Queue('high', routing_key='high'),
                Queue('default', routing_key='default'),
                Queue('low', routing_key='low'),
                Queue('background', routing_key='background'),
            ),
            task_routes={
                'tasks.critical.*': {'queue': 'critical'},
                'tasks.high.*': {'queue': 'high'},
                'tasks.low.*': {'queue': 'low'},
                'tasks.background.*': {'queue': 'background'},
            },
            beat_schedule={
                # Scheduled tasks can be added here
            }
        )

    def create_task(self, name: str, func: callable) -> Task:
        """Create a Celery task."""
        return self.app.task(name=name, bind=True)(func)

    def send_task(
        self,
        task_name: str,
        args: tuple = (),
        kwargs: dict = None,
        queue: str = "default",
        priority: int = 5,
        countdown: Optional[int] = None,
        eta: Optional[datetime] = None,
        expires: Optional[int] = None
    ) -> AsyncResult:
        """Send task to Celery."""
        return self.app.send_task(
            task_name,
            args=args,
            kwargs=kwargs or {},
            queue=queue,
            priority=priority,
            countdown=countdown,
            eta=eta,
            expires=expires
        )

    def get_task_result(self, task_id: str) -> AsyncResult:
        """Get task result by ID."""
        return AsyncResult(task_id, app=self.app)

    def revoke_task(self, task_id: str, terminate: bool = False) -> None:
        """Revoke a task."""
        self.app.control.revoke(task_id, terminate=terminate)

    def get_active_tasks(self) -> Dict[str, Any]:
        """Get all active tasks."""
        inspect = self.app.control.inspect()
        return {
            "active": inspect.active(),
            "scheduled": inspect.scheduled(),
            "reserved": inspect.reserved()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get Celery statistics."""
        inspect = self.app.control.inspect()
        return {
            "stats": inspect.stats(),
            "active_queues": inspect.active_queues(),
            "registered_tasks": inspect.registered()
        }