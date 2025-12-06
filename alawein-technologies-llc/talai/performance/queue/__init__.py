"""
TalAI Performance: Asynchronous Task Queue System
==================================================

Production-grade task queue with Celery, priority management,
scheduling, chaining, and comprehensive monitoring.

© 2024 AlaweinOS. All rights reserved.
"""

from .task_queue import TaskQueue, TaskPriority
from .task_manager import TaskManager, TaskStatus
from .task_scheduler import TaskScheduler
from .task_monitor import TaskMonitor
from .dead_letter_queue import DeadLetterQueue
from .celery_backend import CeleryBackend

__all__ = [
    'TaskQueue',
    'TaskManager',
    'TaskScheduler',
    'TaskMonitor',
    'DeadLetterQueue',
    'CeleryBackend',
    'TaskPriority',
    'TaskStatus'
]

__version__ = "1.0.0"