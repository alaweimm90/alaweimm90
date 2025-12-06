"""
Retry Manager - Advanced retry logic with multiple strategies

Implements various retry strategies including exponential backoff,
jitter, and custom retry policies.
"""

import asyncio
import random
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"
    CUSTOM = "custom"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retry_on: Optional[list] = None  # Specific exceptions to retry
    dont_retry_on: Optional[list] = None  # Exceptions to not retry
    custom_calculator: Optional[Callable[[int], float]] = None

    def should_retry(self, retry_count: int, exception: Exception) -> bool:
        """Determine if should retry based on policy"""
        if retry_count >= self.max_retries:
            return False

        # Check exception filters
        if self.dont_retry_on:
            for exc_type in self.dont_retry_on:
                if isinstance(exception, exc_type):
                    return False

        if self.retry_on:
            for exc_type in self.retry_on:
                if isinstance(exception, exc_type):
                    return True
            return False  # If retry_on is specified, only retry those

        return True

    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** retry_count)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (retry_count + 1)
        elif self.strategy == RetryStrategy.FIBONACCI:
            delay = self._fibonacci_delay(retry_count)
        elif self.strategy == RetryStrategy.CONSTANT:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.CUSTOM and self.custom_calculator:
            delay = self.custom_calculator(retry_count)
        else:
            delay = self.base_delay

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def _fibonacci_delay(self, n: int) -> float:
        """Calculate Fibonacci delay"""
        if n <= 0:
            return self.base_delay
        elif n == 1:
            return self.base_delay * 2

        a, b = self.base_delay, self.base_delay * 2
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class RetryManager:
    """Manages retry execution with various strategies"""

    def __init__(self, default_policy: Optional[RetryPolicy] = None):
        self.default_policy = default_policy or RetryPolicy()
        self.retry_stats: Dict[str, Dict[str, Any]] = {}

    async def execute_with_retry(self,
                                func: Callable,
                                *args,
                                policy: Optional[RetryPolicy] = None,
                                task_id: Optional[str] = None,
                                **kwargs) -> Any:
        """Execute function with retry logic"""
        policy = policy or self.default_policy
        retry_count = 0
        last_exception = None
        task_id = task_id or f"task_{id(func)}"

        # Initialize stats
        self.retry_stats[task_id] = {
            "attempts": 0,
            "failures": 0,
            "success": False,
            "total_delay": 0,
            "start_time": datetime.now()
        }

        while retry_count <= policy.max_retries:
            try:
                # Log attempt
                self.retry_stats[task_id]["attempts"] += 1
                logger.info(f"Executing {task_id}, attempt {retry_count + 1}")

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success
                self.retry_stats[task_id]["success"] = True
                self.retry_stats[task_id]["end_time"] = datetime.now()
                logger.info(f"Task {task_id} succeeded on attempt {retry_count + 1}")
                return result

            except Exception as e:
                last_exception = e
                self.retry_stats[task_id]["failures"] += 1
                logger.warning(f"Task {task_id} failed on attempt {retry_count + 1}: {e}")

                # Check if should retry
                if not policy.should_retry(retry_count, e):
                    logger.error(f"Task {task_id} failed, no retry for exception type: {type(e)}")
                    break

                if retry_count < policy.max_retries:
                    # Calculate delay
                    delay = policy.get_delay(retry_count)
                    self.retry_stats[task_id]["total_delay"] += delay

                    logger.info(f"Retrying {task_id} after {delay:.2f} seconds")
                    await asyncio.sleep(delay)

                retry_count += 1

        # All retries exhausted
        self.retry_stats[task_id]["end_time"] = datetime.now()
        logger.error(f"Task {task_id} failed after {retry_count} attempts")

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Task {task_id} failed without exception")

    def get_stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics"""
        if task_id:
            return self.retry_stats.get(task_id, {})
        return self.retry_stats

    def clear_stats(self, task_id: Optional[str] = None):
        """Clear retry statistics"""
        if task_id:
            self.retry_stats.pop(task_id, None)
        else:
            self.retry_stats.clear()


# Predefined retry policies
AGGRESSIVE_RETRY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL,
    max_retries=10,
    base_delay=0.1,
    max_delay=30,
    exponential_base=2
)

GENTLE_RETRY = RetryPolicy(
    strategy=RetryStrategy.LINEAR,
    max_retries=3,
    base_delay=2,
    max_delay=10
)

NO_RETRY = RetryPolicy(
    max_retries=0
)

API_RETRY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL,
    max_retries=5,
    base_delay=1,
    max_delay=60,
    jitter=True,
    retry_on=[ConnectionError, TimeoutError]
)