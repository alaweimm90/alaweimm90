"""
Enterprise Rate Limiting System

Production-grade rate limiting with multiple algorithms:
- Token Bucket
- Leaky Bucket
- Fixed Window Counter
- Sliding Window Log
- Sliding Window Counter
- Distributed rate limiting via Redis
- Per-user, per-IP, per-endpoint limits
- Rate limit headers support

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"


class RateLimitScope(Enum):
    """Rate limit scopes"""
    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"
    PER_USER_ENDPOINT = "per_user_endpoint"
    PER_IP_ENDPOINT = "per_ip_endpoint"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    limit: int  # Maximum requests
    window: float  # Time window in seconds
    burst_size: Optional[int] = None  # For token bucket
    refill_rate: Optional[float] = None  # For token bucket (tokens per second)
    leak_rate: Optional[float] = None  # For leaky bucket (requests per second)
    block_duration: Optional[float] = None  # How long to block after limit exceeded
    custom_key_func: Optional[callable] = None  # Custom key generation function
    whitelist: List[str] = None  # IPs or users to exclude
    blacklist: List[str] = None  # IPs or users to always block


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    headers: Dict[str, str] = None

    def __post_init__(self):
        """Generate rate limit headers"""
        self.headers = {
            'X-RateLimit-Limit': str(self.limit),
            'X-RateLimit-Remaining': str(max(0, self.remaining)),
            'X-RateLimit-Reset': str(int(self.reset_time)),
        }
        if self.retry_after is not None:
            self.headers['Retry-After'] = str(int(self.retry_after))


class RateLimiter(ABC):
    """Abstract base class for rate limiters"""

    @abstractmethod
    async def check(
        self,
        key: str,
        tokens: int = 1
    ) -> RateLimitResult:
        """Check if request is allowed"""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter"""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize token bucket limiter

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            redis_client: Redis client for distributed limiting
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.redis_client = redis_client
        self.buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = asyncio.Lock()

    async def check(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        if self.redis_client:
            return await self._check_distributed(key, tokens)
        else:
            return await self._check_local(key, tokens)

    async def _check_local(self, key: str, tokens: int) -> RateLimitResult:
        """Check using local state"""
        async with self._lock:
            current_time = time.time()

            if key not in self.buckets:
                # New bucket, start full
                self.buckets[key] = (self.capacity, current_time)

            bucket_tokens, last_update = self.buckets[key]

            # Calculate tokens to add based on time elapsed
            elapsed = current_time - last_update
            tokens_to_add = elapsed * self.refill_rate

            # Update bucket
            bucket_tokens = min(self.capacity, bucket_tokens + tokens_to_add)

            # Check if enough tokens
            if bucket_tokens >= tokens:
                # Consume tokens
                bucket_tokens -= tokens
                self.buckets[key] = (bucket_tokens, current_time)

                # Calculate reset time (when bucket will be full)
                tokens_needed = self.capacity - bucket_tokens
                reset_time = current_time + (tokens_needed / self.refill_rate)

                return RateLimitResult(
                    allowed=True,
                    limit=self.capacity,
                    remaining=int(bucket_tokens),
                    reset_time=reset_time
                )
            else:
                # Not enough tokens, calculate retry time
                tokens_needed = tokens - bucket_tokens
                retry_after = tokens_needed / self.refill_rate

                # Update bucket time but not tokens
                self.buckets[key] = (bucket_tokens, current_time)

                return RateLimitResult(
                    allowed=False,
                    limit=self.capacity,
                    remaining=0,
                    reset_time=current_time + retry_after,
                    retry_after=retry_after
                )

    async def _check_distributed(self, key: str, tokens: int) -> RateLimitResult:
        """Check using Redis for distributed limiting"""
        # Implement Redis-based token bucket
        # This would use Redis Lua scripts for atomic operations
        raise NotImplementedError("Distributed token bucket not implemented yet")

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        if self.redis_client:
            self.redis_client.delete(f"token_bucket:{key}")
        else:
            async with self._lock:
                if key in self.buckets:
                    del self.buckets[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'type': 'token_bucket',
            'capacity': self.capacity,
            'refill_rate': self.refill_rate,
            'active_buckets': len(self.buckets),
        }


class LeakyBucketLimiter(RateLimiter):
    """Leaky bucket rate limiter"""

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize leaky bucket limiter

        Args:
            capacity: Maximum requests in bucket
            leak_rate: Requests processed per second
            redis_client: Redis client for distributed limiting
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.redis_client = redis_client
        self.buckets: Dict[str, deque] = {}  # key -> queue of request times
        self._lock = asyncio.Lock()
        self._leak_task = asyncio.create_task(self._leak_loop())

    async def check(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        async with self._lock:
            current_time = time.time()

            if key not in self.buckets:
                self.buckets[key] = deque()

            bucket = self.buckets[key]

            # Remove leaked requests
            while bucket and bucket[0] < current_time - (1.0 / self.leak_rate):
                bucket.popleft()

            # Check if bucket has space
            if len(bucket) + tokens <= self.capacity:
                # Add requests to bucket
                for _ in range(tokens):
                    bucket.append(current_time)

                # Calculate when bucket will be empty
                reset_time = current_time + (len(bucket) / self.leak_rate)

                return RateLimitResult(
                    allowed=True,
                    limit=self.capacity,
                    remaining=self.capacity - len(bucket),
                    reset_time=reset_time
                )
            else:
                # Bucket is full
                retry_after = (len(bucket) + tokens - self.capacity) / self.leak_rate

                return RateLimitResult(
                    allowed=False,
                    limit=self.capacity,
                    remaining=0,
                    reset_time=current_time + (len(bucket) / self.leak_rate),
                    retry_after=retry_after
                )

    async def _leak_loop(self):
        """Background task to leak buckets"""
        while True:
            try:
                await asyncio.sleep(1.0 / self.leak_rate)

                async with self._lock:
                    current_time = time.time()
                    for key, bucket in list(self.buckets.items()):
                        # Remove leaked requests
                        while bucket and bucket[0] < current_time - (1.0 / self.leak_rate):
                            bucket.popleft()

                        # Remove empty buckets
                        if not bucket:
                            del self.buckets[key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Leak loop error: {e}")

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        async with self._lock:
            if key in self.buckets:
                del self.buckets[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_requests = sum(len(bucket) for bucket in self.buckets.values())
        return {
            'type': 'leaky_bucket',
            'capacity': self.capacity,
            'leak_rate': self.leak_rate,
            'active_buckets': len(self.buckets),
            'total_requests': total_requests,
        }


class FixedWindowLimiter(RateLimiter):
    """Fixed window counter rate limiter"""

    def __init__(
        self,
        limit: int,
        window: float,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize fixed window limiter

        Args:
            limit: Maximum requests per window
            window: Window duration in seconds
            redis_client: Redis client for distributed limiting
        """
        self.limit = limit
        self.window = window
        self.redis_client = redis_client
        self.windows: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)
        self._lock = asyncio.Lock()

    async def check(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        if self.redis_client:
            return await self._check_distributed(key, tokens)
        else:
            return await self._check_local(key, tokens)

    async def _check_local(self, key: str, tokens: int) -> RateLimitResult:
        """Check using local state"""
        async with self._lock:
            current_time = time.time()
            window_start = (current_time // self.window) * self.window

            if key not in self.windows:
                self.windows[key] = (0, window_start)

            count, stored_window = self.windows[key]

            # Check if new window
            if stored_window < window_start:
                count = 0
                stored_window = window_start

            # Check if limit exceeded
            if count + tokens <= self.limit:
                # Allow request
                self.windows[key] = (count + tokens, stored_window)

                return RateLimitResult(
                    allowed=True,
                    limit=self.limit,
                    remaining=self.limit - (count + tokens),
                    reset_time=window_start + self.window
                )
            else:
                # Deny request
                retry_after = window_start + self.window - current_time

                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    reset_time=window_start + self.window,
                    retry_after=retry_after
                )

    async def _check_distributed(self, key: str, tokens: int) -> RateLimitResult:
        """Check using Redis for distributed limiting"""
        current_time = time.time()
        window_start = (current_time // self.window) * self.window
        redis_key = f"fixed_window:{key}:{int(window_start)}"

        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incrby(redis_key, tokens)
        pipe.expire(redis_key, int(self.window) + 1)
        count = pipe.execute()[0]

        if count <= self.limit:
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=self.limit - count,
                reset_time=window_start + self.window
            )
        else:
            # Decrement back if over limit
            self.redis_client.decrby(redis_key, tokens)

            retry_after = window_start + self.window - current_time

            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_time=window_start + self.window,
                retry_after=retry_after
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        if self.redis_client:
            # Delete all windows for this key
            pattern = f"fixed_window:{key}:*"
            for redis_key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(redis_key)
        else:
            async with self._lock:
                if key in self.windows:
                    del self.windows[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'type': 'fixed_window',
            'limit': self.limit,
            'window': self.window,
            'active_windows': len(self.windows),
        }


class SlidingWindowLogLimiter(RateLimiter):
    """Sliding window log rate limiter (most accurate but memory intensive)"""

    def __init__(
        self,
        limit: int,
        window: float,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize sliding window log limiter

        Args:
            limit: Maximum requests per window
            window: Window duration in seconds
            redis_client: Redis client for distributed limiting
        """
        self.limit = limit
        self.window = window
        self.redis_client = redis_client
        self.logs: Dict[str, deque] = {}  # key -> deque of request timestamps
        self._lock = asyncio.Lock()

    async def check(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        async with self._lock:
            current_time = time.time()
            window_start = current_time - self.window

            if key not in self.logs:
                self.logs[key] = deque()

            log = self.logs[key]

            # Remove old entries outside window
            while log and log[0] < window_start:
                log.popleft()

            # Check if limit exceeded
            if len(log) + tokens <= self.limit:
                # Add new entries
                for _ in range(tokens):
                    log.append(current_time)

                # Calculate reset time (when oldest entry expires)
                if log:
                    reset_time = log[0] + self.window
                else:
                    reset_time = current_time + self.window

                return RateLimitResult(
                    allowed=True,
                    limit=self.limit,
                    remaining=self.limit - len(log),
                    reset_time=reset_time
                )
            else:
                # Calculate retry time (when enough old entries expire)
                if len(log) >= self.limit:
                    # Need to wait for oldest entry to expire
                    retry_after = log[len(log) - self.limit] + self.window - current_time
                else:
                    retry_after = 0

                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    reset_time=log[0] + self.window if log else current_time + self.window,
                    retry_after=retry_after
                )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        async with self._lock:
            if key in self.logs:
                del self.logs[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_requests = sum(len(log) for log in self.logs.values())
        return {
            'type': 'sliding_window_log',
            'limit': self.limit,
            'window': self.window,
            'active_logs': len(self.logs),
            'total_requests': total_requests,
        }


class SlidingWindowCounterLimiter(RateLimiter):
    """Sliding window counter (hybrid approach, good balance)"""

    def __init__(
        self,
        limit: int,
        window: float,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize sliding window counter limiter

        Args:
            limit: Maximum requests per window
            window: Window duration in seconds
            redis_client: Redis client for distributed limiting
        """
        self.limit = limit
        self.window = window
        self.redis_client = redis_client
        # key -> (prev_count, prev_window, curr_count, curr_window)
        self.counters: Dict[str, Tuple[int, float, int, float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        async with self._lock:
            current_time = time.time()
            window_start = (current_time // self.window) * self.window

            if key not in self.counters:
                self.counters[key] = (0, window_start - self.window, 0, window_start)

            prev_count, prev_window, curr_count, curr_window = self.counters[key]

            # Update windows if needed
            if curr_window < window_start:
                # Move to new window
                prev_count = curr_count if curr_window == window_start - self.window else 0
                prev_window = curr_window
                curr_count = 0
                curr_window = window_start

            # Calculate weighted count
            prev_weight = 1.0 - (current_time - window_start) / self.window
            weighted_count = prev_count * prev_weight + curr_count

            # Check if limit exceeded
            if weighted_count + tokens <= self.limit:
                # Allow request
                self.counters[key] = (prev_count, prev_window, curr_count + tokens, curr_window)

                return RateLimitResult(
                    allowed=True,
                    limit=self.limit,
                    remaining=int(self.limit - (weighted_count + tokens)),
                    reset_time=window_start + self.window
                )
            else:
                # Deny request
                retry_after = window_start + self.window - current_time

                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    reset_time=window_start + self.window,
                    retry_after=retry_after
                )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        async with self._lock:
            if key in self.counters:
                del self.counters[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'type': 'sliding_window_counter',
            'limit': self.limit,
            'window': self.window,
            'active_counters': len(self.counters),
        }


class RateLimitManager:
    """Manager for multiple rate limiters with different scopes and algorithms"""

    def __init__(self, redis_client: Optional[Any] = None):
        """Initialize rate limit manager"""
        self.redis_client = redis_client
        self.limiters: Dict[str, Tuple[RateLimiter, RateLimitConfig]] = {}
        self._stats = {
            'total_checks': 0,
            'allowed': 0,
            'denied': 0,
            'by_scope': defaultdict(lambda: {'allowed': 0, 'denied': 0}),
        }

    def add_limiter(self, name: str, config: RateLimitConfig) -> None:
        """Add a rate limiter"""
        # Create limiter based on algorithm
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            limiter = TokenBucketLimiter(
                capacity=config.limit,
                refill_rate=config.refill_rate or (config.limit / config.window),
                redis_client=self.redis_client
            )
        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            limiter = LeakyBucketLimiter(
                capacity=config.limit,
                leak_rate=config.leak_rate or (config.limit / config.window),
                redis_client=self.redis_client
            )
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            limiter = FixedWindowLimiter(
                limit=config.limit,
                window=config.window,
                redis_client=self.redis_client
            )
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
            limiter = SlidingWindowLogLimiter(
                limit=config.limit,
                window=config.window,
                redis_client=self.redis_client
            )
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_COUNTER:
            limiter = SlidingWindowCounterLimiter(
                limit=config.limit,
                window=config.window,
                redis_client=self.redis_client
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        self.limiters[name] = (limiter, config)
        logger.info(f"Added rate limiter '{name}': {config.algorithm.value}, {config.scope.value}")

    async def check(
        self,
        limiter_name: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        tokens: int = 1
    ) -> RateLimitResult:
        """Check rate limit"""
        if limiter_name not in self.limiters:
            # No limiter configured, allow by default
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_time=time.time()
            )

        limiter, config = self.limiters[limiter_name]

        # Check whitelist/blacklist
        if config.whitelist:
            if user_id in config.whitelist or ip_address in config.whitelist:
                return RateLimitResult(
                    allowed=True,
                    limit=config.limit,
                    remaining=config.limit,
                    reset_time=time.time() + config.window
                )

        if config.blacklist:
            if user_id in config.blacklist or ip_address in config.blacklist:
                return RateLimitResult(
                    allowed=False,
                    limit=0,
                    remaining=0,
                    reset_time=time.time() + (config.block_duration or 3600),
                    retry_after=config.block_duration or 3600
                )

        # Generate key based on scope
        key = self._generate_key(config, user_id, ip_address, endpoint)

        # Check rate limit
        result = await limiter.check(key, tokens)

        # Update stats
        self._stats['total_checks'] += 1
        if result.allowed:
            self._stats['allowed'] += 1
            self._stats['by_scope'][config.scope.value]['allowed'] += 1
        else:
            self._stats['denied'] += 1
            self._stats['by_scope'][config.scope.value]['denied'] += 1

        return result

    def _generate_key(
        self,
        config: RateLimitConfig,
        user_id: Optional[str],
        ip_address: Optional[str],
        endpoint: Optional[str]
    ) -> str:
        """Generate rate limit key based on scope"""
        if config.custom_key_func:
            return config.custom_key_func(user_id, ip_address, endpoint)

        parts = []

        if config.scope == RateLimitScope.GLOBAL:
            parts = ['global']
        elif config.scope == RateLimitScope.PER_USER:
            parts = ['user', user_id or 'anonymous']
        elif config.scope == RateLimitScope.PER_IP:
            parts = ['ip', ip_address or 'unknown']
        elif config.scope == RateLimitScope.PER_ENDPOINT:
            parts = ['endpoint', endpoint or 'unknown']
        elif config.scope == RateLimitScope.PER_USER_ENDPOINT:
            parts = ['user_endpoint', user_id or 'anonymous', endpoint or 'unknown']
        elif config.scope == RateLimitScope.PER_IP_ENDPOINT:
            parts = ['ip_endpoint', ip_address or 'unknown', endpoint or 'unknown']

        return ':'.join(parts)

    async def reset(self, limiter_name: str, **kwargs) -> None:
        """Reset rate limit for specific key"""
        if limiter_name not in self.limiters:
            return

        limiter, config = self.limiters[limiter_name]
        key = self._generate_key(config, **kwargs)
        await limiter.reset(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        limiter_stats = {}
        for name, (limiter, config) in self.limiters.items():
            limiter_stats[name] = {
                **limiter.get_stats(),
                'scope': config.scope.value,
                'limit': config.limit,
                'window': config.window,
            }

        return {
            'total_checks': self._stats['total_checks'],
            'allowed': self._stats['allowed'],
            'denied': self._stats['denied'],
            'by_scope': dict(self._stats['by_scope']),
            'limiters': limiter_stats,
        }

    def load_config(self, config_path: str) -> None:
        """Load rate limit configuration from YAML file"""
        import yaml

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        for limit_config in config_data.get('rate_limits', []):
            config = RateLimitConfig(
                algorithm=RateLimitAlgorithm(limit_config['algorithm']),
                scope=RateLimitScope(limit_config['scope']),
                limit=limit_config['limit'],
                window=limit_config['window'],
                burst_size=limit_config.get('burst_size'),
                refill_rate=limit_config.get('refill_rate'),
                leak_rate=limit_config.get('leak_rate'),
                block_duration=limit_config.get('block_duration'),
                whitelist=limit_config.get('whitelist', []),
                blacklist=limit_config.get('blacklist', [])
            )

            self.add_limiter(limit_config['name'], config)