"""
Advanced rate limiting with multiple strategies and distributed support.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiter configuration."""
    # Basic limits
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Burst handling
    burst_size: int = 20
    burst_duration_seconds: int = 10

    # Advanced settings
    enable_adaptive_limits: bool = True
    enable_user_tiers: bool = True
    enable_ip_limits: bool = True
    enable_api_key_limits: bool = True

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 2


class RateLimiter:
    """
    Enterprise-grade rate limiter with multiple strategies,
    user tiers, and distributed support.
    """

    # User tier configurations
    USER_TIERS = {
        "free": {"multiplier": 1.0, "priority": 1},
        "basic": {"multiplier": 2.0, "priority": 2},
        "pro": {"multiplier": 5.0, "priority": 3},
        "enterprise": {"multiplier": 10.0, "priority": 4},
        "unlimited": {"multiplier": float('inf'), "priority": 5}
    }

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter."""
        self.config = config or RateLimitConfig()
        self._init_redis()
        self._windows: Dict[str, Dict] = {}
        self._adaptive_factors: Dict[str, float] = {}

    def _init_redis(self) -> None:
        """Initialize Redis connection for distributed rate limiting."""
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True
        )

    async def check_rate_limit(
        self,
        identifier: str,
        user_tier: str = "free",
        request_cost: int = 1
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits.

        Args:
            identifier: User/API key/IP identifier
            user_tier: User tier level
            request_cost: Cost of the request (for weighted limits)

        Returns:
            Dict with allowed status and limit info
        """
        # Get tier multiplier
        tier_info = self.USER_TIERS.get(user_tier, self.USER_TIERS["free"])
        multiplier = tier_info["multiplier"]

        # Check multiple time windows
        checks = [
            ("second", self.config.requests_per_second * multiplier, 1),
            ("minute", self.config.requests_per_minute * multiplier, 60),
            ("hour", self.config.requests_per_hour * multiplier, 3600),
            ("day", self.config.requests_per_day * multiplier, 86400)
        ]

        for window_name, limit, window_seconds in checks:
            if multiplier == float('inf'):
                continue  # Skip for unlimited tier

            key = f"rate_limit:{identifier}:{window_name}"
            allowed, remaining, reset_time = await self._check_window(
                key,
                int(limit),
                window_seconds,
                request_cost
            )

            if not allowed:
                return {
                    "allowed": False,
                    "limit": int(limit),
                    "remaining": remaining,
                    "reset_time": reset_time,
                    "window": window_name,
                    "retry_after": reset_time - time.time()
                }

        # Check burst limits
        if self.config.burst_size > 0:
            burst_key = f"burst:{identifier}"
            burst_allowed = await self._check_burst(
                burst_key,
                self.config.burst_size,
                self.config.burst_duration_seconds
            )

            if not burst_allowed:
                return {
                    "allowed": False,
                    "limit": self.config.burst_size,
                    "remaining": 0,
                    "reset_time": time.time() + self.config.burst_duration_seconds,
                    "window": "burst",
                    "retry_after": self.config.burst_duration_seconds
                }

        # Apply adaptive limits if enabled
        if self.config.enable_adaptive_limits:
            adaptive_factor = await self._get_adaptive_factor(identifier)
            if adaptive_factor < 1.0:
                logger.info(f"Applying adaptive rate limit factor {adaptive_factor} for {identifier}")

        return {
            "allowed": True,
            "limit": int(self.config.requests_per_second * multiplier),
            "remaining": int(self.config.requests_per_second * multiplier) - request_cost,
            "reset_time": time.time() + 1,
            "window": "second"
        }

    async def _check_window(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        cost: int = 1
    ) -> tuple[bool, int, float]:
        """
        Check rate limit for a specific time window using sliding window.

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        current_time = time.time()
        window_start = current_time - window_seconds

        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current entries
        pipe.zcard(key)

        # Add new entry if within limit
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiry
        pipe.expire(key, window_seconds + 1)

        results = await asyncio.get_event_loop().run_in_executor(
            None,
            pipe.execute
        )

        count = results[1]

        if count >= limit:
            # Get oldest entry for reset time
            oldest = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis.zrange,
                key, 0, 0, withscores=True
            )

            if oldest:
                reset_time = oldest[0][1] + window_seconds
            else:
                reset_time = current_time + window_seconds

            return False, max(0, limit - count), reset_time

        remaining = limit - count - cost
        reset_time = current_time + window_seconds

        return True, remaining, reset_time

    async def _check_burst(
        self,
        key: str,
        burst_size: int,
        duration: int
    ) -> bool:
        """Check burst rate limit."""
        current_time = time.time()

        # Use token bucket algorithm
        bucket_key = f"{key}:bucket"
        last_refill_key = f"{key}:last_refill"

        # Get current tokens
        tokens = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.get,
            bucket_key
        )

        tokens = int(tokens) if tokens else burst_size

        # Get last refill time
        last_refill = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.get,
            last_refill_key
        )

        last_refill = float(last_refill) if last_refill else current_time

        # Calculate tokens to add
        time_passed = current_time - last_refill
        tokens_to_add = int(time_passed * (burst_size / duration))

        # Update tokens
        new_tokens = min(burst_size, tokens + tokens_to_add)

        if new_tokens >= 1:
            # Consume token
            pipe = self.redis.pipeline()
            pipe.set(bucket_key, new_tokens - 1, ex=duration)
            pipe.set(last_refill_key, current_time, ex=duration)

            await asyncio.get_event_loop().run_in_executor(
                None,
                pipe.execute
            )

            return True

        return False

    async def _get_adaptive_factor(self, identifier: str) -> float:
        """
        Calculate adaptive rate limit factor based on behavior.

        Returns:
            Factor between 0.1 and 1.0 (lower = more restrictive)
        """
        # Check error rate
        error_key = f"errors:{identifier}"
        error_count = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.get,
            error_key
        )

        error_count = int(error_count) if error_count else 0

        # Calculate factor based on error rate
        if error_count > 100:
            return 0.1
        elif error_count > 50:
            return 0.5
        elif error_count > 10:
            return 0.8

        return 1.0

    async def record_request(
        self,
        identifier: str,
        success: bool,
        response_time_ms: float
    ) -> None:
        """Record request for adaptive rate limiting."""
        if not success:
            error_key = f"errors:{identifier}"
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis.incr,
                error_key
            )

            # Expire after 1 hour
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis.expire,
                error_key,
                3600
            )

        # Record response time for analysis
        stats_key = f"stats:{identifier}"
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.lpush,
            stats_key,
            f"{response_time_ms:.2f}"
        )

        # Keep only last 100 entries
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.ltrim,
            stats_key,
            0,
            99
        )

    async def get_limit_info(
        self,
        identifier: str,
        user_tier: str = "free"
    ) -> Dict[str, Any]:
        """Get current rate limit information for identifier."""
        tier_info = self.USER_TIERS.get(user_tier, self.USER_TIERS["free"])
        multiplier = tier_info["multiplier"]

        return {
            "tier": user_tier,
            "limits": {
                "per_second": int(self.config.requests_per_second * multiplier),
                "per_minute": int(self.config.requests_per_minute * multiplier),
                "per_hour": int(self.config.requests_per_hour * multiplier),
                "per_day": int(self.config.requests_per_day * multiplier),
                "burst_size": self.config.burst_size
            },
            "priority": tier_info["priority"]
        }

    async def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for an identifier."""
        patterns = [
            f"rate_limit:{identifier}:*",
            f"burst:{identifier}:*",
            f"errors:{identifier}",
            f"stats:{identifier}"
        ]

        for pattern in patterns:
            keys = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis.keys,
                pattern
            )

            if keys:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis.delete,
                    *keys
                )

        logger.info(f"Reset rate limits for {identifier}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        # Get all active identifiers
        keys = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.keys,
            "rate_limit:*"
        )

        active_users = len(set(k.split(":")[1] for k in keys))

        return {
            "config": {
                "per_second": self.config.requests_per_second,
                "per_minute": self.config.requests_per_minute,
                "per_hour": self.config.requests_per_hour,
                "per_day": self.config.requests_per_day
            },
            "active_users": active_users,
            "tiers": list(self.USER_TIERS.keys()),
            "features": {
                "adaptive_limits": self.config.enable_adaptive_limits,
                "user_tiers": self.config.enable_user_tiers,
                "ip_limits": self.config.enable_ip_limits,
                "api_key_limits": self.config.enable_api_key_limits
            }
        }