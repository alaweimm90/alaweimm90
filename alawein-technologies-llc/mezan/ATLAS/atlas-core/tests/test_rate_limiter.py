"""
Comprehensive tests for Rate Limiting System

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atlas_core.rate_limiter import (
    RateLimitAlgorithm,
    RateLimitScope,
    RateLimitConfig,
    RateLimitResult,
    TokenBucketLimiter,
    LeakyBucketLimiter,
    FixedWindowLimiter,
    SlidingWindowLogLimiter,
    SlidingWindowCounterLimiter,
    RateLimitManager
)


class TestTokenBucketLimiter:
    """Test Token Bucket rate limiter"""

    @pytest.mark.asyncio
    async def test_token_bucket_basic(self):
        """Test basic token bucket functionality"""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)

        # Initial request should succeed
        result = await limiter.check("test-key", tokens=5)
        assert result.allowed is True
        assert result.remaining == 5

        # Another request should succeed
        result = await limiter.check("test-key", tokens=3)
        assert result.allowed is True
        assert result.remaining == 2

        # Request exceeding capacity should fail
        result = await limiter.check("test-key", tokens=5)
        assert result.allowed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test token bucket refill mechanism"""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)  # 10 tokens per second

        # Consume all tokens
        result = await limiter.check("test-key", tokens=10)
        assert result.allowed is True
        assert result.remaining == 0

        # Wait for refill
        await asyncio.sleep(0.5)  # Should refill 5 tokens

        # Check available tokens
        result = await limiter.check("test-key", tokens=4)
        assert result.allowed is True
        assert result.remaining >= 0

    @pytest.mark.asyncio
    async def test_token_bucket_burst(self):
        """Test token bucket burst capability"""
        limiter = TokenBucketLimiter(capacity=20, refill_rate=1.0)

        # Burst request
        result = await limiter.check("test-key", tokens=15)
        assert result.allowed is True
        assert result.remaining == 5

        # Another burst should fail
        result = await limiter.check("test-key", tokens=10)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_token_bucket_multiple_keys(self):
        """Test token bucket with multiple keys"""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)

        # Different keys should have separate buckets
        result1 = await limiter.check("key1", tokens=8)
        result2 = await limiter.check("key2", tokens=8)

        assert result1.allowed is True
        assert result2.allowed is True

        # Each key maintains its own state
        result1 = await limiter.check("key1", tokens=5)
        assert result1.allowed is False

        result2 = await limiter.check("key2", tokens=2)
        assert result2.allowed is True


class TestLeakyBucketLimiter:
    """Test Leaky Bucket rate limiter"""

    @pytest.mark.asyncio
    async def test_leaky_bucket_basic(self):
        """Test basic leaky bucket functionality"""
        limiter = LeakyBucketLimiter(capacity=10, leak_rate=2.0)  # 2 requests per second

        # Fill bucket
        for i in range(10):
            result = await limiter.check(f"test-key", tokens=1)
            assert result.allowed is True

        # Bucket should be full
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_leaky_bucket_leak(self):
        """Test leaky bucket leak mechanism"""
        limiter = LeakyBucketLimiter(capacity=5, leak_rate=10.0)  # Fast leak

        # Fill bucket
        for i in range(5):
            await limiter.check("test-key", tokens=1)

        # Wait for leak
        await asyncio.sleep(0.3)  # Should leak ~3 requests

        # Should be able to add more
        result = await limiter.check("test-key", tokens=2)
        assert result.allowed is True


class TestFixedWindowLimiter:
    """Test Fixed Window rate limiter"""

    @pytest.mark.asyncio
    async def test_fixed_window_basic(self):
        """Test basic fixed window functionality"""
        limiter = FixedWindowLimiter(limit=10, window=1.0)  # 10 requests per second

        # Make requests within window
        for i in range(10):
            result = await limiter.check("test-key", tokens=1)
            assert result.allowed is True

        # Next request should fail
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False
        assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_fixed_window_reset(self):
        """Test fixed window reset after window expires"""
        limiter = FixedWindowLimiter(limit=5, window=0.5)  # 5 requests per 0.5 seconds

        # Use all requests
        for i in range(5):
            await limiter.check("test-key", tokens=1)

        # Wait for window to reset
        await asyncio.sleep(0.6)

        # Should be able to make requests again
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_fixed_window_distributed(self):
        """Test fixed window with Redis (mocked)"""
        mock_redis = Mock()
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.incrby.return_value = None
        mock_redis.expire.return_value = None
        mock_redis.execute.return_value = [5]  # Current count

        limiter = FixedWindowLimiter(limit=10, window=60, redis_client=mock_redis)

        result = await limiter._check_distributed("test-key", 1)
        assert result.allowed is True
        assert result.remaining == 5


class TestSlidingWindowLogLimiter:
    """Test Sliding Window Log rate limiter"""

    @pytest.mark.asyncio
    async def test_sliding_window_log_basic(self):
        """Test basic sliding window log functionality"""
        limiter = SlidingWindowLogLimiter(limit=10, window=1.0)

        # Make requests
        for i in range(10):
            result = await limiter.check("test-key", tokens=1)
            assert result.allowed is True

        # Next request should fail
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_sliding_window_log_sliding(self):
        """Test sliding window behavior"""
        limiter = SlidingWindowLogLimiter(limit=5, window=1.0)

        # Make 3 requests
        for i in range(3):
            await limiter.check("test-key", tokens=1)
            await asyncio.sleep(0.2)

        # Total time elapsed: 0.6 seconds
        # Wait a bit more
        await asyncio.sleep(0.5)  # Total: 1.1 seconds

        # First requests should have expired
        # Should be able to make more requests
        result = await limiter.check("test-key", tokens=3)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_sliding_window_log_accuracy(self):
        """Test sliding window log maintains exact request times"""
        limiter = SlidingWindowLogLimiter(limit=3, window=1.0)

        # Make requests at specific times
        await limiter.check("test-key", tokens=1)
        await asyncio.sleep(0.3)
        await limiter.check("test-key", tokens=1)
        await asyncio.sleep(0.3)
        await limiter.check("test-key", tokens=1)

        # Should be at limit
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False

        # Wait for first request to expire
        await asyncio.sleep(0.5)

        # Should allow one more
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is True


class TestSlidingWindowCounterLimiter:
    """Test Sliding Window Counter rate limiter"""

    @pytest.mark.asyncio
    async def test_sliding_window_counter_basic(self):
        """Test basic sliding window counter functionality"""
        limiter = SlidingWindowCounterLimiter(limit=10, window=1.0)

        # Make requests
        for i in range(8):
            result = await limiter.check("test-key", tokens=1)
            assert result.allowed is True

        # Should still have capacity
        result = await limiter.check("test-key", tokens=2)
        assert result.allowed is True

        # Should be at limit
        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_sliding_window_counter_weighted(self):
        """Test sliding window counter weighted calculation"""
        limiter = SlidingWindowCounterLimiter(limit=10, window=1.0)

        # Make initial requests
        for i in range(5):
            await limiter.check("test-key", tokens=1)

        # Wait halfway through window
        await asyncio.sleep(0.5)

        # Make more requests
        for i in range(3):
            result = await limiter.check("test-key", tokens=1)
            assert result.allowed is True

        # The weighted count should consider both windows
        # This is approximate due to the hybrid nature


class TestRateLimitManager:
    """Test Rate Limit Manager"""

    @pytest.mark.asyncio
    async def test_manager_multiple_limiters(self):
        """Test manager with multiple limiters"""
        manager = RateLimitManager()

        # Add different limiters
        config1 = RateLimitConfig(
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.PER_USER,
            limit=100,
            window=60,
            refill_rate=1.67
        )
        manager.add_limiter("user_limit", config1)

        config2 = RateLimitConfig(
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.PER_IP,
            limit=50,
            window=60
        )
        manager.add_limiter("ip_limit", config2)

        # Check different limiters
        result1 = await manager.check("user_limit", user_id="user1", tokens=10)
        assert result1.allowed is True

        result2 = await manager.check("ip_limit", ip_address="192.168.1.1", tokens=5)
        assert result2.allowed is True

    @pytest.mark.asyncio
    async def test_manager_scopes(self):
        """Test different rate limit scopes"""
        manager = RateLimitManager()

        # Global scope
        config_global = RateLimitConfig(
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.GLOBAL,
            limit=1000,
            window=60
        )
        manager.add_limiter("global", config_global)

        # Per-user scope
        config_user = RateLimitConfig(
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.PER_USER,
            limit=100,
            window=60
        )
        manager.add_limiter("per_user", config_user)

        # Per-endpoint scope
        config_endpoint = RateLimitConfig(
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.PER_ENDPOINT,
            limit=50,
            window=60
        )
        manager.add_limiter("per_endpoint", config_endpoint)

        # Test different scopes generate different keys
        result_global = await manager.check("global")
        result_user1 = await manager.check("per_user", user_id="user1")
        result_user2 = await manager.check("per_user", user_id="user2")
        result_endpoint = await manager.check("per_endpoint", endpoint="/api/test")

        assert result_global.allowed is True
        assert result_user1.allowed is True
        assert result_user2.allowed is True
        assert result_endpoint.allowed is True

    @pytest.mark.asyncio
    async def test_manager_whitelist_blacklist(self):
        """Test whitelist and blacklist functionality"""
        manager = RateLimitManager()

        config = RateLimitConfig(
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.PER_IP,
            limit=10,
            window=60,
            whitelist=["192.168.1.100"],
            blacklist=["192.168.1.200"]
        )
        manager.add_limiter("ip_limiter", config)

        # Whitelisted IP should always pass
        for i in range(20):
            result = await manager.check("ip_limiter", ip_address="192.168.1.100")
            assert result.allowed is True

        # Blacklisted IP should always fail
        result = await manager.check("ip_limiter", ip_address="192.168.1.200")
        assert result.allowed is False

        # Normal IP should follow limits
        for i in range(10):
            result = await manager.check("ip_limiter", ip_address="192.168.1.50")
            assert result.allowed is True

        result = await manager.check("ip_limiter", ip_address="192.168.1.50")
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_manager_statistics(self):
        """Test rate limit statistics"""
        manager = RateLimitManager()

        config = RateLimitConfig(
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.PER_USER,
            limit=5,
            window=60
        )
        manager.add_limiter("test_limiter", config)

        # Make some requests
        for i in range(5):
            await manager.check("test_limiter", user_id="user1")

        # This should be denied
        await manager.check("test_limiter", user_id="user1")

        stats = manager.get_stats()
        assert stats['total_checks'] == 6
        assert stats['allowed'] == 5
        assert stats['denied'] == 1
        assert stats['by_scope']['per_user']['allowed'] == 5
        assert stats['by_scope']['per_user']['denied'] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self):
        """Test rate limit headers generation"""
        limiter = FixedWindowLimiter(limit=100, window=60)

        result = await limiter.check("test-key", tokens=10)

        assert result.headers is not None
        assert 'X-RateLimit-Limit' in result.headers
        assert result.headers['X-RateLimit-Limit'] == '100'
        assert 'X-RateLimit-Remaining' in result.headers
        assert result.headers['X-RateLimit-Remaining'] == '90'
        assert 'X-RateLimit-Reset' in result.headers

        # When limit is exceeded
        for i in range(90):
            await limiter.check("test-key", tokens=1)

        result = await limiter.check("test-key", tokens=1)
        assert result.allowed is False
        assert 'Retry-After' in result.headers


class TestRateLimitAlgorithms:
    """Test different rate limiting algorithms"""

    @pytest.mark.asyncio
    async def test_algorithm_comparison(self):
        """Compare different algorithms under same conditions"""
        algorithms = {
            "token_bucket": TokenBucketLimiter(capacity=10, refill_rate=10.0),
            "leaky_bucket": LeakyBucketLimiter(capacity=10, leak_rate=10.0),
            "fixed_window": FixedWindowLimiter(limit=10, window=1.0),
            "sliding_log": SlidingWindowLogLimiter(limit=10, window=1.0),
            "sliding_counter": SlidingWindowCounterLimiter(limit=10, window=1.0)
        }

        results = {}

        for name, limiter in algorithms.items():
            allowed = 0
            denied = 0

            # Burst of requests
            for i in range(15):
                result = await limiter.check(f"{name}-key", tokens=1)
                if result.allowed:
                    allowed += 1
                else:
                    denied += 1

            results[name] = {"allowed": allowed, "denied": denied}

        # All algorithms should enforce the limit
        for name, counts in results.items():
            assert counts["allowed"] <= 10, f"{name} allowed too many requests"
            assert counts["denied"] >= 5, f"{name} didn't deny enough requests"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test rate limiting under concurrent requests"""
        limiter = FixedWindowLimiter(limit=100, window=1.0)

        async def make_request(key: str, request_id: int):
            result = await limiter.check(key, tokens=1)
            return result.allowed

        # Create concurrent requests
        tasks = [make_request("concurrent-key", i) for i in range(150)]
        results = await asyncio.gather(*tasks)

        # Should allow exactly 100 requests
        allowed_count = sum(1 for r in results if r)
        assert allowed_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])