"""
Multi-level Cache Manager with intelligent routing and fallback.

This module implements a sophisticated multi-level caching strategy
with L1 (in-memory), L2 (Redis), and L3 (persistent) caching.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import redis
from redis.sentinel import Sentinel

from .cache_analytics import CacheAnalytics
from .memory_cache import InMemoryCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_PERSISTENT = "persistent"


@dataclass
class CacheConfig:
    """Configuration for multi-level cache."""
    memory_size_mb: int = 512
    memory_ttl_seconds: int = 300
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_sentinel_nodes: Optional[List[Tuple[str, int]]] = None
    redis_sentinel_service: Optional[str] = None
    redis_ttl_seconds: int = 3600
    enable_compression: bool = True
    enable_encryption: bool = False
    max_key_length: int = 250
    namespace: str = "talai"
    enable_metrics: bool = True
    cache_warming_enabled: bool = True
    invalidation_strategy: str = "ttl"  # ttl, lru, lfu, manual


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    level: CacheLevel
    size_bytes: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    checksum: Optional[str] = None


class CacheManager:
    """
    Enterprise-grade multi-level cache manager with intelligent routing,
    automatic fallback, and comprehensive monitoring.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the cache manager with configuration."""
        self.config = config or CacheConfig()
        self._initialize_caches()
        self._initialize_analytics()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []
        logger.info(f"CacheManager initialized with namespace: {self.config.namespace}")

    def _initialize_caches(self) -> None:
        """Initialize all cache levels."""
        # L1: In-memory cache
        self.l1_cache = InMemoryCache(
            max_size_mb=self.config.memory_size_mb,
            ttl_seconds=self.config.memory_ttl_seconds,
            enable_compression=False  # No compression for memory cache
        )

        # L2: Redis cache
        if self.config.redis_sentinel_nodes:
            # Use Redis Sentinel for high availability
            sentinel = Sentinel(self.config.redis_sentinel_nodes)
            self.redis_client = sentinel.master_for(
                self.config.redis_sentinel_service,
                socket_timeout=5.0,
                password=self.config.redis_password
            )
        else:
            # Direct Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                connection_pool=redis.ConnectionPool(
                    max_connections=100,
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db
                )
            )

        self.l2_cache = RedisCache(
            client=self.redis_client,
            namespace=self.config.namespace,
            ttl_seconds=self.config.redis_ttl_seconds,
            enable_compression=self.config.enable_compression
        )

        # L3: Persistent cache (could be disk, S3, etc.)
        # For now, using Redis with longer TTL as L3
        self.l3_cache = self.l2_cache  # Simplified for this implementation

    def _initialize_analytics(self) -> None:
        """Initialize cache analytics and monitoring."""
        if self.config.enable_metrics:
            self.analytics = CacheAnalytics(namespace=self.config.namespace)
        else:
            self.analytics = None

    def _generate_key(self, key: str, prefix: Optional[str] = None) -> str:
        """Generate a cache key with namespace and optional prefix."""
        parts = [self.config.namespace]
        if prefix:
            parts.append(prefix)
        parts.append(key)

        full_key = ":".join(parts)

        # Handle long keys by hashing
        if len(full_key) > self.config.max_key_length:
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()[:16]
            full_key = f"{self.config.namespace}:{key_hash}"

        return full_key

    async def get(
        self,
        key: str,
        default: Any = None,
        prefix: Optional[str] = None,
        skip_l1: bool = False
    ) -> Any:
        """
        Get value from cache with multi-level fallback.

        Args:
            key: Cache key
            default: Default value if not found
            prefix: Optional key prefix
            skip_l1: Skip L1 cache lookup

        Returns:
            Cached value or default
        """
        cache_key = self._generate_key(key, prefix)
        start_time = time.time()

        # Try L1 (memory) cache first
        if not skip_l1:
            value = await self.l1_cache.get(cache_key)
            if value is not None:
                self._record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
                return value

        # Try L2 (Redis) cache
        value = await self.l2_cache.get(cache_key)
        if value is not None:
            # Promote to L1 for faster access
            if not skip_l1:
                await self.l1_cache.set(cache_key, value, ttl_seconds=60)
            self._record_hit(CacheLevel.L2_REDIS, time.time() - start_time)
            return value

        # Cache miss
        self._record_miss(time.time() - start_time)
        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        prefix: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        skip_l1: bool = False
    ) -> bool:
        """
        Set value in cache across multiple levels.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            prefix: Optional key prefix
            tags: Tags for cache invalidation
            skip_l1: Skip L1 cache storage

        Returns:
            Success status
        """
        cache_key = self._generate_key(key, prefix)
        ttl = ttl_seconds or self.config.redis_ttl_seconds

        try:
            # Store in L2 (Redis) - primary storage
            success = await self.l2_cache.set(cache_key, value, ttl_seconds=ttl)

            # Store in L1 (memory) for hot data
            if not skip_l1 and success:
                memory_ttl = min(ttl, self.config.memory_ttl_seconds)
                await self.l1_cache.set(cache_key, value, ttl_seconds=memory_ttl)

            # Record tags for invalidation
            if tags and success:
                await self._store_tags(cache_key, tags)

            self._record_set()
            return success

        except Exception as e:
            logger.error(f"Error setting cache key {cache_key}: {e}")
            return False

    async def delete(
        self,
        key: str,
        prefix: Optional[str] = None,
        cascade: bool = True
    ) -> bool:
        """
        Delete value from all cache levels.

        Args:
            key: Cache key
            prefix: Optional key prefix
            cascade: Delete from all levels

        Returns:
            Success status
        """
        cache_key = self._generate_key(key, prefix)

        try:
            # Delete from all levels
            l1_deleted = await self.l1_cache.delete(cache_key)
            l2_deleted = await self.l2_cache.delete(cache_key)

            self._record_delete()
            return l1_deleted or l2_deleted

        except Exception as e:
            logger.error(f"Error deleting cache key {cache_key}: {e}")
            return False

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """
        Invalidate all cache entries with specified tags.

        Args:
            tags: Set of tags to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        for tag in tags:
            tag_key = f"{self.config.namespace}:tags:{tag}"
            keys = await self._get_keys_by_tag(tag)

            for key in keys:
                if await self.delete(key, cascade=True):
                    count += 1

        logger.info(f"Invalidated {count} cache entries for tags: {tags}")
        return count

    async def clear_all(self, pattern: Optional[str] = None) -> int:
        """
        Clear all cache entries or those matching a pattern.

        Args:
            pattern: Optional key pattern

        Returns:
            Number of entries cleared
        """
        count = 0

        # Clear L1
        count += await self.l1_cache.clear(pattern)

        # Clear L2
        count += await self.l2_cache.clear(pattern)

        logger.warning(f"Cleared {count} cache entries")
        return count

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "namespace": self.config.namespace,
            "l1_stats": await self.l1_cache.get_stats(),
            "l2_stats": await self.l2_cache.get_stats(),
        }

        if self.analytics:
            stats["analytics"] = await self.analytics.get_metrics()

        return stats

    async def warm_cache(self, keys: List[str], loader: Callable) -> int:
        """
        Warm cache by pre-loading specified keys.

        Args:
            keys: List of keys to warm
            loader: Async function to load values

        Returns:
            Number of keys warmed
        """
        count = 0

        for key in keys:
            try:
                value = await loader(key)
                if value is not None:
                    await self.set(key, value)
                    count += 1
            except Exception as e:
                logger.error(f"Error warming cache key {key}: {e}")

        logger.info(f"Warmed {count}/{len(keys)} cache keys")
        return count

    async def _store_tags(self, key: str, tags: Set[str]) -> None:
        """Store tag associations for cache invalidation."""
        for tag in tags:
            tag_key = f"{self.config.namespace}:tags:{tag}"
            await self.redis_client.sadd(tag_key, key)
            await self.redis_client.expire(tag_key, self.config.redis_ttl_seconds)

    async def _get_keys_by_tag(self, tag: str) -> Set[str]:
        """Get all keys associated with a tag."""
        tag_key = f"{self.config.namespace}:tags:{tag}"
        keys = await self.redis_client.smembers(tag_key)
        return {key.decode() if isinstance(key, bytes) else key for key in keys}

    def _record_hit(self, level: CacheLevel, latency: float) -> None:
        """Record cache hit metrics."""
        if self.analytics:
            self.analytics.record_hit(level.value, latency)

    def _record_miss(self, latency: float) -> None:
        """Record cache miss metrics."""
        if self.analytics:
            self.analytics.record_miss(latency)

    def _record_set(self) -> None:
        """Record cache set operation."""
        if self.analytics:
            self.analytics.record_set()

    def _record_delete(self) -> None:
        """Record cache delete operation."""
        if self.analytics:
            self.analytics.record_delete()

    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if self.config.cache_warming_enabled:
            task = asyncio.create_task(self._cache_maintenance_loop())
            self._background_tasks.append(task)

    async def _cache_maintenance_loop(self) -> None:
        """Background task for cache maintenance."""
        while not self._shutdown_event.is_set():
            try:
                # Perform periodic maintenance
                await self._evict_expired_entries()
                await self._compact_memory_cache()

                # Sleep for maintenance interval
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(5)

    async def _evict_expired_entries(self) -> None:
        """Evict expired entries from caches."""
        # L1 handles its own eviction
        await self.l1_cache.evict_expired()

    async def _compact_memory_cache(self) -> None:
        """Compact memory cache to free up space."""
        if hasattr(self.l1_cache, 'compact'):
            await self.l1_cache.compact()

    async def shutdown(self) -> None:
        """Gracefully shutdown cache manager."""
        logger.info("Shutting down CacheManager")
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close connections
        if hasattr(self.redis_client, 'close'):
            await self.redis_client.close()

        logger.info("CacheManager shutdown complete")