"""
Redis cache implementation with compression and pipelining.
"""

import asyncio
import json
import logging
import pickle
import zlib
from typing import Any, Dict, List, Optional

import redis
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache with compression, pipelining, and clustering support."""

    def __init__(
        self,
        client: redis.Redis,
        namespace: str = "talai",
        ttl_seconds: int = 3600,
        enable_compression: bool = True,
        compression_threshold: int = 1024
    ):
        """Initialize Redis cache."""
        self.client = client
        self.namespace = namespace
        self.default_ttl = ttl_seconds
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

    async def get(self, key: str) -> Any:
        """Get value from Redis."""
        try:
            data = await self.client.get(key)
            if data is None:
                return None

            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in Redis."""
        try:
            serialized = self._serialize(value)
            ttl = ttl_seconds or self.default_ttl

            await self.client.setex(key, ttl, serialized)
            return True

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            result = await self.client.delete(key)
            return result > 0

        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        try:
            if pattern:
                search_pattern = f"{self.namespace}:{pattern}*"
            else:
                search_pattern = f"{self.namespace}:*"

            count = 0
            cursor = 0

            while True:
                cursor, keys = await self.client.scan(
                    cursor,
                    match=search_pattern,
                    count=100
                )

                if keys:
                    count += len(keys)
                    await self.client.delete(*keys)

                if cursor == 0:
                    break

            return count

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        try:
            values = await self.client.mget(keys)
            result = {}

            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)

            return result

        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return {}

    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set multiple values."""
        try:
            # Serialize all values
            serialized = {
                k: self._serialize(v)
                for k, v in mapping.items()
            }

            # Use pipeline for atomic operation
            pipe = self.client.pipeline()
            ttl = ttl_seconds or self.default_ttl

            for key, value in serialized.items():
                pipe.setex(key, ttl, value)

            await pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = await self.client.info("stats")
            memory_info = await self.client.info("memory")

            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": memory_info.get("used_memory", 0) / (1024 * 1024),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0)
            }

        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)

        if self.enable_compression and len(data) > self.compression_threshold:
            data = b"C" + zlib.compress(data, level=6)
        else:
            data = b"U" + data

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize stored value."""
        if not data:
            return None

        # Check compression flag
        if data[0:1] == b"C":
            data = zlib.decompress(data[1:])
        else:
            data = data[1:]

        return pickle.loads(data)