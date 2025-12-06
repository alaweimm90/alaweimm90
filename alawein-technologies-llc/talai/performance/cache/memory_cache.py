"""
High-performance in-memory cache with LRU eviction and size management.
"""

import asyncio
import pickle
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MemoryCacheEntry:
    """In-memory cache entry."""
    value: Any
    size_bytes: int
    created_at: float
    accessed_at: float
    ttl_seconds: Optional[int]
    access_count: int = 0


class InMemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""

    def __init__(
        self,
        max_size_mb: int = 512,
        ttl_seconds: int = 300,
        enable_compression: bool = False
    ):
        """Initialize in-memory cache."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.enable_compression = enable_compression

        self._cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Any:
        """Get value from memory cache."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if entry.ttl_seconds:
                if time.time() - entry.created_at > entry.ttl_seconds:
                    del self._cache[key]
                    self._current_size -= entry.size_bytes
                    self._misses += 1
                    return None

            # Update LRU
            self._cache.move_to_end(key)
            entry.accessed_at = time.time()
            entry.access_count += 1

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in memory cache."""
        size = sys.getsizeof(pickle.dumps(value))

        async with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size -= old_entry.size_bytes
                del self._cache[key]

            # Check size limit
            while self._current_size + size > self.max_size_bytes and self._cache:
                self._evict_lru()

            # Add new entry
            entry = MemoryCacheEntry(
                value=value,
                size_bytes=size,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl_seconds or self.default_ttl
            )

            self._cache[key] = entry
            self._current_size += size

            return True

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries."""
        async with self._lock:
            if pattern:
                # Clear matching pattern
                count = 0
                keys_to_delete = [
                    k for k in self._cache.keys()
                    if pattern in k
                ]
                for key in keys_to_delete:
                    entry = self._cache[key]
                    self._current_size -= entry.size_bytes
                    del self._cache[key]
                    count += 1
                return count
            else:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._current_size = 0
                return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "type": "memory",
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions
            }

    async def evict_expired(self) -> int:
        """Evict expired entries."""
        async with self._lock:
            current_time = time.time()
            count = 0

            keys_to_delete = []
            for key, entry in self._cache.items():
                if entry.ttl_seconds:
                    if current_time - entry.created_at > entry.ttl_seconds:
                        keys_to_delete.append(key)

            for key in keys_to_delete:
                entry = self._cache[key]
                self._current_size -= entry.size_bytes
                del self._cache[key]
                count += 1

            return count

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes
            self._evictions += 1