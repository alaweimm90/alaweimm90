"""
Adaptive Caching Strategy

Implements:
- LRU cache with size limits
- Adaptive eviction policies
- Cache hit/miss statistics
- TTL-based expiration
"""

from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CACHE ENTRY
# ============================================================================

class CacheEntry:
    """Single cache entry."""

    def __init__(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 1
        self.ttl = ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False

        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl

    def access(self):
        """Record access."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "expired": self.is_expired(),
        }


# ============================================================================
# ADAPTIVE CACHE
# ============================================================================

class AdaptiveCache:
    """Adaptive LRU cache with statistics."""

    def __init__(self, max_size_mb: float = 256.0, default_ttl_seconds: Optional[int] = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            # Rough estimate using JSON serialization
            if isinstance(value, (dict, list)):
                return len(json.dumps(value).encode('utf-8'))
            elif isinstance(value, str):
                return len(value.encode('utf-8'))
            else:
                return 100  # Default estimate for objects
        except:
            return 100

    def _get_current_size(self) -> int:
        """Get current cache size."""
        return sum(self._estimate_size(e.value) for e in self.cache.values())

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )

        del self.cache[lru_key]
        self.evictions += 1
        logger.debug(f"Evicted cache entry: {lru_key}")

    def _make_room(self, needed_bytes: int):
        """Evict entries to make room."""
        current_size = self._get_current_size()

        while current_size + needed_bytes > self.max_size_bytes and self.cache:
            self._evict_lru()
            current_size = self._get_current_size()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None

        entry.access()
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache."""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl

        size = self._estimate_size(value)
        self._make_room(size)

        entry = CacheEntry(key, value, ttl_seconds)
        self.cache[key] = entry
        self.cache.move_to_end(key)

    def delete(self, key: str):
        """Delete entry from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "entries": len(self.cache),
            "size_bytes": self._get_current_size(),
            "size_mb": round(self._get_current_size() / (1024 * 1024), 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.evictions,
        }

    def get_entries(self) -> Dict[str, Dict[str, Any]]:
        """Get all cache entries as dict."""
        # Clean up expired entries first
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for k in expired_keys:
            del self.cache[k]

        return {
            key: entry.to_dict()
            for key, entry in self.cache.items()
        }

    def optimize(self):
        """Optimize cache by removing expired entries."""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for k in expired_keys:
            del self.cache[k]
            logger.debug(f"Cleaned up expired entry: {k}")

    def get_recommendations(self) -> list:
        """Get optimization recommendations."""
        stats = self.get_statistics()
        recommendations = []

        # Check hit rate
        if stats["hit_rate_percent"] < 50:
            recommendations.append(
                "Low cache hit rate. Consider increasing cache size or reviewing caching strategy."
            )

        # Check memory usage
        usage_percent = (stats["size_bytes"] / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
        if usage_percent > 90:
            recommendations.append(
                f"Cache near capacity ({usage_percent:.1f}%). Consider increasing max_size_mb."
            )

        # Check eviction rate
        if stats["evictions"] > stats["entries"] * 2:
            recommendations.append(
                "High eviction rate. Cache may be thrashing. Consider adjusting TTL or size."
            )

        if not recommendations:
            recommendations.append("Cache is operating optimally.")

        return recommendations


# ============================================================================
# GLOBAL CACHE
# ============================================================================

_global_cache: Optional[AdaptiveCache] = None


def initialize_cache(max_size_mb: float = 256.0) -> AdaptiveCache:
    """Initialize global cache."""
    global _global_cache
    _global_cache = AdaptiveCache(max_size_mb)
    return _global_cache


def get_cache() -> AdaptiveCache:
    """Get global cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdaptiveCache()
    return _global_cache
