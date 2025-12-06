"""
Cache invalidation strategies and dependency tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """Advanced cache invalidation with dependency tracking."""

    def __init__(self, cache_manager):
        """Initialize cache invalidator."""
        self.cache = cache_manager
        self._dependencies: Dict[str, Set[str]] = {}

    async def register_dependency(self, key: str, depends_on: List[str]) -> None:
        """Register cache key dependencies."""
        for dep in depends_on:
            if dep not in self._dependencies:
                self._dependencies[dep] = set()
            self._dependencies[dep].add(key)

    async def invalidate_cascade(self, key: str) -> int:
        """Invalidate key and all dependent keys."""
        invalidated = 0

        # Invalidate the key itself
        if await self.cache.delete(key):
            invalidated += 1

        # Invalidate all dependent keys
        if key in self._dependencies:
            for dependent_key in self._dependencies[key]:
                if await self.cache.delete(dependent_key):
                    invalidated += 1

            # Clean up dependencies
            del self._dependencies[key]

        logger.info(f"Cascade invalidated {invalidated} keys starting from {key}")
        return invalidated

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        return await self.cache.clear_all(pattern)

    async def invalidate_by_age(self, max_age_seconds: int) -> int:
        """Invalidate keys older than specified age."""
        # This would need timestamp tracking in cache entries
        # Simplified for this example
        count = 0
        logger.info(f"Invalidated {count} keys older than {max_age_seconds} seconds")
        return count