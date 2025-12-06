"""
Cache warming strategies for preloading frequently accessed data.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Intelligent cache warming with predictive preloading."""

    def __init__(self, cache_manager, concurrency: int = 10):
        """Initialize cache warmer."""
        self.cache = cache_manager
        self.concurrency = concurrency

    async def warm_from_list(
        self,
        keys: List[str],
        loader: Callable[[str], Any],
        ttl_seconds: Optional[int] = None
    ) -> Dict[str, bool]:
        """Warm cache from list of keys."""
        semaphore = asyncio.Semaphore(self.concurrency)
        results = {}

        async def warm_key(key: str) -> tuple[str, bool]:
            async with semaphore:
                try:
                    value = await loader(key)
                    if value is not None:
                        success = await self.cache.set(key, value, ttl_seconds=ttl_seconds)
                        return key, success
                    return key, False
                except Exception as e:
                    logger.error(f"Error warming key {key}: {e}")
                    return key, False

        tasks = [warm_key(key) for key in keys]
        completed = await asyncio.gather(*tasks)

        for key, success in completed:
            results[key] = success

        successful = sum(1 for s in results.values() if s)
        logger.info(f"Warmed {successful}/{len(keys)} keys")

        return results

    async def warm_popular_keys(
        self,
        pattern: str,
        top_n: int = 100,
        loader: Callable[[str], Any]
    ) -> int:
        """Warm most popular keys based on access patterns."""
        # Get analytics to identify hot keys
        stats = await self.cache.get_stats()

        # This would need integration with analytics to track popular keys
        # Simplified for this example
        popular_keys = []  # Would be fetched from analytics

        if popular_keys:
            results = await self.warm_from_list(popular_keys[:top_n], loader)
            return sum(1 for s in results.values() if s)

        return 0

    async def predictive_warming(
        self,
        user_context: Dict[str, Any],
        predictor: Callable[[Dict], List[str]],
        loader: Callable[[str], Any]
    ) -> int:
        """Predictively warm cache based on user context."""
        try:
            # Predict keys likely to be accessed
            predicted_keys = await predictor(user_context)

            if predicted_keys:
                results = await self.warm_from_list(predicted_keys, loader)
                return sum(1 for s in results.values() if s)

        except Exception as e:
            logger.error(f"Predictive warming error: {e}")

        return 0