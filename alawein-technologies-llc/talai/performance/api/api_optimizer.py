"""
Comprehensive API optimization with versioning, caching, and monitoring.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class APIOptimizer:
    """
    API optimizer with request/response compression,
    versioning, and performance monitoring.
    """

    def __init__(
        self,
        enable_response_caching: bool = True,
        enable_request_compression: bool = True,
        api_versions: List[str] = None
    ):
        """Initialize API optimizer."""
        self.enable_response_caching = enable_response_caching
        self.enable_request_compression = enable_request_compression
        self.api_versions = api_versions or ["v1", "v2", "v3"]
        self._cache = {}
        self._metrics = {
            "requests": 0,
            "cache_hits": 0,
            "compression_ratio": 0
        }

    def optimize_request(
        self,
        request_data: Dict[str, Any],
        api_version: str = "v3"
    ) -> Dict[str, Any]:
        """
        Optimize API request.

        Args:
            request_data: Request data
            api_version: API version

        Returns:
            Optimized request
        """
        self._metrics["requests"] += 1

        # Version-specific optimizations
        if api_version == "v3":
            # Remove unnecessary fields for v3
            optimized = {
                k: v for k, v in request_data.items()
                if k not in ["deprecated_field", "legacy_option"]
            }
        elif api_version == "v2":
            # Transform for v2 compatibility
            optimized = self._transform_v2_request(request_data)
        else:
            optimized = request_data

        # Apply compression if large
        if self.enable_request_compression and len(str(optimized)) > 1024:
            import zlib
            compressed = zlib.compress(json.dumps(optimized).encode())
            self._metrics["compression_ratio"] = len(compressed) / len(str(optimized))
            return {"compressed": True, "data": compressed}

        return optimized

    def _transform_v2_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request for v2 API."""
        # Map new field names to v2 format
        field_mapping = {
            "user_id": "userId",
            "request_type": "type",
            "params": "parameters"
        }

        transformed = {}
        for key, value in data.items():
            new_key = field_mapping.get(key, key)
            transformed[new_key] = value

        return transformed

    def cache_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cache API response."""
        if not self.enable_response_caching:
            return

        cache_key = self._generate_cache_key(endpoint, params)
        self._cache[cache_key] = {
            "response": response,
            "timestamp": datetime.utcnow(),
            "ttl": ttl_seconds
        }

    def get_cached_response(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached response if available."""
        if not self.enable_response_caching:
            return None

        cache_key = self._generate_cache_key(endpoint, params)

        if cache_key in self._cache:
            entry = self._cache[cache_key]
            age = (datetime.utcnow() - entry["timestamp"]).total_seconds()

            if age < entry["ttl"]:
                self._metrics["cache_hits"] += 1
                return entry["response"]

            # Expired, remove from cache
            del self._cache[cache_key]

        return None

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for endpoint and parameters."""
        data = f"{endpoint}:{sorted(params.items())}"
        return hashlib.md5(data.encode()).hexdigest()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        cache_hit_rate = (
            self._metrics["cache_hits"] / self._metrics["requests"]
            if self._metrics["requests"] > 0 else 0
        )

        return {
            "total_requests": self._metrics["requests"],
            "cache_hits": self._metrics["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "compression_ratio": self._metrics["compression_ratio"],
            "supported_versions": self.api_versions
        }