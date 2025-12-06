"""
LLM Response Cache for API cost optimization.

Intelligent caching of LLM responses with semantic similarity matching,
response versioning, and cost tracking.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Represents an LLM API request."""
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None


@dataclass
class LLMResponse:
    """Represents an LLM API response with metadata."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: datetime
    request_hash: str
    embedding: Optional[np.ndarray] = None


class LLMResponseCache:
    """
    Specialized cache for LLM responses with semantic matching,
    cost tracking, and response quality scoring.
    """

    # Pricing per 1K tokens (example rates)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(
        self,
        cache_manager,
        similarity_threshold: float = 0.95,
        enable_semantic_match: bool = True,
        max_cache_age_hours: int = 24,
        track_cost_savings: bool = True
    ):
        """
        Initialize LLM response cache.

        Args:
            cache_manager: Underlying cache manager
            similarity_threshold: Threshold for semantic matching
            enable_semantic_match: Enable semantic similarity matching
            max_cache_age_hours: Maximum age for cached responses
            track_cost_savings: Track API cost savings
        """
        self.cache = cache_manager
        self.similarity_threshold = similarity_threshold
        self.enable_semantic_match = enable_semantic_match
        self.max_cache_age = timedelta(hours=max_cache_age_hours)
        self.track_cost_savings = track_cost_savings

        self._cost_savings = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate deterministic cache key for LLM request."""
        # Create a normalized request representation
        key_data = {
            "model": request.model,
            "prompt": request.prompt,
            "temperature": round(request.temperature, 2),
            "max_tokens": request.max_tokens,
            "top_p": round(request.top_p, 2),
            "frequency_penalty": round(request.frequency_penalty, 2),
            "presence_penalty": round(request.presence_penalty, 2),
            "stop_sequences": sorted(request.stop_sequences) if request.stop_sequences else None,
            "system_prompt": request.system_prompt
        }

        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()

        return f"llm:{request.model}:{key_hash[:16]}"

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API call cost."""
        if model not in self.PRICING:
            return 0.0

        pricing = self.PRICING[model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    async def get_cached_response(
        self,
        request: LLMRequest,
        use_semantic_match: bool = True
    ) -> Optional[LLMResponse]:
        """
        Get cached LLM response if available.

        Args:
            request: LLM request
            use_semantic_match: Use semantic similarity matching

        Returns:
            Cached response or None
        """
        self._total_requests += 1

        # Try exact match first
        cache_key = self._generate_cache_key(request)
        cached = await self.cache.get(cache_key)

        if cached:
            response = self._deserialize_response(cached)
            if self._is_response_valid(response):
                self._cache_hits += 1
                if self.track_cost_savings:
                    self._cost_savings += response.cost_usd
                logger.info(f"LLM cache hit (exact): saved ${response.cost_usd:.4f}")
                return response

        # Try semantic match if enabled
        if use_semantic_match and self.enable_semantic_match:
            similar_response = await self._find_similar_response(request)
            if similar_response:
                self._cache_hits += 1
                if self.track_cost_savings:
                    self._cost_savings += similar_response.cost_usd
                logger.info(f"LLM cache hit (semantic): saved ${similar_response.cost_usd:.4f}")
                return similar_response

        self._cache_misses += 1
        return None

    async def cache_response(
        self,
        request: LLMRequest,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ) -> bool:
        """
        Cache an LLM response.

        Args:
            request: Original request
            response: LLM response content
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency_ms: Response latency in milliseconds

        Returns:
            Success status
        """
        cache_key = self._generate_cache_key(request)
        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(request.model, prompt_tokens, completion_tokens)

        llm_response = LLMResponse(
            content=response,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            timestamp=datetime.utcnow(),
            request_hash=cache_key
        )

        # Generate embedding if semantic matching is enabled
        if self.enable_semantic_match:
            llm_response.embedding = await self._generate_embedding(request.prompt)

        # Store in cache
        serialized = self._serialize_response(llm_response)
        ttl_seconds = int(self.max_cache_age.total_seconds())

        success = await self.cache.set(
            cache_key,
            serialized,
            ttl_seconds=ttl_seconds,
            tags={"llm", request.model}
        )

        if success:
            # Store in semantic index if enabled
            if self.enable_semantic_match and llm_response.embedding is not None:
                await self._index_response(request, llm_response)

        logger.info(
            f"Cached LLM response: model={request.model}, "
            f"tokens={total_tokens}, cost=${cost:.4f}"
        )

        return success

    async def _find_similar_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Find semantically similar cached response.

        Args:
            request: LLM request

        Returns:
            Similar cached response or None
        """
        # Generate embedding for request
        request_embedding = await self._generate_embedding(request.prompt)
        if request_embedding is None:
            return None

        # Search semantic index
        pattern = f"llm:{request.model}:*"
        similar_keys = await self._search_semantic_index(
            request_embedding,
            pattern,
            top_k=10
        )

        for key, similarity in similar_keys:
            if similarity >= self.similarity_threshold:
                cached = await self.cache.get(key)
                if cached:
                    response = self._deserialize_response(cached)
                    if self._is_response_valid(response):
                        logger.info(
                            f"Found semantically similar response "
                            f"(similarity={similarity:.3f})"
                        )
                        return response

        return None

    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        # Simplified: using hash-based pseudo-embedding
        # In production, use actual embedding model
        hash_value = hashlib.sha256(text.encode()).digest()
        embedding = np.frombuffer(hash_value, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    async def _index_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Index response for semantic search."""
        index_key = f"llm_index:{request.model}:{response.request_hash}"
        index_data = {
            "request_hash": response.request_hash,
            "embedding": response.embedding.tolist() if response.embedding is not None else None,
            "timestamp": response.timestamp.isoformat()
        }

        await self.cache.set(
            index_key,
            index_data,
            ttl_seconds=int(self.max_cache_age.total_seconds())
        )

    async def _search_semantic_index(
        self,
        query_embedding: np.ndarray,
        pattern: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search semantic index for similar embeddings.

        Args:
            query_embedding: Query embedding
            pattern: Key pattern to search
            top_k: Number of top results

        Returns:
            List of (key, similarity) tuples
        """
        # Simplified implementation
        # In production, use vector database or specialized index
        results = []

        # Get all matching keys
        stats = await self.cache.get_stats()
        # This is a simplified approach
        # Real implementation would use proper key scanning

        return results[:top_k]

    def _serialize_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Serialize LLM response for caching."""
        return {
            "content": response.content,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "timestamp": response.timestamp.isoformat(),
            "request_hash": response.request_hash,
            "embedding": response.embedding.tolist() if response.embedding is not None else None
        }

    def _deserialize_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Deserialize cached LLM response."""
        return LLMResponse(
            content=data["content"],
            model=data["model"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            latency_ms=data["latency_ms"],
            cost_usd=data["cost_usd"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            request_hash=data["request_hash"],
            embedding=np.array(data["embedding"]) if data.get("embedding") else None
        )

    def _is_response_valid(self, response: LLMResponse) -> bool:
        """Check if cached response is still valid."""
        age = datetime.utcnow() - response.timestamp
        return age <= self.max_cache_age

    async def get_stats(self) -> Dict[str, Any]:
        """Get LLM cache statistics."""
        hit_rate = self._cache_hits / self._total_requests if self._total_requests > 0 else 0

        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cost_savings_usd": self._cost_savings,
            "average_savings_per_hit": (
                self._cost_savings / self._cache_hits if self._cache_hits > 0 else 0
            )
        }

    async def invalidate_model_cache(self, model: str) -> int:
        """
        Invalidate all cached responses for a specific model.

        Args:
            model: Model name

        Returns:
            Number of entries invalidated
        """
        return await self.cache.invalidate_by_tags({model})

    async def clear_old_responses(self, max_age_hours: int) -> int:
        """
        Clear responses older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of entries cleared
        """
        # Implementation would scan and delete old entries
        # Simplified for this example
        pattern = f"llm:*"
        count = 0

        logger.info(f"Cleared {count} old LLM responses")
        return count