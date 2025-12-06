"""
Intelligent request batching and deduplication for LLM APIs.
"""

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    """Represents a batched request."""
    id: str
    request_type: str
    payload: Dict[str, Any]
    callback: Callable
    timestamp: datetime
    priority: int = 0


class RequestBatcher:
    """
    Intelligent request batching for API optimization,
    with deduplication and priority management.
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout_ms: int = 100,
        enable_deduplication: bool = True
    ):
        """Initialize request batcher."""
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_deduplication = enable_deduplication

        self._batches: Dict[str, List[BatchedRequest]] = defaultdict(list)
        self._pending_requests: Dict[str, BatchedRequest] = {}
        self._dedup_cache: Dict[str, Any] = {}
        self._processing = False
        self._shutdown = False

        # Statistics
        self._stats = {
            "requests_batched": 0,
            "batches_sent": 0,
            "requests_deduplicated": 0,
            "total_requests": 0
        }

    def _generate_request_hash(self, request_type: str, payload: Dict[str, Any]) -> str:
        """Generate hash for request deduplication."""
        data = f"{request_type}:{str(sorted(payload.items()))}"
        return hashlib.md5(data.encode()).hexdigest()

    async def add_request(
        self,
        request_type: str,
        payload: Dict[str, Any],
        callback: Callable,
        priority: int = 0
    ) -> str:
        """
        Add request to batch queue.

        Args:
            request_type: Type of request for grouping
            payload: Request payload
            callback: Callback for result
            priority: Request priority

        Returns:
            Request ID
        """
        self._stats["total_requests"] += 1

        # Check deduplication
        if self.enable_deduplication:
            request_hash = self._generate_request_hash(request_type, payload)

            if request_hash in self._dedup_cache:
                self._stats["requests_deduplicated"] += 1
                result = self._dedup_cache[request_hash]

                # Call callback with cached result
                await callback(result)
                return f"dedup_{request_hash}"

        # Create batched request
        request = BatchedRequest(
            id=f"req_{datetime.utcnow().timestamp()}",
            request_type=request_type,
            payload=payload,
            callback=callback,
            timestamp=datetime.utcnow(),
            priority=priority
        )

        # Add to batch
        self._batches[request_type].append(request)
        self._pending_requests[request.id] = request

        # Process if batch is full
        if len(self._batches[request_type]) >= self.batch_size:
            await self._process_batch(request_type)
        else:
            # Schedule timeout processing
            asyncio.create_task(self._schedule_batch_timeout(request_type))

        self._stats["requests_batched"] += 1
        return request.id

    async def _schedule_batch_timeout(self, request_type: str) -> None:
        """Schedule batch processing after timeout."""
        await asyncio.sleep(self.batch_timeout_ms / 1000)

        if request_type in self._batches and self._batches[request_type]:
            await self._process_batch(request_type)

    async def _process_batch(self, request_type: str) -> None:
        """Process a batch of requests."""
        if self._processing or request_type not in self._batches:
            return

        batch = self._batches[request_type]
        if not batch:
            return

        self._processing = True

        try:
            # Sort by priority
            batch.sort(key=lambda r: r.priority, reverse=True)

            # Take batch_size requests
            processing_batch = batch[:self.batch_size]
            self._batches[request_type] = batch[self.batch_size:]

            # Process batch based on type
            if request_type.startswith("llm"):
                await self._process_llm_batch(processing_batch)
            else:
                await self._process_generic_batch(processing_batch)

            self._stats["batches_sent"] += 1

        finally:
            self._processing = False

    async def _process_llm_batch(self, batch: List[BatchedRequest]) -> None:
        """Process batch of LLM requests."""
        # Group similar prompts
        grouped = defaultdict(list)

        for request in batch:
            model = request.payload.get("model", "default")
            grouped[model].append(request)

        for model, requests in grouped.items():
            # Combine prompts for batch processing
            combined_prompts = [r.payload["prompt"] for r in requests]

            # Make batch API call (simplified)
            results = await self._batch_llm_call(model, combined_prompts)

            # Distribute results
            for request, result in zip(requests, results):
                await request.callback(result)

                # Cache for deduplication
                if self.enable_deduplication:
                    request_hash = self._generate_request_hash(
                        request.request_type,
                        request.payload
                    )
                    self._dedup_cache[request_hash] = result

                # Remove from pending
                self._pending_requests.pop(request.id, None)

    async def _process_generic_batch(self, batch: List[BatchedRequest]) -> None:
        """Process generic batch of requests."""
        # Execute requests in parallel
        tasks = []

        for request in batch:
            task = asyncio.create_task(
                self._execute_single_request(request)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for request, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Request {request.id} failed: {result}")
                await request.callback(None)
            else:
                await request.callback(result)

            self._pending_requests.pop(request.id, None)

    async def _batch_llm_call(
        self,
        model: str,
        prompts: List[str]
    ) -> List[str]:
        """Make batched LLM API call."""
        # This would integrate with actual LLM API
        # Simplified for this example
        return [f"Response for: {prompt[:50]}..." for prompt in prompts]

    async def _execute_single_request(self, request: BatchedRequest) -> Any:
        """Execute a single request."""
        # This would execute the actual request
        # Simplified for this example
        await asyncio.sleep(0.01)  # Simulate API call
        return {"status": "success", "data": request.payload}

    async def flush_all(self) -> None:
        """Flush all pending batches."""
        for request_type in list(self._batches.keys()):
            if self._batches[request_type]:
                await self._process_batch(request_type)

    def get_pending_count(self) -> int:
        """Get count of pending requests."""
        return sum(len(batch) for batch in self._batches.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "config": {
                "batch_size": self.batch_size,
                "batch_timeout_ms": self.batch_timeout_ms,
                "deduplication_enabled": self.enable_deduplication
            },
            "stats": self._stats,
            "pending_requests": len(self._pending_requests),
            "cache_size": len(self._dedup_cache),
            "active_batches": len(self._batches)
        }