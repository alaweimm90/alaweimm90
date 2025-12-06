"""
Response streaming for large results with compression and chunking.
"""

import asyncio
import gzip
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseStreamer:
    """
    Stream large responses with compression and chunking
    for optimal network utilization.
    """

    def __init__(
        self,
        chunk_size: int = 8192,
        enable_compression: bool = True,
        compression_threshold: int = 1024
    ):
        """Initialize response streamer."""
        self.chunk_size = chunk_size
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

    async def stream_response(
        self,
        data: Any,
        content_type: str = "application/json"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream response data in chunks.

        Args:
            data: Data to stream
            content_type: Content type

        Yields:
            Data chunks
        """
        # Serialize data
        if content_type == "application/json":
            serialized = json.dumps(data).encode()
        else:
            serialized = str(data).encode()

        # Compress if needed
        if self.enable_compression and len(serialized) > self.compression_threshold:
            serialized = gzip.compress(serialized)
            yield b"GZIP:"  # Compression marker

        # Stream in chunks
        for i in range(0, len(serialized), self.chunk_size):
            chunk = serialized[i:i + self.chunk_size]
            yield chunk
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def stream_json_array(
        self,
        items: list,
        transform: Optional[callable] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream JSON array incrementally.

        Args:
            items: List of items to stream
            transform: Optional transformation function

        Yields:
            JSON chunks
        """
        yield b'['

        for i, item in enumerate(items):
            if transform:
                item = await transform(item)

            item_json = json.dumps(item).encode()

            if i > 0:
                yield b','

            yield item_json
            await asyncio.sleep(0)

        yield b']'

    async def stream_ndjson(
        self,
        items: list
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream newline-delimited JSON.

        Args:
            items: List of items

        Yields:
            NDJSON lines
        """
        for item in items:
            line = json.dumps(item).encode() + b'\n'
            yield line
            await asyncio.sleep(0)

    def create_sse_stream(
        self,
        events: list
    ) -> AsyncGenerator[bytes, None]:
        """
        Create Server-Sent Events stream.

        Args:
            events: List of events

        Returns:
            SSE event stream
        """
        async def generate():
            for event in events:
                event_data = f"data: {json.dumps(event)}\n\n".encode()
                yield event_data
                await asyncio.sleep(0.1)

        return generate()

    async def stream_with_progress(
        self,
        data: Any,
        progress_callback: callable
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream data with progress tracking.

        Args:
            data: Data to stream
            progress_callback: Progress callback function

        Yields:
            Data chunks
        """
        serialized = json.dumps(data).encode()
        total_size = len(serialized)
        bytes_sent = 0

        for i in range(0, total_size, self.chunk_size):
            chunk = serialized[i:i + self.chunk_size]
            yield chunk

            bytes_sent += len(chunk)
            progress = bytes_sent / total_size * 100

            await progress_callback(progress)
            await asyncio.sleep(0)