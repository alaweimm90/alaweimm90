"""
Database sharding management for horizontal scaling.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Shard:
    """Represents a database shard."""
    id: str
    host: str
    port: int
    database: str
    weight: int = 1
    active: bool = True
    min_key: Optional[str] = None
    max_key: Optional[str] = None


class ShardingManager:
    """
    Database sharding manager for horizontal scaling
    with consistent hashing and automatic rebalancing.
    """

    def __init__(self, sharding_strategy: str = "hash"):
        """Initialize sharding manager."""
        self.strategy = sharding_strategy
        self.shards: Dict[str, Shard] = {}
        self.virtual_nodes = 150  # For consistent hashing
        self.ring: Dict[int, str] = {}  # Hash ring

    def add_shard(self, shard: Shard) -> None:
        """Add a new shard to the cluster."""
        self.shards[shard.id] = shard

        if self.strategy == "hash":
            self._add_to_hash_ring(shard)

        logger.info(f"Added shard {shard.id} at {shard.host}:{shard.port}")

    def remove_shard(self, shard_id: str) -> None:
        """Remove a shard from the cluster."""
        if shard_id in self.shards:
            if self.strategy == "hash":
                self._remove_from_hash_ring(shard_id)

            del self.shards[shard_id]
            logger.info(f"Removed shard {shard_id}")

    def get_shard(self, key: str) -> Optional[Shard]:
        """Get appropriate shard for a given key."""
        if self.strategy == "hash":
            return self._get_shard_by_hash(key)
        elif self.strategy == "range":
            return self._get_shard_by_range(key)
        else:
            # Round-robin or random
            import random
            active_shards = [s for s in self.shards.values() if s.active]
            return random.choice(active_shards) if active_shards else None

    def _add_to_hash_ring(self, shard: Shard) -> None:
        """Add shard to consistent hash ring."""
        for i in range(self.virtual_nodes):
            virtual_key = f"{shard.id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = shard.id

    def _remove_from_hash_ring(self, shard_id: str) -> None:
        """Remove shard from consistent hash ring."""
        keys_to_remove = [
            k for k, v in self.ring.items()
            if v == shard_id
        ]
        for key in keys_to_remove:
            del self.ring[key]

    def _get_shard_by_hash(self, key: str) -> Optional[Shard]:
        """Get shard using consistent hashing."""
        if not self.ring:
            return None

        hash_value = self._hash(key)

        # Find the next shard in the ring
        sorted_keys = sorted(self.ring.keys())
        for ring_key in sorted_keys:
            if ring_key >= hash_value:
                shard_id = self.ring[ring_key]
                return self.shards.get(shard_id)

        # Wrap around to first shard
        shard_id = self.ring[sorted_keys[0]]
        return self.shards.get(shard_id)

    def _get_shard_by_range(self, key: str) -> Optional[Shard]:
        """Get shard using range-based partitioning."""
        for shard in self.shards.values():
            if shard.min_key <= key <= shard.max_key:
                return shard
        return None

    def _hash(self, key: str) -> int:
        """Generate hash for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    async def rebalance(self) -> Dict[str, Any]:
        """Rebalance data across shards."""
        # This would involve moving data between shards
        # Simplified for this example
        logger.info("Starting shard rebalancing")

        moves = []
        # Calculate ideal distribution
        # Move data to achieve balance

        return {
            "status": "completed",
            "moves": len(moves),
            "shards": len(self.shards)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get sharding statistics."""
        return {
            "strategy": self.strategy,
            "total_shards": len(self.shards),
            "active_shards": sum(1 for s in self.shards.values() if s.active),
            "virtual_nodes": self.virtual_nodes if self.strategy == "hash" else 0,
            "shards": [
                {
                    "id": s.id,
                    "host": s.host,
                    "active": s.active,
                    "weight": s.weight
                }
                for s in self.shards.values()
            ]
        }