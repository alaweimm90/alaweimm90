"""
Read replica management for load distribution and high availability.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Replica:
    """Represents a database replica."""
    id: str
    host: str
    port: int
    is_primary: bool = False
    is_sync: bool = False
    lag_ms: float = 0
    active: bool = True
    weight: int = 1


class ReplicaManager:
    """
    Database replica manager for read scaling
    and automatic failover.
    """

    def __init__(self, routing_policy: str = "round_robin"):
        """Initialize replica manager."""
        self.routing_policy = routing_policy
        self.replicas: Dict[str, Replica] = {}
        self.primary: Optional[Replica] = None
        self.current_index = 0

    def add_replica(self, replica: Replica) -> None:
        """Add a replica to the pool."""
        self.replicas[replica.id] = replica

        if replica.is_primary:
            self.primary = replica

        logger.info(f"Added {'primary' if replica.is_primary else 'replica'} {replica.id}")

    def get_read_replica(self) -> Optional[Replica]:
        """Get a read replica based on routing policy."""
        active_replicas = [
            r for r in self.replicas.values()
            if r.active and not r.is_primary
        ]

        if not active_replicas:
            return self.primary

        if self.routing_policy == "round_robin":
            replica = active_replicas[self.current_index % len(active_replicas)]
            self.current_index += 1
            return replica

        elif self.routing_policy == "least_lag":
            return min(active_replicas, key=lambda r: r.lag_ms)

        elif self.routing_policy == "weighted":
            weights = [r.weight for r in active_replicas]
            return random.choices(active_replicas, weights=weights)[0]

        else:  # random
            return random.choice(active_replicas)

    def get_write_replica(self) -> Optional[Replica]:
        """Get primary replica for writes."""
        return self.primary

    async def promote_replica(self, replica_id: str) -> bool:
        """Promote replica to primary."""
        if replica_id not in self.replicas:
            return False

        # Demote current primary
        if self.primary:
            self.primary.is_primary = False

        # Promote new primary
        new_primary = self.replicas[replica_id]
        new_primary.is_primary = True
        self.primary = new_primary

        logger.info(f"Promoted {replica_id} to primary")
        return True

    async def check_replication_lag(self) -> Dict[str, float]:
        """Check replication lag for all replicas."""
        lag_info = {}

        for replica in self.replicas.values():
            if not replica.is_primary:
                # In real implementation, query replica for lag
                lag_info[replica.id] = replica.lag_ms

        return lag_info

    def get_stats(self) -> Dict[str, Any]:
        """Get replica statistics."""
        return {
            "routing_policy": self.routing_policy,
            "primary": self.primary.id if self.primary else None,
            "total_replicas": len(self.replicas),
            "active_replicas": sum(1 for r in self.replicas.values() if r.active),
            "sync_replicas": sum(1 for r in self.replicas.values() if r.is_sync),
            "replicas": [
                {
                    "id": r.id,
                    "host": r.host,
                    "is_primary": r.is_primary,
                    "lag_ms": r.lag_ms,
                    "active": r.active
                }
                for r in self.replicas.values()
            ]
        }