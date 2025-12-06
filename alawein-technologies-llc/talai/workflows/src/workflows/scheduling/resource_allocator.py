"""
Resource Allocator - Resource allocation and optimization

Manages resource allocation, tracking, and optimization for job execution.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"
    CUSTOM = "custom"


@dataclass
class Resource:
    """Resource definition"""
    name: str
    type: ResourceType
    capacity: float
    available: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def allocate(self, amount: float) -> bool:
        """Allocate resource amount"""
        if amount > self.available:
            return False
        self.available -= amount
        return True

    def release(self, amount: float):
        """Release resource amount"""
        self.available = min(self.available + amount, self.capacity)

    def get_utilization(self) -> float:
        """Get resource utilization percentage"""
        if self.capacity == 0:
            return 0
        return (self.capacity - self.available) / self.capacity


@dataclass
class ResourcePool:
    """Pool of resources"""
    name: str
    resources: Dict[str, Resource] = field(default_factory=dict)
    allocations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_resource(self, resource: Resource):
        """Add resource to pool"""
        self.resources[resource.name] = resource

    def check_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if requirements can be met"""
        for resource_name, amount in requirements.items():
            if resource_name not in self.resources:
                return False
            if self.resources[resource_name].available < amount:
                return False
        return True

    def allocate(self, allocation_id: str,
                requirements: Dict[str, float]) -> bool:
        """Allocate resources"""
        # Check availability first
        if not self.check_availability(requirements):
            return False

        # Allocate resources
        allocated = {}
        for resource_name, amount in requirements.items():
            resource = self.resources[resource_name]
            if resource.allocate(amount):
                allocated[resource_name] = amount
            else:
                # Rollback on failure
                for rollback_name, rollback_amount in allocated.items():
                    self.resources[rollback_name].release(rollback_amount)
                return False

        # Record allocation
        self.allocations[allocation_id] = allocated
        return True

    def release(self, allocation_id: str):
        """Release allocated resources"""
        if allocation_id not in self.allocations:
            return

        allocation = self.allocations[allocation_id]
        for resource_name, amount in allocation.items():
            if resource_name in self.resources:
                self.resources[resource_name].release(amount)

        del self.allocations[allocation_id]


class ResourceAllocator:
    """Manages resource allocation across pools"""

    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.default_pool = ResourcePool(name="default")
        self.pools["default"] = self.default_pool
        self._lock = asyncio.Lock()
        self._init_default_resources()

    def _init_default_resources(self):
        """Initialize default resources"""
        # CPU cores
        self.default_pool.add_resource(Resource(
            name="cpu",
            type=ResourceType.CPU,
            capacity=8.0,
            available=8.0,
            unit="cores"
        ))

        # Memory in GB
        self.default_pool.add_resource(Resource(
            name="memory",
            type=ResourceType.MEMORY,
            capacity=16.0,
            available=16.0,
            unit="GB"
        ))

        # API calls per minute
        self.default_pool.add_resource(Resource(
            name="api_calls",
            type=ResourceType.API_CALLS,
            capacity=100.0,
            available=100.0,
            unit="calls/min"
        ))

    def create_pool(self, pool: ResourcePool):
        """Create resource pool"""
        self.pools[pool.name] = pool
        logger.info(f"Created resource pool: {pool.name}")

    def add_resource(self, resource: Resource, pool_name: str = "default"):
        """Add resource to pool"""
        if pool_name not in self.pools:
            self.pools[pool_name] = ResourcePool(name=pool_name)

        self.pools[pool_name].add_resource(resource)
        logger.info(f"Added resource {resource.name} to pool {pool_name}")

    async def check_availability(self, requirements: Dict[str, float],
                                pool_name: str = "default") -> bool:
        """Check resource availability"""
        async with self._lock:
            if pool_name not in self.pools:
                return False

            return self.pools[pool_name].check_availability(requirements)

    async def allocate(self, allocation_id: str,
                      requirements: Dict[str, float],
                      pool_name: str = "default") -> bool:
        """Allocate resources"""
        async with self._lock:
            if pool_name not in self.pools:
                return False

            success = self.pools[pool_name].allocate(allocation_id, requirements)

            if success:
                logger.info(f"Allocated resources for {allocation_id}: {requirements}")
            else:
                logger.warning(f"Failed to allocate resources for {allocation_id}")

            return success

    async def release(self, allocation_id: str, pool_name: str = "default"):
        """Release allocated resources"""
        async with self._lock:
            if pool_name not in self.pools:
                return

            self.pools[pool_name].release(allocation_id)
            logger.info(f"Released resources for {allocation_id}")

    async def wait_for_resources(self, requirements: Dict[str, float],
                                pool_name: str = "default",
                                timeout: Optional[float] = None) -> bool:
        """Wait for resources to become available"""
        start_time = datetime.now()

        while True:
            if await self.check_availability(requirements, pool_name):
                return True

            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return False

            await asyncio.sleep(0.5)

    def get_utilization(self, resource_name: str,
                       pool_name: str = "default") -> float:
        """Get resource utilization"""
        if pool_name not in self.pools:
            return 0

        pool = self.pools[pool_name]
        if resource_name not in pool.resources:
            return 0

        return pool.resources[resource_name].get_utilization()

    def get_pool_stats(self, pool_name: str = "default") -> Dict[str, Any]:
        """Get statistics for resource pool"""
        if pool_name not in self.pools:
            return {}

        pool = self.pools[pool_name]
        stats = {
            "name": pool_name,
            "resources": {},
            "allocations": len(pool.allocations)
        }

        for resource_name, resource in pool.resources.items():
            stats["resources"][resource_name] = {
                "type": resource.type.value,
                "capacity": resource.capacity,
                "available": resource.available,
                "utilization": resource.get_utilization(),
                "unit": resource.unit
            }

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        return {
            pool_name: self.get_pool_stats(pool_name)
            for pool_name in self.pools
        }

    async def optimize_allocation(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize resource allocation for multiple jobs"""
        # Sort by priority and resource efficiency
        sorted_allocations = sorted(
            allocations,
            key=lambda a: (
                -a.get("priority", 0),  # Higher priority first
                a.get("resource_sum", 0)  # Lower resource usage first
            )
        )

        optimized = []
        temp_allocations = {}

        async with self._lock:
            for allocation in sorted_allocations:
                job_id = allocation["job_id"]
                requirements = allocation["requirements"]
                pool_name = allocation.get("pool", "default")

                # Try to allocate
                if pool_name in self.pools:
                    pool = self.pools[pool_name]

                    # Check if can be allocated
                    can_allocate = True
                    for resource_name, amount in requirements.items():
                        if resource_name not in pool.resources:
                            can_allocate = False
                            break

                        resource = pool.resources[resource_name]
                        # Consider temporary allocations
                        temp_used = sum(
                            alloc.get(resource_name, 0)
                            for alloc in temp_allocations.values()
                        )
                        if resource.available - temp_used < amount:
                            can_allocate = False
                            break

                    if can_allocate:
                        temp_allocations[job_id] = requirements
                        optimized.append(allocation)

        return optimized