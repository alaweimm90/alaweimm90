"""
Resource Management System for MEZAN

This module provides sophisticated resource pool management, priority-based
allocation, and fair scheduling algorithms to prevent resource contention
and ensure efficient utilization in concurrent environments.

Features:
- Resource pool management with bounded capacity
- Priority-based resource allocation
- Fair scheduling algorithms (Round-Robin, Priority Queue, Fair Share)
- Resource contention queue management
- Usage tracking and limits
- Resource leasing with automatic reclamation
- Performance monitoring and metrics

Author: MEZAN Concurrency Team
Version: 1.0.0
"""

import threading
import time
import logging
import heapq
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of managed resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    GPU = "gpu"
    THREAD_POOL = "thread_pool"
    CONNECTION_POOL = "connection_pool"
    CUSTOM = "custom"


class SchedulingAlgorithm(Enum):
    """Resource scheduling algorithms"""
    FIFO = "fifo"                    # First In First Out
    LIFO = "lifo"                    # Last In First Out
    PRIORITY = "priority"            # Priority-based
    ROUND_ROBIN = "round_robin"      # Round-robin fair share
    WEIGHTED_FAIR = "weighted_fair"  # Weighted fair queuing
    SHORTEST_JOB = "shortest_job"    # Shortest job first
    DEADLINE = "deadline"            # Earliest deadline first


class AllocationStatus(Enum):
    """Status of resource allocation request"""
    PENDING = "pending"
    ALLOCATED = "allocated"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RELEASED = "released"


@dataclass
class ResourceSpec:
    """Specification for a resource type"""
    resource_type: ResourceType
    total_capacity: float
    unit: str = "units"
    min_allocation: float = 1.0
    max_allocation: Optional[float] = None
    divisible: bool = True  # Can be allocated in fractions
    preemptible: bool = False
    lease_duration: Optional[float] = None  # Default lease time in seconds


@dataclass
class ResourceRequest:
    """Request for resource allocation"""
    request_id: str
    requester_id: str
    resource_type: ResourceType
    amount: float
    priority: int = 0
    deadline: Optional[float] = None
    lease_duration: Optional[float] = None
    callback: Optional[Callable] = None
    timestamp: float = field(default_factory=time.time)
    estimated_duration: Optional[float] = None
    can_wait: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Allocated resource information"""
    allocation_id: str
    request: ResourceRequest
    amount_allocated: float
    allocation_time: float
    lease_expiry: Optional[float]
    usage_stats: Dict[str, float] = field(default_factory=dict)
    released: bool = False
    release_time: Optional[float] = None


@dataclass
class ResourceUsageStats:
    """Resource usage statistics"""
    total_allocations: int = 0
    total_releases: int = 0
    total_amount_allocated: float = 0.0
    current_utilization: float = 0.0
    peak_utilization: float = 0.0
    avg_wait_time: float = 0.0
    avg_hold_time: float = 0.0
    rejection_count: int = 0
    timeout_count: int = 0
    preemption_count: int = 0


class ResourcePool:
    """
    Resource pool for a specific resource type

    Manages allocation, deallocation, and scheduling for a single resource type.
    """

    def __init__(
        self,
        spec: ResourceSpec,
        scheduling: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY
    ):
        """
        Initialize resource pool

        Args:
            spec: Resource specification
            scheduling: Scheduling algorithm to use
        """
        self.spec = spec
        self.scheduling = scheduling
        self.available_capacity = spec.total_capacity
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.wait_queue: List[ResourceRequest] = []
        self.stats = ResourceUsageStats()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def request(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """
        Request resource allocation

        Returns allocation if successful, None otherwise
        """
        with self._lock:
            # Validate request
            if not self._validate_request(request):
                self.stats.rejection_count += 1
                return None

            # Check immediate availability
            if self._can_allocate(request.amount):
                return self._allocate(request)

            # Add to wait queue if allowed
            if request.can_wait:
                self._add_to_queue(request)
                return None

            self.stats.rejection_count += 1
            return None

    def wait_for_allocation(
        self,
        request: ResourceRequest,
        timeout: Optional[float] = None
    ) -> Optional[ResourceAllocation]:
        """
        Wait for resource allocation

        Blocks until resource is available or timeout
        """
        start_time = time.time()

        with self._lock:
            # Add to queue
            self._add_to_queue(request)

            # Wait for allocation
            while request.request_id not in self.allocations:
                wait_time = None
                if timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        # Timeout - remove from queue
                        self._remove_from_queue(request.request_id)
                        self.stats.timeout_count += 1
                        return None
                    wait_time = timeout - elapsed

                self._condition.wait(timeout=wait_time)

            return self.allocations.get(request.request_id)

    def release(self, allocation_id: str) -> bool:
        """
        Release allocated resource

        Returns True if successful
        """
        with self._lock:
            allocation = self.allocations.get(allocation_id)
            if not allocation or allocation.released:
                return False

            # Mark as released
            allocation.released = True
            allocation.release_time = time.time()

            # Update available capacity
            self.available_capacity += allocation.amount_allocated

            # Update statistics
            self.stats.total_releases += 1
            hold_time = allocation.release_time - allocation.allocation_time
            self.stats.avg_hold_time = (
                (self.stats.avg_hold_time * (self.stats.total_releases - 1) + hold_time) /
                self.stats.total_releases
            )

            # Process wait queue
            self._process_wait_queue()

            # Notify waiters
            self._condition.notify_all()

            logger.debug(f"Released {allocation.amount_allocated} of {self.spec.resource_type.value}")
            return True

    def _validate_request(self, request: ResourceRequest) -> bool:
        """Validate resource request"""
        if request.amount < self.spec.min_allocation:
            logger.warning(f"Request amount {request.amount} below minimum {self.spec.min_allocation}")
            return False

        if self.spec.max_allocation and request.amount > self.spec.max_allocation:
            logger.warning(f"Request amount {request.amount} exceeds maximum {self.spec.max_allocation}")
            return False

        if request.amount > self.spec.total_capacity:
            logger.warning(f"Request amount {request.amount} exceeds total capacity {self.spec.total_capacity}")
            return False

        return True

    def _can_allocate(self, amount: float) -> bool:
        """Check if amount can be allocated"""
        return self.available_capacity >= amount

    def _allocate(self, request: ResourceRequest) -> ResourceAllocation:
        """Allocate resource"""
        allocation_id = f"alloc_{uuid.uuid4().hex[:8]}"

        # Calculate lease expiry
        lease_expiry = None
        if request.lease_duration:
            lease_expiry = time.time() + request.lease_duration
        elif self.spec.lease_duration:
            lease_expiry = time.time() + self.spec.lease_duration

        # Create allocation
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            request=request,
            amount_allocated=request.amount,
            allocation_time=time.time(),
            lease_expiry=lease_expiry
        )

        # Update state
        self.allocations[allocation_id] = allocation
        self.available_capacity -= request.amount

        # Update statistics
        self.stats.total_allocations += 1
        self.stats.total_amount_allocated += request.amount
        self.stats.current_utilization = (
            (self.spec.total_capacity - self.available_capacity) /
            self.spec.total_capacity
        )
        self.stats.peak_utilization = max(
            self.stats.peak_utilization,
            self.stats.current_utilization
        )

        # Execute callback if provided
        if request.callback:
            try:
                request.callback(allocation)
            except Exception as e:
                logger.error(f"Error in allocation callback: {e}")

        logger.debug(f"Allocated {request.amount} of {self.spec.resource_type.value}")
        return allocation

    def _add_to_queue(self, request: ResourceRequest):
        """Add request to wait queue based on scheduling algorithm"""
        if self.scheduling == SchedulingAlgorithm.FIFO:
            self.wait_queue.append(request)

        elif self.scheduling == SchedulingAlgorithm.LIFO:
            self.wait_queue.insert(0, request)

        elif self.scheduling == SchedulingAlgorithm.PRIORITY:
            # Insert in priority order (higher priority first)
            inserted = False
            for i, queued in enumerate(self.wait_queue):
                if request.priority > queued.priority:
                    self.wait_queue.insert(i, request)
                    inserted = True
                    break
            if not inserted:
                self.wait_queue.append(request)

        elif self.scheduling == SchedulingAlgorithm.DEADLINE:
            # Sort by deadline (earliest first)
            if request.deadline:
                inserted = False
                for i, queued in enumerate(self.wait_queue):
                    if queued.deadline and request.deadline < queued.deadline:
                        self.wait_queue.insert(i, request)
                        inserted = True
                        break
                if not inserted:
                    self.wait_queue.append(request)
            else:
                self.wait_queue.append(request)

        elif self.scheduling == SchedulingAlgorithm.SHORTEST_JOB:
            # Sort by estimated duration
            if request.estimated_duration:
                inserted = False
                for i, queued in enumerate(self.wait_queue):
                    if queued.estimated_duration and request.estimated_duration < queued.estimated_duration:
                        self.wait_queue.insert(i, request)
                        inserted = True
                        break
                if not inserted:
                    self.wait_queue.append(request)
            else:
                self.wait_queue.append(request)

        else:
            # Default to FIFO
            self.wait_queue.append(request)

    def _remove_from_queue(self, request_id: str):
        """Remove request from wait queue"""
        self.wait_queue = [r for r in self.wait_queue if r.request_id != request_id]

    def _process_wait_queue(self):
        """Process waiting requests"""
        processed = []

        for request in self.wait_queue:
            if self._can_allocate(request.amount):
                allocation = self._allocate(request)
                processed.append(request.request_id)

                # Update wait time statistics
                wait_time = time.time() - request.timestamp
                self.stats.avg_wait_time = (
                    (self.stats.avg_wait_time * (self.stats.total_allocations - 1) + wait_time) /
                    self.stats.total_allocations
                )

        # Remove processed requests
        for request_id in processed:
            self._remove_from_queue(request_id)

    def check_leases(self) -> List[str]:
        """
        Check for expired leases and reclaim resources

        Returns list of reclaimed allocation IDs
        """
        reclaimed = []
        current_time = time.time()

        with self._lock:
            for alloc_id, allocation in list(self.allocations.items()):
                if allocation.lease_expiry and current_time > allocation.lease_expiry:
                    if not allocation.released:
                        logger.info(f"Reclaiming expired lease: {alloc_id}")
                        self.release(alloc_id)
                        reclaimed.append(alloc_id)

        return reclaimed

    def get_stats(self) -> ResourceUsageStats:
        """Get resource usage statistics"""
        with self._lock:
            return self.stats


class ResourceManager:
    """
    Main resource management system

    Coordinates multiple resource pools and provides unified interface
    for resource allocation across different resource types.
    """

    def __init__(self):
        """Initialize resource manager"""
        self.pools: Dict[ResourceType, ResourcePool] = {}
        self.requests: Dict[str, ResourceRequest] = {}
        self.allocations: Dict[str, Tuple[ResourceType, str]] = {}  # allocation_id -> (type, pool_alloc_id)
        self._lock = threading.RLock()
        self._lease_checker_thread = None
        self._stop_event = threading.Event()

        logger.info("ResourceManager initialized")

    def register_resource(
        self,
        spec: ResourceSpec,
        scheduling: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY
    ):
        """Register a new resource type"""
        with self._lock:
            if spec.resource_type in self.pools:
                raise ValueError(f"Resource type {spec.resource_type} already registered")

            pool = ResourcePool(spec, scheduling)
            self.pools[spec.resource_type] = pool

            logger.info(f"Registered resource: {spec.resource_type.value} with capacity {spec.total_capacity}")

    def request_resource(
        self,
        requester_id: str,
        resource_type: ResourceType,
        amount: float,
        priority: int = 0,
        timeout: Optional[float] = None,
        lease_duration: Optional[float] = None,
        wait: bool = True,
        **kwargs
    ) -> Optional[str]:
        """
        Request resource allocation

        Returns allocation ID if successful, None otherwise
        """
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        request = ResourceRequest(
            request_id=request_id,
            requester_id=requester_id,
            resource_type=resource_type,
            amount=amount,
            priority=priority,
            lease_duration=lease_duration,
            can_wait=wait,
            **kwargs
        )

        with self._lock:
            if resource_type not in self.pools:
                logger.error(f"Resource type {resource_type} not registered")
                return None

            pool = self.pools[resource_type]
            self.requests[request_id] = request

        # Try to allocate
        if wait:
            allocation = pool.wait_for_allocation(request, timeout)
        else:
            allocation = pool.request(request)

        if allocation:
            with self._lock:
                self.allocations[allocation.allocation_id] = (
                    resource_type,
                    allocation.allocation_id
                )
            return allocation.allocation_id

        return None

    def release_resource(self, allocation_id: str) -> bool:
        """Release allocated resource"""
        with self._lock:
            if allocation_id not in self.allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return False

            resource_type, pool_alloc_id = self.allocations[allocation_id]
            pool = self.pools.get(resource_type)

            if not pool:
                logger.error(f"Resource pool for {resource_type} not found")
                return False

        # Release from pool
        success = pool.release(pool_alloc_id)

        if success:
            with self._lock:
                del self.allocations[allocation_id]

        return success

    @contextmanager
    def allocate(
        self,
        requester_id: str,
        resource_type: ResourceType,
        amount: float,
        **kwargs
    ):
        """
        Context manager for resource allocation

        Usage:
            with manager.allocate("worker_1", ResourceType.CPU, 2.0) as alloc_id:
                # Use resource
                pass
            # Resource automatically released
        """
        allocation_id = self.request_resource(
            requester_id,
            resource_type,
            amount,
            **kwargs
        )

        if not allocation_id:
            raise RuntimeError(f"Failed to allocate {amount} of {resource_type}")

        try:
            yield allocation_id
        finally:
            self.release_resource(allocation_id)

    def start_lease_checker(self, interval: float = 10.0):
        """Start background thread to check for expired leases"""
        if self._lease_checker_thread is None or not self._lease_checker_thread.is_alive():
            self._stop_event.clear()
            self._lease_checker_thread = threading.Thread(
                target=self._lease_checker_loop,
                args=(interval,),
                name="LeaseChecker",
                daemon=True
            )
            self._lease_checker_thread.start()
            logger.info("Lease checker started")

    def stop_lease_checker(self):
        """Stop lease checker thread"""
        self._stop_event.set()
        if self._lease_checker_thread:
            self._lease_checker_thread.join(timeout=5.0)
        logger.info("Lease checker stopped")

    def _lease_checker_loop(self, interval: float):
        """Background loop to check for expired leases"""
        while not self._stop_event.is_set():
            try:
                total_reclaimed = 0
                with self._lock:
                    for pool in self.pools.values():
                        reclaimed = pool.check_leases()
                        total_reclaimed += len(reclaimed)

                if total_reclaimed > 0:
                    logger.info(f"Reclaimed {total_reclaimed} expired leases")

            except Exception as e:
                logger.error(f"Error in lease checker: {e}")

            self._stop_event.wait(interval)

    def get_utilization(self) -> Dict[ResourceType, float]:
        """Get current utilization for all resource types"""
        utilization = {}
        with self._lock:
            for resource_type, pool in self.pools.items():
                stats = pool.get_stats()
                utilization[resource_type] = stats.current_utilization

        return utilization

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'total_resource_types': len(self.pools),
            'total_active_requests': len(self.requests),
            'total_active_allocations': len(self.allocations),
            'pools': {}
        }

        with self._lock:
            for resource_type, pool in self.pools.items():
                pool_stats = pool.get_stats()
                stats['pools'][resource_type.value] = {
                    'total_capacity': pool.spec.total_capacity,
                    'available_capacity': pool.available_capacity,
                    'utilization': pool_stats.current_utilization,
                    'peak_utilization': pool_stats.peak_utilization,
                    'total_allocations': pool_stats.total_allocations,
                    'total_releases': pool_stats.total_releases,
                    'avg_wait_time': pool_stats.avg_wait_time,
                    'avg_hold_time': pool_stats.avg_hold_time,
                    'rejection_count': pool_stats.rejection_count,
                    'timeout_count': pool_stats.timeout_count,
                    'wait_queue_size': len(pool.wait_queue),
                    'scheduling': pool.scheduling.value
                }

        return stats

    def set_scheduling_algorithm(
        self,
        resource_type: ResourceType,
        algorithm: SchedulingAlgorithm
    ):
        """Change scheduling algorithm for a resource type"""
        with self._lock:
            if resource_type in self.pools:
                self.pools[resource_type].scheduling = algorithm
                logger.info(f"Changed scheduling for {resource_type.value} to {algorithm.value}")


class FairShareScheduler:
    """
    Fair share scheduling for resource allocation

    Ensures fair distribution of resources among multiple users/groups
    based on configured shares and historical usage.
    """

    def __init__(self, resource_manager: ResourceManager):
        """Initialize fair share scheduler"""
        self.manager = resource_manager
        self.user_shares: Dict[str, float] = {}  # user -> share weight
        self.user_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.history_window = 3600.0  # 1 hour window for usage history
        self._lock = threading.RLock()

    def set_user_share(self, user_id: str, share: float):
        """Set fair share weight for a user"""
        with self._lock:
            self.user_shares[user_id] = share

    def calculate_priority(
        self,
        user_id: str,
        resource_type: ResourceType,
        base_priority: int = 0
    ) -> int:
        """
        Calculate adjusted priority based on fair share

        Users who have used less than their fair share get higher priority
        """
        with self._lock:
            if user_id not in self.user_shares:
                return base_priority

            share = self.user_shares[user_id]
            usage = self.user_usage[user_id][resource_type]

            # Calculate fair share score
            total_shares = sum(self.user_shares.values())
            fair_share_ratio = share / max(1, total_shares)

            # Compare actual usage to fair share
            # Lower usage -> higher priority
            usage_factor = 1.0 - min(1.0, usage / (fair_share_ratio * 100))
            adjusted_priority = base_priority + int(usage_factor * 100)

            return adjusted_priority

    def request_with_fair_share(
        self,
        user_id: str,
        resource_type: ResourceType,
        amount: float,
        **kwargs
    ) -> Optional[str]:
        """Request resource with fair share scheduling"""
        # Calculate fair share priority
        base_priority = kwargs.get('priority', 0)
        adjusted_priority = self.calculate_priority(
            user_id,
            resource_type,
            base_priority
        )

        # Make request with adjusted priority
        kwargs['priority'] = adjusted_priority
        allocation_id = self.manager.request_resource(
            user_id,
            resource_type,
            amount,
            **kwargs
        )

        # Track usage
        if allocation_id:
            with self._lock:
                self.user_usage[user_id][resource_type] += amount

        return allocation_id


# Example usage
def _example_usage():
    """Example usage of resource manager"""

    # Create resource manager
    manager = ResourceManager()

    # Register resources
    manager.register_resource(
        ResourceSpec(
            resource_type=ResourceType.CPU,
            total_capacity=16.0,
            unit="cores",
            min_allocation=0.5,
            max_allocation=8.0,
            lease_duration=60.0  # 1 minute default lease
        ),
        scheduling=SchedulingAlgorithm.PRIORITY
    )

    manager.register_resource(
        ResourceSpec(
            resource_type=ResourceType.MEMORY,
            total_capacity=32768.0,
            unit="MB",
            min_allocation=256.0,
            max_allocation=8192.0
        ),
        scheduling=SchedulingAlgorithm.WEIGHTED_FAIR
    )

    # Start lease checker
    manager.start_lease_checker(interval=5.0)

    # Request resources
    print("Requesting CPU...")
    cpu_alloc = manager.request_resource(
        "worker_1",
        ResourceType.CPU,
        2.0,
        priority=10,
        lease_duration=30.0
    )
    print(f"Allocated CPU: {cpu_alloc}")

    # Use context manager
    print("\nUsing context manager...")
    with manager.allocate("worker_2", ResourceType.MEMORY, 1024.0) as mem_alloc:
        print(f"Allocated memory: {mem_alloc}")
        time.sleep(1)
    print("Memory released")

    # Check utilization
    utilization = manager.get_utilization()
    print(f"\nUtilization: {utilization}")

    # Get statistics
    stats = manager.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        if key != 'pools':
            print(f"  {key}: {value}")

    print("\nPool Statistics:")
    for pool_name, pool_stats in stats['pools'].items():
        print(f"  {pool_name}:")
        for stat_name, stat_value in pool_stats.items():
            print(f"    {stat_name}: {stat_value}")

    # Clean up
    if cpu_alloc:
        manager.release_resource(cpu_alloc)
    manager.stop_lease_checker()

    print("\nResource manager example completed!")


if __name__ == "__main__":
    _example_usage()