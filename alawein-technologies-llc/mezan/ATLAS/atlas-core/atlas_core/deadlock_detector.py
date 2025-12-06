"""
Deadlock Detection and Resolution System for MEZAN

This module provides sophisticated deadlock detection, prevention, and
resolution mechanisms for multi-threaded and multi-process environments.

Features:
- Wait-for graph construction and cycle detection
- Resource allocation graph (RAG) analysis
- Automatic deadlock resolution strategies
- Timeout-based deadlock prevention
- Comprehensive logging and alerting
- Performance monitoring and metrics

Author: MEZAN Concurrency Team
Version: 1.0.0
"""

import threading
import time
import logging
import signal
import os
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import traceback

logger = logging.getLogger(__name__)


class DeadlockResolutionStrategy(Enum):
    """Strategies for resolving detected deadlocks"""
    TIMEOUT = "timeout"                      # Use timeouts to break deadlock
    VICTIM_SELECTION = "victim_selection"    # Choose victim thread to abort
    RESOURCE_PREEMPTION = "resource_preemption"  # Preempt resources
    WAIT_DIE = "wait_die"                   # Older waits, younger dies
    WOUND_WAIT = "wound_wait"               # Older wounds younger
    PRIORITY_BASED = "priority_based"       # Based on thread priority


@dataclass
class Resource:
    """Resource representation for deadlock detection"""
    resource_id: str
    resource_type: str
    max_instances: int = 1
    available_instances: int = 1
    holders: Set[int] = field(default_factory=set)  # Thread IDs holding resource
    waiters: List[int] = field(default_factory=list)  # Thread IDs waiting
    priority: int = 0  # Higher priority resources are less likely to be preempted
    preemptible: bool = True
    acquisition_count: int = 0
    last_acquisition_time: float = 0.0


@dataclass
class ThreadInfo:
    """Information about a thread for deadlock detection"""
    thread_id: int
    thread_name: str
    start_time: float
    priority: int = 0
    resources_held: Set[str] = field(default_factory=set)
    resources_waiting: Optional[str] = None
    wait_start_time: Optional[float] = None
    transaction_id: Optional[str] = None
    can_be_victim: bool = True
    restart_count: int = 0


@dataclass
class DeadlockEvent:
    """Record of a detected deadlock"""
    timestamp: float
    cycle: List[int]  # Thread IDs forming the cycle
    resources_involved: Set[str]
    resolution_strategy: DeadlockResolutionStrategy
    victim_thread: Optional[int] = None
    resolved: bool = False
    resolution_time: Optional[float] = None


class DeadlockDetector:
    """
    Main deadlock detection and resolution system

    Uses multiple algorithms for comprehensive deadlock handling:
    1. Wait-for graph for cycle detection
    2. Resource allocation graph for resource analysis
    3. Banker's algorithm for deadlock prevention
    4. Various resolution strategies
    """

    def __init__(
        self,
        check_interval: float = 1.0,
        enable_auto_resolution: bool = True,
        resolution_strategy: DeadlockResolutionStrategy = DeadlockResolutionStrategy.VICTIM_SELECTION,
        max_wait_time: float = 30.0
    ):
        """
        Initialize deadlock detector

        Args:
            check_interval: How often to check for deadlocks (seconds)
            enable_auto_resolution: Automatically resolve detected deadlocks
            resolution_strategy: Strategy to use for resolution
            max_wait_time: Maximum time to wait before considering it a deadlock
        """
        self.check_interval = check_interval
        self.enable_auto_resolution = enable_auto_resolution
        self.resolution_strategy = resolution_strategy
        self.max_wait_time = max_wait_time

        # Thread and resource tracking
        self.threads: Dict[int, ThreadInfo] = {}
        self.resources: Dict[str, Resource] = {}

        # Wait-for graph: thread -> thread (indirect through resources)
        self.wait_for_graph: Dict[int, Set[int]] = defaultdict(set)

        # Statistics
        self.deadlock_events: List[DeadlockEvent] = []
        self.total_detections = 0
        self.total_resolutions = 0
        self.false_positives = 0

        # Synchronization
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._detector_thread = None

        logger.info(
            f"DeadlockDetector initialized: check_interval={check_interval}s, "
            f"auto_resolution={enable_auto_resolution}, strategy={resolution_strategy.value}"
        )

    def start(self):
        """Start the deadlock detection thread"""
        if self._detector_thread is None or not self._detector_thread.is_alive():
            self._stop_event.clear()
            self._detector_thread = threading.Thread(
                target=self._detection_loop,
                name="DeadlockDetector",
                daemon=True
            )
            self._detector_thread.start()
            logger.info("Deadlock detection started")

    def stop(self):
        """Stop the deadlock detection thread"""
        self._stop_event.set()
        if self._detector_thread:
            self._detector_thread.join(timeout=5.0)
        logger.info("Deadlock detection stopped")

    def register_thread(
        self,
        thread_id: Optional[int] = None,
        thread_name: Optional[str] = None,
        priority: int = 0,
        can_be_victim: bool = True
    ) -> int:
        """Register a thread for deadlock detection"""
        if thread_id is None:
            thread_id = threading.get_ident()
        if thread_name is None:
            thread_name = threading.current_thread().name

        with self._lock:
            self.threads[thread_id] = ThreadInfo(
                thread_id=thread_id,
                thread_name=thread_name,
                start_time=time.time(),
                priority=priority,
                can_be_victim=can_be_victim
            )

        logger.debug(f"Thread registered: {thread_id} ({thread_name})")
        return thread_id

    def unregister_thread(self, thread_id: Optional[int] = None):
        """Unregister a thread"""
        if thread_id is None:
            thread_id = threading.get_ident()

        with self._lock:
            if thread_id in self.threads:
                # Release any held resources
                thread_info = self.threads[thread_id]
                for resource_id in list(thread_info.resources_held):
                    self.release_resource(resource_id, thread_id)

                del self.threads[thread_id]
                logger.debug(f"Thread unregistered: {thread_id}")

    def register_resource(
        self,
        resource_id: str,
        resource_type: str = "lock",
        max_instances: int = 1,
        priority: int = 0,
        preemptible: bool = True
    ) -> str:
        """Register a resource"""
        with self._lock:
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type=resource_type,
                max_instances=max_instances,
                available_instances=max_instances,
                priority=priority,
                preemptible=preemptible
            )

        logger.debug(f"Resource registered: {resource_id} (type={resource_type})")
        return resource_id

    def request_resource(
        self,
        resource_id: str,
        thread_id: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Request a resource (blocking)

        Returns True if acquired, False if deadlock would occur
        """
        if thread_id is None:
            thread_id = threading.get_ident()

        with self._lock:
            # Ensure thread is registered
            if thread_id not in self.threads:
                self.register_thread(thread_id)

            # Ensure resource exists
            if resource_id not in self.resources:
                logger.warning(f"Resource {resource_id} not registered")
                return False

            resource = self.resources[resource_id]
            thread_info = self.threads[thread_id]

            # Check for immediate availability
            if resource.available_instances > 0:
                # Grant resource
                resource.available_instances -= 1
                resource.holders.add(thread_id)
                thread_info.resources_held.add(resource_id)
                resource.acquisition_count += 1
                resource.last_acquisition_time = time.time()
                self._update_wait_for_graph()
                return True

            # Would need to wait - check for potential deadlock
            if self._would_cause_deadlock(thread_id, resource_id):
                logger.warning(
                    f"Deadlock prevention: Thread {thread_id} cannot acquire {resource_id}"
                )
                return False

            # Record wait
            thread_info.resources_waiting = resource_id
            thread_info.wait_start_time = time.time()
            resource.waiters.append(thread_id)
            self._update_wait_for_graph()

        # Wait for resource (simplified - in production would use conditions)
        start_time = time.time()
        while True:
            time.sleep(0.01)  # Polling (simplified)

            with self._lock:
                if thread_id not in self.threads:
                    # Thread was unregistered
                    return False

                if resource.available_instances > 0:
                    # Resource available
                    resource.available_instances -= 1
                    resource.holders.add(thread_id)
                    thread_info.resources_held.add(resource_id)
                    thread_info.resources_waiting = None
                    thread_info.wait_start_time = None
                    if thread_id in resource.waiters:
                        resource.waiters.remove(thread_id)
                    self._update_wait_for_graph()
                    return True

            if timeout and (time.time() - start_time) > timeout:
                # Timeout
                with self._lock:
                    thread_info.resources_waiting = None
                    thread_info.wait_start_time = None
                    if thread_id in resource.waiters:
                        resource.waiters.remove(thread_id)
                    self._update_wait_for_graph()
                return False

    def release_resource(
        self,
        resource_id: str,
        thread_id: Optional[int] = None
    ):
        """Release a resource"""
        if thread_id is None:
            thread_id = threading.get_ident()

        with self._lock:
            if resource_id not in self.resources:
                logger.warning(f"Resource {resource_id} not found")
                return

            resource = self.resources[resource_id]

            if thread_id in resource.holders:
                resource.holders.remove(thread_id)
                resource.available_instances += 1

                if thread_id in self.threads:
                    self.threads[thread_id].resources_held.discard(resource_id)

                self._update_wait_for_graph()
                logger.debug(f"Thread {thread_id} released {resource_id}")

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        while not self._stop_event.is_set():
            try:
                cycles = self.detect_deadlocks()

                if cycles:
                    logger.warning(f"Detected {len(cycles)} deadlock cycles")
                    self.total_detections += len(cycles)

                    if self.enable_auto_resolution:
                        for cycle in cycles:
                            self._resolve_deadlock(cycle)

                # Check for timeout-based deadlocks
                self._check_timeout_deadlocks()

            except Exception as e:
                logger.error(f"Error in deadlock detection: {e}")
                logger.debug(traceback.format_exc())

            self._stop_event.wait(self.check_interval)

    def detect_deadlocks(self) -> List[List[int]]:
        """
        Detect all deadlock cycles in the system

        Returns list of cycles (each cycle is a list of thread IDs)
        """
        with self._lock:
            return self._find_cycles_in_wait_graph()

    def _find_cycles_in_wait_graph(self) -> List[List[int]]:
        """Find all cycles in the wait-for graph using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(thread_id: int, path: List[int]) -> None:
            visited.add(thread_id)
            rec_stack.add(thread_id)
            path.append(thread_id)

            for next_thread in self.wait_for_graph.get(thread_id, []):
                if next_thread not in visited:
                    dfs(next_thread, path[:])
                elif next_thread in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_thread)
                    cycle = path[cycle_start:]
                    # Normalize cycle (start with smallest ID)
                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]
                    if normalized not in cycles:
                        cycles.append(normalized)

            rec_stack.remove(thread_id)

        # Check all threads
        for tid in list(self.threads.keys()):
            if tid not in visited:
                dfs(tid, [])

        return cycles

    def _update_wait_for_graph(self):
        """Update the wait-for graph based on current state"""
        self.wait_for_graph.clear()

        for thread_id, thread_info in self.threads.items():
            if thread_info.resources_waiting:
                # Thread is waiting for a resource
                resource = self.resources.get(thread_info.resources_waiting)
                if resource:
                    # Add edges to all threads holding this resource
                    for holder_id in resource.holders:
                        if holder_id != thread_id:
                            self.wait_for_graph[thread_id].add(holder_id)

    def _would_cause_deadlock(self, thread_id: int, resource_id: str) -> bool:
        """
        Check if granting resource to thread would cause deadlock

        Uses Banker's algorithm concept
        """
        # Simulate granting the resource
        resource = self.resources[resource_id]

        # Check if any holder of this resource is waiting for resources we hold
        thread_info = self.threads[thread_id]
        for holder_id in resource.holders:
            if holder_id in self.threads:
                holder_info = self.threads[holder_id]
                if holder_info.resources_waiting:
                    # Check if holder is waiting for any resource we hold
                    if holder_info.resources_waiting in thread_info.resources_held:
                        # Direct cycle detected
                        return True

        # More sophisticated check: simulate and detect cycles
        # (Simplified for demonstration)
        return False

    def _check_timeout_deadlocks(self):
        """Check for threads waiting too long (timeout-based detection)"""
        current_time = time.time()

        with self._lock:
            for thread_id, thread_info in self.threads.items():
                if thread_info.wait_start_time:
                    wait_duration = current_time - thread_info.wait_start_time
                    if wait_duration > self.max_wait_time:
                        logger.warning(
                            f"Thread {thread_id} has been waiting for "
                            f"{wait_duration:.2f}s (exceeds max {self.max_wait_time}s)"
                        )
                        # Consider this a deadlock
                        self._handle_timeout_deadlock(thread_id)

    def _handle_timeout_deadlock(self, thread_id: int):
        """Handle a timeout-based deadlock"""
        thread_info = self.threads.get(thread_id)
        if not thread_info:
            return

        # Create deadlock event
        event = DeadlockEvent(
            timestamp=time.time(),
            cycle=[thread_id],
            resources_involved={thread_info.resources_waiting} if thread_info.resources_waiting else set(),
            resolution_strategy=DeadlockResolutionStrategy.TIMEOUT
        )

        self.deadlock_events.append(event)

        if self.enable_auto_resolution:
            # Cancel the wait
            thread_info.resources_waiting = None
            thread_info.wait_start_time = None

            # Remove from waiters list
            if thread_info.resources_waiting in self.resources:
                resource = self.resources[thread_info.resources_waiting]
                if thread_id in resource.waiters:
                    resource.waiters.remove(thread_id)

            event.resolved = True
            event.resolution_time = time.time()
            self.total_resolutions += 1

    def _resolve_deadlock(self, cycle: List[int]):
        """
        Resolve a detected deadlock cycle

        Uses the configured resolution strategy
        """
        logger.info(f"Resolving deadlock cycle: {cycle}")

        # Collect resources involved
        resources_involved = set()
        for tid in cycle:
            if tid in self.threads:
                thread_info = self.threads[tid]
                resources_involved.update(thread_info.resources_held)
                if thread_info.resources_waiting:
                    resources_involved.add(thread_info.resources_waiting)

        # Create event
        event = DeadlockEvent(
            timestamp=time.time(),
            cycle=cycle,
            resources_involved=resources_involved,
            resolution_strategy=self.resolution_strategy
        )
        self.deadlock_events.append(event)

        # Apply resolution strategy
        if self.resolution_strategy == DeadlockResolutionStrategy.VICTIM_SELECTION:
            victim = self._select_victim(cycle)
            if victim:
                self._abort_thread(victim)
                event.victim_thread = victim
                event.resolved = True

        elif self.resolution_strategy == DeadlockResolutionStrategy.RESOURCE_PREEMPTION:
            if self._preempt_resources(cycle):
                event.resolved = True

        elif self.resolution_strategy == DeadlockResolutionStrategy.WAIT_DIE:
            self._apply_wait_die(cycle)
            event.resolved = True

        elif self.resolution_strategy == DeadlockResolutionStrategy.WOUND_WAIT:
            self._apply_wound_wait(cycle)
            event.resolved = True

        elif self.resolution_strategy == DeadlockResolutionStrategy.TIMEOUT:
            # Already handled by timeout detection
            pass

        if event.resolved:
            event.resolution_time = time.time()
            self.total_resolutions += 1
            logger.info(f"Deadlock resolved using {self.resolution_strategy.value}")

    def _select_victim(self, cycle: List[int]) -> Optional[int]:
        """
        Select a victim thread to abort

        Criteria:
        1. Can be victim (flag)
        2. Lowest priority
        3. Newest thread
        4. Least resources held
        5. Most restart count (to avoid starvation)
        """
        candidates = []

        for tid in cycle:
            if tid in self.threads:
                thread_info = self.threads[tid]
                if thread_info.can_be_victim:
                    score = (
                        -thread_info.priority * 1000 +           # Lower priority is better victim
                        (time.time() - thread_info.start_time) +  # Newer is better victim
                        len(thread_info.resources_held) * 10 +    # Less resources is better
                        thread_info.restart_count * 100           # More restarts = avoid victim
                    )
                    candidates.append((score, tid))

        if candidates:
            candidates.sort()
            return candidates[0][1]

        # No suitable victim found
        logger.warning("No suitable victim found for deadlock resolution")
        return None

    def _abort_thread(self, thread_id: int):
        """
        Abort a thread (victim)

        In practice, this would send a signal or exception to the thread
        """
        logger.info(f"Aborting thread {thread_id} as deadlock victim")

        thread_info = self.threads.get(thread_id)
        if not thread_info:
            return

        # Release all held resources
        for resource_id in list(thread_info.resources_held):
            self.release_resource(resource_id, thread_id)

        # Clear waiting status
        if thread_info.resources_waiting:
            resource = self.resources.get(thread_info.resources_waiting)
            if resource and thread_id in resource.waiters:
                resource.waiters.remove(thread_id)

        thread_info.resources_waiting = None
        thread_info.wait_start_time = None
        thread_info.restart_count += 1

        self._update_wait_for_graph()

    def _preempt_resources(self, cycle: List[int]) -> bool:
        """
        Preempt resources to break deadlock

        Returns True if successful
        """
        # Find preemptible resources in the cycle
        for tid in cycle:
            if tid in self.threads:
                thread_info = self.threads[tid]
                for resource_id in thread_info.resources_held:
                    resource = self.resources.get(resource_id)
                    if resource and resource.preemptible:
                        # Preempt this resource
                        logger.info(f"Preempting resource {resource_id} from thread {tid}")
                        self.release_resource(resource_id, tid)
                        return True

        return False

    def _apply_wait_die(self, cycle: List[int]):
        """
        Apply Wait-Die algorithm

        Older threads wait, younger threads die (are aborted)
        """
        # Sort by age (start_time)
        threads_by_age = []
        for tid in cycle:
            if tid in self.threads:
                thread_info = self.threads[tid]
                threads_by_age.append((thread_info.start_time, tid))

        threads_by_age.sort()

        # Abort all but the oldest
        for _, tid in threads_by_age[1:]:
            self._abort_thread(tid)

    def _apply_wound_wait(self, cycle: List[int]):
        """
        Apply Wound-Wait algorithm

        Older threads wound (preempt) younger threads
        """
        # Find oldest thread
        oldest_tid = None
        oldest_time = float('inf')

        for tid in cycle:
            if tid in self.threads:
                thread_info = self.threads[tid]
                if thread_info.start_time < oldest_time:
                    oldest_time = thread_info.start_time
                    oldest_tid = tid

        if oldest_tid:
            # Preempt resources from younger threads for the oldest
            for tid in cycle:
                if tid != oldest_tid:
                    self._abort_thread(tid)

    def get_statistics(self) -> Dict[str, Any]:
        """Get deadlock detection statistics"""
        with self._lock:
            active_waits = sum(
                1 for t in self.threads.values()
                if t.resources_waiting is not None
            )

            avg_wait_time = 0.0
            max_wait_time = 0.0
            current_time = time.time()

            for thread_info in self.threads.values():
                if thread_info.wait_start_time:
                    wait_time = current_time - thread_info.wait_start_time
                    avg_wait_time += wait_time
                    max_wait_time = max(max_wait_time, wait_time)

            if active_waits > 0:
                avg_wait_time /= active_waits

            return {
                'total_threads': len(self.threads),
                'total_resources': len(self.resources),
                'active_waits': active_waits,
                'total_detections': self.total_detections,
                'total_resolutions': self.total_resolutions,
                'false_positives': self.false_positives,
                'recent_events': len(self.deadlock_events[-10:]),
                'avg_wait_time': avg_wait_time,
                'max_wait_time': max_wait_time,
                'resolution_strategy': self.resolution_strategy.value,
                'auto_resolution_enabled': self.enable_auto_resolution,
            }

    def visualize_wait_graph(self) -> str:
        """Generate a text visualization of the wait-for graph"""
        with self._lock:
            lines = ["Wait-For Graph:"]
            lines.append("-" * 40)

            for thread_id, waiting_for in self.wait_for_graph.items():
                thread_name = self.threads[thread_id].thread_name if thread_id in self.threads else "Unknown"
                if waiting_for:
                    for target_id in waiting_for:
                        target_name = self.threads[target_id].thread_name if target_id in self.threads else "Unknown"
                        lines.append(f"  {thread_name}({thread_id}) -> {target_name}({target_id})")

            if len(lines) == 2:
                lines.append("  (No waits)")

            lines.append("-" * 40)
            lines.append(f"Cycles detected: {len(self._find_cycles_in_wait_graph())}")

            return "\n".join(lines)


# Example usage
def _example_usage():
    """Example usage of deadlock detector"""

    # Create detector
    detector = DeadlockDetector(
        check_interval=0.5,
        enable_auto_resolution=True,
        resolution_strategy=DeadlockResolutionStrategy.VICTIM_SELECTION
    )

    # Start detection
    detector.start()

    # Register resources
    detector.register_resource("lock_a", "mutex")
    detector.register_resource("lock_b", "mutex")

    # Simulate potential deadlock scenario
    import threading

    def thread_1():
        tid = detector.register_thread(thread_name="Thread-1")
        detector.request_resource("lock_a", tid)
        time.sleep(0.1)
        detector.request_resource("lock_b", tid)
        detector.release_resource("lock_b", tid)
        detector.release_resource("lock_a", tid)
        detector.unregister_thread(tid)

    def thread_2():
        tid = detector.register_thread(thread_name="Thread-2")
        detector.request_resource("lock_b", tid)
        time.sleep(0.1)
        detector.request_resource("lock_a", tid)
        detector.release_resource("lock_a", tid)
        detector.release_resource("lock_b", tid)
        detector.unregister_thread(tid)

    # This would create a deadlock without detection
    t1 = threading.Thread(target=thread_1)
    t2 = threading.Thread(target=thread_2)

    t1.start()
    t2.start()

    # Wait a bit
    time.sleep(2)

    # Check statistics
    stats = detector.get_statistics()
    print("Deadlock Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize wait graph
    print("\n" + detector.visualize_wait_graph())

    # Stop detector
    detector.stop()

    t1.join(timeout=1)
    t2.join(timeout=1)

    print("\nDeadlock detector example completed!")


if __name__ == "__main__":
    _example_usage()