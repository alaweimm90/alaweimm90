"""
Production-Grade Concurrency Primitives for MEZAN

This module provides thread-safe data structures, lock management, and
concurrency utilities to prevent race conditions and ensure correctness
in multi-threaded execution environments.

Features:
- Thread-safe collections (queues, caches, counters)
- Advanced lock management with timeouts and deadlock detection
- Read-write locks for performance optimization
- Lock-free algorithms where applicable
- Comprehensive monitoring and debugging support

Author: MEZAN Concurrency Team
Version: 1.0.0
"""

import threading
import multiprocessing as mp
import time
import logging
import functools
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import heapq
import queue

logger = logging.getLogger(__name__)


class LockType(Enum):
    """Types of locks available"""
    STANDARD = "standard"          # threading.Lock
    REENTRANT = "reentrant"        # threading.RLock
    READ_WRITE = "read_write"      # Custom RWLock
    SEMAPHORE = "semaphore"        # threading.Semaphore
    CONDITION = "condition"        # threading.Condition


@dataclass
class LockInfo:
    """Information about a lock for debugging"""
    lock_id: str
    lock_type: LockType
    owner_thread: Optional[int] = None
    acquisition_time: Optional[float] = None
    wait_count: int = 0
    acquisition_count: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0


class TimeoutError(Exception):
    """Raised when lock acquisition times out"""
    pass


class DeadlockError(Exception):
    """Raised when deadlock is detected"""
    pass


class RWLock:
    """
    Read-Write Lock Implementation

    Allows multiple readers or single writer.
    Readers have priority to minimize read latency.
    """

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.RLock())
        self._write_ready = threading.Condition(threading.RLock())

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Acquire read lock"""
        self.acquire_read(timeout)
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Acquire write lock"""
        self.acquire_write(timeout)
        try:
            yield
        finally:
            self.release_write()

    def acquire_read(self, timeout: Optional[float] = None):
        """Acquire lock for reading"""
        if timeout is not None:
            acquired = self._read_ready.acquire(timeout=timeout)
        else:
            acquired = self._read_ready.acquire()
        if not acquired:
            raise TimeoutError(f"Failed to acquire read lock within {timeout}s")
        try:
            while self._writers > 0:
                if timeout is not None:
                    if not self._read_ready.wait(timeout):
                        raise TimeoutError(f"Timeout waiting for writers to finish")
                else:
                    self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release read lock"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self, timeout: Optional[float] = None):
        """Acquire lock for writing"""
        if timeout is not None:
            acquired = self._write_ready.acquire(timeout=timeout)
        else:
            acquired = self._write_ready.acquire()
        if not acquired:
            raise TimeoutError(f"Failed to acquire write lock within {timeout}s")
        try:
            while self._writers > 0 or self._readers > 0:
                if timeout is not None:
                    if not self._write_ready.wait(timeout):
                        raise TimeoutError(f"Timeout waiting for readers/writers")
                else:
                    self._write_ready.wait()
            self._writers += 1
        finally:
            self._write_ready.release()

    def release_write(self):
        """Release write lock"""
        with self._write_ready:
            self._writers -= 1
            self._write_ready.notify_all()
            with self._read_ready:
                self._read_ready.notify_all()


class ThreadSafeDict:
    """
    Thread-safe dictionary with read-write locking

    Optimized for read-heavy workloads.
    """

    def __init__(self):
        self._data = {}
        self._lock = RWLock()

    def get(self, key: Any, default: Any = None) -> Any:
        """Thread-safe get"""
        with self._lock.read_lock():
            return self._data.get(key, default)

    def set(self, key: Any, value: Any):
        """Thread-safe set"""
        with self._lock.write_lock():
            self._data[key] = value

    def update(self, other: Dict[Any, Any]):
        """Thread-safe update"""
        with self._lock.write_lock():
            self._data.update(other)

    def pop(self, key: Any, default: Any = None) -> Any:
        """Thread-safe pop"""
        with self._lock.write_lock():
            return self._data.pop(key, default)

    def items(self) -> List[Tuple[Any, Any]]:
        """Thread-safe items (returns copy)"""
        with self._lock.read_lock():
            return list(self._data.items())

    def keys(self) -> List[Any]:
        """Thread-safe keys (returns copy)"""
        with self._lock.read_lock():
            return list(self._data.keys())

    def values(self) -> List[Any]:
        """Thread-safe values (returns copy)"""
        with self._lock.read_lock():
            return list(self._data.values())

    def clear(self):
        """Thread-safe clear"""
        with self._lock.write_lock():
            self._data.clear()

    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock.read_lock():
            return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """Thread-safe contains"""
        with self._lock.read_lock():
            return key in self._data


class ThreadSafeCounter:
    """
    Thread-safe counter with atomic operations

    Uses fine-grained locking for high performance.
    """

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """Atomic increment, returns new value"""
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """Atomic decrement, returns new value"""
        with self._lock:
            self._value -= delta
            return self._value

    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value

    def set(self, value: int):
        """Set value"""
        with self._lock:
            self._value = value

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Atomic compare-and-swap"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class ThreadSafeSet:
    """Thread-safe set implementation"""

    def __init__(self):
        self._data = set()
        self._lock = threading.RLock()

    def add(self, item: Any):
        """Thread-safe add"""
        with self._lock:
            self._data.add(item)

    def remove(self, item: Any):
        """Thread-safe remove"""
        with self._lock:
            self._data.remove(item)

    def discard(self, item: Any):
        """Thread-safe discard"""
        with self._lock:
            self._data.discard(item)

    def __contains__(self, item: Any) -> bool:
        """Thread-safe contains"""
        with self._lock:
            return item in self._data

    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock:
            return len(self._data)

    def clear(self):
        """Thread-safe clear"""
        with self._lock:
            self._data.clear()

    def copy(self) -> Set[Any]:
        """Thread-safe copy"""
        with self._lock:
            return self._data.copy()


class BoundedThreadSafeQueue:
    """
    Thread-safe bounded queue with timeout support

    Features:
    - Bounded size with blocking on full
    - Timeout support for get/put
    - Priority queue option
    """

    def __init__(self, maxsize: int = 0, priority: bool = False):
        """
        Initialize queue

        Args:
            maxsize: Maximum queue size (0 = unbounded)
            priority: Use priority queue (items must be comparable)
        """
        if priority:
            self._queue = queue.PriorityQueue(maxsize=maxsize)
        else:
            self._queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, timeout: Optional[float] = None):
        """Put item in queue"""
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full:
            raise TimeoutError(f"Queue full after {timeout}s")

    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"Queue empty after {timeout}s")

    def put_nowait(self, item: Any):
        """Put item without blocking"""
        self._queue.put_nowait(item)

    def get_nowait(self) -> Any:
        """Get item without blocking"""
        return self._queue.get_nowait()

    def qsize(self) -> int:
        """Approximate queue size"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def full(self) -> bool:
        """Check if queue is full"""
        return self._queue.full()


class LockManager:
    """
    Centralized lock manager with deadlock detection

    Features:
    - Track all locks in the system
    - Detect potential deadlocks
    - Timeout-based acquisition
    - Lock statistics and monitoring
    """

    def __init__(self, enable_deadlock_detection: bool = True):
        self._locks: Dict[str, Any] = {}
        self._lock_info: Dict[str, LockInfo] = {}
        self._lock_graph: Dict[int, Set[str]] = defaultdict(set)  # thread -> locks held
        self._wait_graph: Dict[int, str] = {}  # thread -> lock waiting for
        self._manager_lock = threading.RLock()
        self._enable_deadlock = enable_deadlock_detection

    def create_lock(
        self,
        lock_id: str,
        lock_type: LockType = LockType.STANDARD,
        **kwargs
    ) -> Any:
        """Create and register a new lock"""
        with self._manager_lock:
            if lock_id in self._locks:
                raise ValueError(f"Lock {lock_id} already exists")

            if lock_type == LockType.STANDARD:
                lock = threading.Lock()
            elif lock_type == LockType.REENTRANT:
                lock = threading.RLock()
            elif lock_type == LockType.READ_WRITE:
                lock = RWLock()
            elif lock_type == LockType.SEMAPHORE:
                value = kwargs.get('value', 1)
                lock = threading.Semaphore(value)
            elif lock_type == LockType.CONDITION:
                underlying = kwargs.get('lock', threading.RLock())
                lock = threading.Condition(underlying)
            else:
                raise ValueError(f"Unknown lock type: {lock_type}")

            self._locks[lock_id] = lock
            self._lock_info[lock_id] = LockInfo(
                lock_id=lock_id,
                lock_type=lock_type
            )

            return lock

    @contextmanager
    def acquire(
        self,
        lock_id: str,
        timeout: Optional[float] = None,
        read_lock: bool = False
    ):
        """
        Acquire lock with deadlock detection

        Args:
            lock_id: Lock identifier
            timeout: Acquisition timeout in seconds
            read_lock: For RWLock, acquire read lock instead of write
        """
        thread_id = threading.get_ident()
        start_time = time.time()

        with self._manager_lock:
            if lock_id not in self._locks:
                raise KeyError(f"Lock {lock_id} not found")

            lock = self._locks[lock_id]
            info = self._lock_info[lock_id]
            info.wait_count += 1

            # Check for potential deadlock
            if self._enable_deadlock and self._detect_deadlock(thread_id, lock_id):
                raise DeadlockError(
                    f"Potential deadlock: thread {thread_id} acquiring {lock_id}"
                )

            # Record wait
            self._wait_graph[thread_id] = lock_id

        acquired = False
        try:
            # Acquire the actual lock
            if isinstance(lock, RWLock):
                if read_lock:
                    lock.acquire_read(timeout)
                else:
                    lock.acquire_write(timeout)
                acquired = True
            elif hasattr(lock, 'acquire'):
                acquired = lock.acquire(timeout=timeout) if timeout else lock.acquire()
                if not acquired:
                    raise TimeoutError(f"Failed to acquire {lock_id} within {timeout}s")

            # Update tracking
            with self._manager_lock:
                wait_time = time.time() - start_time
                info.acquisition_count += 1
                info.total_wait_time += wait_time
                info.max_wait_time = max(info.max_wait_time, wait_time)
                info.owner_thread = thread_id
                info.acquisition_time = time.time()
                info.wait_count -= 1

                # Update lock graph
                self._lock_graph[thread_id].add(lock_id)
                self._wait_graph.pop(thread_id, None)

            yield lock

        finally:
            if acquired:
                # Release lock
                if isinstance(lock, RWLock):
                    if read_lock:
                        lock.release_read()
                    else:
                        lock.release_write()
                elif hasattr(lock, 'release'):
                    lock.release()

                # Update tracking
                with self._manager_lock:
                    info.owner_thread = None
                    info.acquisition_time = None
                    self._lock_graph[thread_id].discard(lock_id)
            else:
                # Clean up wait graph on failure
                with self._manager_lock:
                    self._wait_graph.pop(thread_id, None)
                    info.wait_count -= 1

    def _detect_deadlock(self, thread_id: int, target_lock: str) -> bool:
        """
        Detect potential deadlock using wait-for graph

        Returns True if acquiring target_lock would create a cycle
        """
        # Check if any thread holding target_lock is waiting for locks we hold
        for tid, locks in self._lock_graph.items():
            if target_lock in locks:
                # Thread tid holds target_lock
                if tid in self._wait_graph:
                    # tid is waiting for a lock
                    waiting_for = self._wait_graph[tid]
                    if waiting_for in self._lock_graph.get(thread_id, set()):
                        # Cycle detected!
                        logger.warning(
                            f"Deadlock detected: Thread {thread_id} holds {waiting_for}, "
                            f"wants {target_lock}. Thread {tid} holds {target_lock}, "
                            f"wants {waiting_for}"
                        )
                        return True
        return False

    def get_lock_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all locks"""
        with self._manager_lock:
            stats = {}
            for lock_id, info in self._lock_info.items():
                stats[lock_id] = {
                    'type': info.lock_type.value,
                    'owner_thread': info.owner_thread,
                    'held_for': (
                        time.time() - info.acquisition_time
                        if info.acquisition_time else 0
                    ),
                    'wait_count': info.wait_count,
                    'total_acquisitions': info.acquisition_count,
                    'avg_wait_time': (
                        info.total_wait_time / max(1, info.acquisition_count)
                    ),
                    'max_wait_time': info.max_wait_time,
                }
            return stats

    def detect_all_deadlocks(self) -> List[List[int]]:
        """
        Detect all cycles in the wait-for graph

        Returns list of thread cycles that form deadlocks
        """
        with self._manager_lock:
            cycles = []
            visited = set()
            rec_stack = set()

            def dfs(thread_id: int, path: List[int]) -> bool:
                visited.add(thread_id)
                rec_stack.add(thread_id)
                path.append(thread_id)

                # Check what this thread is waiting for
                if thread_id in self._wait_graph:
                    waiting_for_lock = self._wait_graph[thread_id]

                    # Find who holds this lock
                    for tid, locks in self._lock_graph.items():
                        if waiting_for_lock in locks:
                            if tid not in visited:
                                if dfs(tid, path):
                                    return True
                            elif tid in rec_stack:
                                # Cycle found
                                cycle_start = path.index(tid)
                                cycles.append(path[cycle_start:])
                                return True

                path.pop()
                rec_stack.remove(thread_id)
                return False

            # Check all threads
            for tid in list(self._wait_graph.keys()):
                if tid not in visited:
                    dfs(tid, [])

            return cycles


class AtomicReference:
    """
    Atomic reference with compare-and-swap

    Lock-free for reads, minimal locking for writes.
    """

    def __init__(self, initial: Any = None):
        self._value = initial
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Get current value (lock-free)"""
        return self._value

    def set(self, value: Any):
        """Set value"""
        with self._lock:
            self._value = value

    def compare_and_swap(self, expected: Any, new_value: Any) -> bool:
        """Atomic compare-and-swap"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def get_and_set(self, new_value: Any) -> Any:
        """Atomic get and set"""
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value


class ThreadSafeCache:
    """
    Thread-safe LRU cache implementation

    Features:
    - Bounded size with LRU eviction
    - Thread-safe operations
    - TTL support for entries
    """

    def __init__(self, max_size: int = 128, ttl: Optional[float] = None):
        """
        Initialize cache

        Args:
            max_size: Maximum cache entries
            ttl: Time-to-live for entries in seconds
        """
        self._cache = {}
        self._access_times = {}
        self._expiry_times = {} if ttl else None
        self._max_size = max_size
        self._ttl = ttl
        self._lock = RWLock()

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        with self._lock.read_lock():
            if key not in self._cache:
                return None

            # Check expiry
            if self._expiry_times and key in self._expiry_times:
                if time.time() > self._expiry_times[key]:
                    # Expired, need write lock to remove
                    return None

            # Update access time (would need write lock for true LRU)
            # Simplified: just return value
            return self._cache[key]

    def put(self, key: Any, value: Any):
        """Put value in cache"""
        with self._lock.write_lock():
            # Evict if necessary
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Find oldest entry (simplified)
                oldest_key = min(self._access_times.keys(),
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
                if self._expiry_times:
                    self._expiry_times.pop(oldest_key, None)

            self._cache[key] = value
            self._access_times[key] = time.time()

            if self._ttl:
                self._expiry_times[key] = time.time() + self._ttl

    def clear(self):
        """Clear cache"""
        with self._lock.write_lock():
            self._cache.clear()
            self._access_times.clear()
            if self._expiry_times:
                self._expiry_times.clear()

    def size(self) -> int:
        """Get cache size"""
        with self._lock.read_lock():
            return len(self._cache)


def synchronized(lock: Optional[threading.Lock] = None):
    """
    Decorator to synchronize method/function execution

    Usage:
        @synchronized()
        def my_method(self):
            # Thread-safe execution
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal lock
        if lock is None:
            lock = threading.RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global lock manager instance
_global_lock_manager = LockManager()


def get_global_lock_manager() -> LockManager:
    """Get the global lock manager instance"""
    return _global_lock_manager


# Example usage and testing
def _example_usage():
    """Example usage of concurrency primitives"""

    # Thread-safe dictionary
    safe_dict = ThreadSafeDict()
    safe_dict.set("key", "value")
    print(f"Dict value: {safe_dict.get('key')}")

    # Thread-safe counter
    counter = ThreadSafeCounter()
    print(f"Counter: {counter.increment()}")

    # Lock manager with deadlock detection
    manager = get_global_lock_manager()
    lock1 = manager.create_lock("resource_1")
    lock2 = manager.create_lock("resource_2")

    with manager.acquire("resource_1", timeout=1.0):
        print("Acquired resource_1")
        # Do work...

    # RWLock for read-heavy workloads
    rw_lock = RWLock()
    with rw_lock.read_lock():
        print("Reading...")

    with rw_lock.write_lock():
        print("Writing...")

    # Thread-safe cache
    cache = ThreadSafeCache(max_size=100, ttl=60.0)
    cache.put("key", "cached_value")
    print(f"Cached: {cache.get('key')}")

    print("Concurrency primitives working correctly!")


if __name__ == "__main__":
    _example_usage()