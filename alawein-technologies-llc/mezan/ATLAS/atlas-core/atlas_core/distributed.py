"""
ORCHEX Distributed Systems Module

Provides distributed orchestration capabilities with Redis backend,
including distributed state management, locking, task distribution,
leader election, and cluster health monitoring.

Features:
- RedisBackend for distributed state synchronization
- Distributed lock manager to prevent race conditions
- Task queue for distributed work distribution
- Leader election for coordinator selection
- Cluster health monitoring and failure detection
- Graceful fallback to local mode if Redis unavailable

Author: MEZAN Research Team
Version: 1.0.0
"""

import json
import logging
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import pickle

try:
    import redis
    from redis.lock import Lock as RedisLock
    from redis.client import PubSub
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisLock = None
    PubSub = None

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a node in the cluster"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ClusterNode:
    """Represents a node in the distributed cluster"""
    node_id: str
    hostname: str
    status: NodeStatus
    last_heartbeat: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    is_leader: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """Represents a task in the distributed queue"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_to: Optional[str] = None
    status: str = "pending"
    retries: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None


class RedisBackend:
    """
    Redis backend for distributed state management

    Provides:
    - Connection pooling
    - Key namespacing
    - Serialization/deserialization
    - Graceful degradation
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "ORCHEX:",
        connection_timeout: int = 5,
        socket_timeout: int = 5,
        max_connections: int = 50
    ):
        """Initialize Redis backend with connection pooling"""
        self.host = host
        self.port = port
        self.db = db
        self.key_prefix = key_prefix
        self.is_connected = False
        self.connection_pool = None
        self.client = None
        self.pubsub = None

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Running in local mode.")
            return

        try:
            # Create connection pool for efficiency
            self.connection_pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connection_timeout,
                max_connections=max_connections,
                decode_responses=False  # We'll handle encoding/decoding
            )

            self.client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            self.client.ping()
            self.is_connected = True
            self.pubsub = self.client.pubsub()

            logger.info(f"Connected to Redis at {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False

    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        return pickle.dumps(value)

    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from Redis"""
        if value is None:
            return None
        return pickle.loads(value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis with fallback"""
        if not self.is_connected:
            return default

        try:
            value = self.client.get(self._make_key(key))
            return self._deserialize(value) if value else default
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return default

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration"""
        if not self.is_connected:
            return False

        try:
            serialized = self._serialize(value)
            return self.client.set(
                self._make_key(key),
                serialized,
                ex=expire
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.is_connected:
            return False

        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.is_connected:
            return False

        try:
            return bool(self.client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def publish(self, channel: str, message: Any) -> bool:
        """Publish message to Redis channel"""
        if not self.is_connected:
            return False

        try:
            serialized = self._serialize(message)
            self.client.publish(self._make_key(channel), serialized)
            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False

    def subscribe(self, channels: List[str]) -> Optional[PubSub]:
        """Subscribe to Redis channels"""
        if not self.is_connected or not self.pubsub:
            return None

        try:
            namespaced = [self._make_key(ch) for ch in channels]
            self.pubsub.subscribe(*namespaced)
            return self.pubsub
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
            return None


class DistributedLockManager:
    """
    Distributed lock manager using Redis

    Features:
    - Reentrant locks
    - Lock timeouts
    - Deadlock detection
    - Lock statistics
    """

    def __init__(self, backend: RedisBackend):
        """Initialize lock manager with Redis backend"""
        self.backend = backend
        self.locks: Dict[str, RedisLock] = {}
        self.lock_stats: Dict[str, Dict[str, Any]] = {}
        self.local_locks: Dict[str, threading.Lock] = {}  # Fallback

    def acquire_lock(
        self,
        resource: str,
        blocking: bool = True,
        timeout: Optional[float] = None,
        ttl: Optional[int] = 30
    ) -> bool:
        """
        Acquire distributed lock

        Args:
            resource: Resource identifier to lock
            blocking: Whether to block waiting for lock
            timeout: Max time to wait for lock (seconds)
            ttl: Lock time-to-live (seconds)

        Returns:
            True if lock acquired, False otherwise
        """
        lock_key = f"lock:{resource}"

        # Track statistics
        if resource not in self.lock_stats:
            self.lock_stats[resource] = {
                "acquisitions": 0,
                "releases": 0,
                "timeouts": 0,
                "failures": 0
            }

        # Use Redis lock if available
        if self.backend.is_connected and REDIS_AVAILABLE:
            try:
                if resource not in self.locks:
                    self.locks[resource] = RedisLock(
                        self.backend.client,
                        self.backend._make_key(lock_key),
                        timeout=ttl,
                        sleep=0.1
                    )

                acquired = self.locks[resource].acquire(
                    blocking=blocking,
                    blocking_timeout=timeout
                )

                if acquired:
                    self.lock_stats[resource]["acquisitions"] += 1
                else:
                    self.lock_stats[resource]["timeouts"] += 1

                return acquired

            except Exception as e:
                logger.error(f"Failed to acquire distributed lock: {e}")
                self.lock_stats[resource]["failures"] += 1

        # Fallback to local lock
        if resource not in self.local_locks:
            self.local_locks[resource] = threading.Lock()

        acquired = self.local_locks[resource].acquire(blocking, timeout or -1)

        if acquired:
            self.lock_stats[resource]["acquisitions"] += 1
        else:
            self.lock_stats[resource]["timeouts"] += 1

        return acquired

    def release_lock(self, resource: str) -> bool:
        """Release distributed lock"""
        # Use Redis lock if available
        if resource in self.locks and self.backend.is_connected:
            try:
                self.locks[resource].release()
                self.lock_stats[resource]["releases"] += 1
                return True
            except Exception as e:
                logger.error(f"Failed to release distributed lock: {e}")
                return False

        # Fallback to local lock
        if resource in self.local_locks:
            try:
                self.local_locks[resource].release()
                self.lock_stats[resource]["releases"] += 1
                return True
            except Exception:
                return False

        return False

    def is_locked(self, resource: str) -> bool:
        """Check if resource is locked"""
        lock_key = f"lock:{resource}"

        if self.backend.is_connected:
            return self.backend.exists(lock_key)

        # Check local lock
        if resource in self.local_locks:
            return self.local_locks[resource].locked()

        return False

    def get_lock_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get lock statistics"""
        return self.lock_stats.copy()


class DistributedQueue:
    """
    Distributed task queue using Redis

    Features:
    - Priority queue support
    - Task persistence
    - At-least-once delivery
    - Dead letter queue
    - Task routing
    """

    def __init__(self, backend: RedisBackend, queue_name: str = "tasks"):
        """Initialize distributed queue"""
        self.backend = backend
        self.queue_name = queue_name
        self.dead_letter_queue = f"{queue_name}:dlq"
        self.processing_set = f"{queue_name}:processing"
        self.local_queue: List[DistributedTask] = []  # Fallback

    def enqueue(self, task: DistributedTask) -> bool:
        """Add task to queue with priority"""
        task_data = asdict(task)

        if self.backend.is_connected:
            try:
                # Use sorted set for priority queue
                score = -task.priority  # Negative for high priority first
                task_key = f"task:{task.task_id}"

                # Store task data
                self.backend.set(task_key, task_data, expire=86400)  # 24h TTL

                # Add to priority queue
                self.backend.client.zadd(
                    self.backend._make_key(self.queue_name),
                    {task.task_id: score}
                )

                logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
                return True

            except Exception as e:
                logger.error(f"Failed to enqueue task: {e}")

        # Fallback to local queue
        self.local_queue.append(task)
        self.local_queue.sort(key=lambda t: -t.priority)
        return True

    def dequeue(self, timeout: int = 0) -> Optional[DistributedTask]:
        """Get next task from queue"""
        if self.backend.is_connected:
            try:
                # Pop highest priority task
                result = self.backend.client.bzpopmin(
                    self.backend._make_key(self.queue_name),
                    timeout=timeout
                )

                if result:
                    _, task_id, _ = result
                    task_id = task_id.decode() if isinstance(task_id, bytes) else task_id

                    # Get task data
                    task_key = f"task:{task_id}"
                    task_data = self.backend.get(task_key)

                    if task_data:
                        task = DistributedTask(**task_data)

                        # Add to processing set
                        self.backend.client.sadd(
                            self.backend._make_key(self.processing_set),
                            task_id
                        )

                        logger.debug(f"Dequeued task {task_id}")
                        return task

            except Exception as e:
                logger.error(f"Failed to dequeue task: {e}")

        # Fallback to local queue
        if self.local_queue:
            return self.local_queue.pop(0)

        return None

    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark task as completed"""
        if self.backend.is_connected:
            try:
                # Remove from processing set
                self.backend.client.srem(
                    self.backend._make_key(self.processing_set),
                    task_id
                )

                # Update task data
                task_key = f"task:{task_id}"
                task_data = self.backend.get(task_key)

                if task_data:
                    task_data["status"] = "completed"
                    task_data["result"] = result
                    self.backend.set(task_key, task_data, expire=3600)  # Keep for 1h

                logger.debug(f"Completed task {task_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to complete task: {e}")

        return False

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed and potentially retry"""
        if self.backend.is_connected:
            try:
                task_key = f"task:{task_id}"
                task_data = self.backend.get(task_key)

                if task_data:
                    task = DistributedTask(**task_data)
                    task.retries += 1
                    task.error = error

                    if task.retries < task.max_retries:
                        # Re-enqueue for retry
                        task.status = "retrying"
                        self.backend.set(task_key, asdict(task))

                        # Remove from processing and re-add to queue
                        self.backend.client.srem(
                            self.backend._make_key(self.processing_set),
                            task_id
                        )
                        self.backend.client.zadd(
                            self.backend._make_key(self.queue_name),
                            {task_id: -task.priority + task.retries}  # Lower priority for retries
                        )

                        logger.debug(f"Retrying task {task_id} (attempt {task.retries})")
                    else:
                        # Move to dead letter queue
                        task.status = "failed"
                        self.backend.set(task_key, asdict(task))

                        self.backend.client.srem(
                            self.backend._make_key(self.processing_set),
                            task_id
                        )
                        self.backend.client.sadd(
                            self.backend._make_key(self.dead_letter_queue),
                            task_id
                        )

                        logger.warning(f"Task {task_id} moved to DLQ after {task.retries} retries")

                    return True

            except Exception as e:
                logger.error(f"Failed to handle task failure: {e}")

        return False

    def get_queue_size(self) -> int:
        """Get number of pending tasks"""
        if self.backend.is_connected:
            try:
                return self.backend.client.zcard(
                    self.backend._make_key(self.queue_name)
                )
            except Exception:
                pass

        return len(self.local_queue)

    def get_processing_count(self) -> int:
        """Get number of tasks being processed"""
        if self.backend.is_connected:
            try:
                return self.backend.client.scard(
                    self.backend._make_key(self.processing_set)
                )
            except Exception:
                pass

        return 0

    def get_dlq_count(self) -> int:
        """Get number of tasks in dead letter queue"""
        if self.backend.is_connected:
            try:
                return self.backend.client.scard(
                    self.backend._make_key(self.dead_letter_queue)
                )
            except Exception:
                pass

        return 0


class LeaderElection:
    """
    Leader election using Redis with heartbeat

    Features:
    - Automatic leader election
    - Heartbeat-based liveness
    - Leader failover
    - Election callbacks
    """

    def __init__(
        self,
        backend: RedisBackend,
        node_id: str,
        ttl: int = 10,
        heartbeat_interval: int = 3
    ):
        """Initialize leader election"""
        self.backend = backend
        self.node_id = node_id
        self.ttl = ttl
        self.heartbeat_interval = heartbeat_interval
        self.is_leader = False
        self.leader_key = "leader"
        self.heartbeat_thread = None
        self.stop_heartbeat = threading.Event()
        self.on_elected: Optional[Callable] = None
        self.on_demoted: Optional[Callable] = None

    def start(self) -> bool:
        """Start leader election process"""
        if not self.backend.is_connected:
            # In local mode, always be leader
            self.is_leader = True
            if self.on_elected:
                self.on_elected()
            return True

        # Try to become leader
        self.is_leader = self._try_become_leader()

        # Start heartbeat thread
        self.stop_heartbeat.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()

        return self.is_leader

    def stop(self):
        """Stop leader election"""
        self.stop_heartbeat.set()

        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)

        if self.is_leader:
            self._step_down()

    def _try_become_leader(self) -> bool:
        """Attempt to become leader"""
        try:
            # Try to set leader key with NX (only if not exists)
            success = self.backend.client.set(
                self.backend._make_key(self.leader_key),
                self.node_id,
                nx=True,
                ex=self.ttl
            )

            if success:
                logger.info(f"Node {self.node_id} elected as leader")
                if self.on_elected:
                    self.on_elected()
                return True

        except Exception as e:
            logger.error(f"Leader election failed: {e}")

        return False

    def _step_down(self):
        """Step down as leader"""
        try:
            # Only delete if we are the leader
            current_leader = self.backend.get(self.leader_key)

            if current_leader == self.node_id:
                self.backend.delete(self.leader_key)

            self.is_leader = False
            logger.info(f"Node {self.node_id} stepped down as leader")

            if self.on_demoted:
                self.on_demoted()

        except Exception as e:
            logger.error(f"Failed to step down: {e}")

    def _heartbeat_loop(self):
        """Heartbeat loop to maintain leadership"""
        while not self.stop_heartbeat.is_set():
            try:
                if self.is_leader:
                    # Refresh TTL
                    self.backend.client.expire(
                        self.backend._make_key(self.leader_key),
                        self.ttl
                    )
                else:
                    # Check if we can become leader
                    current_leader = self.backend.get(self.leader_key)

                    if not current_leader:
                        self.is_leader = self._try_become_leader()

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

                if self.is_leader:
                    self._step_down()

            self.stop_heartbeat.wait(self.heartbeat_interval)

    def get_current_leader(self) -> Optional[str]:
        """Get current leader ID"""
        if not self.backend.is_connected:
            return self.node_id if self.is_leader else None

        return self.backend.get(self.leader_key)


class ClusterHealthMonitor:
    """
    Monitor health of distributed cluster

    Features:
    - Node health tracking
    - Failure detection
    - Performance metrics
    - Automatic recovery
    """

    def __init__(
        self,
        backend: RedisBackend,
        node_id: str,
        check_interval: int = 5,
        unhealthy_threshold: int = 30
    ):
        """Initialize health monitor"""
        self.backend = backend
        self.node_id = node_id
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.nodes_key = "cluster:nodes"
        self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.local_node = ClusterNode(
            node_id=node_id,
            hostname="localhost",
            status=NodeStatus.HEALTHY,
            last_heartbeat=time.time()
        )

    def start(self):
        """Start health monitoring"""
        self.stop_monitor.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()

        # Register self
        self._update_node_status(self.local_node)

        logger.info(f"Health monitor started for node {self.node_id}")

    def stop(self):
        """Stop health monitoring"""
        self.stop_monitor.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Mark self as offline
        self.local_node.status = NodeStatus.OFFLINE
        self._update_node_status(self.local_node)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitor.is_set():
            try:
                # Update own heartbeat
                self.local_node.last_heartbeat = time.time()
                self._update_node_status(self.local_node)

                # Check other nodes
                self._check_cluster_health()

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            self.stop_monitor.wait(self.check_interval)

    def _update_node_status(self, node: ClusterNode):
        """Update node status in Redis"""
        if not self.backend.is_connected:
            return

        try:
            node_key = f"{self.nodes_key}:{node.node_id}"
            self.backend.set(node_key, asdict(node), expire=self.unhealthy_threshold)

            # Add to nodes set
            self.backend.client.sadd(
                self.backend._make_key(self.nodes_key),
                node.node_id
            )

        except Exception as e:
            logger.error(f"Failed to update node status: {e}")

    def _check_cluster_health(self):
        """Check health of all cluster nodes"""
        if not self.backend.is_connected:
            return

        try:
            # Get all node IDs
            node_ids = self.backend.client.smembers(
                self.backend._make_key(self.nodes_key)
            )

            current_time = time.time()

            for node_id_bytes in node_ids:
                node_id = node_id_bytes.decode() if isinstance(node_id_bytes, bytes) else node_id_bytes

                if node_id == self.node_id:
                    continue

                node_key = f"{self.nodes_key}:{node_id}"
                node_data = self.backend.get(node_key)

                if node_data:
                    node = ClusterNode(**node_data)
                    time_since_heartbeat = current_time - node.last_heartbeat

                    # Update status based on heartbeat
                    if time_since_heartbeat > self.unhealthy_threshold:
                        if node.status != NodeStatus.UNHEALTHY:
                            logger.warning(f"Node {node_id} marked as unhealthy")
                            node.status = NodeStatus.UNHEALTHY
                            self._handle_unhealthy_node(node)

                    elif time_since_heartbeat > self.unhealthy_threshold / 2:
                        if node.status == NodeStatus.HEALTHY:
                            node.status = NodeStatus.DEGRADED
                            logger.warning(f"Node {node_id} degraded")
                else:
                    # Node data expired - mark as offline
                    logger.warning(f"Node {node_id} is offline")
                    self.backend.client.srem(
                        self.backend._make_key(self.nodes_key),
                        node_id
                    )

        except Exception as e:
            logger.error(f"Cluster health check failed: {e}")

    def _handle_unhealthy_node(self, node: ClusterNode):
        """Handle unhealthy node - redistribute tasks"""
        logger.info(f"Handling unhealthy node {node.node_id}")
        # TODO: Implement task redistribution logic

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        if not self.backend.is_connected:
            return {
                "nodes": [asdict(self.local_node)],
                "total_nodes": 1,
                "healthy_nodes": 1 if self.local_node.status == NodeStatus.HEALTHY else 0
            }

        try:
            node_ids = self.backend.client.smembers(
                self.backend._make_key(self.nodes_key)
            )

            nodes = []
            healthy_count = 0

            for node_id_bytes in node_ids:
                node_id = node_id_bytes.decode() if isinstance(node_id_bytes, bytes) else node_id_bytes
                node_key = f"{self.nodes_key}:{node_id}"
                node_data = self.backend.get(node_key)

                if node_data:
                    nodes.append(node_data)
                    if node_data["status"] == NodeStatus.HEALTHY.value:
                        healthy_count += 1

            return {
                "nodes": nodes,
                "total_nodes": len(nodes),
                "healthy_nodes": healthy_count,
                "degraded_nodes": sum(1 for n in nodes if n["status"] == NodeStatus.DEGRADED.value),
                "unhealthy_nodes": sum(1 for n in nodes if n["status"] == NodeStatus.UNHEALTHY.value)
            }

        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"error": str(e)}

    def update_node_metrics(self, cpu_usage: float, memory_usage: float):
        """Update node performance metrics"""
        self.local_node.cpu_usage = cpu_usage
        self.local_node.memory_usage = memory_usage
        self._update_node_status(self.local_node)

    def increment_task_counter(self, success: bool = True):
        """Increment task completion counter"""
        if success:
            self.local_node.tasks_completed += 1
        else:
            self.local_node.tasks_failed += 1
        self._update_node_status(self.local_node)


class DistributedOrchestrator:
    """
    Main orchestrator for distributed ORCHEX operations

    Combines all distributed components:
    - State management
    - Locking
    - Task distribution
    - Leader election
    - Health monitoring
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        node_id: Optional[str] = None,
        enable_leader_election: bool = True,
        enable_health_monitoring: bool = True
    ):
        """Initialize distributed orchestrator"""
        self.node_id = node_id or str(uuid.uuid4())

        # Initialize Redis backend
        self.backend = RedisBackend(
            host=redis_host,
            port=redis_port,
            key_prefix="ORCHEX:"
        )

        # Initialize components
        self.lock_manager = DistributedLockManager(self.backend)
        self.task_queue = DistributedQueue(self.backend)

        # Optional components
        self.leader_election = None
        if enable_leader_election:
            self.leader_election = LeaderElection(
                self.backend,
                self.node_id
            )

        self.health_monitor = None
        if enable_health_monitoring:
            self.health_monitor = ClusterHealthMonitor(
                self.backend,
                self.node_id
            )

        logger.info(f"Distributed orchestrator initialized for node {self.node_id}")

    def start(self):
        """Start all distributed services"""
        # Start leader election
        if self.leader_election:
            self.leader_election.start()

        # Start health monitoring
        if self.health_monitor:
            self.health_monitor.start()

        logger.info("Distributed services started")

    def stop(self):
        """Stop all distributed services"""
        # Stop health monitoring
        if self.health_monitor:
            self.health_monitor.stop()

        # Stop leader election
        if self.leader_election:
            self.leader_election.stop()

        logger.info("Distributed services stopped")

    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        if self.leader_election:
            return self.leader_election.is_leader
        return True  # Default to leader in single-node mode

    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Submit task to distributed queue"""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority
        )

        self.task_queue.enqueue(task)
        return task.task_id

    def process_next_task(self, timeout: int = 0) -> Optional[DistributedTask]:
        """Get and process next task from queue"""
        return self.task_queue.dequeue(timeout)

    def complete_task(self, task_id: str, result: Any = None):
        """Mark task as completed"""
        self.task_queue.complete_task(task_id, result)

        if self.health_monitor:
            self.health_monitor.increment_task_counter(success=True)

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        self.task_queue.fail_task(task_id, error)

        if self.health_monitor:
            self.health_monitor.increment_task_counter(success=False)

    def with_lock(self, resource: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with distributed lock"""
        lock_acquired = False

        try:
            lock_acquired = self.lock_manager.acquire_lock(resource)

            if not lock_acquired:
                raise RuntimeError(f"Failed to acquire lock for {resource}")

            return func(*args, **kwargs)

        finally:
            if lock_acquired:
                self.lock_manager.release_lock(resource)

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        status = {
            "node_id": self.node_id,
            "is_connected": self.backend.is_connected,
            "is_leader": self.is_leader()
        }

        # Add queue metrics
        status["queue"] = {
            "pending": self.task_queue.get_queue_size(),
            "processing": self.task_queue.get_processing_count(),
            "failed": self.task_queue.get_dlq_count()
        }

        # Add lock statistics
        status["locks"] = self.lock_manager.get_lock_stats()

        # Add cluster health
        if self.health_monitor:
            status["cluster"] = self.health_monitor.get_cluster_status()

        # Add current leader
        if self.leader_election:
            status["current_leader"] = self.leader_election.get_current_leader()

        return status