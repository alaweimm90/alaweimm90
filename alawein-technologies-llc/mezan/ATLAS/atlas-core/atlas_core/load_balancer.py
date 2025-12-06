"""
Enterprise Load Balancer with Advanced Algorithms

Production-grade load balancing with:
- Multiple algorithms (round-robin, least-conn, IP hash, random, weighted)
- Health-aware routing
- Sticky sessions support
- Backend server pool management
- Automatic failover
- Metrics collection
- Connection pooling
- Circuit breaker integration

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_RESPONSE_TIME = "least_response_time"
    POWER_OF_TWO_CHOICES = "power_of_two_choices"
    RESOURCE_BASED = "resource_based"


class ServerState(Enum):
    """Server states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"  # Gracefully removing from pool
    WARMING = "warming"    # Gradually adding to pool
    DISABLED = "disabled"  # Administratively disabled


@dataclass
class Server:
    """Backend server"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    health_check_url: Optional[str] = None
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    state: ServerState = ServerState.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_health_check: Optional[float] = None
    last_failure: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def url(self) -> str:
        """Get server URL"""
        return f"http://{self.host}:{self.port}"

    @property
    def available_connections(self) -> int:
        """Get available connection capacity"""
        return max(0, self.max_connections - self.current_connections)

    @property
    def avg_response_time(self) -> float:
        """Get average response time"""
        if self.response_times:
            return sum(self.response_times) / len(self.response_times)
        return 0.0

    @property
    def p95_response_time(self) -> float:
        """Get 95th percentile response time"""
        if self.response_times:
            sorted_times = sorted(self.response_times)
            index = int(len(sorted_times) * 0.95)
            return sorted_times[index]
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate"""
        if self.total_requests > 0:
            return 1.0 - (self.failed_requests / self.total_requests)
        return 1.0

    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)"""
        # Consider multiple factors
        connection_ratio = self.current_connections / max(self.max_connections, 1)
        response_time_factor = self.avg_response_time / 1000.0  # Normalize to seconds
        failure_rate = self.failed_requests / max(self.total_requests, 1)

        # Weighted score
        score = (
            connection_ratio * 0.4 +
            response_time_factor * 0.3 +
            failure_rate * 0.2 +
            (self.cpu_usage / 100.0) * 0.05 +
            (self.memory_usage / 100.0) * 0.05
        )

        return score


@dataclass
class ServerPool:
    """Server pool configuration"""
    name: str
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    servers: List[Server] = field(default_factory=list)
    sticky_sessions: bool = False
    session_ttl: float = 3600.0
    health_check_enabled: bool = True
    connection_pool_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    failover_pools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    current_index: int = 0
    sessions: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    consistent_hash_ring: Optional[Dict[int, str]] = None


class ConsistentHashRing:
    """Consistent hash ring for server selection"""

    def __init__(self, servers: List[Server], virtual_nodes: int = 150):
        """Initialize consistent hash ring"""
        self.servers = servers
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, Server] = {}
        self._build_ring()

    def _build_ring(self):
        """Build the hash ring"""
        self.ring.clear()

        for server in self.servers:
            if server.state != ServerState.HEALTHY:
                continue

            # Add virtual nodes for each server
            for i in range(self.virtual_nodes * server.weight):
                virtual_key = f"{server.id}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = server

    def _hash(self, key: str) -> int:
        """Calculate hash value"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_server(self, key: str) -> Optional[Server]:
        """Get server for a key"""
        if not self.ring:
            return None

        hash_value = self._hash(key)

        # Find the first server with hash >= hash_value
        sorted_hashes = sorted(self.ring.keys())
        for ring_hash in sorted_hashes:
            if ring_hash >= hash_value:
                return self.ring[ring_hash]

        # Wrap around to first server
        return self.ring[sorted_hashes[0]]

    def add_server(self, server: Server):
        """Add server to ring"""
        self.servers.append(server)
        self._build_ring()

    def remove_server(self, server_id: str):
        """Remove server from ring"""
        self.servers = [s for s in self.servers if s.id != server_id]
        self._build_ring()


class LoadBalancer:
    """Enterprise load balancer"""

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_health_checks: bool = True,
        redis_client: Optional[Any] = None
    ):
        """Initialize load balancer"""
        self.pools: Dict[str, ServerPool] = {}
        self.enable_metrics = enable_metrics
        self.enable_health_checks = enable_health_checks
        self.redis_client = redis_client

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'requests_by_pool': defaultdict(int),
            'requests_by_server': defaultdict(int),
            'response_times': deque(maxlen=1000),
            'errors_by_type': defaultdict(int),
        }

        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None

        # Connection pools (server_id -> connection pool)
        self.connection_pools: Dict[str, Any] = {}

    def add_pool(self, pool: ServerPool) -> None:
        """Add server pool"""
        self.pools[pool.name] = pool

        # Initialize consistent hash ring if needed
        if pool.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            pool.consistent_hash_ring = ConsistentHashRing(pool.servers)

        logger.info(f"Added pool '{pool.name}' with {len(pool.servers)} servers")

    def add_server(self, pool_name: str, server: Server) -> bool:
        """Add server to pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            return False

        pool.servers.append(server)

        # Update consistent hash ring if needed
        if pool.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH and pool.consistent_hash_ring:
            pool.consistent_hash_ring.add_server(server)

        logger.info(f"Added server {server.id} to pool '{pool_name}'")
        return True

    def remove_server(self, pool_name: str, server_id: str, graceful: bool = True) -> bool:
        """Remove server from pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            return False

        server = next((s for s in pool.servers if s.id == server_id), None)
        if not server:
            return False

        if graceful:
            # Mark server as draining
            server.state = ServerState.DRAINING
            logger.info(f"Server {server_id} marked for graceful removal")

            # Schedule actual removal after connections drain
            asyncio.create_task(self._drain_and_remove_server(pool, server))
        else:
            # Remove immediately
            pool.servers = [s for s in pool.servers if s.id != server_id]

            # Update consistent hash ring if needed
            if pool.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH and pool.consistent_hash_ring:
                pool.consistent_hash_ring.remove_server(server_id)

            logger.info(f"Server {server_id} removed from pool '{pool_name}'")

        return True

    async def _drain_and_remove_server(self, pool: ServerPool, server: Server):
        """Drain connections and remove server"""
        max_wait = 60  # Maximum wait time in seconds
        start_time = time.time()

        while server.current_connections > 0 and time.time() - start_time < max_wait:
            await asyncio.sleep(1)

        # Remove server
        pool.servers = [s for s in pool.servers if s.id != server.id]

        # Update consistent hash ring if needed
        if pool.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH and pool.consistent_hash_ring:
            pool.consistent_hash_ring.remove_server(server.id)

        logger.info(f"Server {server.id} drained and removed")

    async def select_server(
        self,
        pool_name: str,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_key: Optional[str] = None
    ) -> Optional[Server]:
        """Select server from pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            logger.error(f"Pool '{pool_name}' not found")
            return None

        # Get healthy servers
        healthy_servers = [
            s for s in pool.servers
            if s.state == ServerState.HEALTHY and s.available_connections > 0
        ]

        if not healthy_servers:
            # Try failover pools
            for failover_pool_name in pool.failover_pools:
                failover_server = await self.select_server(
                    failover_pool_name,
                    session_id,
                    client_ip,
                    request_key
                )
                if failover_server:
                    logger.info(f"Using failover pool '{failover_pool_name}'")
                    return failover_server

            logger.warning(f"No healthy servers in pool '{pool_name}'")
            return None

        # Handle sticky sessions
        if pool.sticky_sessions and session_id:
            server_id, expiry = pool.sessions.get(session_id, (None, 0))
            if server_id and time.time() < expiry:
                server = next((s for s in healthy_servers if s.id == server_id), None)
                if server:
                    return server

        # Select based on algorithm
        server = await self._select_by_algorithm(
            pool,
            healthy_servers,
            client_ip,
            request_key
        )

        # Update sticky session if needed
        if server and pool.sticky_sessions and session_id:
            pool.sessions[session_id] = (server.id, time.time() + pool.session_ttl)

        return server

    async def _select_by_algorithm(
        self,
        pool: ServerPool,
        servers: List[Server],
        client_ip: Optional[str] = None,
        request_key: Optional[str] = None
    ) -> Optional[Server]:
        """Select server based on algorithm"""
        if not servers:
            return None

        algorithm = pool.algorithm

        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(pool, servers)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(pool, servers)

        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return min(servers, key=lambda s: s.current_connections)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            return min(servers, key=lambda s: s.current_connections / s.weight)

        elif algorithm == LoadBalancingAlgorithm.IP_HASH:
            if client_ip:
                hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
                return servers[hash_value % len(servers)]
            else:
                return servers[0]

        elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            if pool.consistent_hash_ring and request_key:
                return pool.consistent_hash_ring.get_server(request_key)
            else:
                return servers[0]

        elif algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(servers)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_RANDOM:
            return self._weighted_random_select(servers)

        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return min(servers, key=lambda s: s.avg_response_time)

        elif algorithm == LoadBalancingAlgorithm.POWER_OF_TWO_CHOICES:
            # Randomly select two servers and pick the one with fewer connections
            if len(servers) >= 2:
                choices = random.sample(servers, 2)
                return min(choices, key=lambda s: s.current_connections)
            else:
                return servers[0]

        elif algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            # Select based on comprehensive load score
            return min(servers, key=lambda s: s.load_score)

        else:
            logger.warning(f"Unknown algorithm: {algorithm}")
            return servers[0]

    def _round_robin_select(self, pool: ServerPool, servers: List[Server]) -> Server:
        """Round-robin selection"""
        selected = servers[pool.current_index % len(servers)]
        pool.current_index += 1
        return selected

    def _weighted_round_robin_select(self, pool: ServerPool, servers: List[Server]) -> Server:
        """Weighted round-robin selection"""
        # Build weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)

        if weighted_servers:
            selected = weighted_servers[pool.current_index % len(weighted_servers)]
            pool.current_index += 1
            return selected

        return servers[0]

    def _weighted_random_select(self, servers: List[Server]) -> Server:
        """Weighted random selection"""
        total_weight = sum(s.weight for s in servers)
        rand_value = random.random() * total_weight

        cumulative = 0
        for server in servers:
            cumulative += server.weight
            if rand_value <= cumulative:
                return server

        return servers[-1]

    async def execute_request(
        self,
        pool_name: str,
        request_func: Callable,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_key: Optional[str] = None,
        retry_on_failure: bool = True
    ) -> Any:
        """Execute request with load balancing and failover"""
        pool = self.pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")

        max_retries = pool.max_retries if retry_on_failure else 1
        last_error = None

        for attempt in range(max_retries):
            # Select server
            server = await self.select_server(
                pool_name,
                session_id,
                client_ip,
                request_key
            )

            if not server:
                raise RuntimeError(f"No available servers in pool '{pool_name}'")

            try:
                # Update connection count
                server.current_connections += 1
                server.total_requests += 1

                # Record start time
                start_time = time.time()

                # Execute request
                result = await request_func(server)

                # Record response time
                response_time = time.time() - start_time
                server.response_times.append(response_time)
                server.total_response_time += response_time

                # Reset failure counters
                server.consecutive_failures = 0

                # Update metrics
                self._record_success(pool_name, server.id, response_time)

                return result

            except Exception as e:
                # Record failure
                server.failed_requests += 1
                server.consecutive_failures += 1
                server.last_failure = time.time()
                last_error = e

                # Update metrics
                self._record_failure(pool_name, server.id, str(e))

                # Check if server should be marked unhealthy
                if server.consecutive_failures >= server.unhealthy_threshold:
                    server.state = ServerState.UNHEALTHY
                    logger.warning(f"Server {server.id} marked unhealthy")

                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(pool.retry_delay * (2 ** attempt))

            finally:
                # Update connection count
                server.current_connections = max(0, server.current_connections - 1)

        # All retries failed
        raise last_error or RuntimeError("All retry attempts failed")

    async def start(self):
        """Start load balancer"""
        if self.enable_health_checks:
            self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Load balancer started")

    async def stop(self):
        """Stop load balancer"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Load balancer stopped")

    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                for pool in self.pools.values():
                    if pool.health_check_enabled:
                        await self._check_pool_health(pool)

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _check_pool_health(self, pool: ServerPool):
        """Check health of all servers in pool"""
        tasks = []
        for server in pool.servers:
            if server.state != ServerState.DISABLED:
                tasks.append(self._check_server_health(server))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_server_health(self, server: Server):
        """Check health of a single server"""
        if not server.health_check_url:
            return

        # Check if it's time for health check
        if (server.last_health_check and
            time.time() - server.last_health_check < server.health_check_interval):
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    server.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=server.health_check_timeout)
                ) as response:
                    success = response.status == 200

            server.last_health_check = time.time()

            if success:
                server.consecutive_successes += 1
                server.consecutive_failures = 0

                # Mark healthy if threshold met
                if (server.state == ServerState.UNHEALTHY and
                    server.consecutive_successes >= server.healthy_threshold):
                    server.state = ServerState.HEALTHY
                    logger.info(f"Server {server.id} marked healthy")

            else:
                server.consecutive_failures += 1
                server.consecutive_successes = 0

                # Mark unhealthy if threshold met
                if (server.state == ServerState.HEALTHY and
                    server.consecutive_failures >= server.unhealthy_threshold):
                    server.state = ServerState.UNHEALTHY
                    logger.warning(f"Server {server.id} marked unhealthy")

        except Exception as e:
            server.consecutive_failures += 1
            server.consecutive_successes = 0
            server.last_health_check = time.time()

            if (server.state == ServerState.HEALTHY and
                server.consecutive_failures >= server.unhealthy_threshold):
                server.state = ServerState.UNHEALTHY
                logger.warning(f"Server {server.id} marked unhealthy: {e}")

    def _record_success(self, pool_name: str, server_id: str, response_time: float):
        """Record successful request"""
        if not self.enable_metrics:
            return

        self.metrics['total_requests'] += 1
        self.metrics['successful_requests'] += 1
        self.metrics['requests_by_pool'][pool_name] += 1
        self.metrics['requests_by_server'][server_id] += 1
        self.metrics['response_times'].append(response_time)

    def _record_failure(self, pool_name: str, server_id: str, error: str):
        """Record failed request"""
        if not self.enable_metrics:
            return

        self.metrics['total_requests'] += 1
        self.metrics['failed_requests'] += 1
        self.metrics['requests_by_pool'][pool_name] += 1
        self.metrics['requests_by_server'][server_id] += 1

        # Categorize error
        if 'timeout' in error.lower():
            self.metrics['errors_by_type']['timeout'] += 1
        elif 'connection' in error.lower():
            self.metrics['errors_by_type']['connection'] += 1
        else:
            self.metrics['errors_by_type']['other'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics"""
        response_times = list(self.metrics['response_times'])

        metrics = {
            'total_requests': self.metrics['total_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (
                self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
            ),
            'requests_by_pool': dict(self.metrics['requests_by_pool']),
            'requests_by_server': dict(self.metrics['requests_by_server']),
            'errors_by_type': dict(self.metrics['errors_by_type']),
        }

        if response_times:
            sorted_times = sorted(response_times)
            metrics.update({
                'response_time_avg': sum(response_times) / len(response_times),
                'response_time_p50': sorted_times[len(sorted_times) // 2],
                'response_time_p95': sorted_times[int(len(sorted_times) * 0.95)],
                'response_time_p99': sorted_times[int(len(sorted_times) * 0.99)],
            })

        # Pool metrics
        metrics['pools'] = {}
        for pool_name, pool in self.pools.items():
            healthy_servers = sum(1 for s in pool.servers if s.state == ServerState.HEALTHY)
            total_connections = sum(s.current_connections for s in pool.servers)

            metrics['pools'][pool_name] = {
                'algorithm': pool.algorithm.value,
                'total_servers': len(pool.servers),
                'healthy_servers': healthy_servers,
                'total_connections': total_connections,
                'sticky_sessions_count': len(pool.sessions),
                'servers': [
                    {
                        'id': s.id,
                        'state': s.state.value,
                        'current_connections': s.current_connections,
                        'total_requests': s.total_requests,
                        'failed_requests': s.failed_requests,
                        'success_rate': s.success_rate,
                        'avg_response_time': s.avg_response_time,
                        'p95_response_time': s.p95_response_time,
                        'load_score': s.load_score,
                    }
                    for s in pool.servers
                ]
            }

        return metrics

    def get_pool_status(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            return None

        return {
            'name': pool.name,
            'algorithm': pool.algorithm.value,
            'sticky_sessions': pool.sticky_sessions,
            'health_check_enabled': pool.health_check_enabled,
            'servers': [
                {
                    'id': s.id,
                    'host': s.host,
                    'port': s.port,
                    'weight': s.weight,
                    'state': s.state.value,
                    'current_connections': s.current_connections,
                    'max_connections': s.max_connections,
                    'available_connections': s.available_connections,
                    'total_requests': s.total_requests,
                    'failed_requests': s.failed_requests,
                    'success_rate': s.success_rate,
                    'avg_response_time': s.avg_response_time,
                    'p95_response_time': s.p95_response_time,
                    'consecutive_failures': s.consecutive_failures,
                    'consecutive_successes': s.consecutive_successes,
                    'last_health_check': s.last_health_check,
                    'cpu_usage': s.cpu_usage,
                    'memory_usage': s.memory_usage,
                    'load_score': s.load_score,
                }
                for s in pool.servers
            ],
            'failover_pools': pool.failover_pools,
        }

    def update_server_resources(
        self,
        pool_name: str,
        server_id: str,
        cpu_usage: float,
        memory_usage: float
    ) -> bool:
        """Update server resource usage (for resource-based load balancing)"""
        pool = self.pools.get(pool_name)
        if not pool:
            return False

        server = next((s for s in pool.servers if s.id == server_id), None)
        if not server:
            return False

        server.cpu_usage = cpu_usage
        server.memory_usage = memory_usage
        return True