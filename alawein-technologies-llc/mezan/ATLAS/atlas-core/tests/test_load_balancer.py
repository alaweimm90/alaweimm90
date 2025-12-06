"""
Comprehensive tests for Load Balancer

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import pytest
import time
import random
from unittest.mock import Mock, AsyncMock, patch
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atlas_core.load_balancer import (
    LoadBalancer,
    Server,
    ServerPool,
    ServerState,
    LoadBalancingAlgorithm,
    ConsistentHashRing
)


@pytest.fixture
def servers():
    """Create test servers"""
    return [
        Server(
            id="server-1",
            host="localhost",
            port=8001,
            weight=2,
            max_connections=100
        ),
        Server(
            id="server-2",
            host="localhost",
            port=8002,
            weight=1,
            max_connections=100
        ),
        Server(
            id="server-3",
            host="localhost",
            port=8003,
            weight=1,
            max_connections=100
        )
    ]


@pytest.fixture
def server_pool(servers):
    """Create test server pool"""
    return ServerPool(
        name="test-pool",
        algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
        servers=servers,
        health_check_enabled=True
    )


@pytest.fixture
async def load_balancer():
    """Create load balancer instance"""
    lb = LoadBalancer(enable_metrics=True, enable_health_checks=False)
    await lb.start()
    yield lb
    await lb.stop()


class TestServer:
    """Test Server model"""

    def test_server_creation(self):
        """Test server creation"""
        server = Server(
            id="server-1",
            host="localhost",
            port=8080,
            weight=2,
            max_connections=50
        )

        assert server.id == "server-1"
        assert server.url == "http://localhost:8080"
        assert server.weight == 2
        assert server.max_connections == 50
        assert server.state == ServerState.HEALTHY

    def test_server_properties(self):
        """Test server property calculations"""
        server = Server(id="server-1", host="localhost", port=8080)

        # Set runtime stats
        server.current_connections = 25
        server.max_connections = 100
        server.total_requests = 1000
        server.failed_requests = 50

        assert server.available_connections == 75
        assert server.success_rate == 0.95

        # Add response times
        server.response_times.extend([0.1, 0.2, 0.3, 0.4, 0.5])
        assert server.avg_response_time == pytest.approx(0.3, 0.01)

    def test_server_load_score(self):
        """Test server load score calculation"""
        server = Server(id="server-1", host="localhost", port=8080)

        # Low load
        server.current_connections = 10
        server.max_connections = 100
        server.cpu_usage = 20.0
        server.memory_usage = 30.0
        low_score = server.load_score

        # High load
        server.current_connections = 90
        server.cpu_usage = 80.0
        server.memory_usage = 85.0
        high_score = server.load_score

        assert low_score < high_score  # Lower score is better


class TestServerPool:
    """Test ServerPool model"""

    def test_pool_creation(self, servers):
        """Test server pool creation"""
        pool = ServerPool(
            name="test-pool",
            algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
            servers=servers,
            sticky_sessions=True,
            session_ttl=1800.0
        )

        assert pool.name == "test-pool"
        assert pool.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS
        assert len(pool.servers) == 3
        assert pool.sticky_sessions is True


class TestLoadBalancer:
    """Test LoadBalancer functionality"""

    @pytest.mark.asyncio
    async def test_add_pool(self, load_balancer, server_pool):
        """Test adding server pool"""
        load_balancer.add_pool(server_pool)

        assert "test-pool" in load_balancer.pools
        assert len(load_balancer.pools["test-pool"].servers) == 3

    @pytest.mark.asyncio
    async def test_add_server(self, load_balancer, server_pool):
        """Test adding server to pool"""
        load_balancer.add_pool(server_pool)

        new_server = Server(
            id="server-4",
            host="localhost",
            port=8004
        )

        success = load_balancer.add_server("test-pool", new_server)
        assert success is True
        assert len(load_balancer.pools["test-pool"].servers) == 4

    @pytest.mark.asyncio
    async def test_remove_server(self, load_balancer, server_pool):
        """Test removing server from pool"""
        load_balancer.add_pool(server_pool)

        # Remove server (graceful)
        success = load_balancer.remove_server("test-pool", "server-2", graceful=True)
        assert success is True

        # Check server is marked for draining
        server = next(s for s in server_pool.servers if s.id == "server-2")
        assert server.state == ServerState.DRAINING

        # Remove server (immediate)
        success = load_balancer.remove_server("test-pool", "server-3", graceful=False)
        assert success is True
        assert len([s for s in server_pool.servers if s.id == "server-3"]) == 0

    @pytest.mark.asyncio
    async def test_round_robin_selection(self, load_balancer, servers):
        """Test round-robin load balancing"""
        pool = ServerPool(
            name="rr-pool",
            algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Select servers in sequence
        selections = []
        for _ in range(6):
            server = await load_balancer.select_server("rr-pool")
            selections.append(server.id)

        # Should cycle through servers
        assert selections == [
            "server-1", "server-2", "server-3",
            "server-1", "server-2", "server-3"
        ]

    @pytest.mark.asyncio
    async def test_least_connections_selection(self, load_balancer, servers):
        """Test least connections load balancing"""
        # Set different connection counts
        servers[0].current_connections = 50
        servers[1].current_connections = 20
        servers[2].current_connections = 30

        pool = ServerPool(
            name="lc-pool",
            algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Should select server with least connections
        server = await load_balancer.select_server("lc-pool")
        assert server.id == "server-2"  # Has 20 connections

    @pytest.mark.asyncio
    async def test_weighted_selection(self, load_balancer, servers):
        """Test weighted load balancing"""
        pool = ServerPool(
            name="weighted-pool",
            algorithm=LoadBalancingAlgorithm.WEIGHTED_RANDOM,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Statistical test - server-1 has weight 2, others have weight 1
        selections = {"server-1": 0, "server-2": 0, "server-3": 0}
        for _ in range(100):
            server = await load_balancer.select_server("weighted-pool")
            selections[server.id] += 1

        # server-1 should get approximately 50% of requests
        assert 40 <= selections["server-1"] <= 60

    @pytest.mark.asyncio
    async def test_ip_hash_selection(self, load_balancer, servers):
        """Test IP hash load balancing"""
        pool = ServerPool(
            name="ip-hash-pool",
            algorithm=LoadBalancingAlgorithm.IP_HASH,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Same IP should always get same server
        ip1 = "192.168.1.100"
        server1 = await load_balancer.select_server("ip-hash-pool", client_ip=ip1)

        for _ in range(5):
            server = await load_balancer.select_server("ip-hash-pool", client_ip=ip1)
            assert server.id == server1.id

        # Different IP may get different server
        ip2 = "192.168.1.200"
        server2 = await load_balancer.select_server("ip-hash-pool", client_ip=ip2)

        for _ in range(5):
            server = await load_balancer.select_server("ip-hash-pool", client_ip=ip2)
            assert server.id == server2.id

    @pytest.mark.asyncio
    async def test_sticky_sessions(self, load_balancer, servers):
        """Test sticky session support"""
        pool = ServerPool(
            name="sticky-pool",
            algorithm=LoadBalancingAlgorithm.RANDOM,
            servers=servers,
            sticky_sessions=True,
            session_ttl=10.0
        )
        load_balancer.add_pool(pool)

        # First request creates session
        session_id = "session-123"
        server1 = await load_balancer.select_server("sticky-pool", session_id=session_id)

        # Subsequent requests should get same server
        for _ in range(5):
            server = await load_balancer.select_server("sticky-pool", session_id=session_id)
            assert server.id == server1.id

    @pytest.mark.asyncio
    async def test_health_aware_selection(self, load_balancer):
        """Test health-aware server selection"""
        servers = [
            Server(id="healthy-1", host="localhost", port=8001, state=ServerState.HEALTHY),
            Server(id="unhealthy-1", host="localhost", port=8002, state=ServerState.UNHEALTHY),
            Server(id="healthy-2", host="localhost", port=8003, state=ServerState.HEALTHY),
        ]

        pool = ServerPool(name="health-pool", servers=servers)
        load_balancer.add_pool(pool)

        # Should only select healthy servers
        for _ in range(10):
            server = await load_balancer.select_server("health-pool")
            assert server.state == ServerState.HEALTHY
            assert server.id in ["healthy-1", "healthy-2"]

    @pytest.mark.asyncio
    async def test_failover_pools(self, load_balancer):
        """Test failover pool functionality"""
        # Primary pool with unhealthy servers
        primary_servers = [
            Server(id="primary-1", host="localhost", port=8001, state=ServerState.UNHEALTHY)
        ]
        primary_pool = ServerPool(
            name="primary",
            servers=primary_servers,
            failover_pools=["backup"]
        )

        # Backup pool with healthy servers
        backup_servers = [
            Server(id="backup-1", host="localhost", port=9001, state=ServerState.HEALTHY)
        ]
        backup_pool = ServerPool(name="backup", servers=backup_servers)

        load_balancer.add_pool(primary_pool)
        load_balancer.add_pool(backup_pool)

        # Should failover to backup pool
        server = await load_balancer.select_server("primary")
        assert server is not None
        assert server.id == "backup-1"

    @pytest.mark.asyncio
    async def test_execute_request(self, load_balancer, server_pool):
        """Test request execution with load balancing"""
        load_balancer.add_pool(server_pool)

        # Mock request function
        async def mock_request(server):
            return f"Response from {server.id}"

        # Execute request
        result = await load_balancer.execute_request(
            "test-pool",
            mock_request,
            retry_on_failure=False
        )

        assert "Response from server-" in result

    @pytest.mark.asyncio
    async def test_execute_request_with_retry(self, load_balancer, server_pool):
        """Test request execution with retries"""
        load_balancer.add_pool(server_pool)

        # Mock request that fails then succeeds
        attempt_count = 0

        async def flaky_request(server):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception("Connection error")
            return "Success"

        # Execute request with retries
        result = await load_balancer.execute_request(
            "test-pool",
            flaky_request,
            retry_on_failure=True
        )

        assert result == "Success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_metrics_collection(self, load_balancer, server_pool):
        """Test metrics collection"""
        load_balancer.add_pool(server_pool)

        # Execute some requests
        async def mock_request(server):
            await asyncio.sleep(0.01)
            return "OK"

        for _ in range(5):
            await load_balancer.execute_request("test-pool", mock_request)

        metrics = load_balancer.get_metrics()

        assert metrics['total_requests'] == 5
        assert metrics['successful_requests'] == 5
        assert metrics['failed_requests'] == 0
        assert metrics['success_rate'] == 1.0
        assert 'response_time_avg' in metrics
        assert 'pools' in metrics
        assert 'test-pool' in metrics['pools']

    @pytest.mark.asyncio
    async def test_power_of_two_choices(self, load_balancer):
        """Test power of two choices algorithm"""
        servers = [
            Server(id=f"server-{i}", host="localhost", port=8000+i)
            for i in range(10)
        ]

        # Set varying connection counts
        for i, server in enumerate(servers):
            server.current_connections = i * 10

        pool = ServerPool(
            name="p2c-pool",
            algorithm=LoadBalancingAlgorithm.POWER_OF_TWO_CHOICES,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Should tend to select servers with fewer connections
        selections = []
        for _ in range(20):
            server = await load_balancer.select_server("p2c-pool")
            selections.append(int(server.id.split("-")[1]))

        # Average selection should be lower than random (< 5)
        avg_selection = sum(selections) / len(selections)
        assert avg_selection < 5

    @pytest.mark.asyncio
    async def test_resource_based_selection(self, load_balancer):
        """Test resource-based load balancing"""
        servers = [
            Server(id="low-load", host="localhost", port=8001),
            Server(id="high-load", host="localhost", port=8002)
        ]

        # Set resource usage
        servers[0].cpu_usage = 20.0
        servers[0].memory_usage = 30.0
        servers[0].current_connections = 10

        servers[1].cpu_usage = 80.0
        servers[1].memory_usage = 85.0
        servers[1].current_connections = 90

        pool = ServerPool(
            name="resource-pool",
            algorithm=LoadBalancingAlgorithm.RESOURCE_BASED,
            servers=servers
        )
        load_balancer.add_pool(pool)

        # Should select server with lower load score
        for _ in range(5):
            server = await load_balancer.select_server("resource-pool")
            assert server.id == "low-load"


class TestConsistentHashRing:
    """Test Consistent Hash Ring"""

    def test_consistent_hash_creation(self):
        """Test consistent hash ring creation"""
        servers = [
            Server(id=f"server-{i}", host="localhost", port=8000+i, weight=1)
            for i in range(3)
        ]

        ring = ConsistentHashRing(servers, virtual_nodes=150)

        assert len(ring.servers) == 3
        assert len(ring.ring) == 3 * 150  # virtual nodes per server

    def test_consistent_hash_distribution(self):
        """Test key distribution in consistent hash"""
        servers = [
            Server(id=f"server-{i}", host="localhost", port=8000+i, weight=1)
            for i in range(3)
        ]

        ring = ConsistentHashRing(servers, virtual_nodes=150)

        # Distribute keys
        distribution = {f"server-{i}": 0 for i in range(3)}

        for i in range(1000):
            key = f"key-{i}"
            server = ring.get_server(key)
            distribution[server.id] += 1

        # Should be roughly evenly distributed
        for count in distribution.values():
            assert 250 <= count <= 400  # Allow some variance

    def test_consistent_hash_consistency(self):
        """Test that same key always maps to same server"""
        servers = [
            Server(id=f"server-{i}", host="localhost", port=8000+i)
            for i in range(5)
        ]

        ring = ConsistentHashRing(servers)

        # Same key should always get same server
        key = "test-key"
        first_server = ring.get_server(key)

        for _ in range(10):
            server = ring.get_server(key)
            assert server.id == first_server.id

    def test_consistent_hash_add_remove(self):
        """Test adding and removing servers from ring"""
        servers = [
            Server(id=f"server-{i}", host="localhost", port=8000+i)
            for i in range(3)
        ]

        ring = ConsistentHashRing(servers)

        # Map some keys
        key_mappings = {}
        for i in range(100):
            key = f"key-{i}"
            key_mappings[key] = ring.get_server(key).id

        # Add new server
        new_server = Server(id="server-3", host="localhost", port=8003)
        ring.add_server(new_server)

        # Most keys should still map to same server
        same_mapping = 0
        for key, old_server in key_mappings.items():
            new_server = ring.get_server(key)
            if new_server.id == old_server:
                same_mapping += 1

        # At least 60% should remain on same server
        assert same_mapping >= 60

        # Remove server
        ring.remove_server("server-1")

        # Keys should redistribute among remaining servers
        for i in range(100):
            key = f"key-{i}"
            server = ring.get_server(key)
            assert server.id != "server-1"


class TestServerStates:
    """Test server state management"""

    @pytest.mark.asyncio
    async def test_draining_state(self, load_balancer):
        """Test server draining state"""
        servers = [
            Server(id="server-1", host="localhost", port=8001),
            Server(id="server-2", host="localhost", port=8002)
        ]

        pool = ServerPool(name="drain-pool", servers=servers)
        load_balancer.add_pool(pool)

        # Set server to draining
        servers[0].state = ServerState.DRAINING

        # Should not select draining server
        for _ in range(5):
            server = await load_balancer.select_server("drain-pool")
            assert server.id == "server-2"

    @pytest.mark.asyncio
    async def test_warming_state(self, load_balancer):
        """Test server warming state"""
        servers = [
            Server(id="server-1", host="localhost", port=8001),
            Server(id="server-2", host="localhost", port=8002, state=ServerState.WARMING)
        ]

        # Warming servers could gradually receive traffic
        # Implementation would need to handle this specially

    @pytest.mark.asyncio
    async def test_disabled_state(self, load_balancer):
        """Test server disabled state"""
        servers = [
            Server(id="server-1", host="localhost", port=8001),
            Server(id="server-2", host="localhost", port=8002, state=ServerState.DISABLED)
        ]

        pool = ServerPool(name="disabled-pool", servers=servers)
        load_balancer.add_pool(pool)

        # Should not select disabled server
        for _ in range(5):
            server = await load_balancer.select_server("disabled-pool")
            assert server.id == "server-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])