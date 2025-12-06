"""
Distributed Execution Integration Tests for ORCHEX

Tests multi-node, distributed workflow execution and coordination.
"""

import pytest
import asyncio
import time
import uuid
import socket
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import ray
import dask.distributed
from mpi4py import MPI

from atlas_core.engine import ATLASEngine
from atlas_core.distributed import (
    DistributedCoordinator,
    NodeManager,
    TaskScheduler,
    MessageBroker,
    ConsensusProtocol,
)
from atlas_core.blackboard import DistributedBlackboard


class TestDistributedExecution:
    """Test distributed execution across multiple nodes."""

    @pytest.fixture
    def coordinator(self):
        """Create distributed coordinator."""
        return DistributedCoordinator(
            nodes=4,
            strategy="data_parallel",
            fault_tolerance=True,
            consensus_algorithm="raft",
        )

    @pytest.fixture
    def node_cluster(self):
        """Create a test cluster of nodes."""
        nodes = []
        for i in range(4):
            node = NodeManager(
                node_id=f"node_{i}",
                host=f"127.0.0.{i+1}",
                port=8000 + i,
                role="worker" if i > 0 else "master",
            )
            nodes.append(node)
        return nodes

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_node_coordination(self, coordinator, node_cluster):
        """Test coordination between multiple nodes."""
        # Initialize cluster
        await coordinator.initialize_cluster(node_cluster)

        # Verify cluster formation
        assert coordinator.get_cluster_size() == 4
        assert coordinator.get_master_node() == "node_0"

        # Test leader election
        await coordinator.trigger_leader_election()
        new_leader = coordinator.get_master_node()
        assert new_leader in [n.node_id for n in node_cluster]

        # Test heartbeat mechanism
        heartbeats = await coordinator.check_heartbeats()
        assert all(heartbeats.values())  # All nodes should be alive

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_task_distribution_strategies(self, coordinator):
        """Test different task distribution strategies."""
        tasks = [
            {"id": f"task_{i}", "complexity": i % 3 + 1, "data_size": 100 * i}
            for i in range(20)
        ]

        # Test round-robin distribution
        rr_distribution = await coordinator.distribute_tasks(tasks, strategy="round_robin")
        assert len(rr_distribution) == 4  # 4 nodes
        assert all(len(node_tasks) == 5 for node_tasks in rr_distribution.values())

        # Test load-balanced distribution
        lb_distribution = await coordinator.distribute_tasks(tasks, strategy="load_balanced")
        complexities = [
            sum(t["complexity"] for t in node_tasks) for node_tasks in lb_distribution.values()
        ]
        assert max(complexities) - min(complexities) <= 2  # Balanced within threshold

        # Test affinity-based distribution
        affinity_distribution = await coordinator.distribute_tasks(
            tasks, strategy="affinity", affinity_key="complexity"
        )
        # Tasks with same complexity should be on same node
        for node_tasks in affinity_distribution.values():
            complexities = [t["complexity"] for t in node_tasks]
            assert len(set(complexities)) <= 2  # At most 2 different complexities per node

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_blackboard_sync(self, coordinator):
        """Test distributed blackboard synchronization."""
        # Create distributed blackboard
        blackboard = DistributedBlackboard(coordinator=coordinator, sync_interval=0.5)

        # Write from different nodes
        node_writes = {}
        for i in range(4):
            key = f"node_{i}_data"
            value = {"source": f"node_{i}", "timestamp": time.time()}
            await blackboard.write(key, value, source_node=f"node_{i}")
            node_writes[key] = value

        # Allow sync time
        await asyncio.sleep(1)

        # Verify all nodes have all data
        for node_id in range(4):
            node_view = await blackboard.get_node_view(f"node_{node_id}")
            assert len(node_view) == 4
            for key, expected_value in node_writes.items():
                assert key in node_view
                assert node_view[key]["source"] == expected_value["source"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_consensus(self, coordinator):
        """Test distributed consensus protocols."""
        # Test Raft consensus
        raft_proposal = {"action": "update_config", "value": {"max_agents": 100}}

        consensus_result = await coordinator.achieve_consensus(
            raft_proposal, algorithm="raft", timeout=5
        )

        assert consensus_result["achieved"] is True
        assert consensus_result["votes"] >= 3  # Majority in 4-node cluster
        assert consensus_result["term"] > 0

        # Test Byzantine Fault Tolerant consensus
        bft_proposal = {"action": "critical_update", "value": {"security_level": "high"}}

        bft_result = await coordinator.achieve_consensus(bft_proposal, algorithm="pbft", timeout=10)

        assert bft_result["achieved"] is True
        assert bft_result["fault_tolerance"] == 1  # Can tolerate 1 byzantine node in 4-node cluster

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_workflow_partitioning(self, coordinator):
        """Test workflow partitioning across nodes."""
        large_workflow = {
            "id": "large_workflow",
            "tasks": [f"task_{i}" for i in range(100)],
            "dependencies": {
                f"task_{i}": [f"task_{j}" for j in range(max(0, i - 3), i)]
                for i in range(100)
            },
        }

        # Partition workflow
        partitions = await coordinator.partition_workflow(
            large_workflow, strategy="min_cut", target_partitions=4
        )

        # Verify partitioning
        assert len(partitions) == 4

        # Verify all tasks are assigned
        all_tasks = set()
        for partition in partitions.values():
            all_tasks.update(partition["tasks"])
        assert len(all_tasks) == 100

        # Verify edge cuts are minimized
        edge_cuts = coordinator.calculate_edge_cuts(partitions, large_workflow["dependencies"])
        assert edge_cuts < 50  # Less than half of potential edges cross partitions

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_node_failure_recovery(self, coordinator, node_cluster):
        """Test recovery from node failures."""
        # Initialize cluster
        await coordinator.initialize_cluster(node_cluster)

        # Simulate node failure
        failed_node = node_cluster[2]
        await coordinator.simulate_node_failure(failed_node.node_id)

        # Verify detection
        await asyncio.sleep(2)  # Wait for heartbeat timeout
        failed_nodes = await coordinator.detect_failed_nodes()
        assert failed_node.node_id in failed_nodes

        # Test task redistribution
        orphaned_tasks = [f"task_{i}" for i in range(5)]
        coordinator.assign_tasks(failed_node.node_id, orphaned_tasks)

        redistribution = await coordinator.handle_node_failure(failed_node.node_id)
        assert redistribution["success"] is True
        assert redistribution["tasks_redistributed"] == 5
        assert redistribution["new_assignments"] is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_data_streaming(self, coordinator):
        """Test distributed data streaming between nodes."""
        # Create streaming pipeline
        pipeline = {
            "source": "node_0",
            "processors": ["node_1", "node_2"],
            "sink": "node_3",
            "batch_size": 100,
            "stream_rate": 1000,  # records per second
        }

        # Start streaming
        stream_handle = await coordinator.start_data_stream(pipeline)

        # Generate test data
        test_data = [{"id": i, "value": i * 2} for i in range(1000)]

        # Stream data
        await coordinator.stream_data(stream_handle, test_data)

        # Verify processing
        await asyncio.sleep(2)  # Allow processing time

        stats = await coordinator.get_stream_statistics(stream_handle)
        assert stats["records_processed"] == 1000
        assert stats["processing_rate"] >= 500  # At least 500 records/sec
        assert stats["errors"] == 0

        # Stop streaming
        await coordinator.stop_stream(stream_handle)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_resource_management(self, coordinator):
        """Test distributed resource allocation and management."""
        # Define resource pool
        resources = {
            "node_0": {"cpu": 16, "memory": 64, "gpu": 2},
            "node_1": {"cpu": 8, "memory": 32, "gpu": 1},
            "node_2": {"cpu": 8, "memory": 32, "gpu": 1},
            "node_3": {"cpu": 4, "memory": 16, "gpu": 0},
        }

        await coordinator.register_resources(resources)

        # Request resources for tasks
        task_requirements = [
            {"task_id": "heavy_task", "cpu": 4, "memory": 16, "gpu": 1},
            {"task_id": "light_task", "cpu": 1, "memory": 2, "gpu": 0},
            {"task_id": "gpu_task", "cpu": 2, "memory": 8, "gpu": 2},
        ]

        allocations = await coordinator.allocate_resources(task_requirements)

        # Verify allocations
        assert allocations["heavy_task"]["node"] in ["node_0", "node_1", "node_2"]
        assert allocations["light_task"]["node"] is not None
        assert allocations["gpu_task"]["node"] == "node_0"  # Only node with 2 GPUs

        # Test resource exhaustion
        excessive_request = [{"task_id": "huge_task", "cpu": 100, "memory": 500, "gpu": 10}]

        with pytest.raises(InsufficientResourcesError):
            await coordinator.allocate_resources(excessive_request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_checkpointing(self, coordinator):
        """Test distributed checkpointing and recovery."""
        # Create workflow with checkpointing
        workflow = {
            "id": "checkpoint_workflow",
            "checkpoint_interval": 5,  # Every 5 tasks
            "checkpoint_strategy": "coordinated",
        }

        # Execute workflow with checkpoints
        execution_handle = await coordinator.start_workflow(workflow)

        # Simulate progress
        for i in range(23):
            await coordinator.complete_task(execution_handle, f"task_{i}")
            if (i + 1) % 5 == 0:
                # Verify checkpoint created
                checkpoint = await coordinator.get_latest_checkpoint(execution_handle)
                assert checkpoint is not None
                assert checkpoint["completed_tasks"] == i + 1

        # Simulate failure and recovery
        await coordinator.simulate_workflow_failure(execution_handle)

        # Recover from checkpoint
        recovery_result = await coordinator.recover_from_checkpoint(execution_handle)
        assert recovery_result["success"] is True
        assert recovery_result["recovered_from_task"] == 20  # Last checkpoint
        assert recovery_result["tasks_to_retry"] == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ray_integration(self, coordinator):
        """Test integration with Ray distributed computing."""
        # Initialize Ray cluster
        ray.init(num_cpus=16, num_gpus=2, ignore_reinit_error=True)

        # Create Ray-based coordinator
        ray_coordinator = coordinator.with_ray_backend()

        # Define Ray actors
        @ray.remote
        class WorkerActor:
            def process(self, data):
                return {"processed": data, "node": ray.get_runtime_context().node_id}

        # Deploy actors
        actors = [WorkerActor.remote() for _ in range(4)]

        # Distribute work using Ray
        tasks = [{"id": i, "data": f"data_{i}"} for i in range(20)]
        futures = []
        for i, task in enumerate(tasks):
            actor = actors[i % len(actors)]
            futures.append(actor.process.remote(task))

        # Collect results
        results = ray.get(futures)
        assert len(results) == 20
        assert all("processed" in r for r in results)

        ray.shutdown()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dask_integration(self, coordinator):
        """Test integration with Dask distributed computing."""
        # Create Dask cluster
        from dask.distributed import Client, as_completed

        client = Client(n_workers=4, threads_per_worker=2, processes=True)

        # Create Dask-based coordinator
        dask_coordinator = coordinator.with_dask_backend(client)

        # Define computation graph
        def process_chunk(data):
            import time

            time.sleep(0.1)  # Simulate processing
            return sum(data)

        # Create large dataset
        data_chunks = [list(range(i * 1000, (i + 1) * 1000)) for i in range(10)]

        # Submit tasks to Dask
        futures = []
        for chunk in data_chunks:
            future = client.submit(process_chunk, chunk)
            futures.append(future)

        # Process results as they complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

        # Verify results
        assert len(results) == 10
        expected_sum = sum(range(10000))
        actual_sum = sum(results)
        assert actual_sum == expected_sum

        client.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_partition_handling(self, coordinator, node_cluster):
        """Test handling of network partitions."""
        # Initialize cluster
        await coordinator.initialize_cluster(node_cluster)

        # Simulate network partition (split brain)
        partition_groups = [["node_0", "node_1"], ["node_2", "node_3"]]
        await coordinator.simulate_network_partition(partition_groups)

        # Test partition detection
        partitions = await coordinator.detect_network_partitions()
        assert len(partitions) == 2

        # Test quorum maintenance
        quorum_group = await coordinator.determine_quorum_group()
        assert len(quorum_group) == 2  # Half of nodes

        # Test partition recovery
        await coordinator.heal_network_partition()
        healed = await coordinator.verify_cluster_consistency()
        assert healed is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_transaction_coordination(self, coordinator):
        """Test distributed transaction coordination (2PC/3PC)."""
        # Create distributed transaction
        transaction = {
            "id": str(uuid.uuid4()),
            "operations": [
                {"node": "node_0", "action": "write", "data": {"key": "a", "value": 1}},
                {"node": "node_1", "action": "write", "data": {"key": "b", "value": 2}},
                {"node": "node_2", "action": "write", "data": {"key": "c", "value": 3}},
            ],
        }

        # Execute 2-phase commit
        commit_result = await coordinator.execute_2pc(transaction)

        assert commit_result["success"] is True
        assert commit_result["phase"] == "committed"
        assert all(commit_result["participant_votes"])

        # Test transaction rollback
        failing_transaction = {
            "id": str(uuid.uuid4()),
            "operations": [
                {"node": "node_0", "action": "write", "data": {"key": "x", "value": 1}},
                {
                    "node": "node_1",
                    "action": "fail",
                    "data": {},
                },  # This will cause rollback
            ],
        }

        rollback_result = await coordinator.execute_2pc(failing_transaction)
        assert rollback_result["success"] is False
        assert rollback_result["phase"] == "aborted"
        assert rollback_result["rollback_complete"] is True