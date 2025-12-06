"""
Distributed Performance Benchmarking for ORCHEX

Tests multi-node performance, network overhead, and distributed coordination.
"""

import pytest
import asyncio
import time
import json
import statistics
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import ray
import dask.distributed
from mpi4py import MPI
import socket
import hashlib

from atlas_core.engine import ATLASEngine
from atlas_core.distributed import DistributedCoordinator, NodeManager


class DistributedBenchmark:
    """Benchmark suite for distributed system performance."""

    def __init__(self):
        self.coordinator = None
        self.nodes = []
        self.results = {}

    async def setup_cluster(self, num_nodes: int) -> None:
        """Setup distributed cluster for testing."""
        self.coordinator = DistributedCoordinator(
            nodes=num_nodes,
            strategy="data_parallel",
            fault_tolerance=True,
        )

        self.nodes = []
        for i in range(num_nodes):
            node = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                node_id=f"node_{i}",
                is_distributed=True,
                host=f"127.0.0.{i+1}",
                port=8000 + i,
            )
            self.nodes.append(node)

        await self.coordinator.initialize_cluster(self.nodes)

    async def teardown_cluster(self) -> None:
        """Teardown distributed cluster."""
        if self.coordinator:
            await self.coordinator.shutdown_cluster()

    async def benchmark_data_parallel_processing(self) -> Dict[str, Any]:
        """Benchmark data parallel processing across nodes."""
        print("Benchmarking data parallel processing...")

        await self.setup_cluster(4)

        data_sizes = [1000, 10000, 100000, 1000000]
        results = {}

        for size in data_sizes:
            print(f"Processing {size} records...")

            # Generate test data
            data = [{"id": i, "value": np.random.random()} for i in range(size)]

            # Partition data across nodes
            start_time = time.time()
            partitions = await self.coordinator.partition_data(data, num_partitions=4)
            partition_time = time.time() - start_time

            # Process partitions in parallel
            process_start = time.time()
            tasks = []
            for node_id, partition in partitions.items():
                task = self.nodes[int(node_id.split("_")[1])].process_data(partition)
                tasks.append(task)

            partition_results = await asyncio.gather(*tasks)
            process_time = time.time() - process_start

            # Aggregate results
            agg_start = time.time()
            final_result = await self.coordinator.aggregate_results(partition_results)
            agg_time = time.time() - agg_start

            total_time = time.time() - start_time

            results[size] = {
                "records": size,
                "partition_time": partition_time,
                "process_time": process_time,
                "aggregation_time": agg_time,
                "total_time": total_time,
                "throughput": size / total_time,
                "speedup": total_time / (size * 0.001),  # Assuming 1ms per record baseline
            }

        await self.teardown_cluster()
        return results

    async def benchmark_task_parallel_execution(self) -> Dict[str, Any]:
        """Benchmark task parallel execution across nodes."""
        print("Benchmarking task parallel execution...")

        await self.setup_cluster(8)

        task_counts = [10, 50, 100, 500, 1000]
        results = {}

        for count in task_counts:
            print(f"Executing {count} parallel tasks...")

            # Create independent tasks
            tasks = []
            for i in range(count):
                task = {
                    "id": f"task_{i}",
                    "type": "compute",
                    "complexity": np.random.choice(["low", "medium", "high"]),
                    "dependencies": [],
                }
                tasks.append(task)

            # Distribute tasks across nodes
            start_time = time.time()
            task_assignments = await self.coordinator.schedule_tasks(tasks)

            # Execute tasks in parallel
            execution_tasks = []
            for node_id, node_tasks in task_assignments.items():
                node_idx = int(node_id.split("_")[1])
                for task in node_tasks:
                    exec_task = self.nodes[node_idx].execute_task(task)
                    execution_tasks.append(exec_task)

            results_list = await asyncio.gather(*execution_tasks)
            total_time = time.time() - start_time

            successful = sum(1 for r in results_list if r.get("status") == "completed")

            results[count] = {
                "total_tasks": count,
                "successful_tasks": successful,
                "total_time": total_time,
                "tasks_per_second": successful / total_time,
                "avg_task_time": total_time / count,
                "parallel_efficiency": (count / total_time) / 8,  # 8 nodes
            }

        await self.teardown_cluster()
        return results

    async def benchmark_communication_overhead(self) -> Dict[str, Any]:
        """Benchmark inter-node communication overhead."""
        print("Benchmarking communication overhead...")

        await self.setup_cluster(4)

        message_sizes = [100, 1000, 10000, 100000, 1000000]  # bytes
        results = {}

        for size in message_sizes:
            print(f"Testing message size: {size} bytes")

            message = b"x" * size
            latencies = []
            throughputs = []

            # Test all-to-all communication
            for source in range(4):
                for dest in range(4):
                    if source != dest:
                        start = time.time()
                        await self.coordinator.send_message(
                            from_node=f"node_{source}",
                            to_node=f"node_{dest}",
                            message=message,
                        )
                        latency = time.time() - start
                        latencies.append(latency)
                        throughputs.append(size / latency)

            results[size] = {
                "message_size_bytes": size,
                "avg_latency_ms": statistics.mean(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000,
                "avg_throughput_mb_s": statistics.mean(throughputs) / (1024 * 1024),
                "total_messages": len(latencies),
            }

        await self.teardown_cluster()
        return results

    async def benchmark_consensus_protocols(self) -> Dict[str, Any]:
        """Benchmark distributed consensus protocols."""
        print("Benchmarking consensus protocols...")

        await self.setup_cluster(5)  # Odd number for consensus

        protocols = ["raft", "paxos", "pbft"]
        results = {}

        for protocol in protocols:
            print(f"Testing {protocol} protocol...")

            self.coordinator.set_consensus_protocol(protocol)

            # Test consensus rounds
            latencies = []
            success_rate = 0
            total_rounds = 100

            for i in range(total_rounds):
                proposal = {
                    "id": f"proposal_{i}",
                    "value": f"value_{i}",
                    "timestamp": time.time(),
                }

                start = time.time()
                result = await self.coordinator.achieve_consensus(proposal)
                latency = time.time() - start

                if result["achieved"]:
                    success_rate += 1
                    latencies.append(latency)

            results[protocol] = {
                "protocol": protocol,
                "total_rounds": total_rounds,
                "successful_rounds": success_rate,
                "success_rate": success_rate / total_rounds,
                "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000 if latencies else 0,
            }

        await self.teardown_cluster()
        return results

    async def benchmark_distributed_locking(self) -> Dict[str, Any]:
        """Benchmark distributed locking mechanisms."""
        print("Benchmarking distributed locking...")

        await self.setup_cluster(8)

        lock_types = ["mutex", "read_write", "semaphore", "distributed_queue"]
        results = {}

        for lock_type in lock_types:
            print(f"Testing {lock_type} lock...")

            acquisition_times = []
            release_times = []
            contentions = 0

            # Create concurrent lock requests
            async def acquire_and_release(node_id: int, lock_id: str):
                nonlocal contentions

                # Acquire lock
                acquire_start = time.time()
                acquired = False
                attempts = 0

                while not acquired and attempts < 10:
                    acquired = await self.coordinator.acquire_lock(
                        lock_id=lock_id,
                        node_id=f"node_{node_id}",
                        lock_type=lock_type,
                    )
                    if not acquired:
                        contentions += 1
                        await asyncio.sleep(0.01)
                    attempts += 1

                if acquired:
                    acquire_time = time.time() - acquire_start
                    acquisition_times.append(acquire_time)

                    # Hold lock briefly
                    await asyncio.sleep(0.001)

                    # Release lock
                    release_start = time.time()
                    await self.coordinator.release_lock(
                        lock_id=lock_id,
                        node_id=f"node_{node_id}",
                    )
                    release_times.append(time.time() - release_start)

            # Run concurrent lock operations
            tasks = []
            for _ in range(100):
                node_id = np.random.randint(0, 8)
                lock_id = f"lock_{np.random.randint(0, 10)}"  # 10 different locks
                tasks.append(acquire_and_release(node_id, lock_id))

            await asyncio.gather(*tasks)

            results[lock_type] = {
                "lock_type": lock_type,
                "total_acquisitions": len(acquisition_times),
                "avg_acquire_time_ms": statistics.mean(acquisition_times) * 1000 if acquisition_times else 0,
                "avg_release_time_ms": statistics.mean(release_times) * 1000 if release_times else 0,
                "contentions": contentions,
                "contention_rate": contentions / 100,
            }

        await self.teardown_cluster()
        return results

    async def benchmark_distributed_transactions(self) -> Dict[str, Any]:
        """Benchmark distributed transaction performance."""
        print("Benchmarking distributed transactions...")

        await self.setup_cluster(4)

        transaction_sizes = [1, 5, 10, 50, 100]  # operations per transaction
        results = {}

        for size in transaction_sizes:
            print(f"Testing transaction size: {size} operations")

            commit_times = []
            rollback_times = []
            success_count = 0

            for i in range(100):
                # Create transaction
                transaction = {
                    "id": f"tx_{i}",
                    "operations": [],
                }

                for j in range(size):
                    transaction["operations"].append({
                        "type": "write",
                        "key": f"key_{i}_{j}",
                        "value": f"value_{i}_{j}",
                        "node": f"node_{j % 4}",
                    })

                # Execute transaction
                start = time.time()
                result = await self.coordinator.execute_transaction(transaction)
                duration = time.time() - start

                if result["committed"]:
                    success_count += 1
                    commit_times.append(duration)
                else:
                    rollback_times.append(duration)

            results[size] = {
                "operations_per_transaction": size,
                "total_transactions": 100,
                "successful_transactions": success_count,
                "success_rate": success_count / 100,
                "avg_commit_time_ms": statistics.mean(commit_times) * 1000 if commit_times else 0,
                "avg_rollback_time_ms": statistics.mean(rollback_times) * 1000 if rollback_times else 0,
            }

        await self.teardown_cluster()
        return results

    async def benchmark_ray_integration(self) -> Dict[str, Any]:
        """Benchmark Ray distributed computing integration."""
        print("Benchmarking Ray integration...")

        # Initialize Ray
        ray.init(num_cpus=8, num_gpus=0, ignore_reinit_error=True)

        @ray.remote
        class WorkerActor:
            def process(self, data):
                # Simulate processing
                result = sum(data) * 2
                return result

        # Create actors
        num_actors = 8
        actors = [WorkerActor.remote() for _ in range(num_actors)]

        workload_sizes = [100, 1000, 10000, 100000]
        results = {}

        for size in workload_sizes:
            print(f"Processing {size} items with Ray...")

            # Generate data
            data_chunks = np.array_split(range(size), num_actors)

            start = time.time()

            # Distribute work to actors
            futures = []
            for actor, chunk in zip(actors, data_chunks):
                futures.append(actor.process.remote(chunk.tolist()))

            # Collect results
            results_list = ray.get(futures)
            duration = time.time() - start

            results[size] = {
                "workload_size": size,
                "num_actors": num_actors,
                "duration": duration,
                "throughput": size / duration,
                "items_per_actor": size / num_actors,
            }

        ray.shutdown()
        return results

    async def benchmark_network_partitioning(self) -> Dict[str, Any]:
        """Benchmark performance under network partitioning."""
        print("Benchmarking network partitioning resilience...")

        await self.setup_cluster(6)

        partition_scenarios = [
            {"name": "no_partition", "groups": [list(range(6))]},
            {"name": "half_split", "groups": [[0, 1, 2], [3, 4, 5]]},
            {"name": "isolated_node", "groups": [[0, 1, 2, 3, 4], [5]]},
            {"name": "three_way", "groups": [[0, 1], [2, 3], [4, 5]]},
        ]

        results = {}

        for scenario in partition_scenarios:
            print(f"Testing {scenario['name']} scenario...")

            # Apply network partition
            await self.coordinator.simulate_network_partition(scenario["groups"])

            # Try to execute distributed workflow
            start = time.time()
            success_count = 0
            total_attempts = 50

            for i in range(total_attempts):
                workflow = {
                    "id": f"partition_test_{i}",
                    "tasks": ["task1", "task2", "task3"],
                    "requires_consensus": True,
                }

                try:
                    result = await self.coordinator.execute_workflow(workflow)
                    if result.get("status") == "completed":
                        success_count += 1
                except Exception:
                    pass

            duration = time.time() - start

            # Heal partition
            await self.coordinator.heal_network_partition()

            results[scenario["name"]] = {
                "partition_type": scenario["name"],
                "partition_groups": scenario["groups"],
                "total_attempts": total_attempts,
                "successful": success_count,
                "success_rate": success_count / total_attempts,
                "total_duration": duration,
                "avg_latency_ms": (duration / total_attempts) * 1000,
            }

        await self.teardown_cluster()
        return results

    async def benchmark_distributed_caching(self) -> Dict[str, Any]:
        """Benchmark distributed caching performance."""
        print("Benchmarking distributed caching...")

        await self.setup_cluster(4)

        cache_strategies = ["local", "replicated", "distributed", "near_cache"]
        results = {}

        for strategy in cache_strategies:
            print(f"Testing {strategy} caching strategy...")

            await self.coordinator.configure_cache(strategy=strategy)

            # Populate cache
            cache_size = 10000
            for i in range(cache_size):
                key = f"key_{i}"
                value = f"value_{i}" * 100  # ~500 bytes per entry
                await self.coordinator.cache_set(key, value)

            # Measure cache performance
            hits = 0
            misses = 0
            read_times = []

            for _ in range(1000):
                key = f"key_{np.random.randint(0, cache_size * 2)}"  # 50% miss rate expected

                start = time.time()
                result = await self.coordinator.cache_get(key)
                read_time = time.time() - start
                read_times.append(read_time)

                if result is not None:
                    hits += 1
                else:
                    misses += 1

            results[strategy] = {
                "strategy": strategy,
                "cache_size": cache_size,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / (hits + misses),
                "avg_read_time_ms": statistics.mean(read_times) * 1000,
                "p95_read_time_ms": np.percentile(read_times, 95) * 1000,
            }

        await self.teardown_cluster()
        return results

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all distributed performance benchmarks."""
        print("Starting distributed performance benchmarks...\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
        }

        benchmarks = [
            ("data_parallel", self.benchmark_data_parallel_processing()),
            ("task_parallel", self.benchmark_task_parallel_execution()),
            ("communication_overhead", self.benchmark_communication_overhead()),
            ("consensus_protocols", self.benchmark_consensus_protocols()),
            ("distributed_locking", self.benchmark_distributed_locking()),
            ("distributed_transactions", self.benchmark_distributed_transactions()),
            ("ray_integration", self.benchmark_ray_integration()),
            ("network_partitioning", self.benchmark_network_partitioning()),
            ("distributed_caching", self.benchmark_distributed_caching()),
        ]

        for name, benchmark_coro in benchmarks:
            print(f"\n{'='*60}")
            print(f"Running {name} benchmark...")
            print('='*60)
            try:
                results["benchmarks"][name] = await benchmark_coro
                print(f"✓ {name} benchmark completed")
            except Exception as e:
                print(f"✗ {name} benchmark failed: {e}")
                results["benchmarks"][name] = {"error": str(e)}

        return results


async def main():
    """Main function to run distributed benchmarks."""
    benchmark = DistributedBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Save results
    with open("benchmark_distributed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDistributed benchmark results saved to benchmark_distributed_results.json")


if __name__ == "__main__":
    asyncio.run(main())