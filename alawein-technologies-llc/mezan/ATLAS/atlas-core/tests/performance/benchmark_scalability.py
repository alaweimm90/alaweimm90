"""
Scalability Benchmarking for ORCHEX

Tests horizontal and vertical scaling capabilities.
"""

import pytest
import asyncio
import time
import json
import psutil
import statistics
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray

from atlas_core.engine import ATLASEngine
from atlas_core.distributed import DistributedCoordinator


class ScalabilityBenchmark:
    """Benchmark suite for testing system scalability."""

    def __init__(self):
        self.base_engine = ATLASEngine(
            redis_url="redis://localhost:6379/0",
            performance_mode=True,
        )
        self.results = {}

    async def benchmark_horizontal_scaling(self) -> Dict[str, Any]:
        """Test horizontal scaling by adding more nodes."""
        print("Testing horizontal scaling...")

        node_counts = [1, 2, 4, 8, 16]
        results = {}

        for node_count in node_counts:
            print(f"Testing with {node_count} nodes...")

            # Create distributed coordinator
            coordinator = DistributedCoordinator(
                nodes=node_count,
                strategy="load_balanced",
            )

            # Initialize cluster
            nodes = []
            for i in range(node_count):
                node = ATLASEngine(
                    redis_url="redis://localhost:6379/0",
                    node_id=f"node_{i}",
                    is_distributed=True,
                )
                nodes.append(node)

            await coordinator.initialize_cluster(nodes)

            # Run workload
            num_workflows = 100
            workflows = [
                {
                    "id": f"scale_test_{node_count}_{i}",
                    "tasks": ["analyze", "process", "synthesize"] * 3,
                }
                for i in range(num_workflows)
            ]

            start_time = time.time()

            # Distribute and execute workflows
            distributed_workflows = await coordinator.distribute_workflows(workflows)
            results_list = await coordinator.execute_distributed(distributed_workflows)

            duration = time.time() - start_time

            successful = sum(1 for r in results_list if r.get("status") == "completed")

            results[node_count] = {
                "nodes": node_count,
                "total_workflows": num_workflows,
                "successful_workflows": successful,
                "duration_seconds": duration,
                "workflows_per_second": successful / duration,
                "workflows_per_node": successful / node_count,
                "efficiency": (successful / node_count) / (successful / 1) if node_count > 1 else 1.0,
            }

            # Cleanup
            await coordinator.shutdown_cluster()

        return results

    async def benchmark_vertical_scaling(self) -> Dict[str, Any]:
        """Test vertical scaling by increasing resources."""
        print("Testing vertical scaling...")

        resource_configs = [
            {"workers": 1, "memory_mb": 512},
            {"workers": 2, "memory_mb": 1024},
            {"workers": 4, "memory_mb": 2048},
            {"workers": 8, "memory_mb": 4096},
            {"workers": 16, "memory_mb": 8192},
        ]

        results = {}

        for config in resource_configs:
            print(f"Testing with {config['workers']} workers, {config['memory_mb']}MB memory...")

            # Create engine with specific resource configuration
            engine = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                worker_threads=config["workers"],
                max_memory_mb=config["memory_mb"],
            )

            # Run workload
            num_tasks = 200
            tasks = []

            start_time = time.time()

            for i in range(num_tasks):
                task = engine.execute_workflow({
                    "id": f"vertical_test_{i}",
                    "complexity": "medium",
                    "data_size_mb": 10,
                })
                tasks.append(task)

            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time

            successful = sum(1 for r in results_list if isinstance(r, dict) and r.get("status") == "completed")

            # Monitor resource usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()

            results[f"{config['workers']}_workers"] = {
                "workers": config["workers"],
                "memory_limit_mb": config["memory_mb"],
                "tasks_completed": successful,
                "duration_seconds": duration,
                "tasks_per_second": successful / duration,
                "tasks_per_worker": successful / config["workers"],
                "actual_memory_mb": memory_usage,
                "memory_efficiency": (config["memory_mb"] - memory_usage) / config["memory_mb"],
                "cpu_utilization": cpu_percent,
            }

            await engine.shutdown()

        return results

    async def benchmark_database_scaling(self) -> Dict[str, Any]:
        """Test database connection pool scaling."""
        print("Testing database connection pool scaling...")

        pool_sizes = [5, 10, 20, 50, 100]
        results = {}

        for pool_size in pool_sizes:
            print(f"Testing with pool size: {pool_size}")

            # Configure database pool
            engine = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                db_pool_size=pool_size,
                db_max_overflow=pool_size // 2,
            )

            # Run database-intensive workload
            num_operations = 500
            operations = []

            start_time = time.time()

            for i in range(num_operations):
                op = engine.execute_database_operation({
                    "type": "complex_query",
                    "tables": ["workflows", "agents", "results"],
                    "joins": 2,
                })
                operations.append(op)

            results_list = await asyncio.gather(*operations, return_exceptions=True)
            duration = time.time() - start_time

            successful = sum(1 for r in results_list if not isinstance(r, Exception))

            # Calculate connection pool metrics
            pool_stats = await engine.get_pool_statistics()

            results[pool_size] = {
                "pool_size": pool_size,
                "operations": num_operations,
                "successful": successful,
                "duration_seconds": duration,
                "ops_per_second": successful / duration,
                "pool_utilization": pool_stats.get("avg_utilization", 0),
                "pool_wait_time_ms": pool_stats.get("avg_wait_time", 0) * 1000,
                "connection_reuse_rate": pool_stats.get("reuse_rate", 0),
            }

            await engine.shutdown()

        return results

    async def benchmark_cache_scaling(self) -> Dict[str, Any]:
        """Test cache scaling impact on performance."""
        print("Testing cache scaling...")

        cache_sizes = [0, 100, 1000, 10000, 100000]
        results = {}

        # Create test data
        test_keys = [f"key_{i}" for i in range(10000)]
        test_values = {key: f"value_{i}" * 100 for i, key in enumerate(test_keys)}

        for cache_size in cache_sizes:
            print(f"Testing with cache size: {cache_size}")

            engine = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                cache_size=cache_size,
                cache_ttl=300,
            )

            # Warm up cache (if enabled)
            if cache_size > 0:
                for key in test_keys[:min(cache_size, len(test_keys))]:
                    await engine.cache_set(key, test_values[key])

            # Measure performance
            hits = 0
            misses = 0
            total_time = 0

            for _ in range(1000):
                key = test_keys[np.random.randint(0, len(test_keys))]

                start = time.time()
                result = await engine.cache_get(key)
                duration = time.time() - start
                total_time += duration

                if result is not None:
                    hits += 1
                else:
                    misses += 1
                    # Simulate fetching from source
                    await asyncio.sleep(0.01)
                    if cache_size > 0:
                        await engine.cache_set(key, test_values[key])

            results[cache_size] = {
                "cache_size": cache_size,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0,
                "avg_access_time_ms": (total_time / 1000) * 1000,
                "memory_usage_mb": await engine.get_cache_memory_usage() / 1024 / 1024,
            }

            await engine.shutdown()

        return results

    async def benchmark_queue_scaling(self) -> Dict[str, Any]:
        """Test message queue scaling."""
        print("Testing queue scaling...")

        queue_configs = [
            {"workers": 1, "queue_size": 100},
            {"workers": 2, "queue_size": 500},
            {"workers": 4, "queue_size": 1000},
            {"workers": 8, "queue_size": 5000},
            {"workers": 16, "queue_size": 10000},
        ]

        results = {}

        for config in queue_configs:
            print(f"Testing with {config['workers']} workers, queue size {config['queue_size']}")

            engine = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                queue_workers=config["workers"],
                max_queue_size=config["queue_size"],
            )

            # Generate messages
            num_messages = 1000
            messages = [
                {"id": i, "type": "process", "data": f"message_{i}"}
                for i in range(num_messages)
            ]

            # Enqueue all messages
            enqueue_start = time.time()
            for msg in messages:
                await engine.enqueue_message(msg)
            enqueue_duration = time.time() - enqueue_start

            # Process messages
            process_start = time.time()
            await engine.process_queue()
            process_duration = time.time() - process_start

            queue_stats = await engine.get_queue_statistics()

            results[f"{config['workers']}_workers"] = {
                "workers": config["workers"],
                "queue_size": config["queue_size"],
                "messages": num_messages,
                "enqueue_rate": num_messages / enqueue_duration,
                "process_rate": num_messages / process_duration,
                "avg_wait_time_ms": queue_stats.get("avg_wait_time", 0) * 1000,
                "max_queue_depth": queue_stats.get("max_depth", 0),
                "worker_utilization": queue_stats.get("worker_utilization", 0),
            }

            await engine.shutdown()

        return results

    async def benchmark_auto_scaling(self) -> Dict[str, Any]:
        """Test auto-scaling capabilities."""
        print("Testing auto-scaling...")

        # Configure auto-scaling
        auto_scale_config = {
            "min_nodes": 1,
            "max_nodes": 10,
            "scale_up_threshold": 0.8,  # 80% CPU
            "scale_down_threshold": 0.3,  # 30% CPU
            "scale_up_cooldown": 60,
            "scale_down_cooldown": 180,
        }

        coordinator = DistributedCoordinator(
            auto_scale=True,
            auto_scale_config=auto_scale_config,
        )

        # Simulate varying load
        load_pattern = [
            {"duration": 60, "load": "low", "workflows_per_second": 1},
            {"duration": 120, "load": "medium", "workflows_per_second": 10},
            {"duration": 180, "load": "high", "workflows_per_second": 50},
            {"duration": 120, "load": "medium", "workflows_per_second": 10},
            {"duration": 60, "load": "low", "workflows_per_second": 1},
        ]

        results = {
            "scaling_events": [],
            "metrics": [],
        }

        for phase in load_pattern:
            print(f"Testing {phase['load']} load for {phase['duration']} seconds...")

            start_time = time.time()
            phase_end = start_time + phase["duration"]

            while time.time() < phase_end:
                # Generate workflows at specified rate
                workflows_to_create = int(phase["workflows_per_second"])

                for _ in range(workflows_to_create):
                    workflow = {
                        "id": f"autoscale_{time.time()}",
                        "tasks": ["process"] * 5,
                    }
                    asyncio.create_task(coordinator.execute_workflow(workflow))

                # Collect metrics
                metrics = await coordinator.get_cluster_metrics()
                results["metrics"].append({
                    "timestamp": time.time(),
                    "load": phase["load"],
                    "node_count": metrics["node_count"],
                    "cpu_usage": metrics["avg_cpu"],
                    "memory_usage": metrics["avg_memory"],
                    "queue_depth": metrics["queue_depth"],
                })

                # Check for scaling events
                scaling_event = await coordinator.check_scaling()
                if scaling_event:
                    results["scaling_events"].append({
                        "timestamp": time.time(),
                        "type": scaling_event["type"],
                        "from_nodes": scaling_event["from"],
                        "to_nodes": scaling_event["to"],
                        "reason": scaling_event["reason"],
                    })

                await asyncio.sleep(1)

        # Analyze scaling behavior
        results["analysis"] = {
            "total_scaling_events": len(results["scaling_events"]),
            "scale_up_events": sum(1 for e in results["scaling_events"] if e["type"] == "scale_up"),
            "scale_down_events": sum(1 for e in results["scaling_events"] if e["type"] == "scale_down"),
            "max_nodes": max(m["node_count"] for m in results["metrics"]),
            "min_nodes": min(m["node_count"] for m in results["metrics"]),
            "avg_response_time": coordinator.get_avg_response_time(),
        }

        await coordinator.shutdown_cluster()

        return results

    async def benchmark_load_balancing_strategies(self) -> Dict[str, Any]:
        """Test different load balancing strategies."""
        print("Testing load balancing strategies...")

        strategies = [
            "round_robin",
            "least_connections",
            "weighted_round_robin",
            "ip_hash",
            "least_response_time",
            "random",
        ]

        results = {}

        # Create test cluster
        num_nodes = 4
        nodes = []
        for i in range(num_nodes):
            node = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                node_id=f"node_{i}",
                is_distributed=True,
                # Simulate different node capabilities
                worker_threads=2 * (i + 1),  # Node 0: 2 workers, Node 3: 8 workers
            )
            nodes.append(node)

        for strategy in strategies:
            print(f"Testing {strategy} strategy...")

            coordinator = DistributedCoordinator(
                nodes=num_nodes,
                load_balancing_strategy=strategy,
            )

            await coordinator.initialize_cluster(nodes)

            # Generate workload
            num_requests = 1000
            request_latencies = []
            node_distribution = {f"node_{i}": 0 for i in range(num_nodes)}

            for i in range(num_requests):
                request = {
                    "id": f"lb_test_{strategy}_{i}",
                    "client_ip": f"192.168.1.{i % 256}",
                    "task": "process",
                }

                start = time.time()
                result = await coordinator.route_request(request)
                latency = time.time() - start

                request_latencies.append(latency)
                node_distribution[result["node_id"]] += 1

            # Calculate metrics
            results[strategy] = {
                "strategy": strategy,
                "total_requests": num_requests,
                "avg_latency_ms": statistics.mean(request_latencies) * 1000,
                "p95_latency_ms": np.percentile(request_latencies, 95) * 1000,
                "node_distribution": node_distribution,
                "distribution_variance": statistics.variance(list(node_distribution.values())),
                "fairness_index": self._calculate_fairness_index(list(node_distribution.values())),
            }

            await coordinator.shutdown_cluster()

        return results

    def _calculate_fairness_index(self, values: List[int]) -> float:
        """Calculate Jain's fairness index."""
        if not values:
            return 0.0

        sum_values = sum(values)
        sum_squares = sum(v * v for v in values)
        n = len(values)

        if sum_squares == 0:
            return 0.0

        return (sum_values * sum_values) / (n * sum_squares)

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all scalability benchmarks."""
        print("Starting comprehensive scalability benchmarks...\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": mp.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "benchmarks": {},
        }

        benchmarks = [
            ("horizontal_scaling", self.benchmark_horizontal_scaling()),
            ("vertical_scaling", self.benchmark_vertical_scaling()),
            ("database_scaling", self.benchmark_database_scaling()),
            ("cache_scaling", self.benchmark_cache_scaling()),
            ("queue_scaling", self.benchmark_queue_scaling()),
            ("auto_scaling", self.benchmark_auto_scaling()),
            ("load_balancing", self.benchmark_load_balancing_strategies()),
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
    """Main function to run scalability benchmarks."""
    benchmark = ScalabilityBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Save results
    with open("benchmark_scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark results saved to benchmark_scalability_results.json")


if __name__ == "__main__":
    asyncio.run(main())