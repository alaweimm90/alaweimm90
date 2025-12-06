"""
Throughput Benchmarking for ORCHEX

Measures request throughput, task completion rates, and system capacity.
"""

import pytest
import asyncio
import time
import statistics
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from atlas_core.engine import ATLASEngine
from atlas_core.monitoring import MetricsCollector


class ThroughputBenchmark:
    """Benchmark suite for measuring system throughput."""

    def __init__(self):
        self.engine = ATLASEngine(
            redis_url="redis://localhost:6379/0",
            performance_mode=True,
            worker_threads=mp.cpu_count(),
        )
        self.metrics_collector = MetricsCollector()
        self.results = {}

    async def benchmark_workflow_throughput(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Measure workflow execution throughput."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        completed_workflows = 0
        failed_workflows = 0
        latencies = []

        print(f"Running workflow throughput benchmark for {duration_seconds} seconds...")

        async def execute_workflow():
            nonlocal completed_workflows, failed_workflows
            workflow = {
                "id": f"perf_test_{time.time()}",
                "type": "simple",
                "tasks": ["analyze", "process", "report"],
            }

            try:
                workflow_start = time.time()
                result = await self.engine.execute_workflow(workflow)
                workflow_end = time.time()

                if result["status"] == "completed":
                    completed_workflows += 1
                    latencies.append(workflow_end - workflow_start)
                else:
                    failed_workflows += 1
            except Exception:
                failed_workflows += 1

        # Create continuous workload
        tasks = []
        while time.time() < end_time:
            tasks.append(asyncio.create_task(execute_workflow()))
            await asyncio.sleep(0.001)  # Small delay to control rate

        # Wait for remaining tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate metrics
        total_time = time.time() - start_time
        throughput = completed_workflows / total_time

        return {
            "duration_seconds": total_time,
            "completed_workflows": completed_workflows,
            "failed_workflows": failed_workflows,
            "workflows_per_second": throughput,
            "average_latency": statistics.mean(latencies) if latencies else 0,
            "p50_latency": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
        }

    async def benchmark_agent_throughput(self) -> Dict[str, Any]:
        """Measure individual agent processing throughput."""
        agents = [
            "LiteratureReviewAgent",
            "DataAnalysisAgent",
            "SynthesisAgent",
            "HypothesisGeneratorAgent",
            "ExperimentDesignerAgent",
        ]

        agent_metrics = {}

        for agent_name in agents:
            print(f"Benchmarking {agent_name}...")

            completed_tasks = 0
            processing_times = []
            start_time = time.time()

            # Run agent tasks for 30 seconds
            while time.time() - start_time < 30:
                task = {
                    "agent": agent_name,
                    "input": {"data": f"test_input_{time.time()}"},
                }

                task_start = time.time()
                try:
                    result = await self.engine.execute_agent_task(task)
                    if result.get("success"):
                        completed_tasks += 1
                        processing_times.append(time.time() - task_start)
                except Exception:
                    pass

                await asyncio.sleep(0.01)

            agent_metrics[agent_name] = {
                "tasks_completed": completed_tasks,
                "tasks_per_second": completed_tasks / 30,
                "avg_processing_time": (
                    statistics.mean(processing_times) if processing_times else 0
                ),
                "p95_processing_time": (
                    np.percentile(processing_times, 95) if processing_times else 0
                ),
            }

        return agent_metrics

    async def benchmark_blackboard_throughput(self) -> Dict[str, Any]:
        """Measure blackboard read/write throughput."""
        print("Benchmarking blackboard operations...")

        # Write throughput
        write_count = 0
        write_start = time.time()

        while time.time() - write_start < 10:
            key = f"perf_test_{write_count}"
            value = {"data": f"value_{write_count}", "timestamp": time.time()}
            await self.engine.blackboard.write(key, value)
            write_count += 1

        write_throughput = write_count / 10

        # Read throughput
        read_count = 0
        read_start = time.time()

        while time.time() - read_start < 10:
            key = f"perf_test_{read_count % write_count}"
            await self.engine.blackboard.read(key)
            read_count += 1

        read_throughput = read_count / 10

        # Mixed read/write throughput
        mixed_count = 0
        mixed_start = time.time()

        while time.time() - mixed_start < 10:
            if mixed_count % 3 == 0:  # 33% writes
                key = f"mixed_test_{mixed_count}"
                await self.engine.blackboard.write(key, {"value": mixed_count})
            else:  # 67% reads
                key = f"mixed_test_{mixed_count % 100}"
                await self.engine.blackboard.read(key)
            mixed_count += 1

        mixed_throughput = mixed_count / 10

        return {
            "write_ops_per_second": write_throughput,
            "read_ops_per_second": read_throughput,
            "mixed_ops_per_second": mixed_throughput,
            "read_write_ratio": read_throughput / write_throughput if write_throughput > 0 else 0,
        }

    async def benchmark_concurrent_workflows(self) -> Dict[str, Any]:
        """Measure throughput with varying concurrency levels."""
        print("Benchmarking concurrent workflow execution...")

        concurrency_levels = [1, 5, 10, 25, 50, 100, 200]
        results = {}

        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")

            workflows = []
            for i in range(concurrency):
                workflow = {
                    "id": f"concurrent_{concurrency}_{i}",
                    "type": "standard",
                    "tasks": ["task1", "task2", "task3"],
                }
                workflows.append(workflow)

            start_time = time.time()

            # Execute all workflows concurrently
            tasks = [self.engine.execute_workflow(w) for w in workflows]
            workflow_results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time

            successful = sum(
                1 for r in workflow_results
                if isinstance(r, dict) and r.get("status") == "completed"
            )

            results[concurrency] = {
                "total_workflows": concurrency,
                "successful_workflows": successful,
                "duration_seconds": duration,
                "workflows_per_second": successful / duration if duration > 0 else 0,
                "average_completion_time": duration / concurrency if concurrency > 0 else 0,
            }

        return results

    async def benchmark_data_pipeline_throughput(self) -> Dict[str, Any]:
        """Measure data processing pipeline throughput."""
        print("Benchmarking data pipeline throughput...")

        record_sizes = [100, 1000, 10000, 100000]
        results = {}

        for size in record_sizes:
            print(f"Processing {size} records...")

            # Generate test data
            records = [{"id": i, "value": i * 2, "timestamp": time.time()} for i in range(size)]

            pipeline = {
                "id": f"pipeline_{size}",
                "stages": ["validate", "transform", "aggregate", "store"],
                "data": records,
            }

            start_time = time.time()
            result = await self.engine.execute_data_pipeline(pipeline)
            duration = time.time() - start_time

            results[size] = {
                "records_processed": size,
                "duration_seconds": duration,
                "records_per_second": size / duration if duration > 0 else 0,
                "mb_per_second": (size * 0.001) / duration if duration > 0 else 0,  # Assuming ~1KB per record
                "stages_completed": result.get("completed_stages", 0),
            }

        return results

    async def benchmark_api_request_throughput(self) -> Dict[str, Any]:
        """Measure API request handling throughput."""
        print("Benchmarking API request throughput...")

        async with aiohttp.ClientSession() as session:
            # Warm up
            for _ in range(10):
                async with session.get("http://localhost:8080/health") as resp:
                    await resp.text()

            # Measure throughput
            request_counts = {}
            endpoints = [
                "/api/v1/workflow/list",
                "/api/v1/agents/list",
                "/api/v1/status",
                "/api/v1/metrics",
            ]

            for endpoint in endpoints:
                start_time = time.time()
                request_count = 0
                errors = 0

                while time.time() - start_time < 10:
                    try:
                        async with session.get(f"http://localhost:8080{endpoint}") as resp:
                            if resp.status == 200:
                                request_count += 1
                            else:
                                errors += 1
                    except Exception:
                        errors += 1

                request_counts[endpoint] = {
                    "requests_per_second": request_count / 10,
                    "error_rate": errors / (request_count + errors) if (request_count + errors) > 0 else 0,
                }

        return request_counts

    async def benchmark_event_processing_throughput(self) -> Dict[str, Any]:
        """Measure event processing throughput."""
        print("Benchmarking event processing throughput...")

        events_published = 0
        events_processed = 0
        start_time = time.time()

        # Event handler
        async def process_event(event):
            nonlocal events_processed
            # Simulate processing
            await asyncio.sleep(0.001)
            events_processed += 1

        # Subscribe to events
        await self.engine.event_bus.subscribe(["benchmark.event"], process_event)

        # Publish events for 30 seconds
        publish_duration = 30
        while time.time() - start_time < publish_duration:
            event = {
                "id": events_published,
                "type": "benchmark.event",
                "timestamp": time.time(),
                "data": {"value": events_published * 2},
            }
            await self.engine.event_bus.publish("benchmark.event", event)
            events_published += 1
            await asyncio.sleep(0.0001)  # Very small delay

        # Wait for processing to complete
        await asyncio.sleep(2)

        return {
            "events_published": events_published,
            "events_processed": events_processed,
            "publish_rate": events_published / publish_duration,
            "process_rate": events_processed / (time.time() - start_time),
            "processing_efficiency": events_processed / events_published if events_published > 0 else 0,
        }

    async def benchmark_batch_processing_throughput(self) -> Dict[str, Any]:
        """Measure batch processing throughput."""
        print("Benchmarking batch processing...")

        batch_sizes = [10, 50, 100, 500, 1000]
        results = {}

        for batch_size in batch_sizes:
            print(f"Processing batch size: {batch_size}")

            batches_processed = 0
            total_items = 0
            start_time = time.time()

            # Process batches for 30 seconds
            while time.time() - start_time < 30:
                batch = [{"id": i, "data": f"item_{i}"} for i in range(batch_size)]

                batch_start = time.time()
                result = await self.engine.process_batch(batch)
                batch_duration = time.time() - batch_start

                if result.get("success"):
                    batches_processed += 1
                    total_items += batch_size

            duration = time.time() - start_time

            results[batch_size] = {
                "batches_processed": batches_processed,
                "total_items": total_items,
                "batches_per_second": batches_processed / duration,
                "items_per_second": total_items / duration,
                "avg_batch_time": duration / batches_processed if batches_processed > 0 else 0,
            }

        return results

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all throughput benchmarks."""
        print("Starting comprehensive throughput benchmarks...\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": mp.cpu_count(),
                "worker_threads": self.engine.worker_threads,
            },
            "benchmarks": {},
        }

        # Run each benchmark
        benchmarks = [
            ("workflow_throughput", self.benchmark_workflow_throughput()),
            ("agent_throughput", self.benchmark_agent_throughput()),
            ("blackboard_throughput", self.benchmark_blackboard_throughput()),
            ("concurrent_workflows", self.benchmark_concurrent_workflows()),
            ("data_pipeline", self.benchmark_data_pipeline_throughput()),
            ("api_requests", self.benchmark_api_request_throughput()),
            ("event_processing", self.benchmark_event_processing_throughput()),
            ("batch_processing", self.benchmark_batch_processing_throughput()),
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

        # Generate summary
        results["summary"] = self._generate_summary(results["benchmarks"])

        return results

    def _generate_summary(self, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from all benchmarks."""
        summary = {
            "key_metrics": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Extract key metrics
        if "workflow_throughput" in benchmarks:
            wf = benchmarks["workflow_throughput"]
            summary["key_metrics"]["max_workflows_per_second"] = wf.get("workflows_per_second", 0)
            summary["key_metrics"]["p95_latency"] = wf.get("p95_latency", 0)

        if "blackboard_throughput" in benchmarks:
            bb = benchmarks["blackboard_throughput"]
            summary["key_metrics"]["blackboard_ops_per_second"] = bb.get("mixed_ops_per_second", 0)

        if "event_processing" in benchmarks:
            ep = benchmarks["event_processing"]
            summary["key_metrics"]["event_processing_rate"] = ep.get("process_rate", 0)

        # Identify bottlenecks
        if summary["key_metrics"].get("max_workflows_per_second", 0) < 100:
            summary["bottlenecks"].append("Low workflow throughput")

        if summary["key_metrics"].get("p95_latency", 0) > 1.0:
            summary["bottlenecks"].append("High workflow latency")

        # Generate recommendations
        if summary["bottlenecks"]:
            summary["recommendations"].append("Consider scaling horizontally")
            summary["recommendations"].append("Optimize agent processing logic")
            summary["recommendations"].append("Implement caching strategies")

        return summary


async def main():
    """Main function to run benchmarks."""
    benchmark = ThroughputBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Save results
    with open("benchmark_throughput_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for key, value in results["summary"]["key_metrics"].items():
        print(f"{key}: {value:.2f}")

    if results["summary"]["bottlenecks"]:
        print("\nBottlenecks Identified:")
        for bottleneck in results["summary"]["bottlenecks"]:
            print(f"  - {bottleneck}")

    if results["summary"]["recommendations"]:
        print("\nRecommendations:")
        for rec in results["summary"]["recommendations"]:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())