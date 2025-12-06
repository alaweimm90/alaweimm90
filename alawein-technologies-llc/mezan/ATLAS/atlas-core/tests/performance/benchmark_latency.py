"""
Latency Benchmarking for ORCHEX

Measures response times, processing delays, and end-to-end latencies.
"""

import pytest
import asyncio
import time
import statistics
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import aiohttp
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from atlas_core.engine import ATLASEngine


@dataclass
class LatencyMeasurement:
    """Structure for storing latency measurements."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    metadata: Dict[str, Any] = None

    @property
    def latency_ms(self) -> float:
        return self.duration * 1000


class LatencyBenchmark:
    """Comprehensive latency benchmarking suite."""

    def __init__(self):
        self.engine = ATLASEngine(
            redis_url="redis://localhost:6379/0",
            performance_mode=True,
        )
        self.measurements: List[LatencyMeasurement] = []

    def measure_latency(self, operation: str):
        """Decorator to measure operation latency."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    measurement = LatencyMeasurement(
                        operation=operation,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                    )
                    self.measurements.append(measurement)
            return wrapper
        return decorator

    async def benchmark_workflow_latency(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Measure end-to-end workflow latency."""
        print(f"Measuring workflow latency ({num_iterations} iterations)...")

        latencies = []
        stages_latencies = {
            "initialization": [],
            "agent_execution": [],
            "blackboard_sync": [],
            "result_aggregation": [],
            "cleanup": [],
        }

        for i in range(num_iterations):
            workflow = {
                "id": f"latency_test_{i}",
                "type": "standard",
                "tasks": ["analyze", "process", "synthesize"],
            }

            # Measure total latency
            total_start = time.time()

            # Initialization
            init_start = time.time()
            await self.engine.initialize_workflow(workflow)
            stages_latencies["initialization"].append(time.time() - init_start)

            # Agent execution
            agent_start = time.time()
            await self.engine.execute_agents(workflow)
            stages_latencies["agent_execution"].append(time.time() - agent_start)

            # Blackboard sync
            sync_start = time.time()
            await self.engine.sync_blackboard(workflow["id"])
            stages_latencies["blackboard_sync"].append(time.time() - sync_start)

            # Result aggregation
            agg_start = time.time()
            await self.engine.aggregate_results(workflow["id"])
            stages_latencies["result_aggregation"].append(time.time() - agg_start)

            # Cleanup
            cleanup_start = time.time()
            await self.engine.cleanup_workflow(workflow["id"])
            stages_latencies["cleanup"].append(time.time() - cleanup_start)

            total_latency = time.time() - total_start
            latencies.append(total_latency)

        return {
            "total": self._calculate_statistics(latencies),
            "stages": {
                stage: self._calculate_statistics(times)
                for stage, times in stages_latencies.items()
            },
            "breakdown": {
                stage: {
                    "percentage": (sum(times) / sum(latencies)) * 100
                }
                for stage, times in stages_latencies.items()
            }
        }

    async def benchmark_agent_latency(self) -> Dict[str, Any]:
        """Measure individual agent processing latency."""
        print("Measuring agent latencies...")

        agents = [
            "LiteratureReviewAgent",
            "DataAnalysisAgent",
            "SynthesisAgent",
            "HypothesisGeneratorAgent",
            "ExperimentDesignerAgent",
            "QualityCheckAgent",
            "PublicationFormatterAgent",
        ]

        results = {}

        for agent_name in agents:
            latencies = []

            for i in range(50):
                task = {
                    "agent": agent_name,
                    "input": {"data": f"test_data_{i}", "size": "medium"},
                }

                start_time = time.time()
                result = await self.engine.execute_agent_task(task)
                latency = time.time() - start_time

                latencies.append(latency)

            results[agent_name] = self._calculate_statistics(latencies)

        return results

    async def benchmark_api_latency(self) -> Dict[str, Any]:
        """Measure API endpoint latencies."""
        print("Measuring API latencies...")

        endpoints = [
            ("GET", "/api/v1/workflow/list"),
            ("GET", "/api/v1/agents/list"),
            ("POST", "/api/v1/workflow/create"),
            ("GET", "/api/v1/workflow/{id}/status"),
            ("DELETE", "/api/v1/workflow/{id}"),
            ("GET", "/api/v1/metrics"),
        ]

        results = {}

        async with aiohttp.ClientSession() as session:
            for method, endpoint in endpoints:
                latencies = []

                for i in range(100):
                    # Prepare request
                    url = f"http://localhost:8080{endpoint}"
                    if "{id}" in url:
                        url = url.replace("{id}", f"test_{i}")

                    kwargs = {}
                    if method == "POST":
                        kwargs["json"] = {"type": "test", "name": f"test_{i}"}

                    # Measure latency
                    start_time = time.time()
                    try:
                        async with session.request(method, url, **kwargs) as resp:
                            await resp.text()
                            latency = time.time() - start_time
                            latencies.append(latency)
                    except Exception:
                        pass

                if latencies:
                    results[f"{method} {endpoint}"] = self._calculate_statistics(latencies)

        return results

    async def benchmark_blackboard_latency(self) -> Dict[str, Any]:
        """Measure blackboard operation latencies."""
        print("Measuring blackboard latencies...")

        operations = {
            "write": [],
            "read": [],
            "update": [],
            "delete": [],
            "batch_write": [],
            "batch_read": [],
        }

        num_operations = 1000

        # Single operations
        for i in range(num_operations):
            key = f"latency_test_{i}"
            value = {"data": f"value_{i}", "timestamp": time.time()}

            # Write
            start = time.time()
            await self.engine.blackboard.write(key, value)
            operations["write"].append(time.time() - start)

            # Read
            start = time.time()
            await self.engine.blackboard.read(key)
            operations["read"].append(time.time() - start)

            # Update
            start = time.time()
            await self.engine.blackboard.update(key, {"updated": True})
            operations["update"].append(time.time() - start)

            # Delete
            start = time.time()
            await self.engine.blackboard.delete(key)
            operations["delete"].append(time.time() - start)

        # Batch operations
        batch_size = 100
        for i in range(num_operations // batch_size):
            batch_data = {
                f"batch_{i}_{j}": {"value": j}
                for j in range(batch_size)
            }

            # Batch write
            start = time.time()
            await self.engine.blackboard.batch_write(batch_data)
            operations["batch_write"].append(time.time() - start)

            # Batch read
            keys = list(batch_data.keys())
            start = time.time()
            await self.engine.blackboard.batch_read(keys)
            operations["batch_read"].append(time.time() - start)

        return {
            op: self._calculate_statistics(latencies)
            for op, latencies in operations.items()
        }

    async def benchmark_event_latency(self) -> Dict[str, Any]:
        """Measure event processing latencies."""
        print("Measuring event latencies...")

        event_latencies = []
        processing_latencies = []

        received_events = {}

        async def event_handler(event):
            received_time = time.time()
            event_id = event.get("id")
            if event_id in received_events:
                # Calculate end-to-end latency
                publish_time = received_events[event_id]
                latency = received_time - publish_time
                event_latencies.append(latency)

                # Simulate processing
                process_start = time.time()
                await asyncio.sleep(0.001)  # Minimal processing
                processing_latencies.append(time.time() - process_start)

        # Subscribe to events
        await self.engine.event_bus.subscribe(["latency.test"], event_handler)

        # Publish events
        for i in range(1000):
            event_id = f"event_{i}"
            publish_time = time.time()
            received_events[event_id] = publish_time

            event = {
                "id": event_id,
                "type": "latency.test",
                "timestamp": publish_time,
            }

            await self.engine.event_bus.publish("latency.test", event)
            await asyncio.sleep(0.001)

        # Wait for processing
        await asyncio.sleep(2)

        return {
            "publish_to_receive": self._calculate_statistics(event_latencies),
            "processing": self._calculate_statistics(processing_latencies),
        }

    async def benchmark_cold_start_latency(self) -> Dict[str, Any]:
        """Measure cold start latencies."""
        print("Measuring cold start latencies...")

        cold_start_latencies = []
        warm_start_latencies = []

        for i in range(10):
            # Cold start - create new engine instance
            cold_start = time.time()
            cold_engine = ATLASEngine(
                redis_url="redis://localhost:6379/0",
                performance_mode=True,
            )
            await cold_engine.initialize()
            cold_latency = time.time() - cold_start
            cold_start_latencies.append(cold_latency)

            # Warm start - execute on already initialized engine
            warm_start = time.time()
            await cold_engine.execute_workflow({
                "id": f"warmup_{i}",
                "tasks": ["simple_task"],
            })
            warm_latency = time.time() - warm_start
            warm_start_latencies.append(warm_latency)

            # Cleanup
            await cold_engine.shutdown()

        return {
            "cold_start": self._calculate_statistics(cold_start_latencies),
            "warm_start": self._calculate_statistics(warm_start_latencies),
            "improvement_factor": statistics.mean(cold_start_latencies) / statistics.mean(warm_start_latencies),
        }

    async def benchmark_latency_under_load(self) -> Dict[str, Any]:
        """Measure latency under different load conditions."""
        print("Measuring latency under load...")

        load_levels = [1, 10, 50, 100, 200]
        results = {}

        for load in load_levels:
            print(f"Testing load level: {load} concurrent requests")

            latencies = []

            async def single_request():
                start = time.time()
                await self.engine.execute_workflow({
                    "id": f"load_test_{time.time()}",
                    "tasks": ["analyze", "process"],
                })
                return time.time() - start

            # Execute concurrent requests
            tasks = [single_request() for _ in range(load)]
            request_latencies = await asyncio.gather(*tasks)
            latencies.extend(request_latencies)

            results[f"load_{load}"] = self._calculate_statistics(latencies)

        return results

    async def benchmark_network_latency(self) -> Dict[str, Any]:
        """Measure network-related latencies."""
        print("Measuring network latencies...")

        # Simulate different network conditions
        network_conditions = [
            {"name": "local", "delay": 0},
            {"name": "lan", "delay": 0.001},
            {"name": "wan", "delay": 0.050},
            {"name": "satellite", "delay": 0.500},
        ]

        results = {}

        for condition in network_conditions:
            latencies = []

            # Inject network delay
            self.engine.set_network_delay(condition["delay"])

            for i in range(50):
                start = time.time()
                await self.engine.execute_remote_call("test_service", {"data": i})
                latency = time.time() - start
                latencies.append(latency)

            results[condition["name"]] = {
                "statistics": self._calculate_statistics(latencies),
                "injected_delay": condition["delay"],
                "overhead": statistics.mean(latencies) - condition["delay"],
            }

        # Reset network delay
        self.engine.set_network_delay(0)

        return results

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values."""
        if not values:
            return {}

        values_ms = [v * 1000 for v in values]  # Convert to milliseconds

        return {
            "mean_ms": statistics.mean(values_ms),
            "median_ms": statistics.median(values_ms),
            "stdev_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0,
            "min_ms": min(values_ms),
            "max_ms": max(values_ms),
            "p50_ms": np.percentile(values_ms, 50),
            "p75_ms": np.percentile(values_ms, 75),
            "p90_ms": np.percentile(values_ms, 90),
            "p95_ms": np.percentile(values_ms, 95),
            "p99_ms": np.percentile(values_ms, 99),
            "p99_9_ms": np.percentile(values_ms, 99.9),
            "samples": len(values_ms),
        }

    def generate_latency_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted latency report."""
        report = []
        report.append("=" * 80)
        report.append("LATENCY BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        for benchmark_name, data in results.items():
            report.append(f"\n{benchmark_name.upper()}")
            report.append("-" * 40)

            if isinstance(data, dict):
                if "mean_ms" in data:
                    # Single metric
                    report.append(f"  Mean:   {data['mean_ms']:.2f} ms")
                    report.append(f"  Median: {data['median_ms']:.2f} ms")
                    report.append(f"  P95:    {data['p95_ms']:.2f} ms")
                    report.append(f"  P99:    {data['p99_ms']:.2f} ms")
                    report.append(f"  Max:    {data['max_ms']:.2f} ms")
                else:
                    # Nested metrics
                    for sub_name, sub_data in data.items():
                        if isinstance(sub_data, dict) and "mean_ms" in sub_data:
                            report.append(f"  {sub_name}:")
                            report.append(f"    Mean: {sub_data['mean_ms']:.2f} ms")
                            report.append(f"    P95:  {sub_data['p95_ms']:.2f} ms")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all latency benchmarks."""
        print("Starting comprehensive latency benchmarks...\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
        }

        benchmarks = [
            ("workflow_latency", self.benchmark_workflow_latency()),
            ("agent_latency", self.benchmark_agent_latency()),
            ("api_latency", self.benchmark_api_latency()),
            ("blackboard_latency", self.benchmark_blackboard_latency()),
            ("event_latency", self.benchmark_event_latency()),
            ("cold_start_latency", self.benchmark_cold_start_latency()),
            ("latency_under_load", self.benchmark_latency_under_load()),
            ("network_latency", self.benchmark_network_latency()),
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

        # Generate report
        results["report"] = self.generate_latency_report(results["benchmarks"])

        return results


async def main():
    """Main function to run latency benchmarks."""
    benchmark = LatencyBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Save results
    with open("benchmark_latency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print report
    print(results["report"])


if __name__ == "__main__":
    asyncio.run(main())