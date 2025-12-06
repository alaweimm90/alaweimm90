"""
Resource Usage Profiling for ORCHEX

Monitors and profiles CPU, memory, disk I/O, and network usage.
"""

import pytest
import asyncio
import time
import json
import psutil
import os
import statistics
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import tracemalloc
import gc
import resource
import threading
from memory_profiler import profile
import cProfile
import pstats
import io

from atlas_core.engine import ATLASEngine


class ResourceProfiler:
    """Comprehensive resource usage profiling."""

    def __init__(self):
        self.engine = ATLASEngine(
            redis_url="redis://localhost:6379/0",
            performance_mode=True,
        )
        self.monitoring_active = False
        self.resource_samples = []
        self.process = psutil.Process()

    async def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage patterns."""
        print("Profiling memory usage...")

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        initial_memory = self.process.memory_info()

        # Memory usage during different operations
        memory_profiles = {}

        # 1. Idle memory
        gc.collect()
        idle_snapshot = tracemalloc.take_snapshot()
        memory_profiles["idle"] = {
            "rss_mb": self.process.memory_info().rss / 1024 / 1024,
            "vms_mb": self.process.memory_info().vms / 1024 / 1024,
        }

        # 2. Small workflow memory
        small_workflows = []
        for i in range(100):
            workflow = {
                "id": f"small_{i}",
                "data": "x" * 100,  # Small data
            }
            result = await self.engine.execute_workflow(workflow)
            small_workflows.append(result)

        gc.collect()
        small_snapshot = tracemalloc.take_snapshot()
        memory_profiles["small_workflows"] = {
            "rss_mb": self.process.memory_info().rss / 1024 / 1024,
            "vms_mb": self.process.memory_info().vms / 1024 / 1024,
            "top_stats": self._get_top_memory_stats(small_snapshot, idle_snapshot),
        }

        # 3. Large workflow memory
        large_workflows = []
        for i in range(10):
            workflow = {
                "id": f"large_{i}",
                "data": "x" * 1000000,  # ~1MB data
            }
            result = await self.engine.execute_workflow(workflow)
            large_workflows.append(result)

        gc.collect()
        large_snapshot = tracemalloc.take_snapshot()
        memory_profiles["large_workflows"] = {
            "rss_mb": self.process.memory_info().rss / 1024 / 1024,
            "vms_mb": self.process.memory_info().vms / 1024 / 1024,
            "top_stats": self._get_top_memory_stats(large_snapshot, small_snapshot),
        }

        # 4. Memory leak detection
        leak_test_results = await self._test_memory_leaks()
        memory_profiles["leak_test"] = leak_test_results

        # 5. Memory fragmentation analysis
        fragmentation = self._analyze_memory_fragmentation()
        memory_profiles["fragmentation"] = fragmentation

        tracemalloc.stop()

        return memory_profiles

    async def profile_cpu_usage(self) -> Dict[str, Any]:
        """Profile CPU usage patterns."""
        print("Profiling CPU usage...")

        cpu_profiles = {}

        # Set CPU affinity for consistent measurements
        try:
            os.sched_setaffinity(0, range(psutil.cpu_count()))
        except Exception:
            pass  # Not supported on all systems

        # 1. Baseline CPU usage
        baseline_samples = []
        for _ in range(10):
            baseline_samples.append(self.process.cpu_percent(interval=0.1))
        cpu_profiles["baseline"] = {
            "mean": statistics.mean(baseline_samples),
            "max": max(baseline_samples),
        }

        # 2. CPU-intensive workload
        cpu_intensive_samples = []

        async def cpu_intensive_task():
            # Simulate CPU-intensive computation
            result = 0
            for i in range(1000000):
                result += i ** 2
            return result

        monitor_task = asyncio.create_task(self._monitor_cpu(cpu_intensive_samples))

        tasks = [cpu_intensive_task() for _ in range(10)]
        await asyncio.gather(*tasks)

        monitor_task.cancel()

        cpu_profiles["cpu_intensive"] = {
            "mean": statistics.mean(cpu_intensive_samples) if cpu_intensive_samples else 0,
            "max": max(cpu_intensive_samples) if cpu_intensive_samples else 0,
            "samples": len(cpu_intensive_samples),
        }

        # 3. I/O-bound workload
        io_bound_samples = []

        async def io_bound_task():
            # Simulate I/O-bound operation
            for _ in range(100):
                await asyncio.sleep(0.001)
                data = "x" * 10000

        monitor_task = asyncio.create_task(self._monitor_cpu(io_bound_samples))

        tasks = [io_bound_task() for _ in range(10)]
        await asyncio.gather(*tasks)

        monitor_task.cancel()

        cpu_profiles["io_bound"] = {
            "mean": statistics.mean(io_bound_samples) if io_bound_samples else 0,
            "max": max(io_bound_samples) if io_bound_samples else 0,
        }

        # 4. Per-core utilization
        per_core = []
        for _ in range(10):
            per_core.append(psutil.cpu_percent(interval=0.1, percpu=True))

        cpu_profiles["per_core"] = {
            f"core_{i}": {
                "mean": statistics.mean([sample[i] for sample in per_core]),
                "max": max([sample[i] for sample in per_core]),
            }
            for i in range(len(per_core[0]))
        }

        # 5. Context switches and interrupts
        ctx_before = self.process.cpu_stats()
        await asyncio.sleep(1)
        ctx_after = self.process.cpu_stats()

        cpu_profiles["context_switches"] = {
            "voluntary": ctx_after.ctx_switches - ctx_before.ctx_switches,
            "involuntary": ctx_after.ictx_switches - ctx_before.ictx_switches,
        }

        return cpu_profiles

    async def profile_disk_io(self) -> Dict[str, Any]:
        """Profile disk I/O patterns."""
        print("Profiling disk I/O...")

        io_profiles = {}

        # Get initial disk I/O stats
        io_counters_before = psutil.disk_io_counters()
        process_io_before = self.process.io_counters()

        # 1. Sequential write test
        print("Testing sequential writes...")
        write_data = b"x" * 1024 * 1024  # 1MB
        write_times = []

        for i in range(100):
            start = time.time()
            with open(f"/tmp/test_write_{i}.dat", "wb") as f:
                f.write(write_data)
                f.flush()
                os.fsync(f.fileno())
            write_times.append(time.time() - start)

        io_profiles["sequential_write"] = {
            "avg_time_ms": statistics.mean(write_times) * 1000,
            "throughput_mb_s": 1.0 / statistics.mean(write_times),
        }

        # 2. Sequential read test
        print("Testing sequential reads...")
        read_times = []

        for i in range(100):
            start = time.time()
            with open(f"/tmp/test_write_{i}.dat", "rb") as f:
                data = f.read()
            read_times.append(time.time() - start)

        io_profiles["sequential_read"] = {
            "avg_time_ms": statistics.mean(read_times) * 1000,
            "throughput_mb_s": 1.0 / statistics.mean(read_times),
        }

        # 3. Random access test
        print("Testing random access...")
        random_times = []

        for _ in range(100):
            file_num = np.random.randint(0, 100)
            offset = np.random.randint(0, 1024 * 1024)

            start = time.time()
            with open(f"/tmp/test_write_{file_num}.dat", "rb") as f:
                f.seek(offset)
                data = f.read(1024)
            random_times.append(time.time() - start)

        io_profiles["random_access"] = {
            "avg_time_ms": statistics.mean(random_times) * 1000,
            "iops": 1.0 / statistics.mean(random_times),
        }

        # Clean up test files
        for i in range(100):
            try:
                os.remove(f"/tmp/test_write_{i}.dat")
            except Exception:
                pass

        # Get final disk I/O stats
        io_counters_after = psutil.disk_io_counters()
        process_io_after = self.process.io_counters()

        io_profiles["total_io"] = {
            "read_bytes": io_counters_after.read_bytes - io_counters_before.read_bytes,
            "write_bytes": io_counters_after.write_bytes - io_counters_before.write_bytes,
            "read_count": io_counters_after.read_count - io_counters_before.read_count,
            "write_count": io_counters_after.write_count - io_counters_before.write_count,
            "process_read_bytes": process_io_after.read_bytes - process_io_before.read_bytes,
            "process_write_bytes": process_io_after.write_bytes - process_io_before.write_bytes,
        }

        return io_profiles

    async def profile_network_usage(self) -> Dict[str, Any]:
        """Profile network usage patterns."""
        print("Profiling network usage...")

        network_profiles = {}

        # Get initial network stats
        net_before = psutil.net_io_counters()

        # 1. Small packet test
        small_packet_times = []
        for _ in range(1000):
            start = time.time()
            await self.engine.send_network_packet(size=64)  # 64 bytes
            small_packet_times.append(time.time() - start)

        network_profiles["small_packets"] = {
            "avg_latency_ms": statistics.mean(small_packet_times) * 1000,
            "packets_per_second": 1.0 / statistics.mean(small_packet_times),
        }

        # 2. Large packet test
        large_packet_times = []
        for _ in range(100):
            start = time.time()
            await self.engine.send_network_packet(size=65536)  # 64KB
            large_packet_times.append(time.time() - start)

        network_profiles["large_packets"] = {
            "avg_latency_ms": statistics.mean(large_packet_times) * 1000,
            "throughput_mb_s": (64 * 100) / (sum(large_packet_times) * 1024),
        }

        # 3. Concurrent connections test
        connection_counts = [10, 50, 100, 500]
        concurrent_results = {}

        for count in connection_counts:
            start = time.time()
            tasks = [
                self.engine.create_connection(f"conn_{i}")
                for i in range(count)
            ]
            await asyncio.gather(*tasks)
            duration = time.time() - start

            concurrent_results[count] = {
                "connections": count,
                "total_time": duration,
                "connections_per_second": count / duration,
            }

        network_profiles["concurrent_connections"] = concurrent_results

        # Get final network stats
        net_after = psutil.net_io_counters()

        network_profiles["total_network"] = {
            "bytes_sent": net_after.bytes_sent - net_before.bytes_sent,
            "bytes_received": net_after.bytes_recv - net_before.bytes_recv,
            "packets_sent": net_after.packets_sent - net_before.packets_sent,
            "packets_received": net_after.packets_recv - net_before.packets_recv,
            "errors": net_after.errin + net_after.errout,
            "dropped": net_after.dropin + net_after.dropout,
        }

        return network_profiles

    async def profile_thread_usage(self) -> Dict[str, Any]:
        """Profile thread and async task usage."""
        print("Profiling thread usage...")

        thread_profiles = {}

        # 1. Thread count analysis
        thread_counts = []
        for _ in range(10):
            thread_counts.append(threading.active_count())
            await asyncio.sleep(0.1)

        thread_profiles["thread_count"] = {
            "min": min(thread_counts),
            "max": max(thread_counts),
            "mean": statistics.mean(thread_counts),
        }

        # 2. Async task analysis
        task_counts = []
        for _ in range(10):
            tasks = asyncio.all_tasks()
            task_counts.append(len(tasks))
            await asyncio.sleep(0.1)

        thread_profiles["async_tasks"] = {
            "min": min(task_counts),
            "max": max(task_counts),
            "mean": statistics.mean(task_counts),
        }

        # 3. Thread pool executor performance
        from concurrent.futures import ThreadPoolExecutor

        def cpu_bound_work(n):
            result = 0
            for i in range(n):
                result += i ** 2
            return result

        executor_sizes = [1, 2, 4, 8, 16]
        executor_results = {}

        for size in executor_sizes:
            with ThreadPoolExecutor(max_workers=size) as executor:
                start = time.time()
                futures = [
                    executor.submit(cpu_bound_work, 100000)
                    for _ in range(20)
                ]
                results = [f.result() for f in futures]
                duration = time.time() - start

                executor_results[size] = {
                    "workers": size,
                    "duration": duration,
                    "tasks_per_second": 20 / duration,
                }

        thread_profiles["thread_pool_scaling"] = executor_results

        return thread_profiles

    async def _monitor_cpu(self, samples: List[float]):
        """Monitor CPU usage in background."""
        while True:
            samples.append(self.process.cpu_percent(interval=0.1))
            await asyncio.sleep(0.1)

    async def _test_memory_leaks(self) -> Dict[str, Any]:
        """Test for memory leaks."""
        print("Testing for memory leaks...")

        gc.collect()
        initial_objects = len(gc.get_objects())
        initial_memory = self.process.memory_info().rss

        # Run many iterations of workflow
        for i in range(1000):
            workflow = {
                "id": f"leak_test_{i}",
                "data": "x" * 1000,
            }
            result = await self.engine.execute_workflow(workflow)

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        final_objects = len(gc.get_objects())
        final_memory = self.process.memory_info().rss

        memory_growth = final_memory - initial_memory
        object_growth = final_objects - initial_objects

        return {
            "memory_growth_mb": memory_growth / 1024 / 1024,
            "object_growth": object_growth,
            "potential_leak": memory_growth > 10 * 1024 * 1024,  # >10MB growth
            "gc_stats": gc.get_stats(),
        }

    def _analyze_memory_fragmentation(self) -> Dict[str, Any]:
        """Analyze memory fragmentation."""
        # Get memory info
        memory_info = self.process.memory_info()

        # Calculate fragmentation metrics
        fragmentation = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "fragmentation_ratio": memory_info.vms / memory_info.rss if memory_info.rss > 0 else 0,
        }

        # Get malloc info on Linux
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.mallinfo.restype = ctypes.c_void_p
            # Note: mallinfo is system-specific
        except Exception:
            pass

        return fragmentation

    def _get_top_memory_stats(self, snapshot, previous_snapshot=None, limit=10):
        """Get top memory consuming locations."""
        if previous_snapshot:
            stats = snapshot.compare_to(previous_snapshot, "lineno")
        else:
            stats = snapshot.statistics("lineno")

        top_stats = []
        for stat in stats[:limit]:
            top_stats.append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count,
            })

        return top_stats

    async def generate_resource_report(self, profiles: Dict[str, Any]) -> str:
        """Generate comprehensive resource usage report."""
        report = []
        report.append("=" * 80)
        report.append("RESOURCE USAGE PROFILE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Memory section
        if "memory" in profiles:
            report.append("\n## MEMORY USAGE")
            report.append("-" * 40)
            mem = profiles["memory"]

            if "large_workflows" in mem:
                report.append(f"Large Workflows RSS: {mem['large_workflows']['rss_mb']:.2f} MB")

            if "leak_test" in mem:
                leak = mem["leak_test"]
                report.append(f"Memory Growth: {leak['memory_growth_mb']:.2f} MB")
                report.append(f"Potential Leak: {leak['potential_leak']}")

        # CPU section
        if "cpu" in profiles:
            report.append("\n## CPU USAGE")
            report.append("-" * 40)
            cpu = profiles["cpu"]

            if "cpu_intensive" in cpu:
                report.append(f"CPU Intensive Mean: {cpu['cpu_intensive']['mean']:.1f}%")
                report.append(f"CPU Intensive Max: {cpu['cpu_intensive']['max']:.1f}%")

        # Disk I/O section
        if "disk_io" in profiles:
            report.append("\n## DISK I/O")
            report.append("-" * 40)
            io = profiles["disk_io"]

            if "sequential_write" in io:
                report.append(f"Sequential Write: {io['sequential_write']['throughput_mb_s']:.2f} MB/s")
            if "sequential_read" in io:
                report.append(f"Sequential Read: {io['sequential_read']['throughput_mb_s']:.2f} MB/s")

        # Network section
        if "network" in profiles:
            report.append("\n## NETWORK")
            report.append("-" * 40)
            net = profiles["network"]

            if "total_network" in net:
                total = net["total_network"]
                report.append(f"Bytes Sent: {total['bytes_sent'] / 1024 / 1024:.2f} MB")
                report.append(f"Bytes Received: {total['bytes_received'] / 1024 / 1024:.2f} MB")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    async def run_all_profiles(self) -> Dict[str, Any]:
        """Run all resource profiling."""
        print("Starting comprehensive resource profiling...\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage": psutil.disk_usage("/").percent,
            },
            "profiles": {},
        }

        profiles = [
            ("memory", self.profile_memory_usage()),
            ("cpu", self.profile_cpu_usage()),
            ("disk_io", self.profile_disk_io()),
            ("network", self.profile_network_usage()),
            ("threads", self.profile_thread_usage()),
        ]

        for name, profile_coro in profiles:
            print(f"\n{'='*60}")
            print(f"Running {name} profile...")
            print('='*60)
            try:
                results["profiles"][name] = await profile_coro
                print(f"✓ {name} profile completed")
            except Exception as e:
                print(f"✗ {name} profile failed: {e}")
                results["profiles"][name] = {"error": str(e)}

        # Generate report
        results["report"] = await self.generate_resource_report(results["profiles"])

        return results


async def main():
    """Main function to run resource profiling."""
    profiler = ResourceProfiler()
    results = await profiler.run_all_profiles()

    # Save results
    with open("benchmark_resource_usage_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print report
    print(results["report"])


if __name__ == "__main__":
    asyncio.run(main())