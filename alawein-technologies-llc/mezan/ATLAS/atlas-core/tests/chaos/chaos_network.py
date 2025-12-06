"""
Network Chaos Engineering Tests for ORCHEX

Tests system resilience under various network failure conditions.
"""

import pytest
import asyncio
import time
import random
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime
import netem  # Network emulation
import iptables  # Firewall rules
import tc  # Traffic control

from atlas_core.engine import ATLASEngine
from atlas_core.distributed import DistributedCoordinator


class NetworkChaos:
    """Network chaos engineering test suite."""

    def __init__(self):
        self.engine = ATLASEngine(
            redis_url="redis://localhost:6379/0",
            chaos_mode=True,
        )
        self.active_chaos = []

    async def inject_latency(
        self,
        target: str,
        latency_ms: int,
        variation_ms: int = 0,
        correlation: float = 0,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject network latency."""
        print(f"Injecting {latency_ms}ms latency to {target}")

        chaos_id = f"latency_{time.time()}"

        # Apply latency using tc (traffic control)
        tc_rule = {
            "action": "add",
            "dev": target,
            "latency": f"{latency_ms}ms",
            "variation": f"{variation_ms}ms",
            "correlation": f"{correlation}%",
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("latency_test")

        # Auto-remove after duration
        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "latency",
            "target": target,
            "latency_ms": latency_ms,
            "variation_ms": variation_ms,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_packet_loss(
        self,
        target: str,
        loss_percentage: float,
        correlation: float = 0,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject packet loss."""
        print(f"Injecting {loss_percentage}% packet loss to {target}")

        chaos_id = f"packet_loss_{time.time()}"

        # Apply packet loss
        tc_rule = {
            "action": "add",
            "dev": target,
            "loss": f"{loss_percentage}%",
            "correlation": f"{correlation}%",
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("packet_loss_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "packet_loss",
            "target": target,
            "loss_percentage": loss_percentage,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_bandwidth_limit(
        self,
        target: str,
        bandwidth_kbps: int,
        burst_kb: int = 32,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Limit network bandwidth."""
        print(f"Limiting bandwidth to {bandwidth_kbps}kbps on {target}")

        chaos_id = f"bandwidth_{time.time()}"

        # Apply bandwidth limit
        tc_rule = {
            "action": "add",
            "dev": target,
            "rate": f"{bandwidth_kbps}kbit",
            "burst": f"{burst_kb}k",
            "latency": "50ms",  # Buffer size
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("bandwidth_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "bandwidth_limit",
            "target": target,
            "bandwidth_kbps": bandwidth_kbps,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_network_partition(
        self,
        partition_groups: List[List[str]],
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create network partitions between node groups."""
        print(f"Creating network partition: {partition_groups}")

        chaos_id = f"partition_{time.time()}"

        # Apply iptables rules to block communication between groups
        for i, group1 in enumerate(partition_groups):
            for j, group2 in enumerate(partition_groups):
                if i != j:
                    for node1 in group1:
                        for node2 in group2:
                            await self._block_communication(node1, node2)

        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_distributed_workload("partition_test")

        if duration:
            await asyncio.sleep(duration)
            await self._heal_partition(partition_groups)

        return {
            "chaos_id": chaos_id,
            "type": "network_partition",
            "partition_groups": partition_groups,
            "results": results,
            "impact": self._analyze_partition_impact(results),
        }

    async def inject_packet_corruption(
        self,
        target: str,
        corruption_percentage: float,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject packet corruption."""
        print(f"Injecting {corruption_percentage}% packet corruption on {target}")

        chaos_id = f"corruption_{time.time()}"

        # Apply packet corruption
        tc_rule = {
            "action": "add",
            "dev": target,
            "corrupt": f"{corruption_percentage}%",
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("corruption_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "packet_corruption",
            "target": target,
            "corruption_percentage": corruption_percentage,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_packet_duplication(
        self,
        target: str,
        duplication_percentage: float,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject packet duplication."""
        print(f"Injecting {duplication_percentage}% packet duplication on {target}")

        chaos_id = f"duplication_{time.time()}"

        # Apply packet duplication
        tc_rule = {
            "action": "add",
            "dev": target,
            "duplicate": f"{duplication_percentage}%",
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("duplication_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "packet_duplication",
            "target": target,
            "duplication_percentage": duplication_percentage,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_packet_reordering(
        self,
        target: str,
        reorder_percentage: float,
        gap: int = 5,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject packet reordering."""
        print(f"Injecting {reorder_percentage}% packet reordering on {target}")

        chaos_id = f"reordering_{time.time()}"

        # Apply packet reordering
        tc_rule = {
            "action": "add",
            "dev": target,
            "reorder": f"{reorder_percentage}%",
            "gap": str(gap),
        }

        await self._apply_tc_rule(tc_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("reordering_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_tc_rule(target)

        return {
            "chaos_id": chaos_id,
            "type": "packet_reordering",
            "target": target,
            "reorder_percentage": reorder_percentage,
            "gap": gap,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_dns_failure(
        self,
        domains: List[str],
        failure_type: str = "timeout",
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject DNS resolution failures."""
        print(f"Injecting DNS {failure_type} for {domains}")

        chaos_id = f"dns_{time.time()}"

        # Modify /etc/hosts or use iptables to block DNS
        for domain in domains:
            if failure_type == "timeout":
                await self._block_dns_resolution(domain)
            elif failure_type == "wrong_ip":
                await self._redirect_dns(domain, "127.0.0.1")
            elif failure_type == "nxdomain":
                await self._return_nxdomain(domain)

        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("dns_test")

        if duration:
            await asyncio.sleep(duration)
            await self._restore_dns(domains)

        return {
            "chaos_id": chaos_id,
            "type": "dns_failure",
            "domains": domains,
            "failure_type": failure_type,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def inject_connection_reset(
        self,
        target_port: int,
        reset_probability: float,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inject TCP connection resets."""
        print(f"Injecting connection resets on port {target_port}")

        chaos_id = f"reset_{time.time()}"

        # Use iptables to probabilistically reset connections
        iptables_rule = f"""
        iptables -A INPUT -p tcp --dport {target_port} \
            -m statistic --mode random --probability {reset_probability} \
            -j REJECT --reject-with tcp-reset
        """

        await self._execute_command(iptables_rule)
        self.active_chaos.append(chaos_id)

        # Run test workload
        results = await self._run_test_workload("connection_reset_test")

        if duration:
            await asyncio.sleep(duration)
            await self._remove_iptables_rule(target_port)

        return {
            "chaos_id": chaos_id,
            "type": "connection_reset",
            "target_port": target_port,
            "reset_probability": reset_probability,
            "results": results,
            "impact": self._analyze_impact(results),
        }

    async def simulate_network_flapping(
        self,
        target: str,
        interval_seconds: int = 10,
        duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """Simulate network interface flapping."""
        print(f"Simulating network flapping on {target}")

        chaos_id = f"flapping_{time.time()}"
        self.active_chaos.append(chaos_id)

        results = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Bring interface down
            await self._execute_command(f"ip link set {target} down")
            down_time = time.time()

            # Wait half interval
            await asyncio.sleep(interval_seconds / 2)

            # Bring interface up
            await self._execute_command(f"ip link set {target} up")
            up_time = time.time()

            # Test during up period
            test_result = await self._run_quick_test()
            results.append({
                "down_time": down_time,
                "up_time": up_time,
                "down_duration": up_time - down_time,
                "test_result": test_result,
            })

            # Wait remaining interval
            await asyncio.sleep(interval_seconds / 2)

        return {
            "chaos_id": chaos_id,
            "type": "network_flapping",
            "target": target,
            "interval_seconds": interval_seconds,
            "duration_seconds": duration_seconds,
            "flap_count": len(results),
            "results": results,
            "impact": self._analyze_flapping_impact(results),
        }

    async def test_cascading_network_failures(self) -> Dict[str, Any]:
        """Test cascading network failures."""
        print("Testing cascading network failures")

        chaos_sequence = [
            {"type": "latency", "target": "eth0", "value": 100},
            {"type": "packet_loss", "target": "eth0", "value": 5},
            {"type": "latency", "target": "eth1", "value": 200},
            {"type": "packet_loss", "target": "eth1", "value": 10},
            {"type": "bandwidth", "target": "eth0", "value": 1000},
        ]

        results = []

        for i, chaos in enumerate(chaos_sequence):
            print(f"Stage {i+1}: Applying {chaos['type']} to {chaos['target']}")

            # Apply chaos
            if chaos["type"] == "latency":
                result = await self.inject_latency(
                    chaos["target"],
                    chaos["value"],
                    duration=30
                )
            elif chaos["type"] == "packet_loss":
                result = await self.inject_packet_loss(
                    chaos["target"],
                    chaos["value"],
                    duration=30
                )
            elif chaos["type"] == "bandwidth":
                result = await self.inject_bandwidth_limit(
                    chaos["target"],
                    chaos["value"],
                    duration=30
                )

            results.append(result)

            # Test system behavior
            system_test = await self._run_system_test()
            result["system_health"] = system_test

            # Wait before next failure
            await asyncio.sleep(5)

        return {
            "type": "cascading_failures",
            "sequence": chaos_sequence,
            "results": results,
            "total_impact": self._analyze_cascading_impact(results),
        }

    async def _apply_tc_rule(self, rule: Dict[str, Any]) -> None:
        """Apply traffic control rule."""
        cmd = f"tc qdisc {rule['action']} dev {rule['dev']} root netem"
        for key, value in rule.items():
            if key not in ["action", "dev"]:
                cmd += f" {key} {value}"
        await self._execute_command(cmd)

    async def _remove_tc_rule(self, device: str) -> None:
        """Remove traffic control rule."""
        await self._execute_command(f"tc qdisc del dev {device} root")

    async def _block_communication(self, node1: str, node2: str) -> None:
        """Block communication between two nodes."""
        cmd = f"iptables -A INPUT -s {node1} -d {node2} -j DROP"
        await self._execute_command(cmd)

    async def _execute_command(self, command: str) -> None:
        """Execute system command."""
        # In production, use subprocess or similar
        pass

    async def _run_test_workload(self, test_name: str) -> Dict[str, Any]:
        """Run test workload and measure impact."""
        results = {
            "test_name": test_name,
            "start_time": time.time(),
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
        }

        for i in range(100):
            start = time.time()
            try:
                await self.engine.execute_workflow({
                    "id": f"{test_name}_{i}",
                    "tasks": ["test_task"],
                })
                results["successful_requests"] += 1
                results["latencies"].append(time.time() - start)
            except Exception:
                results["failed_requests"] += 1

        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        results["success_rate"] = results["successful_requests"] / 100
        results["avg_latency"] = statistics.mean(results["latencies"]) if results["latencies"] else 0

        return results

    async def _run_distributed_workload(self, test_name: str) -> Dict[str, Any]:
        """Run distributed workload for partition testing."""
        coordinator = DistributedCoordinator()
        results = await coordinator.execute_distributed_test(test_name)
        return results

    async def _run_quick_test(self) -> Dict[str, Any]:
        """Run quick connectivity test."""
        try:
            result = await self.engine.health_check()
            return {"status": "success", "latency": result.get("latency", 0)}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _run_system_test(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        return await self.engine.run_system_diagnostics()

    def _analyze_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of chaos injection."""
        baseline_latency = 0.1  # 100ms baseline
        baseline_success_rate = 0.99

        return {
            "latency_increase": (results.get("avg_latency", 0) - baseline_latency) / baseline_latency,
            "success_rate_decrease": baseline_success_rate - results.get("success_rate", 0),
            "severity": self._calculate_severity(results),
        }

    def _analyze_partition_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of network partition."""
        return {
            "consensus_impact": results.get("consensus_failures", 0),
            "data_inconsistency": results.get("inconsistent_nodes", 0),
            "recovery_time": results.get("recovery_time", 0),
        }

    def _analyze_flapping_impact(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze impact of network flapping."""
        failures = sum(1 for r in results if r["test_result"]["status"] == "failed")
        return {
            "failure_rate": failures / len(results) if results else 0,
            "total_downtime": sum(r["down_duration"] for r in results),
            "recovery_success": all(r["test_result"]["status"] == "success"
                                   for r in results[-3:]) if len(results) >= 3 else False,
        }

    def _analyze_cascading_impact(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze impact of cascading failures."""
        return {
            "stages_impacted": len(results),
            "total_failures": sum(r.get("results", {}).get("failed_requests", 0) for r in results),
            "degradation_pattern": [r.get("system_health", {}).get("health_score", 0) for r in results],
        }

    def _calculate_severity(self, results: Dict[str, Any]) -> str:
        """Calculate severity level of chaos impact."""
        success_rate = results.get("success_rate", 0)
        if success_rate >= 0.95:
            return "low"
        elif success_rate >= 0.80:
            return "medium"
        elif success_rate >= 0.50:
            return "high"
        else:
            return "critical"

    async def cleanup_all_chaos(self) -> None:
        """Remove all active chaos injections."""
        print("Cleaning up all active chaos...")
        for chaos_id in self.active_chaos:
            # Remove all tc rules, iptables rules, etc.
            pass
        self.active_chaos.clear()


async def main():
    """Main function to run network chaos tests."""
    chaos = NetworkChaos()

    try:
        # Run various chaos tests
        results = []

        # Test 1: Latency injection
        result = await chaos.inject_latency("eth0", 200, 50, duration=30)
        results.append(result)

        # Test 2: Packet loss
        result = await chaos.inject_packet_loss("eth0", 10, duration=30)
        results.append(result)

        # Test 3: Network partition
        result = await chaos.inject_network_partition(
            [["node_0", "node_1"], ["node_2", "node_3"]],
            duration=60
        )
        results.append(result)

        # Test 4: Cascading failures
        result = await chaos.test_cascading_network_failures()
        results.append(result)

        print("\n" + "="*60)
        print("NETWORK CHAOS TEST RESULTS")
        print("="*60)
        for result in results:
            print(f"\nTest: {result.get('type', 'unknown')}")
            print(f"Impact: {result.get('impact', {})}")

    finally:
        await chaos.cleanup_all_chaos()


if __name__ == "__main__":
    asyncio.run(main())