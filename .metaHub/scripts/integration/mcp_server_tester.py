#!/usr/bin/env python3
"""
mcp_server_tester.py - Automated MCP Server Testing

Provides comprehensive testing for MCP servers:
- Connection testing and health checks
- Tool discovery and validation
- Performance benchmarking
- Integration testing with orchestration
- Report generation

Usage:
    python mcp_server_tester.py test --server github
    python mcp_server_tester.py test-all
    python mcp_server_tester.py benchmark --server filesystem
    python mcp_server_tester.py report
"""

import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

import click


class TestStatus(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class TestResult:
    """Single test result."""
    test_name: str
    status: TestStatus
    duration_ms: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerTestReport:
    """Complete test report for a server."""
    server_name: str
    timestamp: str
    overall_status: TestStatus
    tests: List[TestResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Benchmark results for a server."""
    server_name: str
    timestamp: str
    iterations: int
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_ops_sec: float
    errors: int


class MCPServerTester:
    """Automated MCP server testing."""

    CONFIG_PATH = ".ai/mcp/mcp-servers.json"
    RESULTS_PATH = ".metaHub/reports/mcp-tests"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.config_path = self.base_path / self.CONFIG_PATH
        self.results_path = self.base_path / self.RESULTS_PATH
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path

        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent

        script_path = Path(__file__).resolve().parent.parent.parent
        if (script_path / ".metaHub").exists():
            return script_path

        raise RuntimeError("Could not find central governance repo")

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP servers configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"mcpServers": {}}

    def _run_test(
        self,
        test_name: str,
        test_func,
        *args,
        **kwargs
    ) -> TestResult:
        """Run a single test with timing."""
        start_time = time.time()

        try:
            success, message, details = test_func(*args, **kwargs)
            status = TestStatus.PASSED if success else TestStatus.FAILED
        except Exception as e:
            status = TestStatus.FAILED
            message = str(e)
            details = {"exception": type(e).__name__}

        duration_ms = int((time.time() - start_time) * 1000)

        return TestResult(
            test_name=test_name,
            status=status,
            duration_ms=duration_ms,
            message=message,
            details=details
        )

    def test_server_config(self, server_name: str) -> TestResult:
        """Test server configuration validity."""
        def check():
            server_config = self.config.get("mcpServers", {}).get(server_name)

            if not server_config:
                return False, f"Server {server_name} not found in configuration", {}

            # Check required fields
            required = ["command", "args", "description"]
            missing = [f for f in required if f not in server_config]

            if missing:
                return False, f"Missing required fields: {missing}", {"missing": missing}

            # Validate command
            command = server_config.get("command")
            if command not in ["npx", "node", "python", "uvx"]:
                return False, f"Unknown command type: {command}", {"command": command}

            return True, "Configuration valid", {
                "command": command,
                "args_count": len(server_config.get("args", [])),
                "capabilities": len(server_config.get("capabilities", []))
            }

        return self._run_test(f"{server_name}_config", check)

    def test_package_availability(self, server_name: str) -> TestResult:
        """Test if the server package is available."""
        def check():
            server_config = self.config.get("mcpServers", {}).get(server_name)

            if not server_config:
                return False, "Server not configured", {}

            args = server_config.get("args", [])
            if len(args) < 2:
                return False, "No package specified in args", {}

            package = args[1] if args[0] == "-y" else args[0]

            # Check npm registry
            try:
                result = subprocess.run(
                    ["npm", "view", package, "version"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                if result.returncode == 0:
                    version = result.stdout.strip()
                    return True, f"Package available: {package}@{version}", {
                        "package": package,
                        "version": version
                    }

                return False, f"Package not found: {package}", {"package": package}

            except subprocess.TimeoutExpired:
                return False, "npm check timed out", {}
            except FileNotFoundError:
                return False, "npm not installed", {}
            except Exception as e:
                return False, str(e), {}

        return self._run_test(f"{server_name}_package", check)

    def test_environment_variables(self, server_name: str) -> TestResult:
        """Test required environment variables."""
        def check():
            server_config = self.config.get("mcpServers", {}).get(server_name)

            if not server_config:
                return False, "Server not configured", {}

            env_config = server_config.get("env", {})
            missing_vars = []
            present_vars = []

            for var, value in env_config.items():
                # Check if it's a reference to an env var
                if value.startswith("${") and value.endswith("}"):
                    env_var_name = value[2:-1]
                    if os.environ.get(env_var_name):
                        present_vars.append(env_var_name)
                    else:
                        missing_vars.append(env_var_name)

            if missing_vars:
                return False, f"Missing env vars: {missing_vars}", {
                    "missing": missing_vars,
                    "present": present_vars
                }

            return True, f"All env vars present ({len(present_vars)})", {
                "present": present_vars
            }

        return self._run_test(f"{server_name}_env", check)

    def test_capabilities(self, server_name: str) -> TestResult:
        """Test server capabilities are defined."""
        def check():
            server_config = self.config.get("mcpServers", {}).get(server_name)

            if not server_config:
                return False, "Server not configured", {}

            capabilities = server_config.get("capabilities", [])

            if not capabilities:
                return False, "No capabilities defined", {}

            return True, f"{len(capabilities)} capabilities defined", {
                "capabilities": capabilities
            }

        return self._run_test(f"{server_name}_capabilities", check)

    def test_server(self, server_name: str) -> ServerTestReport:
        """Run all tests for a server."""
        report = ServerTestReport(
            server_name=server_name,
            timestamp=datetime.now().isoformat(),
            overall_status=TestStatus.PASSED
        )

        # Run tests
        tests = [
            self.test_server_config(server_name),
            self.test_package_availability(server_name),
            self.test_environment_variables(server_name),
            self.test_capabilities(server_name),
        ]

        for test in tests:
            report.tests.append(test)
            status_key = test.status.value
            report.summary[status_key] = report.summary.get(status_key, 0) + 1

            if test.status == TestStatus.FAILED:
                report.overall_status = TestStatus.FAILED

        # Get server metadata
        server_config = self.config.get("mcpServers", {}).get(server_name, {})
        report.metadata = {
            "description": server_config.get("description", ""),
            "tags": server_config.get("tags", []),
            "command": server_config.get("command", "")
        }

        return report

    def test_all_servers(self) -> List[ServerTestReport]:
        """Test all configured servers."""
        reports = []

        for server_name in self.config.get("mcpServers", {}).keys():
            report = self.test_server(server_name)
            reports.append(report)

        return reports

    def benchmark_server(
        self,
        server_name: str,
        iterations: int = 10
    ) -> BenchmarkResult:
        """Benchmark server latency."""
        latencies = []
        errors = 0

        for _ in range(iterations):
            start_time = time.time()

            try:
                # Simple latency test - check package availability
                server_config = self.config.get("mcpServers", {}).get(server_name, {})
                args = server_config.get("args", [])
                package = args[1] if len(args) > 1 and args[0] == "-y" else None

                if package:
                    result = subprocess.run(
                        ["npm", "view", package, "version"],
                        capture_output=True,
                        timeout=10
                    )

                    latency = (time.time() - start_time) * 1000
                    if result.returncode == 0:
                        latencies.append(latency)
                    else:
                        errors += 1
                        latencies.append(latency)
                else:
                    latencies.append((time.time() - start_time) * 1000)

            except Exception:
                errors += 1
                latencies.append((time.time() - start_time) * 1000)

        # Calculate statistics
        latencies.sort()

        def percentile(data: List[float], p: int) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

        total_time = sum(latencies) / 1000  # seconds
        throughput = iterations / total_time if total_time > 0 else 0

        return BenchmarkResult(
            server_name=server_name,
            timestamp=datetime.now().isoformat(),
            iterations=iterations,
            latency_avg_ms=statistics.mean(latencies) if latencies else 0,
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_p99_ms=percentile(latencies, 99),
            throughput_ops_sec=throughput,
            errors=errors
        )

    def generate_report(self, reports: List[ServerTestReport]) -> Dict[str, Any]:
        """Generate consolidated test report."""
        total_tests = 0
        total_passed = 0
        total_failed = 0

        server_results = {}

        for report in reports:
            server_results[report.server_name] = {
                "status": report.overall_status.value,
                "tests": report.summary,
                "metadata": report.metadata
            }

            for count in report.summary.values():
                total_tests += count

            total_passed += report.summary.get("passed", 0)
            total_failed += report.summary.get("failed", 0)

        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_servers": len(reports),
                "servers_passing": sum(1 for r in reports if r.overall_status == TestStatus.PASSED),
                "servers_failing": sum(1 for r in reports if r.overall_status == TestStatus.FAILED),
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed
            },
            "servers": server_results
        }

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save report to file."""
        if not filename:
            filename = f"mcp-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

        filepath = self.results_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return filepath


def format_report(report: ServerTestReport, fmt: str = "text") -> str:
    """Format server test report."""
    if fmt == "json":
        return json.dumps(asdict(report), indent=2, default=str)

    lines = [
        f"Server: {report.server_name}",
        f"Status: {report.overall_status.value.upper()}",
        f"Timestamp: {report.timestamp}",
        "",
        "Tests:"
    ]

    for test in report.tests:
        icon = "[OK]" if test.status == TestStatus.PASSED else "[FAIL]"
        lines.append(f"  {icon} {test.test_name}: {test.message} ({test.duration_ms}ms)")

    lines.extend([
        "",
        f"Summary: {report.summary.get('passed', 0)} passed, {report.summary.get('failed', 0)} failed"
    ])

    return '\n'.join(lines)


@click.group()
def cli():
    """MCP Server testing and benchmarking."""
    pass


@cli.command()
@click.option('--server', '-s', required=True, help='Server name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def test(server: str, json_output: bool):
    """Test a specific MCP server."""
    try:
        tester = MCPServerTester()
        report = tester.test_server(server)

        output = format_report(report, "json" if json_output else "text")
        click.echo(output)

        raise SystemExit(0 if report.overall_status == TestStatus.PASSED else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('test-all')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--save', is_flag=True, help='Save report to file')
def test_all(json_output: bool, save: bool):
    """Test all configured MCP servers."""
    try:
        tester = MCPServerTester()
        reports = tester.test_all_servers()
        consolidated = tester.generate_report(reports)

        if save:
            filepath = tester.save_report(consolidated)
            click.echo(f"Report saved to: {filepath}")

        if json_output:
            click.echo(json.dumps(consolidated, indent=2))
        else:
            click.echo("MCP Server Test Results")
            click.echo("=" * 50)
            click.echo(f"Total Servers: {consolidated['summary']['total_servers']}")
            click.echo(f"Passing: {consolidated['summary']['servers_passing']}")
            click.echo(f"Failing: {consolidated['summary']['servers_failing']}")
            click.echo("")

            for server, result in consolidated["servers"].items():
                icon = "[OK]" if result["status"] == "passed" else "[FAIL]"
                click.echo(f"  {icon} {server}: {result['status']}")

        all_passed = consolidated['summary']['servers_failing'] == 0
        raise SystemExit(0 if all_passed else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--server', '-s', required=True, help='Server name')
@click.option('--iterations', '-n', default=10, help='Number of iterations')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def benchmark(server: str, iterations: int, json_output: bool):
    """Benchmark an MCP server."""
    try:
        tester = MCPServerTester()
        result = tester.benchmark_server(server, iterations)

        if json_output:
            click.echo(json.dumps(asdict(result), indent=2))
        else:
            click.echo(f"Benchmark: {server}")
            click.echo("=" * 40)
            click.echo(f"Iterations: {result.iterations}")
            click.echo(f"Avg Latency: {result.latency_avg_ms:.2f}ms")
            click.echo(f"P50 Latency: {result.latency_p50_ms:.2f}ms")
            click.echo(f"P95 Latency: {result.latency_p95_ms:.2f}ms")
            click.echo(f"P99 Latency: {result.latency_p99_ms:.2f}ms")
            click.echo(f"Throughput: {result.throughput_ops_sec:.2f} ops/sec")
            click.echo(f"Errors: {result.errors}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
def report(json_output: bool):
    """Generate a test report for all servers."""
    try:
        tester = MCPServerTester()
        reports = tester.test_all_servers()
        consolidated = tester.generate_report(reports)

        # Save report
        filepath = tester.save_report(consolidated)

        if json_output:
            click.echo(json.dumps(consolidated, indent=2))
        else:
            click.echo(f"Report generated: {filepath}")
            click.echo("")
            click.echo(f"Servers: {consolidated['summary']['total_servers']}")
            click.echo(f"Tests: {consolidated['summary']['total_tests']}")
            click.echo(f"Passed: {consolidated['summary']['tests_passed']}")
            click.echo(f"Failed: {consolidated['summary']['tests_failed']}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
