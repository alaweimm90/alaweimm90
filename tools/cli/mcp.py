#!/usr/bin/env python3
"""
Unified MCP CLI - MCP server operations and testing

Consolidates:
- mcp_cli_wrapper.py → Core MCP operations
- agent_mcp_integrator.py → Agent integration
- mcp_server_tester.py → Server testing

Usage:
    python tools/cli/mcp.py list
    python tools/cli/mcp.py ping <server>
    python tools/cli/mcp.py test <server>
    python tools/cli/mcp.py integrate --workflow <name>
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import shared libraries
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.validation import Validator
from lib.telemetry import Telemetry, EventType, EventStatus

import yaml


# ============================================================================
# Data Classes and Enums
# ============================================================================

class ServerStatus(Enum):
    """MCP Server status."""
    AVAILABLE = "available"
    CONNECTED = "connected"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


class TestStatus(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""
    name: str
    command: str
    args: List[str]
    description: str
    capabilities: List[str]
    tags: List[str]
    status: ServerStatus = ServerStatus.AVAILABLE


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


@dataclass
class AgentMCPMapping:
    """Mapping between agent and MCP server capabilities."""
    agent_name: str
    agent_role: str
    mcp_servers: List[str]
    capabilities: List[str]
    use_cases: List[str]


# ============================================================================
# Core MCP Operations
# ============================================================================

class MCPManager:
    """Core MCP server management."""
    
    CONFIG_PATH = ".ai/mcp/mcp-servers.json"
    REGISTRY_PATH = ".ai/mcp/server-registry.yaml"
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.config_path = self.base_path / self.CONFIG_PATH
        self.registry_path = self.base_path / self.REGISTRY_PATH
        
        self.config = self._load_config()
        self.registry = self._load_registry()
        self.mcp_cli_available = self._check_mcp_cli()
        
        # Initialize telemetry
        try:
            self.telemetry = Telemetry(self.base_path)
        except Exception:
            self.telemetry = None
    
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
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load server registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _check_mcp_cli(self) -> bool:
        """Check if mcp-cli is installed."""
        return bool(shutil.which("mcp-cli") or shutil.which("uvx") or shutil.which("npx"))
    
    def _run_npx_server(self, server_name: str, timeout: int = 10) -> Tuple[bool, str]:
        """Test if an NPX-based MCP server can start."""
        server_config = self.config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return False, f"Server {server_name} not found in configuration"
        
        command = server_config.get("command", "npx")
        server_args = server_config.get("args", [])
        
        if command == "npx":
            package = server_args[1] if len(server_args) > 1 else None
            if package:
                try:
                    result = subprocess.run(
                        ["npm", "view", package, "version"],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    if result.returncode == 0:
                        return True, f"Package {package} available (v{result.stdout.strip()})"
                    return False, f"Package {package} not found"
                except Exception as e:
                    return False, str(e)
        
        return False, "Unknown server type"
    
    def list_servers(self) -> List[MCPServerInfo]:
        """List all configured MCP servers."""
        servers = []
        
        for name, config in self.config.get("mcpServers", {}).items():
            server = MCPServerInfo(
                name=name,
                command=config.get("command", ""),
                args=config.get("args", []),
                description=config.get("description", ""),
                capabilities=config.get("capabilities", []),
                tags=config.get("tags", [])
            )
            servers.append(server)
        
        return servers
    
    def get_server_groups(self) -> Dict[str, List[str]]:
        """Get server groups from configuration."""
        return self.config.get("serverGroups", {})
    
    def ping_server(self, server_name: str) -> Dict[str, Any]:
        """Ping/test an MCP server."""
        start_time = datetime.now()
        
        result = {
            "server": server_name,
            "timestamp": start_time.isoformat(),
            "status": "unknown",
            "latency_ms": 0,
            "details": {}
        }
        
        if server_name not in self.config.get("mcpServers", {}):
            result["status"] = "not_configured"
            result["error"] = f"Server {server_name} not found in mcp-servers.json"
            return result
        
        success, message = self._run_npx_server(server_name)
        
        end_time = datetime.now()
        result["latency_ms"] = int((end_time - start_time).total_seconds() * 1000)
        
        if success:
            result["status"] = "available"
            result["details"]["message"] = message
        else:
            result["status"] = "error"
            result["error"] = message
        
        # Record telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type=EventType.TOOL_INVOCATION.value,
                status=EventStatus.SUCCESS.value if result["status"] in ["available", "connected"] else EventStatus.FAILURE.value,
                metadata={
                    "tool": f"mcp_ping_{server_name}",
                    "server": server_name,
                    "status": result["status"],
                    "duration_ms": result["latency_ms"]
                }
            )
        
        return result
    
    def list_tools(self, server_name: str) -> Dict[str, Any]:
        """List available tools for an MCP server."""
        result = {
            "server": server_name,
            "tools": [],
            "success": False
        }
        
        server_config = self.config.get("mcpServers", {}).get(server_name, {})
        capabilities = server_config.get("capabilities", [])
        
        for cap in capabilities:
            result["tools"].append({
                "name": cap,
                "description": f"Capability: {cap}",
                "type": "capability"
            })
        
        result["success"] = True
        result["note"] = "Tools listed from configuration capabilities"
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """Run health check on all servers."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "mcp_cli_available": self.mcp_cli_available,
            "servers": {},
            "summary": {
                "total": 0,
                "available": 0,
                "error": 0
            }
        }
        
        servers = self.list_servers()
        results["summary"]["total"] = len(servers)
        
        for server in servers:
            ping_result = self.ping_server(server.name)
            results["servers"][server.name] = ping_result
            
            if ping_result["status"] in ["available", "connected"]:
                results["summary"]["available"] += 1
            else:
                results["summary"]["error"] += 1
        
        return results


# ============================================================================
# MCP Server Testing
# ============================================================================

class MCPTester:
    """MCP server testing functionality."""
    
    def __init__(self, manager: MCPManager):
        self.manager = manager
        self.results_path = manager.base_path / ".metaHub/reports/mcp-tests"
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def _run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
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
            server_config = self.manager.config.get("mcpServers", {}).get(server_name)
            
            if not server_config:
                return False, f"Server {server_name} not found in configuration", {}
            
            required = ["command", "args", "description"]
            missing = [f for f in required if f not in server_config]
            
            if missing:
                return False, f"Missing required fields: {missing}", {"missing": missing}
            
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
            server_config = self.manager.config.get("mcpServers", {}).get(server_name)
            
            if not server_config:
                return False, "Server not configured", {}
            
            args = server_config.get("args", [])
            if len(args) < 2:
                return False, "No package specified in args", {}
            
            package = args[1] if args[0] == "-y" else args[0]
            
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
    
    def test_server(self, server_name: str, suite: str = "smoke") -> ServerTestReport:
        """Run tests for a server."""
        report = ServerTestReport(
            server_name=server_name,
            timestamp=datetime.now().isoformat(),
            overall_status=TestStatus.PASSED
        )
        
        # Run tests based on suite
        tests = [
            self.test_server_config(server_name),
            self.test_package_availability(server_name),
        ]
        
        for test in tests:
            report.tests.append(test)
            status_key = test.status.value
            report.summary[status_key] = report.summary.get(status_key, 0) + 1
            
            if test.status == TestStatus.FAILED:
                report.overall_status = TestStatus.FAILED
        
        return report


# ============================================================================
# Agent-MCP Integration
# ============================================================================

class AgentIntegrator:
    """Agent-MCP integration functionality."""
    
    def __init__(self, manager: MCPManager):
        self.manager = manager
    
    def get_meathead_physicist_mappings(self) -> List[AgentMCPMapping]:
        """Define MeatheadPhysicist agent → MCP mappings."""
        return [
            AgentMCPMapping(
                agent_name="ScientistAgent",
                agent_role="Experimental design and execution",
                mcp_servers=["sequential_thinking", "context", "filesystem"],
                capabilities=["Structured experimental design", "Step-by-step problem decomposition"],
                use_cases=["Design multi-step experiments with error prevention"]
            ),
            AgentMCPMapping(
                agent_name="LiteratureAgent",
                agent_role="Academic literature search and analysis",
                mcp_servers=["brave_search", "context", "git", "filesystem"],
                capabilities=["Web search for academic papers", "Version-controlled literature tracking"],
                use_cases=["Search arXiv, Google Scholar via Brave Search"]
            ),
        ]
    
    def get_turingo_mappings(self) -> List[AgentMCPMapping]:
        """Define Turingo agent → MCP mappings."""
        return [
            AgentMCPMapping(
                agent_name="CodeCowboy",
                agent_role="High-performance code implementation",
                mcp_servers=["github", "git", "filesystem", "playwright"],
                capabilities=["Version-controlled implementation", "Automated testing integration"],
                use_cases=["Implement optimized algorithms with Git tracking"]
            ),
        ]
    
    def get_atlas_mappings(self) -> List[AgentMCPMapping]:
        """Define ATLAS agent → MCP mappings."""
        return [
            AgentMCPMapping(
                agent_name="Workflow_Orchestrator",
                agent_role="DevOps pipeline orchestration",
                mcp_servers=["terraform", "kubernetes", "sequential_thinking", "git", "prometheus"],
                capabilities=["Infrastructure automation", "Container orchestration"],
                use_cases=["Orchestrate Terraform deployments", "Manage Kubernetes clusters"]
            ),
        ]
    
    def generate_integration_report(self, workflow: Optional[str] = None) -> Dict:
        """Generate complete agent-MCP integration report."""
        report = {
            "meathead_physicist": {
                "framework": "MeatheadPhysicist",
                "total_agents": 2,
                "mappings": [asdict(m) for m in self.get_meathead_physicist_mappings()]
            },
            "turingo": {
                "framework": "Turingo",
                "total_agents": 1,
                "mappings": [asdict(m) for m in self.get_turingo_mappings()]
            },
            "atlas": {
                "framework": "ATLAS",
                "total_agents": 1,
                "mappings": [asdict(m) for m in self.get_atlas_mappings()]
            }
        }
        
        # Calculate totals
        all_mappings = []
        for framework in report.values():
            if isinstance(framework, dict) and "mappings" in framework:
                all_mappings.extend(framework["mappings"])
        
        unique_mcps = set()
        for mapping in all_mappings:
            unique_mcps.update(mapping["mcp_servers"])
        
        report["summary"] = {
            "total_frameworks": 3,
            "total_agents": sum(f["total_agents"] for f in report.values() if isinstance(f, dict) and "total_agents" in f),
            "total_mcp_integrations": sum(len(m["mcp_servers"]) for m in all_mappings),
            "unique_mcps_used": sorted(list(unique_mcps))
        }
        
        return report


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_list(args, manager: MCPManager):
    """List configured MCP servers."""
    servers = manager.list_servers()
    
    if args.groups:
        groups = manager.get_server_groups()
        for group, server_list in groups.items():
            print(f"{group}: {', '.join(server_list)}")
    else:
        print(f"{'Server':<20} {'Description':<50} {'Tags'}")
        print("-" * 90)
        for s in servers:
            tags = ', '.join(s.tags[:3])
            print(f"{s.name:<20} {s.description[:50]:<50} {tags}")
    
    return 0


def cmd_ping(args, manager: MCPManager):
    """Ping an MCP server."""
    result = manager.ping_server(args.server)
    
    status_icon = "[OK]" if result["status"] in ["available", "connected"] else "[FAIL]"
    print(f"{status_icon} {args.server}: {result['status']} ({result['latency_ms']}ms)")
    
    if result.get("error"):
        print(f"    Error: {result['error']}")
    
    return 0 if result["status"] in ["available", "connected"] else 1


def cmd_tools(args, manager: MCPManager):
    """List tools for an MCP server."""
    result = manager.list_tools(args.server)
    
    print(f"Tools for {args.server}:")
    for tool in result["tools"]:
        name = tool.get("name", tool)
        desc = tool.get("description", "")
        print(f"  - {name}: {desc[:50]}")
    
    return 0


def cmd_execute(args, manager: MCPManager):
    """Execute MCP tool (placeholder)."""
    print(f"Execute: {args.server}.{args.tool}")
    if args.args:
        print(f"Arguments: {args.args}")
    print("Note: Full execution requires mcp-cli integration")
    return 0


def cmd_test(args, manager: MCPManager):
    """Test MCP server."""
    tester = MCPTester(manager)
    report = tester.test_server(args.server, args.suite)
    
    print(f"Server: {report.server_name}")
    print(f"Status: {report.overall_status.value.upper()}")
    print(f"Timestamp: {report.timestamp}")
    print("")
    print("Tests:")
    
    for test in report.tests:
        icon = "[OK]" if test.status == TestStatus.PASSED else "[FAIL]"
        print(f"  {icon} {test.test_name}: {test.message} ({test.duration_ms}ms)")
    
    print("")
    print(f"Summary: {report.summary.get('passed', 0)} passed, {report.summary.get('failed', 0)} failed")
    
    return 0 if report.overall_status == TestStatus.PASSED else 1


def cmd_integrate(args, manager: MCPManager):
    """Integrate MCP with agents."""
    integrator = AgentIntegrator(manager)
    report = integrator.generate_integration_report(args.workflow)
    
    if args.report:
        output_path = Path(args.report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Integration report saved: {output_path}")
    
    print(f"\n📊 Summary:")
    print(f"   Frameworks: {report['summary']['total_frameworks']}")
    print(f"   Agents: {report['summary']['total_agents']}")
    print(f"   MCP Integrations: {report['summary']['total_mcp_integrations']}")
    print(f"   Unique MCPs: {len(report['summary']['unique_mcps_used'])}")
    print(f"\n🔗 MCPs in use: {', '.join(report['summary']['unique_mcps_used'])}")
    
    return 0


def cmd_health(args, manager: MCPManager):
    """Check MCP ecosystem health."""
    result = manager.health_check()
    
    print("MCP Server Health Check")
    print("=" * 50)
    print(f"MCP CLI Available: {'Yes' if result['mcp_cli_available'] else 'No'}")
    print(f"Total Servers: {result['summary']['total']}")
    print(f"Available: {result['summary']['available']}")
    print(f"Errors: {result['summary']['error']}")
    print("")
    
    for server, status in result["servers"].items():
        icon = "[OK]" if status["status"] in ["available", "connected"] else "[FAIL]"
        print(f"  {icon} {server}: {status['status']}")
    
    return 0


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Unified MCP CLI - MCP server operations and testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # mcp list
    list_parser = subparsers.add_parser('list', help='List configured MCP servers')
    list_parser.add_argument('--groups', action='store_true', help='Show server groups')
    
    # mcp ping
    ping_parser = subparsers.add_parser('ping', help='Ping MCP server')
    ping_parser.add_argument('server', help='Server name')
    
    # mcp tools
    tools_parser = subparsers.add_parser('tools', help='List server tools')
    tools_parser.add_argument('server', help='Server name')
    
    # mcp execute
    execute_parser = subparsers.add_parser('execute', help='Execute MCP tool')
    execute_parser.add_argument('server', help='Server name')
    execute_parser.add_argument('tool', help='Tool name')
    execute_parser.add_argument('--args', help='Tool arguments (JSON)')
    
    # mcp test
    test_parser = subparsers.add_parser('test', help='Test MCP server')
    test_parser.add_argument('server', help='Server name')
    test_parser.add_argument('--suite', choices=['smoke', 'full'], default='smoke', help='Test suite to run')
    
    # mcp integrate
    integrate_parser = subparsers.add_parser('integrate', help='Integrate MCP with agents')
    integrate_parser.add_argument('--workflow', help='Workflow name')
    integrate_parser.add_argument('--report', help='Output report file')
    
    # mcp health
    health_parser = subparsers.add_parser('health', help='Check MCP ecosystem health')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize manager
        manager = MCPManager()
        
        # Route to handlers
        if args.command == 'list':
            return cmd_list(args, manager)
        elif args.command == 'ping':
            return cmd_ping(args, manager)
        elif args.command == 'tools':
            return cmd_tools(args, manager)
        elif args.command == 'execute':
            return cmd_execute(args, manager)
        elif args.command == 'test':
            return cmd_test(args, manager)
        elif args.command == 'integrate':
            return cmd_integrate(args, manager)
        elif args.command == 'health':
            return cmd_health(args, manager)
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())