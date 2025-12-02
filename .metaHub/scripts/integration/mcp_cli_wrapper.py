#!/usr/bin/env python3
"""
mcp_cli_wrapper.py - MCP CLI Integration for Orchestration System

Provides a Python interface to MCP CLI for:
- Server management and testing
- Tool discovery and execution
- Workflow automation
- Integration with orchestration telemetry

Usage:
    python mcp_cli_wrapper.py list-servers
    python mcp_cli_wrapper.py ping --server github
    python mcp_cli_wrapper.py tools --server filesystem
    python mcp_cli_wrapper.py execute --server sqlite --tool list_tables
    python mcp_cli_wrapper.py chat --server context --message "Store this context"
"""

import json
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import click
import yaml


class ServerStatus(Enum):
    """MCP Server status."""
    AVAILABLE = "available"
    CONNECTED = "connected"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


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
    tools: List[Dict[str, Any]] = field(default_factory=list)
    last_ping: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ToolExecutionResult:
    """Result of executing an MCP tool."""
    success: bool
    server: str
    tool: str
    duration_ms: int
    output: Any
    error: Optional[str] = None


class MCPCLIWrapper:
    """Wrapper for MCP CLI integration with orchestration."""

    CONFIG_PATH = ".ai/mcp/mcp-servers.json"
    REGISTRY_PATH = ".ai/mcp/server-registry.yaml"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.config_path = self.base_path / self.CONFIG_PATH
        self.registry_path = self.base_path / self.REGISTRY_PATH

        self.config = self._load_config()
        self.registry = self._load_registry()

        # Check if mcp-cli is available
        self.mcp_cli_available = self._check_mcp_cli()

        # Telemetry integration
        self.telemetry = self._get_telemetry()

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
        # Check for mcp-cli in PATH
        if shutil.which("mcp-cli"):
            return True

        # Check for uvx (can run mcp-cli)
        if shutil.which("uvx"):
            return True

        # Check for npx (can run MCP servers)
        if shutil.which("npx"):
            return True

        return False

    def _get_telemetry(self):
        """Get telemetry instance if available."""
        try:
            from orchestration_telemetry import OrchestrationTelemetry, EventType, EventStatus
            return OrchestrationTelemetry(self.base_path)
        except ImportError:
            return None

    def _run_mcp_command(
        self,
        args: List[str],
        timeout: int = 30,
        capture_output: bool = True
    ) -> Tuple[bool, str, str]:
        """Run an MCP CLI command."""
        # Try mcp-cli first
        if shutil.which("mcp-cli"):
            cmd = ["mcp-cli"] + args
        elif shutil.which("uvx"):
            cmd = ["uvx", "mcp-cli"] + args
        else:
            return False, "", "mcp-cli not found. Install with: pip install mcp-cli or uvx mcp-cli"

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=str(self.base_path)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def _run_npx_server(
        self,
        server_name: str,
        args: List[str] = None,
        timeout: int = 10
    ) -> Tuple[bool, str]:
        """Test if an NPX-based MCP server can start."""
        server_config = self.config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return False, f"Server {server_name} not found in configuration"

        command = server_config.get("command", "npx")
        server_args = server_config.get("args", [])

        # For npx servers, we just check if the package exists
        if command == "npx":
            package = server_args[1] if len(server_args) > 1 else None
            if package:
                try:
                    # Quick check if package info is available
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

        # Check if server exists in config
        if server_name not in self.config.get("mcpServers", {}):
            result["status"] = "not_configured"
            result["error"] = f"Server {server_name} not found in mcp-servers.json"
            return result

        # Try to verify the server package
        success, message = self._run_npx_server(server_name)

        end_time = datetime.now()
        result["latency_ms"] = int((end_time - start_time).total_seconds() * 1000)

        if success:
            result["status"] = "available"
            result["details"]["message"] = message
        else:
            result["status"] = "error"
            result["error"] = message

        # Try MCP CLI ping if available
        if self.mcp_cli_available:
            cli_success, stdout, stderr = self._run_mcp_command(
                ["ping", "--server", server_name, "--config-file", str(self.config_path)],
                timeout=15
            )
            if cli_success:
                result["status"] = "connected"
                result["details"]["cli_output"] = stdout.strip()

        # Record telemetry
        if self.telemetry:
            from orchestration_telemetry import EventType, EventStatus
            self.telemetry.record_event(
                event_type=EventType.TOOL_INVOCATION,
                status=EventStatus.SUCCESS if result["status"] in ["available", "connected"] else EventStatus.FAILURE,
                tool=f"mcp_ping_{server_name}",
                duration_ms=result["latency_ms"],
                metadata={"server": server_name, "status": result["status"]}
            )

        return result

    def list_tools(self, server_name: str) -> Dict[str, Any]:
        """List available tools for an MCP server."""
        result = {
            "server": server_name,
            "tools": [],
            "success": False
        }

        # Check config for capabilities (as proxy for tools)
        server_config = self.config.get("mcpServers", {}).get(server_name, {})
        capabilities = server_config.get("capabilities", [])

        # Map capabilities to tool-like structures
        for cap in capabilities:
            result["tools"].append({
                "name": cap,
                "description": f"Capability: {cap}",
                "type": "capability"
            })

        # Try MCP CLI tools command if available
        if self.mcp_cli_available:
            success, stdout, stderr = self._run_mcp_command(
                ["tools", "--server", server_name, "--config-file", str(self.config_path)],
                timeout=30
            )
            if success and stdout:
                result["success"] = True
                try:
                    # Try to parse JSON output
                    tools_data = json.loads(stdout)
                    if isinstance(tools_data, list):
                        result["tools"] = tools_data
                except json.JSONDecodeError:
                    # Keep capability-based tools
                    result["cli_output"] = stdout.strip()
        else:
            result["success"] = True
            result["note"] = "Tools listed from configuration capabilities (mcp-cli not available)"

        return result

    def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ) -> ToolExecutionResult:
        """Execute a tool on an MCP server."""
        start_time = datetime.now()

        if not self.mcp_cli_available:
            return ToolExecutionResult(
                success=False,
                server=server_name,
                tool=tool_name,
                duration_ms=0,
                output=None,
                error="mcp-cli not available. Install with: pip install mcp-cli"
            )

        # Build command
        cmd_args = [
            "cmd",
            "--server", server_name,
            "--tool", tool_name,
            "--config-file", str(self.config_path)
        ]

        if arguments:
            cmd_args.extend(["--args", json.dumps(arguments)])

        success, stdout, stderr = self._run_mcp_command(cmd_args, timeout=timeout)

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Parse output
        output = None
        if stdout:
            try:
                output = json.loads(stdout)
            except json.JSONDecodeError:
                output = stdout.strip()

        result = ToolExecutionResult(
            success=success,
            server=server_name,
            tool=tool_name,
            duration_ms=duration_ms,
            output=output,
            error=stderr if not success else None
        )

        # Record telemetry
        if self.telemetry:
            from orchestration_telemetry import EventType, EventStatus
            self.telemetry.record_event(
                event_type=EventType.TOOL_INVOCATION,
                status=EventStatus.SUCCESS if success else EventStatus.FAILURE,
                tool=f"{server_name}.{tool_name}",
                duration_ms=duration_ms,
                metadata={"server": server_name, "tool": tool_name}
            )

        return result

    def chat(
        self,
        server_name: str,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a chat message through MCP CLI."""
        if not self.mcp_cli_available:
            return {
                "success": False,
                "error": "mcp-cli not available"
            }

        # Get defaults from config
        defaults = self.config.get("defaults", {})
        provider = provider or defaults.get("provider", "anthropic")
        model = model or defaults.get("model", "claude-sonnet-4-5-20250929")

        # For now, return a placeholder since chat is interactive
        return {
            "success": True,
            "server": server_name,
            "provider": provider,
            "model": model,
            "message": message,
            "note": "Use 'mcp-cli --server {server} --provider {provider} --model {model}' for interactive chat"
        }

    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        provider_config = self.config.get("providerConfig", {}).get(provider, {})
        return provider_config.get("models", [])

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


def format_output(data: Any, fmt: str = "text") -> str:
    """Format output for display."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)

    # Text format
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)

    return str(data)


@click.group()
def cli():
    """MCP CLI wrapper for orchestration integration."""
    pass


@cli.command('list-servers')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def list_servers_cmd(json_output: bool):
    """List all configured MCP servers."""
    try:
        wrapper = MCPCLIWrapper()
        servers = wrapper.list_servers()

        if json_output:
            data = [
                {
                    "name": s.name,
                    "description": s.description,
                    "capabilities": s.capabilities,
                    "tags": s.tags
                }
                for s in servers
            ]
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"{'Server':<15} {'Description':<50} {'Tags'}")
            click.echo("-" * 90)
            for s in servers:
                tags = ', '.join(s.tags[:3])
                click.echo(f"{s.name:<15} {s.description[:50]:<50} {tags}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('list-groups')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def list_groups_cmd(json_output: bool):
    """List server groups."""
    try:
        wrapper = MCPCLIWrapper()
        groups = wrapper.get_server_groups()

        if json_output:
            click.echo(json.dumps(groups, indent=2))
        else:
            for group, servers in groups.items():
                click.echo(f"{group}: {', '.join(servers)}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--server', '-s', required=True, help='Server name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def ping(server: str, json_output: bool):
    """Ping/test an MCP server."""
    try:
        wrapper = MCPCLIWrapper()
        result = wrapper.ping_server(server)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            status_icon = "[OK]" if result["status"] in ["available", "connected"] else "[FAIL]"
            click.echo(f"{status_icon} {server}: {result['status']} ({result['latency_ms']}ms)")
            if result.get("error"):
                click.echo(f"    Error: {result['error']}")

        raise SystemExit(0 if result["status"] in ["available", "connected"] else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--server', '-s', required=True, help='Server name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def tools(server: str, json_output: bool):
    """List tools for an MCP server."""
    try:
        wrapper = MCPCLIWrapper()
        result = wrapper.list_tools(server)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Tools for {server}:")
            for tool in result["tools"]:
                name = tool.get("name", tool)
                desc = tool.get("description", "")
                click.echo(f"  - {name}: {desc[:50]}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--server', '-s', required=True, help='Server name')
@click.option('--tool', '-t', required=True, help='Tool name')
@click.option('--args', '-a', help='Tool arguments as JSON')
@click.option('--timeout', default=60, help='Timeout in seconds')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def execute(server: str, tool: str, args: Optional[str], timeout: int, json_output: bool):
    """Execute a tool on an MCP server."""
    try:
        wrapper = MCPCLIWrapper()

        arguments = json.loads(args) if args else None
        result = wrapper.execute_tool(server, tool, arguments, timeout)

        if json_output:
            click.echo(json.dumps({
                "success": result.success,
                "server": result.server,
                "tool": result.tool,
                "duration_ms": result.duration_ms,
                "output": result.output,
                "error": result.error
            }, indent=2, default=str))
        else:
            status = "SUCCESS" if result.success else "FAILED"
            click.echo(f"{status}: {server}.{tool} ({result.duration_ms}ms)")
            if result.output:
                click.echo(f"Output: {result.output}")
            if result.error:
                click.echo(f"Error: {result.error}")

        raise SystemExit(0 if result.success else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
def health(json_output: bool):
    """Run health check on all servers."""
    try:
        wrapper = MCPCLIWrapper()
        result = wrapper.health_check()

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("MCP Server Health Check")
            click.echo("=" * 50)
            click.echo(f"MCP CLI Available: {'Yes' if result['mcp_cli_available'] else 'No'}")
            click.echo(f"Total Servers: {result['summary']['total']}")
            click.echo(f"Available: {result['summary']['available']}")
            click.echo(f"Errors: {result['summary']['error']}")
            click.echo("")

            for server, status in result["servers"].items():
                icon = "[OK]" if status["status"] in ["available", "connected"] else "[FAIL]"
                click.echo(f"  {icon} {server}: {status['status']}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--provider', '-p', help='Provider name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def models(provider: Optional[str], json_output: bool):
    """List available models for providers."""
    try:
        wrapper = MCPCLIWrapper()

        if provider:
            models_list = wrapper.get_provider_models(provider)
            if json_output:
                click.echo(json.dumps(models_list, indent=2))
            else:
                click.echo(f"Models for {provider}:")
                for m in models_list:
                    click.echo(f"  - {m}")
        else:
            # List all providers and models
            provider_config = wrapper.config.get("providerConfig", {})
            if json_output:
                click.echo(json.dumps(provider_config, indent=2))
            else:
                for prov, config in provider_config.items():
                    click.echo(f"\n{prov}:")
                    click.echo(f"  Default: {config.get('defaultModel', 'N/A')}")
                    click.echo(f"  Models: {', '.join(config.get('models', []))}")

        raise SystemExit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
