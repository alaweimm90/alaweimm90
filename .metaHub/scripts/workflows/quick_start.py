#!/usr/bin/env python3
"""
Quick Start CLI - Simplified Workflow Execution Interface
One-command access to autonomous DevOps workflows

Author: alaweimm90
Last Updated: 2025-11-28
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

class QuickStartCLI:
    """
    Simplified CLI for executing common DevOps workflows
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.scripts_dir = workspace_root / ".metaHub" / "scripts"
        self.config_file = workspace_root / ".ai" / "mcp" / "mcp-servers.json"

    def check_prerequisites(self) -> Dict[str, bool]:
        """Validate system prerequisites"""
        checks = {
            "workspace_exists": self.workspace_root.exists(),
            "mcp_config_exists": self.config_file.exists(),
            "workflow_runner_exists": (self.scripts_dir / "devops_workflow_runner.py").exists(),
            "telemetry_dashboard_exists": (self.scripts_dir / "telemetry_dashboard.py").exists(),
            "python_available": self._check_python(),
            "git_available": self._check_git(),
        }
        return checks

    def _check_python(self) -> bool:
        """Check if Python 3 is available"""
        try:
            result = subprocess.run(
                ["python3", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_git(self) -> bool:
        """Check if Git is available"""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def validate_mcp_config(self) -> Dict[str, any]:
        """Validate MCP configuration"""
        if not self.config_file.exists():
            return {"valid": False, "error": "MCP config not found"}

        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            mcps = config.get("mcpServers", {})
            server_groups = config.get("serverGroups", {})

            return {
                "valid": True,
                "total_mcps": len(mcps),
                "server_groups": len(server_groups),
                "error_free_pipeline": "error-free-pipeline" in server_groups,
                "devops_critical": "devops-critical" in server_groups
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def run_workflow(self, workflow_type: str, problem: str, dry_run: bool = False) -> int:
        """Execute a workflow using devops_workflow_runner.py"""
        workflow_runner = self.scripts_dir / "devops_workflow_runner.py"

        cmd = [
            "python3",
            str(workflow_runner),
            "--problem", problem,
            "--workspace", str(self.workspace_root)
        ]

        if dry_run:
            cmd.append("--dry-run")

        print(f"\n🚀 Executing {workflow_type} workflow...")
        print(f"   Problem: {problem}")
        print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
        print()

        try:
            result = subprocess.run(cmd, cwd=self.workspace_root)
            return result.returncode
        except Exception as e:
            print(f"❌ Error executing workflow: {e}")
            return 1

    def show_dashboard(self) -> int:
        """Display telemetry dashboard"""
        dashboard_script = self.scripts_dir / "telemetry_dashboard.py"

        cmd = [
            "python3",
            str(dashboard_script),
            "--workspace", str(self.workspace_root)
        ]

        try:
            result = subprocess.run(cmd, cwd=self.workspace_root)
            return result.returncode
        except Exception as e:
            print(f"❌ Error displaying dashboard: {e}")
            return 1

    def interactive_mode(self):
        """Interactive workflow selection"""
        print("=" * 80)
        print("🤖 QUICK START - Interactive Workflow Selection")
        print("=" * 80)
        print()

        # Show system health
        print("📊 System Status:")
        checks = self.check_prerequisites()
        for check_name, passed in checks.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check_name.replace('_', ' ').title()}")
        print()

        # Validate MCP config
        mcp_status = self.validate_mcp_config()
        if mcp_status["valid"]:
            print(f"  ✅ MCP Servers: {mcp_status['total_mcps']} configured")
            print(f"  ✅ Server Groups: {mcp_status['server_groups']}")
        else:
            print(f"  ❌ MCP Config: {mcp_status['error']}")
            return 1

        print()
        print("=" * 80)
        print("📋 Available Workflow Templates:")
        print("=" * 80)
        print()

        workflows = {
            "1": {
                "name": "Deploy Feature",
                "description": "Full-stack feature deployment with testing and validation",
                "problem": "Deploy new feature with complete error-free validation"
            },
            "2": {
                "name": "Debug Production Issue",
                "description": "Root cause analysis and bug fixing workflow",
                "problem": "Debug and fix production issue with root cause analysis"
            },
            "3": {
                "name": "Scale Infrastructure",
                "description": "Proactive infrastructure scaling for traffic spikes",
                "problem": "Scale infrastructure for anticipated traffic increase"
            },
            "4": {
                "name": "Security Audit",
                "description": "Comprehensive security scanning and vulnerability assessment",
                "problem": "Run security audit and fix vulnerabilities"
            },
            "5": {
                "name": "Research Workflow",
                "description": "Multi-agent research with literature review and analysis",
                "problem": "Literature review and theoretical framework development"
            },
            "6": {
                "name": "Custom Workflow",
                "description": "Define your own problem statement",
                "problem": None  # Will prompt user
            },
            "7": {
                "name": "View Dashboard",
                "description": "Display telemetry dashboard with system metrics",
                "problem": None  # Not a workflow
            }
        }

        for key, workflow in workflows.items():
            print(f"  [{key}] {workflow['name']}")
            print(f"      {workflow['description']}")
            print()

        print("  [0] Exit")
        print()

        # Get user selection
        try:
            choice = input("Select workflow (0-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting...")
            return 0

        if choice == "0":
            print("Exiting...")
            return 0

        if choice == "7":
            # Show dashboard
            return self.show_dashboard()

        if choice not in workflows:
            print(f"❌ Invalid selection: {choice}")
            return 1

        workflow = workflows[choice]

        # Get problem statement
        if workflow["problem"] is None:
            try:
                problem = input("\nEnter problem statement: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nCancelled.")
                return 0

            if not problem:
                print("❌ Problem statement cannot be empty")
                return 1
        else:
            problem = workflow["problem"]

        # Ask for dry-run mode
        try:
            dry_run_input = input("\nRun in dry-run mode? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nCancelled.")
            return 0

        dry_run = dry_run_input != "n"

        # Execute workflow
        return self.run_workflow(workflow["name"], problem, dry_run)

    def preset_workflow(self, preset: str, dry_run: bool = False) -> int:
        """Execute a preset workflow"""
        presets = {
            "deploy": "Deploy new feature with complete error-free validation",
            "debug": "Debug and fix production issue with root cause analysis",
            "scale": "Scale infrastructure for anticipated traffic increase",
            "security": "Run comprehensive security audit and fix vulnerabilities",
            "research": "Literature review and theoretical framework development",
            "test": "Test the error-free pipeline with all DevOps stages"
        }

        if preset not in presets:
            print(f"❌ Unknown preset: {preset}")
            print(f"   Available: {', '.join(presets.keys())}")
            return 1

        problem = presets[preset]
        return self.run_workflow(preset, problem, dry_run)

    def status_check(self):
        """Display comprehensive status check"""
        print("=" * 80)
        print("🔍 SYSTEM STATUS CHECK")
        print("=" * 80)
        print()

        # Prerequisites
        print("📋 Prerequisites:")
        checks = self.check_prerequisites()
        all_passed = all(checks.values())

        for check_name, passed in checks.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check_name.replace('_', ' ').title()}")
        print()

        # MCP Configuration
        print("🔧 MCP Configuration:")
        mcp_status = self.validate_mcp_config()

        if mcp_status["valid"]:
            print(f"  ✅ Valid configuration")
            print(f"  ✅ MCP Servers: {mcp_status['total_mcps']}")
            print(f"  ✅ Server Groups: {mcp_status['server_groups']}")

            if mcp_status.get("error_free_pipeline"):
                print(f"  ✅ Error-free pipeline configured")
            if mcp_status.get("devops_critical"):
                print(f"  ✅ DevOps critical MCPs configured")
        else:
            print(f"  ❌ Configuration error: {mcp_status['error']}")
            all_passed = False
        print()

        # Overall status
        print("=" * 80)
        if all_passed:
            print("✅ System Ready - All checks passed")
            print()
            print("Quick Start Commands:")
            print("  python .metaHub/scripts/quick_start.py --interactive")
            print("  python .metaHub/scripts/quick_start.py --preset deploy --dry-run")
            print("  python .metaHub/scripts/quick_start.py --dashboard")
        else:
            print("❌ System Not Ready - Some checks failed")
            print()
            print("Please fix the issues above and try again.")
            print("See docs/DEVOPS-MCP-SETUP.md for setup instructions.")
        print("=" * 80)

        return 0 if all_passed else 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quick Start CLI - Simplified Workflow Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python quick_start.py --interactive

  # Run preset workflow
  python quick_start.py --preset deploy --dry-run
  python quick_start.py --preset debug
  python quick_start.py --preset scale --dry-run

  # Custom workflow
  python quick_start.py --problem "Your custom problem statement" --dry-run

  # View dashboard
  python quick_start.py --dashboard

  # Status check
  python quick_start.py --status

Preset Workflows:
  deploy      - Deploy new feature with validation
  debug       - Debug production issue with RCA
  scale       - Scale infrastructure for traffic
  security    - Run security audit
  research    - Multi-agent research workflow
  test        - Test the error-free pipeline
        """
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/mnt/c/Users/mesha/Desktop/GitHub"),
        help="Workspace root directory"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive workflow selection mode"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["deploy", "debug", "scale", "security", "research", "test"],
        help="Run a preset workflow"
    )

    parser.add_argument(
        "--problem",
        type=str,
        help="Custom problem statement for workflow"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (safe, no real changes)"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Display telemetry dashboard"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check system status"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = QuickStartCLI(args.workspace)

    # Route to appropriate mode
    if args.status:
        return cli.status_check()

    if args.dashboard:
        return cli.show_dashboard()

    if args.interactive:
        return cli.interactive_mode()

    if args.preset:
        return cli.preset_workflow(args.preset, args.dry_run)

    if args.problem:
        return cli.run_workflow("custom", args.problem, args.dry_run)

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
