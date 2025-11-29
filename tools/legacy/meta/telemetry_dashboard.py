#!/usr/bin/env python3
"""
Telemetry Dashboard - Real-time MCP Workflow Monitoring
Visualizes workflow execution, agent-MCP integrations, and system health

Author: alaweimm90
Last Updated: 2025-11-28
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

class TelemetryDashboard:
    """
    CLI-based dashboard for MCP workflow telemetry visualization
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.workflows_dir = workspace_root / ".metaHub" / "orchestration" / "workflows"
        self.telemetry_dir = workspace_root / ".metaHub" / "orchestration" / "telemetry"
        self.reports_dir = workspace_root / ".metaHub" / "reports"

    def load_latest_workflow(self) -> Optional[Dict]:
        """Load the most recent workflow execution"""
        if not self.workflows_dir.exists():
            return None

        workflow_files = sorted(
            self.workflows_dir.glob("workflow_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not workflow_files:
            return None

        with open(workflow_files[0], 'r') as f:
            return json.load(f)

    def load_integration_report(self) -> Optional[Dict]:
        """Load agent-MCP integration report"""
        report_path = self.reports_dir / "agent-mcp-integration.json"
        if not report_path.exists():
            return None

        with open(report_path, 'r') as f:
            return json.load(f)

    def render_banner(self):
        """Render dashboard banner"""
        print("=" * 80)
        print("🔍 MCP TELEMETRY DASHBOARD")
        print("Real-time Workflow & Integration Monitoring")
        print("=" * 80)
        print()

    def render_system_health(self, workflow: Dict, integration: Dict):
        """Render system health overview"""
        print("📊 SYSTEM HEALTH")
        print("-" * 80)

        # Workflow Status
        workflow_status = workflow.get("status", "unknown")
        status_icon = "✅" if workflow_status == "success" else "❌" if workflow_status == "failed" else "⏳"

        print(f"  Latest Workflow: {status_icon} {workflow_status.upper()}")
        print(f"  Workflow ID: {workflow.get('workflow_id', 'N/A')}")
        print(f"  Steps Completed: {workflow.get('steps_completed', 0)}")
        print(f"  Errors: {len(workflow.get('errors', []))}")
        print()

        # Integration Stats
        if integration:
            summary = integration.get("summary", {})
            print(f"  Agent Frameworks: {summary.get('total_frameworks', 0)}")
            print(f"  Agents Wired: {summary.get('total_agents', 0)}")
            print(f"  MCP Integrations: {summary.get('total_mcp_integrations', 0)}")
            print(f"  Unique MCPs: {len(summary.get('unique_mcps_used', []))}")

        print()

    def render_workflow_stages(self, workflow: Dict):
        """Render workflow stage execution"""
        print("🚀 WORKFLOW PIPELINE")
        print("-" * 80)

        stages = workflow.get("stages", {})
        if not stages:
            print("  No workflow stages found")
            return

        stage_names = [
            ("analysis", "📋 Analysis"),
            ("git_state", "🔄 Git State"),
            ("tests", "🧪 Tests"),
            ("infrastructure", "🏗️  Infrastructure"),
            ("deployment", "🚢 Deployment"),
            ("monitoring", "📈 Monitoring")
        ]

        for stage_key, stage_label in stage_names:
            if stage_key in stages:
                stage_data = stages[stage_key]

                # Determine status
                if isinstance(stage_data, dict):
                    if "error" in stage_data:
                        status = "❌ ERROR"
                    elif stage_data.get("status") in ["success", "completed", "no_changes"]:
                        status = "✅ SUCCESS"
                    else:
                        status = f"⏳ {stage_data.get('status', 'PENDING').upper()}"
                else:
                    status = "✅ SUCCESS"

                print(f"  {stage_label}: {status}")

                # Show stage details
                if stage_key == "git_state" and isinstance(stage_data, dict):
                    if "branch" in stage_data:
                        print(f"     Branch: {stage_data['branch']}")
                    if "modified_files" in stage_data:
                        print(f"     Modified Files: {stage_data['modified_files']}")

                elif stage_key == "tests" and isinstance(stage_data, dict):
                    total = stage_data.get("total_tests", 0)
                    passed = stage_data.get("passed", 0)
                    if total > 0:
                        print(f"     Tests: {passed}/{total} passed")

                elif stage_key == "infrastructure" and isinstance(stage_data, dict):
                    changes = stage_data.get("changes", {})
                    if changes:
                        add = changes.get("add", 0)
                        change = changes.get("change", 0)
                        destroy = changes.get("destroy", 0)
                        print(f"     Changes: +{add} ~{change} -{destroy}")

        print()

    def render_mcp_usage(self, integration: Dict):
        """Render MCP server usage statistics"""
        print("🔧 MCP SERVER USAGE")
        print("-" * 80)

        if not integration:
            print("  No integration data available")
            return

        # Count MCP usage across frameworks
        mcp_counts = defaultdict(int)

        for framework_key in ["meathead_physicist", "turingo", "atlas"]:
            if framework_key in integration:
                framework_data = integration[framework_key]
                for mapping in framework_data.get("mappings", []):
                    for mcp in mapping.get("mcp_servers", []):
                        mcp_counts[mcp] += 1

        # Sort by usage
        sorted_mcps = sorted(mcp_counts.items(), key=lambda x: x[1], reverse=True)

        for mcp_name, count in sorted_mcps[:10]:  # Top 10
            bar_width = int((count / max(mcp_counts.values())) * 40)
            bar = "█" * bar_width
            print(f"  {mcp_name:25s} {bar} {count}")

        print()

    def render_agent_integrations(self, integration: Dict):
        """Render agent-MCP integration matrix"""
        print("🤖 AGENT-MCP INTEGRATIONS")
        print("-" * 80)

        if not integration:
            print("  No integration data available")
            return

        frameworks = [
            ("meathead_physicist", "MeatheadPhysicist"),
            ("turingo", "Turingo"),
            ("atlas", "ATLAS")
        ]

        for framework_key, framework_name in frameworks:
            if framework_key not in integration:
                continue

            framework_data = integration[framework_key]
            print(f"\n  {framework_name}:")
            print(f"    Agents: {framework_data.get('total_agents', 0)}")
            print(f"    Integrations: {framework_data.get('total_mcp_integrations', 0)}")

            # Show sample agents
            mappings = framework_data.get("mappings", [])[:3]  # First 3
            for mapping in mappings:
                agent_name = mapping.get("agent_name", "Unknown")
                mcp_count = len(mapping.get("mcp_servers", []))
                print(f"      - {agent_name}: {mcp_count} MCPs")

        print()

    def render_recent_activity(self):
        """Render recent workflow activity"""
        print("📅 RECENT ACTIVITY")
        print("-" * 80)

        if not self.workflows_dir.exists():
            print("  No workflow history available")
            return

        workflow_files = sorted(
            self.workflows_dir.glob("workflow_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:5]  # Last 5

        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                data = json.load(f)

            workflow_id = data.get("workflow_id", "unknown")
            status = data.get("status", "unknown")
            steps = data.get("steps_completed", 0)
            errors = len(data.get("errors", []))

            status_icon = "✅" if status == "success" else "❌" if status == "failed" else "⏳"

            # Get file timestamp
            timestamp = datetime.fromtimestamp(workflow_file.stat().st_mtime)
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            print(f"  {status_icon} {workflow_id} ({time_str})")
            print(f"     Steps: {steps}, Errors: {errors}")

        print()

    def render_footer(self):
        """Render dashboard footer"""
        print("=" * 80)
        print("Dashboard auto-refreshes with new workflow executions")
        print("Run workflows with: python .metaHub/scripts/devops_workflow_runner.py")
        print("=" * 80)

    def display(self):
        """Display complete dashboard"""
        self.render_banner()

        # Load data
        workflow = self.load_latest_workflow()
        integration = self.load_integration_report()

        if not workflow:
            print("⚠️  No workflow data available yet")
            print("   Run a workflow first: python .metaHub/scripts/devops_workflow_runner.py --dry-run")
            return

        # Render sections
        self.render_system_health(workflow, integration)
        self.render_workflow_stages(workflow)

        if integration:
            self.render_mcp_usage(integration)
            self.render_agent_integrations(integration)

        self.render_recent_activity()
        self.render_footer()

    def export_json(self, output_path: Path):
        """Export dashboard data as JSON"""
        workflow = self.load_latest_workflow()
        integration = self.load_integration_report()

        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "latest_workflow": workflow,
            "integration_stats": integration,
            "system_health": {
                "workflow_status": workflow.get("status") if workflow else "no_data",
                "total_steps": workflow.get("steps_completed", 0) if workflow else 0,
                "total_errors": len(workflow.get("errors", [])) if workflow else 0,
                "agent_integrations": integration.get("summary", {}).get("total_mcp_integrations", 0) if integration else 0
            }
        }

        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        print(f"✅ Dashboard data exported to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Telemetry Dashboard - MCP Workflow Monitoring"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/mnt/c/Users/mesha/Desktop/GitHub"),
        help="Workspace root directory"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export dashboard data to JSON file"
    )

    args = parser.parse_args()

    dashboard = TelemetryDashboard(args.workspace)

    if args.export:
        dashboard.export_json(args.export)
    else:
        dashboard.display()


if __name__ == "__main__":
    main()
