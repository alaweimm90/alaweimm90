#!/usr/bin/env python3
"""
Technical Debt CLI Commands

Integration with the automation CLI for technical debt scanning and remediation.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
import json

from technical_debt_remediation import (
    TechnicalDebtOrchestrator,
    TechnicalDebtScanner,
    TechnicalDebatePrioritizer,
    TechnicalDebtRemediator,
    TechnicalDebtMonitor
)
from remediation_plan_generator import RemediationPlanGenerator


def _serialize_assessment(assessment):
    """Convert a DebtAssessment to a JSON-serializable dict."""
    return {
        "total_debt_items": assessment.total_debt_items,
        "estimated_remediation_time": assessment.estimated_remediation_time,
        "debt_by_type": {k.value: v for k, v in assessment.debt_by_type.items()},
        "debt_by_severity": {k.value: v for k, v in assessment.debt_by_severity.items()},
        "priority_items_count": len(assessment.priority_items),
        "assessment_timestamp": assessment.assessment_timestamp,
    }


def _serialize_debt_item(item):
    """Convert a TechnicalDebtItem to a JSON-serializable dict."""
    return {
        "id": item.id,
        "type": item.type.value,
        "severity": item.severity.value,
        "description": item.description,
        "file_path": item.file_path,
        "line_number": item.line_number,
        "estimated_effort": item.estimated_effort,
        "business_impact": item.business_impact,
        "metadata": item.metadata,
    }


def cmd_debt_scan(args):
    """Scan for technical debt in the specified path."""

    print(f"🔍 Scanning technical debt in: {args.path}")
    print("=" * 60)

    try:
        # Initialize scanner
        scanner = TechnicalDebtScanner(args.path)

        # Run assessment
        assessment = asyncio.run(scanner.scan_all_issues())

        # Display results
        print(f"\n📊 Technical Debt Assessment Results:")
        print(f"   Total debt items: {assessment.total_debt_items}")
        print(f"   Estimated remediation time: {assessment.estimated_remediation_time} hours")
        print(f"   Priority items: {len(assessment.priority_items)}")

        print(f"\n📈 Debt by Type:")
        for debt_type, count in assessment.debt_by_type.items():
            print(f"   {debt_type.value}: {count}")

        print(f"\n📈 Debt by Severity:")
        for severity, count in assessment.debt_by_severity.items():
            print(f"   {severity.value}: {count}")

        if args.export:
            # Export detailed results (markdown plan for priority items)
            generator = RemediationPlanGenerator(args.path)
            tickets = generator.generate_remediation_plan(assessment.priority_items)
            output_file = generator.export_tickets_to_markdown(tickets, args.export)
            print(f"\n📄 Detailed report exported to: {output_file}")

        if getattr(args, "json", None):
            # Export a compact JSON summary for tooling/MCPS consumption
            summary = {
                "path": args.path,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assessment": _serialize_assessment(assessment),
            }
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"\n📄 JSON summary exported to: {args.json}")

        return 0

    except Exception as e:
        print(f"❌ Error during scan: {e}")
        return 1


def cmd_debt_remediate(args):
    """Run the complete technical debt remediation workflow."""

    print(f"🔧 Starting technical debt remediation workflow")
    print(f"   Path: {args.path}")
    print(f"   Available hours: {args.hours}")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = TechnicalDebtOrchestrator(args.path)

        # Execute complete workflow
        results = asyncio.run(orchestrator.execute_remediation_workflow(args.hours))

        # Display summary
        print(f"\n🎯 Remediation Workflow Summary:")
        print(f"   ✅ Assessment: {results['assessment'].total_debt_items} items identified")
        print(f"   ✅ Prioritization: {len(results['prioritized_items'])} items selected")
        print(f"   ✅ Remediation: {results['remediation_results']['success_rate']:.1%} success rate")
        print(f"   ✅ Monitoring: Baseline established")

        if args.export:
            # Export remediation plan (markdown tickets)
            generator = RemediationPlanGenerator(args.path)
            tickets = generator.generate_remediation_plan(results['prioritized_items'])
            output_file = generator.export_tickets_to_markdown(tickets, args.export)
            print(f"   📄 Remediation plan exported to: {output_file}")

        if getattr(args, "json", None):
            # Export a JSON summary of the remediation workflow
            assessment = results["assessment"]
            baseline = results["monitoring_baseline"]
            summary = {
                "path": args.path,
                "available_hours": args.hours,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assessment": _serialize_assessment(assessment),
                "prioritized_items": [
                    _serialize_debt_item(item)
                    for item in results["prioritized_items"]
                ],
                "remediation_results": results["remediation_results"],
                "monitoring_baseline": _serialize_assessment(baseline),
            }
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"   📄 JSON summary exported to: {args.json}")

        return 0

    except Exception as e:
        print(f"❌ Error during remediation: {e}")
        return 1


def cmd_debt_monitor(args):
    """Start continuous technical debt monitoring."""

    print(f"🔄 Starting continuous technical debt monitoring")
    print(f"   Path: {args.path}")
    print("=" * 60)

    try:
        # Initialize monitor
        monitor = TechnicalDebtMonitor(args.path)

        # Start monitoring
        baseline = asyncio.run(monitor.start_continuous_monitoring())

        print(f"\n📊 Monitoring Setup Complete:")
        print(f"   Baseline debt items: {baseline.total_debt_items}")
        print(f"   Monitoring enabled: Yes")
        print(f"   Alert threshold: {args.threshold}% increase")

        print(f"\n💡 Next Steps:")
        print(f"   • Run 'atlas debt scan' periodically to check for new debt")
        print(f"   • Set up CI/CD integration to run scans automatically")
        print(f"   • Review remediation plan and prioritize high-impact items")

        return 0

    except Exception as e:
        print(f"❌ Error setting up monitoring: {e}")
        return 1


def cmd_debt_plan(args):
    """Generate detailed remediation plan for existing debt items."""

    print(f"📋 Generating technical debt remediation plan")
    print(f"   Path: {args.path}")
    print(f"   Focus: {args.focus}")
    print("=" * 60)

    try:
        # Scan for debt items
        scanner = TechnicalDebtScanner(args.path)
        assessment = asyncio.run(scanner.scan_all_issues())

        # Filter by focus if specified
        if args.focus != "all":
            from technical_debt_remediation import DebtType
            try:
                focus_type = DebtType(args.focus)
                filtered_items = [item for item in scanner.debt_items if item.type == focus_type]
                print(f"   Filtered to {args.focus} debt: {len(filtered_items)} items")
            except ValueError:
                print(f"❌ Invalid focus type: {args.focus}")
                return 1
        else:
            filtered_items = assessment.priority_items

        # Generate remediation plan
        generator = RemediationPlanGenerator(args.path)
        tickets = generator.generate_remediation_plan(filtered_items)

        # Export plan
        output_file = args.export or f"remediation_plan_{args.focus}.md"
        generator.export_tickets_to_markdown(tickets, output_file)

        print(f"\n📊 Plan Generation Results:")
        print(f"   Tickets created: {len(tickets)}")
        print(f"   Total estimated hours: {sum(t.estimated_hours for t in tickets)}")
        print(f"   Export file: {output_file}")

        return 0

    except Exception as e:
        print(f"❌ Error generating plan: {e}")
        return 1


def add_debt_commands(subparsers):
    """Add technical debt commands to the argument parser."""
    debt_parser = subparsers.add_parser(
        'debt',
        help='Technical debt scanning and remediation',
        description='Manage technical debt with AI-powered analysis and remediation planning'
    )
    debt_subparsers = debt_parser.add_subparsers(dest='debt_command', help='Debt management commands')

    _add_scan_parser(debt_subparsers)
    _add_remediate_parser(debt_subparsers)
    _add_monitor_parser(debt_subparsers)
    _add_plan_parser(debt_subparsers)


def _add_scan_parser(subparsers):
    """Add scan subcommand parser."""
    p = subparsers.add_parser('scan', help='Scan for technical debt')
    p.add_argument('--path', default='.', help='Path to scan (default: current directory)')
    p.add_argument('--export', help='Export detailed results to markdown file')
    p.add_argument('--json', help='Write JSON summary to file')
    p.set_defaults(func=cmd_debt_scan)

def _add_remediate_parser(subparsers):
    """Add remediate subcommand parser."""
    p = subparsers.add_parser('remediate', help='Run remediation workflow')
    p.add_argument('--path', default='.', help='Path to remediate (default: current directory)')
    p.add_argument('--hours', type=int, default=40, help='Available hours for remediation (default: 40)')
    p.add_argument('--export', help='Export remediation plan to markdown file')
    p.add_argument('--json', help='Write JSON summary to file')
    p.set_defaults(func=cmd_debt_remediate)

def _add_monitor_parser(subparsers):
    """Add monitor subcommand parser."""
    p = subparsers.add_parser('monitor', help='Start continuous monitoring')
    p.add_argument('--path', default='.', help='Path to monitor (default: current directory)')
    p.add_argument('--threshold', type=int, default=10, help='Alert threshold percentage (default: 10)')
    p.set_defaults(func=cmd_debt_monitor)

def _add_plan_parser(subparsers):
    """Add plan subcommand parser."""
    p = subparsers.add_parser('plan', help='Generate remediation plan')
    p.add_argument('--path', default='.', help='Path to analyze (default: current directory)')
    p.add_argument('--focus', default='all',
                   choices=['all', 'code_complexity', 'naming_consistency', 'documentation_drift',
                            'architectural_violations', 'security_issues', 'performance_bottlenecks',
                            'test_coverage_gaps', 'dependency_bloat'],
                   help='Focus on specific debt type (default: all)')
    p.add_argument('--export', help='Export plan to specific file')
    p.set_defaults(func=cmd_debt_plan)

def main():
    """Main entry point for standalone debt CLI."""
    parser = argparse.ArgumentParser(description='Technical Debt Management CLI', prog='atlas-debt')
    parser.add_argument('--version', action='version', version='1.0.0')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    _add_scan_parser(subparsers)
    _add_remediate_parser(subparsers)
    _add_monitor_parser(subparsers)
    _add_plan_parser(subparsers)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
