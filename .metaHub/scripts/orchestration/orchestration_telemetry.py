#!/usr/bin/env python3
"""
orchestration_telemetry.py - Metrics Collection and Reporting for AI Orchestration

Collects, aggregates, and reports metrics for multi-agent orchestration:
- Execution latency tracking
- Handoff success rates
- Hallucination detection rates
- Tool utilization statistics
- Workflow completion metrics

Usage:
    python orchestration_telemetry.py record --event handoff --tool cline --status success
    python orchestration_telemetry.py report --period 24h
    python orchestration_telemetry.py dashboard
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

import click
import yaml


class EventType(Enum):
    """Types of telemetry events."""
    HANDOFF = "handoff"
    TOOL_INVOCATION = "tool_invocation"
    CHECKPOINT = "checkpoint"
    HALLUCINATION_CHECK = "hallucination_check"
    ERROR = "error"
    RECOVERY = "recovery"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"


class EventStatus(Enum):
    """Status of telemetry events."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_id: str
    event_type: str
    timestamp: str
    status: str
    tool: Optional[str] = None
    workflow: Optional[str] = None
    correlation_id: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Aggregated metrics summary."""
    period_start: str
    period_end: str
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_status: Dict[str, int] = field(default_factory=dict)
    events_by_tool: Dict[str, int] = field(default_factory=dict)

    # Performance metrics
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    # Success rates
    handoff_success_rate: float = 0.0
    workflow_completion_rate: float = 0.0
    hallucination_rate: float = 0.0

    # Tool utilization
    tool_utilization: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class OrchestrationTelemetry:
    """Telemetry collection and reporting for orchestration."""

    TELEMETRY_DIR = ".metaHub/orchestration/telemetry"
    EVENTS_FILE = "events.jsonl"
    AGGREGATES_FILE = "aggregates.json"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.telemetry_dir = self.base_path / self.TELEMETRY_DIR
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.events_file = self.telemetry_dir / self.EVENTS_FILE
        self.aggregates_file = self.telemetry_dir / self.AGGREGATES_FILE

        self.policy = self._load_policy()

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

    def _load_policy(self) -> Dict[str, Any]:
        """Load orchestration governance policy."""
        policy_path = self.base_path / ".metaHub/policies/orchestration-governance.yaml"
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:12]

    def record_event(
        self,
        event_type: EventType,
        status: EventStatus,
        tool: Optional[str] = None,
        workflow: Optional[str] = None,
        correlation_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TelemetryEvent:
        """Record a telemetry event."""
        event = TelemetryEvent(
            event_id=self._generate_event_id(),
            event_type=event_type.value,
            timestamp=datetime.now().isoformat(),
            status=status.value,
            tool=tool,
            workflow=workflow,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        # Append to events file
        with open(self.events_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(event)) + '\n')

        return event

    def record_handoff(
        self,
        source_tool: str,
        target_tool: str,
        success: bool,
        duration_ms: Optional[int] = None,
        workflow: Optional[str] = None,
        correlation_id: Optional[str] = None,
        context_size_kb: Optional[float] = None
    ) -> TelemetryEvent:
        """Record a tool handoff event."""
        return self.record_event(
            event_type=EventType.HANDOFF,
            status=EventStatus.SUCCESS if success else EventStatus.FAILURE,
            tool=f"{source_tool}->{target_tool}",
            workflow=workflow,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata={
                "source_tool": source_tool,
                "target_tool": target_tool,
                "context_size_kb": context_size_kb
            }
        )

    def record_tool_invocation(
        self,
        tool: str,
        action: str,
        success: bool,
        duration_ms: int,
        workflow: Optional[str] = None,
        files_modified: int = 0
    ) -> TelemetryEvent:
        """Record a tool invocation event."""
        return self.record_event(
            event_type=EventType.TOOL_INVOCATION,
            status=EventStatus.SUCCESS if success else EventStatus.FAILURE,
            tool=tool,
            workflow=workflow,
            duration_ms=duration_ms,
            metadata={
                "action": action,
                "files_modified": files_modified
            }
        )

    def record_hallucination_check(
        self,
        passed: bool,
        confidence_score: float,
        tool: Optional[str] = None,
        flagged_claims: int = 0
    ) -> TelemetryEvent:
        """Record a hallucination check event."""
        return self.record_event(
            event_type=EventType.HALLUCINATION_CHECK,
            status=EventStatus.SUCCESS if passed else EventStatus.FAILURE,
            tool=tool,
            metadata={
                "confidence_score": confidence_score,
                "flagged_claims": flagged_claims
            }
        )

    def load_events(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        tool: Optional[str] = None
    ) -> List[TelemetryEvent]:
        """Load events with optional filtering."""
        events = []

        if not self.events_file.exists():
            return events

        with open(self.events_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    event_time = datetime.fromisoformat(data["timestamp"])

                    # Apply filters
                    if since and event_time < since:
                        continue
                    if until and event_time > until:
                        continue
                    if event_type and data["event_type"] != event_type.value:
                        continue
                    if tool and data.get("tool") != tool:
                        continue

                    events.append(TelemetryEvent(**data))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return events

    def generate_summary(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> MetricsSummary:
        """Generate metrics summary for a period."""
        if not since:
            since = datetime.now() - timedelta(hours=24)
        if not until:
            until = datetime.now()

        events = self.load_events(since=since, until=until)

        summary = MetricsSummary(
            period_start=since.isoformat(),
            period_end=until.isoformat(),
            total_events=len(events)
        )

        if not events:
            return summary

        # Aggregate by type, status, and tool
        durations = []
        handoffs_total = 0
        handoffs_success = 0
        workflows_started = 0
        workflows_completed = 0
        hallucination_checks = 0
        hallucinations_detected = 0

        for event in events:
            # By type
            summary.events_by_type[event.event_type] = \
                summary.events_by_type.get(event.event_type, 0) + 1

            # By status
            summary.events_by_status[event.status] = \
                summary.events_by_status.get(event.status, 0) + 1

            # By tool
            if event.tool:
                summary.events_by_tool[event.tool] = \
                    summary.events_by_tool.get(event.tool, 0) + 1

            # Collect durations
            if event.duration_ms is not None:
                durations.append(event.duration_ms)

            # Track handoff success
            if event.event_type == EventType.HANDOFF.value:
                handoffs_total += 1
                if event.status == EventStatus.SUCCESS.value:
                    handoffs_success += 1

            # Track workflow completion
            if event.event_type == EventType.WORKFLOW_START.value:
                workflows_started += 1
            if event.event_type == EventType.WORKFLOW_END.value:
                if event.status == EventStatus.SUCCESS.value:
                    workflows_completed += 1

            # Track hallucinations
            if event.event_type == EventType.HALLUCINATION_CHECK.value:
                hallucination_checks += 1
                if event.status == EventStatus.FAILURE.value:
                    hallucinations_detected += 1

        # Calculate duration percentiles
        if durations:
            durations.sort()
            summary.avg_duration_ms = statistics.mean(durations)
            summary.p50_duration_ms = self._percentile(durations, 50)
            summary.p95_duration_ms = self._percentile(durations, 95)
            summary.p99_duration_ms = self._percentile(durations, 99)

        # Calculate rates
        if handoffs_total > 0:
            summary.handoff_success_rate = handoffs_success / handoffs_total

        if workflows_started > 0:
            summary.workflow_completion_rate = workflows_completed / workflows_started

        if hallucination_checks > 0:
            summary.hallucination_rate = hallucinations_detected / hallucination_checks

        # Tool utilization
        for tool, count in summary.events_by_tool.items():
            tool_events = [e for e in events if e.tool == tool]
            tool_durations = [e.duration_ms for e in tool_events if e.duration_ms]
            tool_successes = sum(1 for e in tool_events if e.status == EventStatus.SUCCESS.value)

            summary.tool_utilization[tool] = {
                "invocations": count,
                "success_rate": tool_successes / count if count > 0 else 0,
                "avg_duration_ms": statistics.mean(tool_durations) if tool_durations else 0
            }

        return summary

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

    def check_targets(self, summary: MetricsSummary) -> Dict[str, Any]:
        """Check metrics against target thresholds."""
        targets = self.policy.get("metrics", {}).get("targets", {})

        results = {
            "all_targets_met": True,
            "checks": []
        }

        checks = [
            ("handoff_success_rate", summary.handoff_success_rate,
             targets.get("handoff_success_rate", 0.95), ">="),
            ("hallucination_rate", summary.hallucination_rate,
             targets.get("hallucination_rate", 0.02), "<="),
            ("workflow_completion", summary.workflow_completion_rate,
             targets.get("workflow_completion", 0.90), ">="),
        ]

        for name, actual, target, operator in checks:
            if operator == ">=":
                passed = actual >= target
            else:
                passed = actual <= target

            results["checks"].append({
                "name": name,
                "actual": actual,
                "target": target,
                "operator": operator,
                "passed": passed
            })

            if not passed:
                results["all_targets_met"] = False

        return results

    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Remove events older than retention period."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        kept_events = []
        removed_count = 0

        if not self.events_file.exists():
            return 0

        with open(self.events_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    event_time = datetime.fromisoformat(data["timestamp"])

                    if event_time >= cutoff:
                        kept_events.append(line)
                    else:
                        removed_count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        # Rewrite file with kept events
        with open(self.events_file, 'w', encoding='utf-8') as f:
            for line in kept_events:
                f.write(line + '\n')

        return removed_count


def format_summary(summary: MetricsSummary, fmt: str = "text") -> str:
    """Format metrics summary for output."""
    if fmt == "json":
        return json.dumps(asdict(summary), indent=2)

    # Text format
    lines = [
        "=" * 60,
        "ORCHESTRATION TELEMETRY REPORT",
        "=" * 60,
        "",
        f"Period: {summary.period_start[:19]} to {summary.period_end[:19]}",
        f"Total Events: {summary.total_events}",
        "",
        "-" * 40,
        "EVENTS BY TYPE",
        "-" * 40,
    ]

    for event_type, count in sorted(summary.events_by_type.items()):
        lines.append(f"  {event_type}: {count}")

    lines.extend([
        "",
        "-" * 40,
        "SUCCESS RATES",
        "-" * 40,
        f"  Handoff Success Rate: {summary.handoff_success_rate:.1%}",
        f"  Workflow Completion:  {summary.workflow_completion_rate:.1%}",
        f"  Hallucination Rate:   {summary.hallucination_rate:.1%}",
    ])

    lines.extend([
        "",
        "-" * 40,
        "PERFORMANCE",
        "-" * 40,
        f"  Average Duration:  {summary.avg_duration_ms:.0f}ms",
        f"  P50 Duration:      {summary.p50_duration_ms:.0f}ms",
        f"  P95 Duration:      {summary.p95_duration_ms:.0f}ms",
        f"  P99 Duration:      {summary.p99_duration_ms:.0f}ms",
    ])

    if summary.tool_utilization:
        lines.extend([
            "",
            "-" * 40,
            "TOOL UTILIZATION",
            "-" * 40,
        ])
        for tool, stats in sorted(summary.tool_utilization.items()):
            lines.append(
                f"  {tool}: {stats['invocations']} calls, "
                f"{stats['success_rate']:.0%} success, "
                f"{stats['avg_duration_ms']:.0f}ms avg"
            )

    lines.extend(["", "=" * 60])
    return '\n'.join(lines)


@click.group()
def cli():
    """Orchestration telemetry collection and reporting."""
    pass


@cli.command()
@click.option('--event', '-e', required=True,
              type=click.Choice(['handoff', 'tool_invocation', 'checkpoint',
                                'hallucination_check', 'error', 'recovery',
                                'workflow_start', 'workflow_end']),
              help='Event type')
@click.option('--status', '-s', required=True,
              type=click.Choice(['success', 'failure', 'partial', 'timeout', 'skipped']),
              help='Event status')
@click.option('--tool', '-t', help='Tool name')
@click.option('--workflow', '-w', help='Workflow name')
@click.option('--duration', '-d', type=int, help='Duration in milliseconds')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def record(event: str, status: str, tool: Optional[str], workflow: Optional[str],
           duration: Optional[int], json_output: bool):
    """Record a telemetry event."""
    try:
        telemetry = OrchestrationTelemetry()

        event_obj = telemetry.record_event(
            event_type=EventType(event),
            status=EventStatus(status),
            tool=tool,
            workflow=workflow,
            duration_ms=duration
        )

        if json_output:
            click.echo(json.dumps(asdict(event_obj), indent=2))
        else:
            click.echo(f"Event recorded: {event_obj.event_id}")
            click.echo(f"  Type: {event_obj.event_type}")
            click.echo(f"  Status: {event_obj.status}")
            if tool:
                click.echo(f"  Tool: {tool}")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--period', '-p', default='24h',
              help='Report period (e.g., 1h, 24h, 7d)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def report(period: str, json_output: bool):
    """Generate telemetry report."""
    try:
        # Parse period
        if period.endswith('h'):
            hours = int(period[:-1])
            since = datetime.now() - timedelta(hours=hours)
        elif period.endswith('d'):
            days = int(period[:-1])
            since = datetime.now() - timedelta(days=days)
        else:
            since = datetime.now() - timedelta(hours=24)

        telemetry = OrchestrationTelemetry()
        summary = telemetry.generate_summary(since=since)

        output = format_summary(summary, "json" if json_output else "text")
        click.echo(output)

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
def check_targets(json_output: bool):
    """Check metrics against target thresholds."""
    try:
        telemetry = OrchestrationTelemetry()
        summary = telemetry.generate_summary()
        results = telemetry.check_targets(summary)

        if json_output:
            click.echo(json.dumps(results, indent=2))
        else:
            status = "ALL TARGETS MET" if results["all_targets_met"] else "TARGETS NOT MET"
            click.echo(f"Status: {status}\n")

            for check in results["checks"]:
                icon = "[OK]" if check["passed"] else "[FAIL]"
                click.echo(
                    f"  {icon} {check['name']}: {check['actual']:.2%} "
                    f"{check['operator']} {check['target']:.2%}"
                )

        raise SystemExit(0 if results["all_targets_met"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--days', '-d', default=30, help='Retention period in days')
def cleanup(days: int):
    """Remove old telemetry events."""
    try:
        telemetry = OrchestrationTelemetry()
        removed = telemetry.cleanup_old_events(retention_days=days)

        click.echo(f"Removed {removed} events older than {days} days")
        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
