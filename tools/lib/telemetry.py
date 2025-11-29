#!/usr/bin/env python3
"""
telemetry.py - Unified Telemetry Library

Provides telemetry collection and reporting for workflows and operations.
Consolidates logic from:
- tools/orchestration/orchestration_telemetry.py
- tools/meta/telemetry_dashboard.py

Usage:
    from tools.lib.telemetry import Telemetry
    
    telemetry = Telemetry()
    telemetry.record_event("handoff", "success", metadata={"tool": "cline"})
    metrics = telemetry.get_metrics(workflow="feature-x")
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics


class EventType(Enum):
    """Types of telemetry events."""
    HANDOFF = "handoff"
    TOOL_INVOCATION = "tool_invocation"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
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
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Telemetry:
    """Unified telemetry collection and reporting."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize telemetry system.
        
        Args:
            base_path: Base path for telemetry storage
        """
        self.base_path = base_path or self._find_base_path()
        self.telemetry_dir = self.base_path / ".metaHub/telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        self.events_file = self.telemetry_dir / "events.jsonl"
    
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
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def record_event(
        self,
        event_type: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a telemetry event.
        
        Args:
            event_type: Type of event (handoff, tool_invocation, etc.)
            status: Status of event (success, failure, etc.)
            metadata: Additional event metadata
            
        Returns:
            Event ID
        """
        event_id = self._generate_event_id()
        
        event = TelemetryEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            status=status,
            tool=metadata.get("tool") if metadata else None,
            workflow=metadata.get("workflow") if metadata else None,
            duration_ms=metadata.get("duration_ms") if metadata else None,
            metadata=metadata or {}
        )
        
        # Append to events file
        with open(self.events_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(event)) + '\n')
        
        return event_id
    
    def get_metrics(
        self,
        workflow: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> dict:
        """
        Get aggregated metrics.
        
        Args:
            workflow: Optional workflow filter
            since: Start time for metrics
            until: End time for metrics
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not since:
            since = datetime.now() - timedelta(hours=24)
        if not until:
            until = datetime.now()
        
        events = self._load_events(workflow=workflow, since=since, until=until)
        
        if not events:
            return {
                "period_start": since.isoformat(),
                "period_end": until.isoformat(),
                "total_events": 0,
                "events_by_type": {},
                "events_by_status": {},
                "success_rate": 0.0,
                "avg_duration_ms": 0.0
            }
        
        # Aggregate metrics
        events_by_type = {}
        events_by_status = {}
        durations = []
        success_count = 0
        
        for event in events:
            # By type
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            
            # By status
            events_by_status[event.status] = events_by_status.get(event.status, 0) + 1
            
            # Success tracking
            if event.status == EventStatus.SUCCESS.value:
                success_count += 1
            
            # Duration tracking
            if event.duration_ms is not None:
                durations.append(event.duration_ms)
        
        return {
            "period_start": since.isoformat(),
            "period_end": until.isoformat(),
            "total_events": len(events),
            "events_by_type": events_by_type,
            "events_by_status": events_by_status,
            "success_rate": success_count / len(events) if events else 0.0,
            "avg_duration_ms": statistics.mean(durations) if durations else 0.0,
            "p50_duration_ms": self._percentile(durations, 50) if durations else 0.0,
            "p95_duration_ms": self._percentile(durations, 95) if durations else 0.0,
            "p99_duration_ms": self._percentile(durations, 99) if durations else 0.0
        }
    
    def _load_events(
        self,
        workflow: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
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
                    if workflow and data.get("workflow") != workflow:
                        continue
                    
                    events.append(TelemetryEvent(**data))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return events
    
    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]) if c != f else sorted_data[f]
    
    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """
        Remove events older than retention period.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of events removed
        """
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
    
    def generate_report(
        self,
        workflow: Optional[str] = None,
        period_hours: int = 24
    ) -> str:
        """
        Generate a text report of metrics.
        
        Args:
            workflow: Optional workflow filter
            period_hours: Hours to include in report
            
        Returns:
            Formatted text report
        """
        since = datetime.now() - timedelta(hours=period_hours)
        metrics = self.get_metrics(workflow=workflow, since=since)
        
        lines = [
            "=" * 60,
            "TELEMETRY REPORT",
            "=" * 60,
            "",
            f"Period: Last {period_hours} hours",
            f"Total Events: {metrics['total_events']}",
            f"Success Rate: {metrics['success_rate']:.1%}",
            "",
            "-" * 40,
            "EVENTS BY TYPE",
            "-" * 40,
        ]
        
        for event_type, count in sorted(metrics['events_by_type'].items()):
            lines.append(f"  {event_type}: {count}")
        
        lines.extend([
            "",
            "-" * 40,
            "PERFORMANCE",
            "-" * 40,
            f"  Average Duration: {metrics['avg_duration_ms']:.0f}ms",
            f"  P50 Duration: {metrics['p50_duration_ms']:.0f}ms",
            f"  P95 Duration: {metrics['p95_duration_ms']:.0f}ms",
            f"  P99 Duration: {metrics['p99_duration_ms']:.0f}ms",
            "",
            "=" * 60
        ])
        
        return '\n'.join(lines)