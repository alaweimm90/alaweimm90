"""
MEZAN Advanced Alerting System.

This module provides comprehensive alerting capabilities including:
- Alert rule DSL (YAML/JSON configuration)
- Multi-channel notifications (Email, Slack, PagerDuty, Webhook)
- Alert aggregation and deduplication
- On-call rotation management
- Alert silencing and acknowledgment
- Rate limiting to prevent alert storms
- Alert correlation and grouping
"""

import json
import yaml
import time
import threading
import logging
import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union
from enum import Enum
from pathlib import Path
import schedule
import jinja2

from .telemetry import (
    Alert, AlertRule, AlertState, AlertSeverity,
    AlertChannel, EmailAlertChannel, SlackAlertChannel,
    PagerDutyAlertChannel, WebhookAlertChannel
)

logger = logging.getLogger(__name__)


@dataclass
class OnCallSchedule:
    """On-call rotation schedule."""
    name: str
    members: List[str]
    rotation_type: str  # 'daily', 'weekly', 'custom'
    current_index: int = 0
    rotation_interval_hours: int = 168  # Weekly default
    last_rotation: float = field(default_factory=time.time)
    overrides: Dict[str, str] = field(default_factory=dict)  # timestamp -> member


@dataclass
class AlertGroup:
    """Group of related alerts."""
    group_id: str
    key: str
    alerts: List[Alert]
    first_alert_time: float
    last_alert_time: float
    total_count: int
    status: AlertState
    labels: Dict[str, str]


@dataclass
class AlertCorrelation:
    """Alert correlation configuration."""
    name: str
    patterns: List[str]  # Regex patterns for correlation
    time_window: int  # Seconds
    min_alerts: int
    action: str  # 'group', 'suppress', 'escalate'
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertRuleEngine:
    """Advanced alert rule evaluation engine."""

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.rule_templates: Dict[str, dict] = {}
        self.jinja_env = jinja2.Environment()
        self._lock = threading.RLock()

    def load_rules_from_file(self, file_path: str):
        """Load alert rules from YAML/JSON file."""
        path = Path(file_path)

        if path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(file_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Load templates if defined
        if 'templates' in config:
            for name, template in config['templates'].items():
                self.rule_templates[name] = template

        # Process rules
        for rule_config in config.get('rules', []):
            rule = self._process_rule_config(rule_config)
            self.add_rule(rule)

        logger.info(f"Loaded {len(config.get('rules', []))} rules from {file_path}")

    def _process_rule_config(self, config: dict) -> AlertRule:
        """Process rule configuration with template support."""
        # Apply template if specified
        if 'template' in config:
            template_name = config['template']
            if template_name in self.rule_templates:
                template = self.rule_templates[template_name].copy()
                template.update(config)
                config = template

        # Process expressions
        if 'expression' in config:
            config['expression'] = self._compile_expression(config['expression'])

        # Convert string severity to enum if needed
        if 'severity' in config and isinstance(config['severity'], str):
            config['severity'] = config['severity'].lower()

        return AlertRule(**config)

    def _compile_expression(self, expr: str) -> str:
        """Compile DSL expression to Python code."""
        # Simple DSL to Python translation
        # Examples:
        # "metric > 90 AND duration > 5m" -> "value > 90 and history_duration > 300"
        # "rate(metric[5m]) > 0.1" -> "calculate_rate(history, 300) > 0.1"

        expr = expr.replace(" AND ", " and ")
        expr = expr.replace(" OR ", " or ")
        expr = expr.replace(" NOT ", " not ")

        # Handle duration units
        expr = re.sub(r'(\d+)m', lambda m: str(int(m.group(1)) * 60), expr)
        expr = re.sub(r'(\d+)h', lambda m: str(int(m.group(1)) * 3600), expr)
        expr = re.sub(r'(\d+)d', lambda m: str(int(m.group(1)) * 86400), expr)

        # Handle rate function
        expr = re.sub(
            r'rate\((\w+)\[(\d+)\]\)',
            r'calculate_rate(history, \2)',
            expr
        )

        # Handle percentile function
        expr = re.sub(
            r'p(\d+)\((\w+)\[(\d+)\]\)',
            r'calculate_percentile(history, \1, \3)',
            expr
        )

        return expr

    def add_rule(self, rule: AlertRule):
        """Add alert rule to engine."""
        with self._lock:
            self.rules.append(rule)

    def evaluate(self, metrics: Dict[str, float],
                history: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> List[AlertRule]:
        """Evaluate all rules against metrics."""
        triggered_rules = []

        with self._lock:
            for rule in self.rules:
                if rule.metric_name in metrics:
                    value = metrics[rule.metric_name]
                    metric_history = history.get(rule.metric_name, []) if history else []

                    if rule.evaluate(value, metric_history):
                        triggered_rules.append(rule)

        return triggered_rules

    def get_rules_by_severity(self, severity: str) -> List[AlertRule]:
        """Get rules by severity level."""
        with self._lock:
            return [r for r in self.rules if r.severity == severity]


class AlertGroupManager:
    """Manages alert grouping and correlation."""

    def __init__(self, group_interval: int = 300, group_wait: int = 30):
        self.groups: Dict[str, AlertGroup] = {}
        self.correlations: List[AlertCorrelation] = []
        self.group_interval = group_interval
        self.group_wait = group_wait
        self._lock = threading.RLock()
        self._pending_groups: Dict[str, List[Alert]] = defaultdict(list)

    def add_correlation(self, correlation: AlertCorrelation):
        """Add alert correlation configuration."""
        with self._lock:
            self.correlations.append(correlation)

    def group_alert(self, alert: Alert) -> Optional[AlertGroup]:
        """Group alert with similar alerts."""
        # Generate grouping key
        group_key = self._generate_group_key(alert)

        with self._lock:
            # Check for existing group
            if group_key in self.groups:
                group = self.groups[group_key]

                # Check if group is still active (within interval)
                if time.time() - group.last_alert_time <= self.group_interval:
                    group.alerts.append(alert)
                    group.last_alert_time = time.time()
                    group.total_count += 1
                    return group
                else:
                    # Create new group
                    del self.groups[group_key]

            # Create new group
            group_id = hashlib.md5(f"{group_key}_{time.time()}".encode()).hexdigest()[:8]
            group = AlertGroup(
                group_id=group_id,
                key=group_key,
                alerts=[alert],
                first_alert_time=time.time(),
                last_alert_time=time.time(),
                total_count=1,
                status=alert.state,
                labels=alert.labels
            )
            self.groups[group_key] = group

            return group

    def _generate_group_key(self, alert: Alert) -> str:
        """Generate grouping key for alert."""
        # Group by metric, severity, and grouping keys
        parts = [
            alert.rule.metric_name,
            alert.rule.severity,
            str(alert.rule.condition)
        ]

        for key in alert.rule.grouping_keys:
            if key in alert.labels:
                parts.append(f"{key}:{alert.labels[key]}")

        return "|".join(parts)

    def correlate_alerts(self, alerts: List[Alert]) -> List[Tuple[AlertCorrelation, List[Alert]]]:
        """Find correlated alerts."""
        correlations = []

        with self._lock:
            for correlation in self.correlations:
                correlated = self._find_correlated_alerts(alerts, correlation)
                if correlated:
                    correlations.append((correlation, correlated))

        return correlations

    def _find_correlated_alerts(self, alerts: List[Alert],
                                correlation: AlertCorrelation) -> List[Alert]:
        """Find alerts matching correlation patterns."""
        current_time = time.time()
        matched = []

        for alert in alerts:
            # Check time window
            if current_time - alert.start_time > correlation.time_window:
                continue

            # Check patterns
            alert_str = f"{alert.rule.metric_name} {json.dumps(alert.labels)}"
            for pattern in correlation.patterns:
                if re.match(pattern, alert_str):
                    matched.append(alert)
                    break

        # Check minimum alerts threshold
        if len(matched) >= correlation.min_alerts:
            return matched

        return []


class OnCallManager:
    """Manages on-call rotation schedules."""

    def __init__(self):
        self.schedules: Dict[str, OnCallSchedule] = {}
        self._lock = threading.RLock()
        self._scheduler_thread = threading.Thread(target=self._rotation_scheduler, daemon=True)
        self._scheduler_thread.start()

    def add_schedule(self, schedule: OnCallSchedule):
        """Add on-call schedule."""
        with self._lock:
            self.schedules[schedule.name] = schedule
            logger.info(f"Added on-call schedule: {schedule.name}")

    def get_current_on_call(self, schedule_name: str) -> Optional[str]:
        """Get current on-call person for schedule."""
        with self._lock:
            if schedule_name not in self.schedules:
                return None

            schedule = self.schedules[schedule_name]

            # Check for overrides
            current_time = time.time()
            for override_time, member in schedule.overrides.items():
                if float(override_time) <= current_time <= float(override_time) + 86400:
                    return member

            # Return regular rotation member
            if schedule.members:
                return schedule.members[schedule.current_index % len(schedule.members)]

            return None

    def override_on_call(self, schedule_name: str, member: str, duration_hours: int = 24):
        """Override on-call for specified duration."""
        with self._lock:
            if schedule_name not in self.schedules:
                return

            schedule = self.schedules[schedule_name]
            override_time = str(time.time())
            schedule.overrides[override_time] = member

            logger.info(f"On-call override: {member} for {schedule_name} ({duration_hours}h)")

    def _rotation_scheduler(self):
        """Background thread for rotation scheduling."""
        while True:
            try:
                current_time = time.time()

                with self._lock:
                    for schedule in self.schedules.values():
                        # Check if rotation is needed
                        time_since_rotation = current_time - schedule.last_rotation
                        rotation_seconds = schedule.rotation_interval_hours * 3600

                        if time_since_rotation >= rotation_seconds:
                            # Rotate to next person
                            schedule.current_index = (schedule.current_index + 1) % len(schedule.members)
                            schedule.last_rotation = current_time

                            logger.info(f"Rotated on-call for {schedule.name} to {schedule.members[schedule.current_index]}")

            except Exception as e:
                logger.error(f"On-call rotation error: {e}")

            time.sleep(3600)  # Check hourly


class AlertManager:
    """Enhanced alert manager with advanced features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rule_engine = AlertRuleEngine()
        self.group_manager = AlertGroupManager()
        self.oncall_manager = OnCallManager()

        # Alert channels
        self.channels: Dict[str, AlertChannel] = {}

        # Alert state
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history = deque(maxlen=10000)
        self._silences: Dict[str, float] = {}
        self._acknowledgments: Dict[str, Dict] = {}

        # Rate limiting
        self.rate_limiter = defaultdict(lambda: deque(maxlen=10))
        self.rate_limit_window = self.config.get('rate_limit_window', 60)
        self.max_alerts_per_window = self.config.get('max_alerts_per_window', 10)

        # Locks
        self._lock = threading.RLock()

        # Initialize from config
        self._init_from_config()

    def _init_from_config(self):
        """Initialize from configuration."""
        if 'rules_file' in self.config:
            self.rule_engine.load_rules_from_file(self.config['rules_file'])

        if 'channels' in self.config:
            self._setup_channels(self.config['channels'])

        if 'oncall_schedules' in self.config:
            self._setup_oncall_schedules(self.config['oncall_schedules'])

    def _setup_channels(self, channels_config: Dict):
        """Setup notification channels from config."""
        for name, config in channels_config.items():
            channel_type = config.get('type')

            if channel_type == 'email':
                channel = EmailAlertChannel(
                    smtp_host=config['smtp_host'],
                    smtp_port=config['smtp_port'],
                    username=config['username'],
                    password=config['password'],
                    from_addr=config['from_addr'],
                    to_addrs=config['to_addrs']
                )
            elif channel_type == 'slack':
                channel = SlackAlertChannel(
                    webhook_url=config['webhook_url'],
                    channel=config.get('channel')
                )
            elif channel_type == 'pagerduty':
                channel = PagerDutyAlertChannel(
                    integration_key=config['integration_key']
                )
            elif channel_type == 'webhook':
                channel = WebhookAlertChannel(
                    url=config['url'],
                    headers=config.get('headers', {}),
                    method=config.get('method', 'POST')
                )
            else:
                logger.warning(f"Unknown channel type: {channel_type}")
                continue

            self.add_channel(name, channel)

    def _setup_oncall_schedules(self, schedules_config: List[Dict]):
        """Setup on-call schedules from config."""
        for config in schedules_config:
            schedule = OnCallSchedule(
                name=config['name'],
                members=config['members'],
                rotation_type=config.get('rotation_type', 'weekly'),
                rotation_interval_hours=config.get('rotation_interval_hours', 168)
            )
            self.oncall_manager.add_schedule(schedule)

    def add_channel(self, name: str, channel: AlertChannel):
        """Add notification channel."""
        self.channels[name] = channel
        logger.info(f"Added channel: {name}")

    def process_metrics(self, metrics: Dict[str, float],
                       history: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                       labels: Optional[Dict[str, str]] = None):
        """Process metrics and generate alerts."""
        labels = labels or {}

        # Evaluate rules
        triggered_rules = self.rule_engine.evaluate(metrics, history)

        for rule in triggered_rules:
            # Create alert
            alert = self._create_alert(rule, metrics[rule.metric_name], labels)

            # Group alert
            group = self.group_manager.group_alert(alert)

            # Check rate limiting
            if not self._should_rate_limit(alert):
                # Send notifications
                self._notify(alert, group)

    def _create_alert(self, rule: AlertRule, value: float, labels: Dict[str, str]) -> Alert:
        """Create alert from triggered rule."""
        # Generate fingerprint
        fingerprint = self._generate_fingerprint(rule, labels)

        # Check if alert already exists
        with self._lock:
            if fingerprint in self._active_alerts:
                # Update existing alert
                alert = self._active_alerts[fingerprint]
                alert.value = value
                return alert

            # Create new alert
            alert = Alert(
                rule=rule,
                value=value,
                state=AlertState.FIRING,
                start_time=time.time(),
                last_notification=0,
                fingerprint=fingerprint,
                labels={**labels, **rule.labels},
                annotations={**rule.annotations, "value": str(value)}
            )

            self._active_alerts[fingerprint] = alert
            self._alert_history.append(alert)

            return alert

    def _generate_fingerprint(self, rule: AlertRule, labels: Dict[str, str]) -> str:
        """Generate unique alert fingerprint."""
        parts = [
            rule.metric_name,
            rule.condition,
            str(rule.threshold),
            json.dumps(labels, sort_keys=True)
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _should_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        current_time = time.time()
        rate_queue = self.rate_limiter[alert.fingerprint]

        # Clean old entries
        while rate_queue and rate_queue[0] < current_time - self.rate_limit_window:
            rate_queue.popleft()

        # Check limit
        if len(rate_queue) >= self.max_alerts_per_window:
            logger.warning(f"Rate limited: {alert.fingerprint}")
            return True

        rate_queue.append(current_time)
        return False

    def _notify(self, alert: Alert, group: Optional[AlertGroup] = None):
        """Send alert notifications."""
        # Check if silenced
        if alert.fingerprint in self._silences:
            if time.time() < self._silences[alert.fingerprint]:
                return

        # Check if acknowledged
        if alert.fingerprint in self._acknowledgments:
            return

        # Determine channels based on severity
        channels_to_use = self._get_channels_for_severity(alert.rule.severity)

        # Add on-call info for critical alerts
        if alert.rule.severity == 'critical':
            oncall = self._get_oncall_info()
            if oncall:
                alert.annotations['oncall'] = oncall

        # Send to channels
        for channel_name in channels_to_use:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    if channel.send(alert):
                        logger.info(f"Alert sent to {channel_name}: {alert.fingerprint}")
                    else:
                        logger.error(f"Failed to send alert to {channel_name}")
                except Exception as e:
                    logger.error(f"Error sending alert to {channel_name}: {e}")

        alert.last_notification = time.time()
        alert.notification_count += 1

    def _get_channels_for_severity(self, severity: str) -> List[str]:
        """Get channels to notify based on severity."""
        severity_mapping = self.config.get('severity_channels', {
            'critical': ['pagerduty', 'slack', 'email'],
            'error': ['slack', 'email'],
            'warning': ['slack'],
            'info': ['slack']
        })
        return severity_mapping.get(severity, ['slack'])

    def _get_oncall_info(self) -> Optional[str]:
        """Get current on-call information."""
        # Get primary on-call schedule
        primary_schedule = self.config.get('primary_oncall_schedule', 'primary')
        oncall_person = self.oncall_manager.get_current_on_call(primary_schedule)
        return oncall_person

    def acknowledge_alert(self, fingerprint: str, acknowledged_by: str, notes: str = ""):
        """Acknowledge an alert."""
        with self._lock:
            if fingerprint in self._active_alerts:
                alert = self._active_alerts[fingerprint]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()

                self._acknowledgments[fingerprint] = {
                    'by': acknowledged_by,
                    'at': time.time(),
                    'notes': notes
                }

                logger.info(f"Alert acknowledged: {fingerprint} by {acknowledged_by}")

    def silence_alert(self, fingerprint: str, duration_hours: int, silenced_by: str, reason: str = ""):
        """Silence an alert for specified duration."""
        silence_until = time.time() + (duration_hours * 3600)

        with self._lock:
            self._silences[fingerprint] = silence_until

            if fingerprint in self._active_alerts:
                self._active_alerts[fingerprint].silence_until = silence_until

            logger.info(f"Alert silenced: {fingerprint} until {datetime.fromtimestamp(silence_until)} by {silenced_by}")

    def resolve_alert(self, fingerprint: str):
        """Resolve an alert."""
        with self._lock:
            if fingerprint in self._active_alerts:
                alert = self._active_alerts[fingerprint]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = time.time()

                # Notify resolution
                for channel_name, channel in self.channels.items():
                    try:
                        channel.send(alert)
                    except:
                        pass

                del self._active_alerts[fingerprint]
                logger.info(f"Alert resolved: {fingerprint}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            active = list(self._active_alerts.values())

            stats = {
                'total_active': len(active),
                'by_severity': defaultdict(int),
                'by_state': defaultdict(int),
                'acknowledged': len(self._acknowledgments),
                'silenced': len([s for s in self._silences.values() if s > time.time()]),
                'groups': len(self.group_manager.groups)
            }

            for alert in active:
                stats['by_severity'][alert.rule.severity] += 1
                stats['by_state'][alert.state.value] += 1

            return dict(stats)


# Example configuration
def create_example_config() -> Dict[str, Any]:
    """Create example alert manager configuration."""
    return {
        'rules_file': 'alert_rules.yaml',
        'rate_limit_window': 60,
        'max_alerts_per_window': 10,
        'channels': {
            'slack': {
                'type': 'slack',
                'webhook_url': 'https://hooks.slack.com/services/xxx'
            },
            'pagerduty': {
                'type': 'pagerduty',
                'integration_key': 'xxx'
            },
            'email': {
                'type': 'email',
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'alerts@example.com',
                'password': 'xxx',
                'from_addr': 'alerts@example.com',
                'to_addrs': ['oncall@example.com']
            },
            'webhook': {
                'type': 'webhook',
                'url': 'https://api.example.com/alerts',
                'headers': {'Authorization': 'Bearer xxx'}
            }
        },
        'severity_channels': {
            'critical': ['pagerduty', 'slack', 'email'],
            'error': ['slack', 'email'],
            'warning': ['slack'],
            'info': []
        },
        'oncall_schedules': [
            {
                'name': 'primary',
                'members': ['alice', 'bob', 'charlie'],
                'rotation_type': 'weekly',
                'rotation_interval_hours': 168
            }
        ],
        'primary_oncall_schedule': 'primary'
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create alert manager
    config = create_example_config()
    manager = AlertManager(config)

    # Process some metrics
    metrics = {
        'cpu_usage': 95.0,
        'memory_usage': 87.0,
        'disk_usage': 45.0,
        'error_rate': 0.05
    }

    manager.process_metrics(metrics, labels={'service': 'mezan', 'environment': 'prod'})

    # Get stats
    stats = manager.get_alert_stats()
    print(f"Alert stats: {json.dumps(stats, indent=2)}")