"""
MEZAN Performance Monitoring and Telemetry System.

This module provides comprehensive observability capabilities including:
- Prometheus-compatible metrics collection
- Performance tracking (latency, throughput, errors)
- Resource monitoring (CPU, memory, disk, network)
- Custom metrics with histograms and summaries
- Real-time anomaly detection
- Alert management and thresholds
- Time-series data storage and export
"""

import json
import psutil
import threading
import time
import logging
import re
import yaml
import smtplib
import requests
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported by the telemetry system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class AlertState(Enum):
    """Alert states."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Configuration for metric alert thresholds."""
    metric_name: str
    threshold: float
    condition: str  # 'above', 'below', 'equals', 'range', 'rate'
    duration_seconds: int = 60
    callback: Optional[Callable] = None
    severity: str = "warning"  # 'info', 'warning', 'critical'
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    grouping_keys: List[str] = field(default_factory=list)
    repeat_interval: int = 3600  # seconds
    mute_duration: int = 0  # seconds
    threshold_low: Optional[float] = None  # for range condition
    rate_window: int = 60  # seconds for rate calculation
    anomaly_detection: bool = False
    expression: Optional[str] = None  # custom expression

    def evaluate(self, value: float, history: Optional[List[Tuple[float, float]]] = None) -> bool:
        """Evaluate if rule is triggered."""
        if self.condition == 'above':
            return value > self.threshold
        elif self.condition == 'below':
            return value < self.threshold
        elif self.condition == 'equals':
            return abs(value - self.threshold) < 0.001
        elif self.condition == 'range':
            return self.threshold_low <= value <= self.threshold
        elif self.condition == 'rate' and history:
            # Calculate rate of change
            if len(history) < 2:
                return False
            recent = history[-min(10, len(history)):]
            rates = [(recent[i+1][1] - recent[i][1]) / (recent[i+1][0] - recent[i][0])
                    for i in range(len(recent)-1)]
            avg_rate = sum(rates) / len(rates) if rates else 0
            return abs(avg_rate) > self.threshold
        elif self.expression:
            # Custom expression evaluation
            try:
                return eval(self.expression, {"value": value, "threshold": self.threshold})
            except:
                return False
        return False


@dataclass
class Alert:
    """Active alert instance."""
    rule: AlertRule
    value: float
    state: AlertState
    start_time: float
    last_notification: float
    fingerprint: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    notification_count: int = 0
    silence_until: Optional[float] = None


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of a metric."""
    name: str
    type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Abstract base class for all metric types."""

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._lock = threading.RLock()
        self._last_updated = time.time()

    @abstractmethod
    def get_value(self) -> float:
        """Get current metric value."""
        pass

    @abstractmethod
    def to_prometheus(self) -> str:
        """Export metric in Prometheus format."""
        pass


class Counter(Metric):
    """Monotonically increasing counter metric."""

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(name, description, labels)
        self._value = 0.0

    def inc(self, value: float = 1.0):
        """Increment counter by value."""
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")
        with self._lock:
            self._value += value
            self._last_updated = time.time()

    def get_value(self) -> float:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        labels_part = f"{{{labels_str}}}" if labels_str else ""
        return f"# HELP {self.name} {self.description}\n# TYPE {self.name} counter\n{self.name}{labels_part} {self._value}"


class Gauge(Metric):
    """Gauge metric that can go up and down."""

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(name, description, labels)
        self._value = 0.0

    def set(self, value: float):
        """Set gauge to specific value."""
        with self._lock:
            self._value = value
            self._last_updated = time.time()

    def inc(self, value: float = 1.0):
        """Increment gauge by value."""
        with self._lock:
            self._value += value
            self._last_updated = time.time()

    def dec(self, value: float = 1.0):
        """Decrement gauge by value."""
        with self._lock:
            self._value -= value
            self._last_updated = time.time()

    def get_value(self) -> float:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        labels_part = f"{{{labels_str}}}" if labels_str else ""
        return f"# HELP {self.name} {self.description}\n# TYPE {self.name} gauge\n{self.name}{labels_part} {self._value}"


class Histogram(Metric):
    """Histogram metric for tracking value distributions."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None,
                 buckets: Optional[List[float]] = None):
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._observations = []
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        """Record an observation."""
        with self._lock:
            self._observations.append(value)
            self._sum += value
            self._count += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._last_updated = time.time()

    def get_value(self) -> float:
        """Return mean value."""
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0

    def get_percentile(self, percentile: float) -> float:
        """Calculate percentile value."""
        with self._lock:
            if not self._observations:
                return 0.0
            sorted_obs = sorted(self._observations)
            index = int(len(sorted_obs) * percentile / 100.0)
            return sorted_obs[min(index, len(sorted_obs) - 1)]

    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        result = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]

        for bucket, count in self._bucket_counts.items():
            bucket_labels = f'{labels_str},le="{bucket}"' if labels_str else f'le="{bucket}"'
            result.append(f"{self.name}_bucket{{{bucket_labels}}} {count}")

        labels_part = f"{{{labels_str}}}" if labels_str else ""
        result.append(f"{self.name}_sum{labels_part} {self._sum}")
        result.append(f"{self.name}_count{labels_part} {self._count}")

        return "\n".join(result)


class ResourceMonitor:
    """System resource monitoring capabilities."""

    def __init__(self, interval_seconds: int = 10):
        self.interval = interval_seconds
        self._running = False
        self._thread = None
        self._metrics = {}
        self._history = defaultdict(lambda: deque(maxlen=1000))

    def start(self):
        """Start resource monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started")

    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                timestamp = time.time()

                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()

                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()

                # Network metrics
                net_io = psutil.net_io_counters()

                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()

                # Store metrics
                metrics = {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_used_gb': memory.used / (1024**3),
                    'swap_percent': swap.percent,
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                    'network_sent_mb': net_io.bytes_sent / (1024**2),
                    'network_recv_mb': net_io.bytes_recv / (1024**2),
                    'process_memory_mb': process_memory.rss / (1024**2),
                    'process_cpu_percent': process_cpu
                }

                self._metrics = metrics

                # Store history
                for key, value in metrics.items():
                    self._history[key].append((timestamp, value))

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

            time.sleep(self.interval)

    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        return self._metrics.copy()

    def get_history(self, metric_name: str, duration_seconds: int = 300) -> List[Tuple[float, float]]:
        """Get metric history for specified duration."""
        if metric_name not in self._history:
            return []

        cutoff_time = time.time() - duration_seconds
        return [(t, v) for t, v in self._history[metric_name] if t >= cutoff_time]


class AnomalyDetector:
    """Real-time anomaly detection for metrics."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._windows = defaultdict(lambda: deque(maxlen=window_size))
        self._anomalies = []

    def update(self, metric_name: str, value: float) -> bool:
        """Update detector with new value and check for anomaly."""
        window = self._windows[metric_name]
        window.append(value)

        if len(window) < 10:  # Need minimum samples
            return False

        # Calculate z-score
        values = np.array(window)
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return False

        z_score = abs((value - mean) / std)

        if z_score > self.z_threshold:
            anomaly = {
                'metric': metric_name,
                'value': value,
                'mean': mean,
                'std': std,
                'z_score': z_score,
                'timestamp': time.time()
            }
            self._anomalies.append(anomaly)
            logger.warning(f"Anomaly detected: {anomaly}")
            return True

        return False

    def get_anomalies(self, since: Optional[float] = None) -> List[Dict]:
        """Get detected anomalies since timestamp."""
        if since is None:
            return self._anomalies.copy()
        return [a for a in self._anomalies if a['timestamp'] >= since]


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test channel connectivity."""
        pass


class EmailAlertChannel(AlertChannel):
    """Email notification channel."""

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_addr: str, to_addrs: List[str], use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{alert.rule.severity.upper()}] Alert: {alert.rule.metric_name}"

            body = f"""
Alert Details:
--------------
Metric: {alert.rule.metric_name}
Condition: {alert.rule.condition} {alert.rule.threshold}
Current Value: {alert.value}
Severity: {alert.rule.severity}
Start Time: {datetime.fromtimestamp(alert.start_time)}
State: {alert.state.value}

Labels: {json.dumps(alert.labels, indent=2)}
Annotations: {json.dumps(alert.annotations, indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test SMTP connectivity."""
        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.quit()
            return True
        except:
            return False


class SlackAlertChannel(AlertChannel):
    """Slack notification channel."""

    def __init__(self, webhook_url: str, channel: Optional[str] = None,
                 username: str = "MEZAN Alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            color_map = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good',
                'error': 'danger'
            }

            payload = {
                "username": self.username,
                "attachments": [{
                    "color": color_map.get(alert.rule.severity, 'warning'),
                    "title": f"Alert: {alert.rule.metric_name}",
                    "fields": [
                        {"title": "Condition", "value": f"{alert.rule.condition} {alert.rule.threshold}", "short": True},
                        {"title": "Current Value", "value": str(alert.value), "short": True},
                        {"title": "Severity", "value": alert.rule.severity.upper(), "short": True},
                        {"title": "State", "value": alert.state.value, "short": True}
                    ],
                    "footer": "MEZAN Monitoring",
                    "ts": int(alert.start_time)
                }]
            }

            if self.channel:
                payload["channel"] = self.channel

            response = requests.post(self.webhook_url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Slack webhook connectivity."""
        try:
            payload = {"text": "Test connection from MEZAN monitoring"}
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            return response.status_code == 200
        except:
            return False


class PagerDutyAlertChannel(AlertChannel):
    """PagerDuty notification channel."""

    def __init__(self, integration_key: str, api_url: str = "https://events.pagerduty.com/v2/enqueue"):
        self.integration_key = integration_key
        self.api_url = api_url

    def send(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        try:
            severity_map = {
                'critical': 'critical',
                'error': 'error',
                'warning': 'warning',
                'info': 'info'
            }

            payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger" if alert.state == AlertState.FIRING else "resolve",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"{alert.rule.metric_name}: {alert.rule.condition} {alert.rule.threshold}",
                    "severity": severity_map.get(alert.rule.severity, 'warning'),
                    "source": "MEZAN",
                    "component": alert.rule.metric_name,
                    "custom_details": {
                        "value": alert.value,
                        "labels": alert.labels,
                        "annotations": alert.annotations
                    }
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=5)
            return response.status_code == 202
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test PagerDuty API connectivity."""
        try:
            # Send a test event that auto-resolves
            test_payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": f"test-{time.time()}",
                "payload": {
                    "summary": "MEZAN test alert",
                    "severity": "info",
                    "source": "MEZAN"
                }
            }
            response = requests.post(self.api_url, json=test_payload, timeout=5)

            if response.status_code == 202:
                # Auto-resolve the test alert
                resolve_payload = test_payload.copy()
                resolve_payload["event_action"] = "resolve"
                requests.post(self.api_url, json=resolve_payload, timeout=5)

            return response.status_code == 202
        except:
            return False


class WebhookAlertChannel(AlertChannel):
    """Generic webhook notification channel."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 method: str = "POST", timeout: int = 10):
        self.url = url
        self.headers = headers or {}
        self.method = method
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            payload = {
                "alert": {
                    "metric": alert.rule.metric_name,
                    "condition": f"{alert.rule.condition} {alert.rule.threshold}",
                    "value": alert.value,
                    "severity": alert.rule.severity,
                    "state": alert.state.value,
                    "start_time": alert.start_time,
                    "fingerprint": alert.fingerprint,
                    "labels": alert.labels,
                    "annotations": alert.annotations
                }
            }

            response = requests.request(
                self.method, self.url, json=payload,
                headers=self.headers, timeout=self.timeout
            )
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test webhook connectivity."""
        try:
            response = requests.request(
                "GET" if self.method == "GET" else self.method,
                self.url, headers=self.headers, timeout=5
            )
            return response.status_code < 500
        except:
            return False


class AlertManager:
    """Enhanced alert management with multiple channels and deduplication."""

    def __init__(self, rate_limit_window: int = 60, max_alerts_per_window: int = 10):
        self.rules = []
        self.channels: Dict[str, AlertChannel] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history = deque(maxlen=10000)
        self._silences: Dict[str, float] = {}  # fingerprint -> silence_until
        self._lock = threading.RLock()
        self._rate_limiter = defaultdict(lambda: deque(maxlen=max_alerts_per_window))
        self.rate_limit_window = rate_limit_window
        self.max_alerts_per_window = max_alerts_per_window
        self._pending_alerts: Dict[str, Tuple[float, Alert]] = {}  # For duration tracking
        self._metric_history = defaultdict(lambda: deque(maxlen=1000))

    def add_channel(self, name: str, channel: AlertChannel):
        """Add notification channel."""
        with self._lock:
            self.channels[name] = channel
            logger.info(f"Added alert channel: {name}")

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        with self._lock:
            self.rules.append(rule)
            logger.info(f"Added alert rule: {rule.metric_name}")

    def load_rules_from_yaml(self, yaml_path: str):
        """Load alert rules from YAML configuration."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        for rule_config in config.get('rules', []):
            rule = AlertRule(**rule_config)
            self.add_rule(rule)

        logger.info(f"Loaded {len(config.get('rules', []))} rules from {yaml_path}")

    def acknowledge_alert(self, fingerprint: str, acknowledged_by: str):
        """Acknowledge an alert."""
        with self._lock:
            if fingerprint in self._active_alerts:
                alert = self._active_alerts[fingerprint]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                logger.info(f"Alert {fingerprint} acknowledged by {acknowledged_by}")

    def silence_alert(self, fingerprint: str, duration_seconds: int):
        """Silence an alert for specified duration."""
        with self._lock:
            silence_until = time.time() + duration_seconds
            self._silences[fingerprint] = silence_until
            if fingerprint in self._active_alerts:
                self._active_alerts[fingerprint].silence_until = silence_until
            logger.info(f"Alert {fingerprint} silenced until {datetime.fromtimestamp(silence_until)}")

    def _generate_fingerprint(self, rule: AlertRule, labels: Dict[str, str]) -> str:
        """Generate unique fingerprint for alert deduplication."""
        parts = [rule.metric_name, rule.condition, str(rule.threshold)]
        for key in sorted(rule.grouping_keys):
            if key in labels:
                parts.append(f"{key}:{labels[key]}")
        return "|".join(parts)

    def _should_rate_limit(self, fingerprint: str) -> bool:
        """Check if alert should be rate limited."""
        current_time = time.time()
        rate_limit_queue = self._rate_limiter[fingerprint]

        # Remove old entries outside the window
        while rate_limit_queue and rate_limit_queue[0] < current_time - self.rate_limit_window:
            rate_limit_queue.popleft()

        # Check if we've exceeded the limit
        if len(rate_limit_queue) >= self.max_alerts_per_window:
            return True

        # Add current time to queue
        rate_limit_queue.append(current_time)
        return False

    def _send_notification(self, alert: Alert):
        """Send notification through configured channels."""
        # Check if alert is silenced
        if alert.silence_until and time.time() < alert.silence_until:
            return

        # Check rate limiting
        if self._should_rate_limit(alert.fingerprint):
            logger.warning(f"Rate limited alert: {alert.fingerprint}")
            return

        # Check repeat interval
        if alert.last_notification and time.time() - alert.last_notification < alert.rule.repeat_interval:
            return

        # Send to all configured channels
        for channel_name, channel in self.channels.items():
            try:
                if channel.send(alert):
                    logger.info(f"Alert sent via {channel_name}: {alert.fingerprint}")
                else:
                    logger.error(f"Failed to send alert via {channel_name}: {alert.fingerprint}")
            except Exception as e:
                logger.error(f"Error sending alert via {channel_name}: {e}")

        alert.last_notification = time.time()
        alert.notification_count += 1

    def check_alerts(self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None):
        """Enhanced alert checking with deduplication and state management."""
        current_time = time.time()
        labels = labels or {}

        with self._lock:
            # Store metric history
            for metric_name, value in metrics.items():
                self._metric_history[metric_name].append((current_time, value))

            for rule in self.rules:
                if rule.metric_name not in metrics:
                    continue

                value = metrics[rule.metric_name]
                history = list(self._metric_history[rule.metric_name])

                # Check if rule is triggered
                triggered = rule.evaluate(value, history)

                # Check for anomaly detection
                if rule.anomaly_detection and not triggered:
                    # Use the anomaly detector if available
                    pass

                # Generate fingerprint for deduplication
                alert_labels = {**labels, **rule.labels}
                fingerprint = self._generate_fingerprint(rule, alert_labels)

                if triggered:
                    if fingerprint in self._pending_alerts:
                        # Check if duration threshold met
                        pending_start, pending_alert = self._pending_alerts[fingerprint]
                        if current_time - pending_start >= rule.duration_seconds:
                            # Move from pending to active
                            if fingerprint not in self._active_alerts:
                                alert = Alert(
                                    rule=rule,
                                    value=value,
                                    state=AlertState.FIRING,
                                    start_time=pending_start,
                                    last_notification=0,
                                    fingerprint=fingerprint,
                                    labels=alert_labels,
                                    annotations={**rule.annotations, "value": str(value)}
                                )
                                self._active_alerts[fingerprint] = alert
                                self._alert_history.append(alert)
                                self._send_notification(alert)
                                del self._pending_alerts[fingerprint]
                                logger.warning(f"Alert firing: {rule.metric_name} {rule.condition} {rule.threshold}, value: {value}")
                    else:
                        # New pending alert
                        if fingerprint not in self._active_alerts:
                            pending_alert = Alert(
                                rule=rule,
                                value=value,
                                state=AlertState.PENDING,
                                start_time=current_time,
                                last_notification=0,
                                fingerprint=fingerprint,
                                labels=alert_labels,
                                annotations={**rule.annotations, "value": str(value)}
                            )
                            self._pending_alerts[fingerprint] = (current_time, pending_alert)
                            logger.debug(f"Alert pending: {fingerprint}")
                else:
                    # Alert condition not met
                    if fingerprint in self._pending_alerts:
                        del self._pending_alerts[fingerprint]
                        logger.debug(f"Pending alert cleared: {fingerprint}")

                    if fingerprint in self._active_alerts:
                        # Alert resolved
                        alert = self._active_alerts[fingerprint]
                        alert.state = AlertState.RESOLVED
                        alert.resolved_at = current_time
                        alert.value = value
                        self._send_notification(alert)
                        del self._active_alerts[fingerprint]
                        logger.info(f"Alert resolved: {rule.metric_name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, since: Optional[float] = None) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            if since is None:
                return list(self._alert_history)
            return [a for a in self._alert_history if a.start_time >= since]

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get active alerts by severity."""
        with self._lock:
            return [a for a in self._active_alerts.values() if a.rule.severity == severity]

    def test_channels(self) -> Dict[str, bool]:
        """Test all configured channels."""
        results = {}
        for name, channel in self.channels.items():
            try:
                results[name] = channel.test_connection()
            except Exception as e:
                logger.error(f"Failed to test channel {name}: {e}")
                results[name] = False
        return results


class MetricsCollector:
    """Central metrics collection and management system."""

    def __init__(self, export_interval: int = 60):
        self.metrics = {}
        self.export_interval = export_interval
        self.resource_monitor = ResourceMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self._lock = threading.RLock()
        self._export_thread = None
        self._running = False
        self._time_series_data = defaultdict(list)

    def register_metric(self, metric: Metric):
        """Register a metric for collection."""
        with self._lock:
            self.metrics[metric.name] = metric
            logger.debug(f"Registered metric: {metric.name}")

    def counter(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Counter:
        """Create and register a counter metric."""
        counter = Counter(name, description, labels)
        self.register_metric(counter)
        return counter

    def gauge(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Create and register a gauge metric."""
        gauge = Gauge(name, description, labels)
        self.register_metric(gauge)
        return gauge

    def histogram(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None,
                  buckets: Optional[List[float]] = None) -> Histogram:
        """Create and register a histogram metric."""
        histogram = Histogram(name, description, labels, buckets)
        self.register_metric(histogram)
        return histogram

    @contextmanager
    def timer(self, metric_name: str):
        """Context manager for timing operations."""
        if metric_name not in self.metrics:
            self.histogram(metric_name, f"Timing for {metric_name}")

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if isinstance(self.metrics[metric_name], Histogram):
                self.metrics[metric_name].observe(duration)

    def start(self):
        """Start metrics collection."""
        self._running = True
        self.resource_monitor.start()
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
        logger.info("Metrics collector started")

    def stop(self):
        """Stop metrics collection."""
        self._running = False
        self.resource_monitor.stop()
        if self._export_thread:
            self._export_thread.join(timeout=5)
        logger.info("Metrics collector stopped")

    def _export_loop(self):
        """Periodic export loop."""
        while self._running:
            try:
                self._collect_and_export()
            except Exception as e:
                logger.error(f"Export error: {e}")
            time.sleep(self.export_interval)

    def _collect_and_export(self):
        """Collect metrics and export."""
        timestamp = time.time()

        # Collect custom metrics
        snapshots = []
        with self._lock:
            for metric in self.metrics.values():
                snapshot = MetricSnapshot(
                    name=metric.name,
                    type=MetricType.COUNTER if isinstance(metric, Counter) else
                         MetricType.GAUGE if isinstance(metric, Gauge) else
                         MetricType.HISTOGRAM,
                    value=metric.get_value(),
                    timestamp=timestamp,
                    labels=metric.labels
                )
                snapshots.append(snapshot)

                # Store time-series data
                self._time_series_data[metric.name].append((timestamp, metric.get_value()))

                # Check for anomalies
                self.anomaly_detector.update(metric.name, metric.get_value())

        # Collect resource metrics
        resource_metrics = self.resource_monitor.get_metrics()
        for name, value in resource_metrics.items():
            self._time_series_data[f"system_{name}"].append((timestamp, value))
            self.anomaly_detector.update(f"system_{name}", value)

        # Check alerts
        all_metrics = {m.name: m.get_value() for m in self.metrics.values()}
        all_metrics.update({f"system_{k}": v for k, v in resource_metrics.items()})
        self.alert_manager.check_alerts(all_metrics)

        logger.debug(f"Exported {len(snapshots)} metrics")

    def get_metrics_json(self) -> str:
        """Export metrics as JSON."""
        data = {
            'timestamp': time.time(),
            'metrics': {},
            'resources': self.resource_monitor.get_metrics(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'recent_anomalies': self.anomaly_detector.get_anomalies(time.time() - 300)
        }

        with self._lock:
            for name, metric in self.metrics.items():
                data['metrics'][name] = {
                    'value': metric.get_value(),
                    'type': type(metric).__name__,
                    'labels': metric.labels
                }

        return json.dumps(data, indent=2)

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        with self._lock:
            for metric in self.metrics.values():
                lines.append(metric.to_prometheus())

        # Add resource metrics
        for name, value in self.resource_monitor.get_metrics().items():
            lines.append(f"system_{name} {value}")

        return "\n".join(lines)


# Example instrumentation
def create_example_metrics():
    """Create example metrics setup."""
    collector = MetricsCollector(export_interval=10)

    # Performance metrics
    request_counter = collector.counter("mezan_requests_total", "Total requests processed")
    error_counter = collector.counter("mezan_errors_total", "Total errors")
    active_connections = collector.gauge("mezan_active_connections", "Active connections")
    request_duration = collector.histogram("mezan_request_duration_seconds", "Request duration",
                                          buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

    # Alert rules
    collector.alert_manager.add_rule(AlertRule(
        metric_name="system_cpu_percent",
        threshold=80.0,
        condition="above",
        duration_seconds=60,
        severity="warning"
    ))

    collector.alert_manager.add_rule(AlertRule(
        metric_name="system_memory_percent",
        threshold=90.0,
        condition="above",
        duration_seconds=30,
        severity="critical"
    ))

    return collector


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    collector = create_example_metrics()
    collector.start()

    try:
        # Simulate some operations
        for i in range(100):
            with collector.timer("operation_duration"):
                time.sleep(0.1)

                # Update metrics
                collector.metrics["mezan_requests_total"].inc()
                if i % 10 == 0:
                    collector.metrics["mezan_errors_total"].inc()
                collector.metrics["mezan_active_connections"].set(i % 20)

                # Get current state
                if i % 20 == 0:
                    print("\n=== Metrics Export ===")
                    print(collector.get_metrics_json())
                    print("\n=== Prometheus Format ===")
                    print(collector.get_prometheus_metrics()[:500])  # First 500 chars

    finally:
        collector.stop()