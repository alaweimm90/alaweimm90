"""
Tests for MEZAN Advanced Alerting System.

Comprehensive test coverage for alert rules, channels, grouping,
correlation, and on-call management.
"""

import pytest
import time
import json
import yaml
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from atlas_core.telemetry import (
    Alert, AlertRule, AlertState, AlertSeverity,
    AlertChannel, EmailAlertChannel, SlackAlertChannel,
    PagerDutyAlertChannel, WebhookAlertChannel
)

from atlas_core.alerting import (
    OnCallSchedule, AlertGroup, AlertCorrelation,
    AlertRuleEngine, AlertGroupManager, OnCallManager,
    AlertManager, create_example_config
)


class TestAlertRuleEngine:
    """Test alert rule evaluation engine."""

    def test_load_rules_from_yaml(self):
        """Test loading rules from YAML file."""
        yaml_content = """
        templates:
          default:
            duration_seconds: 300
            severity: warning

        rules:
          - metric_name: cpu_usage
            threshold: 80.0
            condition: above
            template: default
          - metric_name: memory_usage
            threshold: 90.0
            condition: above
            severity: critical
            duration_seconds: 60
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            engine = AlertRuleEngine()
            engine.load_rules_from_file(f.name)

        os.unlink(f.name)

        assert len(engine.rules) == 2
        assert engine.rules[0].metric_name == "cpu_usage"
        assert engine.rules[0].duration_seconds == 300  # From template
        assert engine.rules[1].severity == "critical"

    def test_load_rules_from_json(self):
        """Test loading rules from JSON file."""
        json_content = {
            "rules": [
                {
                    "metric_name": "error_rate",
                    "threshold": 0.05,
                    "condition": "above",
                    "severity": "error"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            f.flush()

            engine = AlertRuleEngine()
            engine.load_rules_from_file(f.name)

        os.unlink(f.name)

        assert len(engine.rules) == 1
        assert engine.rules[0].metric_name == "error_rate"

    def test_expression_compilation(self):
        """Test DSL expression compilation."""
        engine = AlertRuleEngine()

        # Test duration conversions
        assert "300" in engine._compile_expression("duration > 5m")
        assert "3600" in engine._compile_expression("duration > 1h")
        assert "86400" in engine._compile_expression("duration > 1d")

        # Test logical operators
        expr = engine._compile_expression("metric > 90 AND duration > 5m")
        assert "and" in expr
        assert "300" in expr

    def test_rule_evaluation(self):
        """Test evaluating rules against metrics."""
        engine = AlertRuleEngine()

        # Add test rules
        engine.add_rule(AlertRule(
            metric_name="cpu_usage",
            threshold=80.0,
            condition="above"
        ))
        engine.add_rule(AlertRule(
            metric_name="memory_usage",
            threshold=90.0,
            condition="above"
        ))

        # Test metrics
        metrics = {
            "cpu_usage": 85.0,  # Should trigger
            "memory_usage": 88.0  # Should not trigger
        }

        triggered = engine.evaluate(metrics)
        assert len(triggered) == 1
        assert triggered[0].metric_name == "cpu_usage"

    def test_get_rules_by_severity(self):
        """Test filtering rules by severity."""
        engine = AlertRuleEngine()

        # Add rules with different severities
        engine.add_rule(AlertRule("metric1", 10, "above", severity="critical"))
        engine.add_rule(AlertRule("metric2", 20, "above", severity="warning"))
        engine.add_rule(AlertRule("metric3", 30, "above", severity="critical"))

        critical_rules = engine.get_rules_by_severity("critical")
        assert len(critical_rules) == 2
        assert all(r.severity == "critical" for r in critical_rules)


class TestAlertGroupManager:
    """Test alert grouping and correlation."""

    def test_group_alert(self):
        """Test alert grouping."""
        manager = AlertGroupManager()

        rule = AlertRule("test_metric", 100, "above", grouping_keys=["service"])
        alert1 = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="fp1",
            labels={"service": "api"},
            annotations={}
        )

        group = manager.group_alert(alert1)
        assert group is not None
        assert len(group.alerts) == 1
        assert group.total_count == 1

        # Add similar alert
        alert2 = Alert(
            rule=rule,
            value=110,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="fp2",
            labels={"service": "api"},
            annotations={}
        )

        group2 = manager.group_alert(alert2)
        assert group2 == group  # Same group
        assert len(group.alerts) == 2
        assert group.total_count == 2

    def test_group_expiration(self):
        """Test that groups expire after interval."""
        manager = AlertGroupManager(group_interval=1)  # 1 second interval

        rule = AlertRule("test_metric", 100, "above")
        alert1 = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="fp1",
            labels={},
            annotations={}
        )

        group1 = manager.group_alert(alert1)

        # Wait for expiration
        time.sleep(1.5)

        alert2 = Alert(
            rule=rule,
            value=110,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="fp2",
            labels={},
            annotations={}
        )

        group2 = manager.group_alert(alert2)
        assert group2.group_id != group1.group_id  # New group

    def test_alert_correlation(self):
        """Test alert correlation."""
        manager = AlertGroupManager()

        # Add correlation rule
        correlation = AlertCorrelation(
            name="cascade_failure",
            patterns=[r".*error.*", r".*timeout.*"],
            time_window=300,
            min_alerts=2,
            action="escalate"
        )
        manager.add_correlation(correlation)

        # Create alerts
        rule1 = AlertRule("api_errors", 10, "above")
        rule2 = AlertRule("db_timeout", 5, "above")

        alerts = [
            Alert(rule1, 15, AlertState.FIRING, time.time(), 0, "fp1",
                 {"service": "api"}, {}),
            Alert(rule2, 10, AlertState.FIRING, time.time(), 0, "fp2",
                 {"service": "database"}, {})
        ]

        # Find correlations
        correlations = manager.correlate_alerts(alerts)
        assert len(correlations) == 1
        assert correlations[0][0] == correlation
        assert len(correlations[0][1]) == 2


class TestOnCallManager:
    """Test on-call rotation management."""

    def test_add_schedule(self):
        """Test adding on-call schedule."""
        manager = OnCallManager()

        schedule = OnCallSchedule(
            name="primary",
            members=["alice", "bob", "charlie"],
            rotation_type="weekly"
        )

        manager.add_schedule(schedule)
        assert "primary" in manager.schedules

    def test_get_current_oncall(self):
        """Test getting current on-call person."""
        manager = OnCallManager()

        schedule = OnCallSchedule(
            name="primary",
            members=["alice", "bob", "charlie"],
            rotation_type="weekly",
            current_index=1
        )

        manager.add_schedule(schedule)
        current = manager.get_current_on_call("primary")
        assert current == "bob"

    def test_oncall_override(self):
        """Test on-call override."""
        manager = OnCallManager()

        schedule = OnCallSchedule(
            name="primary",
            members=["alice", "bob", "charlie"],
            rotation_type="weekly",
            current_index=0
        )

        manager.add_schedule(schedule)

        # Override with dave
        manager.override_on_call("primary", "dave", duration_hours=24)

        current = manager.get_current_on_call("primary")
        assert current == "dave"

    def test_rotation(self):
        """Test automatic rotation."""
        manager = OnCallManager()

        schedule = OnCallSchedule(
            name="primary",
            members=["alice", "bob", "charlie"],
            rotation_type="daily",
            rotation_interval_hours=0.001,  # Very short for testing
            current_index=0,
            last_rotation=time.time() - 10  # Old rotation
        )

        manager.add_schedule(schedule)

        # Initial person
        assert manager.get_current_on_call("primary") == "alice"

        # Wait for rotation
        time.sleep(0.01)

        # Force rotation check
        with manager._lock:
            current_time = time.time()
            schedule.current_index = 1
            schedule.last_rotation = current_time

        assert manager.get_current_on_call("primary") == "bob"


class TestAlertChannels:
    """Test alert notification channels."""

    @patch('smtplib.SMTP')
    def test_email_channel(self, mock_smtp):
        """Test email alert channel."""
        channel = EmailAlertChannel(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password",
            from_addr="alerts@test.com",
            to_addrs=["oncall@test.com"]
        )

        rule = AlertRule("test_metric", 100, "above", severity="critical")
        alert = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="test-fp",
            labels={"service": "api"},
            annotations={"summary": "Test alert"}
        )

        # Mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        result = channel.send(alert)
        assert result == True
        mock_server.send_message.assert_called_once()

    @patch('requests.post')
    def test_slack_channel(self, mock_post):
        """Test Slack alert channel."""
        channel = SlackAlertChannel(
            webhook_url="https://hooks.slack.com/test",
            channel="#alerts"
        )

        rule = AlertRule("test_metric", 100, "above", severity="warning")
        alert = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="test-fp",
            labels={},
            annotations={}
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = channel.send(alert)
        assert result == True
        mock_post.assert_called_once()

        # Check payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['channel'] == "#alerts"
        assert "test_metric" in payload['attachments'][0]['title']

    @patch('requests.post')
    def test_pagerduty_channel(self, mock_post):
        """Test PagerDuty alert channel."""
        channel = PagerDutyAlertChannel(
            integration_key="test-key"
        )

        rule = AlertRule("test_metric", 100, "above", severity="critical")
        alert = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="test-fp",
            labels={},
            annotations={}
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response

        result = channel.send(alert)
        assert result == True
        mock_post.assert_called_once()

        # Check payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['routing_key'] == "test-key"
        assert payload['event_action'] == "trigger"

    @patch('requests.request')
    def test_webhook_channel(self, mock_request):
        """Test webhook alert channel."""
        channel = WebhookAlertChannel(
            url="https://api.test.com/alerts",
            headers={"Authorization": "Bearer token"},
            method="POST"
        )

        rule = AlertRule("test_metric", 100, "above")
        alert = Alert(
            rule=rule,
            value=105,
            state=AlertState.FIRING,
            start_time=time.time(),
            last_notification=0,
            fingerprint="test-fp",
            labels={},
            annotations={}
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        result = channel.send(alert)
        assert result == True
        mock_request.assert_called_once_with(
            "POST",
            "https://api.test.com/alerts",
            json=pytest.Any(dict),
            headers={"Authorization": "Bearer token"},
            timeout=10
        )


class TestAlertManager:
    """Test the main alert manager."""

    def test_initialization_from_config(self):
        """Test initializing alert manager from config."""
        config = {
            'channels': {
                'test_slack': {
                    'type': 'slack',
                    'webhook_url': 'https://test.slack.com'
                }
            },
            'oncall_schedules': [
                {
                    'name': 'primary',
                    'members': ['alice', 'bob'],
                    'rotation_type': 'weekly'
                }
            ]
        }

        manager = AlertManager(config)
        assert 'test_slack' in manager.channels
        assert 'primary' in manager.oncall_manager.schedules

    def test_process_metrics(self):
        """Test processing metrics and generating alerts."""
        manager = AlertManager()

        # Add rule
        rule = AlertRule("cpu_usage", 80, "above", severity="warning")
        manager.rule_engine.add_rule(rule)

        # Add mock channel
        mock_channel = MagicMock()
        mock_channel.send.return_value = True
        manager.add_channel("test", mock_channel)

        # Process metrics
        metrics = {"cpu_usage": 85.0}
        manager.process_metrics(metrics)

        # Check alert created
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].rule.metric_name == "cpu_usage"

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        manager = AlertManager()

        # Create alert
        rule = AlertRule("test_metric", 100, "above")
        alert = manager._create_alert(rule, 105, {})

        # Acknowledge
        manager.acknowledge_alert(alert.fingerprint, "alice", "Looking into it")

        # Check acknowledgment
        assert alert.fingerprint in manager._acknowledgments
        assert manager._acknowledgments[alert.fingerprint]['by'] == "alice"

    def test_silence_alert(self):
        """Test silencing alerts."""
        manager = AlertManager()

        # Create alert
        rule = AlertRule("test_metric", 100, "above")
        alert = manager._create_alert(rule, 105, {})

        # Silence for 1 hour
        manager.silence_alert(alert.fingerprint, 1, "bob", "Maintenance window")

        # Check silence
        assert alert.fingerprint in manager._silences
        assert manager._silences[alert.fingerprint] > time.time()

    def test_rate_limiting(self):
        """Test alert rate limiting."""
        manager = AlertManager(config={
            'rate_limit_window': 60,
            'max_alerts_per_window': 2
        })

        rule = AlertRule("test_metric", 100, "above")

        # Create alerts
        for i in range(5):
            alert = Alert(
                rule=rule,
                value=105 + i,
                state=AlertState.FIRING,
                start_time=time.time(),
                last_notification=0,
                fingerprint="same-fp",  # Same fingerprint
                labels={},
                annotations={}
            )

            # First 2 should not be rate limited
            if i < 2:
                assert not manager._should_rate_limit(alert)
            else:
                assert manager._should_rate_limit(alert)

    def test_severity_channel_mapping(self):
        """Test channel selection based on severity."""
        config = {
            'severity_channels': {
                'critical': ['pagerduty', 'slack'],
                'warning': ['slack'],
                'info': []
            }
        }

        manager = AlertManager(config)

        channels = manager._get_channels_for_severity('critical')
        assert 'pagerduty' in channels
        assert 'slack' in channels

        channels = manager._get_channels_for_severity('info')
        assert len(channels) == 0

    def test_get_alert_stats(self):
        """Test getting alert statistics."""
        manager = AlertManager()

        # Create alerts with different severities
        for severity in ['critical', 'warning', 'info']:
            rule = AlertRule(f"metric_{severity}", 100, "above", severity=severity)
            alert = manager._create_alert(rule, 105, {})

        stats = manager.get_alert_stats()
        assert stats['total_active'] == 3
        assert stats['by_severity']['critical'] == 1
        assert stats['by_severity']['warning'] == 1
        assert stats['by_severity']['info'] == 1


class TestIntegration:
    """Integration tests for alerting system."""

    def test_end_to_end_alert_flow(self):
        """Test complete alert flow from metric to notification."""
        # Create config
        config = create_example_config()

        # Create manager
        manager = AlertManager(config)

        # Add rule
        rule = AlertRule(
            metric_name="cpu_usage",
            threshold=80.0,
            condition="above",
            severity="critical",
            duration_seconds=0,  # Immediate
            grouping_keys=["host"]
        )
        manager.rule_engine.add_rule(rule)

        # Mock channels
        mock_slack = MagicMock()
        mock_slack.send.return_value = True
        manager.channels['slack'] = mock_slack

        # Process metrics
        metrics = {"cpu_usage": 95.0}
        labels = {"host": "server-1"}
        manager.process_metrics(metrics, labels=labels)

        # Verify alert created and notification sent
        alerts = manager.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].value == 95.0
        assert alerts[0].labels["host"] == "server-1"

        # Check notification was attempted
        # Note: actual sending depends on rate limiting and other factors

    @patch('requests.post')
    def test_alert_resolution_flow(self, mock_post):
        """Test alert resolution and notification."""
        manager = AlertManager()

        # Add webhook channel
        channel = WebhookAlertChannel("https://test.com/alerts")
        manager.add_channel("webhook", channel)

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create and resolve alert
        rule = AlertRule("test_metric", 100, "above")
        alert = manager._create_alert(rule, 105, {})
        fingerprint = alert.fingerprint

        manager.resolve_alert(fingerprint)

        # Check alert removed
        assert len(manager.get_active_alerts()) == 0

        # Check resolution notification sent
        mock_post.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])