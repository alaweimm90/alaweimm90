#!/usr/bin/env python3
"""
orchestration_validator.py - Handoff Envelope and Orchestration Policy Validator

Validates tool-to-tool handoff envelopes against:
- JSON Schema (handoff-envelope-schema.json)
- Orchestration governance policy (orchestration-governance.yaml)
- Tool routing rules and confidence thresholds

Usage:
    python orchestration_validator.py validate envelope.json
    python orchestration_validator.py route "implement user auth"
    python orchestration_validator.py check-routing --source cline --target kilo
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import click
import yaml

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    level: ValidationLevel
    message: str
    path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    valid: bool
    timestamp: str
    target: str
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def add_result(self, result: ValidationResult):
        self.results.append(result)
        level_key = result.level.value
        self.summary[level_key] = self.summary.get(level_key, 0) + 1
        if not result.passed and result.level == ValidationLevel.ERROR:
            self.valid = False


class OrchestrationValidator:
    """Validates handoff envelopes and orchestration policies."""

    SCHEMA_PATH = ".metaHub/schemas/handoff-envelope-schema.json"
    POLICY_PATH = ".metaHub/policies/orchestration-governance.yaml"

    # Valid tool names
    VALID_TOOLS = {
        "aider", "cline", "cursor", "claude_code", "kilo",
        "blackbox", "windsurf", "continue", "amazon_q",
        "gemini", "codex", "augment", "trae", "copilot"
    }

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.schema = self._load_schema()
        self.policy = self._load_policy()
        self.enforcement_level = self.policy.get("enforcement", {}).get("level", "warning")

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

    def _load_schema(self) -> Dict[str, Any]:
        """Load handoff envelope JSON schema."""
        schema_path = self.base_path / self.SCHEMA_PATH
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_policy(self) -> Dict[str, Any]:
        """Load orchestration governance policy."""
        policy_path = self.base_path / self.POLICY_PATH
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def validate_envelope(self, envelope: Dict[str, Any]) -> ValidationReport:
        """Validate a handoff envelope against schema and policy."""
        report = ValidationReport(
            valid=True,
            timestamp=datetime.now().isoformat(),
            target="handoff_envelope"
        )

        # Schema validation
        self._validate_schema(envelope, report)

        # Policy validation
        self._validate_policy(envelope, report)

        # Tool routing validation
        self._validate_routing(envelope, report)

        # Context validation
        self._validate_context(envelope, report)

        return report

    def _validate_schema(self, envelope: Dict[str, Any], report: ValidationReport):
        """Validate envelope against JSON schema."""
        if not self.schema:
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="Schema not loaded - skipping schema validation",
                path="schema"
            ))
            return

        if not HAS_JSONSCHEMA:
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="jsonschema package not installed - skipping schema validation",
                path="schema"
            ))
            return

        try:
            jsonschema.validate(envelope, self.schema)
            report.add_result(ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Schema validation passed",
                path="schema"
            ))
        except jsonschema.ValidationError as e:
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Schema validation failed: {e.message}",
                path='.'.join(str(p) for p in e.absolute_path),
                details={"schema_path": list(e.schema_path)}
            ))

    def _validate_policy(self, envelope: Dict[str, Any], report: ValidationReport):
        """Validate envelope against orchestration policy."""
        handoff_req = self.policy.get("handoff_requirements", {})
        mandatory_fields = handoff_req.get("mandatory_fields", [])

        # Flatten envelope for field checking
        flat_fields = self._flatten_envelope_fields(envelope)

        # Check mandatory fields
        for field in mandatory_fields:
            if field not in flat_fields:
                report.add_result(ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing mandatory field: {field}",
                    path=f"policy.mandatory_fields.{field}"
                ))

        # Check context size
        validation_rules = handoff_req.get("validation", {})
        max_context_kb = validation_rules.get("max_context_size_kb", 500)

        context_size = len(json.dumps(envelope.get("context", {})).encode('utf-8'))
        if context_size > max_context_kb * 1024:
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Context size ({context_size / 1024:.1f}KB) exceeds limit ({max_context_kb}KB)",
                path="context",
                details={"size_kb": context_size / 1024, "limit_kb": max_context_kb}
            ))
        else:
            report.add_result(ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message=f"Context size: {context_size / 1024:.1f}KB",
                path="context"
            ))

        # Validate timestamp format
        if validation_rules.get("require_timestamp_format") == "ISO8601":
            timestamp = envelope.get("metadata", {}).get("timestamp", "")
            if not self._is_valid_iso8601(timestamp):
                report.add_result(ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid timestamp format: {timestamp}",
                    path="metadata.timestamp"
                ))

    def _flatten_envelope_fields(self, envelope: Dict[str, Any]) -> set:
        """Flatten envelope to set of field names."""
        fields = set()

        # Add top-level fields
        for key in envelope:
            fields.add(key)

        # Metadata fields
        metadata = envelope.get("metadata", {})
        fields.update(metadata.keys())

        # Context fields
        context = envelope.get("context", {})
        fields.update(context.keys())

        # Artifacts fields
        artifacts = envelope.get("artifacts", {})
        fields.update(artifacts.keys())

        # Instructions fields
        instructions = envelope.get("instructions", {})
        fields.update(instructions.keys())

        # Map common field names
        field_mapping = {
            "expected_action": "next_action",
            "primary_output": "validation_status"
        }
        for original, mapped in field_mapping.items():
            if original in fields:
                fields.add(mapped)

        return fields

    def _is_valid_iso8601(self, timestamp: str) -> bool:
        """Check if timestamp is valid ISO8601 format."""
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        return bool(re.match(iso8601_pattern, timestamp))

    def _validate_routing(self, envelope: Dict[str, Any], report: ValidationReport):
        """Validate tool routing rules."""
        metadata = envelope.get("metadata", {})
        source_tool = metadata.get("source_tool", "")
        target_tool = metadata.get("target_tool", "")

        # Validate tool names
        for tool_name, field_name in [(source_tool, "source_tool"), (target_tool, "target_tool")]:
            if tool_name and tool_name not in self.VALID_TOOLS:
                report.add_result(ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid tool name: {tool_name}",
                    path=f"metadata.{field_name}",
                    details={"valid_tools": list(self.VALID_TOOLS)}
                ))

        # Check routing rules
        if source_tool and target_tool:
            routing_check = self.check_routing(source_tool, target_tool)
            if not routing_check["valid"]:
                report.add_result(ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=routing_check["message"],
                    path="metadata.routing",
                    details=routing_check
                ))
            else:
                report.add_result(ValidationResult(
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"Routing valid: {source_tool} -> {target_tool}",
                    path="metadata.routing"
                ))

    def _validate_context(self, envelope: Dict[str, Any], report: ValidationReport):
        """Validate context completeness."""
        context = envelope.get("context", {})

        # Check task description
        task_desc = context.get("task_description", "")
        if len(task_desc) < 10:
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message="Task description too short (min 10 chars)",
                path="context.task_description"
            ))

        # Check for relevant files
        relevant_files = context.get("relevant_files", [])
        if not relevant_files:
            report.add_result(ValidationResult(
                passed=True,
                level=ValidationLevel.WARNING,
                message="No relevant files specified",
                path="context.relevant_files"
            ))

        # Check success criteria
        success_criteria = context.get("success_criteria", [])
        if not success_criteria:
            report.add_result(ValidationResult(
                passed=True,
                level=ValidationLevel.WARNING,
                message="No success criteria defined",
                path="context.success_criteria"
            ))

    def check_routing(self, source_tool: str, target_tool: str) -> Dict[str, Any]:
        """Check if routing between tools is valid."""
        result = {
            "valid": True,
            "source": source_tool,
            "target": target_tool,
            "message": "Routing is valid"
        }

        # Same tool routing is always valid (continuation)
        if source_tool == target_tool:
            result["message"] = "Same-tool continuation"
            return result

        # Get task type affinities
        routing_rules = self.policy.get("tool_routing", {}).get("rules", {})

        source_tasks = set()
        target_tasks = set()

        for task_type, config in routing_rules.items():
            tools = config.get("tools", [])
            if source_tool in tools:
                source_tasks.add(task_type)
            if target_tool in tools:
                target_tasks.add(task_type)

        # Check for complementary routing
        # Architecture -> Implementation is good
        # Implementation -> Testing is good
        # etc.
        complementary_routes = [
            ("architecture", "implementation"),
            ("implementation", "testing"),
            ("implementation", "refactoring"),
            ("debugging", "testing"),
            ("research", "architecture"),
            ("research", "implementation")
        ]

        for source_type in source_tasks:
            for target_type in target_tasks:
                if (source_type, target_type) in complementary_routes:
                    result["message"] = f"Complementary routing: {source_type} -> {target_type}"
                    return result

        # Check if there's any overlap (same task type)
        overlap = source_tasks & target_tasks
        if overlap:
            result["message"] = f"Same-domain routing for: {', '.join(overlap)}"
            return result

        # No clear routing relationship
        result["valid"] = False
        result["message"] = f"No defined routing relationship between {source_tool} and {target_tool}"
        result["source_tasks"] = list(source_tasks)
        result["target_tasks"] = list(target_tasks)

        return result

    def route_task(self, task_description: str) -> Dict[str, Any]:
        """Route a task to the best-fit tool based on intent."""
        intent_config = self.policy.get("tool_routing", {}).get("intent_extraction", {})

        if not intent_config.get("enabled", True):
            return {
                "success": False,
                "message": "Intent extraction disabled"
            }

        keywords = intent_config.get("keywords", {})
        routing_rules = self.policy.get("tool_routing", {}).get("rules", {})

        # Score each task type
        task_lower = task_description.lower()
        scores: Dict[str, int] = {}

        for task_type, task_keywords in keywords.items():
            score = sum(1 for kw in task_keywords if kw in task_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return {
                "success": False,
                "message": "Could not determine task type from description",
                "task": task_description
            }

        # Get best matching task type
        best_type = max(scores, key=scores.get)
        best_tools = routing_rules.get(best_type, {}).get("tools", [])

        # Also check agent frameworks
        agent_routing = self.policy.get("agent_framework_routing", {})
        matching_frameworks = []

        for framework_type, config in agent_routing.items():
            use_when = config.get("use_when", [])
            if any(use_case in task_lower for use_case in use_when):
                matching_frameworks.append({
                    "type": framework_type,
                    "primary": config.get("primary"),
                    "fallback": config.get("fallback"),
                    "agents": config.get("agents", [])
                })

        return {
            "success": True,
            "task": task_description,
            "detected_type": best_type,
            "confidence_scores": scores,
            "recommended_tools": best_tools,
            "primary_tool": best_tools[0] if best_tools else None,
            "agent_frameworks": matching_frameworks
        }

    def validate_file(self, file_path: Path) -> ValidationReport:
        """Validate a handoff envelope from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                envelope = json.load(f)
            return self.validate_envelope(envelope)
        except json.JSONDecodeError as e:
            report = ValidationReport(
                valid=False,
                timestamp=datetime.now().isoformat(),
                target=str(file_path)
            )
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Invalid JSON: {e}",
                path="file"
            ))
            return report
        except OSError as e:
            report = ValidationReport(
                valid=False,
                timestamp=datetime.now().isoformat(),
                target=str(file_path)
            )
            report.add_result(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Could not read file: {e}",
                path="file"
            ))
            return report


def format_report(report: ValidationReport, fmt: str = "text") -> str:
    """Format validation report for output."""
    if fmt == "json":
        return json.dumps({
            "valid": report.valid,
            "timestamp": report.timestamp,
            "target": report.target,
            "summary": report.summary,
            "results": [
                {
                    "passed": r.passed,
                    "level": r.level.value,
                    "message": r.message,
                    "path": r.path,
                    "details": r.details
                }
                for r in report.results
            ]
        }, indent=2)

    # Text format
    lines = [
        "=" * 60,
        "ORCHESTRATION VALIDATION REPORT",
        "=" * 60,
        "",
        f"Target: {report.target}",
        f"Timestamp: {report.timestamp}",
        f"Status: {'VALID' if report.valid else 'INVALID'}",
        "",
        "-" * 40,
        "RESULTS",
        "-" * 40,
    ]

    for result in report.results:
        icon = "[OK]" if result.passed else "[FAIL]" if result.level == ValidationLevel.ERROR else "[WARN]"
        lines.append(f"  {icon} {result.message}")
        if result.path:
            lines.append(f"      Path: {result.path}")

    lines.extend([
        "",
        "-" * 40,
        "SUMMARY",
        "-" * 40,
        f"  Errors: {report.summary.get('error', 0)}",
        f"  Warnings: {report.summary.get('warning', 0)}",
        f"  Info: {report.summary.get('info', 0)}",
        "",
        "=" * 60,
    ])

    return '\n'.join(lines)


@click.group()
def cli():
    """Orchestration validator for handoff envelopes and routing."""
    pass


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--json-output', is_flag=True, help='Output as JSON')
def validate(file: str, json_output: bool):
    """Validate a handoff envelope file."""
    try:
        validator = OrchestrationValidator()
        report = validator.validate_file(Path(file))

        output = format_report(report, "json" if json_output else "text")
        click.echo(output)

        raise SystemExit(0 if report.valid else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument('task', type=str)
@click.option('--json-output', is_flag=True, help='Output as JSON')
def route(task: str, json_output: bool):
    """Route a task description to recommended tools."""
    try:
        validator = OrchestrationValidator()
        result = validator.route_task(task)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            if result["success"]:
                click.echo(f"Task: {result['task']}")
                click.echo(f"Detected Type: {result['detected_type']}")
                click.echo(f"Primary Tool: {result['primary_tool']}")
                click.echo(f"Recommended Tools: {', '.join(result['recommended_tools'])}")

                if result['agent_frameworks']:
                    click.echo("\nMatching Agent Frameworks:")
                    for fw in result['agent_frameworks']:
                        click.echo(f"  - {fw['primary']} ({fw['type']})")
                        if fw.get('agents'):
                            click.echo(f"    Agents: {', '.join(fw['agents'][:3])}...")
            else:
                click.echo(f"Routing failed: {result['message']}")

        raise SystemExit(0 if result["success"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('check-routing')
@click.option('--source', '-s', required=True, help='Source tool name')
@click.option('--target', '-t', required=True, help='Target tool name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def check_routing(source: str, target: str, json_output: bool):
    """Check if routing between two tools is valid."""
    try:
        validator = OrchestrationValidator()
        result = validator.check_routing(source, target)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            status = "VALID" if result["valid"] else "INVALID"
            click.echo(f"Routing {source} -> {target}: {status}")
            click.echo(f"  {result['message']}")

        raise SystemExit(0 if result["valid"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('list-tools')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def list_tools(json_output: bool):
    """List valid tool names and their task affinities."""
    try:
        validator = OrchestrationValidator()
        routing_rules = validator.policy.get("tool_routing", {}).get("rules", {})

        tool_tasks: Dict[str, List[str]] = {tool: [] for tool in validator.VALID_TOOLS}

        for task_type, config in routing_rules.items():
            for tool in config.get("tools", []):
                if tool in tool_tasks:
                    tool_tasks[tool].append(task_type)

        if json_output:
            click.echo(json.dumps(tool_tasks, indent=2))
        else:
            click.echo("Tool Task Affinities:")
            click.echo("-" * 40)
            for tool, tasks in sorted(tool_tasks.items()):
                tasks_str = ', '.join(tasks) if tasks else '(none defined)'
                click.echo(f"  {tool}: {tasks_str}")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
