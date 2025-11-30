#!/usr/bin/env python3
"""
Validation module for automation assets.

Validates:
- Agent definitions against schema
- Workflow definitions against schema
- Orchestration configurations
- Prompt structure and metadata
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

AUTOMATION_PATH = Path(__file__).parent


class Severity(Enum):
    """Validation issue severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: Severity
    message: str
    path: str = ""
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    target: str
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_error(self, message: str, path: str = "", suggestion: str = ""):
        self.issues.append(ValidationIssue(Severity.ERROR, message, path, suggestion))
        self.valid = False

    def add_warning(self, message: str, path: str = "", suggestion: str = ""):
        self.issues.append(ValidationIssue(Severity.WARNING, message, path, suggestion))

    def add_info(self, message: str, path: str = ""):
        self.issues.append(ValidationIssue(Severity.INFO, message, path))

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# ============== AGENT VALIDATION ==============

REQUIRED_AGENT_FIELDS = ["role", "goal", "backstory", "tools", "llm_config"]
VALID_LLM_MODELS = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]


def validate_agent(agent_name: str, agent_config: Dict[str, Any]) -> ValidationResult:
    """Validate a single agent definition."""
    result = ValidationResult(valid=True, target=f"agent:{agent_name}")

    # Check required fields
    for field in REQUIRED_AGENT_FIELDS:
        if field not in agent_config:
            result.add_error(
                f"Missing required field: {field}",
                path=f"agents.{agent_name}",
                suggestion=f"Add '{field}' to the agent definition"
            )

    # Validate role
    role = agent_config.get("role", "")
    if role and len(role) < 3:
        result.add_warning("Role is very short", path=f"agents.{agent_name}.role")

    # Validate goal
    goal = agent_config.get("goal", "")
    if goal and len(goal) < 10:
        result.add_warning("Goal should be more descriptive", path=f"agents.{agent_name}.goal")

    # Validate backstory
    backstory = agent_config.get("backstory", "")
    if backstory and len(backstory) < 50:
        result.add_warning(
            "Backstory is short - consider adding more context",
            path=f"agents.{agent_name}.backstory"
        )

    # Validate tools
    tools = agent_config.get("tools", [])
    if not tools:
        result.add_warning("Agent has no tools assigned", path=f"agents.{agent_name}.tools")

    # Validate LLM config
    llm_config = agent_config.get("llm_config", {})
    if llm_config:
        model = llm_config.get("model", "")
        if model and model not in VALID_LLM_MODELS:
            result.add_warning(
                f"Unknown model: {model}",
                path=f"agents.{agent_name}.llm_config.model",
                suggestion=f"Valid models: {', '.join(VALID_LLM_MODELS)}"
            )

        temp = llm_config.get("temperature", 0.5)
        if not 0 <= temp <= 1:
            result.add_error(
                f"Temperature must be between 0 and 1, got {temp}",
                path=f"agents.{agent_name}.llm_config.temperature"
            )

    return result


def validate_agents_file() -> ValidationResult:
    """Validate the entire agents.yaml file."""
    agents_file = AUTOMATION_PATH / "agents" / "config" / "agents.yaml"

    if not agents_file.exists():
        result = ValidationResult(valid=False, target="agents.yaml")
        result.add_error("agents.yaml not found", path=str(agents_file))
        return result

    config = _load_yaml(agents_file)
    result = ValidationResult(valid=True, target="agents.yaml")

    # Check version
    if "version" not in config:
        result.add_warning("Missing version field", suggestion="Add 'version: \"1.0\"'")

    # Check categories
    categories = config.get("categories", {})
    agents = config.get("agents", {})

    # Validate each agent
    for agent_name, agent_config in agents.items():
        agent_result = validate_agent(agent_name, agent_config)
        result.issues.extend(agent_result.issues)
        if not agent_result.valid:
            result.valid = False

    # Check category references
    for cat_name, cat_config in categories.items():
        for agent_ref in cat_config.get("agents", []):
            if agent_ref not in agents:
                result.add_error(
                    f"Category '{cat_name}' references unknown agent: {agent_ref}",
                    path=f"categories.{cat_name}.agents"
                )

    result.add_info(f"Validated {len(agents)} agents")
    return result


# ============== WORKFLOW VALIDATION ==============

VALID_PATTERNS = ["prompt_chaining", "routing", "parallelization", "orchestrator_workers", "evaluator_optimizer"]


def validate_workflow(wf_name: str, wf_config: Dict[str, Any]) -> ValidationResult:
    """Validate a single workflow definition."""
    result = ValidationResult(valid=True, target=f"workflow:{wf_name}")

    # Check pattern
    pattern = wf_config.get("pattern", "")
    if not pattern:
        result.add_error("Missing pattern field", path=f"workflows.{wf_name}")
    elif pattern not in VALID_PATTERNS:
        result.add_warning(
            f"Unknown pattern: {pattern}",
            path=f"workflows.{wf_name}.pattern",
            suggestion=f"Valid patterns: {', '.join(VALID_PATTERNS)}"
        )

    # Check stages
    stages = wf_config.get("stages", [])
    if not stages:
        result.add_error("Workflow has no stages", path=f"workflows.{wf_name}.stages")

    stage_names = set()
    for i, stage in enumerate(stages):
        stage_name = stage.get("name", f"stage_{i}")

        if stage_name in stage_names:
            result.add_error(
                f"Duplicate stage name: {stage_name}",
                path=f"workflows.{wf_name}.stages[{i}]"
            )
        stage_names.add(stage_name)

        # Check dependencies
        depends_on = stage.get("depends_on", [])
        for dep in depends_on:
            if dep not in stage_names:
                result.add_error(
                    f"Stage '{stage_name}' depends on unknown stage: {dep}",
                    path=f"workflows.{wf_name}.stages[{i}].depends_on"
                )

    # Check success criteria
    if not wf_config.get("success_criteria"):
        result.add_warning(
            "No success criteria defined",
            path=f"workflows.{wf_name}",
            suggestion="Add success_criteria for better quality control"
        )

    return result


def validate_workflows_file() -> ValidationResult:
    """Validate the entire workflows.yaml file."""
    workflows_file = AUTOMATION_PATH / "workflows" / "config" / "workflows.yaml"

    if not workflows_file.exists():
        result = ValidationResult(valid=False, target="workflows.yaml")
        result.add_error("workflows.yaml not found", path=str(workflows_file))
        return result

    config = _load_yaml(workflows_file)
    result = ValidationResult(valid=True, target="workflows.yaml")

    workflows = config.get("workflows", {})

    for wf_name, wf_config in workflows.items():
        wf_result = validate_workflow(wf_name, wf_config)
        result.issues.extend(wf_result.issues)
        if not wf_result.valid:
            result.valid = False

    result.add_info(f"Validated {len(workflows)} workflows")
    return result


# ============== PROMPT VALIDATION ==============

def validate_prompt(prompt_path: Path) -> ValidationResult:
    """Validate a single prompt file."""
    result = ValidationResult(valid=True, target=str(prompt_path.name))

    if not prompt_path.exists():
        result.add_error("Prompt file not found", path=str(prompt_path))
        return result

    content = prompt_path.read_text(encoding='utf-8')

    # Check minimum length
    if len(content) < 100:
        result.add_warning("Prompt is very short", suggestion="Consider adding more detail")

    # Check for heading
    if not content.startswith('#'):
        result.add_warning(
            "Prompt should start with a markdown heading",
            suggestion="Add '# Title' at the beginning"
        )

    # Check for common sections in system prompts
    if "system" in str(prompt_path):
        expected_sections = ["responsibilities", "output", "format"]
        content_lower = content.lower()
        for section in expected_sections:
            if section not in content_lower:
                result.add_info(f"Consider adding a '{section}' section")

    return result


def validate_all_prompts() -> ValidationResult:
    """Validate all prompt files."""
    prompts_path = AUTOMATION_PATH / "prompts"
    result = ValidationResult(valid=True, target="prompts")

    count = 0
    for category in ["system", "project", "tasks"]:
        category_path = prompts_path / category
        if not category_path.exists():
            continue

        for prompt_file in category_path.glob("*.md"):
            prompt_result = validate_prompt(prompt_file)
            result.issues.extend(prompt_result.issues)
            if not prompt_result.valid:
                result.valid = False
            count += 1

    result.add_info(f"Validated {count} prompts")
    return result


# ============== FULL VALIDATION ==============

def validate_all() -> Dict[str, ValidationResult]:
    """Validate all automation assets."""
    return {
        "agents": validate_agents_file(),
        "workflows": validate_workflows_file(),
        "prompts": validate_all_prompts(),
    }


def print_validation_report(results: Dict[str, ValidationResult]):
    """Print a formatted validation report."""
    print("=" * 60)
    print("AUTOMATION VALIDATION REPORT")
    print("=" * 60)

    total_errors = 0
    total_warnings = 0

    for name, result in results.items():
        status = "✓ VALID" if result.valid else "✗ INVALID"
        print(f"\n{name.upper()}: {status}")
        print("-" * 40)

        for issue in result.issues:
            if issue.severity == Severity.ERROR:
                icon = "[ERROR]"
                total_errors += 1
            elif issue.severity == Severity.WARNING:
                icon = "[WARN]"
                total_warnings += 1
            else:
                icon = "[INFO]"

            print(f"  {icon} {issue.message}")
            if issue.path:
                print(f"         Path: {issue.path}")
            if issue.suggestion:
                print(f"         Fix: {issue.suggestion}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_errors} errors, {total_warnings} warnings")
    print("=" * 60)

    return total_errors == 0


if __name__ == "__main__":
    results = validate_all()
    success = print_validation_report(results)
    exit(0 if success else 1)
