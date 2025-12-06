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


def _check_required_fields(agent_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    for field in REQUIRED_AGENT_FIELDS:
        if field not in cfg:
            res.add_error(
                f"Missing required field: {field}",
                path=f"agents.{agent_name}",
                suggestion=f"Add '{field}' to the agent definition"
            )

def _validate_lengths(agent_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    role = cfg.get("role", "")
    if role and len(role) < 3:
        res.add_warning("Role is very short", path=f"agents.{agent_name}.role")
    goal = cfg.get("goal", "")
    if goal and len(goal) < 10:
        res.add_warning("Goal should be more descriptive", path=f"agents.{agent_name}.goal")
    backstory = cfg.get("backstory", "")
    if backstory and len(backstory) < 50:
        res.add_warning("Backstory is short - consider adding more context", path=f"agents.{agent_name}.backstory")

def _validate_tools(agent_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    if not cfg.get("tools", []):
        res.add_warning("Agent has no tools assigned", path=f"agents.{agent_name}.tools")

def _validate_llm(agent_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    llm = cfg.get("llm_config", {})
    if not llm:
        return
    model = llm.get("model", "")
    if model and model not in VALID_LLM_MODELS:
        res.add_warning(
            f"Unknown model: {model}",
            path=f"agents.{agent_name}.llm_config.model",
            suggestion=f"Valid models: {', '.join(VALID_LLM_MODELS)}"
        )
    temp = llm.get("temperature", 0.5)
    if not 0 <= temp <= 1:
        res.add_error(
            f"Temperature must be between 0 and 1, got {temp}",
            path=f"agents.{agent_name}.llm_config.temperature"
        )

def validate_agent(agent_name: str, agent_config: Dict[str, Any]) -> ValidationResult:
    result = ValidationResult(valid=True, target=f"agent:{agent_name}")
    _check_required_fields(agent_name, agent_config, result)
    _validate_lengths(agent_name, agent_config, result)
    _validate_tools(agent_name, agent_config, result)
    _validate_llm(agent_name, agent_config, result)
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


def _validate_pattern(wf_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    pattern = cfg.get("pattern", "")
    if not pattern:
        res.add_error("Missing pattern field", path=f"workflows.{wf_name}")
    elif pattern not in VALID_PATTERNS:
        res.add_warning(
            f"Unknown pattern: {pattern}",
            path=f"workflows.{wf_name}.pattern",
            suggestion=f"Valid patterns: {', '.join(VALID_PATTERNS)}"
        )

def _validate_stages(wf_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    stages = cfg.get("stages", [])
    if not stages:
        res.add_error("Workflow has no stages", path=f"workflows.{wf_name}.stages")
        return
    names = set()
    for i, st in enumerate(stages):
        name = st.get("name", f"stage_{i}")
        if name in names:
            res.add_error(f"Duplicate stage name: {name}", path=f"workflows.{wf_name}.stages[{i}]")
        names.add(name)
        for dep in st.get("depends_on", []):
            if dep not in names:
                res.add_error(f"Stage '{name}' depends on unknown stage: {dep}", path=f"workflows.{wf_name}.stages[{i}].depends_on")

def _validate_success(wf_name: str, cfg: Dict[str, Any], res: ValidationResult) -> None:
    if not cfg.get("success_criteria"):
        res.add_warning("No success criteria defined", path=f"workflows.{wf_name}", suggestion="Add success_criteria for better quality control")

def validate_workflow(wf_name: str, wf_config: Dict[str, Any]) -> ValidationResult:
    result = ValidationResult(valid=True, target=f"workflow:{wf_name}")
    _validate_pattern(wf_name, wf_config, result)
    _validate_stages(wf_name, wf_config, result)
    _validate_success(wf_name, wf_config, result)
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


def _print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)

def _format_issue(issue: ValidationIssue) -> List[str]:
    icon = "[INFO]"
    if issue.severity == Severity.ERROR:
        icon = "[ERROR]"
    elif issue.severity == Severity.WARNING:
        icon = "[WARN]"
    lines = [f"  {icon} {issue.message}"]
    if issue.path:
        lines.append(f"         Path: {issue.path}")
    if issue.suggestion:
        lines.append(f"         Fix: {issue.suggestion}")
    return lines

def print_validation_report(results: Dict[str, ValidationResult]):
    _print_header("AUTOMATION VALIDATION REPORT")
    total_errors = 0
    total_warnings = 0
    for name, result in results.items():
        status = "✓ VALID" if result.valid else "✗ INVALID"
        print(f"\n{name.upper()}: {status}")
        print("-" * 40)
        for issue in result.issues:
            if issue.severity == Severity.ERROR:
                total_errors += 1
            elif issue.severity == Severity.WARNING:
                total_warnings += 1
            for line in _format_issue(issue):
                print(line)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_errors} errors, {total_warnings} warnings")
    print("=" * 60)
    return total_errors == 0


RESET = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"

def _emit(kind: str, msg: str, style: str = "compact"):
    icon = {"success": "✅", "error": "❌", "warn": "⚠️", "info": "ℹ️"}[kind]
    color = {"success": GREEN, "error": RED, "warn": YELLOW, "info": BLUE}[kind]
    if style == "json":
        print(json.dumps({"level": kind, "message": msg}))
        return
    print(f"{color}{icon} {msg}{RESET}")

def _summarize(results: Dict[str, ValidationResult]) -> Dict[str, int]:
    errors = sum(r.error_count for r in results.values())
    warnings = sum(r.warning_count for r in results.values())
    infos = sum(len([i for i in r.issues if i.severity == Severity.INFO]) for r in results.values())
    return {"errors": errors, "warnings": warnings, "infos": infos}

if __name__ == "__main__":
    import argparse, os, json as _json
    parser = argparse.ArgumentParser(description="Validate automation assets")
    parser.add_argument("--style", choices=["compact", "json", "verbose"], default=os.getenv("ATLAS_OUTPUT_STYLE", "compact"))
    args = parser.parse_args()
    results = validate_all()
    if args.style == "verbose":
        success = print_validation_report(results)
        exit(0 if success else 1)
    summary = _summarize(results)
    if args.style == "json":
        print(_json.dumps({"summary": summary}, indent=2))
        exit(0 if summary["errors"] == 0 else 1)
    if summary["errors"] == 0:
        _emit("success", f"validation ok errors={summary['errors']} warnings={summary['warnings']}", args.style)
        exit(0)
    else:
        _emit("error", f"validation fail errors={summary['errors']} warnings={summary['warnings']}", args.style)
        exit(1)
