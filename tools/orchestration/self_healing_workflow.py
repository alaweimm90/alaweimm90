#!/usr/bin/env python3
"""
self_healing_workflow.py - Error Recovery and Graceful Degradation for Orchestration

Implements self-healing capabilities for multi-agent workflows:
- Automatic retry with exponential backoff
- Checkpoint-based recovery
- Graceful degradation levels
- Alternative tool fallback
- Error pattern recognition

Usage:
    python self_healing_workflow.py recover --workflow "feature-x" --from-checkpoint abc123
    python self_healing_workflow.py retry --tool cline --action "implement feature"
    python self_healing_workflow.py status --workflow "feature-x"
    python self_healing_workflow.py degrade --level 2
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

import click
import yaml


class ErrorCategory(Enum):
    """Categories of errors for recovery strategy."""
    TRANSIENT = "transient"      # Retry immediately
    CONTEXT = "context"          # May need context reduction
    PERMANENT = "permanent"      # Don't retry, escalate
    UNKNOWN = "unknown"          # Conservative retry


class DegradationLevel(Enum):
    """Graceful degradation levels."""
    FULL = 1           # All capabilities available
    REDUCED = 2        # Sequential execution, reduced context
    PRIMARY_ONLY = 3   # Only primary tools available
    SAFE_MODE = 4      # Single tool, human approval required


class RecoveryAction(Enum):
    """Recovery actions available."""
    RETRY_SAME = "retry_same_tool"
    ALTERNATIVE_TOOL = "alternative_tool"
    SIMPLIFIED_TASK = "simplified_task"
    HUMAN_ESCALATION = "human_escalation"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    CONTEXT_COMPRESS = "context_compress"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    attempt_id: str
    timestamp: str
    error_category: str
    action_taken: str
    success: bool
    tool: Optional[str] = None
    workflow: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Current state of a workflow for recovery purposes."""
    workflow_name: str
    status: str
    degradation_level: int
    current_tool: Optional[str]
    last_checkpoint_id: Optional[str]
    error_count: int
    recovery_attempts: List[Dict[str, Any]]
    created_at: str
    updated_at: str


class SelfHealingWorkflow:
    """Self-healing capabilities for orchestration workflows."""

    STATE_DIR = ".metaHub/orchestration/state"
    RECOVERY_LOG = ".metaHub/orchestration/recovery.jsonl"

    # Error patterns for classification
    ERROR_PATTERNS = {
        ErrorCategory.TRANSIENT: [
            "timeout", "rate_limit", "connection_refused",
            "service_unavailable", "temporary", "retry"
        ],
        ErrorCategory.CONTEXT: [
            "context_window_exceeded", "memory_error", "token_limit",
            "context_too_large", "truncated"
        ],
        ErrorCategory.PERMANENT: [
            "invalid_request", "authentication_failed", "not_found",
            "permission_denied", "invalid_api_key"
        ]
    }

    # Tool fallback chains
    TOOL_FALLBACKS = {
        "claude_code": ["cline", "aider"],
        "cline": ["cursor", "aider"],
        "cursor": ["aider", "cline"],
        "aider": ["cline", "cursor"],
        "kilo_code": ["claude_code", "cline"],
        "blackbox": ["cursor", "cline"],
        "windsurf": ["cursor", "cline"],
    }

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.state_dir = self.base_path / self.STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.recovery_log = self.base_path / self.RECOVERY_LOG
        self.policy = self._load_policy()

        # Load recovery config from policy
        recovery_config = self.policy.get("error_recovery", {})
        retry_config = recovery_config.get("retry", {})

        self.max_retries = retry_config.get("max_attempts", 3)
        self.base_delay = retry_config.get("base_delay_seconds", 5)
        self.max_delay = retry_config.get("max_delay_seconds", 60)
        self.backoff_type = retry_config.get("backoff", "exponential")

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

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def classify_error(self, error_message: str) -> ErrorCategory:
        """Classify an error into a category for recovery strategy."""
        error_lower = error_message.lower()

        for category, patterns in self.ERROR_PATTERNS.items():
            if any(pattern in error_lower for pattern in patterns):
                return category

        return ErrorCategory.UNKNOWN

    def calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with backoff."""
        if self.backoff_type == "exponential":
            delay = self.base_delay * (2 ** attempt)
        elif self.backoff_type == "linear":
            delay = self.base_delay * (attempt + 1)
        else:
            delay = self.base_delay

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, delay * 0.1)
        delay += jitter

        return min(delay, self.max_delay)

    def get_fallback_tool(self, current_tool: str) -> Optional[str]:
        """Get fallback tool for a failed tool."""
        fallbacks = self.TOOL_FALLBACKS.get(current_tool, [])
        return fallbacks[0] if fallbacks else None

    def get_workflow_state(self, workflow_name: str) -> Optional[WorkflowState]:
        """Load workflow state from disk."""
        state_file = self.state_dir / f"{workflow_name}.json"

        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return WorkflowState(**data)

        return None

    def save_workflow_state(self, state: WorkflowState):
        """Save workflow state to disk."""
        state.updated_at = datetime.now().isoformat()
        state_file = self.state_dir / f"{state.workflow_name}.json"

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=2)

    def create_workflow_state(self, workflow_name: str, tool: Optional[str] = None) -> WorkflowState:
        """Create new workflow state."""
        state = WorkflowState(
            workflow_name=workflow_name,
            status="active",
            degradation_level=DegradationLevel.FULL.value,
            current_tool=tool,
            last_checkpoint_id=None,
            error_count=0,
            recovery_attempts=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.save_workflow_state(state)
        return state

    def log_recovery_attempt(self, attempt: RecoveryAttempt):
        """Log a recovery attempt."""
        with open(self.recovery_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(attempt)) + '\n')

    def determine_recovery_action(
        self,
        error_category: ErrorCategory,
        state: WorkflowState
    ) -> RecoveryAction:
        """Determine best recovery action based on error and state."""
        # Get fallback chain from policy
        fallback_chain = self.policy.get("error_recovery", {}).get(
            "fallback", {}
        ).get("chain", [
            "retry_same_tool",
            "alternative_tool",
            "simplified_task",
            "human_escalation"
        ])

        # Transient errors: retry same tool
        if error_category == ErrorCategory.TRANSIENT:
            if state.error_count < self.max_retries:
                return RecoveryAction.RETRY_SAME
            return RecoveryAction.ALTERNATIVE_TOOL

        # Context errors: compress or simplify
        if error_category == ErrorCategory.CONTEXT:
            return RecoveryAction.CONTEXT_COMPRESS

        # Permanent errors: try alternative or escalate
        if error_category == ErrorCategory.PERMANENT:
            if state.degradation_level < DegradationLevel.PRIMARY_ONLY.value:
                return RecoveryAction.ALTERNATIVE_TOOL
            return RecoveryAction.HUMAN_ESCALATION

        # Unknown: follow fallback chain based on attempts
        attempt_count = len(state.recovery_attempts)
        if attempt_count < len(fallback_chain):
            action_name = fallback_chain[attempt_count]
            try:
                return RecoveryAction(action_name)
            except ValueError:
                pass

        return RecoveryAction.HUMAN_ESCALATION

    def execute_recovery(
        self,
        workflow_name: str,
        error_message: str,
        tool: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute recovery for a failed workflow step."""
        # Get or create workflow state
        state = self.get_workflow_state(workflow_name)
        if not state:
            state = self.create_workflow_state(workflow_name, tool)

        # Classify error
        error_category = self.classify_error(error_message)

        # Increment error count
        state.error_count += 1

        # Determine recovery action
        action = self.determine_recovery_action(error_category, state)

        # Create recovery attempt record
        attempt = RecoveryAttempt(
            attempt_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            error_category=error_category.value,
            action_taken=action.value,
            success=False,
            tool=tool,
            workflow=workflow_name,
            details={"error_message": error_message}
        )

        result = {
            "workflow": workflow_name,
            "error_category": error_category.value,
            "action": action.value,
            "success": False,
            "next_steps": []
        }

        # Execute recovery action
        if action == RecoveryAction.RETRY_SAME:
            delay = self.calculate_delay(state.error_count - 1)
            result["delay_seconds"] = delay
            result["next_steps"] = [
                f"Wait {delay:.1f} seconds",
                f"Retry with {tool or 'same tool'}"
            ]
            result["success"] = True
            attempt.success = True

        elif action == RecoveryAction.ALTERNATIVE_TOOL:
            fallback = self.get_fallback_tool(tool) if tool else None
            if fallback:
                result["fallback_tool"] = fallback
                result["next_steps"] = [
                    f"Switch to {fallback}",
                    "Retry task with new tool"
                ]
                result["success"] = True
                attempt.success = True
                attempt.details["fallback_tool"] = fallback
            else:
                result["next_steps"] = ["No fallback available", "Consider human escalation"]

        elif action == RecoveryAction.CONTEXT_COMPRESS:
            result["next_steps"] = [
                "Compress context to fit limits",
                "Preserve recent messages and key decisions",
                "Retry with reduced context"
            ]
            result["success"] = True
            attempt.success = True

        elif action == RecoveryAction.CHECKPOINT_RESTORE:
            if state.last_checkpoint_id:
                result["checkpoint_id"] = state.last_checkpoint_id
                result["next_steps"] = [
                    f"Restore from checkpoint {state.last_checkpoint_id}",
                    "Resume workflow from last known good state"
                ]
                result["success"] = True
                attempt.success = True
            else:
                result["next_steps"] = ["No checkpoint available"]

        elif action == RecoveryAction.SIMPLIFIED_TASK:
            result["next_steps"] = [
                "Break task into smaller steps",
                "Reduce scope of current operation",
                "Retry with simplified task"
            ]
            result["success"] = True
            attempt.success = True

        elif action == RecoveryAction.HUMAN_ESCALATION:
            result["next_steps"] = [
                "Human intervention required",
                "Review error details",
                "Decide on manual recovery or abort"
            ]
            # Escalation doesn't count as success
            attempt.success = False

        # Update degradation level if needed
        if state.error_count >= self.max_retries:
            new_level = min(state.degradation_level + 1, DegradationLevel.SAFE_MODE.value)
            if new_level != state.degradation_level:
                state.degradation_level = new_level
                result["degradation_level"] = new_level
                result["degradation_message"] = self._get_degradation_message(new_level)

        # Record attempt
        state.recovery_attempts.append(asdict(attempt))
        self.save_workflow_state(state)
        self.log_recovery_attempt(attempt)

        return result

    def _get_degradation_message(self, level: int) -> str:
        """Get message describing degradation level."""
        messages = {
            1: "Full capabilities available",
            2: "Reduced mode: Sequential execution, reduced context windows",
            3: "Primary tools only: claude_code, aider, cline available",
            4: "Safe mode: Single tool (aider), human approval required"
        }
        return messages.get(level, "Unknown degradation level")

    def recover_from_checkpoint(
        self,
        workflow_name: str,
        checkpoint_id: str
    ) -> Dict[str, Any]:
        """Recover workflow from a checkpoint."""
        # Import checkpoint manager
        from orchestration_checkpoint import OrchestrationCheckpointManager

        checkpoint_mgr = OrchestrationCheckpointManager(self.base_path)

        # Validate checkpoint
        validation = checkpoint_mgr.validate_checkpoint(checkpoint_id)
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Checkpoint validation failed",
                "validation": validation
            }

        # Load checkpoint
        checkpoint = checkpoint_mgr.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return {
                "success": False,
                "error": "Checkpoint not found"
            }

        # Restore files (dry run first)
        restore_result = checkpoint_mgr.restore_checkpoint(checkpoint_id, dry_run=True)

        # Update workflow state
        state = self.get_workflow_state(workflow_name)
        if state:
            state.last_checkpoint_id = checkpoint_id
            state.error_count = 0  # Reset error count on successful recovery
            self.save_workflow_state(state)

        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "workflow": workflow_name,
            "context_restored": checkpoint.context,
            "files_to_restore": restore_result.get("files_restored", []),
            "next_steps": [
                "Review restored context",
                f"Resume from tool: {checkpoint.current_tool}",
                f"Sequence number: {checkpoint.sequence_number}"
            ]
        }

    def set_degradation_level(self, workflow_name: str, level: int) -> Dict[str, Any]:
        """Manually set degradation level for a workflow."""
        if level < 1 or level > 4:
            return {
                "success": False,
                "error": "Level must be 1-4"
            }

        state = self.get_workflow_state(workflow_name)
        if not state:
            state = self.create_workflow_state(workflow_name)

        old_level = state.degradation_level
        state.degradation_level = level
        self.save_workflow_state(state)

        # Get available tools for this level
        degradation_config = self.policy.get("degradation_levels", {})
        level_key = f"level_{level}_{'full' if level == 1 else 'reduced' if level == 2 else 'primary_only' if level == 3 else 'safe_mode'}"

        available_tools = ["all"]
        if level >= 3:
            available_tools = degradation_config.get(
                f"level_{level}_primary_only", {}
            ).get("tools", ["claude_code", "aider", "cline"])
        if level == 4:
            available_tools = degradation_config.get(
                "level_4_safe_mode", {}
            ).get("tools", ["aider"])

        return {
            "success": True,
            "workflow": workflow_name,
            "old_level": old_level,
            "new_level": level,
            "message": self._get_degradation_message(level),
            "available_tools": available_tools
        }

    def get_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get current recovery status for a workflow."""
        state = self.get_workflow_state(workflow_name)

        if not state:
            return {
                "exists": False,
                "workflow": workflow_name
            }

        return {
            "exists": True,
            "workflow": workflow_name,
            "status": state.status,
            "degradation_level": state.degradation_level,
            "degradation_message": self._get_degradation_message(state.degradation_level),
            "current_tool": state.current_tool,
            "last_checkpoint": state.last_checkpoint_id,
            "error_count": state.error_count,
            "recovery_attempts": len(state.recovery_attempts),
            "created_at": state.created_at,
            "updated_at": state.updated_at
        }

    def reset_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Reset workflow state to clean slate."""
        state_file = self.state_dir / f"{workflow_name}.json"

        existed = state_file.exists()
        if existed:
            state_file.unlink()

        return {
            "success": True,
            "workflow": workflow_name,
            "was_reset": existed,
            "message": "Workflow state cleared" if existed else "No state to clear"
        }


@click.group()
def cli():
    """Self-healing workflow management for orchestration."""
    pass


@cli.command()
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--error', '-e', required=True, help='Error message')
@click.option('--tool', '-t', help='Tool that failed')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def recover(workflow: str, error: str, tool: Optional[str], json_output: bool):
    """Execute recovery for a failed workflow step."""
    try:
        healer = SelfHealingWorkflow()
        result = healer.execute_recovery(workflow, error, tool)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            status = "Recovery planned" if result["success"] else "Recovery failed"
            click.echo(f"{status} for workflow '{workflow}'")
            click.echo(f"  Error category: {result['error_category']}")
            click.echo(f"  Action: {result['action']}")

            if result.get("delay_seconds"):
                click.echo(f"  Delay: {result['delay_seconds']:.1f}s")
            if result.get("fallback_tool"):
                click.echo(f"  Fallback tool: {result['fallback_tool']}")

            click.echo("\nNext steps:")
            for step in result.get("next_steps", []):
                click.echo(f"  - {step}")

            if result.get("degradation_message"):
                click.echo(f"\n[!] {result['degradation_message']}")

        raise SystemExit(0 if result["success"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('from-checkpoint')
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--checkpoint', '-c', required=True, help='Checkpoint ID')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def from_checkpoint(workflow: str, checkpoint: str, json_output: bool):
    """Recover workflow from a checkpoint."""
    try:
        healer = SelfHealingWorkflow()
        result = healer.recover_from_checkpoint(workflow, checkpoint)

        if json_output:
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            if result["success"]:
                click.echo(f"Recovery from checkpoint {checkpoint} prepared")
                click.echo(f"  Workflow: {workflow}")
                click.echo(f"  Files to restore: {len(result.get('files_to_restore', []))}")
                click.echo("\nNext steps:")
                for step in result.get("next_steps", []):
                    click.echo(f"  - {step}")
            else:
                click.echo(f"Recovery failed: {result.get('error')}")

        raise SystemExit(0 if result["success"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def status(workflow: str, json_output: bool):
    """Get recovery status for a workflow."""
    try:
        healer = SelfHealingWorkflow()
        result = healer.get_status(workflow)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            if result["exists"]:
                click.echo(f"Workflow: {workflow}")
                click.echo(f"  Status: {result['status']}")
                click.echo(f"  Degradation: Level {result['degradation_level']} - {result['degradation_message']}")
                click.echo(f"  Current tool: {result['current_tool'] or 'None'}")
                click.echo(f"  Last checkpoint: {result['last_checkpoint'] or 'None'}")
                click.echo(f"  Error count: {result['error_count']}")
                click.echo(f"  Recovery attempts: {result['recovery_attempts']}")
            else:
                click.echo(f"No state found for workflow '{workflow}'")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--level', '-l', required=True, type=int,
              help='Degradation level (1-4)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def degrade(workflow: str, level: int, json_output: bool):
    """Set degradation level for a workflow."""
    try:
        healer = SelfHealingWorkflow()
        result = healer.set_degradation_level(workflow, level)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            if result["success"]:
                click.echo(f"Degradation level changed for '{workflow}'")
                click.echo(f"  Old level: {result['old_level']}")
                click.echo(f"  New level: {result['new_level']}")
                click.echo(f"  {result['message']}")
                click.echo(f"  Available tools: {', '.join(result['available_tools'])}")
            else:
                click.echo(f"Failed: {result.get('error')}")

        raise SystemExit(0 if result["success"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def reset(workflow: str, json_output: bool):
    """Reset workflow state to clean slate."""
    try:
        healer = SelfHealingWorkflow()
        result = healer.reset_workflow(workflow)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result["message"])

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
