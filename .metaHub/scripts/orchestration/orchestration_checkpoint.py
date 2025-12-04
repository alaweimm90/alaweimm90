#!/usr/bin/env python3
"""
orchestration_checkpoint.py - Workflow State Preservation and Recovery

Manages checkpoints for multi-agent tool orchestration workflows:
- Creates immutable snapshots before tool handoffs
- Enables rollback to known-good states on failure
- Tracks decision history across tool boundaries
- Validates checkpoint integrity for recovery

Usage:
    python orchestration_checkpoint.py create --workflow "feature-x"
    python orchestration_checkpoint.py restore --id abc123
    python orchestration_checkpoint.py list --workflow "feature-x"
    python orchestration_checkpoint.py validate --id abc123
"""

import json
import os
import hashlib
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

import click
import yaml


class CheckpointStatus(Enum):
    """Checkpoint lifecycle states."""
    CREATED = "created"
    VALIDATED = "validated"
    RESTORED = "restored"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"


@dataclass
class WorkflowContext:
    """Context passed between tools in a workflow."""
    task_description: str
    relevant_files: List[str] = field(default_factory=list)
    prior_decisions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    codebase_context: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""
    tool_name: str
    timestamp: str
    action: str
    input_summary: str
    output_summary: str
    files_modified: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    hallucination_check: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationCheckpoint:
    """Complete checkpoint for workflow recovery."""
    checkpoint_id: str
    workflow_name: str
    correlation_id: str
    created_at: str
    status: str

    # Workflow state
    current_tool: str
    sequence_number: int
    context: Dict[str, Any]

    # History
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)

    # Recovery info
    rollback_instructions: str = ""
    parent_checkpoint_id: Optional[str] = None

    # Integrity
    checksum: str = ""
    file_snapshots: List[Dict[str, str]] = field(default_factory=list)


class OrchestrationCheckpointManager:
    """Manages orchestration workflow checkpoints."""

    CHECKPOINT_DIR = ".metaHub/orchestration/checkpoints"
    SNAPSHOT_DIR = ".metaHub/orchestration/snapshots"
    RETENTION_DAYS = 30

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.checkpoint_dir = self.base_path / self.CHECKPOINT_DIR
        self.snapshot_dir = self.base_path / self.SNAPSHOT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Load policy configuration
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

    def _generate_checksum(self, data: Dict[str, Any]) -> str:
        """Generate integrity checksum for checkpoint data."""
        # Exclude checksum field itself from calculation
        data_copy = {k: v for k, v in data.items() if k != 'checksum'}
        json_str = json.dumps(data_copy, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _snapshot_files(self, file_paths: List[str], checkpoint_id: str) -> List[Dict[str, str]]:
        """Create file snapshots for recovery."""
        snapshots = []
        snapshot_subdir = self.snapshot_dir / checkpoint_id
        snapshot_subdir.mkdir(parents=True, exist_ok=True)

        for file_path in file_paths:
            full_path = self.base_path / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = full_path.read_text(encoding='utf-8')
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                    # Save snapshot
                    safe_name = file_path.replace('/', '_').replace('\\', '_')
                    snapshot_file = snapshot_subdir / f"{safe_name}.snapshot"
                    snapshot_file.write_text(content, encoding='utf-8')

                    snapshots.append({
                        "path": file_path,
                        "hash": content_hash,
                        "snapshot_file": str(snapshot_file.relative_to(self.base_path)),
                        "size_bytes": len(content.encode('utf-8'))
                    })
                except (OSError, UnicodeDecodeError) as e:
                    snapshots.append({
                        "path": file_path,
                        "error": str(e)
                    })

        return snapshots

    def create_checkpoint(
        self,
        workflow_name: str,
        current_tool: str,
        context: WorkflowContext,
        tool_invocations: List[ToolInvocation] = None,
        correlation_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        rollback_instructions: str = ""
    ) -> OrchestrationCheckpoint:
        """Create a new orchestration checkpoint."""
        checkpoint_id = str(uuid.uuid4())[:8]
        correlation_id = correlation_id or str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Calculate sequence number
        sequence = len(tool_invocations) + 1 if tool_invocations else 1

        # Snapshot relevant files
        file_snapshots = self._snapshot_files(
            context.relevant_files,
            checkpoint_id
        )

        checkpoint = OrchestrationCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_name=workflow_name,
            correlation_id=correlation_id,
            created_at=timestamp,
            status=CheckpointStatus.CREATED.value,
            current_tool=current_tool,
            sequence_number=sequence,
            context=asdict(context),
            tool_invocations=[asdict(t) for t in (tool_invocations or [])],
            rollback_instructions=rollback_instructions,
            parent_checkpoint_id=parent_checkpoint_id,
            file_snapshots=file_snapshots
        )

        # Generate integrity checksum
        checkpoint_dict = asdict(checkpoint)
        checkpoint.checksum = self._generate_checksum(checkpoint_dict)

        # Save checkpoint
        self._save_checkpoint(checkpoint)

        return checkpoint

    def _save_checkpoint(self, checkpoint: OrchestrationCheckpoint) -> Path:
        """Save checkpoint to disk."""
        checkpoint_data = asdict(checkpoint)

        # Primary checkpoint file
        filename = f"{checkpoint.workflow_name}_{checkpoint.checkpoint_id}.json"
        filepath = self.checkpoint_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Update latest pointer for this workflow
        latest_file = self.checkpoint_dir / f"{checkpoint.workflow_name}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Update index
        self._update_index(checkpoint)

        return filepath

    def _update_index(self, checkpoint: OrchestrationCheckpoint):
        """Update checkpoint index for fast lookups."""
        index_file = self.checkpoint_dir / "index.json"

        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {"checkpoints": [], "workflows": {}}

        # Add to checkpoints list
        index["checkpoints"].append({
            "id": checkpoint.checkpoint_id,
            "workflow": checkpoint.workflow_name,
            "correlation_id": checkpoint.correlation_id,
            "created_at": checkpoint.created_at,
            "tool": checkpoint.current_tool,
            "sequence": checkpoint.sequence_number
        })

        # Update workflow mapping
        if checkpoint.workflow_name not in index["workflows"]:
            index["workflows"][checkpoint.workflow_name] = []
        index["workflows"][checkpoint.workflow_name].append(checkpoint.checkpoint_id)

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[OrchestrationCheckpoint]:
        """Load a checkpoint by ID."""
        # Search for checkpoint file
        for filepath in self.checkpoint_dir.glob(f"*_{checkpoint_id}.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._dict_to_checkpoint(data)

        return None

    def load_latest(self, workflow_name: str) -> Optional[OrchestrationCheckpoint]:
        """Load the latest checkpoint for a workflow."""
        latest_file = self.checkpoint_dir / f"{workflow_name}_latest.json"

        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._dict_to_checkpoint(data)

        return None

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> OrchestrationCheckpoint:
        """Convert dictionary to checkpoint dataclass."""
        return OrchestrationCheckpoint(
            checkpoint_id=data["checkpoint_id"],
            workflow_name=data["workflow_name"],
            correlation_id=data["correlation_id"],
            created_at=data["created_at"],
            status=data["status"],
            current_tool=data["current_tool"],
            sequence_number=data["sequence_number"],
            context=data["context"],
            tool_invocations=data.get("tool_invocations", []),
            rollback_instructions=data.get("rollback_instructions", ""),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            checksum=data.get("checksum", ""),
            file_snapshots=data.get("file_snapshots", [])
        )

    def validate_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Validate checkpoint integrity."""
        checkpoint = self.load_checkpoint(checkpoint_id)

        if not checkpoint:
            return {
                "valid": False,
                "error": "Checkpoint not found",
                "checkpoint_id": checkpoint_id
            }

        result = {
            "valid": True,
            "checkpoint_id": checkpoint_id,
            "workflow": checkpoint.workflow_name,
            "created_at": checkpoint.created_at,
            "checks": {}
        }

        # Verify checksum
        checkpoint_dict = asdict(checkpoint)
        expected_checksum = self._generate_checksum(checkpoint_dict)
        checksum_valid = checkpoint.checksum == expected_checksum
        result["checks"]["checksum"] = {
            "passed": checksum_valid,
            "expected": expected_checksum,
            "actual": checkpoint.checksum
        }
        if not checksum_valid:
            result["valid"] = False

        # Verify file snapshots exist
        snapshot_checks = []
        for snapshot in checkpoint.file_snapshots:
            if "error" in snapshot:
                snapshot_checks.append({
                    "path": snapshot["path"],
                    "exists": False,
                    "error": snapshot["error"]
                })
                continue

            snapshot_path = self.base_path / snapshot["snapshot_file"]
            exists = snapshot_path.exists()

            if exists:
                # Verify snapshot content hash
                content = snapshot_path.read_text(encoding='utf-8')
                current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                hash_match = current_hash == snapshot["hash"]
            else:
                hash_match = False

            snapshot_checks.append({
                "path": snapshot["path"],
                "exists": exists,
                "hash_valid": hash_match
            })

            if not exists or not hash_match:
                result["valid"] = False

        result["checks"]["file_snapshots"] = snapshot_checks

        # Update checkpoint status
        new_status = CheckpointStatus.VALIDATED if result["valid"] else CheckpointStatus.CORRUPTED
        self._update_status(checkpoint_id, new_status)
        result["status"] = new_status.value

        return result

    def _update_status(self, checkpoint_id: str, status: CheckpointStatus):
        """Update checkpoint status."""
        checkpoint = self.load_checkpoint(checkpoint_id)
        if checkpoint:
            checkpoint.status = status.value
            self._save_checkpoint(checkpoint)

    def restore_checkpoint(self, checkpoint_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Restore files from a checkpoint."""
        checkpoint = self.load_checkpoint(checkpoint_id)

        if not checkpoint:
            return {
                "success": False,
                "error": "Checkpoint not found",
                "checkpoint_id": checkpoint_id
            }

        # First validate
        validation = self.validate_checkpoint(checkpoint_id)
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Checkpoint validation failed",
                "validation": validation
            }

        result = {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "workflow": checkpoint.workflow_name,
            "dry_run": dry_run,
            "files_restored": []
        }

        for snapshot in checkpoint.file_snapshots:
            if "error" in snapshot:
                continue

            snapshot_path = self.base_path / snapshot["snapshot_file"]
            target_path = self.base_path / snapshot["path"]

            if not snapshot_path.exists():
                result["files_restored"].append({
                    "path": snapshot["path"],
                    "status": "skipped",
                    "reason": "snapshot missing"
                })
                continue

            if dry_run:
                result["files_restored"].append({
                    "path": snapshot["path"],
                    "status": "would_restore",
                    "from": str(snapshot_path)
                })
            else:
                try:
                    # Create parent directories if needed
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy snapshot to target
                    shutil.copy2(snapshot_path, target_path)

                    result["files_restored"].append({
                        "path": snapshot["path"],
                        "status": "restored"
                    })
                except OSError as e:
                    result["files_restored"].append({
                        "path": snapshot["path"],
                        "status": "error",
                        "error": str(e)
                    })
                    result["success"] = False

        if not dry_run and result["success"]:
            self._update_status(checkpoint_id, CheckpointStatus.RESTORED)

        return result

    def list_checkpoints(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List checkpoints with optional filtering."""
        index_file = self.checkpoint_dir / "index.json"

        if not index_file.exists():
            return []

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        checkpoints = index.get("checkpoints", [])

        if workflow_name:
            checkpoints = [c for c in checkpoints if c["workflow"] == workflow_name]

        # Sort by creation time descending
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)

        return checkpoints[:limit]

    def cleanup_expired(self) -> Dict[str, Any]:
        """Remove checkpoints older than retention period."""
        retention_days = self.policy.get("error_recovery", {}).get(
            "checkpoint", {}
        ).get("retention_days", self.RETENTION_DAYS)

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        result = {
            "removed": [],
            "retained": [],
            "errors": []
        }

        for filepath in self.checkpoint_dir.glob("*.json"):
            if filepath.name in ["index.json"] or filepath.name.endswith("_latest.json"):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                created_at = datetime.fromisoformat(data["created_at"])

                if created_at < cutoff_date:
                    # Remove checkpoint file
                    filepath.unlink()

                    # Remove associated snapshots
                    snapshot_dir = self.snapshot_dir / data["checkpoint_id"]
                    if snapshot_dir.exists():
                        shutil.rmtree(snapshot_dir)

                    result["removed"].append(data["checkpoint_id"])
                else:
                    result["retained"].append(data["checkpoint_id"])

            except (json.JSONDecodeError, KeyError, OSError) as e:
                result["errors"].append({
                    "file": str(filepath),
                    "error": str(e)
                })

        # Rebuild index after cleanup
        self._rebuild_index()

        return result

    def _rebuild_index(self):
        """Rebuild checkpoint index from files."""
        index = {"checkpoints": [], "workflows": {}}

        for filepath in self.checkpoint_dir.glob("*.json"):
            if filepath.name in ["index.json"] or filepath.name.endswith("_latest.json"):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                index["checkpoints"].append({
                    "id": data["checkpoint_id"],
                    "workflow": data["workflow_name"],
                    "correlation_id": data["correlation_id"],
                    "created_at": data["created_at"],
                    "tool": data["current_tool"],
                    "sequence": data["sequence_number"]
                })

                workflow = data["workflow_name"]
                if workflow not in index["workflows"]:
                    index["workflows"][workflow] = []
                index["workflows"][workflow].append(data["checkpoint_id"])

            except (json.JSONDecodeError, KeyError):
                continue

        index_file = self.checkpoint_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)


@click.group()
def cli():
    """Orchestration checkpoint management for multi-agent workflows."""
    pass


@cli.command()
@click.option('--workflow', '-w', required=True, help='Workflow name')
@click.option('--tool', '-t', required=True, help='Current tool name')
@click.option('--task', required=True, help='Task description')
@click.option('--files', '-f', multiple=True, help='Relevant files to snapshot')
@click.option('--correlation-id', help='Correlation ID for tracking')
@click.option('--parent', help='Parent checkpoint ID')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def create(workflow: str, tool: str, task: str, files: tuple,
           correlation_id: Optional[str], parent: Optional[str],
           json_output: bool):
    """Create a new orchestration checkpoint."""
    try:
        mgr = OrchestrationCheckpointManager()

        context = WorkflowContext(
            task_description=task,
            relevant_files=list(files)
        )

        checkpoint = mgr.create_checkpoint(
            workflow_name=workflow,
            current_tool=tool,
            context=context,
            correlation_id=correlation_id,
            parent_checkpoint_id=parent
        )

        if json_output:
            click.echo(json.dumps(asdict(checkpoint), indent=2))
        else:
            click.echo(f"Checkpoint created: {checkpoint.checkpoint_id}")
            click.echo(f"  Workflow: {checkpoint.workflow_name}")
            click.echo(f"  Tool: {checkpoint.current_tool}")
            click.echo(f"  Correlation ID: {checkpoint.correlation_id}")
            click.echo(f"  Files snapshotted: {len(checkpoint.file_snapshots)}")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--id', 'checkpoint_id', required=True, help='Checkpoint ID')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def validate(checkpoint_id: str, json_output: bool):
    """Validate checkpoint integrity."""
    try:
        mgr = OrchestrationCheckpointManager()
        result = mgr.validate_checkpoint(checkpoint_id)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            status = "VALID" if result["valid"] else "INVALID"
            click.echo(f"Checkpoint {checkpoint_id}: {status}")

            for check_name, check_result in result.get("checks", {}).items():
                if isinstance(check_result, dict) and "passed" in check_result:
                    status_icon = "[OK]" if check_result["passed"] else "[FAIL]"
                    click.echo(f"  {status_icon} {check_name}")
                elif isinstance(check_result, list):
                    click.echo(f"  {check_name}:")
                    for item in check_result:
                        status_icon = "[OK]" if item.get("exists") and item.get("hash_valid", True) else "[FAIL]"
                        click.echo(f"    {status_icon} {item['path']}")

        raise SystemExit(0 if result["valid"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--id', 'checkpoint_id', required=True, help='Checkpoint ID')
@click.option('--dry-run/--execute', default=True, help='Preview or execute restore')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def restore(checkpoint_id: str, dry_run: bool, json_output: bool):
    """Restore files from a checkpoint."""
    try:
        mgr = OrchestrationCheckpointManager()
        result = mgr.restore_checkpoint(checkpoint_id, dry_run=dry_run)

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            mode = "[DRY RUN]" if dry_run else "[EXECUTING]"
            status = "SUCCESS" if result["success"] else "FAILED"
            click.echo(f"{mode} Restore checkpoint {checkpoint_id}: {status}")

            for file_info in result.get("files_restored", []):
                click.echo(f"  {file_info['status']}: {file_info['path']}")

        raise SystemExit(0 if result["success"] else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('list')
@click.option('--workflow', '-w', help='Filter by workflow name')
@click.option('--limit', '-n', default=20, help='Maximum results')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def list_cmd(workflow: Optional[str], limit: int, json_output: bool):
    """List orchestration checkpoints."""
    try:
        mgr = OrchestrationCheckpointManager()
        checkpoints = mgr.list_checkpoints(workflow_name=workflow, limit=limit)

        if json_output:
            click.echo(json.dumps(checkpoints, indent=2))
        else:
            if not checkpoints:
                click.echo("No checkpoints found")
            else:
                click.echo(f"{'ID':<10} {'Workflow':<20} {'Tool':<15} {'Created':<20}")
                click.echo("-" * 70)
                for cp in checkpoints:
                    click.echo(f"{cp['id']:<10} {cp['workflow']:<20} {cp['tool']:<15} {cp['created_at'][:19]}")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
def cleanup(json_output: bool):
    """Remove expired checkpoints."""
    try:
        mgr = OrchestrationCheckpointManager()
        result = mgr.cleanup_expired()

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("Cleanup complete:")
            click.echo(f"  Removed: {len(result['removed'])} checkpoints")
            click.echo(f"  Retained: {len(result['retained'])} checkpoints")
            if result['errors']:
                click.echo(f"  Errors: {len(result['errors'])}")

        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
