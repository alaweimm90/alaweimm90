#!/usr/bin/env python3
"""
checkpoint.py - Weekly Drift Detection and Compliance Tracking

Detects configuration drift between catalog snapshots:
- Compares current state to previous checkpoint
- Identifies new, deleted, and changed repositories
- Tracks compliance trends over time
- Generates drift reports in multiple formats

Usage:
    python checkpoint.py                    # Generate checkpoint and detect drift
    python checkpoint.py --baseline         # Create new baseline (no comparison)
    python checkpoint.py --report markdown  # Generate markdown report
"""

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import click
import yaml


class CheckpointManager:
    """Manages drift detection and compliance checkpoints."""

    CHECKPOINT_DIR = ".metaHub/checkpoints"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.checkpoint_dir = self.base_path / self.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_state: Dict[str, Any] = {}
        self.previous_state: Optional[Dict[str, Any]] = None
        self.drift: Dict[str, Any] = {}

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

    def generate_current_state(self) -> Dict[str, Any]:
        """Generate current catalog snapshot."""
        org_path = self.base_path / "organizations"

        self.current_state = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "checksum": "",
            "organizations": {},
            "repositories": {},
            "summary": {
                "total_organizations": 0,
                "total_repositories": 0,
                "compliant": 0,
                "non_compliant": 0,
                "by_tier": {1: 0, 2: 0, 3: 0, 4: 0}
            }
        }

        if not org_path.exists():
            return self.current_state

        # Scan organizations
        for org_dir in sorted(org_path.iterdir()):
            if not org_dir.is_dir() or org_dir.name.startswith('.'):
                continue

            self.current_state["summary"]["total_organizations"] += 1
            org_data = {"repos": {}}

            # Scan repositories
            for repo_dir in sorted(org_dir.iterdir()):
                if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                    continue

                repo_key = f"{org_dir.name}/{repo_dir.name}"
                repo_data = self._scan_repo(repo_dir, org_dir.name)

                org_data["repos"][repo_dir.name] = repo_data
                self.current_state["repositories"][repo_key] = repo_data
                self.current_state["summary"]["total_repositories"] += 1

                # Update compliance counts
                if repo_data["compliant"]:
                    self.current_state["summary"]["compliant"] += 1
                else:
                    self.current_state["summary"]["non_compliant"] += 1

                # Update tier counts
                tier = repo_data.get("tier", 4)
                self.current_state["summary"]["by_tier"][tier] = \
                    self.current_state["summary"]["by_tier"].get(tier, 0) + 1

            self.current_state["organizations"][org_dir.name] = org_data

        # Generate checksum for state comparison
        state_json = json.dumps(self.current_state["repositories"], sort_keys=True)
        self.current_state["checksum"] = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        return self.current_state

    def _scan_repo(self, repo_dir: Path, org_name: str) -> Dict[str, Any]:
        """Scan a single repository and return its state."""
        repo_data = {
            "name": repo_dir.name,
            "organization": org_name,
            "full_name": f"{org_name}/{repo_dir.name}",
            "type": "unknown",
            "language": "unknown",
            "tier": 4,
            "status": "unknown",
            "compliant": False,
            "compliance_checks": {},
            "files_hash": ""
        }

        # Read metadata
        meta_file = repo_dir / ".meta" / "repo.yaml"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f) or {}
                repo_data["type"] = metadata.get("type", "unknown")
                repo_data["language"] = metadata.get("language", "unknown")
                repo_data["tier"] = metadata.get("tier", 4)
                repo_data["status"] = metadata.get("status", "unknown")
                repo_data["compliance_checks"]["has_metadata"] = True
            except (yaml.YAMLError, IOError):
                repo_data["compliance_checks"]["has_metadata"] = False
        else:
            repo_data["compliance_checks"]["has_metadata"] = False

        # Check compliance items
        repo_data["compliance_checks"]["has_readme"] = (repo_dir / "README.md").exists()
        repo_data["compliance_checks"]["has_codeowners"] = \
            (repo_dir / ".github" / "CODEOWNERS").exists()
        repo_data["compliance_checks"]["has_ci"] = \
            (repo_dir / ".github" / "workflows").exists()
        repo_data["compliance_checks"]["has_tests"] = \
            (repo_dir / "tests").is_dir() or (repo_dir / "test").is_dir()

        # Determine overall compliance based on tier
        tier = repo_data["tier"]
        required_checks = {
            1: ["has_metadata", "has_readme", "has_codeowners", "has_ci", "has_tests"],
            2: ["has_metadata", "has_readme", "has_codeowners", "has_ci"],
            3: ["has_metadata", "has_readme"],
            4: ["has_readme"]
        }

        required = required_checks.get(tier, required_checks[4])
        repo_data["compliant"] = all(
            repo_data["compliance_checks"].get(check, False)
            for check in required
        )

        # Generate hash of key files for change detection
        key_files = [".meta/repo.yaml", "README.md", ".github/CODEOWNERS"]
        file_contents = []
        for rel_path in key_files:
            file_path = repo_dir / rel_path
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    file_contents.append(f"{rel_path}:{hashlib.md5(content.encode()).hexdigest()}")
                except:
                    pass
        repo_data["files_hash"] = hashlib.md5('|'.join(file_contents).encode()).hexdigest()[:8]

        return repo_data

    def load_previous_checkpoint(self, checkpoint_file: Optional[Path] = None) -> bool:
        """Load the most recent checkpoint for comparison."""
        if checkpoint_file:
            target_file = checkpoint_file
        else:
            # Find most recent checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.json"), reverse=True)
            if not checkpoints:
                return False
            target_file = checkpoints[0]

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                self.previous_state = json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def detect_drift(self) -> Dict[str, Any]:
        """Compare current and previous states to detect drift."""
        self.drift = {
            "timestamp": datetime.now().isoformat(),
            "previous_checkpoint": self.previous_state.get("timestamp") if self.previous_state else None,
            "current_checkpoint": self.current_state.get("timestamp"),
            "has_drift": False,
            "changes": {
                "new_repos": [],
                "deleted_repos": [],
                "changed_repos": [],
                "compliance_improved": [],
                "compliance_degraded": []
            },
            "summary": {
                "new_count": 0,
                "deleted_count": 0,
                "changed_count": 0,
                "improved_count": 0,
                "degraded_count": 0
            }
        }

        if not self.previous_state:
            self.drift["has_drift"] = True
            self.drift["changes"]["new_repos"] = list(self.current_state.get("repositories", {}).keys())
            self.drift["summary"]["new_count"] = len(self.drift["changes"]["new_repos"])
            return self.drift

        prev_repos = set(self.previous_state.get("repositories", {}).keys())
        curr_repos = set(self.current_state.get("repositories", {}).keys())

        # New repositories
        new_repos = curr_repos - prev_repos
        self.drift["changes"]["new_repos"] = list(new_repos)
        self.drift["summary"]["new_count"] = len(new_repos)

        # Deleted repositories
        deleted_repos = prev_repos - curr_repos
        self.drift["changes"]["deleted_repos"] = list(deleted_repos)
        self.drift["summary"]["deleted_count"] = len(deleted_repos)

        # Changed repositories
        for repo_key in curr_repos & prev_repos:
            curr_data = self.current_state["repositories"][repo_key]
            prev_data = self.previous_state["repositories"][repo_key]

            # Check for changes
            if curr_data.get("files_hash") != prev_data.get("files_hash"):
                self.drift["changes"]["changed_repos"].append({
                    "repo": repo_key,
                    "previous_hash": prev_data.get("files_hash"),
                    "current_hash": curr_data.get("files_hash")
                })
                self.drift["summary"]["changed_count"] += 1

            # Check compliance changes
            was_compliant = prev_data.get("compliant", False)
            is_compliant = curr_data.get("compliant", False)

            if not was_compliant and is_compliant:
                self.drift["changes"]["compliance_improved"].append(repo_key)
                self.drift["summary"]["improved_count"] += 1
            elif was_compliant and not is_compliant:
                self.drift["changes"]["compliance_degraded"].append(repo_key)
                self.drift["summary"]["degraded_count"] += 1

        # Determine if there's meaningful drift
        self.drift["has_drift"] = (
            self.drift["summary"]["new_count"] > 0 or
            self.drift["summary"]["deleted_count"] > 0 or
            self.drift["summary"]["changed_count"] > 0 or
            self.drift["summary"]["improved_count"] > 0 or
            self.drift["summary"]["degraded_count"] > 0
        )

        return self.drift

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """Save current state as a checkpoint."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"checkpoint-{timestamp}.json"

        checkpoint_file = self.checkpoint_dir / filename
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_state, f, indent=2)

        # Also save as 'latest' for easy access
        latest_file = self.checkpoint_dir / "checkpoint-latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_state, f, indent=2)

        return checkpoint_file

    def generate_report(self, fmt: str = 'text') -> str:
        """Generate drift report."""
        if fmt == 'json':
            return json.dumps({
                "drift": self.drift,
                "current_state": {
                    "timestamp": self.current_state.get("timestamp"),
                    "summary": self.current_state.get("summary")
                },
                "previous_state": {
                    "timestamp": self.previous_state.get("timestamp") if self.previous_state else None,
                    "summary": self.previous_state.get("summary") if self.previous_state else None
                }
            }, indent=2)

        elif fmt == 'markdown':
            return self._generate_markdown_report()

        else:  # text
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate text drift report."""
        lines = [
            "="*60,
            "DRIFT DETECTION REPORT",
            "="*60,
            "",
            f"Current Checkpoint: {self.current_state.get('timestamp', 'N/A')}",
            f"Previous Checkpoint: {self.previous_state.get('timestamp', 'N/A') if self.previous_state else 'None (first run)'}",
            f"Drift Detected: {'Yes' if self.drift.get('has_drift') else 'No'}",
            "",
            "-"*40,
            "SUMMARY",
            "-"*40,
            "",
            f"Total Organizations: {self.current_state['summary']['total_organizations']}",
            f"Total Repositories:  {self.current_state['summary']['total_repositories']}",
            f"Compliant:           {self.current_state['summary']['compliant']}",
            f"Non-Compliant:       {self.current_state['summary']['non_compliant']}",
            "",
        ]

        if self.drift.get("has_drift"):
            lines.extend([
                "-"*40,
                "CHANGES DETECTED",
                "-"*40,
                "",
            ])

            if self.drift["changes"]["new_repos"]:
                lines.append(f"[NEW] {len(self.drift['changes']['new_repos'])} new repositories:")
                for repo in self.drift["changes"]["new_repos"][:10]:
                    lines.append(f"  + {repo}")
                if len(self.drift["changes"]["new_repos"]) > 10:
                    lines.append(f"  ... and {len(self.drift['changes']['new_repos']) - 10} more")
                lines.append("")

            if self.drift["changes"]["deleted_repos"]:
                lines.append(f"[DELETED] {len(self.drift['changes']['deleted_repos'])} removed repositories:")
                for repo in self.drift["changes"]["deleted_repos"][:10]:
                    lines.append(f"  - {repo}")
                lines.append("")

            if self.drift["changes"]["changed_repos"]:
                lines.append(f"[CHANGED] {len(self.drift['changes']['changed_repos'])} modified repositories:")
                for change in self.drift["changes"]["changed_repos"][:10]:
                    lines.append(f"  ~ {change['repo']}")
                lines.append("")

            if self.drift["changes"]["compliance_improved"]:
                lines.append(f"[IMPROVED] {len(self.drift['changes']['compliance_improved'])} repos now compliant:")
                for repo in self.drift["changes"]["compliance_improved"]:
                    lines.append(f"  [OK] {repo}")
                lines.append("")

            if self.drift["changes"]["compliance_degraded"]:
                lines.append(f"[DEGRADED] {len(self.drift['changes']['compliance_degraded'])} repos lost compliance:")
                for repo in self.drift["changes"]["compliance_degraded"]:
                    lines.append(f"  [WARN] {repo}")
                lines.append("")

        lines.extend([
            "",
            "="*60,
        ])

        return '\n'.join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown drift report."""
        lines = [
            "# Drift Detection Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Current Checkpoint | {self.current_state.get('timestamp', 'N/A')} |",
            f"| Previous Checkpoint | {self.previous_state.get('timestamp', 'N/A') if self.previous_state else 'None'} |",
            f"| Drift Detected | {'Yes' if self.drift.get('has_drift') else 'No'} |",
            f"| Total Repositories | {self.current_state['summary']['total_repositories']} |",
            f"| Compliant | {self.current_state['summary']['compliant']} |",
            f"| Non-Compliant | {self.current_state['summary']['non_compliant']} |",
            "",
        ]

        if self.drift.get("has_drift"):
            lines.extend([
                "## Changes",
                "",
                f"- New repositories: {self.drift['summary']['new_count']}",
                f"- Deleted repositories: {self.drift['summary']['deleted_count']}",
                f"- Changed repositories: {self.drift['summary']['changed_count']}",
                f"- Compliance improved: {self.drift['summary']['improved_count']}",
                f"- Compliance degraded: {self.drift['summary']['degraded_count']}",
                "",
            ])

            if self.drift["changes"]["new_repos"]:
                lines.extend([
                    "### New Repositories",
                    "",
                ])
                for repo in self.drift["changes"]["new_repos"]:
                    lines.append(f"- `{repo}`")
                lines.append("")

            if self.drift["changes"]["compliance_degraded"]:
                lines.extend([
                    "### Compliance Degraded (Action Required)",
                    "",
                ])
                for repo in self.drift["changes"]["compliance_degraded"]:
                    lines.append(f"- `{repo}`")
                lines.append("")

        return '\n'.join(lines)


@click.command()
@click.option('--baseline', is_flag=True, help='Create new baseline (skip comparison)')
@click.option('--report', 'report_fmt', type=click.Choice(['text', 'markdown', 'json']),
              default='text', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
@click.option('--checkpoint', type=click.Path(exists=True),
              help='Specific checkpoint file to compare against')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def main(baseline: bool, report_fmt: str, output: Optional[str],
         checkpoint: Optional[str], quiet: bool):
    """
    Detect drift between governance checkpoints.

    Creates a snapshot of current repository states and compares against
    the previous checkpoint to identify changes.

    Examples:

        python checkpoint.py                     # Normal drift detection

        python checkpoint.py --baseline          # Create new baseline

        python checkpoint.py --report markdown   # Markdown report
    """
    try:
        mgr = CheckpointManager()

        if not quiet:
            click.echo("Generating current state snapshot...")

        mgr.generate_current_state()

        if not baseline:
            if checkpoint:
                loaded = mgr.load_previous_checkpoint(Path(checkpoint))
            else:
                loaded = mgr.load_previous_checkpoint()

            if not loaded and not quiet:
                click.echo("No previous checkpoint found - this will be the baseline")

            if not quiet:
                click.echo("Detecting drift...")

            mgr.detect_drift()

        # Save checkpoint
        checkpoint_file = mgr.save_checkpoint()
        if not quiet:
            click.echo(f"Checkpoint saved: {checkpoint_file}")

        # Generate report
        report = mgr.generate_report(fmt=report_fmt)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            if not quiet:
                click.echo(f"Report written to: {output}")
        else:
            click.echo(report)

        # Also save drift report
        if mgr.drift.get("has_drift"):
            drift_file = mgr.checkpoint_dir / f"drift-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            with open(drift_file, 'w', encoding='utf-8') as f:
                json.dump(mgr.drift, f, indent=2)
            if not quiet:
                click.echo(f"Drift report saved: {drift_file}")

        # Exit with code 0 - drift is informational, not failure
        raise SystemExit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
