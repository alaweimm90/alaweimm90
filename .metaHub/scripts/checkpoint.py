"""
Checkpoint Manager - Drift Detection System.

Generates state snapshots and detects drift between checkpoints
for compliance tracking and change management.
"""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class CheckpointManager:
    """Manages state checkpoints and drift detection."""

    # Tier-based requirements
    TIER_REQUIREMENTS = {
        1: ["metadata", "readme", "license", "ci", "codeowners", "tests", "docs"],
        2: ["metadata", "readme", "license", "ci", "codeowners", "tests"],
        3: ["metadata", "readme"],
        4: ["readme"],
    }

    def __init__(self, base_path: Path = None):
        """
        Initialize checkpoint manager.

        Args:
            base_path: Base path of the repository
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.org_path = self.base_path / "organizations"
        self.checkpoint_dir = self.base_path / ".metaHub" / "checkpoints"
        self.current_state: Dict[str, Any] = {}
        self.previous_state: Dict[str, Any] = {}
        self.drift_result: Dict[str, Any] = {}

    def generate_current_state(self) -> Dict[str, Any]:
        """
        Generate current state snapshot.

        Returns:
            State snapshot dictionary
        """
        repos = {}

        if self.org_path.exists():
            for org_dir in self.org_path.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    repo_key = f"{org_dir.name}/{repo_dir.name}"
                    repos[repo_key] = self._analyze_repo(repo_dir)

        self.current_state = {
            "timestamp": datetime.now().isoformat(),
            "repositories": repos,
            "summary": {
                "total_repositories": len(repos),
                "compliant_count": sum(1 for r in repos.values() if r.get("compliant")),
                "non_compliant_count": sum(1 for r in repos.values() if not r.get("compliant")),
            },
            "checksum": self._generate_checksum(repos)
        }

        return self.current_state

    def _analyze_repo(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze a single repository."""
        meta_file = repo_path / ".meta" / "repo.yaml"
        tier = 4  # Default tier

        # Load metadata
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    metadata = yaml.safe_load(f) or {}
                    tier = metadata.get("tier", 4)
            except Exception:
                pass

        # Check compliance based on tier
        checks = {
            "metadata": (repo_path / ".meta" / "repo.yaml").exists(),
            "readme": (repo_path / "README.md").exists(),
            "license": (repo_path / "LICENSE").exists() or (repo_path / "LICENSE.md").exists(),
            "ci": (repo_path / ".github" / "workflows").exists(),
            "codeowners": (repo_path / ".github" / "CODEOWNERS").exists(),
            "tests": (repo_path / "tests").exists() or (repo_path / "test").exists(),
            "docs": (repo_path / "docs").exists(),
        }

        # Determine compliance based on tier requirements
        required = self.TIER_REQUIREMENTS.get(tier, self.TIER_REQUIREMENTS[4])
        compliant = all(checks.get(req, False) for req in required)

        return {
            "path": str(repo_path),
            "tier": tier,
            "checks": checks,
            "compliant": compliant,
            "missing": [req for req in required if not checks.get(req, False)]
        }

    def _generate_checksum(self, data: Dict) -> str:
        """Generate a checksum for state data."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save_checkpoint(self, filename: str = None) -> Path:
        """
        Save current state as checkpoint.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved checkpoint file
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            checkpoint_file = self.checkpoint_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"checkpoint-{timestamp}.json"

        checkpoint_file.write_text(json.dumps(self.current_state, indent=2))

        # Update latest symlink/copy
        latest_file = self.checkpoint_dir / "checkpoint-latest.json"
        if latest_file.exists():
            latest_file.unlink()
        shutil.copy(checkpoint_file, latest_file)

        return checkpoint_file

    def load_previous_checkpoint(self, checkpoint_path: Path = None) -> bool:
        """
        Load a previous checkpoint for comparison.

        Args:
            checkpoint_path: Optional specific checkpoint file

        Returns:
            True if loaded successfully
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint-latest.json"

        if not checkpoint_path.exists():
            self.previous_state = {}
            return False

        try:
            self.previous_state = json.loads(checkpoint_path.read_text())
            return True
        except Exception:
            self.previous_state = {}
            return False

    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect drift between previous and current state.

        Returns:
            Drift detection result
        """
        prev_repos = self.previous_state.get("repositories", {})
        curr_repos = self.current_state.get("repositories", {})

        prev_keys = set(prev_repos.keys())
        curr_keys = set(curr_repos.keys())

        new_repos = list(curr_keys - prev_keys)
        deleted_repos = list(prev_keys - curr_keys)
        common_repos = prev_keys & curr_keys

        compliance_improved = []
        compliance_degraded = []
        changed_repos = []

        for repo_key in common_repos:
            prev = prev_repos[repo_key]
            curr = curr_repos[repo_key]

            if prev.get("compliant") != curr.get("compliant"):
                if curr.get("compliant"):
                    compliance_improved.append(repo_key)
                else:
                    compliance_degraded.append(repo_key)
                changed_repos.append(repo_key)
            elif prev.get("checks") != curr.get("checks"):
                changed_repos.append(repo_key)

        has_drift = bool(new_repos or deleted_repos or changed_repos)

        self.drift_result = {
            "has_drift": has_drift,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "new_repos": new_repos,
                "deleted_repos": deleted_repos,
                "changed_repos": changed_repos,
                "compliance_improved": compliance_improved,
                "compliance_degraded": compliance_degraded,
            },
            "summary": {
                "new_count": len(new_repos),
                "deleted_count": len(deleted_repos),
                "changed_count": len(changed_repos),
                "improved_count": len(compliance_improved),
                "degraded_count": len(compliance_degraded),
            }
        }

        return self.drift_result

    def generate_report(self, fmt: str = "text") -> str:
        """
        Generate drift detection report.

        Args:
            fmt: Output format ('text', 'markdown', or 'json')

        Returns:
            Formatted report string
        """
        if fmt == "json":
            return json.dumps({
                "drift": self.drift_result,
                "current_state": self.current_state,
            }, indent=2)

        if fmt == "markdown":
            return self._markdown_report()

        return self._text_report()

    def _text_report(self) -> str:
        """Generate text format report."""
        lines = [
            "=" * 60,
            "DRIFT DETECTION REPORT",
            "=" * 60,
            "",
            f"Timestamp: {self.current_state.get('timestamp', 'N/A')}",
            f"Total Repositories: {self.current_state.get('summary', {}).get('total_repositories', 0)}",
            "",
            "Changes:",
            f"  New Repos: {self.drift_result.get('summary', {}).get('new_count', 0)}",
            f"  Deleted Repos: {self.drift_result.get('summary', {}).get('deleted_count', 0)}",
            f"  Changed Repos: {self.drift_result.get('summary', {}).get('changed_count', 0)}",
            f"  Compliance Improved: {self.drift_result.get('summary', {}).get('improved_count', 0)}",
            f"  Compliance Degraded: {self.drift_result.get('summary', {}).get('degraded_count', 0)}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _markdown_report(self) -> str:
        """Generate markdown format report."""
        summary = self.current_state.get("summary", {})
        drift_summary = self.drift_result.get("summary", {})

        lines = [
            "# Drift Detection Report",
            "",
            f"**Generated:** {self.current_state.get('timestamp', 'N/A')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Repositories | {summary.get('total_repositories', 0)} |",
            f"| Compliant | {summary.get('compliant_count', 0)} |",
            f"| Non-Compliant | {summary.get('non_compliant_count', 0)} |",
            "",
            "## Changes",
            "",
            "| Change Type | Count |",
            "|-------------|-------|",
            f"| New Repos | {drift_summary.get('new_count', 0)} |",
            f"| Deleted Repos | {drift_summary.get('deleted_count', 0)} |",
            f"| Changed Repos | {drift_summary.get('changed_count', 0)} |",
            f"| Compliance Improved | {drift_summary.get('improved_count', 0)} |",
            f"| Compliance Degraded | {drift_summary.get('degraded_count', 0)} |",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint and drift detection")
    parser.add_argument("action", choices=["snapshot", "drift", "report"])
    parser.add_argument("--format", "-f", choices=["text", "markdown", "json"], default="text")
    parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    mgr = CheckpointManager()

    if args.action == "snapshot":
        mgr.generate_current_state()
        path = mgr.save_checkpoint()
        print(f"Checkpoint saved: {path}")

    elif args.action == "drift":
        mgr.load_previous_checkpoint()
        mgr.generate_current_state()
        drift = mgr.detect_drift()
        print(json.dumps(drift, indent=2))

    elif args.action == "report":
        mgr.load_previous_checkpoint()
        mgr.generate_current_state()
        mgr.detect_drift()
        report = mgr.generate_report(fmt=args.format)
        if args.output:
            Path(args.output).write_text(report)
        else:
            print(report)
