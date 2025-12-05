#!/usr/bin/env python3
"""
sync_governance.py - Sync governance rules to all organization repositories

Automatically updates governance files, CI/CD workflows, and policies across
all repositories in the portfolio.
"""

import yaml
import shutil
from pathlib import Path
from typing import Dict, Any

class GovernanceSyncer:
    """Syncs governance rules across all repositories."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or self._find_base_path()
        self.org_path = self.base_path / "organizations"

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent
        return Path.cwd()

    def sync_repo_governance(self, repo_path: Path) -> Dict[str, Any]:
        """Sync governance files to a single repository."""
        changes = {
            "repo": repo_path.name,
            "files_synced": [],
            "errors": []
        }

        try:
            # Read repo metadata
            meta_file = repo_path / ".meta" / "repo.yaml"
            if not meta_file.exists():
                changes["errors"].append("No .meta/repo.yaml found")
                return changes

            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f) or {}

            # Sync CI/CD workflows
            self._sync_ci_workflows(repo_path, metadata, changes)

            # Sync governance files
            self._sync_governance_files(repo_path, changes)

            # Sync pre-commit config
            self._sync_precommit_config(repo_path, changes)

            # Sync editor configs
            self._sync_editor_configs(repo_path, changes)

        except Exception as e:
            changes["errors"].append(f"Sync error: {e}")

        return changes

    def _sync_ci_workflows(self, repo_path: Path, metadata: Dict[str, Any], changes: Dict[str, Any]):
        """Sync CI/CD workflows based on repo metadata."""
        workflows_dir = repo_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        language = metadata.get("language", "unknown")
        tier = metadata.get("tier", 4)

        # Copy appropriate workflow files
        source_workflows = self.base_path / ".github" / "workflows"

        # Always sync policy check workflow
        self._copy_workflow_file(source_workflows / "reusable-policy.yml", workflows_dir, changes)

        # Language-specific workflows
        if language == 'python':
            self._copy_workflow_file(source_workflows / "reusable-python-ci.yml", workflows_dir, changes)
        elif language == 'typescript':
            self._copy_workflow_file(source_workflows / "reusable-ts-ci.yml", workflows_dir, changes)

        # Security scans for high-tier repos
        if tier <= 2:
            self._copy_workflow_file(source_workflows / "scorecard.yml", workflows_dir, changes)

    def _sync_governance_files(self, repo_path: Path, changes: Dict[str, Any]):
        """Sync governance documentation and configs."""
        # Sync SECURITY.md
        security_src = self.base_path / "SECURITY.md"
        security_dst = repo_path / "SECURITY.md"
        if security_src.exists():
            shutil.copy2(security_src, security_dst)
            changes["files_synced"].append("SECURITY.md")

        # Sync CONTRIBUTING.md if it exists
        contrib_src = self.base_path / "CONTRIBUTING.md"
        contrib_dst = repo_path / "CONTRIBUTING.md"
        if contrib_src.exists():
            shutil.copy2(contrib_src, contrib_dst)
            changes["files_synced"].append("CONTRIBUTING.md")

        # Sync CODEOWNERS for tier 1-2 repos
        codeowners_src = self.base_path / ".github" / "CODEOWNERS"
        codeowners_dst = repo_path / ".github" / "CODEOWNERS"
        if codeowners_src.exists():
            codeowners_dst.parent.mkdir(exist_ok=True)
            shutil.copy2(codeowners_src, codeowners_dst)
            changes["files_synced"].append(".github/CODEOWNERS")

    def _sync_precommit_config(self, repo_path: Path, changes: Dict[str, Any]):
        """Sync pre-commit configuration."""
        precommit_src = self.base_path / ".pre-commit-config.yaml"
        precommit_dst = repo_path / ".pre-commit-config.yaml"

        if precommit_src.exists():
            shutil.copy2(precommit_src, precommit_dst)
            changes["files_synced"].append(".pre-commit-config.yaml")

    def _sync_editor_configs(self, repo_path: Path, changes: Dict[str, Any]):
        """Sync editor configuration files."""
        config_files = [
            ".editorconfig",
            ".yamllint.yml",
            ".ruff.toml",
            "pyrightconfig.json"
        ]

        for config_file in config_files:
            src = self.base_path / config_file
            dst = repo_path / config_file

            if src.exists():
                shutil.copy2(src, dst)
                changes["files_synced"].append(config_file)

    def _copy_workflow_file(self, src: Path, dst_dir: Path, changes: Dict[str, Any]):
        """Copy a workflow file if it exists."""
        if src.exists():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            changes["files_synced"].append(f".github/workflows/{src.name}")

    def sync_organization(self, org_name: str) -> Dict[str, Any]:
        """Sync governance to all repos in an organization."""
        org_path = self.org_path / org_name
        if not org_path.exists():
            return {"error": f"Organization {org_name} not found"}

        results = {
            "organization": org_name,
            "total_repos": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "repos": []
        }

        for repo_dir in org_path.iterdir():
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue

            results["total_repos"] += 1
            sync_result = self.sync_repo_governance(repo_dir)

            results["repos"].append(sync_result)

            if sync_result["errors"]:
                results["failed_syncs"] += 1
            else:
                results["successful_syncs"] += 1

        return results

    def sync_all_organizations(self) -> Dict[str, Any]:
        """Sync governance to all organizations."""
        if not self.org_path.exists():
            return {"error": "organizations/ directory not found"}

        overall_results = {
            "total_organizations": 0,
            "total_repos": 0,
            "total_successful": 0,
            "total_failed": 0,
            "organizations": []
        }

        organizations = ["alawein-business", "alawein-science", "alawein-tools", "AlaweinOS", "MeatheadPhysicist"]

        for org_name in organizations:
            overall_results["total_organizations"] += 1
            org_results = self.sync_organization(org_name)
            overall_results["organizations"].append(org_results)

            overall_results["total_repos"] += org_results["total_repos"]
            overall_results["total_successful"] += org_results["successful_syncs"]
            overall_results["total_failed"] += org_results["failed_syncs"]

        return overall_results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sync governance rules to repositories")
    parser.add_argument('--org', help='Specific organization to sync')
    parser.add_argument('--all', action='store_true', help='Sync all organizations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')

    args = parser.parse_args()

    syncer = GovernanceSyncer()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        return

    if args.org:
        results = syncer.sync_organization(args.org)
        successful = results["successful_syncs"]
        total = results["total_repos"]
        print(f"Organization {args.org}: {successful}/{total} repos synced")
    elif args.all:
        results = syncer.sync_all_organizations()
        successful = results["total_successful"]
        total = results["total_repos"]
        print(f"All organizations: {successful}/{total} repos synced")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()