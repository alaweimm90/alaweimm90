#!/usr/bin/env python3
"""
push_monorepos.py - Sync organization content to existing GitHub monorepos

Clones existing monorepos, applies local changes, and pushes updates.
Preserves git history by updating rather than replacing.
"""

import os
import subprocess
import shutil
import stat
import time
from pathlib import Path
from typing import Dict, Any, List


def robust_rmtree(path: Path, retries: int = 3) -> bool:
    """Remove directory tree with Windows-compatible retry logic."""
    def on_error(func, path, exc_info):
        # Make file writable and retry
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for attempt in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path, onerror=on_error)
            return True
        except (OSError, PermissionError):
            if attempt < retries - 1:
                time.sleep(0.5)  # Wait before retry
            continue
    return False


class MonorepoPusher:
    """Syncs organization content to GitHub monorepos."""

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

    def _get_remote_url(self, org_name: str) -> str:
        """Get the correct GitHub remote URL for an organization."""
        # Actual repos are at {org}/monorepo, not alaweimm90/{org}-monorepo
        return f"https://github.com/{org_name}/monorepo.git"

    def _run_git(self, args: List[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command with proper error handling."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
        return result

    def push_monorepo(self, org_name: str, commit_message: str = None) -> Dict[str, Any]:
        """Sync organization directory to its GitHub monorepo."""
        result = {
            "org": org_name,
            "success": False,
            "files_changed": 0,
            "message": ""
        }

        temp_repo_dir = None
        try:
            org_dir = self.org_path / org_name
            if not org_dir.exists():
                result["message"] = f"Organization directory {org_name} not found"
                return result

            # Create temp directory for clone
            temp_repo_dir = self.base_path / f"temp-{org_name}-sync"
            if temp_repo_dir.exists():
                robust_rmtree(temp_repo_dir)

            # Clone existing monorepo
            remote_url = self._get_remote_url(org_name)
            print(f"Cloning {remote_url}...")
            self._run_git(["clone", "--depth=1", remote_url, str(temp_repo_dir)], cwd=self.base_path)

            # Copy local organization content over cloned repo (excluding .git)
            print(f"Syncing local changes for {org_name}...")
            for item in org_dir.iterdir():
                if item.name == ".git":
                    continue
                dest = temp_repo_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        robust_rmtree(dest)
                    else:
                        dest.unlink()
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            # Check for changes
            self._run_git(["add", "-A"], cwd=temp_repo_dir)
            status = self._run_git(["status", "--porcelain"], cwd=temp_repo_dir, check=False)

            if not status.stdout.strip():
                result["success"] = True
                result["message"] = "No changes to push"
                return result

            # Count changed files
            result["files_changed"] = len([line for line in status.stdout.strip().split('\n') if line])

            # Commit changes
            msg = commit_message or f"chore(governance): sync {org_name} with governance fixes\n\n- Docker security: EXPOSE 8080, USER directive, pinned versions\n- Applied by push_monorepos.py"
            self._run_git(["commit", "-m", msg], cwd=temp_repo_dir)

            # Push to origin
            print(f"Pushing {result['files_changed']} changed files to {org_name}/monorepo...")
            self._run_git(["push", "origin", "main"], cwd=temp_repo_dir)

            result["success"] = True
            result["message"] = f"Pushed {result['files_changed']} files"
            print(f"[OK] {org_name}: {result['message']}")
            return result

        except subprocess.CalledProcessError as e:
            result["message"] = f"Git error: {e.stderr or e.stdout or str(e)}"
            print(f"[FAIL] {org_name}: {result['message']}")
            return result
        except Exception as e:
            result["message"] = f"Error: {str(e)}"
            print(f"[FAIL] {org_name}: {result['message']}")
            return result
        finally:
            # Clean up temp directory
            if temp_repo_dir and temp_repo_dir.exists():
                if not robust_rmtree(temp_repo_dir):
                    print(f"Warning: Could not clean up {temp_repo_dir}")

    def push_all_monorepos(self, commit_message: str = None) -> Dict[str, Any]:
        """Sync all organization monorepos to GitHub."""
        results = {
            "total_organizations": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "no_changes": 0,
            "total_files_changed": 0,
            "organizations": []
        }

        organizations = ["alaweimm90-business", "alaweimm90-science", "alaweimm90-tools", "AlaweinOS", "MeatheadPhysicist"]

        print(f"\n{'='*60}")
        print("Syncing Governance Fixes to Organization Monorepos")
        print(f"{'='*60}\n")

        for org_name in organizations:
            results["total_organizations"] += 1
            org_result = self.push_monorepo(org_name, commit_message)
            results["organizations"].append(org_result)

            if org_result["success"]:
                if org_result["files_changed"] > 0:
                    results["successful_syncs"] += 1
                    results["total_files_changed"] += org_result["files_changed"]
                else:
                    results["no_changes"] += 1
            else:
                results["failed_syncs"] += 1

        print(f"\n{'='*60}")
        print(f"Summary: {results['successful_syncs']} synced, {results['no_changes']} unchanged, {results['failed_syncs']} failed")
        print(f"Total files changed: {results['total_files_changed']}")
        print(f"{'='*60}\n")

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sync organization content to GitHub monorepos")
    parser.add_argument('--org', help='Specific organization to sync')
    parser.add_argument('--all', action='store_true', help='Sync all organizations')
    parser.add_argument('--message', '-m', help='Custom commit message')

    args = parser.parse_args()

    pusher = MonorepoPusher()

    if args.org:
        result = pusher.push_monorepo(args.org, args.message)
        if result["success"]:
            print(f"\nSuccess: {result['message']}")
        else:
            print(f"\nFailed: {result['message']}")

    elif args.all:
        results = pusher.push_all_monorepos(args.message)
        exit(0 if results['failed_syncs'] == 0 else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()