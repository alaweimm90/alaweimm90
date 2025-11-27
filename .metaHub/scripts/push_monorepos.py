#!/usr/bin/env python3
"""
push_monorepos.py - Push organization monorepos to existing GitHub repositories
"""

import os
import subprocess
import shutil
from pathlib import Path

class MonorepoPusher:
    """Pushes organization monorepos to GitHub."""

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

    def push_monorepo(self, org_name: str) -> bool:
        """Push organization directory as a monorepo."""
        try:
            org_dir = self.org_path / org_name
            if not org_dir.exists():
                print(f"Organization directory {org_name} not found")
                return False

            # Create a temporary directory for the monorepo
            temp_repo_dir = self.base_path / f"temp-{org_name}-monorepo"
            if temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)

            temp_repo_dir.mkdir()

            # Copy organization content to temp directory
            for item in org_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, temp_repo_dir / item.name)
                else:
                    shutil.copy2(item, temp_repo_dir / item.name)

            # Initialize git repo in temp directory
            subprocess.run(["git", "init"], cwd=temp_repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=temp_repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Initial commit: {org_name} organization monorepo"], cwd=temp_repo_dir, check=True, capture_output=True)

            # Add remote and push
            repo_name = f"{org_name}-monorepo"
            remote_url = f"https://github.com/alaweimm90/{repo_name}.git"

            # Remove existing remote if it exists
            subprocess.run(["git", "remote", "remove", "origin"], cwd=temp_repo_dir, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=temp_repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=temp_repo_dir, check=True, capture_output=True)

            # Clean up (don't fail if cleanup fails)
            try:
                shutil.rmtree(temp_repo_dir)
            except (OSError, PermissionError):
                print(f"Warning: Could not clean up {temp_repo_dir}, manual cleanup may be needed")
                # Don't fail the operation for cleanup issues

            print(f"Pushed {org_name} monorepo to GitHub")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Git command failed for {org_name}: {e}")
            # Clean up on failure
            if temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)
            return False
        except Exception as e:
            print(f"Error pushing {org_name} monorepo: {e}")
            # Clean up on failure
            if 'temp_repo_dir' in locals() and temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)
            return False

    def push_all_monorepos(self) -> Dict[str, Any]:
        """Push all organization monorepos."""
        results = {
            "total_organizations": 0,
            "successful_pushes": 0,
            "failed_pushes": 0,
            "organizations": []
        }

        organizations = ["alaweimm90-business", "alaweimm90-science", "alaweimm90-tools", "AlaweinOS", "MeatheadPhysicist"]

        for org_name in organizations:
            results["total_organizations"] += 1

            org_result = {
                "name": org_name,
                "pushed": False
            }

            if self.push_monorepo(org_name):
                org_result["pushed"] = True
                results["successful_pushes"] += 1
            else:
                results["failed_pushes"] += 1

            results["organizations"].append(org_result)

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Push organization monorepos to GitHub")
    parser.add_argument('--org', help='Specific organization to push')
    parser.add_argument('--all', action='store_true', help='Push all organizations')

    args = parser.parse_args()

    pusher = MonorepoPusher()

    if args.org:
        if pusher.push_monorepo(args.org):
            print(f"Successfully pushed {args.org} monorepo")
        else:
            print(f"Failed to push {args.org} monorepo")

    elif args.all:
        results = pusher.push_all_monorepos()
        print(f"Pushed {results['successful_pushes']}/{results['total_organizations']} monorepos")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()