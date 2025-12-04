#!/usr/bin/env python3
"""
create_github_repos.py - Create GitHub repositories for organization monorepos

Creates GitHub repositories for each organization and pushes the monorepo content.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any

class GitHubRepoCreator:
    """Creates GitHub repositories for organization monorepos."""

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

    def create_github_repo(self, org_name: str, repo_name: str, description: str = "") -> bool:
        """Create a GitHub repository using GitHub CLI."""
        try:
            # Create the repository
            cmd = [
                "gh", "repo", "create",
                f"alaweimm90/{repo_name}",
                "--public",
                "--description", f"{org_name} organization monorepo - {description}",
                "--confirm"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_path)
            if result.returncode != 0:
                print(f"Failed to create GitHub repo {repo_name}: {result.stderr}")
                return False

            print(f"Created GitHub repository: alaweimm90/{repo_name}")
            return True

        except Exception as e:
            print(f"Error creating GitHub repo {repo_name}: {e}")
            return False

    def push_organization_monorepo(self, org_name: str) -> bool:
        """Push organization directory as a monorepo."""
        try:
            org_dir = self.org_path / org_name
            if not org_dir.exists():
                print(f"Organization directory {org_name} not found")
                return False

            # Create a temporary directory for the monorepo
            temp_repo_dir = self.base_path / f"temp-{org_name}-monorepo"
            if temp_repo_dir.exists():
                import shutil
                shutil.rmtree(temp_repo_dir)

            temp_repo_dir.mkdir()

            # Copy organization content to temp directory
            import shutil
            for item in org_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, temp_repo_dir / item.name)
                else:
                    shutil.copy2(item, temp_repo_dir / item.name)

            # Initialize git repo in temp directory
            subprocess.run(["git", "init"], cwd=temp_repo_dir, check=True)
            subprocess.run(["git", "add", "."], cwd=temp_repo_dir, check=True)
            subprocess.run(["git", "commit", "-m", f"Initial commit: {org_name} organization monorepo"], cwd=temp_repo_dir, check=True)

            # Add remote and push
            repo_name = f"{org_name}-monorepo"
            remote_url = f"https://github.com/alaweimm90/{repo_name}.git"

            # Remove existing remote if it exists
            subprocess.run(["git", "remote", "remove", "origin"], cwd=temp_repo_dir, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=temp_repo_dir, check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=temp_repo_dir, check=True)

            # Clean up
            shutil.rmtree(temp_repo_dir)

            print(f"Pushed {org_name} monorepo to GitHub")
            return True

        except Exception as e:
            print(f"Error pushing {org_name} monorepo: {e}")
            return False

    def create_and_push_all(self) -> Dict[str, Any]:
        """Create GitHub repos and push all organization monorepos."""
        results = {
            "total_organizations": 0,
            "successful_creations": 0,
            "successful_pushes": 0,
            "organizations": []
        }

        organizations = ["alaweimm90-business", "alaweimm90-science", "alaweimm90-tools", "AlaweinOS", "MeatheadPhysicist"]

        for org_name in organizations:
            results["total_organizations"] += 1

            org_result = {
                "name": org_name,
                "repo_created": False,
                "repo_pushed": False
            }

            # Create GitHub repository
            repo_name = f"{org_name}-monorepo"
            description = f"Monorepo containing all {org_name} projects"

            if self.create_github_repo(org_name, repo_name, description):
                org_result["repo_created"] = True
                results["successful_creations"] += 1

                # Push the monorepo content
                if self.push_organization_monorepo(org_name):
                    org_result["repo_pushed"] = True
                    results["successful_pushes"] += 1

            results["organizations"].append(org_result)

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Create GitHub repositories for organization monorepos")
    parser.add_argument('--org', help='Specific organization to create repo for')
    parser.add_argument('--all', action='store_true', help='Create repos for all organizations')
    parser.add_argument('--push-only', action='store_true', help='Only push existing repos')

    args = parser.parse_args()

    creator = GitHubRepoCreator()

    if args.org:
        repo_name = f"{args.org}-monorepo"
        description = f"Monorepo containing all {args.org} projects"

        if creator.create_github_repo(args.org, repo_name, description):
            if creator.push_organization_monorepo(args.org):
                print(f"Successfully created and pushed {args.org} monorepo")
            else:
                print(f"Created repo but failed to push {args.org} monorepo")
        else:
            print(f"Failed to create GitHub repo for {args.org}")

    elif args.all:
        results = creator.create_and_push_all()
        print(f"Created {results['successful_creations']}/{results['total_organizations']} GitHub repos")
        print(f"Pushed {results['successful_pushes']}/{results['total_organizations']} monorepos")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()