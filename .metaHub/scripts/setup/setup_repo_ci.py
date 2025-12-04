#!/usr/bin/env python3
"""
setup_repo_ci.py - Setup CI/CD workflows for organization repositories

Adds reusable workflow calls to organization repositories so they get
actual CI/CD execution on GitHub.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class RepoCISetup:
    """Sets up CI/CD workflows for repositories."""

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

    def setup_ci_for_repo(self, repo_path: Path) -> bool:
        """Set up CI workflow for a single repository."""
        try:
            # Read repo metadata
            meta_file = repo_path / ".meta" / "repo.yaml"
            if not meta_file.exists():
                print(f"Skipping {repo_path.name}: no .meta/repo.yaml")
                return False

            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f) or {}

            language = metadata.get("language", "unknown")
            tier = metadata.get("tier", 4)

            # Create .github/workflows directory
            workflows_dir = repo_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)

            # Create CI workflow
            ci_workflow = self._create_ci_workflow(language, tier)
            ci_file = workflows_dir / "ci.yml"
            with open(ci_file, 'w', encoding='utf-8') as f:
                yaml.dump(ci_workflow, f, default_flow_style=False, sort_keys=False)

            # Create CODEOWNERS if tier 1-2
            if tier <= 2:
                codeowners_file = repo_path / ".github" / "CODEOWNERS"
                codeowners_file.parent.mkdir(exist_ok=True)
                with open(codeowners_file, 'w', encoding='utf-8') as f:
                    f.write("# Repository maintainers\n")
                    f.write("* @alaweimm90\n")

            print(f"Set up CI for {repo_path.name} ({language}, tier {tier})")
            return True

        except Exception as e:
            print(f"Error setting up CI for {repo_path.name}: {e}")
            return False

    def _create_ci_workflow(self, language: str, tier: int) -> Dict[str, Any]:
        """Create CI workflow based on language and tier."""
        workflow = {
            'name': 'CI',
            'on': {
                'push': {'branches': ['main', 'master', 'develop']},
                'pull_request': {'branches': ['main', 'master']}
            },
            'permissions': {
                'contents': 'read',
                'pull-requests': 'write'
            },
            'jobs': {}
        }

        # Language-specific CI job
        if language == 'python':
            workflow['jobs']['python-ci'] = {
                'name': 'Python CI',
                'uses': 'alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main',
                'with': {'python-version': '3.11'}
            }
        elif language == 'typescript':
            workflow['jobs']['typescript-ci'] = {
                'name': 'TypeScript CI',
                'uses': 'alaweimm90/alaweimm90/.github/workflows/reusable-ts-ci.yml@main'
            }
        else:
            # Generic CI for other languages
            workflow['jobs']['ci'] = {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {'name': 'Checkout', 'uses': 'actions/checkout@v4'},
                    {'name': 'Run tests', 'run': 'echo "Add tests for ${language}"'}
                ]
            }

        # Policy check job (always included)
        workflow['jobs']['policy-check'] = {
            'name': 'Policy Check',
            'uses': 'alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main'
        }

        # Security scan for tier 1-2
        if tier <= 2:
            workflow['jobs']['security'] = {
                'name': 'Security Scan',
                'uses': 'alaweimm90/alaweimm90/.github/workflows/scorecard.yml@main'
            }

        return workflow

    def setup_ci_for_organization(self, org_name: str) -> Dict[str, Any]:
        """Set up CI for all repos in an organization."""
        org_path = self.org_path / org_name
        if not org_path.exists():
            return {"error": f"Organization {org_name} not found"}

        results = {
            "organization": org_name,
            "total_repos": 0,
            "successful": 0,
            "failed": 0,
            "repos": []
        }

        for repo_dir in org_path.iterdir():
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue

            results["total_repos"] += 1
            success = self.setup_ci_for_repo(repo_dir)

            results["repos"].append({
                "name": repo_dir.name,
                "success": success
            })

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

        return results

    def setup_ci_for_all(self) -> Dict[str, Any]:
        """Set up CI for all organizations."""
        if not self.org_path.exists():
            return {"error": "organizations/ directory not found"}

        overall_results = {
            "total_organizations": 0,
            "total_repos": 0,
            "total_successful": 0,
            "total_failed": 0,
            "organizations": []
        }

        for org_dir in self.org_path.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith('.'):
                continue

            overall_results["total_organizations"] += 1
            org_results = self.setup_ci_for_organization(org_dir.name)
            overall_results["organizations"].append(org_results)

            overall_results["total_repos"] += org_results["total_repos"]
            overall_results["total_successful"] += org_results["successful"]
            overall_results["total_failed"] += org_results["failed"]

        return overall_results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up CI/CD for repositories")
    parser.add_argument('--org', help='Specific organization to set up')
    parser.add_argument('--all', action='store_true', help='Set up all organizations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')

    args = parser.parse_args()

    setup = RepoCISetup()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        return

    if args.org:
        results = setup.setup_ci_for_organization(args.org)
        print(f"Organization {args.org}: {results['successful']}/{results['total_repos']} repos set up")
    elif args.all:
        results = setup.setup_ci_for_all()
        print(f"All organizations: {results['total_successful']}/{results['total_repos']} repos set up")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()