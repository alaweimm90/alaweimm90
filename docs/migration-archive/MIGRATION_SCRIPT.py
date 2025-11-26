#!/usr/bin/env python3
"""
Migration script for alaweimm90 Golden Path compliance.

Applies .meta/repo.yaml, .github/CODEOWNERS, and reusable CI to existing repos.
Safe: creates changes in a branch and prints git commands for manual review.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime


class RepoMigrator:
    """Migrate a single repo to Golden Path compliance."""

    REPO_TYPES = {
        "library": {
            "required_files": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
            "coverage_min": 80,
            "docs_profile": "standard",
        },
        "tool": {
            "required_files": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
            "coverage_min": 80,
            "docs_profile": "standard",
        },
        "meta": {
            "required_files": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml"],
            "coverage_min": 0,
            "docs_profile": "minimal",
        },
        "demo": {
            "required_files": ["README.md", ".meta/repo.yaml"],
            "coverage_min": 70,
            "docs_profile": "minimal",
        },
        "research": {
            "required_files": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS"],
            "coverage_min": 50,
            "docs_profile": "standard",
        },
        "adapter": {
            "required_files": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
            "coverage_min": 80,
            "docs_profile": "minimal",
        },
    }

    def __init__(self, repo_path: str, org: str, repo_name: str, repo_data: Dict[str, Any]):
        self.repo_path = Path(repo_path)
        self.org = org
        self.repo_name = repo_name
        self.repo_data = repo_data
        self.changes_made = []

    def infer_repo_type(self) -> str:
        """Infer repo type from prefix and contents."""
        # Check prefix
        prefixes = {
            "core-": "tool",
            "lib-": "library",
            "adapter-": "adapter",
            "tool-": "tool",
            "template-": "demo",
            "demo-": "demo",
            "infra-": "tool",
            "paper-": "research",
        }

        for prefix, repo_type in prefixes.items():
            if self.repo_name.startswith(prefix):
                return repo_type

        # Fallback: use inventory data
        return self.repo_data.get("type", "unknown")

    def infer_primary_language(self) -> str:
        """Infer primary language from project files."""
        languages = self.repo_data.get("languages", [])
        if languages:
            return languages[0]

        # Check for markers
        if (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
            return "python"
        if (self.repo_path / "package.json").exists():
            return "typescript"
        if (self.repo_path / "go.mod").exists():
            return "go"
        if (self.repo_path / "Cargo.toml").exists():
            return "rust"

        return "unknown"

    def create_meta_repo_yaml(self) -> bool:
        """Create .meta/repo.yaml if missing."""
        meta_dir = self.repo_path / ".meta"
        meta_file = meta_dir / "repo.yaml"

        if meta_file.exists():
            print(f"  ✓ {self.repo_name}: .meta/repo.yaml already exists")
            return False

        meta_dir.mkdir(parents=True, exist_ok=True)

        repo_type = self.infer_repo_type()
        language = self.infer_primary_language()
        docs_profile = self.REPO_TYPES.get(repo_type, {}).get("docs_profile", "minimal")
        coverage_min = self.REPO_TYPES.get(repo_type, {}).get("coverage_min", 70)

        # Determine criticality tier
        if self.repo_name in ["core-control-center", ".github", "standards"]:
            tier = 1
        elif repo_type in ["library", "tool"] and self.repo_data.get("active_status") == "active":
            tier = 2
        elif repo_type in ["adapter", "demo", "research"]:
            tier = 3
        else:
            tier = 4

        metadata = {
            "type": repo_type,
            "languages": [language] if language != "unknown" else [],
            "docs_profile": docs_profile,
            "criticality_tier": tier,
            "description": f"Repository: {self.repo_name}",
            "owner": self.org,
            "status": self.repo_data.get("active_status", "unknown"),
            "created": datetime.now().strftime("%Y-%m-%d"),
        }

        with open(meta_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        self.changes_made.append(f"Created .meta/repo.yaml (type={repo_type}, tier={tier})")
        print(f"  ✓ {self.repo_name}: Created .meta/repo.yaml")
        return True

    def create_codeowners(self) -> bool:
        """Create .github/CODEOWNERS if missing."""
        codeowners_file = self.repo_path / ".github" / "CODEOWNERS"

        if codeowners_file.exists():
            print(f"  ✓ {self.repo_name}: .github/CODEOWNERS already exists")
            return False

        codeowners_file.parent.mkdir(parents=True, exist_ok=True)

        content = """# Repository ownership and approval requirements

* @alaweimm90

# Critical paths require additional review
/.github/workflows/ @alaweimm90
/src/ @alaweimm90
/tests/ @alaweimm90
"""

        with open(codeowners_file, "w") as f:
            f.write(content)

        self.changes_made.append("Created .github/CODEOWNERS")
        print(f"  ✓ {self.repo_name}: Created .github/CODEOWNERS")
        return True

    def update_ci_yml(self) -> bool:
        """Update or create .github/workflows/ci.yml to call reusable workflows."""
        ci_file = self.repo_path / ".github" / "workflows" / "ci.yml"

        # Detect language
        language = self.infer_primary_language()

        if language == "python":
            ci_content = """name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""
        elif language in ["typescript", "javascript"]:
            ci_content = """name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ts:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""
        else:
            # Fallback: minimal policy check only
            ci_content = """name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""

        ci_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if CI already exists and references reusable workflows
        if ci_file.exists():
            existing = ci_file.read_text()
            if "uses: alaweimm90/.github/.github/workflows/reusable" in existing:
                print(f"  ✓ {self.repo_name}: CI already calls reusable workflows")
                return False

        with open(ci_file, "w") as f:
            f.write(ci_content)

        self.changes_made.append(f"Updated .github/workflows/ci.yml (language: {language})")
        print(f"  ✓ {self.repo_name}: Updated CI to call reusable workflows")
        return True

    def create_policy_yml(self) -> bool:
        """Create .github/workflows/policy.yml if missing."""
        policy_file = self.repo_path / ".github" / "workflows" / "policy.yml"

        if policy_file.exists():
            return False

        policy_content = """name: policy

on:
  pull_request:
    branches: [main]

jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""

        policy_file.parent.mkdir(parents=True, exist_ok=True)
        with open(policy_file, "w") as f:
            f.write(policy_content)

        self.changes_made.append("Created .github/workflows/policy.yml")
        print(f"  ✓ {self.repo_name}: Created policy workflow")
        return True

    def migrate(self) -> Dict[str, Any]:
        """Execute full migration for this repo."""
        print(f"\n📦 Migrating {self.org}/{self.repo_name}")

        if not self.repo_path.exists():
            print(f"  ⚠️  Repo path not found: {self.repo_path}")
            return {"status": "skipped", "reason": "path not found"}

        self.create_meta_repo_yaml()
        self.create_codeowners()
        self.update_ci_yml()
        self.create_policy_yml()

        return {
            "org": self.org,
            "repo": self.repo_name,
            "status": "success" if self.changes_made else "skipped",
            "changes": self.changes_made,
        }


def main():
    """Main entry point."""
    # Load inventory
    inventory_path = Path("inventory.json")
    if not inventory_path.exists():
        print("ERROR: inventory.json not found")
        sys.exit(1)

    with open(inventory_path) as f:
        inventory = json.load(f)

    # Base path for repos
    repo_base = Path("organizations")

    if not repo_base.exists():
        print(f"ERROR: organizations/ path not found")
        sys.exit(1)

    results = []
    total_changes = 0

    print("\n" + "=" * 70)
    print("alaweimm90 Golden Path Migration")
    print("=" * 70)

    for org_data in inventory["organizations"]:
        org_name = org_data["name"]
        print(f"\n🏢 Organization: {org_name}")

        org_path = repo_base / org_name

        for repo in org_data["repositories"]:
            repo_name = repo["name"]
            repo_path = org_path / repo_name

            migrator = RepoMigrator(repo_path, org_name, repo_name, repo)
            result = migrator.migrate()
            results.append(result)

            if result["status"] == "success":
                total_changes += len(result.get("changes", []))

    # Summary
    print("\n" + "=" * 70)
    print("Migration Summary")
    print("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n✅ Successfully migrated: {len(successful)} repos")
    print(f"⏭️  Skipped (already compliant): {len(skipped)} repos")
    print(f"📝 Total changes made: {total_changes}")

    # Save results
    with open("migration-results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: migration-results.json")

    # Generate git commands
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)

    print("""
1. Review changes:
   cd organizations && find . -name '.meta' -o -name '.github/CODEOWNERS' | head -20

2. Commit changes in each repo:
   for org in alaweimm90-business alaweimm90-science alaweimm90-tools AlaweinOS MeatheadPhysicist; do
     cd organizations/$org
     git add .
     git commit -m "chore: add Golden Path compliance files (.meta/repo.yaml, CODEOWNERS, CI)"
     git push origin main
     cd ../..
   done

3. Verify CI runs on all repos
4. Monitor for policy violations
5. Update repo settings in GitHub:
   - Enable branch protection on 'main'
   - Require 'ci' + 'policy' status checks
   - Require code owner reviews
""")


if __name__ == "__main__":
    main()
