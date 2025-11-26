import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime

# Configure structured logging
def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def get_central_repo_path() -> Path:
    """Get central repo path from environment or auto-detect.

    Priority:
    1. GOLDEN_PATH_ROOT environment variable
    2. Auto-detect by searching for .metaHub directory
    3. Current working directory if it contains .metaHub
    """
    # Check environment variable first
    if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
        path = Path(env_path)
        if path.exists() and (path / ".metaHub").exists():
            return path
        logger.warning(f"GOLDEN_PATH_ROOT set to {env_path} but .metaHub not found there")

    # Auto-detect: search up from current directory
    current = Path.cwd()
    while current != current.parent:
        if (current / ".metaHub").exists():
            return current
        current = current.parent

    # Fallback: check if cwd has .metaHub
    if (Path.cwd() / ".metaHub").exists():
        return Path.cwd()

    raise RuntimeError(
        "Could not find central repo (no .metaHub directory found). "
        "Set GOLDEN_PATH_ROOT environment variable or run from repo root."
    )

class RepoEnforcer:
    """Enforce Golden Path compliance on a single repo."""

    REPO_TYPES = {
        # Define REPO_TYPES as in MIGRATION_SCRIPT.py
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

    def __init__(self, repo_path: Path, org: str, repo_name: str, repo_data: Dict[str, Any],
                 central_repo_path: Path, dry_run: bool = False):
        self.repo_path = repo_path
        self.org = org
        self.repo_name = repo_name
        self.repo_data = repo_data
        self.central_repo_path = central_repo_path
        self.dry_run = dry_run
        self.changes_made: List[str] = []

    def _write_file(self, path: Path, content: str, encoding: str = 'utf-8') -> bool:
        """Write content to file, respecting dry-run mode."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would write to {path}")
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return True

    def _log_change(self, message: str):
        self.changes_made.append(message)
        print(f"  [OK] {self.repo_name}: {message}")

    def infer_repo_type(self) -> str:
        # Re-use logic from MIGRATION_SCRIPT.py
        """Infer repo type from prefix and contents."""
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

        return self.repo_data.get("type", "unknown")

    def infer_primary_language(self) -> str:
        # Re-use logic from MIGRATION_SCRIPT.py
        """Infer primary language from project files."""
        languages = self.repo_data.get("languages", [])
        if languages:
            return languages[0]

        if (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
            return "python"
        if (self.repo_path / "package.json").exists():
            return "typescript"
        if (self.repo_path / "go.mod").exists():
            return "go"
        if (self.repo_path / "Cargo.toml").exists():
            return "rust"

        return "unknown"

    def create_or_update_meta_repo_yaml(self) -> bool:
        """Create or update .meta/repo.yaml."""
        meta_dir = self.repo_path / ".meta"
        meta_file = meta_dir / "repo.yaml"

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

        # Schema-compliant metadata (matches .metaHub/schemas/repo-schema.json)
        metadata = {
            "type": repo_type,
            "language": language if language != "unknown" else "mixed",  # singular per schema
            "tier": tier,  # integer 1-3 per schema (not criticality_tier)
            "coverage": {"target": coverage_min},
            "docs": {"profile": docs_profile},
            "owner": self.org,
            "description": self.repo_data.get("description", f"Repository: {self.repo_name}"),
            "status": self.repo_data.get("active_status", "unknown"),
            "created": self.repo_data.get("created", datetime.now().strftime("%Y-%m-%d")),
        }

        existing_metadata = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                existing_metadata = yaml.safe_load(f) or {}

        if existing_metadata != metadata:
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            self._log_change("Updated .meta/repo.yaml")
            return True
        else:
            print(f"  [OK] {self.repo_name}: .meta/repo.yaml is up-to-date")
            return False

    def create_or_update_codeowners(self) -> bool:
        """Create or update .github/CODEOWNERS."""
        codeowners_file = self.repo_path / ".github" / "CODEOWNERS"
        codeowners_file.parent.mkdir(parents=True, exist_ok=True)

        # Use a standardized CODEOWNERS content
        content = f"""# Repository ownership and approval requirements

* @{self.org}

# Critical paths require additional review
/.github/workflows/ @{self.org}
/src/ @{self.org}
/tests/ @{self.org}
"""
        existing_content = ""
        if codeowners_file.exists():
            existing_content = codeowners_file.read_text()

        if existing_content.strip() != content.strip():
            with open(codeowners_file, "w") as f:
                f.write(content)
            self._log_change("Updated .github/CODEOWNERS")
            return True
        else:
            print(f"  [OK] {self.repo_name}: .github/CODEOWNERS is up-to-date")
            return False

    def enforce_ci_yml(self) -> bool:
        """Enforce .github/workflows/ci.yml to call reusable workflows."""
        ci_file = self.repo_path / ".github" / "workflows" / "ci.yml"
        ci_file.parent.mkdir(parents=True, exist_ok=True)

        language = self.infer_primary_language()

        # Get the name of the central GitHub repo (e.g., 'GitHub' from C:\Users\mesha\Desktop\GitHub)
        central_repo_name = self.central_repo_path.name


        if language == "python":
            ci_content = f"""name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python:
    uses: {central_repo_name}/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: {central_repo_name}/.github/workflows/reusable-policy.yml@main
"""
        elif language == "typescript":
             ci_content = f"""name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ts:
    uses: {central_repo_name}/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'

  policy:
    uses: {central_repo_name}/.github/workflows/reusable-policy.yml@main
"""
        else: # Fallback for unknown languages or other repo types
            ci_content = f"""name: ci

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  policy:
    uses: {central_repo_name}/.github/workflows/reusable-policy.yml@main
"""

        existing_content = ""
        if ci_file.exists():
            existing_content = ci_file.read_text()

        if existing_content.strip() != ci_content.strip():
            with open(ci_file, "w") as f:
                f.write(ci_content)
            self._log_change(f"Updated .github/workflows/ci.yml (language: {language})")
            return True
        else:
            print(f"  [OK] {self.repo_name}: CI is up-to-date")
            return False

    def enforce_policy_yml(self) -> bool:
        """Enforce .github/workflows/policy.yml to call reusable workflows."""
        policy_file = self.repo_path / ".github" / "workflows" / "policy.yml"
        policy_file.parent.mkdir(parents=True, exist_ok=True)

        central_repo_name = self.central_repo_path.name

        policy_content = f"""name: policy

on:
  pull_request:
    branches: [main]

jobs:
  policy:
    uses: {central_repo_name}/.github/workflows/reusable-policy.yml@main
"""
        existing_content = ""
        if policy_file.exists():
            existing_content = policy_file.read_text()

        if existing_content.strip() != policy_content.strip():
            with open(policy_file, "w") as f:
                f.write(policy_content)
            self._log_change("Updated .github/workflows/policy.yml")
            return True
        else:
            print(f"  [OK] {self.repo_name}: Policy workflow is up-to-date")
            return False

    def enforce_pre_commit_config(self) -> bool:
        """Enforce a standard .pre-commit-config.yaml."""
        pre_commit_file = self.repo_path / ".pre-commit-config.yaml"
        # Assuming a standard pre-commit config is in the central .metaHub templates
        template_path = self.central_repo_path / ".metaHub" / "templates" / "pre-commit"

        language = self.infer_primary_language()
        template_content = ""
        template_name = ""

        if language == "python":
            template_name = "python.yaml"
        # Assuming standard typescript pre-commit config is also available
        elif language in ["typescript", "javascript"]:
            template_name = "typescript.yaml"
        else:
            template_name = "generic.yaml"

        specific_template = template_path / template_name
        if specific_template.exists():
            with open(specific_template, "r") as f:
                template_content = f.read()
        else:
            print(f"  [WARN] {self.repo_name}: No specific pre-commit template for {language} at {specific_template}. Skipping.")
            return False

        existing_content = ""
        if pre_commit_file.exists():
            existing_content = pre_commit_file.read_text()

        if existing_content.strip() != template_content.strip():
            with open(pre_commit_file, "w") as f:
                f.write(template_content)
            self._log_change(f"Updated .pre-commit-config.yaml for {language}")
            return True
        else:
            print(f"  [OK] {self.repo_name}: .pre-commit-config.yaml is up-to-date")
            return False


    def enforce_dockerfile(self) -> bool:
        """Create a Dockerfile if the project is Python and one doesn't exist."""
        if self.infer_primary_language() == "python":
            dockerfile_path = self.repo_path / "Dockerfile"
            if not dockerfile_path.exists():
                template_dockerfile = self.central_repo_path / ".metaHub" / "templates" / "docker" / "python.Dockerfile"
                if template_dockerfile.exists():
                    with open(template_dockerfile, 'r') as src, open(dockerfile_path, 'w') as dst:
                        dst.write(src.read())
                    self._log_change("Created Dockerfile from template")
                    return True
                else:
                    print(f"  [WARN] {self.repo_name}: Dockerfile template not found at {template_dockerfile}. Skipping Dockerfile creation.")
            else:
                print(f"  [OK] {self.repo_name}: Dockerfile already exists")
        return False

    def enforce_readme(self) -> bool:
        """Enforce a standardized README.md using a template."""
        readme_template_path = self.central_repo_path / ".metaHub" / "templates" / "README.md.template"
        if not readme_template_path.exists():
            print(f"  [WARN] {self.repo_name}: README template not found at {readme_template_path}. Skipping README enforcement.")
            return False

        with open(readme_template_path, 'r', encoding='utf-8') as f: # Added encoding='utf-8'
            template_content = f.read()

        # Fill placeholders
        repo_name = self.repo_name
        org_name = self.org
        repo_description = self.repo_data.get("description", "A brief description of the repository.")
        license_type = self.repo_data.get("license", "MIT") # Default to MIT if not specified

        generated_readme_content = template_content.replace("{{ REPO_NAME }}", repo_name)
        generated_readme_content = generated_readme_content.replace("{{ REPO_DESCRIPTION }}", repo_description)
        generated_readme_content = generated_readme_content.replace("{{ ORG_NAME }}", org_name)
        generated_readme_content = generated_readme_content.replace("{{ LICENSE_TYPE }}", license_type)


        readme_file = self.repo_path / "README.md"
        existing_readme_content = ""
        if readme_file.exists():
            # Try decoding with common encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings_to_try:
                try:
                    with open(readme_file, 'r', encoding=encoding) as f_existing:
                        existing_readme_content = f_existing.read()
                    break # Successfully read, break out of loop
                except UnicodeDecodeError:
                    print(f"  [WARN] {self.repo_name}: Failed to read README.md with {encoding} encoding. Trying next.")
                    continue
            else: # If loop completes without break
                print(f"  [ERROR] {self.repo_name}: Failed to read README.md with any common encoding. Skipping README update for this repo.")
                return False # Skip update for this repo due to encoding issue


        if existing_readme_content.strip() != generated_readme_content.strip():
            with open(readme_file, "w", encoding='utf-8') as f: # Always write as UTF-8
                f.write(generated_readme_content)
            self._log_change("Updated README.md")
            return True
        else:
            print(f"  [OK] {self.repo_name}: README.md is up-to-date")
            return False


    def enforce_golden_path(self) -> Dict[str, Any]:
        """Execute full enforcement for this repo."""
        print(f"\n[REPO] Enforcing Golden Path for {self.org}/{self.repo_name}")

        if not self.repo_path.exists():
            print(f"  [WARN] Repo path not found: {self.repo_path}")
            return {"status": "skipped", "reason": "path not found"}

        self.create_or_update_meta_repo_yaml()
        self.create_or_update_codeowners()
        self.enforce_ci_yml()
        self.enforce_policy_yml()
        self.enforce_pre_commit_config()
        self.enforce_dockerfile()
        self.enforce_readme() # Add this line to call the new function

        return {
            "org": self.org,
            "repo": self.repo_name,
            "status": "success" if self.changes_made else "skipped",
            "changes": self.changes_made,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enforce Golden Path compliance across all repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_and_enforce_golden_path.py                    # Run enforcement
    python verify_and_enforce_golden_path.py --dry-run          # Preview changes without writing
    python verify_and_enforce_golden_path.py --verbose --log    # Verbose with log file
    GOLDEN_PATH_ROOT=/path/to/repo python verify_and_enforce_golden_path.py  # Custom root
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing any files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write output to timestamped log file"
    )
    return parser.parse_args()


def main():
    """Main entry point for Golden Path enforcement."""
    args = parse_args()

    # Setup logging
    log_file = None
    if args.log:
        log_file = Path(f"enforcement-{datetime.now():%Y%m%d-%H%M%S}.log")
    setup_logging(verbose=args.verbose, log_file=log_file)

    # Auto-detect central repo path (no more hardcoded paths!)
    try:
        central_repo_path = get_central_repo_path()
        logger.info(f"Central repo detected at: {central_repo_path}")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY-RUN] No files will be modified")

    # Load inventory from the archived location
    inventory_path = central_repo_path / "docs" / "migration-archive" / "inventory.json"
    if not inventory_path.exists():
        logger.error(f"inventory.json not found at {inventory_path}")
        sys.exit(1)

    with open(inventory_path) as f:
        inventory = json.load(f)

    # Base path for repos
    repo_base_path = central_repo_path / "organizations"

    if not repo_base_path.exists():
        logger.error(f"organizations/ path not found at {repo_base_path}")
        sys.exit(1)

    results = []
    total_changes = 0

    print("\n" + "=" * 70)
    print("alaweimm90 Golden Path Enforcement Script")
    if args.dry_run:
        print("                    [DRY-RUN MODE]")
    print("=" * 70)

    for org_data in inventory["organizations"]:
        org_name = org_data["name"]
        print(f"\n[ORG] Organization: {org_name}")

        org_path = repo_base_path / org_name

        if not org_path.exists():
            logger.warning(f"Organization path not found: {org_path}. Skipping.")
            continue

        for repo in org_data["repositories"]:
            repo_name = repo["name"]
            repo_path = org_path / repo_name

            enforcer = RepoEnforcer(
                repo_path, org_name, repo_name, repo,
                central_repo_path, dry_run=args.dry_run
            )
            result = enforcer.enforce_golden_path()
            results.append(result)

            if result["status"] == "success":
                total_changes += len(result.get("changes", []))

    # Summary
    print("\n" + "=" * 70)
    print("Enforcement Summary")
    print("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n[SUCCESS] Enforced: {len(successful)} repos")
    print(f"[SKIPPED] Already compliant: {len(skipped)} repos")
    if failed:
        print(f"[FAILED] Failed to enforce: {len(failed)} repos")
    print(f"[CHANGES] Total changes made: {total_changes}")

    # Save results (unless dry-run)
    if not args.dry_run:
        results_path = central_repo_path / "enforcement-results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Detailed results saved to: {results_path}")
    else:
        print("\n[DRY-RUN] Results not saved (use without --dry-run to save)")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)

    print(f"""
1. Review changes in affected repositories within '{repo_base_path}'.
2. Commit and push these changes to their respective remote repositories.
3. Verify CI runs for all updated repositories.
""")


if __name__ == "__main__":
    main()
