"""
Meta Auditor & Project Promotion System.

Provides comprehensive portfolio auditing and project promotion
capabilities for governance compliance.
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class GapSeverity(Enum):
    """Severity levels for governance gaps."""
    P0 = "critical"  # Blocks promotion
    P1 = "high"      # Should fix before promotion
    P2 = "medium"    # Recommended to fix
    P3 = "low"       # Nice to have


@dataclass
class GovernanceGap:
    """Represents a governance compliance gap."""
    severity: GapSeverity
    category: str
    message: str
    file_path: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ProjectAudit:
    """Audit result for a single project."""
    name: str
    path: str
    organization: str
    language: Optional[str]
    repo_type: str
    has_metadata: bool
    gaps: List[GovernanceGap] = field(default_factory=list)
    compliance_score: float = 0.0

    @property
    def promotion_ready(self) -> bool:
        """Check if project is ready for promotion (no P0 gaps)."""
        return not any(g.severity == GapSeverity.P0 for g in self.gaps)


class MetaAuditor:
    """Audits portfolio projects for governance compliance."""

    # Type inference from name prefixes
    TYPE_PREFIXES = {
        "lib-": "library",
        "tool-": "tool",
        "svc-": "service",
        "app-": "application",
        "demo-": "demo",
        "infra-": "infrastructure",
        "pkg-": "package",
    }

    # Language detection files
    LANGUAGE_INDICATORS = {
        "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
        "typescript": ["tsconfig.json", "package.json"],
        "javascript": ["package.json"],
        "go": ["go.mod", "go.sum"],
        "rust": ["Cargo.toml"],
        "java": ["pom.xml", "build.gradle"],
    }

    def __init__(self, base_path: Path = None):
        """
        Initialize meta auditor.

        Args:
            base_path: Base path of the repository
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.org_path = self.base_path / "organizations"
        self.templates_path = self.base_path / ".metaHub" / "templates"
        self.audits: List[ProjectAudit] = []

    def scan_all_projects(self, org_filter: str = None) -> List[ProjectAudit]:
        """
        Scan all projects in portfolio.

        Args:
            org_filter: Optional organization name to filter

        Returns:
            List of project audits
        """
        self.audits = []

        if not self.org_path.exists():
            return self.audits

        for org_dir in self.org_path.iterdir():
            if not org_dir.is_dir():
                continue

            if org_filter and org_dir.name != org_filter:
                continue

            for project_dir in org_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                audit = self._audit_project(project_dir, org_dir.name)
                self.audits.append(audit)

        return self.audits

    def _audit_project(self, project_path: Path, organization: str) -> ProjectAudit:
        """Audit a single project."""
        name = project_path.name
        meta_file = project_path / ".meta" / "repo.yaml"
        has_metadata = meta_file.exists()

        # Detect language and type
        language = self._detect_language(project_path)
        repo_type = self._infer_type(name)

        # Load metadata if exists
        tier = 4
        if has_metadata:
            try:
                with open(meta_file) as f:
                    metadata = yaml.safe_load(f) or {}
                    tier = metadata.get("tier", 4)
                    if metadata.get("type"):
                        repo_type = metadata["type"]
                    if metadata.get("language"):
                        language = metadata["language"]
            except Exception:
                pass

        # Create audit
        audit = ProjectAudit(
            name=name,
            path=str(project_path),
            organization=organization,
            language=language,
            repo_type=repo_type,
            has_metadata=has_metadata,
            gaps=[],
            compliance_score=100.0
        )

        # Run gap detection
        self._detect_gaps(project_path, audit, tier)

        # Calculate compliance score
        audit.compliance_score = self._calculate_score(audit)

        return audit

    def _detect_language(self, project_path: Path) -> Optional[str]:
        """Detect project language from files."""
        for language, indicators in self.LANGUAGE_INDICATORS.items():
            if any((project_path / ind).exists() for ind in indicators):
                return language
        return None

    def _infer_type(self, name: str) -> str:
        """Infer project type from name prefix."""
        for prefix, repo_type in self.TYPE_PREFIXES.items():
            if name.startswith(prefix):
                return repo_type
        return "unknown"

    def _detect_gaps(self, project_path: Path, audit: ProjectAudit, tier: int) -> None:
        """Detect governance gaps in project."""
        gaps = []

        # P0: Missing metadata
        if not audit.has_metadata:
            gaps.append(GovernanceGap(
                severity=GapSeverity.P0,
                category="metadata",
                message="Missing .meta/repo.yaml metadata file",
                suggestion="Create .meta/repo.yaml with type, language, and tier"
            ))

        # P0: Missing README (for all tiers)
        readme = project_path / "README.md"
        if not readme.exists():
            gaps.append(GovernanceGap(
                severity=GapSeverity.P0,
                category="documentation",
                message="Missing README.md file",
                suggestion="Create README.md with project description"
            ))
        elif len(readme.read_text()) < 50:
            gaps.append(GovernanceGap(
                severity=GapSeverity.P2,
                category="documentation",
                message="README.md is too short",
                suggestion="Add more content to README.md"
            ))

        # Missing LICENSE (severity based on tier)
        if not any((project_path / f).exists() for f in ("LICENSE", "LICENSE.md")):
            gaps.append(GovernanceGap(
                severity=GapSeverity.P1 if tier <= 2 else GapSeverity.P3,
                category="legal",
                message="Missing LICENSE file",
                suggestion="Add LICENSE file with appropriate license"
            ))

        # P1: Missing CODEOWNERS (tier 1-2)
        if tier <= 2:
            codeowners = project_path / ".github" / "CODEOWNERS"
            if not codeowners.exists():
                gaps.append(GovernanceGap(
                    severity=GapSeverity.P1,
                    category="ownership",
                    message="Missing .github/CODEOWNERS file",
                    suggestion="Create CODEOWNERS file with team ownership"
                ))

        # P1: Missing CI (tier 1-2)
        if tier <= 2:
            ci_dir = project_path / ".github" / "workflows"
            if not ci_dir.exists() or not any(ci_dir.glob("*.yml")):
                gaps.append(GovernanceGap(
                    severity=GapSeverity.P1,
                    category="ci",
                    message="Missing CI/CD workflows",
                    suggestion="Add GitHub Actions workflow for CI"
                ))

        # P2: Missing tests (tier 1-2)
        if tier <= 2 and not any((project_path / d).exists() for d in ("tests", "test")):
            gaps.append(GovernanceGap(
                severity=GapSeverity.P2,
                category="testing",
                message="Missing tests directory",
                suggestion="Add tests directory with test files"
            ))

        audit.gaps = gaps

    def _calculate_score(self, audit: ProjectAudit) -> float:
        """Calculate compliance score based on gaps."""
        if not audit.gaps:
            return 100.0

        deductions = {GapSeverity.P0: 30, GapSeverity.P1: 15, GapSeverity.P2: 5, GapSeverity.P3: 2}
        return max(0.0, 100.0 - sum(deductions[g.severity] for g in audit.gaps))

    def generate_report(self, fmt: str = "text") -> str:
        """
        Generate audit report.

        Args:
            fmt: Output format ('text', 'markdown', or 'json')

        Returns:
            Formatted report string
        """
        if fmt == "json":
            return json.dumps({
                "total_projects": len(self.audits),
                "projects": [
                    {
                        "name": a.name,
                        "organization": a.organization,
                        "language": a.language,
                        "type": a.repo_type,
                        "compliance_score": a.compliance_score,
                        "promotion_ready": a.promotion_ready,
                        "gaps": [
                            {"severity": g.severity.name, "category": g.category, "message": g.message}
                            for g in a.gaps
                        ]
                    }
                    for a in self.audits
                ],
                "summary": {
                    "avg_score": sum(a.compliance_score for a in self.audits) / len(self.audits) if self.audits else 0,
                    "promotion_ready": sum(1 for a in self.audits if a.promotion_ready),
                    "needs_work": sum(1 for a in self.audits if not a.promotion_ready),
                }
            }, indent=2)

        if fmt == "markdown":
            return self._markdown_report()

        return self._text_report()

    def _text_report(self) -> str:
        """Generate text format report."""
        lines = [
            "=" * 60,
            "META AUDITOR - Portfolio Compliance Report",
            "=" * 60,
            "",
            f"Total Projects: {len(self.audits)}",
            f"Promotion Ready: {sum(1 for a in self.audits if a.promotion_ready)}",
            f"Needs Work: {sum(1 for a in self.audits if not a.promotion_ready)}",
            "",
        ]

        for audit in self.audits:
            status = "✅" if audit.promotion_ready else "❌"
            lines.append(f"{status} {audit.organization}/{audit.name}")
            lines.append(f"   Score: {audit.compliance_score:.1f}%")
            if audit.gaps:
                for gap in audit.gaps:
                    lines.append(f"   - [{gap.severity.name}] {gap.message}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _markdown_report(self) -> str:
        """Generate markdown format report."""
        lines = [
            "# Portfolio Compliance Audit Report",
            "",
            "## Summary",
            "",
            f"- **Total Projects:** {len(self.audits)}",
            f"- **Promotion Ready:** {sum(1 for a in self.audits if a.promotion_ready)}",
            f"- **Needs Work:** {sum(1 for a in self.audits if not a.promotion_ready)}",
            "",
            "## Projects",
            "",
        ]

        for audit in self.audits:
            status = "✅" if audit.promotion_ready else "❌"
            lines.append(f"### {status} {audit.organization}/{audit.name}")
            lines.append("")
            lines.append(f"- **Score:** {audit.compliance_score:.1f}%")
            lines.append(f"- **Language:** {audit.language or 'Unknown'}")
            lines.append(f"- **Type:** {audit.repo_type}")
            lines.append("")
            if audit.gaps:
                lines.append("**Gaps:**")
                for gap in audit.gaps:
                    lines.append(f"- [{gap.severity.name}] {gap.message}")
                lines.append("")

        return "\n".join(lines)


class ProjectPromoter:
    """Promotes projects to governance compliance."""

    def __init__(self, base_path: Path = None):
        """
        Initialize project promoter.

        Args:
            base_path: Base path of the repository
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.templates_path = self.base_path / ".metaHub" / "templates"

    def promote_project(
        self,
        project_path: Path,
        organization: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Promote a project to governance compliance.

        Args:
            project_path: Path to the project
            organization: Organization name
            dry_run: If True, don't create files

        Returns:
            Promotion result dictionary
        """
        project_path = Path(project_path)

        if not project_path.exists():
            return {"success": False, "error": "Project path does not exist"}

        result = {
            "success": True,
            "project": project_path.name,
            "organization": organization,
            "files_created": [],
            "dry_run": dry_run
        }

        # Detect language
        language = self._detect_language(project_path)

        # Create .meta/repo.yaml
        meta_dir = project_path / ".meta"
        meta_file = meta_dir / "repo.yaml"
        if not meta_file.exists():
            metadata = {
                "type": self._infer_type(project_path.name),
                "language": language or "unknown",
                "tier": 3,  # Default tier
                "status": "active"
            }
            if not dry_run:
                meta_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_file, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)
            result["files_created"].append(str(meta_file))

        # Create .github/CODEOWNERS
        github_dir = project_path / ".github"
        codeowners_file = github_dir / "CODEOWNERS"
        if not codeowners_file.exists():
            content = f"* @{organization}\n"
            if not dry_run:
                github_dir.mkdir(parents=True, exist_ok=True)
                codeowners_file.write_text(content)
            result["files_created"].append(str(codeowners_file))

        # Create .github/workflows/ci.yml
        workflows_dir = github_dir / "workflows"
        ci_file = workflows_dir / "ci.yml"
        if not ci_file.exists():
            ci_content = self._generate_ci_workflow(language)
            if not dry_run:
                workflows_dir.mkdir(parents=True, exist_ok=True)
                ci_file.write_text(ci_content)
            result["files_created"].append(str(ci_file))

        # Create .pre-commit-config.yaml
        precommit_file = project_path / ".pre-commit-config.yaml"
        if not precommit_file.exists():
            precommit_content = self._get_precommit_config(language)
            if precommit_content and not dry_run:
                precommit_file.write_text(precommit_content)
            if precommit_content:
                result["files_created"].append(str(precommit_file))

        return result

    def _detect_language(self, project_path: Path) -> Optional[str]:
        """Detect project language from files."""
        lang_map = {
            "pyproject.toml": "python",
            "tsconfig.json": "typescript",
            "package.json": "javascript",
            "go.mod": "go",
            "Cargo.toml": "rust"
        }
        for file, lang in lang_map.items():
            if (project_path / file).exists():
                return lang
        return None

    def _infer_type(self, name: str) -> str:
        """Infer project type from name prefix."""
        for prefix, repo_type in self.TYPE_PREFIXES.items():
            if name.startswith(prefix):
                return repo_type
        return "unknown"

    def _generate_ci_workflow(self, language: str) -> str:
        """Generate CI workflow for language."""
        base = {
            "name": "CI",
            "on": {"push": {"branches": ["main"]}, "pull_request": {"branches": ["main"]}},
            "permissions": {"contents": "read"},
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"}
                    ]
                }
            }
        }

        if language == "python":
            base["jobs"]["build"]["steps"].extend([
                {"name": "Set up Python", "uses": "actions/setup-python@v5", "with": {"python-version": "3.11"}},
                {"name": "Install dependencies", "run": "pip install -e '.[dev]'"},
                {"name": "Run tests", "run": "pytest"}
            ])
        elif language in ("typescript", "javascript"):
            base["jobs"]["build"]["steps"].extend([
                {"name": "Set up Node", "uses": "actions/setup-node@v4", "with": {"node-version": "20"}},
                {"name": "Install dependencies", "run": "npm ci"},
                {"name": "Run tests", "run": "npm test"}
            ])

        return yaml.dump(base, default_flow_style=False)

    def _get_precommit_config(self, language: str) -> Optional[str]:
        """Get pre-commit config for language."""
        # Try to load from templates
        template_file = self.templates_path / "pre-commit" / f"{language}.yaml"
        if template_file.exists():
            return template_file.read_text()

        generic_file = self.templates_path / "pre-commit" / "generic.yaml"
        if generic_file.exists():
            return generic_file.read_text()

        # Default minimal config
        return yaml.dump({
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v4.5.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"}
                    ]
                }
            ]
        }, default_flow_style=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta Auditor & Promoter")
    parser.add_argument("action", choices=["audit", "promote"])
    parser.add_argument("--org", help="Organization filter")
    parser.add_argument("--project", help="Project path for promotion")
    parser.add_argument("--format", "-f", choices=["text", "markdown", "json"], default="text")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.action == "audit":
        auditor = MetaAuditor()
        auditor.scan_all_projects(org_filter=args.org)
        print(auditor.generate_report(fmt=args.format))

    elif args.action == "promote":
        if not args.project or not args.org:
            print("Error: --project and --org required for promote")
        else:
            promoter = ProjectPromoter()
            result = promoter.promote_project(
                Path(args.project),
                args.org,
                dry_run=args.dry_run
            )
            print(json.dumps(result, indent=2))
