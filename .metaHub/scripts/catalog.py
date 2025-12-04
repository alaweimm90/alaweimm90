"""
Service Catalog Generator.

Scans organizations and repositories to generate a service catalog
with metadata, compliance status, and summary statistics.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class CatalogBuilder:
    """Builds a service catalog from organization structure."""

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

    def __init__(
        self,
        base_path: Path = None,
        org_path: str = None
    ):
        """
        Initialize catalog builder.

        Args:
            base_path: Base path of the repository
            org_path: Path to organizations directory
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.org_path = Path(org_path) if org_path else self.base_path / "organizations"
        self.catalog: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "organizations": [],
            "summary": {
                "total_organizations": 0,
                "total_repositories": 0,
                "by_language": {},
                "by_type": {},
                "by_tier": defaultdict(int),
                "compliance_rate": 0.0
            }
        }

    def scan_organizations(self) -> Dict[str, Any]:
        """
        Scan all organizations and their repositories.

        Returns:
            Catalog dictionary with organizations and summary
        """
        if not self.org_path.exists():
            # Return empty catalog for missing directory (graceful handling)
            return self.catalog

        organizations = []
        total_repos = 0
        compliant_repos = 0
        by_language = defaultdict(int)
        by_type = defaultdict(int)
        by_tier = defaultdict(int)

        for org_dir in sorted(self.org_path.iterdir()):
            if not org_dir.is_dir():
                continue

            org_data = {
                "name": org_dir.name,
                "repos": []
            }

            for repo_dir in sorted(org_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue

                repo_data = self._scan_repository(repo_dir)
                org_data["repos"].append(repo_data)
                total_repos += 1

                # Aggregate statistics
                if repo_data.get("language"):
                    by_language[repo_data["language"]] += 1
                if repo_data.get("type"):
                    by_type[repo_data["type"]] += 1

                tier = repo_data.get("tier", 4)
                by_tier[tier] += 1

                if repo_data.get("compliance", {}).get("is_compliant", False):
                    compliant_repos += 1

            organizations.append(org_data)

        self.catalog["organizations"] = organizations
        self.catalog["summary"]["total_organizations"] = len(organizations)
        self.catalog["summary"]["total_repositories"] = total_repos
        self.catalog["summary"]["by_language"] = dict(by_language)
        self.catalog["summary"]["by_type"] = dict(by_type)
        self.catalog["summary"]["by_tier"] = dict(by_tier)
        self.catalog["summary"]["compliance_rate"] = (
            (compliant_repos / total_repos * 100) if total_repos > 0 else 0
        )

        return self.catalog

    def _scan_repository(self, repo_path: Path) -> Dict[str, Any]:
        """
        Scan a single repository for metadata and compliance.

        Args:
            repo_path: Path to the repository

        Returns:
            Repository data dictionary
        """
        repo_name = repo_path.name
        meta_file = repo_path / ".meta" / "repo.yaml"

        # Initialize repo data
        repo_data = {
            "name": repo_name,
            "path": str(repo_path),
            "has_metadata": meta_file.exists(),
            "type": self._infer_type(repo_name),
            "language": None,
            "tier": 4,  # Default tier
            "status": "unknown",
            "compliance": self._check_compliance(repo_path)
        }

        # Load metadata if exists
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    metadata = yaml.safe_load(f)
                    if metadata:
                        repo_data["type"] = metadata.get("type", repo_data["type"])
                        repo_data["language"] = metadata.get("language")
                        repo_data["tier"] = metadata.get("tier", 4)
                        repo_data["status"] = metadata.get("status", "unknown")
            except (yaml.YAMLError, Exception):
                # Handle invalid YAML gracefully
                pass

        return repo_data

    def _infer_type(self, name: str) -> str:
        """Infer repository type from name prefix."""
        for prefix, repo_type in self.TYPE_PREFIXES.items():
            if name.startswith(prefix):
                return repo_type
        return "unknown"

    def _check_compliance(self, repo_path: Path) -> Dict[str, bool]:
        """
        Check repository compliance indicators.

        Args:
            repo_path: Path to the repository

        Returns:
            Compliance status dictionary
        """
        has_readme = (repo_path / "README.md").exists()
        has_ci = (repo_path / ".github" / "workflows").exists()
        has_metadata = (repo_path / ".meta" / "repo.yaml").exists()
        has_license = (repo_path / "LICENSE").exists() or (repo_path / "LICENSE.md").exists()
        has_tests = (repo_path / "tests").exists() or (repo_path / "test").exists()

        return {
            "has_readme": has_readme,
            "has_ci": has_ci,
            "has_metadata": has_metadata,
            "has_license": has_license,
            "has_tests": has_tests,
            "is_compliant": has_readme and has_metadata
        }

    def generate_json(self, output_file: Path = None) -> str:
        """
        Generate JSON output of the catalog.

        Args:
            output_file: Optional file to write output to

        Returns:
            JSON string
        """
        output = json.dumps(self.catalog, indent=2, default=str)
        if output_file:
            output_file.write_text(output)
        return output

    def generate_markdown(self, output_file: Path = None) -> str:
        """
        Generate Markdown output of the catalog.

        Args:
            output_file: Optional file to write output to

        Returns:
            Markdown string
        """
        lines = [
            "# Service Catalog",
            "",
            f"Generated: {self.catalog['generated_at']}",
            "",
            "## Summary",
            "",
            f"- **Total Organizations:** {self.catalog['summary']['total_organizations']}",
            f"- **Total Repositories:** {self.catalog['summary']['total_repositories']}",
            f"- **Compliance Rate:** {self.catalog['summary']['compliance_rate']:.1f}%",
            "",
            "## Organizations",
            ""
        ]

        for org in self.catalog["organizations"]:
            lines.append(f"### {org['name']}")
            lines.append("")
            lines.append("| Repository | Type | Language | Tier | Compliant |")
            lines.append("|------------|------|----------|------|-----------|")

            for repo in org["repos"]:
                compliant = "✅" if repo["compliance"].get("is_compliant") else "❌"
                lines.append(
                    f"| {repo['name']} | {repo['type']} | "
                    f"{repo.get('language', 'N/A')} | {repo['tier']} | {compliant} |"
                )
            lines.append("")

        output = "\n".join(lines)
        if output_file:
            output_file.write_text(output)
        return output

    def generate_html(self, output_file: Path = None) -> str:
        """
        Generate HTML output of the catalog.

        Args:
            output_file: Optional file to write output to

        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Service Catalog</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .compliant {{ color: green; }}
        .non-compliant {{ color: red; }}
    </style>
</head>
<body>
    <h1>Service Catalog</h1>
    <p>Generated: {self.catalog['generated_at']}</p>

    <h2>Summary</h2>
    <ul>
        <li><strong>Total Organizations:</strong> {self.catalog['summary']['total_organizations']}</li>
        <li><strong>Total Repositories:</strong> {self.catalog['summary']['total_repositories']}</li>
        <li><strong>Compliance Rate:</strong> {self.catalog['summary']['compliance_rate']:.1f}%</li>
    </ul>
"""

        for org in self.catalog["organizations"]:
            html += f"""
    <h3>{org['name']}</h3>
    <table>
        <tr>
            <th>Repository</th>
            <th>Type</th>
            <th>Language</th>
            <th>Tier</th>
            <th>Compliant</th>
        </tr>
"""
            for repo in org["repos"]:
                compliant_class = "compliant" if repo["compliance"].get("is_compliant") else "non-compliant"
                compliant_text = "Yes" if repo["compliance"].get("is_compliant") else "No"
                html += f"""        <tr>
            <td>{repo['name']}</td>
            <td>{repo['type']}</td>
            <td>{repo.get('language', 'N/A')}</td>
            <td>{repo['tier']}</td>
            <td class="{compliant_class}">{compliant_text}</td>
        </tr>
"""
            html += "    </table>\n"

        html += """</body>
</html>"""

        if output_file:
            output_file.write_text(html)
        return html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate service catalog")
    parser.add_argument("--org-path", help="Path to organizations directory")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "markdown", "html"], default="json")

    args = parser.parse_args()

    builder = CatalogBuilder(org_path=args.org_path)
    builder.scan_organizations()

    if args.format == "json":
        output = builder.generate_json(Path(args.output) if args.output else None)
    elif args.format == "markdown":
        output = builder.generate_markdown(Path(args.output) if args.output else None)
    else:
        output = builder.generate_html(Path(args.output) if args.output else None)

    if not args.output:
        print(output)
