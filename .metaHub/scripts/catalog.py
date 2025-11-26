#!/usr/bin/env python3
"""
catalog.py - Organization Catalog Generator

Generates a comprehensive service catalog from repository metadata:
- Scans all organizations and repositories
- Reads .meta/repo.yaml for each repo
- Generates catalog.json with full inventory
- Supports multiple output formats (JSON, Markdown, HTML)

Usage:
    python catalog.py                               # Generate catalog.json
    python catalog.py --format markdown --output catalog.md
    python catalog.py --format html --output catalog.html
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import click
import yaml
from tabulate import tabulate


class CatalogBuilder:
    """Builds a comprehensive service catalog from repository metadata."""

    def __init__(self, base_path: Optional[Path] = None,
                 org_path: Optional[str] = None):
        self.base_path = base_path or self._find_base_path()
        self.org_path = Path(org_path) if org_path else (self.base_path / "organizations")
        self.catalog: Dict[str, Any] = {
            "version": "2.0",
            "generated_at": datetime.now().isoformat(),
            "generator": "catalog.py",
            "organizations": [],
            "summary": {
                "total_organizations": 0,
                "total_repositories": 0,
                "by_language": {},
                "by_type": {},
                "by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
                "by_status": {}
            }
        }

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        # Check environment variable
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path

        # Search up from current directory
        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent

        # Fallback to script location
        script_path = Path(__file__).resolve().parent.parent.parent
        if (script_path / ".metaHub").exists():
            return script_path

        raise RuntimeError("Could not find central governance repo (.metaHub directory)")

    def scan_organizations(self) -> Dict[str, Any]:
        """Scan and catalog all repositories."""
        if not self.org_path.exists():
            raise RuntimeError(f"Organizations path not found: {self.org_path}")

        # Scan each organization directory
        for org_dir in sorted(self.org_path.iterdir()):
            if not org_dir.is_dir():
                continue
            if org_dir.name.startswith('.'):
                continue

            org_entry = self._scan_organization(org_dir)
            if org_entry["repos"]:  # Only add if has repos
                self.catalog["organizations"].append(org_entry)
                self.catalog["summary"]["total_organizations"] += 1

        # Sort organizations by name
        self.catalog["organizations"].sort(key=lambda x: x["name"])

        return self.catalog

    def _scan_organization(self, org_dir: Path) -> Dict[str, Any]:
        """Scan a single organization directory."""
        org_entry = {
            "name": org_dir.name,
            "path": str(org_dir),
            "repos": [],
            "summary": {
                "total_repos": 0,
                "by_language": {},
                "by_type": {},
                "by_tier": {1: 0, 2: 0, 3: 0, 4: 0}
            }
        }

        # Scan each repository
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            if repo_dir.name.startswith('.'):
                continue

            repo_entry = self._scan_repository(repo_dir, org_dir.name)
            org_entry["repos"].append(repo_entry)
            org_entry["summary"]["total_repos"] += 1
            self.catalog["summary"]["total_repositories"] += 1

            # Update summaries
            self._update_summaries(repo_entry, org_entry)

        # Sort repos by name
        org_entry["repos"].sort(key=lambda x: x["name"])

        return org_entry

    def _scan_repository(self, repo_dir: Path, org_name: str) -> Dict[str, Any]:
        """Scan a single repository."""
        repo_entry = {
            "name": repo_dir.name,
            "path": str(repo_dir),
            "organization": org_name,
            "full_name": f"{org_name}/{repo_dir.name}",
            "metadata": None,
            "has_metadata": False,
            "compliance": {
                "has_readme": (repo_dir / "README.md").exists(),
                "has_codeowners": (repo_dir / ".github" / "CODEOWNERS").exists(),
                "has_ci": (repo_dir / ".github" / "workflows").exists(),
                "has_tests": (repo_dir / "tests").is_dir() or (repo_dir / "test").is_dir(),
                "has_dockerfile": any(repo_dir.glob("**/Dockerfile*"))
            }
        }

        # Read metadata
        meta_file = repo_dir / ".meta" / "repo.yaml"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f) or {}
                repo_entry["metadata"] = metadata
                repo_entry["has_metadata"] = True

                # Extract key fields for easy access
                repo_entry["type"] = metadata.get("type", "unknown")
                repo_entry["language"] = metadata.get("language", "unknown")
                repo_entry["tier"] = metadata.get("tier", 4)
                repo_entry["status"] = metadata.get("status", "unknown")
                repo_entry["owner"] = metadata.get("owner", org_name)
                repo_entry["description"] = metadata.get("description", "")

            except (yaml.YAMLError, IOError) as e:
                repo_entry["metadata_error"] = str(e)
        else:
            # Infer from directory structure
            repo_entry["type"] = self._infer_type(repo_dir)
            repo_entry["language"] = self._infer_language(repo_dir)
            repo_entry["tier"] = 4
            repo_entry["status"] = "unknown"
            repo_entry["owner"] = org_name
            repo_entry["description"] = ""

        return repo_entry

    def _infer_type(self, repo_dir: Path) -> str:
        """Infer repository type from name."""
        name = repo_dir.name.lower()
        prefixes = {
            "lib-": "library",
            "tool-": "tool",
            "adapter-": "adapter",
            "demo-": "demo",
            "paper-": "research",
            "core-": "tool",
            "infra-": "tool",
            "template-": "demo"
        }
        for prefix, repo_type in prefixes.items():
            if name.startswith(prefix):
                return repo_type
        return "unknown"

    def _infer_language(self, repo_dir: Path) -> str:
        """Infer primary language from project files."""
        if (repo_dir / "pyproject.toml").exists() or (repo_dir / "setup.py").exists():
            return "python"
        if (repo_dir / "package.json").exists():
            return "typescript"
        if (repo_dir / "go.mod").exists():
            return "go"
        if (repo_dir / "Cargo.toml").exists():
            return "rust"
        if (repo_dir / "pom.xml").exists():
            return "java"
        return "unknown"

    def _update_summaries(self, repo_entry: Dict, org_entry: Dict):
        """Update catalog and organization summaries."""
        language = repo_entry.get("language", "unknown")
        repo_type = repo_entry.get("type", "unknown")
        tier = repo_entry.get("tier", 4)
        status = repo_entry.get("status", "unknown")

        # Update catalog summary
        self.catalog["summary"]["by_language"][language] = \
            self.catalog["summary"]["by_language"].get(language, 0) + 1
        self.catalog["summary"]["by_type"][repo_type] = \
            self.catalog["summary"]["by_type"].get(repo_type, 0) + 1
        self.catalog["summary"]["by_tier"][tier] = \
            self.catalog["summary"]["by_tier"].get(tier, 0) + 1
        self.catalog["summary"]["by_status"][status] = \
            self.catalog["summary"]["by_status"].get(status, 0) + 1

        # Update org summary
        org_entry["summary"]["by_language"][language] = \
            org_entry["summary"]["by_language"].get(language, 0) + 1
        org_entry["summary"]["by_type"][repo_type] = \
            org_entry["summary"]["by_type"].get(repo_type, 0) + 1
        org_entry["summary"]["by_tier"][tier] = \
            org_entry["summary"]["by_tier"].get(tier, 0) + 1

    def generate_json(self, output_file: Optional[Path] = None) -> str:
        """Generate catalog as JSON."""
        json_output = json.dumps(self.catalog, indent=2, default=str)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
        return json_output

    def generate_markdown(self, output_file: Optional[Path] = None) -> str:
        """Generate catalog as Markdown."""
        lines = [
            "# Service Catalog",
            "",
            f"Generated: {self.catalog['generated_at']}",
            "",
            "## Summary",
            "",
            f"- **Total Organizations**: {self.catalog['summary']['total_organizations']}",
            f"- **Total Repositories**: {self.catalog['summary']['total_repositories']}",
            "",
            "### By Language",
            "",
        ]

        # Language breakdown
        for lang, count in sorted(self.catalog["summary"]["by_language"].items()):
            lines.append(f"- {lang}: {count}")

        lines.extend([
            "",
            "### By Type",
            "",
        ])

        # Type breakdown
        for repo_type, count in sorted(self.catalog["summary"]["by_type"].items()):
            lines.append(f"- {repo_type}: {count}")

        lines.extend([
            "",
            "### By Tier",
            "",
            f"- Tier 1 (Mission-Critical): {self.catalog['summary']['by_tier'].get(1, 0)}",
            f"- Tier 2 (Important): {self.catalog['summary']['by_tier'].get(2, 0)}",
            f"- Tier 3 (Experimental): {self.catalog['summary']['by_tier'].get(3, 0)}",
            f"- Tier 4 (Unknown): {self.catalog['summary']['by_tier'].get(4, 0)}",
            "",
            "---",
            "",
        ])

        # Organizations and repos
        for org in self.catalog["organizations"]:
            lines.extend([
                f"## {org['name']}",
                "",
                f"Repositories: {org['summary']['total_repos']}",
                "",
                "| Repository | Type | Language | Tier | Status |",
                "|------------|------|----------|------|--------|",
            ])

            for repo in org["repos"]:
                lines.append(
                    f"| {repo['name']} | {repo.get('type', '-')} | "
                    f"{repo.get('language', '-')} | {repo.get('tier', '-')} | "
                    f"{repo.get('status', '-')} |"
                )

            lines.extend(["", "---", ""])

        md_output = '\n'.join(lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_output)
        return md_output

    def generate_html(self, output_file: Optional[Path] = None) -> str:
        """Generate catalog as HTML."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Catalog</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        tr:hover {{ background: #f9f9f9; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .summary-card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .summary-card h3 {{ margin-top: 0; }}
        .tier-1 {{ background: #ffebee; }}
        .tier-2 {{ background: #fff3e0; }}
        .tier-3 {{ background: #e3f2fd; }}
        .tier-4 {{ background: #f5f5f5; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-python {{ background: #3776ab; color: white; }}
        .badge-typescript {{ background: #3178c6; color: white; }}
        .badge-go {{ background: #00add8; color: white; }}
        .badge-rust {{ background: #dea584; color: black; }}
        .timestamp {{ color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>Service Catalog</h1>
    <p class="timestamp">Generated: {self.catalog['generated_at']}</p>

    <div class="summary">
        <div class="summary-card">
            <h3>Organizations</h3>
            <p style="font-size: 24px; font-weight: bold;">{self.catalog['summary']['total_organizations']}</p>
        </div>
        <div class="summary-card">
            <h3>Repositories</h3>
            <p style="font-size: 24px; font-weight: bold;">{self.catalog['summary']['total_repositories']}</p>
        </div>
        <div class="summary-card tier-1">
            <h3>Tier 1 (Critical)</h3>
            <p style="font-size: 24px; font-weight: bold;">{self.catalog['summary']['by_tier'].get(1, 0)}</p>
        </div>
        <div class="summary-card tier-2">
            <h3>Tier 2 (Important)</h3>
            <p style="font-size: 24px; font-weight: bold;">{self.catalog['summary']['by_tier'].get(2, 0)}</p>
        </div>
    </div>

    <h2>By Language</h2>
    <table>
        <tr><th>Language</th><th>Count</th></tr>
"""
        for lang, count in sorted(self.catalog["summary"]["by_language"].items(),
                                   key=lambda x: -x[1]):
            html += f"        <tr><td>{lang}</td><td>{count}</td></tr>\n"

        html += """    </table>

    <h2>By Type</h2>
    <table>
        <tr><th>Type</th><th>Count</th></tr>
"""
        for repo_type, count in sorted(self.catalog["summary"]["by_type"].items(),
                                        key=lambda x: -x[1]):
            html += f"        <tr><td>{repo_type}</td><td>{count}</td></tr>\n"

        html += "    </table>\n"

        # Organizations and repos
        for org in self.catalog["organizations"]:
            html += f"""
    <h2>{org['name']}</h2>
    <p>Repositories: {org['summary']['total_repos']}</p>
    <table>
        <tr>
            <th>Repository</th>
            <th>Type</th>
            <th>Language</th>
            <th>Tier</th>
            <th>Status</th>
            <th>Compliance</th>
        </tr>
"""
            for repo in org["repos"]:
                tier_class = f"tier-{repo.get('tier', 4)}"
                compliance = repo.get("compliance", {})
                compliance_icons = []
                if compliance.get("has_readme"):
                    compliance_icons.append("README")
                if compliance.get("has_ci"):
                    compliance_icons.append("CI")
                if compliance.get("has_tests"):
                    compliance_icons.append("Tests")

                html += f"""        <tr class="{tier_class}">
            <td><strong>{repo['name']}</strong></td>
            <td>{repo.get('type', '-')}</td>
            <td><span class="badge badge-{repo.get('language', 'unknown')}">{repo.get('language', '-')}</span></td>
            <td>{repo.get('tier', '-')}</td>
            <td>{repo.get('status', '-')}</td>
            <td>{', '.join(compliance_icons) if compliance_icons else '-'}</td>
        </tr>
"""
            html += "    </table>\n"

        html += """
</body>
</html>"""

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
        return html


@click.command()
@click.option('--org-path', type=click.Path(exists=True),
              help='Path to organizations directory')
@click.option('--format', 'fmt', type=click.Choice(['json', 'markdown', 'html']),
              default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def main(org_path: Optional[str], fmt: str, output: Optional[str], quiet: bool):
    """
    Generate a service catalog from repository metadata.

    Scans all organizations and repositories, reading .meta/repo.yaml
    files to build a comprehensive catalog.

    Examples:

        python catalog.py

        python catalog.py --format markdown --output catalog.md

        python catalog.py --format html --output service-catalog.html
    """
    try:
        builder = CatalogBuilder(org_path=org_path)

        if not quiet:
            click.echo(f"Scanning organizations at: {builder.org_path}")

        catalog = builder.scan_organizations()

        if not quiet:
            click.echo(f"\nFound {catalog['summary']['total_organizations']} organizations")
            click.echo(f"Found {catalog['summary']['total_repositories']} repositories")

        # Generate output
        output_path = Path(output) if output else None

        if fmt == 'json':
            result = builder.generate_json(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.json"
        elif fmt == 'markdown':
            result = builder.generate_markdown(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.md"
        else:  # html
            result = builder.generate_html(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.html"

        # Write to default location if no output specified
        if not output_path:
            default_file.parent.mkdir(parents=True, exist_ok=True)
            with open(default_file, 'w', encoding='utf-8') as f:
                f.write(result)
            if not quiet:
                click.echo(f"\nCatalog written to: {default_file}")
        else:
            if not quiet:
                click.echo(f"\nCatalog written to: {output_path}")

        # Print summary
        if not quiet:
            click.echo("\n" + "="*50)
            click.echo("CATALOG SUMMARY")
            click.echo("="*50)
            click.echo(f"\nBy Language:")
            for lang, count in sorted(catalog["summary"]["by_language"].items(),
                                       key=lambda x: -x[1]):
                click.echo(f"  {lang}: {count}")

            click.echo(f"\nBy Type:")
            for repo_type, count in sorted(catalog["summary"]["by_type"].items(),
                                            key=lambda x: -x[1]):
                click.echo(f"  {repo_type}: {count}")

            click.echo(f"\nBy Tier:")
            for tier in [1, 2, 3, 4]:
                count = catalog["summary"]["by_tier"].get(tier, 0)
                click.echo(f"  Tier {tier}: {count}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
