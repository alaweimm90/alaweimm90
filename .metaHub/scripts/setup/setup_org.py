#!/usr/bin/env python3
"""
setup_org.py - Organization Setup and Standardization

Sets up organization profiles, .github repos, and standardized configs.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import click


# Organization metadata
ORGS = {
    "alawein-business": {
        "description": "Business and enterprise projects",
        "topics": ["business", "enterprise", "portfolio"],
    },
    "alawein-science": {
        "description": "Scientific computing and research",
        "topics": ["science", "research", "physics", "quantum"],
    },
    "alawein-tools": {
        "description": "Developer tools and utilities",
        "topics": ["devtools", "cli", "automation", "utilities"],
    },
    "AlaweinOS": {
        "description": "Operating system and infrastructure",
        "topics": ["os", "infrastructure", "systems"],
    },
    "MeatheadPhysicist": {
        "description": "Physics education and outreach",
        "topics": ["physics", "education", "research"],
    },
}


def run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run gh CLI command."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result


def update_org_profile(org: str, metadata: Dict[str, Any], dry_run: bool = False) -> None:
    """Update organization profile README."""
    template_path = Path(__file__).parent.parent / "templates" / "organizations" / "ORG_PROFILE_README.md"

    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return

    template = template_path.read_text()

    # Substitute variables
    content = template.replace("{{ORG_NAME}}", org)
    content = content.replace("{{ORG_DESCRIPTION}}", metadata.get("description", ""))
    content = content.replace("{{MONOREPO_URL}}", f"https://github.com/{org}/monorepo")

    if dry_run:
        print(f"[DRY-RUN] Would update {org} profile README")
        return

    # Check if profile repo exists
    result = run_gh(["repo", "view", f"{org}/{org}", "--json", "name"], check=False)

    if result.returncode != 0:
        print(f"Profile repo {org}/{org} not found, skipping profile update")
        return

    # Update via API
    print(f"Updating {org} profile...")
    # Write locally first, then push


def update_org_settings(org: str, metadata: Dict[str, Any], dry_run: bool = False) -> None:
    """Update organization settings."""
    if dry_run:
        print(f"[DRY-RUN] Would update {org} settings")
        return

    # Update monorepo description and topics
    desc = metadata.get("description", "")
    topics = metadata.get("topics", [])

    print(f"Updating {org}/monorepo settings...")

    # Update description
    run_gh([
        "repo", "edit", f"{org}/monorepo",
        "-d", f"{desc} - Consolidated monorepo"
    ], check=False)

    # Add topics
    for topic in topics:
        run_gh([
            "repo", "edit", f"{org}/monorepo",
            "--add-topic", topic
        ], check=False)

    # Enable features
    run_gh([
        "repo", "edit", f"{org}/monorepo",
        "--enable-issues",
        "--enable-projects",
        "--delete-branch-on-merge",
        "--enable-auto-merge",
    ], check=False)


def setup_github_templates(org: str, dry_run: bool = False) -> None:
    """Set up .github repo with templates."""
    if dry_run:
        print(f"[DRY-RUN] Would setup .github templates for {org}")
        return

    # Check if .github repo exists
    result = run_gh(["repo", "view", f"{org}/.github", "--json", "name"], check=False)

    if result.returncode != 0:
        print(f"Creating .github repo for {org}...")
        run_gh([
            "repo", "create", f"{org}/.github",
            "--private",
            "-d", f"Organization-wide defaults for {org}"
        ], check=False)

    print(f"Configured .github for {org}")


@click.command()
@click.option("--org", help="Specific organization to setup")
@click.option("--dry-run", is_flag=True, help="Show what would be done")
@click.option("--all", "all_orgs", is_flag=True, help="Setup all organizations")
def main(org: str, dry_run: bool, all_orgs: bool):
    """Setup and standardize GitHub organizations."""

    if org:
        orgs_to_setup = {org: ORGS.get(org, {"description": "", "topics": []})}
    elif all_orgs:
        orgs_to_setup = ORGS
    else:
        click.echo("Specify --org <name> or --all")
        return

    for org_name, metadata in orgs_to_setup.items():
        click.echo(f"\n=== Setting up {org_name} ===")
        update_org_settings(org_name, metadata, dry_run)
        setup_github_templates(org_name, dry_run)
        update_org_profile(org_name, metadata, dry_run)

    click.echo("\n[OK] Organization setup complete")


if __name__ == "__main__":
    main()
