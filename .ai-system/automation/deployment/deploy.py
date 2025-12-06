#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Deployment Script
Deploy any project from the registry with a single command.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set console output to UTF-8 on Windows
if sys.platform == 'win32':
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import yaml

AUTOMATION_PATH = Path(__file__).parent.parent
REPO_ROOT = AUTOMATION_PATH.parent
REGISTRY_PATH = AUTOMATION_PATH / "deployment" / "registry.yaml"
TEMPLATES_PATH = AUTOMATION_PATH / "deployment" / "templates"


def load_registry() -> Dict[str, Any]:
    """Load the deployment registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def find_project(name: str, registry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a project in the registry by name."""
    name_lower = name.lower()

    for org_name, org_data in registry.get("organizations", {}).items():
        for project in org_data.get("projects", []):
            if project.get("name", "").lower() == name_lower:
                project["organization"] = org_name
                return project

    return None


def list_projects(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List all projects in the registry."""
    projects = []

    for org_name, org_data in registry.get("organizations", {}).items():
        for project in org_data.get("projects", []):
            project["organization"] = org_name
            projects.append(project)

    return projects


def get_template(template_type: str) -> Optional[Path]:
    """Get the path to a deployment template."""
    templates = {
        "netlify": TEMPLATES_PATH / "netlify.toml",
        "vercel": TEMPLATES_PATH / "vercel.json",
        "docker": TEMPLATES_PATH / "docker-compose.yaml",
        "kubernetes": TEMPLATES_PATH / "kubernetes.yaml",
        "github-actions": TEMPLATES_PATH / "github-actions.yaml",
        "python-ci": TEMPLATES_PATH / "python-ci.yaml",
        "pypi-publish": TEMPLATES_PATH / "pypi-publish.yaml",
        "fastapi-docker": TEMPLATES_PATH / "fastapi-docker.yaml",
    }
    return templates.get(template_type)


def copy_template(template_type: str, dest_path: Path, variables: Dict[str, str] = None) -> bool:
    """Copy a template to the destination with variable substitution."""
    template_path = get_template(template_type)

    if not template_path or not template_path.exists():
        print(f"Template not found: {template_type}")
        return False

    content = template_path.read_text()

    # Substitute variables
    if variables:
        for key, value in variables.items():
            content = content.replace(f"${{{key}}}", value)
            content = content.replace(f"${key}", value)

    dest_path.write_text(content)
    print(f"Created: {dest_path}")
    return True


def deploy_netlify(project_path: Path, site_name: str = None, prod: bool = False) -> bool:
    """Deploy to Netlify."""
    print(f"\n🚀 Deploying to Netlify...")

    cmd = ["netlify", "deploy"]
    if prod:
        cmd.append("--prod")
    if site_name:
        cmd.extend(["--site", site_name])

    cmd.extend(["--dir", str(project_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("Error: netlify CLI not found. Install with: npm install -g netlify-cli")
        return False


def deploy_vercel(project_path: Path, prod: bool = False) -> bool:
    """Deploy to Vercel."""
    print(f"\n🚀 Deploying to Vercel...")

    cmd = ["vercel"]
    if prod:
        cmd.append("--prod")

    try:
        result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("Error: vercel CLI not found. Install with: npm install -g vercel")
        return False


def deploy_docker(project_path: Path, compose_file: str = "docker-compose.yaml") -> bool:
    """Deploy with Docker Compose."""
    print(f"\n🐳 Deploying with Docker Compose...")

    compose_path = project_path / compose_file
    if not compose_path.exists():
        print(f"Error: {compose_file} not found in {project_path}")
        return False

    cmd = ["docker-compose", "-f", str(compose_path), "up", "-d", "--build"]

    try:
        result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("Error: docker-compose not found")
        return False


def cmd_list(args):
    """List all projects in the registry."""
    registry = load_registry()
    projects = list_projects(registry)

    print("\n[PROJECTS] Registered Projects")
    print("=" * 60)

    current_org = None
    for project in projects:
        if project["organization"] != current_org:
            current_org = project["organization"]
            print(f"\n{current_org}:")

        name = project.get("name", "Unknown")
        ptype = project.get("type", "unknown")
        stack = ", ".join(project.get("stack", [])[:3])

        print(f"  • {name:<20} [{ptype}] {stack}")

    print(f"\nTotal: {len(projects)} projects")
    return 0


def cmd_info(args):
    """Show info about a project."""
    registry = load_registry()
    project = find_project(args.project, registry)

    if not project:
        print(f"Project not found: {args.project}")
        return 1

    print(f"\n[PROJECT] {project.get('name')}")
    print("=" * 40)
    print(f"Organization: {project.get('organization')}")
    print(f"Type: {project.get('type')}")
    print(f"Path: {project.get('path')}")

    if project.get("stack"):
        print(f"Stack: {', '.join(project['stack'])}")

    if project.get("domain"):
        print(f"Domain: {project['domain']}")

    guides = project.get("deployment_guides") or [project.get("deployment_guide")]
    if guides and guides[0]:
        print("\nDeployment Guides:")
        for guide in guides:
            if guide:
                print(f"  • {guide}")

    return 0


def cmd_deploy(args):
    """Deploy a project."""
    registry = load_registry()
    project = find_project(args.project, registry)

    if not project:
        print(f"Project not found: {args.project}")
        return 1

    project_path = REPO_ROOT / project.get("path", "")

    if not project_path.exists():
        print(f"Project path not found: {project_path}")
        return 1

    platform = args.platform or "netlify"

    print(f"\n🚀 Deploying {project.get('name')} to {platform}")
    print(f"   Path: {project_path}")

    if platform == "netlify":
        return 0 if deploy_netlify(project_path, prod=args.prod) else 1
    elif platform == "vercel":
        return 0 if deploy_vercel(project_path, prod=args.prod) else 1
    elif platform == "docker":
        return 0 if deploy_docker(project_path) else 1
    else:
        print(f"Unknown platform: {platform}")
        return 1


def cmd_template(args):
    """Copy a deployment template to a project."""
    dest = Path(args.dest).resolve()

    template_files = {
        "netlify": "netlify.toml",
        "vercel": "vercel.json",
        "docker": "docker-compose.yaml",
        "kubernetes": "kubernetes.yaml",
        "github-actions": ".github/workflows/deploy.yaml",
    }

    filename = template_files.get(args.type)
    if not filename:
        print(f"Unknown template type: {args.type}")
        return 1

    dest_file = dest / filename
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    variables = {}
    if args.name:
        variables["APP_NAME"] = args.name
        variables["PROJECT_NAME"] = args.name

    if copy_template(args.type, dest_file, variables):
        print(f"\n✓ Template copied to {dest_file}")
        return 0
    return 1


def cmd_templates(args):
    """List available templates."""
    print("\n[TEMPLATES] Available Deployment Templates")
    print("=" * 40)

    templates = [
        ("netlify", "Netlify static site deployment"),
        ("vercel", "Vercel serverless deployment"),
        ("docker", "Docker Compose multi-container"),
        ("kubernetes", "Kubernetes full stack"),
        ("github-actions", "GitHub Actions CI/CD"),
        ("python-ci", "Python CI/CD workflow"),
        ("pypi-publish", "PyPI publish workflow"),
        ("fastapi-docker", "FastAPI Docker Compose"),
    ]

    for name, desc in templates:
        template_path = get_template(name)
        exists = "✓" if template_path and template_path.exists() else "✗"
        print(f"  {exists} {name:<15} - {desc}")

    print("\nUsage: deploy template <type> --dest <path>")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified deployment tool for all projects"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all projects")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show project info")
    info_parser.add_argument("project", help="Project name")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a project")
    deploy_parser.add_argument("project", help="Project name")
    deploy_parser.add_argument("--platform", "-p", choices=["netlify", "vercel", "docker"],
                               help="Deployment platform")
    deploy_parser.add_argument("--prod", action="store_true", help="Deploy to production")

    # Template command
    template_parser = subparsers.add_parser("template", help="Copy deployment template")
    template_parser.add_argument("type", choices=[
        "netlify", "vercel", "docker", "kubernetes", "github-actions",
        "python-ci", "pypi-publish", "fastapi-docker"
    ], help="Template type")
    template_parser.add_argument("--dest", "-d", required=True, help="Destination directory")
    template_parser.add_argument("--name", "-n", help="Project name for substitution")

    # Templates command
    templates_parser = subparsers.add_parser("templates", help="List available templates")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "deploy": cmd_deploy,
        "template": cmd_template,
        "templates": cmd_templates,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
