#!/usr/bin/env python3
"""
Deployment CLI - Command-line interface for deployment operations.
"""

import argparse
import json
import sys
import os
from pathlib import Path

from .portfolio import PortfolioDeployer, DeploymentConfig
from .knowledge_base import KnowledgeBaseDeployer
from .web_generator import WebInterfaceGenerator, WebConfig


RESET = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"

NO_COLOR = bool(os.getenv("NO_COLOR"))
if NO_COLOR:
    RESET = GREEN = RED = YELLOW = BLUE = ""

def _emit(kind: str, msg: str, style: str = "compact"):
    icon = {"success": "✅", "error": "❌", "warn": "⚠️", "info": "ℹ️"}[kind]
    color = {"success": GREEN, "error": RED, "warn": YELLOW, "info": BLUE}[kind]
    if style == "json":
        print(json.dumps({"level": kind, "message": msg}))
        return
    print(f"{color}{icon} {msg}{RESET}")

def cmd_deploy_portfolio(args):
    """Deploy a portfolio site."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else source / "deploy"

    config = DeploymentConfig(
        name=args.name or source.name,
        source_path=source,
        output_path=output,
        platform=args.platform,
        build_command=args.build,
        domain=args.domain,
        accessibility_level=args.accessibility or "AA"
    )

    deployer = PortfolioDeployer(config)

    if args.dry_run:
        _emit("info", f"portfolio (dry) {config.name} {args.platform} {source} → {output}", getattr(args, "style", "compact"))
        return 0

    result = deployer.deploy()

    if result.success:
        msg = f"portfolio {config.name} ok"
        if result.url:
            msg += f" url={result.url}"
        msg += f" t={result.duration_seconds:.1f}s acc={result.accessibility_score:.0%}"
        _emit("success", msg, getattr(args, "style", "compact"))
        return 0
    else:
        _emit("error", f"portfolio {config.name} failed errors={len(result.errors)}", getattr(args, "style", "compact"))
        return 1


def cmd_deploy_knowledge_base(args):
    """Deploy a knowledge base."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else source / "ORGANIZED"

    deployer = KnowledgeBaseDeployer(source, output)
    result = deployer.deploy(dry_run=args.dry_run)

    if result["success"]:
        msg = f"kb ok organized={result['organized']}/{result['total']} categories={len(result['categories'])}"
        if result.get("web_path"):
            msg += f" web=file://{result['web_path']}/index.html"
        _emit("success", msg, getattr(args, "style", "compact"))
        return 0
    else:
        _emit("error", "kb failed", getattr(args, "style", "compact"))
        return 1


def cmd_generate_web(args):
    """Generate a web interface."""
    output = Path(args.output).resolve()

    config = WebConfig(
        title=args.title or "Web Interface",
        description=args.description or "",
        primary_color=args.color or "#3498db",
        enable_dark_mode=not args.no_dark_mode
    )

    generator = WebInterfaceGenerator(output, config)

    if args.type == "dashboard":
        # Demo dashboard
        result = generator.generate_dashboard(
            sections=[
                {"id": "welcome", "title": "Welcome", "content": "Your dashboard is ready."},
            ],
            stats={"Status": "Ready", "Version": "1.0"}
        )
    elif args.type == "file-browser":
        # Demo file browser
        result = generator.generate_file_browser(files=[])
    elif args.type == "documentation":
        result = generator.generate_documentation(docs=[
            {"id": "intro", "title": "Introduction", "content": "Welcome to the documentation."}
        ])
    elif args.type == "portfolio":
        result = generator.generate_portfolio(projects=[])
    else:
        print(f"Unknown template type: {args.type}")
        return 1

    _emit("success", f"web {args.type} generated path={result}", getattr(args, "style", "compact"))
    return 0


def cmd_organize_downloads(args):
    """Organize Downloads folder."""
    downloads = Path(args.path or Path.home() / "Downloads")

    if not downloads.exists():
        print(f"Error: Path does not exist: {downloads}")
        return 1

    deployer = KnowledgeBaseDeployer(downloads)
    result = deployer.deploy(dry_run=args.dry_run)

    if result["success"]:
        msg = f"downloads ok organized={result['organized']} cats={len(result['categories'])}"
        if result.get("web_path") and not args.dry_run:
            msg += f" web=file://{result['web_path']}/index.html"
        _emit("success", msg, getattr(args, "style", "compact"))

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deployment CLI - Deploy portfolios, knowledge bases, and web interfaces")
    subs = parser.add_subparsers(dest="command", help="Commands")

    p = subs.add_parser("portfolio", help="Deploy a portfolio site")
    p.add_argument("source", help="Source directory")
    p.add_argument("--name", help="Project name")
    p.add_argument("--output", "-o", help="Output directory")
    p.add_argument("--platform", "-p", default="local", choices=["netlify", "vercel", "github-pages", "local"], help="Deployment platform")
    p.add_argument("--build", "-b", help="Build command")
    p.add_argument("--domain", "-d", help="Custom domain")
    p.add_argument("--accessibility", "-a", choices=["A", "AA", "AAA"], help="Accessibility level")
    p.add_argument("--dry-run", action="store_true", help="Preview without deploying")

    kb = subs.add_parser("knowledge-base", help="Deploy a knowledge base")
    kb.add_argument("source", help="Source directory")
    kb.add_argument("--output", "-o", help="Output directory")
    kb.add_argument("--dry-run", action="store_true", help="Preview without deploying")
    kb.add_argument("--web-only", action="store_true", help="Only generate web interface")
    kb.add_argument("--no-web", action="store_true", help="Skip web interface")

    web = subs.add_parser("web", help="Generate a web interface")
    web.add_argument("--type", "-t", required=True, choices=["dashboard", "file-browser", "documentation", "portfolio"], help="Interface type")
    web.add_argument("--output", "-o", required=True, help="Output directory")
    web.add_argument("--title", help="Page title")
    web.add_argument("--description", help="Page description")
    web.add_argument("--color", help="Primary color (hex)")
    web.add_argument("--no-dark-mode", action="store_true", help="Disable dark mode")

    dl = subs.add_parser("organize-downloads", help="Organize Downloads folder")
    dl.add_argument("--path", help="Path to Downloads folder")
    dl.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    return parser

def _dispatch(args: argparse.Namespace) -> int:
    if not args.command:
        return 1
    if args.command == "portfolio":
        return cmd_deploy_portfolio(args)
    if args.command == "knowledge-base":
        return cmd_deploy_knowledge_base(args)
    if args.command == "web":
        return cmd_generate_web(args)
    if args.command == "organize-downloads":
        return cmd_organize_downloads(args)
    return 0

def main():
    parser = _build_parser()
    default_style = os.getenv("ATLAS_OUTPUT_STYLE", "compact")
    parser.add_argument("--style", choices=["compact", "json"], default=default_style, help="Output style")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    code = _dispatch(args)
    if code == 0:
        _emit("success", "summary ok", getattr(args, "style", "compact"))
    else:
        _emit("error", "summary fail", getattr(args, "style", "compact"))
    return code


if __name__ == "__main__":
    sys.exit(main())
