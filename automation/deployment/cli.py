#!/usr/bin/env python3
"""
Deployment CLI - Command-line interface for deployment operations.
"""

import argparse
import json
import sys
from pathlib import Path

from .portfolio import PortfolioDeployer, DeploymentConfig
from .knowledge_base import KnowledgeBaseDeployer
from .web_generator import WebInterfaceGenerator, WebConfig


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
        print("[DRY RUN] Would deploy portfolio:")
        print(f"  Source: {source}")
        print(f"  Output: {output}")
        print(f"  Platform: {args.platform}")
        return 0

    result = deployer.deploy()

    if result.success:
        print("\n✓ Deployment successful!")
        if result.url:
            print(f"  URL: {result.url}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Accessibility Score: {result.accessibility_score:.0%}")
        return 0
    else:
        print("\n✗ Deployment failed!")
        for error in result.errors:
            print(f"  Error: {error}")
        return 1


def cmd_deploy_knowledge_base(args):
    """Deploy a knowledge base."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else source / "ORGANIZED"

    deployer = KnowledgeBaseDeployer(source, output)
    result = deployer.deploy(dry_run=args.dry_run)

    if result["success"]:
        print("\n✓ Knowledge base deployed!")
        print(f"  Files organized: {result['organized']}/{result['total']}")
        print(f"  Categories: {len(result['categories'])}")
        if result.get("web_path"):
            print(f"  Web interface: file://{result['web_path']}/index.html")
        return 0
    else:
        print("\n✗ Deployment failed!")
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

    print(f"✓ Generated: {result}")
    print(f"  Open: file://{result}")
    return 0


def cmd_organize_downloads(args):
    """Organize Downloads folder."""
    downloads = Path(args.path or Path.home() / "Downloads")

    if not downloads.exists():
        print(f"Error: Path does not exist: {downloads}")
        return 1

    deployer = KnowledgeBaseDeployer(downloads)
    result = deployer.deploy(dry_run=args.dry_run)

    print("\n" + "=" * 50)
    print("DOWNLOADS ORGANIZATION COMPLETE")
    print("=" * 50)

    if result["success"]:
        print(f"\nOrganized {result['organized']} files into {len(result['categories'])} categories:")
        for cat, count in result["categories"].items():
            print(f"  {cat}: {count} files")

        if result.get("web_path") and not args.dry_run:
            print(f"\n📁 Web interface ready:")
            print(f"   file://{result['web_path']}/index.html")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Deployment CLI - Deploy portfolios, knowledge bases, and web interfaces"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Portfolio deployment
    portfolio_parser = subparsers.add_parser("portfolio", help="Deploy a portfolio site")
    portfolio_parser.add_argument("source", help="Source directory")
    portfolio_parser.add_argument("--name", help="Project name")
    portfolio_parser.add_argument("--output", "-o", help="Output directory")
    portfolio_parser.add_argument("--platform", "-p", default="local",
                                  choices=["netlify", "vercel", "github-pages", "local"],
                                  help="Deployment platform")
    portfolio_parser.add_argument("--build", "-b", help="Build command")
    portfolio_parser.add_argument("--domain", "-d", help="Custom domain")
    portfolio_parser.add_argument("--accessibility", "-a", choices=["A", "AA", "AAA"],
                                  help="Accessibility level")
    portfolio_parser.add_argument("--dry-run", action="store_true", help="Preview without deploying")

    # Knowledge base deployment
    kb_parser = subparsers.add_parser("knowledge-base", help="Deploy a knowledge base")
    kb_parser.add_argument("source", help="Source directory")
    kb_parser.add_argument("--output", "-o", help="Output directory")
    kb_parser.add_argument("--dry-run", action="store_true", help="Preview without deploying")
    kb_parser.add_argument("--web-only", action="store_true", help="Only generate web interface")
    kb_parser.add_argument("--no-web", action="store_true", help="Skip web interface")

    # Web generator
    web_parser = subparsers.add_parser("web", help="Generate a web interface")
    web_parser.add_argument("--type", "-t", required=True,
                           choices=["dashboard", "file-browser", "documentation", "portfolio"],
                           help="Interface type")
    web_parser.add_argument("--output", "-o", required=True, help="Output directory")
    web_parser.add_argument("--title", help="Page title")
    web_parser.add_argument("--description", help="Page description")
    web_parser.add_argument("--color", help="Primary color (hex)")
    web_parser.add_argument("--no-dark-mode", action="store_true", help="Disable dark mode")

    # Downloads organizer
    downloads_parser = subparsers.add_parser("organize-downloads", help="Organize Downloads folder")
    downloads_parser.add_argument("--path", help="Path to Downloads folder")
    downloads_parser.add_argument("--dry-run", action="store_true", help="Preview without moving files")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "portfolio":
        return cmd_deploy_portfolio(args)
    elif args.command == "knowledge-base":
        return cmd_deploy_knowledge_base(args)
    elif args.command == "web":
        return cmd_generate_web(args)
    elif args.command == "organize-downloads":
        return cmd_organize_downloads(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
