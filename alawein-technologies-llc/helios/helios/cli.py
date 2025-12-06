#!/usr/bin/env python
"""
HELIOS Command Line Interface

Usage:
    helios --version
    helios --help
    helios generate <topic> [--domain=<domain>] [--num=<n>]
    helios validate <hypothesis> [--domain=<domain>]
    helios research <topic> [--domain=<domain>] [--full]
    helios domain list
    helios domain info <domain>
"""

import sys
import argparse
from typing import Optional, List


def print_banner() -> None:
    """Print HELIOS banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║         HELIOS: Autonomous Research Discovery System          ║
║  Hypothesis Exploration & Learning Intelligence Orchestration ║
║                     Version 0.1.0 MVP                         ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def list_domains() -> None:
    """List all available research domains."""
    try:
        from helios.domains import DOMAINS

        print("\nAvailable Research Domains:\n")
        for i, (name, domain_class) in enumerate(DOMAINS.items(), 1):
            try:
                domain = domain_class()
                display = getattr(domain, 'display_name', name)
                description = getattr(domain, 'description', 'No description')
                print(f"{i}. {display} ({name})")
                print(f"   {description}\n")
            except Exception:
                print(f"{i}. {name}\n")
    except ImportError:
        print("Error: Could not import domains")
        sys.exit(1)


def domain_info(domain_name: str) -> None:
    """Show information about a specific domain."""
    try:
        from helios.domains import DOMAINS

        if domain_name not in DOMAINS:
            print(f"Error: Unknown domain '{domain_name}'")
            print(f"Available: {', '.join(DOMAINS.keys())}")
            sys.exit(1)

        domain = DOMAINS[domain_name]()

        print(f"\n{'='*60}")
        print(f"Domain: {getattr(domain, 'display_name', domain_name)}")
        print(f"{'='*60}\n")

        print(f"Name: {domain_name}")
        print(f"Description: {getattr(domain, 'description', 'N/A')}\n")

        # Try to get additional info
        if hasattr(domain, 'get_benchmarks'):
            try:
                benchmarks = domain.get_benchmarks()
                print(f"Benchmarks: {list(benchmarks.keys())[:5]}...")
            except Exception:
                pass

    except ImportError:
        print("Error: Could not import domains")
        sys.exit(1)


def generate_hypotheses(topic: str, domain: Optional[str] = None, num: int = 5) -> None:
    """Generate hypotheses for a research topic."""
    print(f"\nGenerating {num} hypotheses for: '{topic}'")
    if domain:
        print(f"Domain: {domain}\n")

    try:
        from helios import HypothesisGenerator

        generator = HypothesisGenerator(num_hypotheses=num)
        hypotheses = generator.generate(topic, domain=domain)

        print(f"Generated {len(hypotheses)} hypotheses:\n")
        for i, h in enumerate(hypotheses, 1):
            text = h.get('text', 'Hypothesis')[:70]
            score = h.get('novelty_score', 0)
            print(f"{i}. [{score:.1f}] {text}...")

    except ImportError:
        print("Error: Could not import HypothesisGenerator")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def validate_hypothesis(hypothesis: str, domain: Optional[str] = None) -> None:
    """Validate a research hypothesis."""
    print(f"\nValidating hypothesis: '{hypothesis}'")
    if domain:
        print(f"Domain: {domain}\n")

    try:
        from helios import TuringValidator

        validator = TuringValidator()
        result = validator.validate([hypothesis])

        if result:
            r = result[0]
            print(f"Validation Results:")
            print(f"  Logical Score:     {r.get('logical_score', 0):.1f}/100")
            print(f"  Empirical Score:   {r.get('empirical_score', 0):.1f}/100")
            print(f"  Analogical Score:  {r.get('analogical_score', 0):.1f}/100")
            print(f"  Boundary Score:    {r.get('boundary_score', 0):.1f}/100")
            print(f"  Mechanism Score:   {r.get('mechanism_score', 0):.1f}/100")
            print(f"\n  Overall Score:     {r.get('overall_score', 0):.1f}/100\n")

            if r.get('weaknesses'):
                print("Identified Weaknesses:")
                for w in r['weaknesses'][:5]:
                    print(f"  - {w}")

    except ImportError:
        print("Error: Could not import TuringValidator")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_research(topic: str, domain: Optional[str] = None, full: bool = False) -> None:
    """Run full research workflow."""
    print(f"\nStarting research workflow for: '{topic}'")
    if domain:
        print(f"Domain: {domain}")
    if full:
        print("Mode: Full workflow (with paper generation)")
    print()

    try:
        from helios.core.orchestration import WorkflowOrchestrator

        orchestrator = WorkflowOrchestrator()

        print("This feature requires full HELIOS setup and API keys.")
        print("See: helios/docs/GETTING_STARTED.md for setup instructions")

    except ImportError:
        print("Error: Could not import WorkflowOrchestrator")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='HELIOS: Autonomous Research Discovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  helios generate "machine learning"
  helios generate "quantum computing" --domain=quantum --num=10
  helios validate "Hypothesis text"
  helios research "Materials discovery" --domain=materials --full
  helios domain list
  helios domain info quantum
        """
    )

    parser.add_argument('--version', action='version', version='HELIOS 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate hypotheses')
    gen_parser.add_argument('topic', help='Research topic')
    gen_parser.add_argument('--domain', help='Research domain')
    gen_parser.add_argument('--num', type=int, default=5, help='Number of hypotheses')

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate hypothesis')
    val_parser.add_argument('hypothesis', help='Hypothesis text')
    val_parser.add_argument('--domain', help='Research domain')

    # Research command
    res_parser = subparsers.add_parser('research', help='Run research workflow')
    res_parser.add_argument('topic', help='Research topic')
    res_parser.add_argument('--domain', help='Research domain')
    res_parser.add_argument('--full', action='store_true', help='Full workflow with paper')

    # Domain command
    dom_parser = subparsers.add_parser('domain', help='Domain management')
    dom_subparsers = dom_parser.add_subparsers(dest='domain_action')
    dom_subparsers.add_parser('list', help='List all domains')
    dom_info_parser = dom_subparsers.add_parser('info', help='Domain information')
    dom_info_parser.add_argument('name', help='Domain name')

    # Parse arguments
    parsed = parser.parse_args(args)

    # Print banner on first use
    if len(sys.argv) > 1:
        print_banner()

    # Handle commands
    if parsed.command == 'generate':
        generate_hypotheses(parsed.topic, parsed.domain, parsed.num)
        return 0

    elif parsed.command == 'validate':
        validate_hypothesis(parsed.hypothesis, parsed.domain)
        return 0

    elif parsed.command == 'research':
        run_research(parsed.topic, parsed.domain, parsed.full)
        return 0

    elif parsed.command == 'domain':
        if parsed.domain_action == 'list':
            list_domains()
            return 0
        elif parsed.domain_action == 'info':
            domain_info(parsed.domain_action if hasattr(parsed, 'domain_action') else parsed.name)
            return 0

    else:
        print_banner()
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
