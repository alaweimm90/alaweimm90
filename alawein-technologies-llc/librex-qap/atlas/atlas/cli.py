"""
ORCHEX CLI
"""

import asyncio
import json
import os
from typing import Optional

import typer
from rich.console import Console

from ORCHEX import ATLASProtocol

DEFAULT_PREFLIGHT = os.getenv("ATLAS_PREFLIGHT", "1").lower() not in {"0", "false", "no"}

app = typer.Typer(help="ORCHEX - Autonomous Research System (NOT Nobel Turing Challenge)")
console = Console()


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic"),
    domain: str = typer.Option("optimization", help="Scientific domain"),
    hypotheses: int = typer.Option(5, help="Number of hypotheses to generate"),
    with_orchestrator: bool = typer.Option(False, help="Use AI Orchestrator"),
    no_meta_learning: bool = typer.Option(False, help="Disable meta-learning (personality agents)"),
    debug: Optional[bool] = typer.Option(
        None,
        "--debug/--no-debug",
        help="Enable verbose debug logs (overrides ATLAS_DEBUG)",
    ),
    preflight: bool = typer.Option(
        DEFAULT_PREFLIGHT,
        "--preflight/--skip-preflight",
        help="Run diagnostics before executing the pipeline",
    ),
):
    """
    Run autonomous research pipeline with personality agents!

    Example:
        ORCHEX research "Reinforcement learning for QAP solving" --domain optimization

    Features:
        - 🧠 Self-learning from past projects (meta-learning)
        - 🎭 Personality agents (Grumpy Refuter 😠, Skeptical Steve 🤨, etc.)
        - 📊 Performance tracking and optimization
    """
    console.print("\n[bold cyan]ORCHEX - Autonomous Research System[/bold cyan]")
    console.print("[yellow]⚠️  NOT Nobel Turing Challenge - computational prototype only[/yellow]\n")

    # CLI flag overrides env var when provided; otherwise mirror ATLAS_DEBUG
    atlas_debug = (
        debug
        if debug is not None
        else os.getenv("ATLAS_DEBUG", "0").lower() not in {"0", "false", "no"}
    )
    if atlas_debug:
        console.print("[dim]ATLAS_DEBUG=1 — verbose logging enabled[/dim]")

    if preflight:
        from ORCHEX.diagnostics import run_diagnostics, summarize_diagnostics

        diagnostics = run_diagnostics()
        console.print(summarize_diagnostics(diagnostics))
        if not diagnostics["healthy"]:
            console.print("[red]Preflight failed. Resolve missing critical modules or pass --skip-preflight to override.[/red]")
            raise typer.Exit(code=1)
        console.print("[green]✓ Preflight checks passed[/green]\n")

    orchestrator = None
    if with_orchestrator:
        try:
            from atlas_orchestrator import Orchestrator
            if atlas_debug:
                console.print("[dim]Initializing Orchestrator…[/dim]")
            orchestrator = Orchestrator()
            console.print("[green]✓[/green] Using AI Orchestrator\n")
        except ImportError:
            console.print("[yellow]⚠️[/yellow] ORCHEX-orchestrator not available\n")

    # Meta-learning enabled by default!
    enable_meta_learning = not no_meta_learning

    protocol = ATLASProtocol(
        orchestrator=orchestrator,
        enable_meta_learning=enable_meta_learning
    )

    if atlas_debug:
        console.print("[dim]Starting protocol.run_research…[/dim]")
    project = asyncio.run(
        protocol.run_research(topic=topic, domain=domain, num_hypotheses=hypotheses)
    )
    if atlas_debug:
        console.print("[dim]protocol.run_research completed[/dim]")

    console.print(f"\n[green]✓ Research complete![/green]")
    console.print(f"Output: {project.output_dir}\n")


@app.command()
def info():
    """Show system information"""
    console.print("\n[bold]ORCHEX - Autonomous Theorist & Laboratory Autonomous System[/bold]\n")
    console.print("Version: 0.1.0\n")

    console.print("[yellow]⚠️  IMPORTANT DISCLAIMER[/yellow]")
    console.print("This is NOT the Nobel Turing Challenge.")
    console.print("ORCHEX is a computational research prototype inspired by")
    console.print("autonomous discovery systems, but with significant limitations:\n")

    console.print("  • Computational experiments only (no wet lab)")
    console.print("  • Hours/days timeline (not years)")
    console.print("  • Incremental discoveries (not Nobel-level)")
    console.print("  • Prototype quality (not production-ready)\n")

    console.print("[cyan]What ORCHEX CAN do:[/cyan]")
    console.print("  ✓ Generate research hypotheses from topics")
    console.print("  ✓ Validate hypotheses (self-refutation + interrogation)")
    console.print("  ✓ Learn from failures (Hall of Failures)")
    console.print("  ✓ Create project repositories")
    console.print("  ⏳ Run computational experiments (coming soon)")
    console.print("  ⏳ Generate papers (coming soon)\n")


@app.command()
def diagnostics(json_output: bool = typer.Option(False, "--json", help="Return JSON instead of text")):
    """Run dependency + API key diagnostics."""
    from ORCHEX.diagnostics import run_diagnostics, summarize_diagnostics

    result = run_diagnostics()
    if json_output:
        console.print_json(data=result)
    else:
        console.print(summarize_diagnostics(result))
    if not result["healthy"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
