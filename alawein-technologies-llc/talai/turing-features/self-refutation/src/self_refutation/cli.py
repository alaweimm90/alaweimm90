"""
CLI for Self-Refutation Protocol

Usage:
    refute "Your hypothesis claim here" --domain optimization
    refute-file hypothesis.json
    refute-batch hypotheses.json
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from self_refutation import (
    SelfRefutationProtocol,
    Hypothesis,
    HypothesisDomain,
)

app = typer.Typer(help="Self-Refutation Protocol CLI - Popperian Falsification for AI")
console = Console()


@app.command()
def refute(
    claim: str = typer.Argument(..., help="Hypothesis claim to test"),
    domain: str = typer.Option("general", help="Scientific domain"),
    context: Optional[str] = typer.Option(None, help="Additional context"),
    with_orchestrator: bool = typer.Option(False, help="Use AI Orchestrator for deeper analysis"),
    parallel: bool = typer.Option(True, help="Run strategies in parallel"),
    output: Optional[Path] = typer.Option(None, help="Save result to JSON file"),
):
    """
    Test a single hypothesis using the Self-Refutation Protocol

    Example:
        refute "Our new QAP solver achieves 10% better results" --domain optimization
    """
    # Validate domain
    try:
        hypothesis_domain = HypothesisDomain(domain)
    except ValueError:
        console.print(f"[red]Invalid domain: {domain}[/red]")
        console.print(f"Valid domains: {', '.join([d.value for d in HypothesisDomain])}")
        sys.exit(1)

    # Create hypothesis
    hypothesis = Hypothesis(
        claim=claim,
        domain=hypothesis_domain,
        context=context,
    )

    # Initialize protocol
    orchestrator = None
    if with_orchestrator:
        try:
            from atlas_orchestrator import Orchestrator
            orchestrator = Orchestrator()
            console.print("[green]✓[/green] Using AI Orchestrator for enhanced analysis")
        except ImportError:
            console.print("[yellow]⚠[/yellow] ORCHEX-orchestrator not available, running without AI")

    protocol = SelfRefutationProtocol(orchestrator=orchestrator, parallel=parallel)

    # Run refutation
    console.print("\n[bold]🔬 Running Self-Refutation Protocol...[/bold]\n")

    result = asyncio.run(protocol.refute(hypothesis))

    # Display results
    _display_result(result)

    # Save to file if requested
    if output:
        _save_result(result, output)
        console.print(f"\n[green]✓[/green] Results saved to {output}")


@app.command()
def refute_file(
    file_path: Path = typer.Argument(..., help="Path to hypothesis JSON file"),
    with_orchestrator: bool = typer.Option(False, help="Use AI Orchestrator"),
    output: Optional[Path] = typer.Option(None, help="Save result to JSON file"),
):
    """
    Test hypothesis from JSON file

    JSON format:
    {
        "claim": "Your hypothesis here",
        "domain": "optimization",
        "context": "Optional context",
        "evidence": ["evidence 1", "evidence 2"],
        "assumptions": ["assumption 1"]
    }
    """
    # Load hypothesis
    try:
        with open(file_path) as f:
            data = json.load(f)

        hypothesis = Hypothesis(**data)
    except Exception as e:
        console.print(f"[red]Error loading hypothesis: {e}[/red]")
        sys.exit(1)

    # Initialize protocol
    orchestrator = None
    if with_orchestrator:
        try:
            from atlas_orchestrator import Orchestrator
            orchestrator = Orchestrator()
        except ImportError:
            console.print("[yellow]⚠[/yellow] ORCHEX-orchestrator not available")

    protocol = SelfRefutationProtocol(orchestrator=orchestrator)

    # Run refutation
    console.print(f"\n[bold]🔬 Testing hypothesis from {file_path}...[/bold]\n")
    result = asyncio.run(protocol.refute(hypothesis))

    # Display results
    _display_result(result)

    # Save if requested
    if output:
        _save_result(result, output)


@app.command()
def refute_batch(
    file_path: Path = typer.Argument(..., help="Path to hypotheses JSON file"),
    with_orchestrator: bool = typer.Option(False, help="Use AI Orchestrator"),
    output: Optional[Path] = typer.Option(None, help="Save results to JSON file"),
    min_score: float = typer.Option(0.0, help="Only show hypotheses with score >= this"),
):
    """
    Test multiple hypotheses from JSON file

    JSON format:
    [
        {"claim": "Hypothesis 1", "domain": "optimization"},
        {"claim": "Hypothesis 2", "domain": "machine_learning"}
    ]
    """
    # Load hypotheses
    try:
        with open(file_path) as f:
            data = json.load(f)

        hypotheses = [Hypothesis(**h) for h in data]
    except Exception as e:
        console.print(f"[red]Error loading hypotheses: {e}[/red]")
        sys.exit(1)

    # Initialize protocol
    orchestrator = None
    if with_orchestrator:
        try:
            from atlas_orchestrator import Orchestrator
            orchestrator = Orchestrator()
        except ImportError:
            pass

    protocol = SelfRefutationProtocol(orchestrator=orchestrator)

    # Run batch refutation
    console.print(f"\n[bold]🔬 Testing {len(hypotheses)} hypotheses...[/bold]\n")
    results = asyncio.run(protocol.refute_batch(hypotheses))

    # Display summary table
    _display_batch_results(results, min_score)

    # Save if requested
    if output:
        output_data = [
            {
                "hypothesis": r.hypothesis.dict(),
                "strength_score": r.strength_score,
                "refuted": r.refuted,
                "confidence": r.confidence.value,
                "interpretation": r.interpretation,
            }
            for r in results
        ]
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]✓[/green] Results saved to {output}")


@app.command()
def example():
    """
    Run example hypothesis through protocol
    """
    hypothesis = Hypothesis(
        claim="Our new QAP solver always finds the global optimum in polynomial time",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 20 QAPLIB instances",
        assumptions=[
            "Algorithm uses simulated annealing",
            "Temperature schedule is exponential"
        ],
    )

    console.print("\n[bold]🔬 Running Example Hypothesis...[/bold]")
    console.print(f"\n[cyan]Claim:[/cyan] {hypothesis.claim}")
    console.print(f"[cyan]Domain:[/cyan] {hypothesis.domain.value}")
    console.print(f"[cyan]Context:[/cyan] {hypothesis.context}\n")

    protocol = SelfRefutationProtocol()
    result = asyncio.run(protocol.refute(hypothesis))

    _display_result(result)


def _display_result(result):
    """Display refutation result in nice format"""
    # Main score panel
    score_color = "green" if result.strength_score >= 61 else "yellow" if result.strength_score >= 41 else "red"

    score_panel = Panel(
        f"[bold {score_color}]{result.strength_score:.1f}/100[/bold {score_color}]\n\n"
        f"{result.interpretation}\n\n"
        f"Passed: {result.strategies_passed}/{result.total_strategies} strategies\n"
        f"Confidence: {result.confidence.value.upper()}",
        title="Hypothesis Strength Score",
        border_style=score_color,
    )
    console.print(score_panel)

    # Strategy results table
    table = Table(title="Strategy Results", show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Result", justify="center")
    table.add_column("Confidence")
    table.add_column("Severity", justify="right")
    table.add_column("Reasoning", max_width=50)

    for sr in result.strategy_results:
        result_icon = "✓" if sr.passed else "✗"
        result_color = "green" if sr.passed else "red"

        table.add_row(
            sr.strategy.value.replace("_", " ").title(),
            f"[{result_color}]{result_icon}[/{result_color}]",
            sr.confidence.value,
            f"{sr.severity:.2f}",
            sr.reasoning,
        )

    console.print("\n")
    console.print(table)

    # Evidence/Issues found
    if any(sr.evidence for sr in result.strategy_results):
        console.print("\n[bold]📋 Evidence Found:[/bold]\n")
        for sr in result.strategy_results:
            if sr.evidence:
                console.print(f"[yellow]▪[/yellow] [bold]{sr.strategy.value}:[/bold]")
                for evidence in sr.evidence[:3]:  # Show first 3
                    console.print(f"  • {evidence}")
                if len(sr.evidence) > 3:
                    console.print(f"  ... and {len(sr.evidence) - 3} more")

    # Recommendations
    if result.recommendations:
        console.print("\n[bold]💡 Recommendations:[/bold]\n")
        for rec in result.recommendations:
            console.print(f"  {rec}")

    console.print()


def _display_batch_results(results, min_score: float = 0.0):
    """Display batch results summary"""
    # Filter by min score
    filtered = [r for r in results if r.strength_score >= min_score]

    if not filtered:
        console.print(f"[yellow]No hypotheses with score >= {min_score}[/yellow]")
        return

    # Summary statistics
    avg_score = sum(r.strength_score for r in results) / len(results)
    strong = sum(1 for r in results if r.strength_score >= 61)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total: {len(results)}")
    console.print(f"  Average Score: {avg_score:.1f}/100")
    console.print(f"  Strong (≥61): {strong}")
    console.print()

    # Results table
    table = Table(title="Batch Results", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Claim", max_width=50)
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Interpretation")

    for i, result in enumerate(filtered, 1):
        score_color = "green" if result.strength_score >= 61 else "yellow" if result.strength_score >= 41 else "red"
        status = "✓ Pass" if not result.refuted else "✗ Refuted"
        status_color = "green" if not result.refuted else "red"

        table.add_row(
            str(i),
            result.hypothesis.claim,
            f"[{score_color}]{result.strength_score:.1f}[/{score_color}]",
            f"[{status_color}]{status}[/{status_color}]",
            result.interpretation,
        )

    console.print(table)


def _save_result(result, output_path: Path):
    """Save result to JSON file"""
    # Convert to dict
    data = {
        "hypothesis": result.hypothesis.dict(),
        "strength_score": result.strength_score,
        "strategies_passed": result.strategies_passed,
        "total_strategies": result.total_strategies,
        "refuted": result.refuted,
        "refutation_reason": result.refutation_reason,
        "confidence": result.confidence.value,
        "interpretation": result.interpretation,
        "strategy_results": [
            {
                "strategy": sr.strategy.value,
                "passed": sr.passed,
                "confidence": sr.confidence.value,
                "reasoning": sr.reasoning,
                "evidence": sr.evidence,
                "severity": sr.severity,
            }
            for sr in result.strategy_results
        ],
        "recommendations": result.recommendations,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    app()
