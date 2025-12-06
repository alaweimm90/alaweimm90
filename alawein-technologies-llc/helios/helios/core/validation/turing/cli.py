"""
CLI for 200-Question Interrogation Framework
"""

import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from interrogation import InterrogationProtocol
from self_refutation import Hypothesis, HypothesisDomain

app = typer.Typer(help="200-Question Interrogation Framework CLI")
console = Console()


@app.command()
def interrogate(
    claim: str = typer.Argument(..., help="Hypothesis claim to interrogate"),
    domain: str = typer.Option("general", help="Scientific domain"),
    context: Optional[str] = typer.Option(None, help="Additional context"),
    with_orchestrator: bool = typer.Option(False, help="Use AI Orchestrator"),
    use_consensus: bool = typer.Option(False, help="Use multi-model consensus"),
    categories: Optional[str] = typer.Option(None, help="Comma-separated category names"),
    questions_limit: Optional[int] = typer.Option(None, help="Questions per category limit"),
):
    """
    Interrogate a hypothesis with 200 questions

    Example:
        interrogate "Our optimizer achieves 15% improvement" --domain optimization
    """
    # Create hypothesis
    try:
        hyp_domain = HypothesisDomain(domain)
    except ValueError:
        console.print(f"[red]Invalid domain: {domain}[/red]")
        return

    hypothesis = Hypothesis(
        claim=claim,
        domain=hyp_domain,
        context=context,
    )

    # Initialize protocol
    orchestrator = None
    if with_orchestrator:
        try:
            from atlas_orchestrator import Orchestrator
            orchestrator = Orchestrator()
            console.print("[green]✓[/green] Using AI Orchestrator")
        except ImportError:
            console.print("[yellow]⚠[/yellow] ORCHEX-orchestrator not available")

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(',')]

    protocol = InterrogationProtocol(
        orchestrator=orchestrator,
        use_consensus=use_consensus,
        categories=cat_list,
        questions_per_category=questions_limit,
    )

    # Run interrogation
    console.print("\n[bold]🔍 Running 200-Question Interrogation...[/bold]\n")
    result = asyncio.run(protocol.interrogate(hypothesis))

    # Display results
    _display_result(result)


@app.command()
def info():
    """Show information about the question database"""
    from interrogation import QuestionLoader

    loader = QuestionLoader()
    info_data = loader.get_info()

    console.print("\n[bold]Question Database Info:[/bold]")
    console.print(f"  Version: {info_data['version']}")
    console.print(f"  Total Questions: {info_data['total_questions']}")
    console.print(f"  Categories: {info_data['categories']}")
    console.print(f"  Last Updated: {info_data['last_updated']}")
    console.print(f"  Database Path: {info_data['database_path']}\n")

    # Show categories
    console.print("[bold]Categories:[/bold]")
    table = Table(show_header=True)
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Weight", justify="center")
    table.add_column("Questions", justify="center")

    for cat in loader.get_all_categories():
        table.add_row(
            str(cat.id),
            cat.name,
            f"{cat.weight}x",
            str(cat.question_count),
        )

    console.print(table)


@app.command()
def validate_db():
    """Validate the question database"""
    from interrogation import QuestionLoader

    loader = QuestionLoader()
    validation = loader.validate_database()

    if validation["valid"]:
        console.print("\n[green]✓ Database is valid![/green]\n")
    else:
        console.print("\n[red]✗ Database has issues:[/red]\n")
        for issue in validation["issues"]:
            console.print(f"  [red]• {issue}[/red]")

    if validation["warnings"]:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in validation["warnings"]:
            console.print(f"  [yellow]• {warning}[/yellow]")

    console.print(f"\nTotal Questions: {validation['total_questions']}")
    console.print(f"Total Categories: {validation['total_categories']}\n")


@app.command()
def example():
    """Run example interrogation"""
    hypothesis = Hypothesis(
        claim="Our new optimization algorithm always finds the global optimum in polynomial time",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 20 QAPLIB instances",
        assumptions=["Algorithm uses simulated annealing", "Cooling schedule is exponential"],
    )

    console.print("\n[bold]🔍 Running Example Interrogation...[/bold]")
    console.print(f"\n[cyan]Claim:[/cyan] {hypothesis.claim}\n")

    protocol = InterrogationProtocol(
        questions_per_category=5  # Limit for demo
    )

    result = asyncio.run(protocol.interrogate(hypothesis))
    _display_result(result)


def _display_result(result):
    """Display interrogation result"""
    # Overall score panel
    score_color = "green" if result.overall_score >= 70 else "yellow" if result.overall_score >= 55 else "red"

    panel = Panel(
        f"{result.status_emoji} [bold {score_color}]{result.overall_score:.1f}/100[/bold {score_color}]\n\n"
        f"{result.interpretation}\n\n"
        f"Questions Asked: {result.total_questions}\n"
        f"Execution Time: {result.execution_time_seconds:.1f}s",
        title="Overall Interrogation Score",
        border_style=score_color,
    )
    console.print(panel)

    # Category breakdown
    table = Table(title="\nCategory Breakdown", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Weight", justify="center")
    table.add_column("Status")

    for cr in result.category_results:
        score = cr.raw_score
        if score >= 80:
            status = "[green]✓ Strong[/green]"
        elif score >= 60:
            status = "[yellow]✓ Adequate[/yellow]"
        elif score >= 40:
            status = "[orange]⚠ Weak[/orange]"
        else:
            status = "[red]✗ Critical[/red]"

        score_color_cat = "green" if score >= 60 else "red"

        table.add_row(
            cr.category_name,
            f"[{score_color_cat}]{score:.1f}/100[/{score_color_cat}]",
            f"{cr.category_weight}x",
            status,
        )

    console.print(table)

    # Strengths and Weaknesses
    if result.strong_categories:
        console.print("\n[bold green]✓ Strong Points:[/bold green]")
        for cat in result.strong_categories:
            console.print(f"  • {cat}")

    if result.weak_categories or result.critical_categories:
        console.print("\n[bold yellow]⚠ Weak Points:[/bold yellow]")
        for cat in result.weak_categories + result.critical_categories:
            console.print(f"  • {cat}")

    # Failure Points
    if result.failure_points:
        console.print("\n[bold red]✗ Failure Points:[/bold red]")
        for fp in result.failure_points[:5]:
            console.print(f"  • {fp}")

    # Recommendations
    if result.recommendations:
        console.print("\n[bold]💡 Recommendations:[/bold]")
        for rec in result.recommendations[:7]:
            console.print(f"  {rec}")

    console.print()


if __name__ == "__main__":
    app()
