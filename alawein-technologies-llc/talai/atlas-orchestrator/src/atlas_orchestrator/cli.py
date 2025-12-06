"""
CLI interface for ORCHEX Orchestrator
"""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from atlas_orchestrator import Orchestrator, Task, TaskType

app = typer.Typer(help="ORCHEX Orchestrator - Multi-model AI orchestration")
console = Console()


@app.command()
def execute(
    prompt: str = typer.Argument(..., help="Prompt to send to AI"),
    task_type: str = typer.Option("simple", help="Type of task (code, research, analysis, etc.)"),
    max_tokens: int = typer.Option(1000, help="Maximum tokens in response"),
    temperature: float = typer.Option(0.7, help="Sampling temperature (0-1)"),
):
    """Execute a single prompt"""
    async def _execute():
        orchestrator = Orchestrator()

        task = Task(
            prompt=prompt,
            task_type=TaskType(task_type),
            max_tokens=max_tokens,
            temperature=temperature
        )

        console.print(f"[bold blue]Executing task...[/bold blue]")
        result = await orchestrator.execute(task)

        if result.success:
            console.print(Panel(result.content, title=f"Response from {result.model}"))
            console.print(f"\n[green]✓ Success![/green]")
            console.print(f"Model: {result.model}")
            console.print(f"Cost: ${result.cost:.4f}")
            console.print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
            console.print(f"Latency: {result.latency:.2f}s")
        else:
            console.print(f"[red]✗ Error: {result.error}[/red]")

    asyncio.run(_execute())


@app.command()
def models():
    """List available models"""
    orchestrator = Orchestrator()
    available = orchestrator.list_available_models()

    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Max Tokens", style="yellow")
    table.add_column("Strengths", style="magenta")

    for model_id in available:
        info = orchestrator.get_model_info(model_id)
        if info:
            strengths = ", ".join(s.value for s in info.strengths[:3])
            table.add_row(
                model_id,
                info.provider,
                str(info.max_tokens),
                strengths
            )

    console.print(table)


@app.command()
def cost():
    """Show cost report"""
    orchestrator = Orchestrator()
    report = orchestrator.get_cost_report()
    savings = orchestrator.get_savings_report()

    console.print(Panel.fit(
        f"""[bold]Cost Report[/bold]

Total Cost: ${report.total_cost:.2f}
Total Requests: {report.total_requests}
Average per Request: ${report.avg_cost_per_request:.4f}

[bold green]Savings vs All-GPT-4:[/bold green]
Savings: ${savings['savings']:.2f} ({savings['savings_percent']:.1f}%)
""",
        title="Cost Analysis"
    ))

    # Cost by model
    if report.cost_by_model:
        table = Table(title="Cost by Model")
        table.add_column("Model", style="cyan")
        table.add_column("Cost", style="green")

        for model, cost in sorted(report.cost_by_model.items(), key=lambda x: x[1], reverse=True):
            table.add_row(model, f"${cost:.2f}")

        console.print(table)


@app.command()
def benchmark(
    prompt: str = typer.Option("Explain quantum physics in simple terms", help="Test prompt"),
    iterations: int = typer.Option(5, help="Number of iterations per model"),
):
    """Benchmark different models"""
    async def _benchmark():
        orchestrator = Orchestrator()
        available = orchestrator.list_available_models()

        console.print(f"[bold]Benchmarking {len(available)} models with {iterations} iterations each...[/bold]\n")

        results = {}
        for model_id in available:
            console.print(f"Testing {model_id}...")
            model_results = []

            for i in range(iterations):
                task = Task(prompt=prompt, task_type=TaskType.SIMPLE)
                task.fallback_chain = [model_id]  # Force specific model

                result = await orchestrator.execute(task)
                if result.success:
                    model_results.append({
                        "cost": result.cost,
                        "latency": result.latency,
                        "tokens": result.total_tokens
                    })

            if model_results:
                results[model_id] = {
                    "avg_cost": sum(r["cost"] for r in model_results) / len(model_results),
                    "avg_latency": sum(r["latency"] for r in model_results) / len(model_results),
                    "avg_tokens": sum(r["tokens"] for r in model_results) / len(model_results),
                }

        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Model", style="cyan")
        table.add_column("Avg Cost", style="green")
        table.add_column("Avg Latency", style="yellow")
        table.add_column("Avg Tokens", style="magenta")

        for model_id, stats in sorted(results.items(), key=lambda x: x[1]["avg_cost"]):
            table.add_row(
                model_id,
                f"${stats['avg_cost']:.4f}",
                f"{stats['avg_latency']:.2f}s",
                f"{int(stats['avg_tokens'])}"
            )

        console.print(table)

    asyncio.run(_benchmark())


if __name__ == "__main__":
    app()
