"""
Basic usage examples for ORCHEX Orchestrator
"""

import asyncio
from atlas_orchestrator import Orchestrator, Task, TaskType


async def example_1_simple_task():
    """Example 1: Execute a simple task"""
    print("=" * 50)
    print("Example 1: Simple Task")
    print("=" * 50)

    orchestrator = Orchestrator()

    task = Task(
        prompt="Explain quantum entanglement in simple terms",
        task_type=TaskType.EXPLANATION,
        max_tokens=500
    )

    result = await orchestrator.execute(task)

    print(f"\nResult from {result.model}:")
    print(result.content)
    print(f"\nCost: ${result.cost:.4f}")
    print(f"Latency: {result.latency:.2f}s")


async def example_2_different_task_types():
    """Example 2: Different task types get routed to different models"""
    print("\n" + "=" * 50)
    print("Example 2: Task Type Routing")
    print("=" * 50)

    orchestrator = Orchestrator()

    tasks = [
        Task(prompt="Write a Python function to calculate fibonacci", task_type=TaskType.CODE),
        Task(prompt="Summarize recent advances in quantum computing", task_type=TaskType.RESEARCH),
        Task(prompt="Analyze the pros and cons of remote work", task_type=TaskType.ANALYSIS),
    ]

    results = await orchestrator.execute_batch(tasks)

    for i, result in enumerate(results):
        print(f"\nTask {i+1} ({tasks[i].task_type.value}) -> {result.model}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"Preview: {result.content[:100]}...")


async def example_3_parallel_execution():
    """Example 3: Execute tasks in parallel"""
    print("\n" + "=" * 50)
    print("Example 3: Parallel Execution")
    print("=" * 50)

    orchestrator = Orchestrator()

    tasks = [
        Task(prompt=f"Tell me a fact about the number {i}", task_type=TaskType.SIMPLE)
        for i in range(1, 6)
    ]

    import time
    start = time.time()
    results = await orchestrator.execute_parallel(tasks)
    elapsed = time.time() - start

    print(f"\nExecuted {len(tasks)} tasks in {elapsed:.2f}s (parallel)")
    print(f"Average time per task: {elapsed / len(tasks):.2f}s")

    for i, result in enumerate(results):
        print(f"\nTask {i+1}: {result.content[:80]}...")


async def example_4_cost_tracking():
    """Example 4: Cost tracking and reporting"""
    print("\n" + "=" * 50)
    print("Example 4: Cost Tracking")
    print("=" * 50)

    orchestrator = Orchestrator()

    # Execute some tasks
    for i in range(10):
        task = Task(
            prompt=f"Quick task {i}",
            task_type=TaskType.SIMPLE
        )
        await orchestrator.execute(task)

    # Get cost report
    report = orchestrator.get_cost_report()
    savings = orchestrator.get_savings_report()

    print(f"\nCost Report:")
    print(f"Total cost: ${report.total_cost:.4f}")
    print(f"Total requests: {report.total_requests}")
    print(f"Average per request: ${report.avg_cost_per_request:.4f}")

    print(f"\nCost by model:")
    for model, cost in report.cost_by_model.items():
        print(f"  {model}: ${cost:.4f}")

    print(f"\nSavings vs all-GPT-4:")
    print(f"  Savings: ${savings['savings']:.4f} ({savings['savings_percent']:.1f}%)")


async def example_5_budget_control():
    """Example 5: Budget control"""
    print("\n" + "=" * 50)
    print("Example 5: Budget Control")
    print("=" * 50)

    orchestrator = Orchestrator()

    # Set a low budget
    orchestrator.set_budget(max_daily_cost=0.10, max_per_request=0.01)

    print("Budget set: $0.10 daily, $0.01 per request")

    # This should work
    task1 = Task(prompt="Short task", task_type=TaskType.SIMPLE, max_tokens=50)
    result1 = await orchestrator.execute(task1)
    print(f"\nTask 1: {'✓ Success' if result1.success else '✗ Failed'}")

    # This might hit budget limit
    task2 = Task(prompt="Longer task", task_type=TaskType.RESEARCH, max_tokens=2000)
    result2 = await orchestrator.execute(task2)
    print(f"Task 2: {'✓ Success' if result2.success else '✗ Failed (budget)'}")

    print(f"\nCurrent spending: ${orchestrator.get_today_cost():.4f}")


async def main():
    """Run all examples"""
    await example_1_simple_task()
    await example_2_different_task_types()
    await example_3_parallel_execution()
    await example_4_cost_tracking()
    await example_5_budget_control()


if __name__ == "__main__":
    asyncio.run(main())
