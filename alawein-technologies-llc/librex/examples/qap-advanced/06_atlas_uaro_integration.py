"""
Example 6: ORCHEX-UARO Integration

Demonstrates:
- Converting research tasks to UARO problems
- Solving research tasks with universal primitives
- Extracting workflows as reusable primitives
- Publishing to marketplace
- Generating research reports with proofs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro.atlas_integration import (
    ATLASUAROIntegration,
    ResearchTask,
    create_atlas_uaro_integration
)


def success_criteria_literature_review(state):
    """Evaluate literature review quality"""
    score = 0.0

    # Check for hypotheses
    if len(state['hypotheses']) >= 2:
        score += 0.3

    # Check for evidence
    if len(state['evidence']) >= 3:
        score += 0.3

    # Check phase completion
    if state['phase'] == 'complete':
        score += 0.2

    # Check confidence
    score += state['confidence'] * 0.2

    return min(1.0, score)


def success_criteria_hypothesis_test(state):
    """Evaluate hypothesis testing quality"""
    score = 0.0

    # Check for experiments
    if len(state['experiments']) >= 1:
        score += 0.4

    # Check for results
    if state['results']:
        score += 0.3

    # Check validation
    if state['confidence'] > 0.7:
        score += 0.3

    return min(1.0, score)


def main():
    """Demonstrate ORCHEX-UARO integration"""

    print("=" * 70)
    print("UARO Example 6: ORCHEX-UARO Integration")
    print("=" * 70)
    print()

    # Create integration
    integration = create_atlas_uaro_integration()

    # Example 1: Literature Review Task
    print("Task 1: Literature Review")
    print("-" * 70)

    task1 = ResearchTask(
        task_id="lit_review_001",
        task_type="literature_review",
        query="What are the latest advances in transformer architectures?",
        context={
            'domain': 'machine_learning',
            'time_range': '2023-2024',
            'focus_areas': ['attention_mechanisms', 'efficiency', 'scaling']
        },
        success_criteria=success_criteria_literature_review,
        metadata={'priority': 'high'}
    )

    print(f"Query: {task1.query}")
    print(f"Type: {task1.task_type}")
    print(f"Context: {task1.context}")
    print()

    print("Solving with UARO...")
    result1 = integration.solve_research_task(task1, max_iterations=10)

    print()
    print("Results:")
    print(f"  Success: {result1['success']}")
    print(f"  Confidence: {result1['confidence']:.2%}")
    print(f"  Iterations: {result1['solution'].iterations}")
    print(f"  Duration: {result1['solution'].duration_seconds:.3f}s")

    if result1['extracted_primitive']:
        print(f"  Extracted Primitive: {result1['extracted_primitive'].name}")

    print()

    # Example 2: Hypothesis Testing Task
    print("Task 2: Hypothesis Testing")
    print("-" * 70)

    task2 = ResearchTask(
        task_id="hyp_test_001",
        task_type="hypothesis_test",
        query="Does increasing model size improve performance on reasoning tasks?",
        context={
            'domain': 'machine_learning',
            'hypothesis': 'Larger models perform better on complex reasoning',
            'metrics': ['accuracy', 'inference_time', 'parameters']
        },
        success_criteria=success_criteria_hypothesis_test,
        metadata={'priority': 'medium'}
    )

    print(f"Query: {task2.query}")
    print(f"Type: {task2.task_type}")
    print(f"Hypothesis: {task2.context['hypothesis']}")
    print()

    print("Solving with UARO...")
    result2 = integration.solve_research_task(task2, max_iterations=10)

    print()
    print("Results:")
    print(f"  Success: {result2['success']}")
    print(f"  Confidence: {result2['confidence']:.2%}")
    print(f"  Iterations: {result2['solution'].iterations}")
    print(f"  Duration: {result2['solution'].duration_seconds:.3f}s")

    if result2['extracted_primitive']:
        print(f"  Extracted Primitive: {result2['extracted_primitive'].name}")

    print()

    # Show reasoning trace for first task
    print("Reasoning Trace (Task 1, first 5 steps):")
    print("-" * 70)
    for step in result1['reasoning_trace'][:5]:
        print(f"Step {step.iteration}: {step.primitive_name}")
        print(f"  Success: {step.success}")
        print(f"  Confidence: {step.confidence:.2%}")
        print(f"  Reasoning: {step.reasoning}")
        print()

    if len(result1['reasoning_trace']) > 5:
        print(f"... ({len(result1['reasoning_trace']) - 5} more steps)")
        print()

    # Export research reports
    print("Exporting Research Reports...")
    print("-" * 70)

    proofs_dir = Path(__file__).parent / "proofs"
    proofs_dir.mkdir(exist_ok=True)

    # Export as markdown
    report1_path = proofs_dir / "atlas_literature_review.md"
    integration.export_research_report(result1, report1_path, format="markdown")
    print(f"Report 1: {report1_path}")

    report2_path = proofs_dir / "atlas_hypothesis_test.md"
    integration.export_research_report(result2, report2_path, format="markdown")
    print(f"Report 2: {report2_path}")

    # Export as HTML
    html_path = proofs_dir / "atlas_literature_review.html"
    integration.export_research_report(result1, html_path, format="html")
    print(f"Report 1 (HTML): {html_path}")

    print()

    # Publish extracted primitive (if any)
    if result1['extracted_primitive']:
        print("Publishing Extracted Workflow to Marketplace...")
        print("-" * 70)

        listing_id = integration.publish_learned_workflow(
            primitive=result1['extracted_primitive'],
            author="ORCHEX Research System",
            description="Automated literature review workflow extracted from successful research task",
            tags=['literature_review', 'research', 'ORCHEX', 'machine_learning']
        )

        print(f"Published listing: {listing_id}")
        print(f"Primitive: {result1['extracted_primitive'].name}")
        print()

    # Discover research primitives
    print("Discovering Research Primitives from Marketplace...")
    print("-" * 70)

    listings = integration.discover_research_primitives(task_type="literature_review")

    print(f"Found {len(listings)} relevant primitives")
    for i, listing in enumerate(listings[:3], 1):
        print(f"\n{i}. {listing.primitive.name}")
        print(f"   Author: {listing.author}")
        print(f"   Rating: {listing.rating:.2f}/5.0")
        print(f"   Uses: {listing.usage_count}")
        print(f"   Success Rate: {listing.success_rate():.1%}")

    print()

    # Show statistics
    print("Integration Statistics:")
    print("-" * 70)

    stats = integration.get_statistics()
    print(f"Total Tasks Solved: {stats['total_tasks_solved']}")
    print(f"Successful Tasks: {stats['successful_tasks']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Extracted Primitives: {stats['extracted_primitives']}")
    print(f"Extraction Rate: {stats['extraction_rate']:.1%}")

    print()

    # Key insights
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("1. Research Task Formulation:")
    print("   - ORCHEX tasks automatically converted to UARO problems")
    print("   - Universal primitives apply to research workflows")
    print("   - Success criteria guide solver toward quality solutions")
    print()
    print("2. Workflow Extraction:")
    print("   - Successful research workflows extracted as primitives")
    print("   - Primitives can be shared via marketplace")
    print("   - Other researchers benefit from proven workflows")
    print()
    print("3. Explainable Research:")
    print("   - Every research output includes proof document")
    print("   - Complete reasoning trace from query to conclusion")
    print("   - Multiple export formats (MD, HTML, LaTeX, JSON)")
    print()
    print("4. Network Effects:")
    print("   - More research → More primitives → Better solutions")
    print("   - Marketplace creates viral growth loop")
    print("   - Community benefits from collective intelligence")
    print()

    # Advanced: Compare primitive performance
    print("=" * 70)
    print("Advanced: Primitive Performance Comparison")
    print("=" * 70)
    print()

    # Show which primitives were most effective
    primitive_usage = {}
    for step in result1['reasoning_trace']:
        name = step.primitive_name
        if name not in primitive_usage:
            primitive_usage[name] = {'count': 0, 'success': 0}
        primitive_usage[name]['count'] += 1
        if step.success:
            primitive_usage[name]['success'] += 1

    print("Primitive Performance (Task 1):")
    for name, stats in sorted(primitive_usage.items(), key=lambda x: x[1]['success'], reverse=True):
        success_rate = stats['success'] / stats['count'] * 100 if stats['count'] > 0 else 0
        print(f"  {name}: {stats['success']}/{stats['count']} ({success_rate:.1f}%)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
