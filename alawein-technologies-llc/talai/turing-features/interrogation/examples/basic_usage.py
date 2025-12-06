"""
Basic 200-Question Interrogation Example

Demonstrates how to interrogate a hypothesis.
"""

import asyncio
from interrogation import InterrogationProtocol
from self_refutation import Hypothesis, HypothesisDomain


async def main():
    """Run basic interrogation example"""

    # Create hypothesis
    hypothesis = Hypothesis(
        claim="Our new QAP solver achieves 15% improvement over state-of-the-art on QAPLIB instances",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 20 tai instances with 10 runs each",
        evidence=[
            "Average gap to best-known reduced by 15.3%",
            "Statistical significance p < 0.01",
        ],
        assumptions=[
            "Same computational budget for all methods",
            "Standard benchmark instances",
            "Best-known solutions as baseline",
        ],
        predictions=[
            "Improvement will hold on untested tai instances",
            "May not generalize to non-tai instances",
        ]
    )

    print("=" * 80)
    print("200-QUESTION INTERROGATION FRAMEWORK")
    print("=" * 80)
    print(f"\nHypothesis: {hypothesis.claim}")
    print(f"Domain: {hypothesis.domain.value}\n")

    # Initialize protocol
    # For quick demo, limit questions
    protocol = InterrogationProtocol(
        questions_per_category=5  # Ask 5 questions per category (50 total)
    )

    # Get database info
    db_info = protocol.get_database_info()
    print(f"Using database: {db_info['total_questions']} questions in {db_info['categories']} categories\n")

    print("Running interrogation (this may take a moment without AI Orchestrator)...\n")

    # Run interrogation
    result = await protocol.interrogate(hypothesis)

    # Display results
    print("=" * 80)
    print("INTERROGATION RESULTS")
    print("=" * 80)

    print(f"\n{result.status_emoji} OVERALL SCORE: {result.overall_score:.1f}/100")
    print(f"   {result.interpretation}\n")

    print(f"Questions Asked: {result.total_questions}")
    print(f"Execution Time: {result.execution_time_seconds:.1f}s\n")

    # Category breakdown
    print("-" * 80)
    print("CATEGORY BREAKDOWN")
    print("-" * 80)

    for cr in result.category_results:
        status_emoji = "✓" if cr.raw_score >= 60 else "⚠" if cr.raw_score >= 40 else "✗"
        print(f"\n  {status_emoji} {cr.category_name}")
        print(f"     Score: {cr.raw_score:.1f}/100 (weight: {cr.category_weight}x)")
        print(f"     Questions: {cr.questions_asked}")

        if cr.strengths:
            print(f"     Strengths: {cr.strengths[0]}")

        if cr.weaknesses:
            print(f"     Weaknesses: {cr.weaknesses[0]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if result.strong_categories:
        print(f"\n✓ Strong Categories ({len(result.strong_categories)}):")
        for cat in result.strong_categories:
            print(f"   • {cat}")

    if result.weak_categories:
        print(f"\n⚠ Weak Categories ({len(result.weak_categories)}):")
        for cat in result.weak_categories:
            print(f"   • {cat}")

    if result.critical_categories:
        print(f"\n✗ Critical Categories ({len(result.critical_categories)}):")
        for cat in result.critical_categories:
            print(f"   • {cat}")

    # Failure points
    if result.failure_points:
        print(f"\n✗ FAILURE POINTS:")
        for fp in result.failure_points[:5]:
            print(f"   • {fp}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result.recommendations[:5]:
        print(f"   {rec}")

    print("\n" + "=" * 80)
    print("INTERROGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
