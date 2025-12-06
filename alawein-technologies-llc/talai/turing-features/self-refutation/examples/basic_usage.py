"""
Basic Self-Refutation Protocol Example

Demonstrates basic hypothesis testing with the protocol.
"""

import asyncio
from self_refutation import SelfRefutationProtocol, Hypothesis, HypothesisDomain


async def main():
    """Run basic example"""

    # Create a hypothesis
    hypothesis = Hypothesis(
        claim="Our new optimization algorithm improves performance by 30% on all test cases",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 50 random instances",
        assumptions=[
            "Algorithm uses gradient descent",
            "Test instances are representative of real-world problems"
        ],
        predictions=[
            "Will maintain 30% improvement on new test cases",
            "Performance improvement scales linearly with problem size"
        ]
    )

    print("=" * 80)
    print("SELF-REFUTATION PROTOCOL - Basic Example")
    print("=" * 80)
    print(f"\nHypothesis: {hypothesis.claim}")
    print(f"Domain: {hypothesis.domain.value}")
    print(f"Context: {hypothesis.context}\n")

    # Initialize protocol (without AI Orchestrator for this example)
    protocol = SelfRefutationProtocol()

    # Run refutation
    print("Running refutation strategies...\n")
    result = await protocol.refute(hypothesis)

    # Display results
    print("-" * 80)
    print(f"STRENGTH SCORE: {result.strength_score:.1f}/100")
    print(f"STATUS: {'REFUTED' if result.refuted else 'SURVIVED'}")
    print(f"CONFIDENCE: {result.confidence.value.upper()}")
    print(f"STRATEGIES PASSED: {result.strategies_passed}/{result.total_strategies}")
    print("-" * 80)

    print("\nSTRATEGY RESULTS:")
    for sr in result.strategy_results:
        status = "✓ PASS" if sr.passed else "✗ FAIL"
        print(f"\n  {sr.strategy.value}:")
        print(f"    Status: {status}")
        print(f"    Confidence: {sr.confidence.value}")
        print(f"    Severity: {sr.severity:.2f}")
        print(f"    Reasoning: {sr.reasoning}")

        if sr.evidence:
            print(f"    Evidence:")
            for evidence in sr.evidence[:2]:
                print(f"      • {evidence}")

    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print(f"  {result.interpretation}")
    print("-" * 80)

    if result.recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"  • {rec}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
