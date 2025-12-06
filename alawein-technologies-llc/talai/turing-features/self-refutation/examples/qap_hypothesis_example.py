"""
QAP Hypothesis Testing Example

Tests a hypothesis about QAP (Quadratic Assignment Problem) solving.
"""

import asyncio
from self_refutation import SelfRefutationProtocol, Hypothesis, HypothesisDomain


async def test_hypothesis(hypothesis: Hypothesis, protocol: SelfRefutationProtocol):
    """Test a single hypothesis and display results"""
    print(f"\n{'=' * 80}")
    print(f"Testing: {hypothesis.claim}")
    print('=' * 80)

    result = await protocol.refute(hypothesis)

    score_emoji = "🟢" if result.strength_score >= 61 else "🟡" if result.strength_score >= 41 else "🔴"
    print(f"\n{score_emoji} STRENGTH SCORE: {result.strength_score:.1f}/100")
    print(f"   Status: {'REFUTED ❌' if result.refuted else 'SURVIVED ✓'}")
    print(f"   Passed: {result.strategies_passed}/{result.total_strategies} strategies")

    # Show any failures
    failures = [sr for sr in result.strategy_results if not sr.passed]
    if failures:
        print(f"\n   Failed strategies:")
        for sr in failures:
            print(f"     • {sr.strategy.value}: {sr.reasoning}")

    print(f"\n   → {result.interpretation}")

    return result


async def main():
    """Run QAP hypothesis examples"""
    print("=" * 80)
    print("QAP HYPOTHESIS TESTING")
    print("Testing 4 hypotheses with different strength levels")
    print("=" * 80)

    protocol = SelfRefutationProtocol()

    # Hypothesis 1: STRONG - Specific, bounded claim
    h1 = Hypothesis(
        claim="Our modified simulated annealing achieves average 8% improvement over baseline on QAPLIB tai instances",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 20 tai instances (tai12a-tai50a), 10 runs each, statistical significance p<0.05",
        assumptions=[
            "Baseline is standard simulated annealing with exponential cooling",
            "Improvement measured by relative gap to best-known solutions",
            "Same computational budget for both methods"
        ],
        predictions=[
            "Improvement will hold on tai instances not yet tested",
            "Improvement will degrade on non-tai QAPLIB instances",
            "Performance depends on cooling schedule tuning"
        ]
    )

    # Hypothesis 2: MODERATE - Less specific
    h2 = Hypothesis(
        claim="Our new QAP solver improves performance by 15% compared to existing methods",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on 30 QAPLIB instances",
        assumptions=[
            "Solver uses genetic algorithm approach"
        ]
    )

    # Hypothesis 3: WEAK - Vague claim with absolute terms
    h3 = Hypothesis(
        claim="Our algorithm always finds better solutions for QAP problems",
        domain=HypothesisDomain.OPTIMIZATION,
        context="Tested on some QAPLIB instances"
    )

    # Hypothesis 4: VERY WEAK - Impossible claim
    h4 = Hypothesis(
        claim="Our polynomial-time algorithm always finds the global optimum for QAP",
        domain=HypothesisDomain.OPTIMIZATION,
        context="QAP is NP-hard, algorithm runs in O(n^3) time"
    )

    # Test all hypotheses
    hypotheses = [
        ("Strong Hypothesis", h1),
        ("Moderate Hypothesis", h2),
        ("Weak Hypothesis", h3),
        ("Very Weak Hypothesis", h4),
    ]

    results = []
    for name, hyp in hypotheses:
        print(f"\n\n{'#' * 80}")
        print(f"# {name}")
        print('#' * 80)
        result = await test_hypothesis(hyp, protocol)
        results.append((name, result))

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    print(f"\n{'Name':<25} {'Score':<10} {'Status':<15} {'Interpretation'}")
    print('-' * 80)

    for name, result in results:
        status = 'REFUTED' if result.refuted else 'SURVIVED'
        print(f"{name:<25} {result.strength_score:>5.1f}/100  {status:<15} {result.interpretation}")

    print('\n' + '=' * 80)
    print("KEY INSIGHTS:")
    print('=' * 80)
    print("""
1. Strong hypotheses are SPECIFIC with bounded claims
   → "8% improvement on tai instances" (not "always better")

2. State your ASSUMPTIONS explicitly
   → What baseline? What budget? What metric?

3. Include BOUNDARY CONDITIONS
   → Where does it work? Where might it fail?

4. Avoid ABSOLUTE TERMS
   → "always", "all", "never" are easily falsifiable

5. Make it TESTABLE
   → Clear predictions that can be verified or refuted
    """)


if __name__ == "__main__":
    asyncio.run(main())
