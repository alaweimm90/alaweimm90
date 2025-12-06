"""
Batch Hypothesis Testing Example

Demonstrates testing multiple hypotheses in parallel.
"""

import asyncio
from self_refutation import SelfRefutationProtocol, Hypothesis, HypothesisDomain


async def main():
    """Run batch testing example"""
    print("=" * 80)
    print("BATCH HYPOTHESIS TESTING")
    print("=" * 80)

    # Create multiple hypotheses across different domains
    hypotheses = [
        # Optimization
        Hypothesis(
            claim="Genetic algorithm with adaptive mutation rate converges 20% faster than standard GA",
            domain=HypothesisDomain.OPTIMIZATION,
            context="Tested on 15 benchmark functions"
        ),

        # Machine Learning
        Hypothesis(
            claim="Our neural architecture achieves 95% accuracy on ImageNet validation set",
            domain=HypothesisDomain.MACHINE_LEARNING,
            context="Using ResNet-50 backbone with custom attention mechanism",
            assumptions=["Training data includes ImageNet training set", "Standard augmentation applied"]
        ),

        # Computer Science
        Hypothesis(
            claim="New hash table implementation reduces lookup time by 40% in average case",
            domain=HypothesisDomain.COMPUTER_SCIENCE,
            context="Compared against standard chaining with load factor 0.75",
            assumptions=["Hash function has good distribution properties", "Keys are uniformly distributed"]
        ),

        # Mathematics
        Hypothesis(
            claim="Proposed algorithm computes matrix inverse in O(n^2.5) time for sparse matrices",
            domain=HypothesisDomain.MATHEMATICS,
            context="Sparsity pattern is block-diagonal",
            assumptions=["Matrix is invertible", "Sparsity ≥ 90%"]
        ),

        # General (weak hypothesis for comparison)
        Hypothesis(
            claim="Our method always improves results in all cases",
            domain=HypothesisDomain.GENERAL,
            context="Preliminary testing"
        ),
    ]

    # Initialize protocol
    protocol = SelfRefutationProtocol()

    # Run batch refutation (parallel execution)
    print(f"\nTesting {len(hypotheses)} hypotheses in parallel...\n")
    results = await protocol.refute_batch(hypotheses)

    # Display results
    print("-" * 80)
    print(f"{'#':<4} {'Domain':<20} {'Score':<10} {'Status':<12} {'Pass Rate'}")
    print("-" * 80)

    for i, (hyp, result) in enumerate(zip(hypotheses, results), 1):
        status_emoji = "✓" if not result.refuted else "✗"
        score_color = "🟢" if result.strength_score >= 61 else "🟡" if result.strength_score >= 41 else "🔴"

        print(f"{i:<4} {hyp.domain.value:<20} {score_color}{result.strength_score:>5.1f}/100  "
              f"{status_emoji} {('PASS' if not result.refuted else 'REFUTED'):<10} "
              f"{result.strategies_passed}/{result.total_strategies}")

    print("-" * 80)

    # Detailed analysis of strongest and weakest
    sorted_results = sorted(zip(hypotheses, results), key=lambda x: x[1].strength_score, reverse=True)

    print("\n" + "=" * 80)
    print("STRONGEST HYPOTHESIS")
    print("=" * 80)
    best_hyp, best_result = sorted_results[0]
    print(f"\nClaim: {best_hyp.claim}")
    print(f"Score: {best_result.strength_score:.1f}/100")
    print(f"Interpretation: {best_result.interpretation}")
    print(f"\nWhy it's strong:")
    for sr in best_result.strategy_results:
        if sr.passed:
            print(f"  ✓ {sr.strategy.value}: {sr.reasoning}")

    print("\n" + "=" * 80)
    print("WEAKEST HYPOTHESIS")
    print("=" * 80)
    worst_hyp, worst_result = sorted_results[-1]
    print(f"\nClaim: {worst_hyp.claim}")
    print(f"Score: {worst_result.strength_score:.1f}/100")
    print(f"Interpretation: {worst_result.interpretation}")
    print(f"\nIssues found:")
    for sr in worst_result.strategy_results:
        if not sr.passed:
            print(f"  ✗ {sr.strategy.value}: {sr.reasoning}")
            if sr.evidence:
                for evidence in sr.evidence[:1]:
                    print(f"     • {evidence}")

    # Statistics
    print("\n" + "=" * 80)
    print("BATCH STATISTICS")
    print("=" * 80)

    avg_score = sum(r.strength_score for _, r in sorted_results) / len(sorted_results)
    strong_count = sum(1 for _, r in sorted_results if r.strength_score >= 61)
    refuted_count = sum(1 for _, r in sorted_results if r.refuted)

    print(f"\n  Average Score: {avg_score:.1f}/100")
    print(f"  Strong Hypotheses (≥61): {strong_count}/{len(hypotheses)}")
    print(f"  Refuted: {refuted_count}/{len(hypotheses)}")
    print(f"  Survival Rate: {((len(hypotheses) - refuted_count) / len(hypotheses) * 100):.0f}%")

    # Recommendations for batch
    print("\n" + "=" * 80)
    print("BATCH RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
  Proceed to Experiment:
    {strong_count} hypotheses scored ≥61 and are ready for experimental validation.

  Need Revision:
    {len(hypotheses) - strong_count - refuted_count} hypotheses need minor to moderate revision.

  Reject/Major Revision:
    {refuted_count} hypotheses have critical flaws and need major revision or rejection.

  Next Steps:
    1. Focus resources on the {strong_count} strong hypotheses
    2. Revise moderate hypotheses addressing specific strategy failures
    3. Reformulate or discard critically flawed hypotheses
    4. This filtering saved you from {refuted_count} potentially failed experiments!
    """)


if __name__ == "__main__":
    asyncio.run(main())
