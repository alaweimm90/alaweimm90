"""
Basic Hall of Failures Example
"""

import asyncio
from hall_of_failures import HallOfFailures, Failure, FailureType


async def main():
    """Demonstrate Hall of Failures"""

    print("=" * 80)
    print("HALL OF FAILURES - Learning from Failures")
    print("=" * 80)

    # Initialize Hall of Failures
    hall = HallOfFailures(db_path="example_failures.db")

    # Get current stats
    stats = hall.get_stats()
    print(f"\nCurrent database: {stats['total_failures']} failures recorded")

    # Example 1: Record a hypothesis failure
    print("\n" + "-" * 80)
    print("Example 1: Recording a hypothesis failure")
    print("-" * 80)

    failure1 = Failure(
        hypothesis="Our algorithm always finds the global optimum",
        failure_type=FailureType.HYPOTHESIS,
        description="Hypothesis is unfalsifiable - no way to prove it 'always' works. Found counter-examples on tai50a.",
        context={
            "domain": "optimization",
            "problem": "QAP",
        }
    )

    analysis1 = await hall.record_failure(failure1)

    print(f"\n✓ Failure recorded (ID: {analysis1.failure.id})")
    print(f"  Type: {analysis1.failure.failure_type.value}")
    print(f"  Severity: {analysis1.failure.severity.value}")

    print(f"\n📚 Lessons Learned:")
    for lesson in analysis1.failure.lessons_learned:
        print(f"  • {lesson}")

    print(f"\n🛡️ Prevention Strategies:")
    for strategy in analysis1.failure.prevention_strategies:
        print(f"  • {strategy}")

    if analysis1.similar_failures:
        print(f"\n⚠️ Similar Past Failures: {len(analysis1.similar_failures)}")

    # Example 2: Record an experimental failure
    print("\n" + "-" * 80)
    print("Example 2: Recording an experimental failure")
    print("-" * 80)

    failure2 = Failure(
        hypothesis="Method X improves performance by 30%",
        failure_type=FailureType.EXPERIMENTAL,
        description="Study was underpowered - sample size of 20 insufficient for claimed effect size",
        context={
            "sample_size": 20,
            "effect_size": 0.30,
            "power": 0.35,
        }
    )

    analysis2 = await hall.record_failure(failure2)

    print(f"\n✓ Failure recorded")
    print(f"\n📚 Lessons Learned:")
    for lesson in analysis2.failure.lessons_learned:
        print(f"  • {lesson}")

    # Example 3: Risk Assessment
    print("\n" + "-" * 80)
    print("Example 3: Risk assessment for new hypothesis")
    print("-" * 80)

    new_hypothesis = "Our optimizer guarantees finding optimal solution on all instances"

    risk = await hall.assess_risk(new_hypothesis)

    print(f"\nHypothesis: {new_hypothesis}")
    print(f"\n🎯 Risk Assessment:")
    print(f"  Risk Level: {risk.risk_level}")
    print(f"  Risk Score: {risk.risk_score:.2f}")

    if risk.warnings:
        print(f"\n⚠️ Warnings:")
        for warning in risk.warnings:
            print(f"  • {warning}")

    if risk.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in risk.recommendations[:3]:
            print(f"  • {rec}")

    # Example 4: Query failures
    print("\n" + "-" * 80)
    print("Example 4: Query all hypothesis failures")
    print("-" * 80)

    hypothesis_failures = hall.query(failure_type=FailureType.HYPOTHESIS)

    print(f"\nFound {len(hypothesis_failures)} hypothesis failures:")
    for f in hypothesis_failures[:3]:
        print(f"  • {f.hypothesis[:60]}...")

    # Example 5: Pattern Analysis
    print("\n" + "-" * 80)
    print("Example 5: Analyze failure patterns")
    print("-" * 80)

    all_failures = hall.query(limit=100)
    patterns = hall.analyze_patterns(all_failures)

    print(f"\nIdentified {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"\n  Pattern: {pattern.pattern_name}")
        print(f"  Failures: {pattern.failure_count}")
        print(f"  Common causes: {', '.join(pattern.common_root_causes[:2])}")

    # Final stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    final_stats = hall.get_stats()
    print(f"\nTotal failures recorded: {final_stats['total_failures']}")
    print(f"By type: {final_stats['by_type']}")
    print(f"By severity: {final_stats['by_severity']}")

    print("\n✅ Hall of Failures demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
