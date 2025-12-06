"""
Real-World Integration Example #2: Climate Model Validation

Demonstrates using TalAI to validate climate science hypotheses before
publishing research or informing policy decisions.

Use Case:
- Climate researchers have a new model predicting regional temperature changes
- Model predictions will inform billion-dollar policy decisions
- Need rigorous validation to ensure model reliability

Expected Outcome:
- Identify model weaknesses before publication
- Strengthen confidence in reliable predictions
- Prevent policy decisions based on flawed models
"""

import asyncio
from typing import Dict, List

from self_refutation import Hypothesis, HypothesisDomain
from turing_challenge_system import TuringChallengeSystem, ValidationMode


async def main():
    """Run climate model validation pipeline."""

    print("=" * 80)
    print("CLIMATE MODEL VALIDATION - Turing Challenge System")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Define Climate Hypothesis
    # ========================================================================

    hypothesis = Hypothesis(
        claim="""
        A new ensemble climate model predicts that the Pacific Northwest region
        will experience a 2.5°C ± 0.5°C increase in average temperature by 2050,
        with 95% confidence, leading to 15% reduction in snowpack and increased
        wildfire risk.
        """,
        domain=HypothesisDomain.CLIMATE_SCIENCE,
        evidence=[
            "Ensemble of 47 climate models shows consistent warming trend",
            "Historical data (1950-2023) validates model hindcasts with R²=0.89",
            "Sea surface temperature projections align with IPCC AR6 scenarios",
            "Model captures El Niño/La Niña patterns with 82% accuracy",
            "Downscaling from global to regional scale using RCM techniques",
            "Validation against 30 weather stations across region",
            "Peer review by 3 climate scientists",
        ],
        assumptions=[
            "GHG emissions follow RCP 4.5 scenario (moderate mitigation)",
            "Ocean circulation patterns remain stable",
            "No major volcanic eruptions that would cool climate",
            "Land use changes follow projected urbanization trends",
            "Model parameterizations accurately capture cloud physics",
            "Downscaling preserves statistical properties at regional scale",
            "Extreme events scale predictably with mean temperature",
        ],
        supporting_data={
            "model_count": 47,
            "r_squared": 0.89,
            "confidence_level": 0.95,
            "temperature_increase": 2.5,
            "uncertainty": 0.5,
            "snowpack_reduction": 0.15,
            "historical_period": "1950-2023",
            "stations": 30,
            "policy_impact_estimate": 5_000_000_000,  # $5B in policy decisions
        },
    )

    print(f"🌍 Climate Hypothesis:")
    print(f"   {hypothesis.claim[:120]}...")
    print(f"   Domain: {hypothesis.domain}")
    print(f"   Evidence: {len(hypothesis.evidence)} data points")
    print(f"   Assumptions: {len(hypothesis.assumptions)}")
    print()

    # ========================================================================
    # Step 2: Run Comprehensive Turing Challenge
    # ========================================================================

    print("🔬 Running COMPREHENSIVE Turing Challenge Validation...")
    print("   (Using highest rigor mode for policy-critical research)")
    print()
    print("   This will run ALL 8 validation protocols:")
    print("   1. Self-Refutation Protocol")
    print("   2. 200-Question Interrogation")
    print("   3. Hall of Failures Database Check")
    print("   4. Meta-Learning from Past Errors")
    print("   5. Devil's Advocate Adversarial Attack")
    print("   6. Agent Tournament Comparison")
    print("   7. Swarm Intelligence Democratic Vote")
    print("   8. Emergent Behavior Monitoring")
    print()

    system = TuringChallengeSystem()

    result = await system.validate_hypothesis_complete(
        hypothesis=hypothesis, mode=ValidationMode.COMPREHENSIVE
    )

    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 3: Detailed Analysis
    # ========================================================================

    print(f"📊 Overall Score: {result.overall_score:.1f}/100")
    print(f"📋 Recommendation: {result.recommendation}")
    print(f"🎯 Verdict: {result.overall_verdict}")
    print()

    # Protocol-by-Protocol Breakdown
    print("🔍 PROTOCOL-BY-PROTOCOL BREAKDOWN:")
    print()

    # 1. Self-Refutation
    print("1️⃣  Self-Refutation Protocol:")
    if result.self_refutation_result:
        sr = result.self_refutation_result
        print(f"    Self-refuting: {sr.self_refuting}")
        if not sr.self_refuting:
            print(f"    ✅ No internal contradictions found")
        else:
            print(f"    ⚠️  Contradictions: {len(sr.contradictions)}")
            for c in sr.contradictions[:2]:
                print(f"       - {c.description}")
    print()

    # 2. Interrogation
    print("2️⃣  200-Question Interrogation:")
    if result.interrogation_result:
        ir = result.interrogation_result
        print(f"    Questions answered: {ir.total_questions_answered}/200")
        print(f"    Overall confidence: {ir.overall_confidence:.1f}%")
        print(f"    Weak areas: {len([d for d in ir.dimensions if d.score < 60])}")
        if ir.unanswered_count > 0:
            print(f"    ⚠️  Unanswered: {ir.unanswered_count} questions")
    print()

    # 3. Hall of Failures
    print("3️⃣  Hall of Failures Check:")
    if result.hall_of_failures_result:
        hof = result.hall_of_failures_result
        print(f"    Similar past failures: {hof.similar_failures_found}")
        if hof.similar_failures_found > 0:
            print(f"    ⚠️  WARNING: {hof.similar_failures_found} similar hypotheses have failed")
            print(f"    Risk level: {hof.risk_level}")
    print()

    # 4. Meta-Learning
    print("4️⃣  Meta-Learning Core:")
    if result.meta_learning_result:
        ml = result.meta_learning_result
        print(f"    Historical accuracy: {ml.historical_accuracy:.1f}%")
        print(f"    Improvement trajectory: {ml.trajectory}")
    print()

    # 5. Devil's Advocate
    print("5️⃣  Devil's Advocate Attack:")
    if result.devils_advocate_result:
        da = result.devils_advocate_result
        print(f"    Total flaws: {da.total_flaws_found}")
        print(f"    Critical: {len(da.critical_flaws)}")
        print(f"    High: {len(da.high_flaws)}")
        print(f"    Medium: {len(da.medium_flaws)}")
        print(f"    Robustness: {da.robustness_score:.1f}/100")

        if da.critical_flaws:
            print(f"\n    🚨 CRITICAL FLAWS:")
            for flaw in da.critical_flaws:
                print(f"       - {flaw.description}")
                print(f"         Mitigation: {flaw.mitigation[:60]}...")
    print()

    # 6. Agent Tournaments
    print("6️⃣  Agent Tournament Results:")
    if result.tournament_result:
        tr = result.tournament_result
        print(f"    Champion: {tr.champion.agent_name}")
        print(f"    Solution quality: {tr.champion.best_solution_quality:.1f}/100")
        print(f"    Competitive pressure effect: {tr.competitive_pressure_effect:.1f}%")
    print()

    # 7. Swarm Voting
    print("7️⃣  Swarm Intelligence Vote:")
    if result.swarm_voting_result:
        sv = result.swarm_voting_result
        print(f"    Consensus: {sv.consensus_level}")
        print(f"    Decision: {sv.winning_option}")
        print(f"    Agreement: {sv.vote_percentage:.1f}%")
        print(f"    Reliability: {sv.reliability:.1f}/100")
        if sv.groupthink_detected:
            print(f"    ⚠️  Groupthink detected")
    print()

    # 8. Emergent Behavior
    print("8️⃣  Emergent Behavior Monitoring:")
    if result.emergent_behavior_result:
        eb = result.emergent_behavior_result
        print(f"    System health: {eb.health_score:.1f}/100")
        print(f"    Beneficial patterns: {len(eb.beneficial_patterns)}")
        print(f"    Harmful patterns: {len(eb.harmful_patterns)}")
        print(f"    Emergence rate: {eb.emergence_rate:.1f} patterns/min")
    print()

    # ========================================================================
    # Step 4: Strengths & Weaknesses
    # ========================================================================

    print("=" * 80)
    print("STRENGTHS & WEAKNESSES")
    print("=" * 80)
    print()

    print("💪 STRENGTHS:")
    for i, strength in enumerate(result.strengths[:5], 1):
        print(f"   {i}. {strength}")
    print()

    print("⚠️  WEAKNESSES:")
    for i, weakness in enumerate(result.weaknesses[:5], 1):
        print(f"   {i}. {weakness}")
    print()

    if result.critical_issues:
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(result.critical_issues, 1):
            print(f"   {i}. {issue}")
        print()

    # ========================================================================
    # Step 5: Policy Implications
    # ========================================================================

    print("=" * 80)
    print("POLICY IMPLICATIONS")
    print("=" * 80)
    print()

    policy_budget = hypothesis.supporting_data["policy_impact_estimate"]

    if result.recommendation == "PROCEED":
        print("✅ RECOMMENDATION: PUBLISH AND USE FOR POLICY DECISIONS")
        print()
        print(f"   Policy Budget at Risk: ${policy_budget / 1e9:.1f}B")
        print(f"   Confidence in Model: HIGH")
        print(f"   Scientific Rigor: EXCELLENT")
        print()
        print("   Recommended Policy Actions:")
        print("   1. Invest in wildfire prevention infrastructure ($2B)")
        print("   2. Update building codes for climate resilience ($500M)")
        print("   3. Expand water storage capacity ($1.5B)")
        print("   4. Forest management programs ($1B)")
        print()
        print("   Publication Strategy:")
        print("   - Submit to Nature Climate Change or Science")
        print("   - Prepare press release emphasizing model validation")
        print("   - Brief policymakers with confidence intervals")

    elif result.recommendation == "REVISE":
        print("⚠️  RECOMMENDATION: ADDRESS ISSUES BEFORE PUBLICATION")
        print()
        print(f"   Policy Budget at Risk: ${policy_budget / 1e9:.1f}B")
        print(f"   Confidence: MODERATE")
        print(f"   Risk: MEDIUM")
        print()
        print("   Required Improvements:")
        print("   1. Address critical flaws identified by Devil's Advocate")
        print("   2. Expand validation dataset (more stations)")
        print("   3. Test model against independent climate datasets")
        print("   4. Conduct sensitivity analysis on key assumptions")
        print("   5. Re-run Turing Challenge after improvements")
        print()
        print("   Timeline: 3-6 months for revisions")

    else:  # REJECT
        print("🛑 RECOMMENDATION: DO NOT USE FOR POLICY - MAJOR FLAWS")
        print()
        print(f"   Policy Budget Saved: ${policy_budget / 1e9:.1f}B")
        print(f"   (by avoiding decisions based on flawed model)")
        print()
        print("   Critical Problems:")
        print("   - Model has fundamental flaws that invalidate predictions")
        print("   - Using this model would lead to misguided policy")
        print("   - Return to model development phase")
        print()
        print("   Next Steps:")
        print("   1. Conduct root cause analysis of model failures")
        print("   2. Revise model physics or parameterizations")
        print("   3. Expand ensemble with different model families")
        print("   4. Increase validation against paleo-climate data")

    print()
    print("=" * 80)

    # ========================================================================
    # Step 6: Uncertainty Quantification
    # ========================================================================

    print()
    print("📊 UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    print()

    temp_increase = hypothesis.supporting_data["temperature_increase"]
    uncertainty = hypothesis.supporting_data["uncertainty"]

    print(f"   Predicted Temperature Increase: {temp_increase}°C ± {uncertainty}°C")
    print(f"   Confidence Interval: [{temp_increase - uncertainty}°C, {temp_increase + uncertainty}°C]")
    print()

    if result.overall_score >= 70:
        print("   ✅ Turing Challenge validation INCREASES confidence in prediction")
        print(f"   Validated confidence: 95% → 98%")
    elif result.overall_score >= 50:
        print("   ⚠️  Turing Challenge identifies moderate uncertainty")
        print(f"   Adjusted confidence: 95% → 85%")
        print(f"   Widen uncertainty band to ±{uncertainty * 1.5:.1f}°C")
    else:
        print("   🚨 Turing Challenge identifies HIGH uncertainty")
        print(f"   Model predictions NOT RELIABLE for policy decisions")
        print(f"   Do not use current prediction ranges")

    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

    print()
    print("📌 KEY TAKEAWAYS:")
    print()
    print("1. Climate models inform billion-dollar policy decisions")
    print("2. TalAI provides independent validation of model reliability")
    print("3. Identifies weaknesses before publication/policy use")
    print("4. Quantifies confidence in predictions")
    print("5. Prevents costly mistakes from flawed models")
    print()
    print("🌍 Rigorous science for a changing climate!")
