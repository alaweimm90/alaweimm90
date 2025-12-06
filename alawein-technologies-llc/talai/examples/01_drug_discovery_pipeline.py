"""
Real-World Integration Example #1: Drug Discovery Pipeline

Demonstrates using TalAI Turing Challenge System to validate pharmaceutical
research hypotheses before expensive clinical trials.

Use Case:
- Pharmaceutical company has a hypothesis about a new drug compound
- Need to validate hypothesis rigorously before investing $100M+ in trials
- Use Turing Challenge to catch flaws early

Expected Outcome:
- Save millions by catching bad hypotheses early
- Accelerate discovery of promising compounds
- Reduce time to market for life-saving drugs
"""

import asyncio
from typing import Dict, List, Any

# TalAI imports
from self_refutation import Hypothesis, HypothesisDomain
from turing_challenge_system import TuringChallengeSystem, ValidationMode


async def main():
    """Run drug discovery validation pipeline."""

    print("=" * 80)
    print("DRUG DISCOVERY PIPELINE - Turing Challenge Validation")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Define Drug Hypothesis
    # ========================================================================

    hypothesis = Hypothesis(
        claim="""
        Compound XR-47 shows 40% greater efficacy than current standard of care
        for treating Type 2 Diabetes, with fewer side effects.
        """,
        domain=HypothesisDomain.BIOLOGY,
        evidence=[
            "In vitro studies show 40% increase in glucose uptake",
            "Animal trials (n=50 mice) show improved glycemic control",
            "Preliminary toxicology screen shows acceptable safety profile",
            "Molecular docking suggests high affinity for GLP-1 receptor",
            "Pharmacokinetics show oral bioavailability >60%"
        ],
        assumptions=[
            "Mouse model accurately reflects human physiology",
            "In vitro efficacy translates to in vivo efficacy",
            "Safety profile in animals predicts human safety",
            "GLP-1 pathway is primary mechanism",
            "Oral formulation will be stable",
            "No drug-drug interactions with common diabetic medications"
        ],
        supporting_data={
            "in_vitro_efficacy": 1.4,  # 40% improvement
            "animal_trial_n": 50,
            "bioavailability": 0.62,
            "receptor_affinity": "high",
            "cost_per_dose": 2.50,
            "estimated_trial_cost": 150_000_000  # $150M for Phase III trial
        }
    )

    print(f"📊 Hypothesis: {hypothesis.claim[:100]}...")
    print(f"   Domain: {hypothesis.domain}")
    print(f"   Evidence points: {len(hypothesis.evidence)}")
    print(f"   Assumptions: {len(hypothesis.assumptions)}")
    print()

    # ========================================================================
    # Step 2: Run Turing Challenge Validation
    # ========================================================================

    print("🔬 Running Turing Challenge Validation (Standard Mode)...")
    print("   This will run 6 validation protocols:")
    print("   1. Self-Refutation Protocol")
    print("   2. 200-Question Interrogation")
    print("   3. Hall of Failures Check")
    print("   4. Devil's Advocate Attack")
    print("   5. Agent Tournament Comparison")
    print("   6. Swarm Intelligence Vote")
    print()

    system = TuringChallengeSystem()

    result = await system.validate_hypothesis_complete(
        hypothesis=hypothesis,
        mode=ValidationMode.STANDARD  # Quick enough for this example
    )

    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 3: Analyze Results
    # ========================================================================

    print(f"✅ Overall Score: {result.overall_score:.1f}/100")
    print(f"📋 Recommendation: {result.recommendation}")
    print(f"🎯 Verdict: {result.overall_verdict}")
    print()

    # Self-Refutation Results
    print("🔄 Self-Refutation Protocol:")
    if result.self_refutation_result:
        print(f"   Self-refuting: {result.self_refutation_result.self_refuting}")
        if result.self_refutation_result.contradictions:
            print(f"   Contradictions found: {len(result.self_refutation_result.contradictions)}")
            for contra in result.self_refutation_result.contradictions[:2]:
                print(f"     - {contra.description[:60]}...")
    print()

    # Interrogation Results
    print("❓ Interrogation Results:")
    if result.interrogation_result:
        print(f"   Questions answered: {result.interrogation_result.total_questions_answered}")
        print(f"   Confidence: {result.interrogation_result.overall_confidence:.1f}%")
        print(f"   Unanswered questions: {result.interrogation_result.unanswered_count}")
    print()

    # Devil's Advocate Results
    print("👿 Devil's Advocate Attack:")
    if result.devils_advocate_result:
        print(f"   Total flaws found: {result.devils_advocate_result.total_flaws_found}")
        print(f"   Critical flaws: {len(result.devils_advocate_result.critical_flaws)}")
        print(f"   High severity: {len(result.devils_advocate_result.high_flaws)}")
        print(f"   Robustness score: {result.devils_advocate_result.robustness_score:.1f}/100")

        if result.devils_advocate_result.critical_flaws:
            print("\n   ⚠️  CRITICAL FLAWS IDENTIFIED:")
            for flaw in result.devils_advocate_result.critical_flaws[:3]:
                print(f"     - {flaw.description}")
                print(f"       Mitigation: {flaw.mitigation[:80]}...")
    print()

    # Swarm Voting Results
    print("🐝 Swarm Intelligence Vote:")
    if result.swarm_voting_result:
        print(f"   Consensus: {result.swarm_voting_result.consensus_level}")
        print(f"   Winner: {result.swarm_voting_result.winning_option}")
        print(f"   Agreement: {result.swarm_voting_result.vote_percentage:.1f}%")
        print(f"   Reliability: {result.swarm_voting_result.reliability:.1f}/100")
        if result.swarm_voting_result.groupthink_detected:
            print("   ⚠️  Groupthink detected - opinions too uniform")
    print()

    # ========================================================================
    # Step 4: Business Decision
    # ========================================================================

    print("=" * 80)
    print("BUSINESS DECISION FRAMEWORK")
    print("=" * 80)
    print()

    trial_cost = hypothesis.supporting_data["estimated_trial_cost"]

    if result.recommendation == "PROCEED":
        print("✅ RECOMMENDATION: PROCEED TO CLINICAL TRIALS")
        print(f"   Investment: ${trial_cost:,.0f}")
        print(f"   Confidence: HIGH")
        print(f"   Risk: LOW")
        print()
        print("   Next Steps:")
        print("   1. Finalize IND (Investigational New Drug) application")
        print("   2. Design Phase I safety trial (n=20-80 healthy volunteers)")
        print("   3. Budget $15M for Phase I")
        print("   4. Timeline: 18-24 months to Phase II")

    elif result.recommendation == "REVISE":
        print("⚠️  RECOMMENDATION: REVISE HYPOTHESIS BEFORE PROCEEDING")
        print(f"   Potential savings: ${trial_cost * 0.3:,.0f} (by catching issues early)")
        print(f"   Confidence: MODERATE")
        print(f"   Risk: MEDIUM")
        print()
        print("   Required Actions:")
        print("   1. Address critical flaws identified by Devil's Advocate")
        print("   2. Conduct additional preclinical studies to strengthen evidence")
        print("   3. Re-run Turing Challenge after revisions")

        if result.critical_issues:
            print()
            print("   Critical Issues to Address:")
            for issue in result.critical_issues[:5]:
                print(f"     - {issue}")

    else:  # REJECT
        print("🛑 RECOMMENDATION: REJECT - DO NOT PROCEED")
        print(f"   Potential savings: ${trial_cost:,.0f} (trial avoided)")
        print(f"   Confidence: HIGH")
        print(f"   Risk: HIGH if proceeded")
        print()
        print("   Rationale:")
        print("   - Multiple critical flaws identified")
        print("   - Low probability of success in clinical trials")
        print("   - Better to pivot resources to alternative compounds")
        print()
        print("   Alternative Actions:")
        print("   1. Return to lead optimization phase")
        print("   2. Explore backup compounds")
        print("   3. Consider different molecular scaffolds")

    print()
    print("=" * 80)

    # ========================================================================
    # Step 5: ROI Calculation
    # ========================================================================

    print()
    print("💰 RETURN ON INVESTMENT ANALYSIS")
    print("=" * 80)
    print()

    talair_cost = 5000  # $5K for Turing Challenge validation

    if result.recommendation == "REJECT":
        savings = trial_cost
        roi = (savings / talair_cost) * 100
        print(f"   TalAI Validation Cost: ${talair_cost:,.0f}")
        print(f"   Trial Cost Avoided: ${savings:,.0f}")
        print(f"   Net Savings: ${savings - talair_cost:,.0f}")
        print(f"   ROI: {roi:.0f}x")
        print()
        print(f"   ✨ By catching this bad hypothesis early, TalAI saved")
        print(f"      {roi:.0f}x its cost in avoided trial expenses!")

    elif result.recommendation == "REVISE":
        # Assume revisions save 30% of trial cost by improving success rate
        savings = trial_cost * 0.30
        roi = (savings / talair_cost) * 100
        print(f"   TalAI Validation Cost: ${talair_cost:,.0f}")
        print(f"   Estimated Savings: ${savings:,.0f}")
        print(f"   (from improved trial design & higher success rate)")
        print(f"   ROI: {roi:.0f}x")

    else:  # PROCEED
        # Assume validation increases trial success rate from 20% to 30%
        expected_value_increase = trial_cost * 0.10  # 10% improvement
        print(f"   TalAI Validation Cost: ${talair_cost:,.0f}")
        print(f"   Expected Value Increase: ${expected_value_increase:,.0f}")
        print(f"   (from higher trial success rate)")
        print(f"   ROI: {expected_value_increase / talair_cost:.0f}x")

    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

    print()
    print("📌 KEY TAKEAWAYS:")
    print()
    print("1. TalAI Turing Challenge provides rigorous hypothesis validation")
    print("2. Catches critical flaws before expensive clinical trials")
    print("3. Typical ROI: 1000x-30000x (by avoiding bad trials)")
    print("4. Accelerates drug discovery by focusing resources on best candidates")
    print("5. Complements (not replaces) traditional drug development processes")
    print()
    print("💊 Happy Drug Discovering!")
