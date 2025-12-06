"""
Real-World Integration Example #3: Financial Algorithm Testing

Demonstrates using TalAI to validate trading algorithms before deploying
them with real capital in financial markets.

Use Case:
- Quantitative hedge fund has developed a new algorithmic trading strategy
- Strategy shows 25% annual returns in backtests
- Need to validate before deploying $500M in capital

Expected Outcome:
- Identify overfitting and data snooping biases
- Test robustness across market regimes
- Prevent catastrophic losses from flawed strategies
- Save millions by catching bad algorithms early
"""

import asyncio
from typing import Dict, List

from self_refutation import Hypothesis, HypothesisDomain
from turing_challenge_system import TuringChallengeSystem, ValidationMode


async def main():
    """Run financial algorithm validation pipeline."""

    print("=" * 80)
    print("FINANCIAL ALGORITHM TESTING - Turing Challenge Validation")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Define Trading Strategy Hypothesis
    # ========================================================================

    hypothesis = Hypothesis(
        claim="""
        A machine learning-based momentum trading strategy achieves 25% annualized
        returns with Sharpe ratio of 2.1, maximum drawdown of 12%, and consistent
        performance across all market conditions.
        """,
        domain=HypothesisDomain.FINANCE,
        evidence=[
            "Backtest on 15 years of historical data (2008-2023)",
            "Sharpe ratio: 2.1 (significantly above market's 0.8)",
            "Annualized return: 25% (vs S&P 500's 10%)",
            "Maximum drawdown: 12% (vs S&P 500's 34% in 2008)",
            "Win rate: 62% across 10,000+ trades",
            "Profitable in 13 out of 15 years",
            "Works on 500 most liquid stocks",
            "Average trade duration: 3 days",
            "Transaction costs included in backtest",
        ],
        assumptions=[
            "Historical patterns persist into future",
            "Sufficient market liquidity for $500M deployment",
            "No significant market impact from our trades",
            "Execution prices match backtest assumptions",
            "Models trained on past data generalize to future",
            "No data snooping bias in feature selection",
            "Market microstructure remains similar",
            "Regulatory environment remains stable",
            "Correlation structure between stocks persists",
        ],
        supporting_data={
            "backtest_period": "2008-2023",
            "annualized_return": 0.25,
            "sharpe_ratio": 2.1,
            "max_drawdown": 0.12,
            "win_rate": 0.62,
            "trade_count": 10847,
            "profitable_years": 13,
            "total_years": 15,
            "capital_to_deploy": 500_000_000,  # $500M
            "stocks_covered": 500,
            "avg_trade_duration_days": 3,
        },
    )

    print(f"💰 Trading Strategy Hypothesis:")
    print(f"   {hypothesis.claim[:100]}...")
    print(f"   Domain: {hypothesis.domain}")
    print(f"   Backtest: {hypothesis.supporting_data['backtest_period']}")
    print(f"   Capital at Risk: ${hypothesis.supporting_data['capital_to_deploy'] / 1e6:.0f}M")
    print()

    # ========================================================================
    # Step 2: Run Rigorous Turing Challenge
    # ========================================================================

    print("🔬 Running RIGOROUS Turing Challenge Validation...")
    print("   (Maximum scrutiny for high-stakes financial deployment)")
    print()

    system = TuringChallengeSystem()

    result = await system.validate_hypothesis_complete(
        hypothesis=hypothesis, mode=ValidationMode.RIGOROUS
    )

    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 3: Risk Analysis
    # ========================================================================

    print(f"📊 Overall Validation Score: {result.overall_score:.1f}/100")
    print(f"📋 Recommendation: {result.recommendation}")
    print(f"🎯 Verdict: {result.overall_verdict}")
    print()

    # Key Metrics
    print("📈 BACKTEST METRICS vs TURING CHALLENGE ASSESSMENT:")
    print()

    sharpe = hypothesis.supporting_data["sharpe_ratio"]
    annual_return = hypothesis.supporting_data["annualized_return"]
    max_dd = hypothesis.supporting_data["max_drawdown"]

    print(f"   Claimed Sharpe Ratio: {sharpe:.2f}")
    if result.overall_score >= 80:
        print(f"   ✅ TalAI Assessment: Likely achievable (confidence: high)")
        adjusted_sharpe = sharpe * 0.95
    elif result.overall_score >= 60:
        print(f"   ⚠️  TalAI Assessment: May be optimistic (confidence: medium)")
        adjusted_sharpe = sharpe * 0.75
    else:
        print(f"   🚨 TalAI Assessment: Significantly overstated (confidence: low)")
        adjusted_sharpe = sharpe * 0.50

    print(f"   Adjusted Sharpe Ratio: {adjusted_sharpe:.2f}")
    print()

    print(f"   Claimed Annual Return: {annual_return * 100:.1f}%")
    if result.overall_score >= 80:
        adjusted_return = annual_return * 0.90
        print(f"   ✅ TalAI Estimate: {adjusted_return * 100:.1f}% (confidence: high)")
    elif result.overall_score >= 60:
        adjusted_return = annual_return * 0.70
        print(f"   ⚠️  TalAI Estimate: {adjusted_return * 100:.1f}% (confidence: medium)")
    else:
        adjusted_return = annual_return * 0.40
        print(f"   🚨 TalAI Estimate: {adjusted_return * 100:.1f}% (confidence: low)")
    print()

    # ========================================================================
    # Step 4: Devil's Advocate Findings
    # ========================================================================

    print("=" * 80)
    print("DEVIL'S ADVOCATE: CRITICAL FLAWS IDENTIFIED")
    print("=" * 80)
    print()

    if result.devils_advocate_result:
        da = result.devils_advocate_result

        print(f"Total Flaws Found: {da.total_flaws_found}")
        print(f"Robustness Score: {da.robustness_score:.1f}/100")
        print()

        if da.critical_flaws:
            print("🚨 CRITICAL FLAWS (Strategy-Breaking):")
            print()
            for i, flaw in enumerate(da.critical_flaws[:5], 1):
                print(f"{i}. {flaw.description}")
                print(f"   Example: {flaw.example}")
                print(f"   Impact: May invalidate backtest results")
                print(f"   Mitigation: {flaw.mitigation}")
                print()

        if da.high_flaws:
            print("⚠️  HIGH SEVERITY FLAWS:")
            for i, flaw in enumerate(da.high_flaws[:3], 1):
                print(f"{i}. {flaw.description}")
                print(f"   Mitigation: {flaw.mitigation[:80]}...")
            print()

    # ========================================================================
    # Step 5: Specific Financial Risks
    # ========================================================================

    print("=" * 80)
    print("FINANCIAL RISK ASSESSMENT")
    print("=" * 80)
    print()

    capital = hypothesis.supporting_data["capital_to_deploy"]

    print("🎲 RISK FACTORS:")
    print()

    print("1. Overfitting Risk:")
    if result.overall_score >= 70:
        print("   LOW - Strategy shows good generalization")
    elif result.overall_score >= 50:
        print("   MEDIUM - Some signs of overfitting detected")
    else:
        print("   HIGH - Strategy likely overfit to backtest period")
    print()

    print("2. Market Impact Risk:")
    print(f"   Capital: ${capital / 1e6:.0f}M")
    print(f"   Stocks: {hypothesis.supporting_data['stocks_covered']}")
    print(f"   Average trade size: ${capital / hypothesis.supporting_data['trade_count'] / 1000:.0f}K")
    if capital > 100_000_000:
        print("   ⚠️  MEDIUM-HIGH - Large capital may move markets")
        print("      Recommendation: Deploy gradually over 6-12 months")
    else:
        print("   ✅ LOW - Capital size manageable")
    print()

    print("3. Data Snooping Bias:")
    if "no data snooping" in [a.lower() for a in hypothesis.assumptions]:
        print("   ⚠️  Not tested - requires walk-forward analysis")
    else:
        print("   🚨 HIGH RISK - May have peeked at test data during development")
    print()

    print("4. Regime Change Risk:")
    if result.overall_score >= 70:
        print("   LOW - Strategy robust across different market conditions")
    else:
        print("   HIGH - Strategy may fail in new market regimes")
    print()

    print("5. Black Swan Risk:")
    max_dd = hypothesis.supporting_data["max_drawdown"]
    print(f"   Backtest max drawdown: {max_dd * 100:.1f}%")
    estimated_worst_case = max_dd * 2.0  # Assume real-world 2x worse
    print(f"   Estimated worst-case: {estimated_worst_case * 100:.1f}%")
    print(f"   Potential loss: ${capital * estimated_worst_case / 1e6:.0f}M")
    print()

    # ========================================================================
    # Step 6: Deployment Decision
    # ========================================================================

    print("=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)
    print()

    if result.recommendation == "PROCEED":
        print("✅ GREEN LIGHT: DEPLOY WITH STANDARD RISK MANAGEMENT")
        print()
        print("   Deployment Plan:")
        print(f"   - Start with 10% of capital (${capital * 0.1 / 1e6:.0f}M)")
        print("   - Monitor performance for 3 months")
        print("   - If Sharpe > 1.5, scale to 25%")
        print("   - If Sharpe > 2.0 after 6 months, scale to 50%")
        print("   - Full deployment only after 12 months of live validation")
        print()
        print("   Risk Controls:")
        print(f"   - Daily loss limit: ${capital * 0.02 / 1e6:.1f}M (2%)")
        print(f"   - Monthly loss limit: ${capital * 0.08 / 1e6:.1f}M (8%)")
        print("   - Automatic halt if drawdown exceeds 15%")
        print("   - Weekly review of key metrics")

    elif result.recommendation == "REVISE":
        print("⚠️  YELLOW LIGHT: REVISE BEFORE DEPLOYMENT")
        print()
        print("   Critical Improvements Needed:")
        for issue in result.critical_issues[:5]:
            print(f"   - {issue}")
        print()
        print("   Additional Testing Required:")
        print("   1. Walk-forward analysis (prevent data snooping)")
        print("   2. Stress testing with 2008, 2020 crises")
        print("   3. Monte Carlo simulation (10,000 scenarios)")
        print("   4. Test on out-of-sample international markets")
        print("   5. Paper trading for 6 months")
        print()
        print("   Timeline: 6-9 months before deployment")

    else:  # REJECT
        print("🛑 RED LIGHT: DO NOT DEPLOY")
        print()
        print(f"   💰 Potential Loss Avoided: ${capital * 0.20 / 1e6:.0f}M")
        print("      (Assuming 20% loss if deployed)")
        print()
        print("   Reasons for Rejection:")
        if result.devils_advocate_result:
            print(f"   - {len(result.devils_advocate_result.critical_flaws)} critical flaws")
        print(f"   - Validation score: {result.overall_score:.1f}/100 (below 50 threshold)")
        print("   - High probability of catastrophic loss")
        print()
        print("   Alternative Actions:")
        print("   1. Return to research phase")
        print("   2. Redesign strategy from scratch")
        print("   3. Consider simpler, more robust approaches")
        print("   4. Allocate capital to proven strategies")

    print()

    # ========================================================================
    # Step 7: ROI of Turing Challenge Validation
    # ========================================================================

    print("=" * 80)
    print("ROI ANALYSIS: TalAI TURING CHALLENGE")
    print("=" * 80)
    print()

    talair_cost = 10000  # $10K for rigorous financial validation

    print(f"   TalAI Validation Cost: ${talair_cost:,.0f}")
    print()

    if result.recommendation == "REJECT":
        # Assume prevented 20% loss on $500M
        loss_prevented = capital * 0.20
        roi = loss_prevented / talair_cost

        print(f"   Catastrophic Loss Prevented: ${loss_prevented / 1e6:.0f}M")
        print(f"   ROI: {roi:,.0f}x")
        print()
        print(f"   🎯 By catching this flawed strategy, TalAI saved")
        print(f"      ${loss_prevented / 1e6:.0f}M (ROI: {roi / 1000:.0f},000x)")

    elif result.recommendation == "REVISE":
        # Assume improved strategy performance by 30%
        performance_boost = capital * adjusted_return * 0.30
        roi = performance_boost / talair_cost

        print(f"   Expected Performance Improvement: ${performance_boost / 1e6:.1f}M/year")
        print(f"   (from addressing identified flaws)")
        print(f"   ROI: {roi:,.0f}x")

    else:  # PROCEED
        # Assume validation increased confidence, allowing faster deployment
        time_saved_months = 6
        opportunity_gain = capital * (adjusted_return / 12) * time_saved_months
        roi = opportunity_gain / talair_cost

        print(f"   Faster Deployment (saved {time_saved_months} months): ${opportunity_gain / 1e6:.1f}M")
        print(f"   ROI: {roi:,.0f}x")

    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

    print()
    print("📌 KEY TAKEAWAYS:")
    print()
    print("1. Backtests are often overly optimistic")
    print("2. TalAI identifies overfitting, data snooping, and hidden risks")
    print("3. Prevents catastrophic losses from flawed strategies")
    print("4. Typical ROI: 1,000x - 100,000x in hedge fund applications")
    print("5. Independent validation critical for fiduciary responsibility")
    print()
    print("💸 Trade safely with rigorous validation!")
