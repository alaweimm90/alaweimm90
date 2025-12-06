"""
Devil's Advocate Protocol

Adversarial agent that tries to break every hypothesis.
"""

import time
import asyncio
from typing import List, Optional

from devils_advocate.core.models import AttackResult, AttackStrategy, Flaw, EdgeCase
from devils_advocate.strategies.edge_case_finder import EdgeCaseFinder
from devils_advocate.strategies.assumption_challenger import AssumptionChallenger
from devils_advocate.strategies.scaling_attacker import ScalingAttacker


class DevilsAdvocateProtocol:
    """
    Devil's Advocate Agent

    Systematically tries to break hypotheses through:
    1. Edge case identification
    2. Assumption challenging
    3. Scaling attacks
    4. Composition attacks
    5. Temporal attacks

    Usage:
        protocol = DevilsAdvocateProtocol()
        result = await protocol.attack(hypothesis)
        print(f"Flaws found: {len(result.flaws)}")
    """

    def __init__(self, orchestrator=None, **config):
        self.orchestrator = orchestrator
        self.config = config

        # Initialize attack strategies
        self.edge_case_finder = EdgeCaseFinder()
        self.assumption_challenger = AssumptionChallenger()
        self.scaling_attacker = ScalingAttacker()

    async def attack(self, hypothesis, iterations: int = 3) -> AttackResult:
        """
        Attack hypothesis with all strategies

        Args:
            hypothesis: Hypothesis to attack
            iterations: Number of adversarial iterations

        Returns:
            AttackResult with all flaws found
        """
        start_time = time.time()

        print(f"\n😈 DEVIL'S ADVOCATE ATTACK")
        print(f"   Target: {hypothesis.claim if hasattr(hypothesis, 'claim') else str(hypothesis)}")
        print(f"   Iterations: {iterations}")

        all_flaws = []
        all_edge_cases = []

        for iteration in range(iterations):
            print(f"\n   Attack Round {iteration + 1}...")

            # Strategy 1: Edge Case Identification
            print("      🔍 Finding edge cases...")
            edge_cases = await self.edge_case_finder.find_edge_cases(hypothesis)
            all_edge_cases.extend(edge_cases)
            print(f"         Found {len(edge_cases)} edge cases")

            # Strategy 2: Assumption Challenging
            print("      ⚡ Challenging assumptions...")
            assumption_flaws = await self.assumption_challenger.challenge(hypothesis)
            all_flaws.extend(assumption_flaws)
            print(f"         Found {len(assumption_flaws)} assumption flaws")

            # Strategy 3: Scaling Attacks
            print("      📈 Testing scaling...")
            scaling_flaws = await self.scaling_attacker.attack(hypothesis)
            all_flaws.extend(scaling_flaws)
            print(f"         Found {len(scaling_flaws)} scaling issues")

        # Categorize flaws by severity
        critical_flaws = [f for f in all_flaws if f.severity == "critical"]
        high_flaws = [f for f in all_flaws if f.severity == "high"]
        medium_flaws = [f for f in all_flaws if f.severity == "medium"]
        low_flaws = [f for f in all_flaws if f.severity == "low"]

        # Generate summary
        summary = self._generate_summary(all_flaws, all_edge_cases)

        result = AttackResult(
            hypothesis=hypothesis,
            total_flaws_found=len(all_flaws),
            critical_flaws=critical_flaws,
            high_flaws=high_flaws,
            medium_flaws=medium_flaws,
            low_flaws=low_flaws,
            edge_cases=all_edge_cases,
            attack_strategies_used=[
                AttackStrategy.EDGE_CASE,
                AttackStrategy.ASSUMPTION,
                AttackStrategy.SCALING
            ],
            iterations=iterations,
            execution_time_seconds=time.time() - start_time,
            summary=summary
        )

        print(f"\n😈 ATTACK COMPLETE")
        print(f"   Total flaws: {result.total_flaws_found}")
        print(f"   Critical: {len(critical_flaws)}")
        print(f"   High: {len(high_flaws)}")
        print(f"   Edge cases: {len(all_edge_cases)}")
        print(f"   Verdict: {result.verdict}")

        return result

    def _generate_summary(self, flaws: List[Flaw], edge_cases: List[EdgeCase]) -> str:
        """Generate attack summary"""
        if not flaws and not edge_cases:
            return "No significant flaws found. Hypothesis appears robust."

        summary_parts = []

        if flaws:
            summary_parts.append(f"Found {len(flaws)} potential flaws:")
            critical = [f for f in flaws if f.severity == "critical"]
            if critical:
                summary_parts.append(f"  - {len(critical)} CRITICAL issues that may invalidate the hypothesis")

        if edge_cases:
            summary_parts.append(f"Identified {len(edge_cases)} edge cases that need consideration")

        return "\n".join(summary_parts)
