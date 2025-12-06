"""Assumption Challenger Strategy"""

from typing import List
from devils_advocate.core.models import Flaw, AttackStrategy


class AssumptionChallenger:
    """Challenges assumptions in hypotheses"""

    async def challenge(self, hypothesis) -> List[Flaw]:
        """Challenge assumptions"""
        flaws = []

        # Challenge assumption of independence
        flaws.append(Flaw(
            flaw_id="assumption_independence",
            description="Assumes variables are independent when they may not be",
            severity="medium",
            attack_strategy=AttackStrategy.ASSUMPTION,
            example="Variables X and Y may have hidden correlations",
            mitigation="Test for correlations explicitly"
        ))

        # Challenge assumption of linearity
        flaws.append(Flaw(
            flaw_id="assumption_linearity",
            description="Assumes linear relationships when non-linear may exist",
            severity="medium",
            attack_strategy=AttackStrategy.ASSUMPTION,
            example="Effect may saturate or show diminishing returns",
            mitigation="Test for non-linear relationships"
        ))

        # Challenge assumption of stationarity
        flaws.append(Flaw(
            flaw_id="assumption_stationarity",
            description="Assumes conditions remain constant over time",
            severity="high",
            attack_strategy=AttackStrategy.ASSUMPTION,
            example="Environment may drift, making hypothesis invalid",
            mitigation="Add temporal validation"
        ))

        return flaws
