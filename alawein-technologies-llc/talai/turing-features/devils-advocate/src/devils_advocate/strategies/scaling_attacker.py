"""Scaling Attacker Strategy"""

from typing import List
from devils_advocate.core.models import Flaw, AttackStrategy


class ScalingAttacker:
    """Tests scaling properties"""

    async def attack(self, hypothesis) -> List[Flaw]:
        """Attack with scaling tests"""
        flaws = []

        # Small scale failures
        flaws.append(Flaw(
            flaw_id="small_scale_failure",
            description="May not work at very small scales",
            severity="medium",
            attack_strategy=AttackStrategy.SCALING,
            example="N=1 or N=2 may produce degenerate cases",
            mitigation="Specify minimum viable scale"
        ))

        # Large scale failures
        flaws.append(Flaw(
            flaw_id="large_scale_failure",
            description="May not scale to large instances",
            severity="high",
            attack_strategy=AttackStrategy.SCALING,
            example="Computational complexity may explode at scale",
            mitigation="Analyze algorithmic complexity"
        ))

        return flaws


class CompositionAttacker:
    """Tests composition failures"""

    async def attack(self, hypothesis) -> List[Flaw]:
        """Attack with composition tests"""
        flaws = []

        flaws.append(Flaw(
            flaw_id="composition_conflict",
            description="Components may conflict when combined",
            severity="medium",
            attack_strategy=AttackStrategy.COMPOSITION,
            example="Method A + Method B may have interface mismatches",
            mitigation="Test component integration explicitly"
        ))

        return flaws
