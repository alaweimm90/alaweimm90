"""Edge Case Finder Strategy"""

from typing import List
from devils_advocate.core.models import EdgeCase


class EdgeCaseFinder:
    """Finds edge cases in hypotheses"""

    async def find_edge_cases(self, hypothesis) -> List[EdgeCase]:
        """Find edge cases"""
        edge_cases = []

        # Zero values
        edge_cases.append(EdgeCase(
            case_id="zero_value",
            description="What happens when input is zero?",
            input_conditions="Input parameter = 0",
            expected_behavior="Should handle gracefully",
            probability=0.3
        ))

        # Negative values
        edge_cases.append(EdgeCase(
            case_id="negative_value",
            description="What happens with negative inputs?",
            input_conditions="Input parameter < 0",
            expected_behavior="Should reject or handle appropriately",
            probability=0.4
        ))

        # Extreme values
        edge_cases.append(EdgeCase(
            case_id="extreme_value",
            description="What happens at extreme scales?",
            input_conditions="Input parameter >> normal range",
            expected_behavior="Should scale or fail gracefully",
            probability=0.2
        ))

        # Empty inputs
        edge_cases.append(EdgeCase(
            case_id="empty_input",
            description="What happens with empty/null inputs?",
            input_conditions="Input is empty/null/undefined",
            expected_behavior="Should validate and reject or use defaults",
            probability=0.5
        ))

        return edge_cases
