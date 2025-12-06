"""Groupthink Detector"""

from typing import List
from swarm_voting.core.models import Vote


class GroupthinkDetector:
    """Detects groupthink in voting"""

    def detect(self, votes: List[Vote], threshold: float = 0.15) -> bool:
        """
        Detect if groupthink occurred

        Args:
            votes: All votes cast
            threshold: Variance threshold below which groupthink is detected

        Returns:
            True if groupthink detected
        """
        if len(votes) < 3:
            return False

        # Calculate variance in votes
        vote_options = [v.option for v in votes]
        unique_options = set(vote_options)

        # If too few unique options, might be groupthink
        if len(unique_options) <= 2 and len(votes) > 10:
            # Check if one option dominates heavily (>90%)
            max_count = max(vote_options.count(opt) for opt in unique_options)
            if max_count / len(votes) > 0.90:
                return True

        # Check confidence variance (low variance = groupthink)
        confidences = [v.confidence for v in votes]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf)**2 for c in confidences) / len(confidences)

        return variance < threshold
