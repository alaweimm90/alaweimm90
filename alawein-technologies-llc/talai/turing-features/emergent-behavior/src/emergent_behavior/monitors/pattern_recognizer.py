"""Pattern Recognizer"""

from typing import List, Dict, Optional
from datetime import datetime
from emergent_behavior.core.models import EmergentPattern, BehaviorType


class PatternRecognizer:
    """Recognizes emergent patterns from anomalies"""

    def recognize(self, anomaly: Dict, interaction_history: List[Dict]) -> Optional[EmergentPattern]:
        """Recognize if anomaly represents an emergent pattern"""

        if anomaly["type"] == "uniformity":
            # Uniform behavior could be beneficial coordination
            return EmergentPattern(
                pattern_id=f"pattern_{int(datetime.now().timestamp())}",
                description="Agents converging on coordinated strategy",
                behavior_type=BehaviorType.BENEFICIAL,
                frequency=len(interaction_history),
                agents_involved=["multiple"],
                impact_score=60.0,
                first_observed=datetime.now(),
                example_instances=[anomaly["description"]]
            )

        elif anomaly["type"] == "high_failure":
            # High failure rate is harmful
            return EmergentPattern(
                pattern_id=f"pattern_{int(datetime.now().timestamp())}",
                description="System-wide failure cascade",
                behavior_type=BehaviorType.HARMFUL,
                frequency=len([i for i in interaction_history if i.get("outcome") == "failure"]),
                agents_involved=["multiple"],
                impact_score=80.0,
                first_observed=datetime.now(),
                example_instances=[anomaly["description"]]
            )

        return None
