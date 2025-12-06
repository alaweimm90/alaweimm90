"""Anomaly Detector"""

from typing import List, Dict


class AnomalyDetector:
    """Detects anomalous patterns in agent interactions"""

    def detect(self, interactions: List[Dict]) -> List[Dict]:
        """Detect anomalies in recent interactions"""
        anomalies = []

        if len(interactions) < 3:
            return anomalies

        # Simple anomaly detection: look for unusual patterns
        # In practice: statistical methods, ML models

        # Check for sudden changes in interaction types
        types = [i.get("type") for i in interactions]
        if len(set(types)) == 1 and len(types) > 5:
            # All same type = potential emerging pattern
            anomalies.append({
                "type": "uniformity",
                "description": f"All interactions are {types[0]}",
                "severity": "medium"
            })

        # Check for unexpected outcomes
        outcomes = [i.get("outcome") for i in interactions]
        failure_rate = outcomes.count("failure") / len(outcomes)
        if failure_rate > 0.7:
            anomalies.append({
                "type": "high_failure",
                "description": "High failure rate detected",
                "severity": "high"
            })

        return anomalies
