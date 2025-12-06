"""
Emergent Behavior Monitoring Protocol

Monitor multi-agent interactions for emergent behaviors.
"""

import time
import asyncio
from typing import List, Optional
from datetime import datetime

from emergent_behavior.core.models import (
    EmergentPattern,
    BehaviorType,
    MonitoringResult,
)
from emergent_behavior.monitors.anomaly_detector import AnomalyDetector
from emergent_behavior.monitors.pattern_recognizer import PatternRecognizer


class EmergentBehaviorProtocol:
    """
    Emergent Behavior Monitoring

    Monitors agent interactions and:
    1. Detects anomalies (unexpected behaviors)
    2. Classifies as beneficial/harmful/neutral
    3. Amplifies beneficial behaviors
    4. Suppresses harmful behaviors

    Usage:
        protocol = EmergentBehaviorProtocol()
        result = await protocol.monitor(agents, duration=60)
    """

    def __init__(self, orchestrator=None, **config):
        self.orchestrator = orchestrator
        self.config = config

        # Initialize monitors
        self.anomaly_detector = AnomalyDetector()
        self.pattern_recognizer = PatternRecognizer()

        # Tracking
        self.observed_patterns = []
        self.interaction_log = []

    async def monitor(
        self,
        agents: List[Any],
        duration_seconds: float = 60,
        interaction_callback: Optional[callable] = None
    ) -> MonitoringResult:
        """
        Monitor agents for emergent behaviors

        Args:
            agents: List of agents to monitor
            duration_seconds: How long to monitor
            interaction_callback: Called for each interaction

        Returns:
            MonitoringResult with patterns detected
        """
        start_time = time.time()

        print(f"\n👁️  EMERGENT BEHAVIOR MONITORING")
        print(f"   Agents: {len(agents)}")
        print(f"   Duration: {duration_seconds}s")

        beneficial_patterns = []
        harmful_patterns = []
        neutral_patterns = []
        total_interactions = 0

        # Monitor interactions
        end_time = start_time + duration_seconds

        while time.time() < end_time:
            # Simulate agent interactions (in practice, hook into actual interactions)
            interaction = await self._observe_interaction(agents)
            self.interaction_log.append(interaction)
            total_interactions += 1

            if interaction_callback:
                interaction_callback(interaction)

            # Check for anomalies
            if total_interactions % 10 == 0:  # Check periodically
                anomalies = self.anomaly_detector.detect(self.interaction_log[-10:])

                for anomaly in anomalies:
                    # Recognize if it's a pattern
                    pattern = self.pattern_recognizer.recognize(
                        anomaly, self.interaction_log
                    )

                    if pattern and pattern not in self.observed_patterns:
                        self.observed_patterns.append(pattern)

                        # Classify pattern
                        if pattern.behavior_type == BehaviorType.BENEFICIAL:
                            beneficial_patterns.append(pattern)
                            await self._amplify_pattern(pattern)
                        elif pattern.behavior_type == BehaviorType.HARMFUL:
                            harmful_patterns.append(pattern)
                            await self._suppress_pattern(pattern)
                        else:
                            neutral_patterns.append(pattern)

            await asyncio.sleep(0.1)  # Small delay

        # Calculate statistics
        duration = time.time() - start_time
        total_patterns = len(beneficial_patterns) + len(harmful_patterns) + len(neutral_patterns)
        emergence_rate = total_patterns / (duration / 60) if duration > 0 else 0

        beneficial_ratio = (len(beneficial_patterns) / total_patterns * 100) if total_patterns > 0 else 0

        result = MonitoringResult(
            monitoring_duration_seconds=duration,
            total_interactions=total_interactions,
            beneficial_patterns=beneficial_patterns,
            harmful_patterns=harmful_patterns,
            neutral_patterns=neutral_patterns,
            amplified_patterns=[p.pattern_id for p in beneficial_patterns],
            suppressed_patterns=[p.pattern_id for p in harmful_patterns],
            total_patterns=total_patterns,
            emergence_rate=emergence_rate,
            beneficial_ratio=beneficial_ratio
        )

        print(f"\n👁️  MONITORING COMPLETE")
        print(f"   {result.verdict}")
        print(f"   Health Score: {result.health_score:.0f}/100")
        print(f"   Emergence Rate: {emergence_rate:.2f} patterns/min")

        return result

    async def _observe_interaction(self, agents: List[Any]) -> dict:
        """Observe an interaction between agents"""
        # Placeholder - in practice, hook into actual agent communication
        import random
        return {
            "timestamp": datetime.now(),
            "agents": random.sample([a.id if hasattr(a, 'id') else str(a) for a in agents], min(2, len(agents))),
            "type": random.choice(["collaboration", "competition", "information_sharing"]),
            "outcome": random.choice(["success", "failure", "neutral"])
        }

    async def _amplify_pattern(self, pattern: EmergentPattern):
        """Amplify a beneficial pattern"""
        print(f"      ✨ AMPLIFYING: {pattern.description}")
        # In practice: increase likelihood, reward agents, modify incentives

    async def _suppress_pattern(self, pattern: EmergentPattern):
        """Suppress a harmful pattern"""
        print(f"      🚫 SUPPRESSING: {pattern.description}")
        # In practice: intervene, modify agent behavior, add constraints
