"""
Emergent Behavior Monitoring

Monitor and amplify beneficial emergent behaviors in multi-agent systems.

Usage:
    from emergent_behavior import EmergentBehaviorProtocol

    protocol = EmergentBehaviorProtocol()
    await protocol.monitor(agents, duration=60)
"""

__version__ = "0.1.0"

from emergent_behavior.protocol import EmergentBehaviorProtocol
from emergent_behavior.core.models import (
    EmergentPattern,
    BehaviorType,
    MonitoringResult,
)

__all__ = [
    "EmergentBehaviorProtocol",
    "EmergentPattern",
    "BehaviorType",
    "MonitoringResult",
]
