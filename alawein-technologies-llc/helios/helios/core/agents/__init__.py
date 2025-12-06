"""
Agents Module: Personality-based research agents
"""

try:
    from helios.core.agents.personality_agent import PersonalityAgent
except ImportError:
    pass

__all__ = ['PersonalityAgent']
