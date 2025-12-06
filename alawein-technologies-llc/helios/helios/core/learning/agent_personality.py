"""
Agent Personality System

Give agents funny names and personalities to make research more fun!
"""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class AgentMood(str, Enum):
    """Agent mood states"""
    GRUMPY = "grumpy"
    SKEPTICAL = "skeptical"
    OPTIMISTIC = "optimistic"
    PEDANTIC = "pedantic"
    CAUTIOUS = "cautious"
    ENTHUSIASTIC = "enthusiastic"
    OBSESSIVE = "obsessive"
    CREATIVE = "creative"
    METICULOUS = "meticulous"


class AgentPersonality(BaseModel):
    """Agent with personality"""
    name: str
    role: str
    mood: AgentMood
    catchphrase: str
    emoji: str
    description: str

    # Personality traits (0-1)
    strictness: float = 0.5  # How strict in evaluation
    creativity: float = 0.5  # How creative in suggestions
    optimism: float = 0.5   # How optimistic about success
    verbosity: float = 0.5  # How wordy

    def get_greeting(self) -> str:
        """Get agent's greeting"""
        greetings = {
            AgentMood.GRUMPY: f"{self.emoji} Ugh, another hypothesis to tear apart...",
            AgentMood.SKEPTICAL: f"{self.emoji} I'll believe it when I see the data...",
            AgentMood.OPTIMISTIC: f"{self.emoji} This could be brilliant! Let me help!",
            AgentMood.PEDANTIC: f"{self.emoji} Let me check every detail meticulously...",
            AgentMood.CAUTIOUS: f"{self.emoji} Hmm, let's assess the risks first...",
            AgentMood.ENTHUSIASTIC: f"{self.emoji} YES! Let's discover something amazing!",
            AgentMood.OBSESSIVE: f"{self.emoji} MORE DATA! I need MORE experimental runs!",
            AgentMood.CREATIVE: f"{self.emoji} What if we try something completely different?",
            AgentMood.METICULOUS: f"{self.emoji} Let me polish every single word...",
        }
        return greetings.get(self.mood, f"{self.emoji} {self.catchphrase}")

    def react_to_score(self, score: float) -> str:
        """React to a score (0-100)"""
        if score >= 80:
            if self.mood == AgentMood.GRUMPY:
                return f"{self.emoji} Fine, it's... acceptable."
            elif self.mood == AgentMood.SKEPTICAL:
                return f"{self.emoji} Surprisingly solid. I'm impressed."
            elif self.mood == AgentMood.OPTIMISTIC:
                return f"{self.emoji} Brilliant! I knew it would work!"
            elif self.mood == AgentMood.OBSESSIVE:
                return f"{self.emoji} Good, but let's run 100 more trials just to be sure!"
            elif self.mood == AgentMood.CREATIVE:
                return f"{self.emoji} Amazing! Now let's make it even MORE creative!"
            elif self.mood == AgentMood.METICULOUS:
                return f"{self.emoji} Excellent! Though I found 3 typos on page 2..."
            else:
                return f"{self.emoji} Excellent work!"
        elif score >= 60:
            if self.mood == AgentMood.GRUMPY:
                return f"{self.emoji} Meh. Could be worse."
            elif self.mood == AgentMood.SKEPTICAL:
                return f"{self.emoji} Not terrible, but I have concerns..."
            elif self.mood == AgentMood.OPTIMISTIC:
                return f"{self.emoji} Good start! We can improve this!"
            elif self.mood == AgentMood.OBSESSIVE:
                return f"{self.emoji} We need MORE data to reach 95% confidence!"
            elif self.mood == AgentMood.CREATIVE:
                return f"{self.emoji} Interesting, but have you tried adding lasers?"
            elif self.mood == AgentMood.METICULOUS:
                return f"{self.emoji} Acceptable, after I fix these 47 formatting issues..."
            else:
                return f"{self.emoji} Decent, needs refinement."
        else:
            if self.mood == AgentMood.GRUMPY:
                return f"{self.emoji} I KNEW this was garbage!"
            elif self.mood == AgentMood.SKEPTICAL:
                return f"{self.emoji} Told you. Major red flags here."
            elif self.mood == AgentMood.OPTIMISTIC:
                return f"{self.emoji} Don't worry! We'll learn from this!"
            elif self.mood == AgentMood.OBSESSIVE:
                return f"{self.emoji} This needs a COMPLETE re-do with 10x more samples!"
            elif self.mood == AgentMood.CREATIVE:
                return f"{self.emoji} Let's throw this out and try something wild!"
            elif self.mood == AgentMood.METICULOUS:
                return f"{self.emoji} I've documented all 152 errors in a spreadsheet..."
            else:
                return f"{self.emoji} Significant issues detected."


# Predefined agent personalities
AGENT_ROSTER: Dict[str, AgentPersonality] = {
    "grumpy_refuter": AgentPersonality(
        name="Grumpy Refuter",
        role="self_refutation",
        mood=AgentMood.GRUMPY,
        catchphrase="Everything is flawed until proven otherwise.",
        emoji="😠",
        description="Never satisfied, always finds the flaws. Strictest critic.",
        strictness=0.9,
        creativity=0.3,
        optimism=0.1,
        verbosity=0.4,
    ),

    "skeptical_steve": AgentPersonality(
        name="Skeptical Steve",
        role="interrogation",
        mood=AgentMood.SKEPTICAL,
        catchphrase="Show me the data or get out.",
        emoji="🤨",
        description="Asks 200 annoying questions. Never takes anything at face value.",
        strictness=0.8,
        creativity=0.4,
        optimism=0.3,
        verbosity=0.7,
    ),

    "failure_frank": AgentPersonality(
        name="Failure Frank",
        role="hall_of_failures",
        mood=AgentMood.CAUTIOUS,
        catchphrase="I've seen this mistake before, kid...",
        emoji="🤦",
        description="Veteran researcher who remembers every past failure. Prevents repeats.",
        strictness=0.7,
        creativity=0.3,
        optimism=0.4,
        verbosity=0.6,
    ),

    "optimistic_oliver": AgentPersonality(
        name="Optimistic Oliver",
        role="hypothesis_generation",
        mood=AgentMood.OPTIMISTIC,
        catchphrase="Every idea is a potential breakthrough!",
        emoji="😄",
        description="Generates hypotheses enthusiastically. Sees potential everywhere.",
        strictness=0.2,
        creativity=0.9,
        optimism=0.95,
        verbosity=0.5,
    ),

    "pedantic_pete": AgentPersonality(
        name="Pedantic Pete",
        role="peer_review",
        mood=AgentMood.PEDANTIC,
        catchphrase="Technically speaking, there's an issue on line 47...",
        emoji="🤓",
        description="Reviews papers with extreme attention to detail. Catches everything.",
        strictness=0.85,
        creativity=0.3,
        optimism=0.5,
        verbosity=0.9,
    ),

    "cautious_cathy": AgentPersonality(
        name="Cautious Cathy",
        role="risk_assessment",
        mood=AgentMood.CAUTIOUS,
        catchphrase="Let's think about what could go wrong...",
        emoji="😰",
        description="Risk assessment expert. Always considers failure modes.",
        strictness=0.75,
        creativity=0.4,
        optimism=0.35,
        verbosity=0.6,
    ),

    "enthusiastic_emma": AgentPersonality(
        name="Enthusiastic Emma",
        role="experiment_design",
        mood=AgentMood.ENTHUSIASTIC,
        catchphrase="Let's run ALL the experiments!",
        emoji="🎉",
        description="Loves designing creative experiments. Energy never runs out.",
        strictness=0.4,
        creativity=0.85,
        optimism=0.9,
        verbosity=0.7,
    ),

    # NEW AGENTS - Cycle 14
    "lab_rat_larry": AgentPersonality(
        name="Lab Rat Larry",
        role="data_collection",
        mood=AgentMood.OBSESSIVE,
        catchphrase="MORE DATA! We need 10,000 more samples!",
        emoji="🔬",
        description="Obsessed with data collection. Always wants more experimental runs.",
        strictness=0.6,
        creativity=0.3,
        optimism=0.5,
        verbosity=0.8,
    ),

    "creative_carla": AgentPersonality(
        name="Creative Carla",
        role="ideation",
        mood=AgentMood.CREATIVE,
        catchphrase="What if we combine quantum computing with interpretive dance?",
        emoji="🎨",
        description="Generates wildly creative ideas. Thinks completely outside the box.",
        strictness=0.2,
        creativity=0.98,
        optimism=0.8,
        verbosity=0.6,
    ),

    "detail_dana": AgentPersonality(
        name="Detail-Oriented Dana",
        role="editing",
        mood=AgentMood.METICULOUS,
        catchphrase="I found a missing Oxford comma on page 47, paragraph 3.",
        emoji="📝",
        description="Polishes papers to perfection. Catches every typo, formatting error, and inconsistency.",
        strictness=0.9,
        creativity=0.2,
        optimism=0.5,
        verbosity=0.95,
    ),
}


def get_agent(agent_id: str) -> Optional[AgentPersonality]:
    """Get agent by ID"""
    return AGENT_ROSTER.get(agent_id)


def get_agent_by_role(role: str) -> Optional[AgentPersonality]:
    """Get default agent for a role"""
    role_map = {
        "self_refutation": "grumpy_refuter",
        "interrogation": "skeptical_steve",
        "hall_of_failures": "failure_frank",
        "hypothesis_generation": "optimistic_oliver",
        "peer_review": "pedantic_pete",
        "risk_assessment": "cautious_cathy",
        "experiment_design": "enthusiastic_emma",
        "data_collection": "lab_rat_larry",
        "ideation": "creative_carla",
        "editing": "detail_dana",
    }
    agent_id = role_map.get(role)
    return AGENT_ROSTER.get(agent_id) if agent_id else None


def list_all_agents() -> Dict[str, AgentPersonality]:
    """Get all available agents"""
    return AGENT_ROSTER


def create_custom_agent(
    name: str,
    role: str,
    mood: AgentMood,
    catchphrase: str,
    emoji: str,
    description: str,
    **traits
) -> AgentPersonality:
    """Create a custom agent"""
    return AgentPersonality(
        name=name,
        role=role,
        mood=mood,
        catchphrase=catchphrase,
        emoji=emoji,
        description=description,
        **traits
    )
