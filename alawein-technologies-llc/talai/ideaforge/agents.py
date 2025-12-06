"""
Specialized Agents for IdeaForge

17 Agents that work in parallel to generate breakthrough ideas:
1. Meta-Orchestrator
2. Planner
3. Researcher
4. Critic
5. Evolver
6. Executor
7. Synthesizer
8. Chaos Monkey
9. Domain Expert
10. Futurist
11. Historian
12. Economist
13. Ethicist
14. UX Visionary
15. Technical Architect
16. Market Analyst
17. Wild Card
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AgentInsight:
    """Represents an insight from an agent"""
    agent_name: str
    insight_type: str
    content: str
    confidence: float  # 0-1
    novelty: float  # 0-1


class Agent:
    """Base class for all agents"""

    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        """Analyze context and return insights"""
        raise NotImplementedError


# ========== Agent 1: Meta-Orchestrator ==========
class MetaOrchestrator(Agent):
    """Coordinates all other agents, identifies meta-patterns"""

    def __init__(self):
        super().__init__("meta_orchestrator", "coordination_and_meta_patterns")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        # Identify meta-patterns across framework outputs
        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="meta_pattern",
            content="Detected convergence between first_principles and biomimicry",
            confidence=0.8,
            novelty=0.7
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="coordination",
            content="Prioritizing inversion framework - shows highest novelty potential",
            confidence=0.9,
            novelty=0.6
        ))

        return insights


# ========== Agent 2: Planner ==========
class Planner(Agent):
    """Creates implementation roadmaps for ideas"""

    def __init__(self):
        super().__init__("planner", "roadmapping_and_execution")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="roadmap",
            content="Phase 1: Validate in 4 weeks. Phase 2: MVP in 8 weeks.",
            confidence=0.85,
            novelty=0.4
        ))

        return insights


# ========== Agent 3: Researcher ==========
class Researcher(Agent):
    """Finds related work, prior art, state-of-the-art"""

    def __init__(self):
        super().__init__("researcher", "literature_and_prior_art")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="prior_art",
            content="Found 23 related papers. None solve the exact problem.",
            confidence=0.9,
            novelty=0.8
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="sota",
            content="Current SOTA: XYZ method. Opportunity to improve by 15%+",
            confidence=0.75,
            novelty=0.7
        ))

        return insights


# ========== Agent 4: Critic ==========
class Critic(Agent):
    """Devil's advocate, finds flaws and risks"""

    def __init__(self):
        super().__init__("critic", "risk_assessment_and_flaws")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="risk",
            content="Risk: Scaling may be prohibitively expensive",
            confidence=0.7,
            novelty=0.3
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="flaw",
            content="Assumption: Users want this. Needs validation.",
            confidence=0.9,
            novelty=0.5
        ))

        return insights


# ========== Agent 5: Evolver ==========
class Evolver(Agent):
    """Takes existing ideas and mutates/evolves them"""

    def __init__(self):
        super().__init__("evolver", "idea_mutation_and_evolution")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="mutation",
            content="Mutated idea: Add AI component to existing concept",
            confidence=0.8,
            novelty=0.85
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="evolution",
            content="Evolved: Take idea from B2C to B2B2C model",
            confidence=0.7,
            novelty=0.75
        ))

        return insights


# ========== Agent 6: Executor ==========
class Executor(Agent):
    """Focuses on practical implementation details"""

    def __init__(self):
        super().__init__("executor", "implementation_pragmatics")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="technical",
            content="Tech stack: Python FastAPI + React + PostgreSQL",
            confidence=0.85,
            novelty=0.4
        ))

        return insights


# ========== Agent 7: Synthesizer ==========
class Synthesizer(Agent):
    """Combines insights from all other agents"""

    def __init__(self):
        super().__init__("synthesizer", "insight_integration")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="synthesis",
            content="Combined 5 frameworks → coherent meta-idea",
            confidence=0.9,
            novelty=0.9
        ))

        return insights


# ========== Agent 8: Chaos Monkey ==========
class ChaosMonkey(Agent):
    """Introduces random disruptions, wild ideas"""

    def __init__(self):
        super().__init__("chaos_monkey", "randomness_and_disruption")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="wild_idea",
            content="What if we did this entirely in VR?",
            confidence=0.3,
            novelty=0.95
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="disruption",
            content="Challenge: Make it 100x cheaper, not 10% better",
            confidence=0.5,
            novelty=0.9
        ))

        return insights


# ========== Agent 9: Domain Expert ==========
class DomainExpert(Agent):
    """Deep domain knowledge"""

    def __init__(self, domain: str = "general"):
        super().__init__("domain_expert", f"expertise_in_{domain}")
        self.domain = domain

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="domain_knowledge",
            content=f"In {self.domain}, the key constraint is...",
            confidence=0.95,
            novelty=0.6
        ))

        return insights


# ========== Agent 10: Futurist ==========
class Futurist(Agent):
    """Projects trends 5-10 years forward"""

    def __init__(self):
        super().__init__("futurist", "trend_projection")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="future_trend",
            content="By 2030, this space will be dominated by AI-first solutions",
            confidence=0.75,
            novelty=0.7
        ))

        return insights


# ========== Agent 11: Historian ==========
class Historian(Agent):
    """Analyzes patterns from history"""

    def __init__(self):
        super().__init__("historian", "historical_patterns")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="historical_pattern",
            content="Similar transition happened in 1990s with internet adoption",
            confidence=0.8,
            novelty=0.65
        ))

        return insights


# ========== Agent 12: Economist ==========
class Economist(Agent):
    """Analyzes market dynamics, pricing, incentives"""

    def __init__(self):
        super().__init__("economist", "market_economics")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="market_size",
            content="TAM: $500M, SAM: $100M, SOM: $10M (year 3)",
            confidence=0.7,
            novelty=0.5
        ))

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="pricing",
            content="Pricing: $99/mo SMB, $999/mo enterprise",
            confidence=0.75,
            novelty=0.4
        ))

        return insights


# ========== Agent 13: Ethicist ==========
class Ethicist(Agent):
    """Evaluates ethical implications"""

    def __init__(self):
        super().__init__("ethicist", "ethical_analysis")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="ethical_concern",
            content="Privacy concern: User data storage. Mitigate with encryption.",
            confidence=0.85,
            novelty=0.5
        ))

        return insights


# ========== Agent 14: UX Visionary ==========
class UXVisionary(Agent):
    """Imagines perfect user experiences"""

    def __init__(self):
        super().__init__("ux_visionary", "user_experience")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="ux_insight",
            content="One-click solution. No learning curve. Magical experience.",
            confidence=0.8,
            novelty=0.75
        ))

        return insights


# ========== Agent 15: Technical Architect ==========
class TechnicalArchitect(Agent):
    """Designs system architecture"""

    def __init__(self):
        super().__init__("technical_architect", "system_design")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="architecture",
            content="Microservices: API gateway → 5 services → event bus",
            confidence=0.9,
            novelty=0.5
        ))

        return insights


# ========== Agent 16: Market Analyst ==========
class MarketAnalyst(Agent):
    """Analyzes competitive landscape"""

    def __init__(self):
        super().__init__("market_analyst", "competitive_analysis")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="competition",
            content="3 direct competitors. None have feature X. Opportunity.",
            confidence=0.85,
            novelty=0.7
        ))

        return insights


# ========== Agent 17: Wild Card ==========
class WildCard(Agent):
    """Completely unexpected perspectives"""

    def __init__(self):
        super().__init__("wild_card", "unexpected_angles")

    async def analyze(self, context: Dict[str, Any]) -> List[AgentInsight]:
        insights = []

        insights.append(AgentInsight(
            agent_name=self.name,
            insight_type="wild_perspective",
            content="What if we gamified it? What if users were the product?",
            confidence=0.4,
            novelty=0.95
        ))

        return insights


# ========== Agent Registry ==========
ALL_AGENTS = {
    "meta_orchestrator": MetaOrchestrator,
    "planner": Planner,
    "researcher": Researcher,
    "critic": Critic,
    "evolver": Evolver,
    "executor": Executor,
    "synthesizer": Synthesizer,
    "chaos_monkey": ChaosMonkey,
    "domain_expert": DomainExpert,
    "futurist": Futurist,
    "historian": Historian,
    "economist": Economist,
    "ethicist": Ethicist,
    "ux_visionary": UXVisionary,
    "technical_architect": TechnicalArchitect,
    "market_analyst": MarketAnalyst,
    "wild_card": WildCard,
}


def get_agent(name: str) -> Agent:
    """Factory function to get agent by name"""
    if name not in ALL_AGENTS:
        raise ValueError(f"Unknown agent: {name}")
    return ALL_AGENTS[name]()


def get_all_agents() -> List[Agent]:
    """Get instances of all agents"""
    return [cls() for cls in ALL_AGENTS.values()]
