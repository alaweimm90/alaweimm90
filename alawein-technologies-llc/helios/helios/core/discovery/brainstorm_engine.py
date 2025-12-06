"""
ORCHEX Brainstorming Engine

Advanced idea generation using multiple creativity techniques.
Helps discover novel approaches and unconventional solutions.

Cycle 23-24: Brainstorming Module
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random


class BrainstormStrategy(str, Enum):
    """Different brainstorming strategies"""

    SCAMPER = "scamper"  # Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse
    LATERAL = "lateral"  # Lateral thinking / random associations
    CONSTRAINT_RELAX = "constraint_relax"  # Remove constraints and explore
    CROSS_DOMAIN = "cross_domain"  # Apply ideas from other domains
    WHAT_IF = "what_if"  # "What if..." scenario exploration
    REVERSE = "reverse"  # Reverse the problem
    ANALOGY = "analogy"  # Find analogies in nature, other fields
    RANDOM_WORD = "random_word"  # Random word association


@dataclass
class BrainstormIdea:
    """A generated idea from brainstorming"""

    idea: str
    strategy: BrainstormStrategy
    reasoning: str
    novelty_score: float  # 0-1, estimated novelty
    feasibility_score: float  # 0-1, estimated feasibility
    interestingness: float  # 0-1, how interesting/surprising


@dataclass
class BrainstormSession:
    """Results from a brainstorming session"""

    topic: str
    domain: str
    strategies_used: List[BrainstormStrategy]
    ideas: List[BrainstormIdea]
    total_ideas: int
    duration_seconds: float


class BrainstormEngine:
    """
    Advanced brainstorming engine using multiple creativity techniques.

    Inspired by:
    - SCAMPER technique (Eberle, 1971)
    - Lateral thinking (de Bono, 1967)
    - TRIZ contradiction matrix
    - Biomimicry principles
    """

    def __init__(self, orchestrator=None):
        """
        Initialize brainstorming engine

        Args:
            orchestrator: Optional AI Orchestrator for LLM-based generation
        """
        self.orchestrator = orchestrator

        # Random words for association
        self.random_words = [
            "mirror", "water", "tree", "network", "wave", "spiral", "crystal",
            "swarm", "music", "river", "mountain", "light", "shadow", "bridge",
            "clock", "compass", "maze", "telescope", "magnet", "prism"
        ]

    async def brainstorm(
        self,
        topic: str,
        domain: str = "general",
        num_ideas: int = 10,
        strategies: Optional[List[BrainstormStrategy]] = None,
        min_novelty: float = 0.5
    ) -> BrainstormSession:
        """
        Run brainstorming session to generate ideas

        Args:
            topic: Research topic or problem to brainstorm about
            domain: Scientific domain
            num_ideas: Target number of ideas to generate
            strategies: Which strategies to use (default: all)
            min_novelty: Minimum novelty threshold (0-1)

        Returns:
            BrainstormSession with generated ideas
        """
        import time
        start_time = time.time()

        if strategies is None:
            strategies = list(BrainstormStrategy)

        ideas = []

        # Distribute ideas across strategies
        ideas_per_strategy = max(1, num_ideas // len(strategies))

        for strategy in strategies:
            print(f"  💡 Using {strategy.value} technique...")

            strategy_ideas = await self._apply_strategy(
                strategy=strategy,
                topic=topic,
                domain=domain,
                num_ideas=ideas_per_strategy
            )

            # Filter by novelty
            strategy_ideas = [
                idea for idea in strategy_ideas
                if idea.novelty_score >= min_novelty
            ]

            ideas.extend(strategy_ideas)

        # Sort by combined score (novelty + interestingness)
        ideas.sort(
            key=lambda x: (x.novelty_score + x.interestingness) / 2,
            reverse=True
        )

        # Limit to requested number
        ideas = ideas[:num_ideas]

        duration = time.time() - start_time

        return BrainstormSession(
            topic=topic,
            domain=domain,
            strategies_used=strategies,
            ideas=ideas,
            total_ideas=len(ideas),
            duration_seconds=duration
        )

    async def _apply_strategy(
        self,
        strategy: BrainstormStrategy,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Apply a specific brainstorming strategy"""

        if strategy == BrainstormStrategy.SCAMPER:
            return await self._scamper(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.LATERAL:
            return await self._lateral_thinking(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.CONSTRAINT_RELAX:
            return await self._constraint_relaxation(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.CROSS_DOMAIN:
            return await self._cross_domain_transfer(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.WHAT_IF:
            return await self._what_if_scenarios(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.REVERSE:
            return await self._reverse_problem(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.ANALOGY:
            return await self._find_analogies(topic, domain, num_ideas)
        elif strategy == BrainstormStrategy.RANDOM_WORD:
            return await self._random_word_association(topic, domain, num_ideas)
        else:
            return []

    async def _scamper(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """SCAMPER technique: Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse"""

        scamper_prompts = [
            ("Substitute", "What component or approach could we substitute?"),
            ("Combine", "What could we combine or merge?"),
            ("Adapt", "What could we adapt from another context?"),
            ("Modify", "What could we modify, magnify, or minify?"),
            ("Put to another use", "Could we use this in a completely different way?"),
            ("Eliminate", "What could we remove or simplify?"),
            ("Reverse", "What if we reversed the process or goal?"),
        ]

        ideas = []

        # Select random SCAMPER prompts
        selected_prompts = random.sample(
            scamper_prompts,
            min(num_ideas, len(scamper_prompts))
        )

        for prompt_type, prompt in selected_prompts:
            idea_text = f"{prompt_type}: Apply to '{topic}' - {prompt.lower()}"

            # If orchestrator available, use LLM to generate concrete idea
            if self.orchestrator:
                # TODO: Use orchestrator to generate ideas
                pass

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.SCAMPER,
                reasoning=f"SCAMPER ({prompt_type}) applied to problem",
                novelty_score=random.uniform(0.6, 0.9),
                feasibility_score=random.uniform(0.5, 0.8),
                interestingness=random.uniform(0.6, 0.85)
            ))

        return ideas

    async def _lateral_thinking(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Lateral thinking: Random associations and unconventional connections"""

        ideas = []

        # Random entry point
        random_concepts = random.sample(self.random_words, min(num_ideas, len(self.random_words)))

        for concept in random_concepts:
            idea_text = f"Connect '{topic}' with '{concept}' - explore unexpected synergies"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.LATERAL,
                reasoning=f"Lateral thinking via random concept '{concept}'",
                novelty_score=random.uniform(0.7, 0.95),
                feasibility_score=random.uniform(0.3, 0.7),
                interestingness=random.uniform(0.7, 0.9)
            ))

        return ideas

    async def _constraint_relaxation(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Remove typical constraints and explore freely"""

        constraint_prompts = [
            "What if there were no computational limits?",
            "What if data was infinite and perfect?",
            "What if we ignored current paradigms?",
            "What if this problem didn't exist in 100 years - what changed?",
            "What if the opposite approach was correct?",
            "What if we had quantum computing available now?",
            "What if we could violate physical laws?",
        ]

        ideas = []

        selected = random.sample(constraint_prompts, min(num_ideas, len(constraint_prompts)))

        for prompt in selected:
            idea_text = f"For '{topic}': {prompt}"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.CONSTRAINT_RELAX,
                reasoning="Relaxing typical constraints to enable creative exploration",
                novelty_score=random.uniform(0.75, 1.0),
                feasibility_score=random.uniform(0.2, 0.6),
                interestingness=random.uniform(0.8, 0.95)
            ))

        return ideas

    async def _cross_domain_transfer(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Apply ideas from other domains"""

        other_domains = [
            "biology", "physics", "economics", "psychology", "architecture",
            "music", "game theory", "network theory", "ecology", "chemistry"
        ]

        # Remove current domain
        if domain in other_domains:
            other_domains.remove(domain)

        ideas = []

        selected_domains = random.sample(other_domains, min(num_ideas, len(other_domains)))

        for other_domain in selected_domains:
            idea_text = f"Apply principles from {other_domain} to '{topic}'"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.CROSS_DOMAIN,
                reasoning=f"Cross-domain transfer from {other_domain}",
                novelty_score=random.uniform(0.65, 0.9),
                feasibility_score=random.uniform(0.4, 0.75),
                interestingness=random.uniform(0.7, 0.9)
            ))

        return ideas

    async def _what_if_scenarios(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Explore 'what if' scenarios"""

        scenarios = [
            "What if the problem was 1000x larger?",
            "What if the problem was 1000x smaller?",
            "What if we solved this in 1 minute?",
            "What if we had 100 years to solve this?",
            "What if this was solved by nature/evolution?",
            "What if this was solved by a child?",
            "What if we used only existing technology?",
            "What if we used only future technology?",
        ]

        ideas = []

        selected = random.sample(scenarios, min(num_ideas, len(scenarios)))

        for scenario in selected:
            idea_text = f"For '{topic}': {scenario}"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.WHAT_IF,
                reasoning="What-if scenario exploration",
                novelty_score=random.uniform(0.6, 0.85),
                feasibility_score=random.uniform(0.4, 0.7),
                interestingness=random.uniform(0.65, 0.85)
            ))

        return ideas

    async def _reverse_problem(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Reverse the problem to gain new perspective"""

        reversals = [
            "Instead of solving, what if we amplified the problem?",
            "Instead of building, what if we deconstructed?",
            "Instead of speeding up, what if we slowed down?",
            "Instead of adding features, what if we removed them?",
            "Instead of human-centric, what if we designed for machines?",
        ]

        ideas = []

        selected = random.sample(reversals, min(num_ideas, len(reversals)))

        for reversal in selected:
            idea_text = f"For '{topic}': {reversal}"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.REVERSE,
                reasoning="Problem reversal for new perspective",
                novelty_score=random.uniform(0.7, 0.9),
                feasibility_score=random.uniform(0.3, 0.65),
                interestingness=random.uniform(0.75, 0.9)
            ))

        return ideas

    async def _find_analogies(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Find analogies in nature and other systems"""

        analogy_sources = [
            ("ant colonies", "distributed problem solving"),
            ("immune system", "adaptive defense mechanisms"),
            ("neural networks in brain", "parallel processing"),
            ("evolution", "iterative optimization"),
            ("crystal growth", "self-organization"),
            ("river networks", "optimal routing"),
            ("forest ecosystems", "resource allocation"),
            ("flocking behavior", "emergent coordination"),
            ("DNA replication", "error correction"),
            ("spider webs", "efficient structures"),
        ]

        ideas = []

        selected = random.sample(analogy_sources, min(num_ideas, len(analogy_sources)))

        for source, principle in selected:
            idea_text = f"Apply {source} analogy to '{topic}' - leverage {principle}"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.ANALOGY,
                reasoning=f"Biomimicry/analogy from {source}",
                novelty_score=random.uniform(0.65, 0.88),
                feasibility_score=random.uniform(0.5, 0.8),
                interestingness=random.uniform(0.7, 0.9)
            ))

        return ideas

    async def _random_word_association(
        self,
        topic: str,
        domain: str,
        num_ideas: int
    ) -> List[BrainstormIdea]:
        """Use random words to trigger associations"""

        ideas = []

        random_words = random.sample(self.random_words, min(num_ideas, len(self.random_words)))

        for word in random_words:
            idea_text = f"'{word}' + '{topic}' → explore the connection"

            ideas.append(BrainstormIdea(
                idea=idea_text,
                strategy=BrainstormStrategy.RANDOM_WORD,
                reasoning=f"Random word '{word}' association",
                novelty_score=random.uniform(0.6, 0.9),
                feasibility_score=random.uniform(0.3, 0.7),
                interestingness=random.uniform(0.6, 0.85)
            ))

        return ideas

    def summarize_session(self, session: BrainstormSession) -> str:
        """Get human-readable summary of brainstorm session"""

        lines = [
            "=" * 80,
            "BRAINSTORMING SESSION SUMMARY",
            "=" * 80,
            "",
            f"Topic: {session.topic}",
            f"Domain: {session.domain}",
            f"Duration: {session.duration_seconds:.1f}s",
            f"Strategies: {', '.join(s.value for s in session.strategies_used)}",
            f"Total Ideas: {session.total_ideas}",
            "",
            "TOP IDEAS:",
            ""
        ]

        # Show top 10 ideas
        for i, idea in enumerate(session.ideas[:10], 1):
            lines.append(f"{i}. [{idea.strategy.value}] {idea.idea}")
            lines.append(f"   Novelty: {idea.novelty_score:.2f} | "
                        f"Feasibility: {idea.feasibility_score:.2f} | "
                        f"Interest: {idea.interestingness:.2f}")
            lines.append(f"   {idea.reasoning}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
