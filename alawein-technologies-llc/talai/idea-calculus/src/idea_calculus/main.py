#!/usr/bin/env python3
"""
IdeaCalculus - Mathematical Framework for Ideas

Formal calculus operations on ideas:
- Derivative: d/dt(idea) = rate of novelty decay
- Integral: ∫(ideas) = synthesis/accumulation
- Limit: lim(idea→x) = ultimate implication
- Composition: f(g(idea)) = nested concepts
- Transform: Fourier/Laplace for idea spaces

Usage:
    python calculus.py derivative --idea "deep learning" --variable "time"
    python calculus.py integrate --ideas "ML,stats,optimization" --output synthesis.json
    python calculus.py limit --idea "AI capabilities" --approach "infinity"
    python calculus.py compose --ideas "quantum,ML" --output composition.json
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable


@dataclass
class Idea:
    """Mathematical representation of an idea"""
    name: str
    novelty: float  # 0-1, decreases over time
    impact: float  # 0-1, area under curve
    complexity: float  # 0-1, derivative magnitude
    domain: str
    timestamp: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class IdeaDerivative:
    """Derivative of an idea with respect to a variable"""
    original_idea: str
    variable: str  # "time" | "knowledge" | "resources"
    rate_of_change: float
    interpretation: str
    future_projection: Dict[str, float]  # variable_value -> predicted_novelty


@dataclass
class IdeaSynthesis:
    """Integration/synthesis of multiple ideas"""
    component_ideas: List[str]
    synthesized_idea: str
    emergence_score: float  # How much > sum of parts (0-1)
    synergy_effects: List[str]
    applications: List[str]


@dataclass
class IdeaLimit:
    """Limit of an idea as variable approaches value"""
    original_idea: str
    variable: str
    approach_value: str  # "infinity" | "zero" | number
    limit_value: str
    interpretation: str
    paradoxes: List[str]


@dataclass
class IdeaComposition:
    """Composition of ideas f(g(x))"""
    outer_idea: str
    inner_idea: str
    composition: str
    emergent_properties: List[str]
    novel_applications: List[str]


class IdeaCalculusEngine:
    """Core calculus engine for idea operations"""

    def __init__(self):
        # Novelty decay models
        self.decay_models = {
            'exponential': lambda t, a: a * math.exp(-0.1 * t),
            'power_law': lambda t, a: a / (t + 1),
            'sigmoid': lambda t, a: a * (1 / (1 + math.exp(0.5 * (t - 10)))),
        }

        # Domain knowledge base
        self.domains = {
            'ai': ['deep learning', 'neural networks', 'transformers', 'reinforcement learning'],
            'physics': ['quantum mechanics', 'relativity', 'thermodynamics', 'particle physics'],
            'biology': ['genetics', 'evolution', 'molecular biology', 'ecology'],
            'math': ['calculus', 'algebra', 'topology', 'number theory'],
            'economics': ['game theory', 'market dynamics', 'behavioral economics'],
            'chemistry': ['organic chemistry', 'catalysis', 'materials science'],
        }

    def derivative(self, idea_name: str, variable: str = "time") -> IdeaDerivative:
        """
        Calculate derivative of idea with respect to variable

        d/dt(novelty) = rate at which idea becomes commonplace
        d/dk(complexity) = how complexity changes with knowledge
        """

        # Estimate initial novelty (simplified - would use real data)
        initial_novelty = self._estimate_novelty(idea_name)

        # Calculate rate of change
        if variable == "time":
            # Novelty decays over time
            rate = -0.1 * initial_novelty  # Exponential decay coefficient

            interpretation = (
                f"The novelty of '{idea_name}' is decreasing at rate {abs(rate):.3f}/year. "
                f"In 5 years, novelty will be ~{initial_novelty * math.exp(-0.1 * 5):.2f}."
            )

            # Project future
            future = {
                str(t): initial_novelty * math.exp(-0.1 * t)
                for t in [1, 3, 5, 10, 20]
            }

        elif variable == "knowledge":
            # Complexity increases with knowledge
            rate = 0.05 * initial_novelty

            interpretation = (
                f"The complexity of '{idea_name}' increases at rate {rate:.3f} "
                f"per unit of new knowledge. Mastery requires deep expertise."
            )

            future = {
                str(k): min(1.0, initial_novelty + 0.05 * k)
                for k in [10, 50, 100, 500, 1000]
            }

        elif variable == "resources":
            # Impact scales with resources
            rate = 0.03

            interpretation = (
                f"Impact of '{idea_name}' grows at rate {rate:.3f} "
                f"per resource unit invested. Strong returns to scale."
            )

            future = {
                f"${k}k": min(1.0, initial_novelty * (1 + 0.03 * k))
                for k in [10, 100, 1000, 10000]
            }
        else:
            rate = 0.0
            interpretation = f"Unknown variable: {variable}"
            future = {}

        return IdeaDerivative(
            original_idea=idea_name,
            variable=variable,
            rate_of_change=round(rate, 4),
            interpretation=interpretation,
            future_projection=future
        )

    def integrate(self, idea_names: List[str]) -> IdeaSynthesis:
        """
        Integrate/synthesize multiple ideas

        ∫(idea1, idea2, ..., ideaN) = synthesized concept
        """

        if len(idea_names) < 2:
            raise ValueError("Need at least 2 ideas to synthesize")

        # Generate synthesized idea name
        synthesized = self._synthesize_name(idea_names)

        # Calculate emergence (how much > sum of parts)
        component_novelties = [self._estimate_novelty(idea) for idea in idea_names]
        sum_novelty = sum(component_novelties) / len(component_novelties)

        # Synthesis creates emergence
        emergence = min(1.0, sum_novelty * 1.3 + random.uniform(0.1, 0.3))
        emergence_score = emergence - sum_novelty

        # Identify synergy effects
        synergies = self._identify_synergies(idea_names)

        # Generate applications
        applications = self._generate_applications(idea_names, synthesized)

        return IdeaSynthesis(
            component_ideas=idea_names,
            synthesized_idea=synthesized,
            emergence_score=round(emergence_score, 3),
            synergy_effects=synergies,
            applications=applications
        )

    def limit(self, idea_name: str, variable: str, approach: str) -> IdeaLimit:
        """
        Calculate limit of idea as variable approaches value

        lim(idea) as time → ∞
        lim(idea) as knowledge → 0
        """

        if approach == "infinity":
            if variable == "time":
                limit_val = "Complete commoditization - idea becomes universal knowledge"
                interpretation = (
                    f"As time approaches infinity, '{idea_name}' becomes completely "
                    f"integrated into common knowledge. Novelty approaches zero. "
                    f"All potential applications are discovered and exploited."
                )
                paradoxes = [
                    "Zeno's paradox: Idea never truly reaches zero novelty",
                    "Ship of Theseus: Is it still the same idea after infinite iteration?",
                    "Heat death: All ideas eventually reach equilibrium state"
                ]

            elif variable == "knowledge":
                limit_val = "Perfect understanding - complete mastery of concept"
                interpretation = (
                    f"With infinite knowledge, '{idea_name}' is perfectly understood. "
                    f"All implications, edge cases, and applications are known. "
                    f"No surprises remain."
                )
                paradoxes = [
                    "Gödel's incompleteness: Some truths remain unprovable",
                    "Uncertainty principle: Perfect knowledge impossible",
                    "Halting problem: Can't predict all consequences"
                ]

            elif variable == "resources":
                limit_val = "Maximum impact - idea reaches full potential"
                interpretation = (
                    f"With unlimited resources, '{idea_name}' achieves maximum impact. "
                    f"Every possible application is realized. Saturation point."
                )
                paradoxes = [
                    "Diminishing returns: Impact grows sublinearly",
                    "Resource curse: Too much funding can harm innovation",
                    "Coordination problems: Infinite resources, finite coordination"
                ]
            else:
                limit_val = "Unknown"
                interpretation = "Undefined variable"
                paradoxes = []

        elif approach == "zero":
            if variable == "time":
                limit_val = "Initial discovery - pure novelty"
                interpretation = (
                    f"At time zero (first discovery), '{idea_name}' has maximum novelty. "
                    f"This is the Eureka moment, before any diffusion."
                )
                paradoxes = [
                    "Invention vs discovery: Was it always there?",
                    "Simultaneous discovery: Multiple zero points?",
                    "Precursors: Is there truly a first moment?"
                ]

            elif variable == "knowledge":
                limit_val = "Ignorance - idea unknown/unknowable"
                interpretation = (
                    f"With zero knowledge, '{idea_name}' doesn't exist yet. "
                    f"This is pre-discovery state."
                )
                paradoxes = [
                    "Known unknowns: Can we know what we don't know?",
                    "Platonic forms: Do ideas exist before discovery?",
                    "Observer effect: Does thinking about it create it?"
                ]
            else:
                limit_val = "Theoretical existence only"
                interpretation = f"With zero resources, '{idea_name}' remains purely theoretical"
                paradoxes = []
        else:
            limit_val = f"Approaching {approach}"
            interpretation = f"Limit of '{idea_name}' as {variable} → {approach}"
            paradoxes = []

        return IdeaLimit(
            original_idea=idea_name,
            variable=variable,
            approach_value=approach,
            limit_value=limit_val,
            interpretation=interpretation,
            paradoxes=paradoxes
        )

    def compose(self, outer_idea: str, inner_idea: str) -> IdeaComposition:
        """
        Compose ideas: f(g(x))

        Apply outer idea to result of inner idea
        """

        # Generate composition
        composition = f"{outer_idea}({inner_idea})"

        # Identify emergent properties
        emergent = self._find_emergent_properties(outer_idea, inner_idea)

        # Generate novel applications
        applications = self._compose_applications(outer_idea, inner_idea)

        return IdeaComposition(
            outer_idea=outer_idea,
            inner_idea=inner_idea,
            composition=composition,
            emergent_properties=emergent,
            novel_applications=applications
        )

    def _estimate_novelty(self, idea_name: str) -> float:
        """Estimate current novelty of an idea (0-1)"""
        # Simplified heuristic - in production would use real data

        # Check if idea is in common knowledge domains
        idea_lower = idea_name.lower()

        # Very common ideas (low novelty)
        common = ['addition', 'subtraction', 'wheel', 'fire', 'writing']
        if any(c in idea_lower for c in common):
            return random.uniform(0.05, 0.15)

        # Emerging ideas (high novelty)
        emerging = ['quantum', 'transformer', 'crispr', 'gpt', 'diffusion model']
        if any(e in idea_lower for e in emerging):
            return random.uniform(0.7, 0.95)

        # Established but evolving (medium novelty)
        return random.uniform(0.3, 0.6)

    def _synthesize_name(self, ideas: List[str]) -> str:
        """Generate name for synthesized idea"""
        if len(ideas) == 2:
            return f"{ideas[0]}-powered {ideas[1]}"
        else:
            return f"Hybrid {'/'.join(ideas[:2])} system"

    def _identify_synergies(self, ideas: List[str]) -> List[str]:
        """Identify synergy effects from combining ideas"""
        synergies = []

        # Cross-domain combinations create synergy
        domains_involved = set()
        for idea in ideas:
            for domain, concepts in self.domains.items():
                if any(concept in idea.lower() for concept in concepts):
                    domains_involved.add(domain)

        if len(domains_involved) >= 2:
            synergies.append(f"Cross-domain synergy: {' + '.join(domains_involved)}")

        # Complementary capabilities
        synergies.append(f"Complementary strengths from {len(ideas)} components")

        # Emergent behaviors
        synergies.append("Non-linear interactions between components")

        return synergies

    def _generate_applications(self, ideas: List[str], synthesis: str) -> List[str]:
        """Generate potential applications of synthesized idea"""
        applications = []

        # Domain-specific applications
        applications.append(f"Novel approach to {ideas[0]} using insights from {ideas[1] if len(ideas) > 1 else 'other domains'}")

        # Solve previously unsolvable problems
        applications.append(f"Address limitations of standalone {ideas[0]}")

        # New research directions
        applications.append(f"Open new research directions in {synthesis}")

        return applications

    def _find_emergent_properties(self, outer: str, inner: str) -> List[str]:
        """Find emergent properties of composition"""
        properties = []

        properties.append(f"Applies {outer} methodology to {inner} domain")
        properties.append(f"Transforms {inner} outputs using {outer} framework")
        properties.append(f"Creates feedback loop between {outer} and {inner}")

        return properties

    def _compose_applications(self, outer: str, inner: str) -> List[str]:
        """Generate applications of composed ideas"""
        applications = []

        applications.append(f"Use {outer} to optimize {inner} performance")
        applications.append(f"Apply {inner} solutions to {outer} problems")
        applications.append(f"Meta-analysis of {inner} through {outer} lens")

        return applications


def main():
    parser = argparse.ArgumentParser(
        description="IdeaCalculus - Mathematical Framework for Ideas"
    )
    subparsers = parser.add_subparsers(dest='command', help='Calculus operations')

    # Derivative
    deriv_parser = subparsers.add_parser('derivative', help='Calculate idea derivative')
    deriv_parser.add_argument('--idea', required=True, help='Idea name')
    deriv_parser.add_argument('--variable', default='time', choices=['time', 'knowledge', 'resources'])
    deriv_parser.add_argument('--output', help='Output JSON file')

    # Integrate
    integrate_parser = subparsers.add_parser('integrate', help='Synthesize ideas')
    integrate_parser.add_argument('--ideas', required=True, help='Comma-separated idea names')
    integrate_parser.add_argument('--output', help='Output JSON file')

    # Limit
    limit_parser = subparsers.add_parser('limit', help='Calculate idea limit')
    limit_parser.add_argument('--idea', required=True, help='Idea name')
    limit_parser.add_argument('--variable', default='time', choices=['time', 'knowledge', 'resources'])
    limit_parser.add_argument('--approach', default='infinity', help='Value to approach (infinity, zero, number)')
    limit_parser.add_argument('--output', help='Output JSON file')

    # Compose
    compose_parser = subparsers.add_parser('compose', help='Compose ideas f(g(x))')
    compose_parser.add_argument('--outer', required=True, help='Outer idea')
    compose_parser.add_argument('--inner', required=True, help='Inner idea')
    compose_parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = IdeaCalculusEngine()

    if args.command == 'derivative':
        result = engine.derivative(args.idea, args.variable)

        print(f"\n{'='*70}")
        print(f"DERIVATIVE: d({args.idea})/d({args.variable})")
        print(f"{'='*70}\n")
        print(f"Rate of change: {result.rate_of_change:.4f}")
        print(f"\n{result.interpretation}\n")
        print(f"Future projections:")
        for var_val, novelty in result.future_projection.items():
            print(f"  {var_val:10s} → novelty: {novelty:.3f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'integrate':
        ideas = [i.strip() for i in args.ideas.split(',')]
        result = engine.integrate(ideas)

        print(f"\n{'='*70}")
        print(f"INTEGRATION: ∫({', '.join(ideas)})")
        print(f"{'='*70}\n")
        print(f"Synthesized idea: {result.synthesized_idea}")
        print(f"Emergence score: {result.emergence_score:.3f}")
        print(f"\nSynergy effects:")
        for synergy in result.synergy_effects:
            print(f"  - {synergy}")
        print(f"\nApplications:")
        for app in result.applications:
            print(f"  - {app}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'limit':
        result = engine.limit(args.idea, args.variable, args.approach)

        print(f"\n{'='*70}")
        print(f"LIMIT: lim({args.idea}) as {args.variable} → {args.approach}")
        print(f"{'='*70}\n")
        print(f"Limit value: {result.limit_value}\n")
        print(f"{result.interpretation}\n")

        if result.paradoxes:
            print(f"Philosophical paradoxes:")
            for paradox in result.paradoxes:
                print(f"  - {paradox}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'compose':
        result = engine.compose(args.outer, args.inner)

        print(f"\n{'='*70}")
        print(f"COMPOSITION: {args.outer}({args.inner})")
        print(f"{'='*70}\n")
        print(f"Result: {result.composition}\n")
        print(f"Emergent properties:")
        for prop in result.emergent_properties:
            print(f"  - {prop}")
        print(f"\nNovel applications:")
        for app in result.novel_applications:
            print(f"  - {app}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
