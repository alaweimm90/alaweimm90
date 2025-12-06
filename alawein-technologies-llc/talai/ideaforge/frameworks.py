"""
Thinking Frameworks for IdeaForge

Implements all 15 thinking frameworks:
1. First Principles
2. Inversion
3. Second-Order Thinking
4. Lateral Thinking
5. SCAMPER
6. TRIZ (Inventive Principles)
7. Biomimicry
8. Cross-Domain Analogy
9. Constraint Relaxation
10. Exaptation (Repurposing)
11. Conceptual Blending
12. Oblique Strategies
13. Six Thinking Hats
14. Morphological Analysis
15. Future Backcasting
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class KnowledgeGraph:
    """Represents extracted knowledge from input"""
    concepts: List[str]
    relationships: List[tuple]
    problems: List[str]
    opportunities: List[str]
    constraints: List[str]
    domain: str
    source: str


class ThinkingFramework:
    """Base class for all thinking frameworks"""

    def __init__(self, name: str):
        self.name = name

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Analyze knowledge graph and return insights"""
        raise NotImplementedError


# ========== Framework 1: First Principles ==========
class FirstPrinciplesFramework(ThinkingFramework):
    """Break down problem to fundamental truths, rebuild from scratch"""

    def __init__(self):
        super().__init__("first_principles")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "fundamental_truths": [],
            "assumptions_challenged": [],
            "rebuilt_approaches": []
        }

        for concept in graph.concepts:
            insights["fundamental_truths"].append(
                f"Core: What is {concept} really trying to achieve?"
            )
            insights["assumptions_challenged"].append(
                f"Why does {concept} have to work the current way?"
            )
            insights["rebuilt_approaches"].append(
                f"Build {concept} from physics/math/biology fundamentals"
            )

        return insights


# ========== Framework 2: Inversion ==========
class InversionFramework(ThinkingFramework):
    """Invert the problem - solve backwards, do the opposite"""

    def __init__(self):
        super().__init__("inversion")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "inverted_problems": [],
            "anti_solutions": [],
            "backward_chains": []
        }

        for problem in graph.problems:
            insights["inverted_problems"].append(
                f"Instead of solving {problem}, what if we embraced it?"
            )
            insights["anti_solutions"].append(
                f"What would make {problem} worse? Invert that."
            )

        return insights


# ========== Framework 3: Second-Order Thinking ==========
class SecondOrderThinkingFramework(ThinkingFramework):
    """Think through consequences of consequences"""

    def __init__(self):
        super().__init__("second_order")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "second_order_effects": [],
            "nth_order_cascades": [],
            "unintended_consequences": []
        }

        for opportunity in graph.opportunities:
            insights["second_order_effects"].append(
                f"If we pursue {opportunity}, what happens next?"
            )
            insights["nth_order_cascades"].append(
                f"And then what? (3rd, 4th, 5th order effects)"
            )

        return insights


# ========== Framework 4: Lateral Thinking ==========
class LateralThinkingFramework(ThinkingFramework):
    """Random stimuli, provocations, po-statements"""

    def __init__(self):
        super().__init__("lateral")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        random_words = ["purple", "quantum", "backwards", "liquid", "fractal", "musical"]

        insights = {
            "framework": self.name,
            "provocations": [],
            "random_stimuli": [],
            "po_statements": []
        }

        for concept in graph.concepts[:3]:
            random_word = random_words[hash(concept) % len(random_words)]
            insights["provocations"].append(
                f"What if {concept} was {random_word}?"
            )
            insights["po_statements"].append(
                f"PO: {concept} happens before it starts"
            )

        return insights


# ========== Framework 5: SCAMPER ==========
class SCAMPERFramework(ThinkingFramework):
    """Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse"""

    def __init__(self):
        super().__init__("scamper")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "substitute": [],
            "combine": [],
            "adapt": [],
            "modify": [],
            "put_to_other_uses": [],
            "eliminate": [],
            "reverse": []
        }

        for concept in graph.concepts[:3]:
            insights["substitute"].append(f"What if we replaced {concept} with...?")
            insights["combine"].append(f"Merge {concept} with another concept")
            insights["adapt"].append(f"Adapt {concept} from another industry")
            insights["modify"].append(f"Change {concept}'s scale, speed, or form")
            insights["put_to_other_uses"].append(f"Use {concept} for different purpose")
            insights["eliminate"].append(f"What if {concept} didn't exist?")
            insights["reverse"].append(f"Flip {concept} inside-out or backwards")

        return insights


# ========== Framework 6: TRIZ ==========
class TRIZFramework(ThinkingFramework):
    """40 Inventive Principles from TRIZ methodology"""

    def __init__(self):
        super().__init__("triz")
        self.principles = [
            "Segmentation", "Taking out", "Local quality", "Asymmetry",
            "Merging", "Universality", "Nested doll", "Anti-weight",
            "Preliminary anti-action", "Preliminary action"
        ]

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "inventive_principles": [],
            "contradictions": []
        }

        for i, problem in enumerate(graph.problems[:5]):
            principle = self.principles[i % len(self.principles)]
            insights["inventive_principles"].append(
                f"Apply '{principle}' to {problem}"
            )

        return insights


# ========== Framework 7: Biomimicry ==========
class BiomimicryFramework(ThinkingFramework):
    """Look to nature's 3.8 billion years of R&D"""

    def __init__(self):
        super().__init__("biomimicry")
        self.nature_examples = [
            "ant colonies", "spider webs", "trees", "coral reefs",
            "bird flocks", "immune systems", "DNA", "mycelium networks"
        ]

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "nature_analogues": [],
            "biological_strategies": []
        }

        for i, concept in enumerate(graph.concepts[:4]):
            nature = self.nature_examples[i % len(self.nature_examples)]
            insights["nature_analogues"].append(
                f"How do {nature} solve {concept}?"
            )

        return insights


# ========== Framework 8: Cross-Domain Analogy ==========
class CrossDomainAnalogyFramework(ThinkingFramework):
    """Map solutions from completely different domains"""

    def __init__(self):
        super().__init__("cross_domain_analogy")
        self.domains = [
            "music composition", "professional sports", "cooking",
            "urban planning", "video game design", "film editing",
            "gardening", "chess strategy", "fashion design"
        ]

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "analogies": [],
            "cross_domain_insights": []
        }

        for i, concept in enumerate(graph.concepts[:3]):
            for j in range(2):  # 2 domains per concept
                domain = self.domains[(i + j) % len(self.domains)]
                insights["analogies"].append(
                    f"Problem '{concept}' in {graph.domain} is like {domain}..."
                )

        return insights


# ========== Framework 9: Constraint Relaxation ==========
class ConstraintRelaxationFramework(ThinkingFramework):
    """What if constraints didn't exist? What becomes possible?"""

    def __init__(self):
        super().__init__("constraint_relaxation")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "relaxed_constraints": [],
            "unconstrained_solutions": [],
            "partial_relaxations": []
        }

        for constraint in graph.constraints:
            insights["relaxed_constraints"].append(
                f"What if {constraint} wasn't a limitation?"
            )
            insights["partial_relaxations"].append(
                f"What if we could reduce {constraint} by 90%?"
            )

        # Add implicit constraints
        insights["unconstrained_solutions"].append(
            "What if we had unlimited compute/time/money?"
        )

        return insights


# ========== Framework 10: Exaptation ==========
class ExaptationFramework(ThinkingFramework):
    """Repurpose existing solutions for new problems"""

    def __init__(self):
        super().__init__("exaptation")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "repurposed_solutions": [],
            "unexpected_applications": []
        }

        for opportunity in graph.opportunities:
            insights["repurposed_solutions"].append(
                f"What existing tool could we repurpose for {opportunity}?"
            )

        return insights


# ========== Framework 11: Conceptual Blending ==========
class ConceptualBlendingFramework(ThinkingFramework):
    """Blend concepts to create novel hybrids"""

    def __init__(self):
        super().__init__("conceptual_blending")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "blends": [],
            "chimeras": []
        }

        concepts = graph.concepts
        # Blend every pair
        for i in range(min(3, len(concepts))):
            for j in range(i + 1, min(4, len(concepts))):
                insights["blends"].append(
                    f"Blend: {concepts[i]} + {concepts[j]} = ?"
                )

        return insights


# ========== Framework 12: Oblique Strategies ==========
class ObliqueStrategiesFramework(ThinkingFramework):
    """Brian Eno's oblique strategies for creative blocks"""

    def __init__(self):
        super().__init__("oblique")
        self.strategies = [
            "Use an old idea",
            "State the problem in words as clearly as possible",
            "What would your closest friend do?",
            "Do the washing up",
            "Turn it upside down",
            "Look at the order in which you do things",
            "What mistakes did you make last time?",
            "Take away the elements in order of apparent non-importance"
        ]

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "strategies_applied": []
        }

        for i in range(5):
            strategy = self.strategies[i % len(self.strategies)]
            insights["strategies_applied"].append(
                f"Strategy: {strategy}"
            )

        return insights


# ========== Framework 13: Six Thinking Hats ==========
class SixThinkingHatsFramework(ThinkingFramework):
    """Edward de Bono's six perspectives"""

    def __init__(self):
        super().__init__("six_hats")
        self.hats = {
            "white": "Facts and information",
            "red": "Emotions and feelings",
            "black": "Risks and negatives",
            "yellow": "Benefits and positives",
            "green": "Creativity and new ideas",
            "blue": "Process and overview"
        }

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "perspectives": {}
        }

        for color, description in self.hats.items():
            insights["perspectives"][color] = f"{description}: Applied to {graph.domain}"

        return insights


# ========== Framework 14: Morphological Analysis ==========
class MorphologicalAnalysisFramework(ThinkingFramework):
    """Break problem into dimensions, combine attributes"""

    def __init__(self):
        super().__init__("morphological")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "dimensions": [],
            "combinations": []
        }

        # Create dimensions from concepts
        if len(graph.concepts) >= 2:
            insights["dimensions"].append(f"Dimension 1: {graph.concepts[0]}")
            insights["dimensions"].append(f"Dimension 2: {graph.concepts[1]}")
            insights["combinations"].append(
                f"Combine attributes from both dimensions"
            )

        return insights


# ========== Framework 15: Future Backcasting ==========
class FutureBackcastingFramework(ThinkingFramework):
    """Start from desired future, work backwards"""

    def __init__(self):
        super().__init__("future_backcasting")

    async def analyze(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        insights = {
            "framework": self.name,
            "future_visions": [],
            "backward_steps": []
        }

        for opportunity in graph.opportunities:
            insights["future_visions"].append(
                f"Imagine {opportunity} is fully solved in 2030. What does that look like?"
            )
            insights["backward_steps"].append(
                "Work backwards: What had to happen in 2029? 2028? 2027?"
            )

        return insights


# ========== Framework Registry ==========
ALL_FRAMEWORKS = {
    "first_principles": FirstPrinciplesFramework,
    "inversion": InversionFramework,
    "second_order": SecondOrderThinkingFramework,
    "lateral": LateralThinkingFramework,
    "scamper": SCAMPERFramework,
    "triz": TRIZFramework,
    "biomimicry": BiomimicryFramework,
    "cross_domain_analogy": CrossDomainAnalogyFramework,
    "constraint_relaxation": ConstraintRelaxationFramework,
    "exaptation": ExaptationFramework,
    "conceptual_blending": ConceptualBlendingFramework,
    "oblique": ObliqueStrategiesFramework,
    "six_hats": SixThinkingHatsFramework,
    "morphological": MorphologicalAnalysisFramework,
    "future_backcasting": FutureBackcastingFramework,
}


def get_framework(name: str) -> ThinkingFramework:
    """Factory function to get framework by name"""
    if name not in ALL_FRAMEWORKS:
        raise ValueError(f"Unknown framework: {name}")
    return ALL_FRAMEWORKS[name]()


def get_all_frameworks() -> List[ThinkingFramework]:
    """Get instances of all frameworks"""
    return [cls() for cls in ALL_FRAMEWORKS.values()]
