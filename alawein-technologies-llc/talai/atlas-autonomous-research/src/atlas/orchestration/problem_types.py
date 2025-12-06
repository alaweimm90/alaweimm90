"""
ORCHEX Orchestration - Problem Types

Defines different types of research problems and their characteristics.
Enables adaptive workflow configuration based on user intent.

Cycle 21-22: Adaptive Orchestration
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class ProblemType(str, Enum):
    """Types of research problems ORCHEX can handle"""

    # Discovery & Novelty
    NOVELTY_CHECK = "novelty_check"  # "Is this method novel? Find similar work"
    DISCOVERY = "discovery"  # "Find new approaches/methods for X"
    IDEATION = "ideation"  # "Brainstorm ideas for X"

    # Validation & Testing
    VALIDATION = "validation"  # "Validate this hypothesis/method"
    BENCHMARK = "benchmark"  # "Benchmark this algorithm/system"
    COMPARISON = "comparison"  # "Compare X and Y"

    # Optimization & Improvement
    OPTIMIZATION = "optimization"  # "Optimize this algorithm/system"
    PARAMETER_TUNING = "parameter_tuning"  # "Find best parameters for X"

    # Analysis & Understanding
    ANALYSIS = "analysis"  # "Analyze this phenomenon/system"
    EXPLANATION = "explanation"  # "Explain why X works/fails"

    # Synthesis & Creation
    SYNTHESIS = "synthesis"  # "Combine X and Y in novel ways"
    DESIGN = "design"  # "Design a system/algorithm for X"

    # General research (fallback)
    GENERAL_RESEARCH = "general_research"


@dataclass
class ProblemCharacteristics:
    """Characteristics and requirements for a problem type"""

    problem_type: ProblemType

    # Pipeline configuration
    requires_literature_search: bool = True
    requires_hypothesis_generation: bool = True
    requires_validation: bool = True
    requires_experimentation: bool = True
    requires_paper: bool = True

    # Agent preferences (role -> priority score 0-1)
    preferred_agents: Dict[str, float] = None

    # Stage emphasis (how much to focus on each stage, 0-1)
    hypothesis_emphasis: float = 0.5
    validation_emphasis: float = 0.5
    experiment_emphasis: float = 0.5
    paper_emphasis: float = 0.5

    # Special features
    enable_brainstorming: bool = False
    enable_cross_domain_search: bool = False
    enable_novelty_scoring: bool = False
    parallel_hypothesis_generation: bool = False

    # Meta-learning context
    meta_learning_contexts: List[str] = None

    # Resource allocation
    max_hypotheses: int = 5
    max_validation_rounds: int = 3
    max_experiment_variants: int = 3

    def __post_init__(self):
        if self.preferred_agents is None:
            self.preferred_agents = {}
        if self.meta_learning_contexts is None:
            self.meta_learning_contexts = []


# Problem type configurations
PROBLEM_TYPE_CONFIGS = {
    ProblemType.NOVELTY_CHECK: ProblemCharacteristics(
        problem_type=ProblemType.NOVELTY_CHECK,
        hypothesis_emphasis=0.3,
        validation_emphasis=0.8,  # High validation for novelty
        experiment_emphasis=0.4,
        paper_emphasis=0.6,
        enable_novelty_scoring=True,
        enable_cross_domain_search=True,
        preferred_agents={
            "research": 0.9,
            "criticism": 0.8,
            "interrogation": 0.7,
            "peer_review": 0.8,
        },
        max_hypotheses=3,
        max_validation_rounds=5,  # Thorough validation
        meta_learning_contexts=["novelty_detection", "literature_review"],
    ),

    ProblemType.DISCOVERY: ProblemCharacteristics(
        problem_type=ProblemType.DISCOVERY,
        hypothesis_emphasis=0.9,  # High hypothesis generation
        validation_emphasis=0.6,
        experiment_emphasis=0.5,
        paper_emphasis=0.5,
        enable_brainstorming=True,
        enable_cross_domain_search=True,
        parallel_hypothesis_generation=True,
        preferred_agents={
            "research": 0.9,
            "ideation": 0.95,  # Creative Carla
            "criticism": 0.5,
        },
        max_hypotheses=10,  # Generate many ideas
        max_validation_rounds=2,
        meta_learning_contexts=["discovery", "creativity"],
    ),

    ProblemType.IDEATION: ProblemCharacteristics(
        problem_type=ProblemType.IDEATION,
        requires_experimentation=False,  # Just ideas, no experiments
        requires_paper=False,
        hypothesis_emphasis=0.95,  # Almost all focus on ideation
        validation_emphasis=0.3,
        experiment_emphasis=0.0,
        paper_emphasis=0.0,
        enable_brainstorming=True,
        enable_cross_domain_search=True,
        parallel_hypothesis_generation=True,
        preferred_agents={
            "research": 0.7,
            "ideation": 1.0,  # Creative Carla leads
            "criticism": 0.2,  # Low criticism for ideation
        },
        max_hypotheses=20,  # Many ideas!
        max_validation_rounds=1,
        meta_learning_contexts=["ideation", "creativity", "brainstorming"],
    ),

    ProblemType.VALIDATION: ProblemCharacteristics(
        problem_type=ProblemType.VALIDATION,
        requires_hypothesis_generation=False,  # User provides hypothesis
        hypothesis_emphasis=0.1,
        validation_emphasis=0.95,  # All focus on validation
        experiment_emphasis=0.7,
        paper_emphasis=0.6,
        preferred_agents={
            "criticism": 0.9,
            "interrogation": 0.9,
            "risk_assessment": 0.8,
            "peer_review": 0.8,
        },
        max_validation_rounds=5,  # Very thorough
        meta_learning_contexts=["validation", "hypothesis_testing"],
    ),

    ProblemType.BENCHMARK: ProblemCharacteristics(
        problem_type=ProblemType.BENCHMARK,
        hypothesis_emphasis=0.3,
        validation_emphasis=0.4,
        experiment_emphasis=0.95,  # All focus on experiments
        paper_emphasis=0.7,
        preferred_agents={
            "experiment_design": 0.9,
            "data_collection": 0.95,  # Lab Rat Larry
            "criticism": 0.6,
        },
        max_experiment_variants=5,  # Many benchmark variants
        meta_learning_contexts=["benchmarking", "experimentation"],
    ),

    ProblemType.COMPARISON: ProblemCharacteristics(
        problem_type=ProblemType.COMPARISON,
        hypothesis_emphasis=0.4,
        validation_emphasis=0.6,
        experiment_emphasis=0.9,
        paper_emphasis=0.8,
        preferred_agents={
            "experiment_design": 0.9,
            "criticism": 0.7,
            "peer_review": 0.8,
        },
        max_hypotheses=2,  # One per comparison target
        max_experiment_variants=4,
        meta_learning_contexts=["comparison", "benchmarking"],
    ),

    ProblemType.OPTIMIZATION: ProblemCharacteristics(
        problem_type=ProblemType.OPTIMIZATION,
        hypothesis_emphasis=0.6,
        validation_emphasis=0.5,
        experiment_emphasis=0.95,  # Heavy experimentation
        paper_emphasis=0.6,
        enable_brainstorming=True,
        preferred_agents={
            "research": 0.7,
            "ideation": 0.7,
            "experiment_design": 0.9,
            "data_collection": 0.8,
        },
        max_hypotheses=8,
        max_experiment_variants=6,
        meta_learning_contexts=["optimization", "experimentation"],
    ),

    ProblemType.PARAMETER_TUNING: ProblemCharacteristics(
        problem_type=ProblemType.PARAMETER_TUNING,
        hypothesis_emphasis=0.5,
        validation_emphasis=0.4,
        experiment_emphasis=0.95,
        paper_emphasis=0.5,
        preferred_agents={
            "experiment_design": 0.95,
            "data_collection": 0.9,
        },
        max_experiment_variants=10,  # Many parameter combinations
        meta_learning_contexts=["parameter_tuning", "optimization"],
    ),

    ProblemType.ANALYSIS: ProblemCharacteristics(
        problem_type=ProblemType.ANALYSIS,
        hypothesis_emphasis=0.7,
        validation_emphasis=0.7,
        experiment_emphasis=0.6,
        paper_emphasis=0.8,
        preferred_agents={
            "research": 0.8,
            "criticism": 0.7,
            "interrogation": 0.8,
            "peer_review": 0.9,
        },
        meta_learning_contexts=["analysis", "understanding"],
    ),

    ProblemType.EXPLANATION: ProblemCharacteristics(
        problem_type=ProblemType.EXPLANATION,
        hypothesis_emphasis=0.8,
        validation_emphasis=0.7,
        experiment_emphasis=0.5,
        paper_emphasis=0.9,  # Heavy paper emphasis
        enable_cross_domain_search=True,
        preferred_agents={
            "research": 0.9,
            "criticism": 0.6,
            "peer_review": 0.9,
            "editing": 0.8,  # Detail Dana
        },
        meta_learning_contexts=["explanation", "understanding"],
    ),

    ProblemType.SYNTHESIS: ProblemCharacteristics(
        problem_type=ProblemType.SYNTHESIS,
        hypothesis_emphasis=0.9,
        validation_emphasis=0.6,
        experiment_emphasis=0.7,
        paper_emphasis=0.7,
        enable_brainstorming=True,
        enable_cross_domain_search=True,
        parallel_hypothesis_generation=True,
        preferred_agents={
            "research": 0.8,
            "ideation": 0.95,
            "experiment_design": 0.7,
        },
        max_hypotheses=8,
        meta_learning_contexts=["synthesis", "creativity", "integration"],
    ),

    ProblemType.DESIGN: ProblemCharacteristics(
        problem_type=ProblemType.DESIGN,
        hypothesis_emphasis=0.8,
        validation_emphasis=0.6,
        experiment_emphasis=0.8,
        paper_emphasis=0.6,
        enable_brainstorming=True,
        preferred_agents={
            "research": 0.7,
            "ideation": 0.9,
            "experiment_design": 0.9,
            "criticism": 0.6,
        },
        max_hypotheses=6,
        meta_learning_contexts=["design", "engineering"],
    ),

    ProblemType.GENERAL_RESEARCH: ProblemCharacteristics(
        problem_type=ProblemType.GENERAL_RESEARCH,
        # Balanced defaults
        hypothesis_emphasis=0.5,
        validation_emphasis=0.5,
        experiment_emphasis=0.5,
        paper_emphasis=0.5,
        preferred_agents={},
        max_hypotheses=5,
        max_validation_rounds=3,
        max_experiment_variants=3,
        meta_learning_contexts=["general"],
    ),
}


def get_problem_config(problem_type: ProblemType) -> ProblemCharacteristics:
    """Get configuration for a problem type"""
    return PROBLEM_TYPE_CONFIGS.get(problem_type, PROBLEM_TYPE_CONFIGS[ProblemType.GENERAL_RESEARCH])


# Intent detection keywords
INTENT_KEYWORDS = {
    ProblemType.NOVELTY_CHECK: [
        "novel", "novelty", "new", "original", "unique", "similar work",
        "existing methods", "prior art", "is this new", "has anyone",
        "already exists", "been done before", "state of the art"
    ],

    ProblemType.DISCOVERY: [
        "find new", "discover", "explore", "what are possible", "alternatives",
        "other approaches", "different methods", "new ways", "innovative"
    ],

    ProblemType.IDEATION: [
        "brainstorm", "ideas for", "generate ideas", "think of ways",
        "come up with", "creative solutions", "possibilities"
    ],

    ProblemType.VALIDATION: [
        "validate", "verify", "test hypothesis", "prove", "confirm",
        "is this true", "does this work", "check if"
    ],

    ProblemType.BENCHMARK: [
        "benchmark", "performance", "measure", "evaluate performance",
        "how fast", "throughput", "latency", "speed test", "compare performance"
    ],

    ProblemType.COMPARISON: [
        "compare", "versus", "vs", "difference between", "better than",
        "which is", "advantages of", "disadvantages"
    ],

    ProblemType.OPTIMIZATION: [
        "optimize", "improve", "make better", "enhance", "increase performance",
        "faster", "more efficient", "reduce", "minimize", "maximize"
    ],

    ProblemType.PARAMETER_TUNING: [
        "tune", "hyperparameter", "best parameters", "optimal settings",
        "configuration", "parameter search", "grid search"
    ],

    ProblemType.ANALYSIS: [
        "analyze", "analysis", "understand", "investigate", "examine",
        "study", "look into", "what causes", "why does"
    ],

    ProblemType.EXPLANATION: [
        "explain", "why", "how does", "reasoning", "rationale",
        "mechanism", "what makes", "interpretation"
    ],

    ProblemType.SYNTHESIS: [
        "combine", "integrate", "merge", "synthesize", "unify",
        "bring together", "fusion", "hybrid"
    ],

    ProblemType.DESIGN: [
        "design", "create", "build", "develop", "construct",
        "engineer", "architect", "implement"
    ],
}
