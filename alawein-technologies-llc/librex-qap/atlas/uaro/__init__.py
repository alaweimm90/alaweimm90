"""
UARO - Universal Autonomous Reasoning Orchestrator

Problem-agnostic reasoning system with universal primitives.

Components:
- ReasoningPrimitives: Problem-agnostic operations
- UniversalSolver: Meta-algorithm that applies primitives
- ProblemClassifier: Recognize problem structure
- PrimitiveRegistry: Central registry of primitives
- ExplainabilityEngine: "Show Your Work" proof documents
- PrimitiveMarketplace: Share and monetize primitives

Cycles 27-32: Universal Reasoning System

Based on ideas from:
- Universal Problem Solving (Newell & Simon, 1972)
- General Problem Solver (GPS)
- Modern AI planning and search
- Constraint programming
- Logic programming
"""

__version__ = "0.1.0"

from uaro.reasoning_primitives import (
    # Abstract interfaces
    Problem,
    ReasoningPrimitive,

    # Decomposition primitives
    DivideAndConquer,
    HierarchicalDecomposition,

    # Search primitives
    BreadthFirstSearch,
    DepthFirstSearch,
    BestFirstSearch,
    BeamSearch,

    # Constraint primitives
    ConstraintPropagation,
    BacktrackingSearch,

    # Logic primitives
    ForwardChaining,
    BackwardChaining,

    # Optimization primitives
    LocalSearch,
    SimulatedAnnealing,

    # Registry
    PrimitiveRegistry,
)

from uaro.universal_solver import (
    UniversalSolver,
    ProblemClassifier,
    ReasoningStep,
    SolutionResult,
    ProblemSignature,
    solve_with_uaro,
)

from uaro.explainability import (
    ExplainabilityEngine,
    ProofDocument,
    explain_solution,
)

from uaro.marketplace import (
    PrimitiveMarketplace,
    PrimitiveListing,
    PrimitiveReview,
    UsageRecord,
    PricingModel,
    create_marketplace,
)

from uaro.atlas_integration import (
    ATLASUAROIntegration,
    ResearchTask,
    ResearchProblem,
    WorkflowExtractor,
    create_atlas_uaro_integration,
    solve_atlas_task,
)

__all__ = [
    # Interfaces
    "Problem",
    "ReasoningPrimitive",

    # Decomposition
    "DivideAndConquer",
    "HierarchicalDecomposition",

    # Search
    "BreadthFirstSearch",
    "DepthFirstSearch",
    "BestFirstSearch",
    "BeamSearch",

    # Constraints
    "ConstraintPropagation",
    "BacktrackingSearch",

    # Logic
    "ForwardChaining",
    "BackwardChaining",

    # Optimization
    "LocalSearch",
    "SimulatedAnnealing",

    # Registry
    "PrimitiveRegistry",

    # Solver
    "UniversalSolver",
    "ProblemClassifier",
    "ReasoningStep",
    "SolutionResult",
    "ProblemSignature",
    "solve_with_uaro",

    # Explainability
    "ExplainabilityEngine",
    "ProofDocument",
    "explain_solution",

    # Marketplace
    "PrimitiveMarketplace",
    "PrimitiveListing",
    "PrimitiveReview",
    "UsageRecord",
    "PricingModel",
    "create_marketplace",

    # ORCHEX Integration
    "ATLASUAROIntegration",
    "ResearchTask",
    "ResearchProblem",
    "WorkflowExtractor",
    "create_atlas_uaro_integration",
    "solve_atlas_task",
]
