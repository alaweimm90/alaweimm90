"""
ORCHEX Orchestration Module

Adaptive workflow orchestration based on problem type.
Dynamically configures ORCHEX pipeline for optimal results.

Components:
- ProblemTypes: Classification of research problems
- IntentClassifier: Detect user intent and problem type
- WorkflowOrchestrator: Configure pipeline based on intent

Cycle 21-22: Adaptive Orchestration
"""

__version__ = "0.1.0"

from ORCHEX.orchestration.problem_types import (
    ProblemType,
    ProblemCharacteristics,
    PROBLEM_TYPE_CONFIGS,
    INTENT_KEYWORDS,
    get_problem_config
)

from ORCHEX.orchestration.intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    classify_intent
)

from ORCHEX.orchestration.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfig,
    plan_research_workflow
)

__all__ = [
    # Problem types
    "ProblemType",
    "ProblemCharacteristics",
    "PROBLEM_TYPE_CONFIGS",
    "INTENT_KEYWORDS",
    "get_problem_config",

    # Intent classification
    "IntentClassifier",
    "ClassificationResult",
    "classify_intent",

    # Workflow orchestration
    "WorkflowOrchestrator",
    "WorkflowConfig",
    "plan_research_workflow",
]
