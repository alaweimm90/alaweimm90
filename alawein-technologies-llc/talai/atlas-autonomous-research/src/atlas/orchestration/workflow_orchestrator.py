"""
ORCHEX Orchestration - Workflow Orchestrator

Dynamically configures ORCHEX pipeline based on problem type.
Adapts agent selection, resource allocation, and execution strategy.

Cycle 21-22: Adaptive Orchestration
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ORCHEX.orchestration.problem_types import (
    ProblemType,
    ProblemCharacteristics,
    get_problem_config
)
from ORCHEX.orchestration.intent_classifier import (
    IntentClassifier,
    ClassificationResult
)


@dataclass
class WorkflowConfig:
    """Configuration for ORCHEX workflow execution"""

    # Problem classification
    problem_type: ProblemType
    classification: ClassificationResult

    # Stage execution
    execute_hypothesis_generation: bool = True
    execute_validation: bool = True
    execute_experimentation: bool = True
    execute_paper_generation: bool = True

    # Resource limits
    max_hypotheses: int = 5
    max_validation_rounds: int = 3
    max_experiment_variants: int = 3
    max_paper_sections: int = 6

    # Agent preferences (role -> priority multiplier)
    agent_priorities: Dict[str, float] = field(default_factory=dict)

    # Special features
    enable_brainstorming: bool = False
    enable_novelty_scoring: bool = False
    enable_cross_domain_search: bool = False
    parallel_hypothesis_generation: bool = False

    # Meta-learning
    meta_learning_contexts: List[str] = field(default_factory=list)

    # Timeouts (seconds)
    hypothesis_timeout: int = 600  # 10 min
    validation_timeout: int = 1800  # 30 min
    experiment_timeout: int = 3600  # 1 hour
    paper_timeout: int = 900  # 15 min

    # Quality thresholds
    hypothesis_quality_threshold: float = 0.6
    validation_quality_threshold: float = 0.7
    experiment_quality_threshold: float = 0.6

    # Iteration limits
    max_refinement_iterations: int = 3
    max_retry_attempts: int = 2


class WorkflowOrchestrator:
    """
    Orchestrates ORCHEX workflow based on problem type.

    Responsibilities:
    1. Classify user intent
    2. Configure pipeline stages
    3. Select optimal agents
    4. Allocate resources
    5. Adapt execution strategy
    """

    def __init__(self, orchestrator=None):
        """
        Initialize workflow orchestrator

        Args:
            orchestrator: Optional AI Orchestrator
        """
        self.orchestrator = orchestrator
        self.intent_classifier = IntentClassifier(orchestrator=orchestrator)

    def plan_workflow(
        self,
        query: str,
        domain: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> WorkflowConfig:
        """
        Plan ORCHEX workflow for a research query

        Args:
            query: User's research question/request
            domain: Research domain
            user_preferences: Optional user preferences to override defaults

        Returns:
            WorkflowConfig with execution plan
        """
        # Step 1: Classify intent
        classification = self.intent_classifier.classify(query, domain)
        problem_config = classification.problem_config

        # Step 2: Build base configuration
        config = self._build_base_config(classification, problem_config)

        # Step 3: Apply user preferences
        if user_preferences:
            config = self._apply_user_preferences(config, user_preferences)

        # Step 4: Validate and optimize
        config = self._validate_config(config)

        return config

    def _build_base_config(
        self,
        classification: ClassificationResult,
        problem_config: ProblemCharacteristics
    ) -> WorkflowConfig:
        """Build base workflow configuration from problem characteristics"""

        # Stage execution
        execute_hypothesis = problem_config.requires_hypothesis_generation
        execute_validation = problem_config.requires_validation
        execute_experiment = problem_config.requires_experimentation
        execute_paper = problem_config.requires_paper

        # Resource limits
        max_hypotheses = problem_config.max_hypotheses
        max_validation = problem_config.max_validation_rounds
        max_experiments = problem_config.max_experiment_variants

        # Agent priorities (convert to dict)
        agent_priorities = dict(problem_config.preferred_agents)

        # Special features
        enable_brainstorming = problem_config.enable_brainstorming
        enable_novelty = problem_config.enable_novelty_scoring
        enable_cross_domain = problem_config.enable_cross_domain_search
        parallel_hyp = problem_config.parallel_hypothesis_generation

        # Meta-learning contexts
        ml_contexts = list(problem_config.meta_learning_contexts)

        # Timeouts based on emphasis
        hyp_timeout = int(600 * problem_config.hypothesis_emphasis + 300)
        val_timeout = int(1800 * problem_config.validation_emphasis + 600)
        exp_timeout = int(3600 * problem_config.experiment_emphasis + 1200)
        paper_timeout = int(900 * problem_config.paper_emphasis + 300)

        # Quality thresholds (stricter for high-emphasis stages)
        hyp_quality = 0.5 + 0.2 * problem_config.hypothesis_emphasis
        val_quality = 0.6 + 0.2 * problem_config.validation_emphasis
        exp_quality = 0.5 + 0.2 * problem_config.experiment_emphasis

        return WorkflowConfig(
            problem_type=classification.primary_type,
            classification=classification,
            execute_hypothesis_generation=execute_hypothesis,
            execute_validation=execute_validation,
            execute_experimentation=execute_experiment,
            execute_paper_generation=execute_paper,
            max_hypotheses=max_hypotheses,
            max_validation_rounds=max_validation,
            max_experiment_variants=max_experiments,
            agent_priorities=agent_priorities,
            enable_brainstorming=enable_brainstorming,
            enable_novelty_scoring=enable_novelty,
            enable_cross_domain_search=enable_cross_domain,
            parallel_hypothesis_generation=parallel_hyp,
            meta_learning_contexts=ml_contexts,
            hypothesis_timeout=hyp_timeout,
            validation_timeout=val_timeout,
            experiment_timeout=exp_timeout,
            paper_timeout=paper_timeout,
            hypothesis_quality_threshold=hyp_quality,
            validation_quality_threshold=val_quality,
            experiment_quality_threshold=exp_quality,
        )

    def _apply_user_preferences(
        self,
        config: WorkflowConfig,
        preferences: Dict[str, Any]
    ) -> WorkflowConfig:
        """Apply user preferences to override default configuration"""

        # Allow overriding resource limits
        if "max_hypotheses" in preferences:
            config.max_hypotheses = preferences["max_hypotheses"]

        if "max_validation_rounds" in preferences:
            config.max_validation_rounds = preferences["max_validation_rounds"]

        if "max_experiment_variants" in preferences:
            config.max_experiment_variants = preferences["max_experiment_variants"]

        # Allow overriding stage execution
        if "skip_experiments" in preferences and preferences["skip_experiments"]:
            config.execute_experimentation = False

        if "skip_paper" in preferences and preferences["skip_paper"]:
            config.execute_paper_generation = False

        # Allow enabling/disabling features
        if "enable_brainstorming" in preferences:
            config.enable_brainstorming = preferences["enable_brainstorming"]

        if "enable_novelty_scoring" in preferences:
            config.enable_novelty_scoring = preferences["enable_novelty_scoring"]

        # Allow timeout overrides
        if "timeout_multiplier" in preferences:
            multiplier = preferences["timeout_multiplier"]
            config.hypothesis_timeout = int(config.hypothesis_timeout * multiplier)
            config.validation_timeout = int(config.validation_timeout * multiplier)
            config.experiment_timeout = int(config.experiment_timeout * multiplier)
            config.paper_timeout = int(config.paper_timeout * multiplier)

        return config

    def _validate_config(self, config: WorkflowConfig) -> WorkflowConfig:
        """Validate and optimize configuration"""

        # Ensure at least one stage is enabled
        if not any([
            config.execute_hypothesis_generation,
            config.execute_validation,
            config.execute_experimentation,
            config.execute_paper_generation
        ]):
            # Re-enable all stages
            config.execute_hypothesis_generation = True
            config.execute_validation = True
            config.execute_experimentation = True
            config.execute_paper_generation = True

        # Ensure reasonable limits
        config.max_hypotheses = max(1, min(50, config.max_hypotheses))
        config.max_validation_rounds = max(1, min(10, config.max_validation_rounds))
        config.max_experiment_variants = max(1, min(20, config.max_experiment_variants))

        # Ensure reasonable timeouts
        config.hypothesis_timeout = max(60, min(3600, config.hypothesis_timeout))
        config.validation_timeout = max(300, min(7200, config.validation_timeout))
        config.experiment_timeout = max(600, min(14400, config.experiment_timeout))
        config.paper_timeout = max(300, min(3600, config.paper_timeout))

        # Ensure quality thresholds in valid range
        config.hypothesis_quality_threshold = max(0.0, min(1.0, config.hypothesis_quality_threshold))
        config.validation_quality_threshold = max(0.0, min(1.0, config.validation_quality_threshold))
        config.experiment_quality_threshold = max(0.0, min(1.0, config.experiment_quality_threshold))

        return config

    def get_agent_priority(
        self,
        config: WorkflowConfig,
        agent_role: str,
        base_priority: float = 0.5
    ) -> float:
        """
        Get priority for an agent based on workflow config

        Args:
            config: Workflow configuration
            agent_role: Agent role (e.g., "research", "criticism")
            base_priority: Base priority from meta-learning

        Returns:
            Adjusted priority (0-1)
        """
        if agent_role in config.agent_priorities:
            role_priority = config.agent_priorities[agent_role]
            # Combine with base priority (weighted average)
            return 0.6 * role_priority + 0.4 * base_priority
        return base_priority

    def should_skip_stage(
        self,
        config: WorkflowConfig,
        stage: str
    ) -> bool:
        """
        Check if a stage should be skipped

        Args:
            config: Workflow configuration
            stage: Stage name ("hypothesis", "validation", "experiment", "paper")

        Returns:
            True if stage should be skipped
        """
        stage_map = {
            "hypothesis": not config.execute_hypothesis_generation,
            "validation": not config.execute_validation,
            "experiment": not config.execute_experimentation,
            "paper": not config.execute_paper_generation,
        }
        return stage_map.get(stage, False)

    def get_workflow_summary(self, config: WorkflowConfig) -> str:
        """Get human-readable summary of workflow configuration"""

        summary_parts = [
            "=" * 80,
            "ORCHEX ADAPTIVE WORKFLOW CONFIGURATION",
            "=" * 80,
            "",
            f"Problem Type: {config.problem_type.value.upper()}",
            f"Confidence: {config.classification.confidence:.1%}",
            "",
            "PIPELINE STAGES:",
            f"  {'✓' if config.execute_hypothesis_generation else '✗'} Hypothesis Generation (max: {config.max_hypotheses})",
            f"  {'✓' if config.execute_validation else '✗'} Validation (max rounds: {config.max_validation_rounds})",
            f"  {'✓' if config.execute_experimentation else '✗'} Experimentation (max variants: {config.max_experiment_variants})",
            f"  {'✓' if config.execute_paper_generation else '✗'} Paper Generation",
            "",
            "SPECIAL FEATURES:",
        ]

        features = []
        if config.enable_brainstorming:
            features.append("  ✓ Brainstorming enabled")
        if config.enable_novelty_scoring:
            features.append("  ✓ Novelty scoring enabled")
        if config.enable_cross_domain_search:
            features.append("  ✓ Cross-domain search enabled")
        if config.parallel_hypothesis_generation:
            features.append("  ✓ Parallel hypothesis generation")

        if features:
            summary_parts.extend(features)
        else:
            summary_parts.append("  (none)")

        summary_parts.append("")
        summary_parts.append("RESOURCE LIMITS:")
        summary_parts.append(f"  • Hypothesis timeout: {config.hypothesis_timeout // 60}min")
        summary_parts.append(f"  • Validation timeout: {config.validation_timeout // 60}min")
        summary_parts.append(f"  • Experiment timeout: {config.experiment_timeout // 60}min")
        summary_parts.append(f"  • Paper timeout: {config.paper_timeout // 60}min")

        if config.agent_priorities:
            summary_parts.append("")
            summary_parts.append("AGENT PRIORITIES:")
            for role, priority in sorted(
                config.agent_priorities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                summary_parts.append(f"  • {role}: {priority:.0%}")

        summary_parts.append("")
        summary_parts.append("QUALITY THRESHOLDS:")
        summary_parts.append(f"  • Hypothesis: {config.hypothesis_quality_threshold:.1%}")
        summary_parts.append(f"  • Validation: {config.validation_quality_threshold:.1%}")
        summary_parts.append(f"  • Experiment: {config.experiment_quality_threshold:.1%}")

        summary_parts.append("")
        summary_parts.append("=" * 80)

        return "\n".join(summary_parts)

    def optimize_for_speed(self, config: WorkflowConfig) -> WorkflowConfig:
        """Optimize configuration for speed (sacrifice thoroughness)"""
        config.max_hypotheses = min(3, config.max_hypotheses)
        config.max_validation_rounds = min(2, config.max_validation_rounds)
        config.max_experiment_variants = min(2, config.max_experiment_variants)
        config.hypothesis_timeout = int(config.hypothesis_timeout * 0.5)
        config.validation_timeout = int(config.validation_timeout * 0.5)
        config.experiment_timeout = int(config.experiment_timeout * 0.7)
        config.paper_timeout = int(config.paper_timeout * 0.7)
        return config

    def optimize_for_quality(self, config: WorkflowConfig) -> WorkflowConfig:
        """Optimize configuration for quality (sacrifice speed)"""
        config.max_hypotheses = min(10, config.max_hypotheses * 2)
        config.max_validation_rounds = min(6, config.max_validation_rounds * 2)
        config.max_experiment_variants = min(5, config.max_experiment_variants + 2)
        config.hypothesis_quality_threshold = min(0.8, config.hypothesis_quality_threshold + 0.1)
        config.validation_quality_threshold = min(0.85, config.validation_quality_threshold + 0.1)
        config.experiment_quality_threshold = min(0.75, config.experiment_quality_threshold + 0.1)
        config.max_refinement_iterations = 5
        return config


# Convenience function
def plan_research_workflow(
    query: str,
    domain: Optional[str] = None,
    orchestrator=None,
    preferences: Optional[Dict[str, Any]] = None
) -> WorkflowConfig:
    """
    Plan ORCHEX research workflow for a query

    Args:
        query: User's research question/request
        domain: Research domain
        orchestrator: Optional AI Orchestrator
        preferences: Optional user preferences

    Returns:
        WorkflowConfig with execution plan
    """
    orchestrator_obj = WorkflowOrchestrator(orchestrator=orchestrator)
    return orchestrator_obj.plan_workflow(query, domain, preferences)
