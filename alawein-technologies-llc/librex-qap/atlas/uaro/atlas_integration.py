"""
UARO-ORCHEX Integration

Connects Universal Autonomous Reasoning Orchestrator (UARO) with
ORCHEX Autonomous Research System.

This integration enables:
1. ORCHEX research tasks → UARO Problems (automatic formulation)
2. ORCHEX workflows → UARO Primitives (automatic extraction)
3. UARO solutions → ORCHEX research outputs (with proof documents)
4. Bidirectional learning (ORCHEX learns problem-solving, UARO learns domain knowledge)
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Callable
from pathlib import Path
import json

from uaro.reasoning_primitives import ReasoningPrimitive, Problem
from uaro.universal_solver import UniversalSolver, SolutionResult
from uaro.explainability import ExplainabilityEngine
from uaro.marketplace import PrimitiveMarketplace


@dataclass
class ResearchTask:
    """
    ORCHEX research task representation

    Attributes:
        task_id: Unique task identifier
        task_type: Type of research task (literature_review, hypothesis_test, etc.)
        query: Research question or hypothesis
        context: Background information and constraints
        success_criteria: How to evaluate solution quality
        metadata: Additional task-specific information
    """
    task_id: str
    task_type: str
    query: str
    context: Dict[str, Any]
    success_criteria: Callable[[Any], float]  # Returns score 0-1
    metadata: Dict[str, Any] = None


class ResearchProblem(Problem):
    """
    Adapter that converts ORCHEX research tasks into UARO Problems

    This allows UARO's universal primitives to reason about research tasks.
    """

    def __init__(self, task: ResearchTask):
        """
        Initialize research problem from ORCHEX task

        Args:
            task: ORCHEX research task to convert
        """
        self.task = task
        self.current_state = {
            'phase': 'planning',
            'evidence': [],
            'hypotheses': [],
            'experiments': [],
            'results': {},
            'confidence': 0.0
        }

    def initial_state(self) -> Dict[str, Any]:
        """Get initial research state"""
        return self.current_state.copy()

    def goal_test(self, state: Dict[str, Any]) -> bool:
        """
        Check if research task is complete

        Args:
            state: Current research state

        Returns:
            True if task solved (confidence > 0.8 and results validated)
        """
        if state['phase'] != 'complete':
            return False

        # Check confidence threshold
        if state['confidence'] < 0.8:
            return False

        # Evaluate against success criteria
        score = self.task.success_criteria(state)
        return score >= 0.7

    def actions(self, state: Dict[str, Any]) -> List[str]:
        """
        Get available research actions for current state

        Args:
            state: Current research state

        Returns:
            List of valid action names
        """
        actions = []

        phase = state['phase']

        if phase == 'planning':
            actions.extend([
                'formulate_hypotheses',
                'identify_knowledge_gaps',
                'design_experiments',
                'gather_background'
            ])

        elif phase == 'investigation':
            actions.extend([
                'run_experiment',
                'collect_evidence',
                'analyze_data',
                'test_hypothesis'
            ])

        elif phase == 'synthesis':
            actions.extend([
                'integrate_findings',
                'validate_results',
                'identify_limitations',
                'draw_conclusions'
            ])

        elif phase == 'complete':
            actions.append('refine_results')

        # Universal actions available in all phases
        actions.extend([
            'revise_approach',
            'seek_clarification',
            'progress_phase'
        ])

        return actions

    def result(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Apply research action to state

        Args:
            state: Current research state
            action: Action to apply

        Returns:
            New research state after action
        """
        new_state = state.copy()

        if action == 'formulate_hypotheses':
            # Generate hypotheses based on task query
            new_state['hypotheses'].append({
                'text': f"Hypothesis for {self.task.query}",
                'confidence': 0.5
            })

        elif action == 'collect_evidence':
            # Gather evidence
            new_state['evidence'].append({
                'source': 'search',
                'content': 'Evidence data'
            })

        elif action == 'progress_phase':
            # Move to next research phase
            phase_progression = {
                'planning': 'investigation',
                'investigation': 'synthesis',
                'synthesis': 'complete'
            }
            if state['phase'] in phase_progression:
                new_state['phase'] = phase_progression[state['phase']]

        elif action == 'validate_results':
            # Validate findings
            new_state['confidence'] = min(1.0, state['confidence'] + 0.2)

        return new_state

    def cost(self, state: Dict[str, Any], action: str) -> float:
        """
        Estimate cost of research action

        Args:
            state: Current research state
            action: Action to evaluate

        Returns:
            Estimated cost (time, resources)
        """
        # Expensive actions
        if action in ['run_experiment', 'analyze_data']:
            return 10.0

        # Medium cost actions
        if action in ['collect_evidence', 'validate_results']:
            return 5.0

        # Cheap actions
        return 1.0


class WorkflowExtractor:
    """
    Extracts successful ORCHEX workflows as reusable UARO primitives

    Learns from ORCHEX's domain expertise by converting proven workflows
    into primitives that can be shared via marketplace.
    """

    def __init__(self):
        """Initialize workflow extractor"""
        self.extracted_primitives = {}
        self.workflow_success_threshold = 0.8

    def extract_from_solution(
        self,
        solution: SolutionResult,
        task: ResearchTask
    ) -> Optional[ReasoningPrimitive]:
        """
        Extract reusable primitive from successful research solution

        Args:
            solution: UARO solution result
            task: Original research task

        Returns:
            Extracted primitive or None if not extractable
        """
        if not solution.success or solution.confidence < self.workflow_success_threshold:
            return None

        # Analyze reasoning trace to identify pattern
        pattern = self._identify_pattern(solution.reasoning_trace)

        if not pattern:
            return None

        # Create custom primitive from pattern
        class ExtractedPrimitive(ReasoningPrimitive):
            """Primitive extracted from ORCHEX workflow"""

            def __init__(self, pattern_data, task_type):
                super().__init__(
                    name=f"atlas_{task_type}_workflow",
                    category="research"
                )
                self.pattern = pattern_data
                self.task_type = task_type

            def is_applicable(self, problem: Any) -> bool:
                """Check if applicable to problem"""
                if isinstance(problem, ResearchProblem):
                    return problem.task.task_type == self.task_type
                return False

            def apply(self, problem: Any) -> Any:
                """Apply learned workflow"""
                # Execute pattern steps
                state = problem.initial_state()
                for step in self.pattern['steps']:
                    state = problem.result(state, step['action'])
                return state

        primitive = ExtractedPrimitive(pattern, task.task_type)
        self.extracted_primitives[primitive.name] = primitive

        return primitive

    def _identify_pattern(self, reasoning_trace: List) -> Optional[Dict]:
        """
        Identify reusable pattern in reasoning trace

        Args:
            reasoning_trace: List of reasoning steps

        Returns:
            Pattern data or None
        """
        if len(reasoning_trace) < 3:
            return None

        # Extract successful steps
        successful_steps = [
            step for step in reasoning_trace
            if step.success
        ]

        if not successful_steps:
            return None

        # Build pattern
        pattern = {
            'steps': [
                {
                    'action': step.primitive_name,
                    'reasoning': step.reasoning
                }
                for step in successful_steps
            ],
            'confidence': successful_steps[-1].confidence
        }

        return pattern


class ATLASUAROIntegration:
    """
    Main integration class connecting ORCHEX and UARO

    Provides:
    - Automatic problem formulation (ORCHEX tasks → UARO problems)
    - Workflow extraction (ORCHEX workflows → UARO primitives)
    - Explainable research (UARO proofs for ORCHEX outputs)
    - Marketplace integration (share research primitives)
    """

    def __init__(
        self,
        solver: UniversalSolver = None,
        marketplace: PrimitiveMarketplace = None
    ):
        """
        Initialize ORCHEX-UARO integration

        Args:
            solver: UARO solver instance (creates if None)
            marketplace: Primitive marketplace (creates if None)
        """
        self.solver = solver or UniversalSolver()
        self.marketplace = marketplace or PrimitiveMarketplace()
        self.explainer = ExplainabilityEngine()
        self.extractor = WorkflowExtractor()

        # Track solved tasks for learning
        self.solved_tasks = []

    def solve_research_task(
        self,
        task: ResearchTask,
        max_iterations: int = 100,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Solve ORCHEX research task using UARO

        Args:
            task: Research task to solve
            max_iterations: Maximum solver iterations
            explain: Generate proof document

        Returns:
            Dictionary with solution, proof, and metadata
        """
        # Convert task to UARO problem
        problem = ResearchProblem(task)

        # Create solver with custom max_iterations
        solver = UniversalSolver(
            primitive_registry=self.solver.registry,
            max_iterations=max_iterations
        )

        # Solve with UARO
        solution = solver.solve(problem)

        # Generate proof document if requested
        proof_doc = None
        if explain:
            proof_doc = self.explainer.generate_proof_document(
                solution,
                title=f"Research Task: {task.query}"
            )

        # Try to extract workflow as primitive
        extracted_primitive = self.extractor.extract_from_solution(
            solution,
            task
        )

        # Track successful solution
        if solution.success:
            self.solved_tasks.append({
                'task': task,
                'solution': solution,
                'primitive': extracted_primitive
            })

        return {
            'solution': solution,
            'proof_document': proof_doc,
            'extracted_primitive': extracted_primitive,
            'success': solution.success,
            'confidence': solution.confidence,
            'reasoning_trace': solution.reasoning_trace
        }

    def publish_learned_workflow(
        self,
        primitive: ReasoningPrimitive,
        author: str,
        description: str,
        tags: List[str] = None
    ) -> str:
        """
        Publish extracted workflow to marketplace

        Args:
            primitive: Extracted primitive
            author: Author name
            description: Primitive description
            tags: Tags for discovery

        Returns:
            Listing ID
        """
        listing_id = self.marketplace.publish(
            primitive=primitive,
            author=author,
            description=description,
            tags=tags or ['ORCHEX', 'research', 'workflow']
        )

        return listing_id

    def discover_research_primitives(
        self,
        task_type: str = None
    ) -> List:
        """
        Discover relevant primitives from marketplace

        Args:
            task_type: Filter by research task type

        Returns:
            List of relevant primitive listings
        """
        tags = ['research', 'ORCHEX']
        if task_type:
            tags.append(task_type)

        listings = self.marketplace.discover(
            category='research',
            tags=tags,
            verified_only=True,
            sort_by='success_rate'
        )

        return listings

    def export_research_report(
        self,
        result: Dict[str, Any],
        output_path: Path,
        format: str = "markdown"
    ) -> None:
        """
        Export research result with proof as report

        Args:
            result: Result from solve_research_task
            output_path: Where to save report
            format: Export format (markdown, html, latex, json)
        """
        proof_doc = result['proof_document']

        if format == "markdown":
            content = self.explainer.export_markdown(proof_doc)
        elif format == "html":
            content = self.explainer.export_html(proof_doc)
        elif format == "latex":
            content = self.explainer.export_latex(proof_doc)
        elif format == "json":
            content = self.explainer.export_json(proof_doc)
        else:
            raise ValueError(f"Unsupported format: {format}")

        output_path.write_text(content)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics

        Returns:
            Statistics dictionary
        """
        total_tasks = len(self.solved_tasks)
        successful_tasks = sum(
            1 for t in self.solved_tasks
            if t['solution'].success
        )
        extracted_primitives = sum(
            1 for t in self.solved_tasks
            if t['primitive'] is not None
        )

        return {
            'total_tasks_solved': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'extracted_primitives': extracted_primitives,
            'extraction_rate': extracted_primitives / successful_tasks if successful_tasks > 0 else 0.0
        }


# Convenience functions

def create_atlas_uaro_integration() -> ATLASUAROIntegration:
    """Create new ORCHEX-UARO integration instance"""
    return ATLASUAROIntegration()


def solve_atlas_task(
    task: ResearchTask,
    integration: ATLASUAROIntegration = None
) -> Dict[str, Any]:
    """
    Solve ORCHEX research task (convenience function)

    Args:
        task: Research task to solve
        integration: Integration instance (creates if None)

    Returns:
        Solution result
    """
    if integration is None:
        integration = create_atlas_uaro_integration()

    return integration.solve_research_task(task)
