"""
UARO - Universal Solver

The meta-algorithm that applies reasoning primitives until convergence.
Self-iterating loop that works on ANY problem structure.

Cycle 27-28: Universal Reasoning Primitives
"""

from typing import Any, List, Optional, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

from uaro.reasoning_primitives import (
    Problem,
    ReasoningPrimitive,
    PrimitiveRegistry
)


@dataclass
class ReasoningStep:
    """A single step in reasoning process"""

    iteration: int
    timestamp: float
    primitive_name: str
    input_state: Any
    output_state: Any
    confidence: float
    success: bool
    reasoning: str


@dataclass
class SolutionResult:
    """Result from universal solver"""

    problem: Any
    solution: Any
    success: bool
    iterations: int
    duration_seconds: float
    confidence: float
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    primitives_used: List[str] = field(default_factory=list)


class UniversalSolver:
    """
    The core UARO engine: Applies reasoning primitives iteratively until:
    - Goal is reached
    - Convergence detected
    - Maximum iterations exceeded
    - High confidence achieved
    """

    def __init__(
        self,
        primitive_registry: Optional[PrimitiveRegistry] = None,
        max_iterations: int = 1000,
        confidence_threshold: float = 0.95
    ):
        """
        Initialize universal solver

        Args:
            primitive_registry: Registry of available primitives
            max_iterations: Maximum reasoning iterations
            confidence_threshold: Stop when confidence exceeds this
        """
        self.registry = primitive_registry or PrimitiveRegistry()
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Meta-learning: Track which primitives work well
        self.primitive_scores: Dict[str, float] = {}

    def solve(
        self,
        problem: Any,
        initial_state: Optional[Any] = None,
        goal: Optional[Any] = None
    ) -> SolutionResult:
        """
        Universal problem solving loop

        Args:
            problem: Problem to solve
            initial_state: Starting state (if None, extract from problem)
            goal: Goal condition (if None, use problem's goal_test)

        Returns:
            SolutionResult with solution and reasoning trace
        """
        start_time = time.time()

        # Initialize state
        if initial_state is None:
            if isinstance(problem, Problem):
                state = problem.initial_state()
            else:
                state = problem

        if goal is None and isinstance(problem, Problem):
            goal = problem.goal_test

        # Reasoning trace
        trace = []
        iteration = 0
        confidence = 0.5  # Start uncertain

        # Main reasoning loop
        while iteration < self.max_iterations:
            iteration += 1

            # Check termination criteria
            if self._is_done(state, goal, confidence):
                duration = time.time() - start_time
                return SolutionResult(
                    problem=problem,
                    solution=state,
                    success=True,
                    iterations=iteration,
                    duration_seconds=duration,
                    confidence=confidence,
                    reasoning_trace=trace,
                    primitives_used=list(set(step.primitive_name for step in trace))
                )

            # Get applicable primitives
            applicable = self.registry.get_applicable(state)
            apply_target = state  # Track what to apply primitives to

            # If no primitives applicable to state, try problem directly
            if not applicable:
                applicable = self.registry.get_applicable(problem)
                apply_target = problem  # Apply to problem instead

            if not applicable:
                # No primitives can help - record and exit
                step = ReasoningStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    primitive_name="none_applicable",
                    input_state=self._snapshot(state),
                    output_state=self._snapshot(state),
                    confidence=confidence,
                    success=False,
                    reasoning="No applicable primitives found for current state"
                )
                trace.append(step)

                duration = time.time() - start_time
                return SolutionResult(
                    problem=problem,
                    solution=state,
                    success=False,
                    iterations=iteration,
                    duration_seconds=duration,
                    confidence=confidence,
                    reasoning_trace=trace,
                    primitives_used=list(set(step.primitive_name for step in trace if step.primitive_name != "none_applicable"))
                )

            # Select best primitive
            primitive = self._select_primitive(applicable, state, trace)

            # Apply primitive to correct target (state or problem)
            try:
                new_state = primitive.apply(apply_target)
                success = True
                reasoning = f"Applied {primitive.name} successfully"

                # Update confidence based on progress
                progress = self._estimate_progress(state, new_state, goal)
                confidence = min(0.99, confidence + progress * 0.1)

            except Exception as e:
                new_state = state
                success = False
                reasoning = f"Failed to apply {primitive.name}: {str(e)}"
                confidence = max(0.1, confidence - 0.05)

            # Record step
            step = ReasoningStep(
                iteration=iteration,
                timestamp=time.time(),
                primitive_name=primitive.name,
                input_state=self._snapshot(state),
                output_state=self._snapshot(new_state),
                confidence=confidence,
                success=success,
                reasoning=reasoning
            )
            trace.append(step)

            # Update primitive performance
            primitive.record_usage(success)
            self._update_primitive_score(primitive.name, success)

            # Move to new state
            state = new_state

        # Exceeded max iterations
        duration = time.time() - start_time
        return SolutionResult(
            problem=problem,
            solution=state,
            success=False,
            iterations=iteration,
            duration_seconds=duration,
            confidence=confidence,
            reasoning_trace=trace,
            primitives_used=list(set(step.primitive_name for step in trace))
        )

    def _is_done(
        self,
        state: Any,
        goal: Optional[Callable],
        confidence: float
    ) -> bool:
        """Check termination criteria"""

        # High confidence
        if confidence >= self.confidence_threshold:
            return True

        # Goal reached
        if goal is not None:
            if callable(goal):
                try:
                    if goal(state):
                        return True
                except:
                    pass
            elif state == goal:
                return True

        # State indicates completion
        if hasattr(state, 'is_solution'):
            if state.is_solution():
                return True

        return False

    def _select_primitive(
        self,
        applicable: List[ReasoningPrimitive],
        state: Any,
        history: List[ReasoningStep]
    ) -> ReasoningPrimitive:
        """
        Meta-reasoning: Choose which primitive to apply next

        Scoring based on:
        1. Historical success rate
        2. Primitive-specific heuristics
        3. Diversity (avoid repeating same primitive)
        """
        scores = []

        # Count recent usage
        recent_usage = {}
        for step in history[-5:]:  # Last 5 steps
            recent_usage[step.primitive_name] = recent_usage.get(step.primitive_name, 0) + 1

        for primitive in applicable:
            # Base score: historical success
            hist_score = primitive.success_rate()

            # Global score from meta-learning
            global_score = self.primitive_scores.get(primitive.name, 0.5)

            # Diversity bonus: penalize recently used
            diversity_score = 1.0 - (recent_usage.get(primitive.name, 0) / 10.0)

            # Composite score
            total_score = (
                0.4 * hist_score +
                0.3 * global_score +
                0.3 * diversity_score
            )

            scores.append((total_score, primitive))

        # Return best (sort by score only, not primitive object)
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _estimate_progress(
        self,
        old_state: Any,
        new_state: Any,
        goal: Optional[Callable]
    ) -> float:
        """
        Estimate how much progress was made (0-1)

        Heuristics:
        - State changed significantly = progress
        - State unchanged = no progress
        - Closer to goal = progress
        """
        # No change = no progress
        if self._states_equal(old_state, new_state):
            return 0.0

        # Some change = some progress
        return 0.5

    def _states_equal(self, s1: Any, s2: Any) -> bool:
        """Check if two states are equal"""
        try:
            return s1 == s2
        except:
            return str(s1) == str(s2)

    def _snapshot(self, state: Any) -> Any:
        """Create snapshot of state for trace"""
        if hasattr(state, 'copy'):
            return state.copy()
        elif hasattr(state, '__dict__'):
            return str(state.__dict__)
        else:
            return str(state)

    def _update_primitive_score(self, primitive_name: str, success: bool):
        """Update global primitive scores (meta-learning)"""
        if primitive_name not in self.primitive_scores:
            self.primitive_scores[primitive_name] = 0.5

        # Exponential moving average
        alpha = 0.1
        reward = 1.0 if success else 0.0
        self.primitive_scores[primitive_name] = (
            (1 - alpha) * self.primitive_scores[primitive_name] +
            alpha * reward
        )

    def get_primitive_stats(self) -> Dict[str, Any]:
        """Get statistics on primitive usage and performance"""
        stats = {}

        for name, primitive in self.registry.primitives.items():
            stats[name] = {
                "category": primitive.category,
                "usage_count": primitive.usage_count,
                "success_count": primitive.success_count,
                "success_rate": primitive.success_rate(),
                "global_score": self.primitive_scores.get(name, 0.5)
            }

        return stats


# ==================== PROBLEM STRUCTURE RECOGNITION ====================

@dataclass
class ProblemSignature:
    """Signature for recognizing problem types"""

    name: str
    required_attributes: List[str]
    recommended_primitives: List[str]


class ProblemClassifier:
    """
    Recognize problem structure from attributes.
    Maps problems to appropriate reasoning primitives.
    """

    SIGNATURES = [
        ProblemSignature(
            name="search_problem",
            required_attributes=["initial_state", "goal_test", "actions", "result"],
            recommended_primitives=[
                "breadth_first_search",
                "depth_first_search",
                "best_first_search",
                "beam_search"
            ]
        ),
        ProblemSignature(
            name="constraint_satisfaction",
            required_attributes=["variables", "domains", "constraints"],
            recommended_primitives=[
                "constraint_propagation",
                "backtracking_search"
            ]
        ),
        ProblemSignature(
            name="optimization",
            required_attributes=["objective", "neighbors"],
            recommended_primitives=[
                "local_search",
                "simulated_annealing"
            ]
        ),
        ProblemSignature(
            name="logic_problem",
            required_attributes=["facts", "rules"],
            recommended_primitives=[
                "forward_chaining",
                "backward_chaining"
            ]
        ),
        ProblemSignature(
            name="decomposable",
            required_attributes=["split"],
            recommended_primitives=[
                "divide_and_conquer",
                "hierarchical_decomposition"
            ]
        ),
    ]

    def classify(self, problem: Any) -> List[str]:
        """
        Identify problem types from structure

        Args:
            problem: Problem to classify

        Returns:
            List of matching problem type names
        """
        matches = []

        for signature in self.SIGNATURES:
            if all(hasattr(problem, attr) for attr in signature.required_attributes):
                matches.append(signature.name)

        return matches if matches else ["unknown"]

    def get_recommended_primitives(self, problem: Any) -> List[str]:
        """
        Get recommended primitive names for problem

        Args:
            problem: Problem to analyze

        Returns:
            List of recommended primitive names
        """
        problem_types = self.classify(problem)

        recommended = []
        for signature in self.SIGNATURES:
            if signature.name in problem_types:
                recommended.extend(signature.recommended_primitives)

        return list(set(recommended))


# ==================== CONVENIENCE FUNCTIONS ====================

def solve_with_uaro(
    problem: Any,
    max_iterations: int = 1000,
    confidence_threshold: float = 0.95
) -> SolutionResult:
    """
    Convenience function: Solve any problem with UARO

    Args:
        problem: Problem to solve
        max_iterations: Max reasoning iterations
        confidence_threshold: Stop when confidence exceeds this

    Returns:
        SolutionResult
    """
    solver = UniversalSolver(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold
    )
    return solver.solve(problem)
