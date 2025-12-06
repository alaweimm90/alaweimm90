"""
UARO - Universal Reasoning Primitives Library

Problem-agnostic reasoning operations that work across all domains.
Inspired by universal problem-solving techniques from AI, algorithms, and logic.

Cycle 27-28: Universal Reasoning Primitives

Based on:
- Search algorithms (AI textbooks)
- Constraint satisfaction (Tsang, 1993)
- Logic programming (Prolog, Datalog)
- Optimization theory
- Meta-reasoning (Russell & Wefald, 1991)
"""

from typing import Any, List, Callable, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import random


# ==================== ABSTRACT INTERFACES ====================

class Problem(ABC):
    """Abstract problem representation"""

    @abstractmethod
    def initial_state(self) -> Any:
        """Get initial state"""
        pass

    @abstractmethod
    def goal_test(self, state: Any) -> bool:
        """Check if state satisfies goal"""
        pass

    @abstractmethod
    def actions(self, state: Any) -> List[Any]:
        """Get available actions from state"""
        pass

    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        """Apply action to state"""
        pass

    @abstractmethod
    def cost(self, state: Any, action: Any) -> float:
        """Cost of taking action from state"""
        pass


class ReasoningPrimitive(ABC):
    """Abstract reasoning primitive"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.usage_count = 0
        self.success_count = 0

    @abstractmethod
    def apply(self, input_data: Any) -> Any:
        """Apply this primitive to input"""
        pass

    @abstractmethod
    def is_applicable(self, input_data: Any) -> bool:
        """Check if primitive can be applied"""
        pass

    def success_rate(self) -> float:
        """Historical success rate"""
        if self.usage_count == 0:
            return 0.5  # Unknown
        return self.success_count / self.usage_count

    def record_usage(self, success: bool):
        """Track usage statistics"""
        self.usage_count += 1
        if success:
            self.success_count += 1


# ==================== DECOMPOSITION PRIMITIVES ====================

class DivideAndConquer(ReasoningPrimitive):
    """Split problem into independent subproblems"""

    def __init__(self):
        super().__init__("divide_and_conquer", "decomposition")

    def apply(self, problem: Any) -> List[Any]:
        """Recursively divide until atomic"""
        if hasattr(problem, 'is_atomic') and problem.is_atomic():
            return [problem]

        if hasattr(problem, 'split'):
            left, right = problem.split()
            return self.apply(left) + self.apply(right)

        return [problem]

    def is_applicable(self, problem: Any) -> bool:
        return hasattr(problem, 'split')


class HierarchicalDecomposition(ReasoningPrimitive):
    """Create tree of abstraction levels"""

    def __init__(self):
        super().__init__("hierarchical_decomposition", "decomposition")

    def apply(self, problem: Any) -> Dict[str, Any]:
        """Build hierarchy from abstract to concrete"""
        hierarchy = {"root": problem}

        if hasattr(problem, 'max_depth'):
            max_depth = problem.max_depth()
            for level in range(max_depth):
                if hasattr(problem, 'expand_level'):
                    hierarchy[f"level_{level}"] = problem.expand_level(level)

        return hierarchy

    def is_applicable(self, problem: Any) -> bool:
        return hasattr(problem, 'expand_level')


# ==================== SEARCH PRIMITIVES ====================

class BreadthFirstSearch(ReasoningPrimitive):
    """Explore level by level"""

    def __init__(self):
        super().__init__("breadth_first_search", "search")

    def apply(self, problem: Problem) -> Optional[Any]:
        """BFS to find goal"""
        frontier = deque([problem.initial_state()])
        explored = set()

        while frontier:
            state = frontier.popleft()

            if problem.goal_test(state):
                return state

            explored.add(str(state))  # Convert to hashable

            for action in problem.actions(state):
                child = problem.result(state, action)
                if str(child) not in explored:
                    frontier.append(child)

        return None

    def is_applicable(self, problem: Any) -> bool:
        return isinstance(problem, Problem)


class DepthFirstSearch(ReasoningPrimitive):
    """Explore deeply before backtracking"""

    def __init__(self):
        super().__init__("depth_first_search", "search")

    def apply(self, problem: Problem, max_depth: int = 1000) -> Optional[Any]:
        """DFS with depth limit"""
        return self._recursive_dfs(
            problem,
            problem.initial_state(),
            set(),
            0,
            max_depth
        )

    def _recursive_dfs(
        self,
        problem: Problem,
        state: Any,
        explored: Set[str],
        depth: int,
        max_depth: int
    ) -> Optional[Any]:
        """Recursive DFS helper"""
        if depth > max_depth:
            return None

        if problem.goal_test(state):
            return state

        explored.add(str(state))

        for action in problem.actions(state):
            child = problem.result(state, action)
            if str(child) not in explored:
                result = self._recursive_dfs(
                    problem, child, explored, depth + 1, max_depth
                )
                if result is not None:
                    return result

        return None

    def is_applicable(self, problem: Any) -> bool:
        return isinstance(problem, Problem)


class BestFirstSearch(ReasoningPrimitive):
    """Always explore most promising path"""

    def __init__(self):
        super().__init__("best_first_search", "search")

    def apply(
        self,
        problem: Problem,
        heuristic: Callable[[Any], float]
    ) -> Optional[Any]:
        """Best-first search with heuristic"""
        frontier = [(heuristic(problem.initial_state()), problem.initial_state())]
        explored = set()

        while frontier:
            # Sort by heuristic value
            frontier.sort(reverse=True)
            _, state = frontier.pop(0)

            if problem.goal_test(state):
                return state

            explored.add(str(state))

            for action in problem.actions(state):
                child = problem.result(state, action)
                if str(child) not in explored:
                    frontier.append((heuristic(child), child))

        return None

    def is_applicable(self, problem: Any) -> bool:
        return isinstance(problem, Problem)


class BeamSearch(ReasoningPrimitive):
    """Keep top-k paths at each level"""

    def __init__(self):
        super().__init__("beam_search", "search")

    def apply(
        self,
        problem: Problem,
        beam_width: int = 5,
        heuristic: Optional[Callable[[Any], float]] = None
    ) -> Optional[Any]:
        """Beam search with width constraint"""
        if heuristic is None:
            heuristic = lambda s: 0.5

        frontier = [(heuristic(problem.initial_state()), problem.initial_state())]

        for _ in range(1000):  # Max iterations
            if not frontier:
                return None

            # Check if any goal
            for score, state in frontier:
                if problem.goal_test(state):
                    return state

            # Expand all
            candidates = []
            for score, state in frontier:
                for action in problem.actions(state):
                    child = problem.result(state, action)
                    candidates.append((heuristic(child), child))

            # Keep only top beam_width
            candidates.sort(reverse=True)
            frontier = candidates[:beam_width]

        return None

    def is_applicable(self, problem: Any) -> bool:
        return isinstance(problem, Problem)


# ==================== CONSTRAINT PRIMITIVES ====================

class ConstraintPropagation(ReasoningPrimitive):
    """Apply constraints to reduce search space"""

    def __init__(self):
        super().__init__("constraint_propagation", "constraint")

    def apply(self, csp: Any) -> Any:
        """Propagate constraints throughout CSP"""
        if not hasattr(csp, 'variables') or not hasattr(csp, 'domains'):
            return csp

        changed = True
        while changed:
            changed = False

            for var in csp.variables:
                original_size = len(csp.domains[var])

                # Remove values that violate constraints
                if hasattr(csp, 'constraints'):
                    valid_values = []
                    for value in csp.domains[var]:
                        # Check if value is consistent with constraints
                        is_valid = True
                        for constraint in csp.constraints:
                            if hasattr(constraint, 'check'):
                                if not constraint.check(var, value):
                                    is_valid = False
                                    break
                        if is_valid:
                            valid_values.append(value)

                    csp.domains[var] = valid_values

                if len(csp.domains[var]) < original_size:
                    changed = True

        return csp

    def is_applicable(self, csp: Any) -> bool:
        return hasattr(csp, 'variables') and hasattr(csp, 'domains')


class BacktrackingSearch(ReasoningPrimitive):
    """Systematic search with backtracking"""

    def __init__(self):
        super().__init__("backtracking_search", "constraint")

    def apply(self, csp: Any) -> Optional[Dict[Any, Any]]:
        """Backtracking CSP solver"""
        return self._backtrack({}, csp)

    def _backtrack(self, assignment: Dict, csp: Any) -> Optional[Dict]:
        """Recursive backtracking"""
        # Check if complete
        if hasattr(csp, 'is_complete'):
            if csp.is_complete(assignment):
                return assignment
        elif len(assignment) == len(csp.variables):
            return assignment

        # Select unassigned variable
        var = self._select_unassigned_variable(assignment, csp)

        # Try each value in domain
        for value in csp.domains[var]:
            if self._is_consistent(var, value, assignment, csp):
                assignment[var] = value

                result = self._backtrack(assignment, csp)
                if result is not None:
                    return result

                del assignment[var]

        return None

    def _select_unassigned_variable(self, assignment: Dict, csp: Any) -> Any:
        """Choose next variable to assign"""
        for var in csp.variables:
            if var not in assignment:
                return var
        return None

    def _is_consistent(self, var: Any, value: Any, assignment: Dict, csp: Any) -> bool:
        """Check if assignment is consistent with constraints"""
        if not hasattr(csp, 'constraints'):
            return True

        for constraint in csp.constraints:
            if hasattr(constraint, 'is_satisfied'):
                temp_assignment = assignment.copy()
                temp_assignment[var] = value
                if not constraint.is_satisfied(temp_assignment):
                    return False

        return True

    def is_applicable(self, csp: Any) -> bool:
        return hasattr(csp, 'variables') and hasattr(csp, 'domains')


# ==================== LOGIC PRIMITIVES ====================

class ForwardChaining(ReasoningPrimitive):
    """Apply rules until no new facts derived"""

    def __init__(self):
        super().__init__("forward_chaining", "logic")

    def apply(self, kb: Any) -> Set[Any]:
        """Forward chaining inference"""
        if not hasattr(kb, 'facts') or not hasattr(kb, 'rules'):
            return set()

        inferred = set(kb.facts)
        changed = True
        iterations = 0
        max_iterations = 100

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for rule in kb.rules:
                # rule format: (premise, conclusion)
                if hasattr(rule, '__iter__') and len(rule) == 2:
                    premise, conclusion = rule

                    # Check if premise is satisfied
                    if premise in inferred or (hasattr(premise, '__iter__') and all(p in inferred for p in premise)):
                        if conclusion not in inferred:
                            inferred.add(conclusion)
                            changed = True

        return inferred

    def is_applicable(self, kb: Any) -> bool:
        return hasattr(kb, 'facts') and hasattr(kb, 'rules')


class BackwardChaining(ReasoningPrimitive):
    """Work backwards from goal to prove it"""

    def __init__(self):
        super().__init__("backward_chaining", "logic")

    def apply(self, kb: Any, goal: Any) -> Tuple[bool, List[Any]]:
        """Backward chaining to prove goal"""
        if not hasattr(kb, 'facts') or not hasattr(kb, 'rules'):
            return False, []

        return self._prove(goal, kb, set())

    def _prove(self, goal: Any, kb: Any, explored: Set[Any]) -> Tuple[bool, List[Any]]:
        """Recursive backward chaining"""
        # Base case: goal is a known fact
        if goal in kb.facts:
            return True, [goal]

        # Avoid infinite loops
        if goal in explored:
            return False, []

        explored.add(goal)

        # Try each rule
        for rule in kb.rules:
            if hasattr(rule, '__iter__') and len(rule) == 2:
                premise, conclusion = rule

                if conclusion == goal:
                    # Recursively prove premise
                    if isinstance(premise, (list, tuple)):
                        # Multiple premises (AND)
                        all_proved = True
                        proof_path = []
                        for p in premise:
                            proved, path = self._prove(p, kb, explored)
                            if not proved:
                                all_proved = False
                                break
                            proof_path.extend(path)

                        if all_proved:
                            proof_path.append(goal)
                            return True, proof_path
                    else:
                        # Single premise (string or other)
                        proved, path = self._prove(premise, kb, explored)
                        if proved:
                            path.append(goal)
                            return True, path

        return False, []

    def is_applicable(self, kb: Any) -> bool:
        return hasattr(kb, 'facts') and hasattr(kb, 'rules')


# ==================== OPTIMIZATION PRIMITIVES ====================

class LocalSearch(ReasoningPrimitive):
    """Hill climbing / gradient descent"""

    def __init__(self):
        super().__init__("local_search", "optimization")

    def apply(
        self,
        objective: Callable[[Any], float],
        initial_state: Any,
        max_iterations: int = 1000
    ) -> Any:
        """Hill climbing to local optimum"""
        current = initial_state
        current_value = objective(current)

        for _ in range(max_iterations):
            # Get neighbors
            if hasattr(current, 'neighbors'):
                neighbors = current.neighbors()
            else:
                break

            # Find best neighbor
            best_neighbor = None
            best_value = current_value

            for neighbor in neighbors:
                value = objective(neighbor)
                if value > best_value:
                    best_value = value
                    best_neighbor = neighbor

            # Stop if no improvement
            if best_neighbor is None:
                break

            current = best_neighbor
            current_value = best_value

        return current

    def is_applicable(self, state: Any) -> bool:
        return hasattr(state, 'neighbors')


class SimulatedAnnealing(ReasoningPrimitive):
    """Accept worse solutions probabilistically"""

    def __init__(self):
        super().__init__("simulated_annealing", "optimization")

    def apply(
        self,
        objective: Callable[[Any], float],
        initial_state: Any,
        temperature_schedule: Optional[List[float]] = None
    ) -> Any:
        """Simulated annealing optimization"""
        if temperature_schedule is None:
            temperature_schedule = [1.0 * (0.95 ** i) for i in range(1000)]

        current = initial_state
        current_value = objective(current)

        for temp in temperature_schedule:
            if not hasattr(current, 'random_neighbor'):
                break

            neighbor = current.random_neighbor()
            neighbor_value = objective(neighbor)

            delta = neighbor_value - current_value

            # Accept if better, or probabilistically if worse
            if delta > 0 or random.random() < math.exp(delta / temp):
                current = neighbor
                current_value = neighbor_value

        return current

    def is_applicable(self, state: Any) -> bool:
        return hasattr(state, 'random_neighbor')


# ==================== PRIMITIVE REGISTRY ====================

class PrimitiveRegistry:
    """Central registry of all reasoning primitives"""

    def __init__(self):
        self.primitives: Dict[str, ReasoningPrimitive] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all built-in primitives"""
        default_primitives = [
            # Decomposition
            DivideAndConquer(),
            HierarchicalDecomposition(),

            # Search
            BreadthFirstSearch(),
            DepthFirstSearch(),
            BestFirstSearch(),
            BeamSearch(),

            # Constraints
            ConstraintPropagation(),
            BacktrackingSearch(),

            # Logic
            ForwardChaining(),
            BackwardChaining(),

            # Optimization
            LocalSearch(),
            SimulatedAnnealing(),
        ]

        for primitive in default_primitives:
            self.register(primitive)

    def register(self, primitive: ReasoningPrimitive):
        """Add a primitive to registry"""
        self.primitives[primitive.name] = primitive

    def get(self, name: str) -> Optional[ReasoningPrimitive]:
        """Get primitive by name"""
        return self.primitives.get(name)

    def get_by_category(self, category: str) -> List[ReasoningPrimitive]:
        """Get all primitives in category"""
        return [
            p for p in self.primitives.values()
            if p.category == category
        ]

    def get_applicable(self, problem: Any) -> List[ReasoningPrimitive]:
        """Get primitives that can be applied to problem"""
        return [
            p for p in self.primitives.values()
            if p.is_applicable(problem)
        ]

    def list_all(self) -> List[str]:
        """List all registered primitive names"""
        return list(self.primitives.keys())
