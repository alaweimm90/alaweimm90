"""
Reasoning and inference over knowledge graphs.

Implements multi-hop reasoning, path finding, link prediction, and conflict resolution.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """Represents a reasoning path in the knowledge graph."""
    start_entity: str
    end_entity: str
    path: List[Tuple[str, str, str]]  # List of (entity, relation, entity)
    confidence: float
    evidence: List[str]


@dataclass
class Inference:
    """Represents an inferred fact."""
    subject: str
    predicate: str
    object: str
    confidence: float
    derivation: List[str]  # Steps used to derive this fact
    supporting_facts: List[Tuple[str, str, str]]


@dataclass
class Conflict:
    """Represents a conflict in the knowledge graph."""
    fact1: Tuple[str, str, str]
    fact2: Tuple[str, str, str]
    conflict_type: str  # 'contradiction', 'inconsistency', 'temporal'
    resolution: Optional[str] = None
    confidence_fact1: float = 0.5
    confidence_fact2: float = 0.5


class KnowledgeReasoner:
    """Perform reasoning over knowledge graphs."""

    def __init__(self, knowledge_graph: Dict[str, List[Tuple[str, str]]]):
        """Initialize reasoner.

        Args:
            knowledge_graph: Dict mapping subjects to list of (predicate, object) tuples
        """
        self.kg = knowledge_graph
        self.reverse_kg = self._build_reverse_graph()
        self.rules = self._initialize_rules()
        self.inferences = []

    def _build_reverse_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """Build reverse graph for efficient backward reasoning."""
        reverse = defaultdict(list)

        for subject, predicates in self.kg.items():
            for predicate, obj in predicates:
                reverse[obj].append((predicate, subject))

        return dict(reverse)

    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize reasoning rules."""
        return [
            # Transitivity rules
            {
                'name': 'subclass_transitivity',
                'premise': [('?x', 'subclass_of', '?y'), ('?y', 'subclass_of', '?z')],
                'conclusion': ('?x', 'subclass_of', '?z'),
                'confidence': 0.9
            },
            {
                'name': 'part_of_transitivity',
                'premise': [('?x', 'part_of', '?y'), ('?y', 'part_of', '?z')],
                'conclusion': ('?x', 'part_of', '?z'),
                'confidence': 0.85
            },
            # Inverse rules
            {
                'name': 'inverse_author',
                'premise': [('?x', 'has_author', '?y')],
                'conclusion': ('?y', 'author_of', '?x'),
                'confidence': 1.0
            },
            # Composition rules
            {
                'name': 'uses_transitivity',
                'premise': [('?x', 'uses_technique', '?y'), ('?y', 'based_on', '?z')],
                'conclusion': ('?x', 'indirectly_uses', '?z'),
                'confidence': 0.7
            },
            # Domain-specific rules
            {
                'name': 'evaluation_implies_application',
                'premise': [('?x', 'evaluated_on', '?y')],
                'conclusion': ('?x', 'applicable_to', '?y'),
                'confidence': 0.8
            },
            {
                'name': 'improvement_hierarchy',
                'premise': [('?x', 'outperforms', '?y'), ('?y', 'outperforms', '?z')],
                'conclusion': ('?x', 'outperforms', '?z'),
                'confidence': 0.75
            }
        ]

    def forward_chain(self, max_iterations: int = 10) -> List[Inference]:
        """Perform forward chaining inference.

        Args:
            max_iterations: Maximum inference iterations

        Returns:
            List of inferred facts
        """
        inferred = []
        new_facts = True
        iteration = 0

        while new_facts and iteration < max_iterations:
            new_facts = False
            iteration += 1

            for rule in self.rules:
                # Find all possible bindings
                bindings = self._find_rule_bindings(rule['premise'])

                for binding in bindings:
                    # Apply rule with binding
                    conclusion = self._apply_binding(rule['conclusion'], binding)

                    # Check if fact already exists
                    if not self._fact_exists(conclusion):
                        # Add new inference
                        inference = Inference(
                            subject=conclusion[0],
                            predicate=conclusion[1],
                            object=conclusion[2],
                            confidence=rule['confidence'],
                            derivation=[rule['name']],
                            supporting_facts=[self._apply_binding(p, binding) for p in rule['premise']]
                        )

                        inferred.append(inference)
                        self._add_fact(conclusion)
                        new_facts = True

        self.inferences.extend(inferred)
        return inferred

    def backward_chain(self, query: Tuple[str, str, str],
                      max_depth: int = 5) -> List[ReasoningPath]:
        """Perform backward chaining to prove a query.

        Args:
            query: Query triple (subject, predicate, object)
            max_depth: Maximum reasoning depth

        Returns:
            List of reasoning paths that prove the query
        """
        paths = []
        visited = set()

        def prove(goal: Tuple[str, str, str], depth: int,
                 current_path: List[Tuple[str, str, str]]) -> bool:
            """Recursively prove a goal."""
            if depth > max_depth:
                return False

            if goal in visited:
                return False

            visited.add(goal)

            # Check if fact exists
            if self._fact_exists(goal):
                paths.append(ReasoningPath(
                    start_entity=current_path[0][0] if current_path else goal[0],
                    end_entity=goal[2],
                    path=current_path + [goal],
                    confidence=1.0,
                    evidence=['direct_fact']
                ))
                return True

            # Try to prove using rules
            found = False
            for rule in self.rules:
                # Check if rule conclusion matches goal
                unifier = self._unify(rule['conclusion'], goal)
                if unifier:
                    # Try to prove premises
                    all_premises_proved = True
                    premise_paths = []

                    for premise in rule['premise']:
                        premise_goal = self._apply_binding(premise, unifier)
                        if not prove(premise_goal, depth + 1, current_path + [goal]):
                            all_premises_proved = False
                            break
                        premise_paths.append(premise_goal)

                    if all_premises_proved:
                        paths.append(ReasoningPath(
                            start_entity=current_path[0][0] if current_path else goal[0],
                            end_entity=goal[2],
                            path=current_path + premise_paths + [goal],
                            confidence=rule['confidence'],
                            evidence=[rule['name']]
                        ))
                        found = True

            return found

        prove(query, 0, [])
        return paths

    def _find_rule_bindings(self, premises: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Find all possible variable bindings for rule premises."""
        if not premises:
            return []

        bindings = []

        # Get first premise
        first_premise = premises[0]

        # Find all facts matching first premise
        for subject, predicates in self.kg.items():
            for predicate, obj in predicates:
                # Try to unify with first premise
                unifier = self._unify(first_premise, (subject, predicate, obj))

                if unifier:
                    # Check if remaining premises can be satisfied
                    if len(premises) == 1:
                        bindings.append(unifier)
                    else:
                        # Recursively check remaining premises
                        remaining = [self._apply_binding(p, unifier) for p in premises[1:]]
                        if self._all_facts_exist(remaining):
                            bindings.append(unifier)

        return bindings

    def _unify(self, pattern: Tuple[str, str, str],
               fact: Tuple[str, str, str]) -> Optional[Dict[str, str]]:
        """Unify a pattern with a fact."""
        binding = {}

        for p, f in zip(pattern, fact):
            if p.startswith('?'):
                # Variable
                if p in binding:
                    if binding[p] != f:
                        return None  # Inconsistent binding
                else:
                    binding[p] = f
            else:
                # Constant
                if p != f:
                    return None  # Mismatch

        return binding

    def _apply_binding(self, pattern: Tuple[str, str, str],
                      binding: Dict[str, str]) -> Tuple[str, str, str]:
        """Apply variable binding to pattern."""
        result = []
        for element in pattern:
            if element.startswith('?'):
                result.append(binding.get(element, element))
            else:
                result.append(element)
        return tuple(result)

    def _fact_exists(self, fact: Tuple[str, str, str]) -> bool:
        """Check if a fact exists in the knowledge graph."""
        subject, predicate, obj = fact

        if subject in self.kg:
            for pred, o in self.kg[subject]:
                if pred == predicate and o == obj:
                    return True
        return False

    def _all_facts_exist(self, facts: List[Tuple[str, str, str]]) -> bool:
        """Check if all facts exist."""
        return all(self._fact_exists(fact) for fact in facts)

    def _add_fact(self, fact: Tuple[str, str, str]):
        """Add a fact to the knowledge graph."""
        subject, predicate, obj = fact

        if subject not in self.kg:
            self.kg[subject] = []

        self.kg[subject].append((predicate, obj))

    def explain_inference(self, inference: Inference) -> str:
        """Generate human-readable explanation for an inference."""
        explanation = f"Inferred: {inference.subject} {inference.predicate} {inference.object}\n"
        explanation += f"Confidence: {inference.confidence:.2f}\n"
        explanation += "Derivation:\n"

        for step in inference.derivation:
            explanation += f"  - Applied rule: {step}\n"

        explanation += "Supporting facts:\n"
        for fact in inference.supporting_facts:
            explanation += f"  - {fact[0]} {fact[1]} {fact[2]}\n"

        return explanation


class PathFinder:
    """Find paths between entities in knowledge graph."""

    def __init__(self, knowledge_graph: Dict[str, List[Tuple[str, str]]]):
        """Initialize path finder.

        Args:
            knowledge_graph: Knowledge graph
        """
        self.kg = knowledge_graph

    def find_paths(self, start: str, end: str,
                  max_length: int = 3,
                  relation_filter: Optional[List[str]] = None) -> List[ReasoningPath]:
        """Find all paths between two entities.

        Args:
            start: Start entity
            end: End entity
            max_length: Maximum path length
            relation_filter: Optional list of allowed relations

        Returns:
            List of paths
        """
        paths = []
        visited = set()

        def dfs(current: str, target: str, current_path: List[Tuple[str, str, str]],
               depth: int):
            """Depth-first search for paths."""
            if depth > max_length:
                return

            if current == target and len(current_path) > 0:
                # Found a path
                paths.append(ReasoningPath(
                    start_entity=start,
                    end_entity=end,
                    path=current_path,
                    confidence=self._calculate_path_confidence(current_path),
                    evidence=[f"path_length_{len(current_path)}"]
                ))
                return

            if current not in self.kg:
                return

            for predicate, obj in self.kg[current]:
                # Check relation filter
                if relation_filter and predicate not in relation_filter:
                    continue

                # Avoid cycles
                path_key = (current, predicate, obj)
                if path_key in visited:
                    continue

                visited.add(path_key)
                dfs(obj, target, current_path + [path_key], depth + 1)
                visited.remove(path_key)

        dfs(start, end, [], 0)
        return paths

    def find_shortest_path(self, start: str, end: str,
                         weighted: bool = False) -> Optional[ReasoningPath]:
        """Find shortest path using BFS or Dijkstra.

        Args:
            start: Start entity
            end: End entity
            weighted: Whether to use weighted edges

        Returns:
            Shortest path or None
        """
        if weighted:
            return self._dijkstra_path(start, end)
        else:
            return self._bfs_path(start, end)

    def _bfs_path(self, start: str, end: str) -> Optional[ReasoningPath]:
        """Find shortest path using BFS."""
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == end:
                return ReasoningPath(
                    start_entity=start,
                    end_entity=end,
                    path=path,
                    confidence=self._calculate_path_confidence(path),
                    evidence=['shortest_path']
                )

            if current not in self.kg:
                continue

            for predicate, obj in self.kg[current]:
                if obj not in visited:
                    visited.add(obj)
                    new_path = path + [(current, predicate, obj)]
                    queue.append((obj, new_path))

        return None

    def _dijkstra_path(self, start: str, end: str) -> Optional[ReasoningPath]:
        """Find shortest weighted path using Dijkstra."""
        import heapq

        # Priority queue: (cost, node, path)
        pq = [(0, start, [])]
        visited = set()
        costs = {start: 0}

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                return ReasoningPath(
                    start_entity=start,
                    end_entity=end,
                    path=path,
                    confidence=1.0 / (1 + cost),  # Convert cost to confidence
                    evidence=['weighted_shortest_path']
                )

            if current not in self.kg:
                continue

            for predicate, obj in self.kg[current]:
                if obj not in visited:
                    # Edge weight (can be customized)
                    edge_weight = self._get_edge_weight(current, predicate, obj)
                    new_cost = cost + edge_weight

                    if obj not in costs or new_cost < costs[obj]:
                        costs[obj] = new_cost
                        new_path = path + [(current, predicate, obj)]
                        heapq.heappush(pq, (new_cost, obj, new_path))

        return None

    def _get_edge_weight(self, source: str, relation: str, target: str) -> float:
        """Get edge weight for Dijkstra."""
        # Can be customized based on relation type
        relation_weights = {
            'subclass_of': 0.5,
            'part_of': 0.6,
            'uses_technique': 0.7,
            'evaluated_on': 0.8,
            'outperforms': 0.9
        }
        return relation_weights.get(relation, 1.0)

    def _calculate_path_confidence(self, path: List[Tuple[str, str, str]]) -> float:
        """Calculate confidence score for a path."""
        if not path:
            return 0.0

        # Decay based on path length
        length_factor = 1.0 / (1 + len(path))

        # Relation reliability
        relation_scores = {
            'has_type': 1.0,
            'subclass_of': 0.9,
            'uses_technique': 0.8,
            'evaluated_on': 0.85,
            'outperforms': 0.7
        }

        avg_relation_score = np.mean([
            relation_scores.get(r, 0.5) for _, r, _ in path
        ])

        return length_factor * avg_relation_score

    def find_common_neighbors(self, entity1: str, entity2: str) -> List[str]:
        """Find entities connected to both input entities."""
        neighbors1 = set()
        neighbors2 = set()

        if entity1 in self.kg:
            neighbors1 = {obj for _, obj in self.kg[entity1]}

        if entity2 in self.kg:
            neighbors2 = {obj for _, obj in self.kg[entity2]}

        return list(neighbors1.intersection(neighbors2))


class LinkPredictor:
    """Predict missing links in knowledge graph."""

    def __init__(self, knowledge_graph: Dict[str, List[Tuple[str, str]]],
                 embeddings: Optional[Dict[str, np.ndarray]] = None):
        """Initialize link predictor.

        Args:
            knowledge_graph: Knowledge graph
            embeddings: Optional entity embeddings
        """
        self.kg = knowledge_graph
        self.embeddings = embeddings
        self.path_finder = PathFinder(knowledge_graph)

    def predict_links(self, entity: str, relation: str,
                     candidates: Optional[List[str]] = None,
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict missing links for a given entity and relation.

        Args:
            entity: Source entity
            relation: Relation type
            candidates: Optional candidate entities
            top_k: Number of top predictions

        Returns:
            List of (entity, score) predictions
        """
        if candidates is None:
            candidates = list(self.kg.keys())

        predictions = []

        for candidate in candidates:
            # Skip if link already exists
            if self._link_exists(entity, relation, candidate):
                continue

            score = self._score_link(entity, relation, candidate)
            predictions.append((candidate, score))

        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def _link_exists(self, entity: str, relation: str, target: str) -> bool:
        """Check if a link already exists."""
        if entity in self.kg:
            for rel, obj in self.kg[entity]:
                if rel == relation and obj == target:
                    return True
        return False

    def _score_link(self, entity: str, relation: str, target: str) -> float:
        """Score a potential link."""
        score = 0.0

        # Path-based features
        paths = self.path_finder.find_paths(entity, target, max_length=3)
        if paths:
            # Shorter paths indicate stronger connection
            min_length = min(len(p.path) for p in paths)
            path_score = 1.0 / (1 + min_length)
            score += 0.3 * path_score

        # Common neighbors
        common = self.path_finder.find_common_neighbors(entity, target)
        neighbor_score = len(common) / (len(self.kg.get(entity, [])) + 1)
        score += 0.2 * neighbor_score

        # Embedding similarity (if available)
        if self.embeddings and entity in self.embeddings and target in self.embeddings:
            emb_sim = np.dot(self.embeddings[entity], self.embeddings[target])
            emb_sim = emb_sim / (np.linalg.norm(self.embeddings[entity]) *
                                 np.linalg.norm(self.embeddings[target]) + 1e-8)
            score += 0.3 * (emb_sim + 1) / 2  # Normalize to [0, 1]

        # Relation-specific patterns
        relation_score = self._relation_pattern_score(entity, relation, target)
        score += 0.2 * relation_score

        return min(score, 1.0)

    def _relation_pattern_score(self, entity: str, relation: str, target: str) -> float:
        """Score based on relation-specific patterns."""
        # Count how often this relation pattern occurs
        pattern_count = 0
        total_relations = 0

        for subj in self.kg:
            for rel, _ in self.kg[subj]:
                if rel == relation:
                    pattern_count += 1
                total_relations += 1

        if total_relations == 0:
            return 0.5

        # Relation frequency
        rel_frequency = pattern_count / total_relations

        # Type compatibility (simplified)
        type_compatible = self._check_type_compatibility(entity, relation, target)

        return 0.5 * rel_frequency + 0.5 * type_compatible

    def _check_type_compatibility(self, entity: str, relation: str, target: str) -> float:
        """Check if entity types are compatible with relation."""
        # Simplified type checking
        # In practice, would use entity type information

        compatible_patterns = {
            'uses_technique': ('method', 'method'),
            'evaluated_on': ('method', 'dataset'),
            'has_author': ('paper', 'author'),
            'outperforms': ('method', 'method'),
        }

        if relation in compatible_patterns:
            # Would check actual entity types here
            return 0.8
        return 0.5


class ConflictResolver:
    """Resolve conflicts in knowledge graph."""

    def __init__(self, knowledge_graph: Dict[str, List[Tuple[str, str]]]):
        """Initialize conflict resolver.

        Args:
            knowledge_graph: Knowledge graph
        """
        self.kg = knowledge_graph
        self.conflicts = []

    def detect_conflicts(self) -> List[Conflict]:
        """Detect conflicts in the knowledge graph."""
        conflicts = []

        # Check for contradictions
        contradictions = self._detect_contradictions()
        conflicts.extend(contradictions)

        # Check for inconsistencies
        inconsistencies = self._detect_inconsistencies()
        conflicts.extend(inconsistencies)

        # Check for temporal conflicts
        temporal = self._detect_temporal_conflicts()
        conflicts.extend(temporal)

        self.conflicts = conflicts
        return conflicts

    def _detect_contradictions(self) -> List[Conflict]:
        """Detect direct contradictions."""
        conflicts = []
        contradictory_predicates = [
            ('outperforms', 'underperforms'),
            ('improves', 'degrades'),
            ('is_a', 'is_not_a')
        ]

        for subject in self.kg:
            predicates = self.kg[subject]

            for pred1, obj1 in predicates:
                for pred2, obj2 in predicates:
                    # Check for contradictory predicates with same object
                    for contra1, contra2 in contradictory_predicates:
                        if pred1 == contra1 and pred2 == contra2 and obj1 == obj2:
                            conflicts.append(Conflict(
                                fact1=(subject, pred1, obj1),
                                fact2=(subject, pred2, obj2),
                                conflict_type='contradiction'
                            ))

        return conflicts

    def _detect_inconsistencies(self) -> List[Conflict]:
        """Detect logical inconsistencies."""
        conflicts = []

        # Check for cyclic outperformance
        for subject in self.kg:
            visited = set()
            if self._has_cycle(subject, 'outperforms', visited):
                # Simplified: report first found cycle
                conflicts.append(Conflict(
                    fact1=(subject, 'outperforms', 'cycle_detected'),
                    fact2=(subject, 'outperforms', 'cycle_detected'),
                    conflict_type='inconsistency'
                ))

        return conflicts

    def _detect_temporal_conflicts(self) -> List[Conflict]:
        """Detect temporal conflicts."""
        conflicts = []

        # Check for anachronisms (simplified)
        for subject in self.kg:
            years = []
            for pred, obj in self.kg[subject]:
                if pred == 'published_in':
                    try:
                        years.append(int(obj))
                    except ValueError:
                        pass

            if len(years) > 1 and len(set(years)) > 1:
                conflicts.append(Conflict(
                    fact1=(subject, 'published_in', str(min(years))),
                    fact2=(subject, 'published_in', str(max(years))),
                    conflict_type='temporal'
                ))

        return conflicts

    def _has_cycle(self, start: str, relation: str, visited: Set[str]) -> bool:
        """Check for cycles in a specific relation."""
        if start in visited:
            return True

        visited.add(start)

        if start in self.kg:
            for pred, obj in self.kg[start]:
                if pred == relation:
                    if self._has_cycle(obj, relation, visited.copy()):
                        return True

        return False

    def resolve_conflict(self, conflict: Conflict,
                        strategy: str = 'confidence') -> Conflict:
        """Resolve a conflict using specified strategy.

        Args:
            conflict: Conflict to resolve
            strategy: Resolution strategy ('confidence', 'recency', 'majority', 'manual')

        Returns:
            Resolved conflict
        """
        if strategy == 'confidence':
            resolution = self._resolve_by_confidence(conflict)
        elif strategy == 'recency':
            resolution = self._resolve_by_recency(conflict)
        elif strategy == 'majority':
            resolution = self._resolve_by_majority(conflict)
        else:
            resolution = 'manual_review_required'

        conflict.resolution = resolution
        return conflict

    def _resolve_by_confidence(self, conflict: Conflict) -> str:
        """Resolve based on confidence scores."""
        if conflict.confidence_fact1 > conflict.confidence_fact2:
            return f"Keep fact1: {conflict.fact1}"
        elif conflict.confidence_fact2 > conflict.confidence_fact1:
            return f"Keep fact2: {conflict.fact2}"
        else:
            return "Equal confidence - manual review needed"

    def _resolve_by_recency(self, conflict: Conflict) -> str:
        """Resolve by keeping more recent fact."""
        # Would need timestamp information
        return "Recency information not available"

    def _resolve_by_majority(self, conflict: Conflict) -> str:
        """Resolve by majority vote from sources."""
        # Count supporting sources for each fact
        # Simplified implementation
        return "Majority voting requires multiple sources"

    def auto_resolve_all(self, strategy: str = 'confidence') -> List[Conflict]:
        """Automatically resolve all detected conflicts.

        Args:
            strategy: Resolution strategy

        Returns:
            List of resolved conflicts
        """
        if not self.conflicts:
            self.detect_conflicts()

        resolved = []
        for conflict in self.conflicts:
            resolved.append(self.resolve_conflict(conflict, strategy))

        return resolved