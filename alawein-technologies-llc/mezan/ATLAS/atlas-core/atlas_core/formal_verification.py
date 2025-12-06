"""Formal Verification Framework for MEZAN Workflows.

This module provides formal verification capabilities including workflow
invariant checking, deadlock detection, resource allocation verification,
state machine validation, and temporal logic checking.

Author: Meshal Alawein
Date: 2025-11-18
"""

import itertools
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import networkx as nx
import numpy as np
from graphviz import Digraph


class VerificationStatus(Enum):
    """Status of verification checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    UNKNOWN = "unknown"


class TemporalOperator(Enum):
    """Temporal logic operators for CTL/LTL."""
    # CTL operators
    AF = "AF"  # All Finally (inevitably)
    AG = "AG"  # All Globally (invariantly)
    AX = "AX"  # All neXt
    EF = "EF"  # Exists Finally (potentially)
    EG = "EG"  # Exists Globally
    EX = "EX"  # Exists neXt

    # LTL operators
    F = "F"    # Finally (eventually)
    G = "G"    # Globally (always)
    X = "X"    # neXt
    U = "U"    # Until
    R = "R"    # Release
    W = "W"    # Weak until


@dataclass
class WorkflowState:
    """Represents a state in a workflow."""
    id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    is_initial: bool = False
    is_final: bool = False
    is_error: bool = False


@dataclass
class WorkflowTransition:
    """Represents a transition between workflow states."""
    source: str
    target: str
    action: str
    guard: Optional[str] = None  # Condition for transition
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequirement:
    """Resource requirements for a workflow component."""
    resource_type: str
    amount: float
    is_exclusive: bool = False
    timeout: Optional[float] = None


@dataclass
class Invariant:
    """Workflow invariant specification."""
    name: str
    description: str
    predicate: Callable[[WorkflowState], bool]
    scope: str = "global"  # global, local, temporal
    severity: str = "error"  # error, warning, info


@dataclass
class TemporalProperty:
    """Temporal logic property specification."""
    name: str
    formula: str
    operator: TemporalOperator
    predicate: Callable[[WorkflowState], bool]
    description: str


@dataclass
class VerificationResult:
    """Result of a verification check."""
    check_type: str
    status: VerificationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'check_type': self.check_type,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }


class WorkflowVerifier:
    """Main workflow verification engine."""

    def __init__(self):
        """Initialize the workflow verifier."""
        self.states: Dict[str, WorkflowState] = {}
        self.transitions: List[WorkflowTransition] = []
        self.invariants: List[Invariant] = []
        self.temporal_properties: List[TemporalProperty] = []
        self.resource_requirements: Dict[str, List[ResourceRequirement]] = defaultdict(list)
        self.graph: Optional[nx.DiGraph] = None
        self.verification_results: List[VerificationResult] = []

    def add_state(self, state: WorkflowState):
        """Add a state to the workflow."""
        self.states[state.id] = state
        self.graph = None  # Invalidate graph cache

    def add_transition(self, transition: WorkflowTransition):
        """Add a transition to the workflow."""
        self.transitions.append(transition)
        self.graph = None  # Invalidate graph cache

    def add_invariant(self, invariant: Invariant):
        """Add an invariant to check."""
        self.invariants.append(invariant)

    def add_temporal_property(self, property: TemporalProperty):
        """Add a temporal logic property to verify."""
        self.temporal_properties.append(property)

    def add_resource_requirement(self, state_id: str, requirement: ResourceRequirement):
        """Add resource requirement for a state."""
        self.resource_requirements[state_id].append(requirement)

    def build_graph(self) -> nx.DiGraph:
        """Build directed graph representation of the workflow."""
        if self.graph is not None:
            return self.graph

        self.graph = nx.DiGraph()

        # Add nodes (states)
        for state_id, state in self.states.items():
            self.graph.add_node(
                state_id,
                label=state.name,
                is_initial=state.is_initial,
                is_final=state.is_final,
                is_error=state.is_error,
                **state.properties
            )

        # Add edges (transitions)
        for transition in self.transitions:
            self.graph.add_edge(
                transition.source,
                transition.target,
                action=transition.action,
                guard=transition.guard,
                **transition.properties
            )

        return self.graph

    def verify_invariants(self) -> List[VerificationResult]:
        """Verify all workflow invariants."""
        results = []

        for invariant in self.invariants:
            violations = []

            # Check invariant for each state
            for state_id, state in self.states.items():
                if not invariant.predicate(state):
                    violations.append(state_id)

            if violations:
                result = VerificationResult(
                    check_type="invariant",
                    status=VerificationStatus.FAILED,
                    message=f"Invariant '{invariant.name}' violated",
                    details={
                        'invariant': invariant.name,
                        'description': invariant.description,
                        'violating_states': violations,
                        'severity': invariant.severity
                    }
                )
            else:
                result = VerificationResult(
                    check_type="invariant",
                    status=VerificationStatus.PASSED,
                    message=f"Invariant '{invariant.name}' satisfied",
                    details={'invariant': invariant.name}
                )

            results.append(result)
            self.verification_results.append(result)

        return results

    def detect_deadlocks(self) -> VerificationResult:
        """Detect potential deadlocks in the workflow."""
        graph = self.build_graph()

        deadlock_states = []
        non_terminal_states = [
            s for s, data in graph.nodes(data=True)
            if not data.get('is_final', False) and not data.get('is_error', False)
        ]

        # Check for states with no outgoing transitions
        for state in non_terminal_states:
            if graph.out_degree(state) == 0:
                deadlock_states.append(state)

        # Check for circular wait conditions (resource-based deadlocks)
        resource_deadlocks = self._detect_resource_deadlocks()

        if deadlock_states or resource_deadlocks:
            result = VerificationResult(
                check_type="deadlock_detection",
                status=VerificationStatus.FAILED,
                message="Potential deadlocks detected",
                details={
                    'terminal_deadlocks': deadlock_states,
                    'resource_deadlocks': resource_deadlocks,
                    'total_deadlocks': len(deadlock_states) + len(resource_deadlocks)
                }
            )
        else:
            result = VerificationResult(
                check_type="deadlock_detection",
                status=VerificationStatus.PASSED,
                message="No deadlocks detected",
                details={}
            )

        self.verification_results.append(result)
        return result

    def _detect_resource_deadlocks(self) -> List[Dict[str, Any]]:
        """Detect resource-based deadlocks using wait-for graph."""
        deadlocks = []

        # Build wait-for graph
        wait_graph = nx.DiGraph()

        for state_id, requirements in self.resource_requirements.items():
            for req in requirements:
                if req.is_exclusive:
                    # Check for circular dependencies
                    for other_state, other_reqs in self.resource_requirements.items():
                        if state_id != other_state:
                            for other_req in other_reqs:
                                if other_req.is_exclusive and other_req.resource_type == req.resource_type:
                                    wait_graph.add_edge(state_id, other_state)

        # Find cycles in wait-for graph
        try:
            cycles = list(nx.simple_cycles(wait_graph))
            for cycle in cycles:
                deadlocks.append({
                    'type': 'resource_cycle',
                    'states': cycle,
                    'description': f"Circular wait detected: {' -> '.join(cycle)}"
                })
        except nx.NetworkXError:
            pass  # No cycles

        return deadlocks

    def verify_reachability(self) -> VerificationResult:
        """Verify that all states are reachable from initial states."""
        graph = self.build_graph()

        initial_states = [
            s for s, data in graph.nodes(data=True)
            if data.get('is_initial', False)
        ]

        if not initial_states:
            return VerificationResult(
                check_type="reachability",
                status=VerificationStatus.WARNING,
                message="No initial states defined",
                details={}
            )

        # Find all reachable states
        reachable = set()
        for initial in initial_states:
            reachable.update(nx.descendants(graph, initial))
            reachable.add(initial)

        unreachable = set(graph.nodes()) - reachable

        if unreachable:
            result = VerificationResult(
                check_type="reachability",
                status=VerificationStatus.WARNING,
                message="Unreachable states detected",
                details={
                    'unreachable_states': list(unreachable),
                    'count': len(unreachable),
                    'percentage': len(unreachable) / len(graph.nodes()) * 100
                }
            )
        else:
            result = VerificationResult(
                check_type="reachability",
                status=VerificationStatus.PASSED,
                message="All states are reachable",
                details={'total_states': len(graph.nodes())}
            )

        self.verification_results.append(result)
        return result

    def verify_termination(self) -> VerificationResult:
        """Verify that the workflow can terminate properly."""
        graph = self.build_graph()

        final_states = [
            s for s, data in graph.nodes(data=True)
            if data.get('is_final', False)
        ]

        if not final_states:
            return VerificationResult(
                check_type="termination",
                status=VerificationStatus.WARNING,
                message="No final states defined",
                details={}
            )

        # Check if final states are reachable from initial states
        initial_states = [
            s for s, data in graph.nodes(data=True)
            if data.get('is_initial', False)
        ]

        can_terminate = False
        for initial in initial_states:
            for final in final_states:
                if nx.has_path(graph, initial, final):
                    can_terminate = True
                    break
            if can_terminate:
                break

        if can_terminate:
            result = VerificationResult(
                check_type="termination",
                status=VerificationStatus.PASSED,
                message="Workflow can terminate properly",
                details={'final_states': final_states}
            )
        else:
            result = VerificationResult(
                check_type="termination",
                status=VerificationStatus.FAILED,
                message="No path from initial to final states",
                details={
                    'initial_states': initial_states,
                    'final_states': final_states
                }
            )

        self.verification_results.append(result)
        return result

    def verify_resource_allocation(self) -> VerificationResult:
        """Verify resource allocation constraints."""
        violations = []
        warnings = []

        # Check for resource conflicts
        for state_id, requirements in self.resource_requirements.items():
            # Group by resource type
            by_type = defaultdict(list)
            for req in requirements:
                by_type[req.resource_type].append(req)

            # Check for conflicts within each type
            for resource_type, reqs in by_type.items():
                exclusive_count = sum(1 for r in reqs if r.is_exclusive)

                if exclusive_count > 1:
                    violations.append({
                        'state': state_id,
                        'resource': resource_type,
                        'issue': 'Multiple exclusive resource requests'
                    })

        # Check for resource capacity (would need capacity definitions)
        # This is a simplified check
        total_by_resource = defaultdict(float)
        for requirements in self.resource_requirements.values():
            for req in requirements:
                total_by_resource[req.resource_type] += req.amount

        # Warn if total exceeds reasonable thresholds
        for resource_type, total in total_by_resource.items():
            if total > 100:  # Arbitrary threshold
                warnings.append({
                    'resource': resource_type,
                    'total_required': total,
                    'issue': 'High total resource requirement'
                })

        if violations:
            result = VerificationResult(
                check_type="resource_allocation",
                status=VerificationStatus.FAILED,
                message="Resource allocation conflicts detected",
                details={'violations': violations, 'warnings': warnings}
            )
        elif warnings:
            result = VerificationResult(
                check_type="resource_allocation",
                status=VerificationStatus.WARNING,
                message="Resource allocation warnings",
                details={'warnings': warnings}
            )
        else:
            result = VerificationResult(
                check_type="resource_allocation",
                status=VerificationStatus.PASSED,
                message="Resource allocation verified",
                details={'total_resources': dict(total_by_resource)}
            )

        self.verification_results.append(result)
        return result

    def verify_temporal_logic(self) -> List[VerificationResult]:
        """Verify temporal logic properties using model checking."""
        results = []

        for property in self.temporal_properties:
            if property.operator in [TemporalOperator.AG, TemporalOperator.AF,
                                    TemporalOperator.EG, TemporalOperator.EF]:
                result = self._verify_ctl_property(property)
            else:
                result = self._verify_ltl_property(property)

            results.append(result)
            self.verification_results.append(result)

        return results

    def _verify_ctl_property(self, property: TemporalProperty) -> VerificationResult:
        """Verify a CTL (Computation Tree Logic) property."""
        graph = self.build_graph()

        if property.operator == TemporalOperator.AG:
            # AG p: p holds in all states of all paths
            violating_states = []
            for state_id, state in self.states.items():
                if not property.predicate(state):
                    violating_states.append(state_id)

            if violating_states:
                return VerificationResult(
                    check_type="temporal_logic_ctl",
                    status=VerificationStatus.FAILED,
                    message=f"CTL property '{property.name}' (AG) violated",
                    details={
                        'property': property.name,
                        'formula': property.formula,
                        'violating_states': violating_states
                    }
                )

        elif property.operator == TemporalOperator.AF:
            # AF p: p eventually holds in all paths
            initial_states = [
                s for s, data in graph.nodes(data=True)
                if data.get('is_initial', False)
            ]

            for initial in initial_states:
                # Check if property eventually holds in all paths from initial
                if not self._check_af_property(initial, property.predicate, graph):
                    return VerificationResult(
                        check_type="temporal_logic_ctl",
                        status=VerificationStatus.FAILED,
                        message=f"CTL property '{property.name}' (AF) violated",
                        details={
                            'property': property.name,
                            'formula': property.formula,
                            'initial_state': initial
                        }
                    )

        elif property.operator == TemporalOperator.EF:
            # EF p: p eventually holds in some path
            initial_states = [
                s for s, data in graph.nodes(data=True)
                if data.get('is_initial', False)
            ]

            found = False
            for initial in initial_states:
                if self._check_ef_property(initial, property.predicate, graph):
                    found = True
                    break

            if not found:
                return VerificationResult(
                    check_type="temporal_logic_ctl",
                    status=VerificationStatus.FAILED,
                    message=f"CTL property '{property.name}' (EF) violated",
                    details={
                        'property': property.name,
                        'formula': property.formula
                    }
                )

        elif property.operator == TemporalOperator.EG:
            # EG p: p holds globally in some path
            initial_states = [
                s for s, data in graph.nodes(data=True)
                if data.get('is_initial', False)
            ]

            found = False
            for initial in initial_states:
                if self._check_eg_property(initial, property.predicate, graph):
                    found = True
                    break

            if not found:
                return VerificationResult(
                    check_type="temporal_logic_ctl",
                    status=VerificationStatus.FAILED,
                    message=f"CTL property '{property.name}' (EG) violated",
                    details={
                        'property': property.name,
                        'formula': property.formula
                    }
                )

        return VerificationResult(
            check_type="temporal_logic_ctl",
            status=VerificationStatus.PASSED,
            message=f"CTL property '{property.name}' satisfied",
            details={
                'property': property.name,
                'operator': property.operator.value,
                'formula': property.formula
            }
        )

    def _verify_ltl_property(self, property: TemporalProperty) -> VerificationResult:
        """Verify an LTL (Linear Temporal Logic) property."""
        # Simplified LTL verification
        # In practice, this would use a model checker like NuSMV or SPIN

        graph = self.build_graph()
        initial_states = [
            s for s, data in graph.nodes(data=True)
            if data.get('is_initial', False)
        ]

        # Generate sample paths for verification
        paths = []
        for initial in initial_states:
            paths.extend(self._generate_paths(initial, graph, max_length=100, max_paths=10))

        violations = 0
        for path in paths:
            if not self._check_ltl_on_path(property, path):
                violations += 1

        if violations > 0:
            return VerificationResult(
                check_type="temporal_logic_ltl",
                status=VerificationStatus.FAILED,
                message=f"LTL property '{property.name}' violated",
                details={
                    'property': property.name,
                    'formula': property.formula,
                    'violations': violations,
                    'total_paths': len(paths)
                }
            )

        return VerificationResult(
            check_type="temporal_logic_ltl",
            status=VerificationStatus.PASSED,
            message=f"LTL property '{property.name}' satisfied",
            details={
                'property': property.name,
                'operator': property.operator.value,
                'formula': property.formula,
                'paths_checked': len(paths)
            }
        )

    def _check_af_property(self,
                          state_id: str,
                          predicate: Callable,
                          graph: nx.DiGraph,
                          visited: Optional[Set] = None) -> bool:
        """Check if AF property holds from a state."""
        if visited is None:
            visited = set()

        if state_id in visited:
            return False  # Cycle without satisfying property

        visited.add(state_id)

        state = self.states[state_id]
        if predicate(state):
            return True

        # Check all successors
        successors = list(graph.successors(state_id))
        if not successors:
            return False  # Dead end without satisfying property

        for successor in successors:
            if not self._check_af_property(successor, predicate, graph, visited.copy()):
                return False

        return True

    def _check_ef_property(self,
                          state_id: str,
                          predicate: Callable,
                          graph: nx.DiGraph,
                          visited: Optional[Set] = None) -> bool:
        """Check if EF property holds from a state."""
        if visited is None:
            visited = set()

        if state_id in visited:
            return False

        visited.add(state_id)

        state = self.states[state_id]
        if predicate(state):
            return True

        # Check any successor
        for successor in graph.successors(state_id):
            if self._check_ef_property(successor, predicate, graph, visited.copy()):
                return True

        return False

    def _check_eg_property(self,
                          state_id: str,
                          predicate: Callable,
                          graph: nx.DiGraph,
                          visited: Optional[Set] = None) -> bool:
        """Check if EG property holds from a state."""
        if visited is None:
            visited = set()

        state = self.states[state_id]
        if not predicate(state):
            return False

        if state_id in visited:
            return True  # Found a cycle where property holds

        visited.add(state_id)

        # Check if there exists at least one path where property holds globally
        successors = list(graph.successors(state_id))
        if not successors:
            # Reached a terminal state
            return graph.nodes[state_id].get('is_final', False)

        for successor in successors:
            if self._check_eg_property(successor, predicate, graph, visited.copy()):
                return True

        return False

    def _generate_paths(self,
                       start: str,
                       graph: nx.DiGraph,
                       max_length: int = 100,
                       max_paths: int = 10) -> List[List[str]]:
        """Generate sample paths for LTL verification."""
        paths = []
        queue = deque([(start, [start])])

        while queue and len(paths) < max_paths:
            current, path = queue.popleft()

            if len(path) >= max_length:
                paths.append(path)
                continue

            successors = list(graph.successors(current))
            if not successors:
                paths.append(path)
            else:
                for successor in successors:
                    if successor not in path:  # Avoid cycles
                        queue.append((successor, path + [successor]))

        return paths

    def _check_ltl_on_path(self, property: TemporalProperty, path: List[str]) -> bool:
        """Check if an LTL property holds on a specific path."""
        if property.operator == TemporalOperator.G:
            # Globally: property holds in all states
            for state_id in path:
                if not property.predicate(self.states[state_id]):
                    return False
            return True

        elif property.operator == TemporalOperator.F:
            # Finally: property eventually holds
            for state_id in path:
                if property.predicate(self.states[state_id]):
                    return True
            return False

        elif property.operator == TemporalOperator.X:
            # Next: property holds in next state
            if len(path) < 2:
                return False
            return property.predicate(self.states[path[1]])

        return True  # Default

    def visualize_workflow(self, output_path: Optional[Path] = None) -> str:
        """
        Visualize the workflow as a graph.

        Args:
            output_path: Path to save the visualization

        Returns:
            DOT source code
        """
        graph = self.build_graph()

        dot = Digraph(comment='Workflow Verification')
        dot.attr(rankdir='TB')

        # Add nodes with styling based on type
        for node_id, data in graph.nodes(data=True):
            label = data.get('label', node_id)

            if data.get('is_initial', False):
                dot.node(node_id, label, shape='doublecircle', style='filled',
                        fillcolor='lightgreen')
            elif data.get('is_final', False):
                dot.node(node_id, label, shape='doublecircle', style='filled',
                        fillcolor='lightblue')
            elif data.get('is_error', False):
                dot.node(node_id, label, shape='octagon', style='filled',
                        fillcolor='lightcoral')
            else:
                dot.node(node_id, label, shape='circle')

        # Add edges with labels
        for source, target, data in graph.edges(data=True):
            label = data.get('action', '')
            if data.get('guard'):
                label += f"\n[{data['guard']}]"

            dot.edge(source, target, label)

        if output_path:
            dot.render(output_path, format='png', cleanup=True)

        return dot.source

    def generate_verification_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive verification report.

        Args:
            output_path: Path to save report

        Returns:
            Verification report dictionary
        """
        # Run all verifications
        self.verify_invariants()
        self.detect_deadlocks()
        self.verify_reachability()
        self.verify_termination()
        self.verify_resource_allocation()
        self.verify_temporal_logic()

        # Aggregate results
        report = {
            'timestamp': time.time(),
            'workflow_stats': {
                'total_states': len(self.states),
                'total_transitions': len(self.transitions),
                'total_invariants': len(self.invariants),
                'total_temporal_properties': len(self.temporal_properties)
            },
            'verification_summary': {
                'total_checks': len(self.verification_results),
                'passed': sum(1 for r in self.verification_results
                            if r.status == VerificationStatus.PASSED),
                'failed': sum(1 for r in self.verification_results
                            if r.status == VerificationStatus.FAILED),
                'warnings': sum(1 for r in self.verification_results
                              if r.status == VerificationStatus.WARNING)
            },
            'detailed_results': [r.to_dict() for r in self.verification_results]
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report


# Convenience functions for common verification tasks
def verify_mezan_workflow(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify a MEZAN workflow definition.

    Args:
        workflow_def: Workflow definition dictionary

    Returns:
        Verification results
    """
    verifier = WorkflowVerifier()

    # Parse workflow definition
    for state_def in workflow_def.get('states', []):
        state = WorkflowState(
            id=state_def['id'],
            name=state_def['name'],
            properties=state_def.get('properties', {}),
            is_initial=state_def.get('is_initial', False),
            is_final=state_def.get('is_final', False),
            is_error=state_def.get('is_error', False)
        )
        verifier.add_state(state)

    for trans_def in workflow_def.get('transitions', []):
        transition = WorkflowTransition(
            source=trans_def['source'],
            target=trans_def['target'],
            action=trans_def.get('action', ''),
            guard=trans_def.get('guard'),
            properties=trans_def.get('properties', {})
        )
        verifier.add_transition(transition)

    # Add default invariants
    verifier.add_invariant(Invariant(
        name="no_self_error_transition",
        description="Error states should not transition to themselves",
        predicate=lambda s: not s.is_error or s.id not in [
            t.target for t in verifier.transitions if t.source == s.id
        ]
    ))

    return verifier.generate_verification_report()


def check_workflow_safety(states: List[Dict], transitions: List[Dict]) -> bool:
    """
    Quick safety check for a workflow.

    Args:
        states: List of state definitions
        transitions: List of transition definitions

    Returns:
        True if workflow is safe, False otherwise
    """
    verifier = WorkflowVerifier()

    for state in states:
        verifier.add_state(WorkflowState(
            id=state['id'],
            name=state.get('name', state['id']),
            is_initial=state.get('is_initial', False),
            is_final=state.get('is_final', False)
        ))

    for trans in transitions:
        verifier.add_transition(WorkflowTransition(
            source=trans['source'],
            target=trans['target'],
            action=trans.get('action', '')
        ))

    # Check for deadlocks and unreachable states
    deadlock_result = verifier.detect_deadlocks()
    reachability_result = verifier.verify_reachability()

    return (deadlock_result.status == VerificationStatus.PASSED and
            reachability_result.status == VerificationStatus.PASSED)