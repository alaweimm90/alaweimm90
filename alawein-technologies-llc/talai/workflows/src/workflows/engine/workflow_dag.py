"""
Workflow DAG - Directed Acyclic Graph representation for workflows

Provides the core DAG structure for representing complex workflows with
conditional branching, parallel execution paths, and dynamic node creation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import uuid
import networkx as nx


class NodeStatus(Enum):
    """Status of a workflow node"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Type of workflow node"""
    START = "start"
    END = "end"
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    FORK = "fork"
    JOIN = "join"
    SUBWORKFLOW = "subworkflow"
    LOOP = "loop"
    WAIT = "wait"
    WEBHOOK = "webhook"
    HUMAN_APPROVAL = "human_approval"


@dataclass
class NodeCondition:
    """Conditional logic for node execution"""
    expression: str
    evaluate: Optional[Callable[[Dict[str, Any]], bool]] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None

    def check(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context"""
        if self.evaluate:
            return self.evaluate(context)
        # Simple expression evaluation
        try:
            return eval(self.expression, {"__builtins__": {}}, context)
        except:
            return False


@dataclass
class NodeMetadata:
    """Metadata for workflow nodes"""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: int = 5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    cost_estimate: Optional[float] = None


@dataclass
class WorkflowNode:
    """Individual node in workflow DAG"""
    id: str
    name: str
    type: NodeType
    handler: Optional[Callable] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.PENDING
    condition: Optional[NodeCondition] = None
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.name}_{uuid.uuid4().hex[:8]}"

    def is_ready(self, completed_nodes: Set[str]) -> bool:
        """Check if node is ready to execute"""
        if self.status != NodeStatus.PENDING:
            return False
        return all(dep in completed_nodes for dep in self.dependencies)

    def add_dependency(self, node_id: str):
        """Add dependency to this node"""
        self.dependencies.add(node_id)

    def add_dependent(self, node_id: str):
        """Add dependent node"""
        self.dependents.add(node_id)

    def reset(self):
        """Reset node state for re-execution"""
        self.status = NodeStatus.PENDING
        self.error = None
        self.start_time = None
        self.end_time = None
        self.retry_count = 0
        self.outputs = {}


class WorkflowDAG:
    """Directed Acyclic Graph for workflow representation"""

    def __init__(self, workflow_id: Optional[str] = None, name: str = "workflow"):
        self.id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.graph = nx.DiGraph()
        self.start_node_id: Optional[str] = None
        self.end_node_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "checksum": None
        }

        # Create default start and end nodes
        self._create_sentinel_nodes()

    def _create_sentinel_nodes(self):
        """Create start and end nodes"""
        start_node = WorkflowNode(
            id="start",
            name="Start",
            type=NodeType.START
        )
        end_node = WorkflowNode(
            id="end",
            name="End",
            type=NodeType.END
        )

        self.add_node(start_node)
        self.add_node(end_node)
        self.start_node_id = "start"
        self.end_node_id = "end"

    def add_node(self, node: WorkflowNode) -> str:
        """Add node to DAG"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node=node)
        return node.id

    def add_edge(self, from_node_id: str, to_node_id: str,
                 condition: Optional[NodeCondition] = None):
        """Add edge between nodes"""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found")

        self.graph.add_edge(from_node_id, to_node_id, condition=condition)
        self.nodes[from_node_id].add_dependent(to_node_id)
        self.nodes[to_node_id].add_dependency(from_node_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_node_id, to_node_id)
            self.nodes[from_node_id].dependents.discard(to_node_id)
            self.nodes[to_node_id].dependencies.discard(from_node_id)
            raise ValueError(f"Adding edge creates cycle: {from_node_id} -> {to_node_id}")

    def remove_node(self, node_id: str):
        """Remove node from DAG"""
        if node_id not in self.nodes:
            return

        # Update dependencies
        node = self.nodes[node_id]
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.discard(node_id)

        for dependent_id in node.dependents:
            if dependent_id in self.nodes:
                self.nodes[dependent_id].dependencies.discard(node_id)

        # Remove from graph and nodes
        self.graph.remove_node(node_id)
        del self.nodes[node_id]

    def get_ready_nodes(self, completed_nodes: Set[str]) -> List[WorkflowNode]:
        """Get all nodes ready for execution"""
        ready = []
        for node in self.nodes.values():
            if node.is_ready(completed_nodes):
                ready.append(node)
        return sorted(ready, key=lambda n: n.metadata.priority, reverse=True)

    def get_execution_order(self) -> List[List[str]]:
        """Get topological levels for execution"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Workflow contains cycles")

        # Get topological generations
        generations = list(nx.topological_generations(self.graph))
        return generations

    def get_parallel_groups(self) -> List[Set[str]]:
        """Identify groups of nodes that can execute in parallel"""
        generations = self.get_execution_order()
        parallel_groups = []

        for generation in generations:
            # Nodes in same generation can run in parallel
            parallel_groups.append(set(generation))

        return parallel_groups

    def validate(self) -> List[str]:
        """Validate DAG structure"""
        errors = []

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Workflow contains cycles")

        # Check for unreachable nodes
        if self.start_node_id:
            reachable = nx.descendants(self.graph, self.start_node_id)
            reachable.add(self.start_node_id)
            unreachable = set(self.nodes.keys()) - reachable
            if unreachable:
                errors.append(f"Unreachable nodes: {unreachable}")

        # Check for nodes without path to end
        if self.end_node_id:
            ancestors = nx.ancestors(self.graph, self.end_node_id)
            ancestors.add(self.end_node_id)
            no_end_path = set(self.nodes.keys()) - ancestors
            if no_end_path and no_end_path != {self.start_node_id}:
                errors.append(f"Nodes without path to end: {no_end_path}")

        # Validate node handlers
        for node in self.nodes.values():
            if node.type == NodeType.TASK and not node.handler:
                errors.append(f"Task node {node.id} missing handler")

        return errors

    def optimize(self):
        """Optimize DAG for execution"""
        # Remove redundant edges (transitive reduction)
        reduced = nx.transitive_reduction(self.graph)

        # Update edges
        self.graph = reduced

        # Recalculate dependencies
        for node_id in self.nodes:
            node = self.nodes[node_id]
            node.dependencies = set(self.graph.predecessors(node_id))
            node.dependents = set(self.graph.successors(node_id))

    def clone(self) -> 'WorkflowDAG':
        """Create a deep copy of the DAG"""
        new_dag = WorkflowDAG(name=f"{self.name}_clone")

        # Copy nodes
        for node_id, node in self.nodes.items():
            if node_id not in ["start", "end"]:
                new_node = WorkflowNode(
                    id=node.id,
                    name=node.name,
                    type=node.type,
                    handler=node.handler,
                    inputs=node.inputs.copy(),
                    condition=node.condition,
                    metadata=node.metadata
                )
                new_dag.add_node(new_node)

        # Copy edges
        for from_id, to_id, data in self.graph.edges(data=True):
            if from_id in new_dag.nodes and to_id in new_dag.nodes:
                new_dag.add_edge(from_id, to_id, data.get("condition"))

        return new_dag

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type.value,
                    "status": node.status.value,
                    "dependencies": list(node.dependencies),
                    "dependents": list(node.dependents),
                    "metadata": {
                        "priority": node.metadata.priority,
                        "timeout_seconds": node.metadata.timeout_seconds,
                        "max_retries": node.metadata.max_retries
                    }
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {"from": u, "to": v}
                for u, v in self.graph.edges()
            ],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowDAG':
        """Deserialize DAG from dictionary"""
        dag = cls(workflow_id=data["id"], name=data["name"])

        # Clear default nodes if custom ones provided
        if len(data["nodes"]) > 2:
            dag.nodes.clear()
            dag.graph.clear()

        # Add nodes
        for node_data in data["nodes"].values():
            if node_data["id"] not in ["start", "end"]:
                node = WorkflowNode(
                    id=node_data["id"],
                    name=node_data["name"],
                    type=NodeType(node_data["type"])
                )
                dag.add_node(node)

        # Add edges
        for edge in data["edges"]:
            if edge["from"] in dag.nodes and edge["to"] in dag.nodes:
                dag.add_edge(edge["from"], edge["to"])

        dag.metadata = data.get("metadata", {})
        return dag

    def calculate_checksum(self) -> str:
        """Calculate checksum for DAG structure"""
        dag_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(dag_str.encode()).hexdigest()