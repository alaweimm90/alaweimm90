"""
Workflow Templates - Pre-built workflow patterns and template library

Provides common workflow patterns that can be reused and customized.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from .workflow_dag import WorkflowDAG, WorkflowNode, NodeType


@dataclass
class WorkflowTemplate:
    """Template for creating workflows"""
    id: str
    name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]
    builder: Callable[[Dict[str, Any]], WorkflowDAG]
    tags: List[str] = None
    author: str = ""
    version: str = "1.0.0"

    def build(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build workflow DAG from template with parameters"""
        # Validate required parameters
        required = [p["name"] for p in self.parameters if p.get("required", False)]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Build DAG
        return self.builder(params)

    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for template"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class TemplateLibrary:
    """Library of workflow templates"""

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._register_builtin_templates()

    def _register_builtin_templates(self):
        """Register built-in workflow templates"""

        # Sequential Pipeline Template
        self.register_template(WorkflowTemplate(
            id="sequential_pipeline",
            name="Sequential Pipeline",
            description="Execute tasks in sequence",
            category="basic",
            parameters=[
                {"name": "tasks", "type": "list", "required": True,
                 "description": "List of task definitions"},
                {"name": "name", "type": "string", "required": False}
            ],
            builder=self._build_sequential_pipeline,
            tags=["basic", "pipeline"]
        ))

        # Parallel Pipeline Template
        self.register_template(WorkflowTemplate(
            id="parallel_pipeline",
            name="Parallel Pipeline",
            description="Execute tasks in parallel",
            category="basic",
            parameters=[
                {"name": "tasks", "type": "list", "required": True},
                {"name": "name", "type": "string", "required": False}
            ],
            builder=self._build_parallel_pipeline,
            tags=["basic", "parallel"]
        ))

        # Map-Reduce Template
        self.register_template(WorkflowTemplate(
            id="map_reduce",
            name="Map-Reduce",
            description="Map-reduce pattern for data processing",
            category="data",
            parameters=[
                {"name": "mapper", "type": "dict", "required": True},
                {"name": "reducer", "type": "dict", "required": True},
                {"name": "data_source", "type": "dict", "required": True}
            ],
            builder=self._build_map_reduce,
            tags=["data", "parallel", "batch"]
        ))

        # Conditional Workflow Template
        self.register_template(WorkflowTemplate(
            id="conditional_flow",
            name="Conditional Workflow",
            description="Workflow with conditional branching",
            category="control",
            parameters=[
                {"name": "condition", "type": "dict", "required": True},
                {"name": "true_branch", "type": "list", "required": True},
                {"name": "false_branch", "type": "list", "required": True}
            ],
            builder=self._build_conditional_flow,
            tags=["control", "branching"]
        ))

        # Retry Pattern Template
        self.register_template(WorkflowTemplate(
            id="retry_pattern",
            name="Retry Pattern",
            description="Task with retry logic",
            category="resilience",
            parameters=[
                {"name": "task", "type": "dict", "required": True},
                {"name": "max_retries", "type": "int", "default": 3},
                {"name": "retry_delay", "type": "int", "default": 5}
            ],
            builder=self._build_retry_pattern,
            tags=["resilience", "retry"]
        ))

        # Fan-out/Fan-in Template
        self.register_template(WorkflowTemplate(
            id="fan_out_fan_in",
            name="Fan-out/Fan-in",
            description="Distribute work and aggregate results",
            category="parallel",
            parameters=[
                {"name": "splitter", "type": "dict", "required": True},
                {"name": "workers", "type": "list", "required": True},
                {"name": "aggregator", "type": "dict", "required": True}
            ],
            builder=self._build_fan_out_fan_in,
            tags=["parallel", "distribution"]
        ))

        # Data Validation Pipeline
        self.register_template(WorkflowTemplate(
            id="data_validation",
            name="Data Validation Pipeline",
            description="Validate data through multiple checks",
            category="validation",
            parameters=[
                {"name": "data_source", "type": "dict", "required": True},
                {"name": "validators", "type": "list", "required": True},
                {"name": "error_handler", "type": "dict", "required": False}
            ],
            builder=self._build_data_validation,
            tags=["validation", "data", "quality"]
        ))

        # ETL Pipeline Template
        self.register_template(WorkflowTemplate(
            id="etl_pipeline",
            name="ETL Pipeline",
            description="Extract, Transform, Load data pipeline",
            category="data",
            parameters=[
                {"name": "extractors", "type": "list", "required": True},
                {"name": "transformers", "type": "list", "required": True},
                {"name": "loaders", "type": "list", "required": True}
            ],
            builder=self._build_etl_pipeline,
            tags=["data", "etl", "pipeline"]
        ))

    def register_template(self, template: WorkflowTemplate):
        """Register new template"""
        self.templates[template.id] = template

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)

    def list_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """List available templates"""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def _build_sequential_pipeline(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build sequential pipeline DAG"""
        dag = WorkflowDAG(name=params.get("name", "sequential_pipeline"))

        tasks = params["tasks"]
        prev_node_id = "start"

        for i, task in enumerate(tasks):
            node = WorkflowNode(
                id=f"task_{i}",
                name=task.get("name", f"Task {i}"),
                type=NodeType.TASK,
                inputs=task.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(prev_node_id, node.id)
            prev_node_id = node.id

        dag.add_edge(prev_node_id, "end")
        return dag

    def _build_parallel_pipeline(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build parallel pipeline DAG"""
        dag = WorkflowDAG(name=params.get("name", "parallel_pipeline"))

        tasks = params["tasks"]

        # Create fork node
        fork_node = WorkflowNode(
            id="fork",
            name="Fork",
            type=NodeType.FORK
        )
        dag.add_node(fork_node)
        dag.add_edge("start", fork_node.id)

        # Create parallel tasks
        task_ids = []
        for i, task in enumerate(tasks):
            node = WorkflowNode(
                id=f"task_{i}",
                name=task.get("name", f"Task {i}"),
                type=NodeType.TASK,
                inputs=task.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(fork_node.id, node.id)
            task_ids.append(node.id)

        # Create join node
        join_node = WorkflowNode(
            id="join",
            name="Join",
            type=NodeType.JOIN
        )
        dag.add_node(join_node)

        for task_id in task_ids:
            dag.add_edge(task_id, join_node.id)

        dag.add_edge(join_node.id, "end")
        return dag

    def _build_map_reduce(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build map-reduce DAG"""
        dag = WorkflowDAG(name="map_reduce")

        # Data source
        source = WorkflowNode(
            id="data_source",
            name="Data Source",
            type=NodeType.TASK,
            inputs=params["data_source"]
        )
        dag.add_node(source)
        dag.add_edge("start", source.id)

        # Mapper
        mapper = WorkflowNode(
            id="mapper",
            name="Mapper",
            type=NodeType.PARALLEL,
            inputs=params["mapper"]
        )
        dag.add_node(mapper)
        dag.add_edge(source.id, mapper.id)

        # Reducer
        reducer = WorkflowNode(
            id="reducer",
            name="Reducer",
            type=NodeType.TASK,
            inputs=params["reducer"]
        )
        dag.add_node(reducer)
        dag.add_edge(mapper.id, reducer.id)

        dag.add_edge(reducer.id, "end")
        return dag

    def _build_conditional_flow(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build conditional workflow DAG"""
        dag = WorkflowDAG(name="conditional_flow")

        # Condition node
        from .workflow_dag import NodeCondition
        condition_node = WorkflowNode(
            id="condition",
            name="Condition",
            type=NodeType.CONDITION,
            inputs=params["condition"]
        )
        dag.add_node(condition_node)
        dag.add_edge("start", condition_node.id)

        # True branch
        true_branch_start = WorkflowNode(
            id="true_branch_start",
            name="True Branch",
            type=NodeType.TASK
        )
        dag.add_node(true_branch_start)

        for i, task in enumerate(params["true_branch"]):
            node = WorkflowNode(
                id=f"true_task_{i}",
                name=task.get("name", f"True Task {i}"),
                type=NodeType.TASK,
                inputs=task.get("inputs", {})
            )
            dag.add_node(node)
            if i == 0:
                dag.add_edge(true_branch_start.id, node.id)
            else:
                dag.add_edge(f"true_task_{i-1}", node.id)

        # False branch
        false_branch_start = WorkflowNode(
            id="false_branch_start",
            name="False Branch",
            type=NodeType.TASK
        )
        dag.add_node(false_branch_start)

        for i, task in enumerate(params["false_branch"]):
            node = WorkflowNode(
                id=f"false_task_{i}",
                name=task.get("name", f"False Task {i}"),
                type=NodeType.TASK,
                inputs=task.get("inputs", {})
            )
            dag.add_node(node)
            if i == 0:
                dag.add_edge(false_branch_start.id, node.id)
            else:
                dag.add_edge(f"false_task_{i-1}", node.id)

        # Connect condition to branches
        condition = NodeCondition(
            expression="result == True",
            true_branch=true_branch_start.id,
            false_branch=false_branch_start.id
        )
        dag.add_edge(condition_node.id, true_branch_start.id, condition=condition)
        dag.add_edge(condition_node.id, false_branch_start.id)

        # Join branches
        join = WorkflowNode(
            id="join",
            name="Join",
            type=NodeType.JOIN
        )
        dag.add_node(join)

        if params["true_branch"]:
            dag.add_edge(f"true_task_{len(params['true_branch'])-1}", join.id)
        else:
            dag.add_edge(true_branch_start.id, join.id)

        if params["false_branch"]:
            dag.add_edge(f"false_task_{len(params['false_branch'])-1}", join.id)
        else:
            dag.add_edge(false_branch_start.id, join.id)

        dag.add_edge(join.id, "end")
        return dag

    def _build_retry_pattern(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build retry pattern DAG"""
        dag = WorkflowDAG(name="retry_pattern")

        task_data = params["task"]
        task = WorkflowNode(
            id="main_task",
            name=task_data.get("name", "Main Task"),
            type=NodeType.TASK,
            inputs=task_data.get("inputs", {})
        )
        task.metadata.max_retries = params.get("max_retries", 3)
        task.metadata.retry_delay_seconds = params.get("retry_delay", 5)

        dag.add_node(task)
        dag.add_edge("start", task.id)
        dag.add_edge(task.id, "end")

        return dag

    def _build_fan_out_fan_in(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build fan-out/fan-in DAG"""
        dag = WorkflowDAG(name="fan_out_fan_in")

        # Splitter
        splitter = WorkflowNode(
            id="splitter",
            name="Splitter",
            type=NodeType.TASK,
            inputs=params["splitter"]
        )
        dag.add_node(splitter)
        dag.add_edge("start", splitter.id)

        # Workers
        worker_ids = []
        for i, worker in enumerate(params["workers"]):
            node = WorkflowNode(
                id=f"worker_{i}",
                name=worker.get("name", f"Worker {i}"),
                type=NodeType.TASK,
                inputs=worker.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(splitter.id, node.id)
            worker_ids.append(node.id)

        # Aggregator
        aggregator = WorkflowNode(
            id="aggregator",
            name="Aggregator",
            type=NodeType.TASK,
            inputs=params["aggregator"]
        )
        dag.add_node(aggregator)

        for worker_id in worker_ids:
            dag.add_edge(worker_id, aggregator.id)

        dag.add_edge(aggregator.id, "end")
        return dag

    def _build_data_validation(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build data validation pipeline DAG"""
        dag = WorkflowDAG(name="data_validation")

        # Data source
        source = WorkflowNode(
            id="data_source",
            name="Data Source",
            type=NodeType.TASK,
            inputs=params["data_source"]
        )
        dag.add_node(source)
        dag.add_edge("start", source.id)

        # Validators
        prev_node_id = source.id
        for i, validator in enumerate(params["validators"]):
            node = WorkflowNode(
                id=f"validator_{i}",
                name=validator.get("name", f"Validator {i}"),
                type=NodeType.TASK,
                inputs=validator.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(prev_node_id, node.id)
            prev_node_id = node.id

        # Error handler (optional)
        if params.get("error_handler"):
            handler = WorkflowNode(
                id="error_handler",
                name="Error Handler",
                type=NodeType.TASK,
                inputs=params["error_handler"]
            )
            dag.add_node(handler)
            # Connect validators to error handler on failure
            for i in range(len(params["validators"])):
                dag.add_edge(f"validator_{i}", handler.id)

        dag.add_edge(prev_node_id, "end")
        return dag

    def _build_etl_pipeline(self, params: Dict[str, Any]) -> WorkflowDAG:
        """Build ETL pipeline DAG"""
        dag = WorkflowDAG(name="etl_pipeline")

        prev_node_id = "start"

        # Extractors
        for i, extractor in enumerate(params["extractors"]):
            node = WorkflowNode(
                id=f"extractor_{i}",
                name=extractor.get("name", f"Extractor {i}"),
                type=NodeType.TASK,
                inputs=extractor.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(prev_node_id, node.id)
            prev_node_id = node.id

        # Transformers
        for i, transformer in enumerate(params["transformers"]):
            node = WorkflowNode(
                id=f"transformer_{i}",
                name=transformer.get("name", f"Transformer {i}"),
                type=NodeType.TASK,
                inputs=transformer.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(prev_node_id, node.id)
            prev_node_id = node.id

        # Loaders
        for i, loader in enumerate(params["loaders"]):
            node = WorkflowNode(
                id=f"loader_{i}",
                name=loader.get("name", f"Loader {i}"),
                type=NodeType.TASK,
                inputs=loader.get("inputs", {})
            )
            dag.add_node(node)
            dag.add_edge(prev_node_id, node.id)
            prev_node_id = node.id

        dag.add_edge(prev_node_id, "end")
        return dag