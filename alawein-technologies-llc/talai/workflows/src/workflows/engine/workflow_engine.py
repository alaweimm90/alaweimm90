"""
Workflow Engine - Main orchestration engine for workflow execution

Manages workflow lifecycle, versioning, and high-level orchestration.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import pickle
import logging
from pathlib import Path

from .workflow_dag import WorkflowDAG, WorkflowNode
from .workflow_executor import WorkflowExecutor, ExecutionContext, ExecutionResult


logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Workflow lifecycle states"""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    dag: WorkflowDAG
    version: str
    state: WorkflowState = WorkflowState.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class WorkflowEngine:
    """Main workflow orchestration engine"""

    def __init__(self, storage_path: Optional[str] = None,
                 max_concurrent_workflows: int = 10):
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / ".workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executors: Dict[str, WorkflowExecutor] = {}
        self.running_workflows: Dict[str, ExecutionContext] = {}
        self.max_concurrent = max_concurrent_workflows
        self.execution_history: List[ExecutionResult] = []
        self.handlers: Dict[str, Callable] = {}

        # Load persisted workflows
        self._load_workflows()

    def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """Register new workflow"""
        if workflow.id in self.workflows:
            raise ValueError(f"Workflow {workflow.id} already exists")

        # Validate DAG
        errors = workflow.dag.validate()
        if errors:
            raise ValueError(f"Invalid workflow DAG: {errors}")

        workflow.state = WorkflowState.READY
        workflow.updated_at = datetime.now()
        self.workflows[workflow.id] = workflow

        # Persist workflow
        self._save_workflow(workflow)

        logger.info(f"Registered workflow: {workflow.id}")
        return workflow.id

    def update_workflow(self, workflow_id: str,
                       updates: Dict[str, Any]) -> WorkflowDefinition:
        """Update existing workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        # Check if workflow is running
        if workflow_id in self.running_workflows:
            raise ValueError(f"Cannot update running workflow {workflow_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(workflow, key):
                setattr(workflow, key, value)

        workflow.updated_at = datetime.now()

        # Re-validate if DAG updated
        if "dag" in updates:
            errors = workflow.dag.validate()
            if errors:
                raise ValueError(f"Invalid workflow DAG: {errors}")

        # Persist changes
        self._save_workflow(workflow)

        return workflow

    def delete_workflow(self, workflow_id: str):
        """Delete workflow"""
        if workflow_id not in self.workflows:
            return

        # Check if running
        if workflow_id in self.running_workflows:
            raise ValueError(f"Cannot delete running workflow {workflow_id}")

        # Archive instead of delete
        workflow = self.workflows[workflow_id]
        workflow.state = WorkflowState.ARCHIVED
        self._save_workflow(workflow)

        del self.workflows[workflow_id]
        logger.info(f"Deleted workflow: {workflow_id}")

    async def execute_workflow(self, workflow_id: str,
                              inputs: Optional[Dict[str, Any]] = None,
                              context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """Execute workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        # Check concurrent execution limit
        if len(self.running_workflows) >= self.max_concurrent:
            raise RuntimeError(f"Maximum concurrent workflows ({self.max_concurrent}) reached")

        # Check workflow state
        if workflow.state not in [WorkflowState.READY, WorkflowState.COMPLETED]:
            raise ValueError(f"Workflow {workflow_id} not ready for execution")

        # Create executor
        executor = WorkflowExecutor()

        # Register handlers
        for handler_name, handler in self.handlers.items():
            executor.register_handler(handler_name, handler)

        # Create context if not provided
        if not context:
            context = ExecutionContext(
                workflow_id=workflow_id,
                execution_id=f"{workflow_id}_{datetime.now().timestamp()}",
                inputs=inputs or {}
            )

        # Mark workflow as running
        workflow.state = WorkflowState.RUNNING
        self.running_workflows[workflow_id] = context
        self.executors[workflow_id] = executor

        try:
            # Execute workflow
            logger.info(f"Starting workflow execution: {workflow_id}")
            result = await executor.execute(workflow.dag, context)

            # Update workflow state
            if result.status == "completed":
                workflow.state = WorkflowState.COMPLETED
            elif result.status == "failed":
                workflow.state = WorkflowState.FAILED
            else:
                workflow.state = WorkflowState.READY

            # Store result
            self.execution_history.append(result)

            # Persist execution result
            self._save_execution_result(result)

            logger.info(f"Workflow execution completed: {workflow_id}")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.state = WorkflowState.FAILED
            raise

        finally:
            # Clean up
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
            if workflow_id in self.executors:
                del self.executors[workflow_id]

    async def pause_workflow(self, workflow_id: str):
        """Pause running workflow"""
        if workflow_id not in self.running_workflows:
            raise ValueError(f"Workflow {workflow_id} not running")

        workflow = self.workflows[workflow_id]
        workflow.state = WorkflowState.PAUSED

        # TODO: Implement actual pause logic for executor
        logger.info(f"Paused workflow: {workflow_id}")

    async def resume_workflow(self, workflow_id: str):
        """Resume paused workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        if workflow.state != WorkflowState.PAUSED:
            raise ValueError(f"Workflow {workflow_id} not paused")

        workflow.state = WorkflowState.RUNNING

        # TODO: Implement actual resume logic
        logger.info(f"Resumed workflow: {workflow_id}")

    async def cancel_workflow(self, workflow_id: str):
        """Cancel running workflow"""
        if workflow_id not in self.running_workflows:
            raise ValueError(f"Workflow {workflow_id} not running")

        workflow = self.workflows[workflow_id]
        workflow.state = WorkflowState.CANCELLED

        # Clean up
        if workflow_id in self.running_workflows:
            del self.running_workflows[workflow_id]
        if workflow_id in self.executors:
            del self.executors[workflow_id]

        logger.info(f"Cancelled workflow: {workflow_id}")

    def register_handler(self, name: str, handler: Callable):
        """Register node handler"""
        self.handlers[name] = handler
        logger.info(f"Registered handler: {name}")

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition"""
        return self.workflows.get(workflow_id)

    def list_workflows(self, state: Optional[WorkflowState] = None) -> List[WorkflowDefinition]:
        """List workflows"""
        workflows = list(self.workflows.values())
        if state:
            workflows = [w for w in workflows if w.state == state]
        return workflows

    def get_execution_status(self, workflow_id: str) -> Optional[ExecutionContext]:
        """Get execution status for running workflow"""
        return self.running_workflows.get(workflow_id)

    def get_execution_history(self, workflow_id: Optional[str] = None,
                            limit: int = 100) -> List[ExecutionResult]:
        """Get execution history"""
        history = self.execution_history
        if workflow_id:
            history = [r for r in history if r.workflow_id == workflow_id]
        return history[-limit:]

    def create_workflow_from_template(self, template_name: str,
                                     params: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow from template"""
        from .workflow_templates import TemplateLibrary

        library = TemplateLibrary()
        template = library.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")

        # Build workflow from template
        dag = template.build(params)
        workflow = WorkflowDefinition(
            id=f"{template_name}_{datetime.now().timestamp()}",
            name=params.get("name", template.name),
            description=params.get("description", template.description),
            dag=dag,
            version="1.0.0",
            author=params.get("author", ""),
            tags=params.get("tags", [])
        )

        return workflow

    def _save_workflow(self, workflow: WorkflowDefinition):
        """Persist workflow to storage"""
        filepath = self.storage_path / f"workflow_{workflow.id}.json"
        data = {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "state": workflow.state.value,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "author": workflow.author,
            "tags": workflow.tags,
            "metadata": workflow.metadata,
            "dag": workflow.dag.to_dict()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_workflows(self):
        """Load persisted workflows"""
        if not self.storage_path.exists():
            return

        for filepath in self.storage_path.glob("workflow_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                dag = WorkflowDAG.from_dict(data["dag"])
                workflow = WorkflowDefinition(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    dag=dag,
                    version=data["version"],
                    state=WorkflowState(data["state"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    author=data["author"],
                    tags=data["tags"],
                    metadata=data.get("metadata", {})
                )
                self.workflows[workflow.id] = workflow
            except Exception as e:
                logger.error(f"Failed to load workflow from {filepath}: {e}")

    def _save_execution_result(self, result: ExecutionResult):
        """Persist execution result"""
        filepath = self.storage_path / f"execution_{result.execution_id}.json"
        data = {
            "execution_id": result.execution_id,
            "workflow_id": result.workflow_id,
            "status": result.status,
            "outputs": result.outputs,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
            "node_results": result.node_results,
            "metadata": result.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)