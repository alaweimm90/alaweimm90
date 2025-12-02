#!/usr/bin/env python3
"""
Task Executor - Execute workflows and route tasks to agents.

This module provides:
- Workflow execution engine
- Task routing
- Agent invocation
- Checkpoint management
- Telemetry collection
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

AUTOMATION_PATH = Path(__file__).parent


class TaskStatus(Enum):
    """Status of a task or workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Context passed through workflow execution."""
    workflow_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, TaskResult] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def get_stage_output(self, stage_name: str) -> Any:
        """Get output from a completed stage."""
        if stage_name in self.stage_results:
            return self.stage_results[stage_name].output
        return None

    def set_output(self, key: str, value: Any):
        """Set a workflow output."""
        self.outputs[key] = value

    def checkpoint(self, stage_name: str):
        """Create a checkpoint at current state."""
        self.checkpoints.append({
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "outputs": dict(self.outputs),
            "stage_results": {k: v.status.value for k, v in self.stage_results.items()}
        })


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class WorkflowExecutor:
    """Execute workflows defined in YAML."""

    def __init__(self, agent_handler: Optional[Callable] = None):
        """
        Initialize executor.

        Args:
            agent_handler: Callable that executes agent tasks.
                          Signature: (agent_name, action, inputs) -> output
        """
        self.agent_handler = agent_handler or self._default_agent_handler
        self.workflows = self._load_workflows()
        self.agents = self._load_agents()
        self.telemetry: List[Dict[str, Any]] = []

    def _load_workflows(self) -> Dict[str, Any]:
        """Load workflow definitions."""
        workflows_file = AUTOMATION_PATH / "workflows" / "config" / "workflows.yaml"
        if workflows_file.exists():
            config = load_yaml_file(workflows_file)
            return config.get("workflows", {})
        return {}

    def _load_agents(self) -> Dict[str, Any]:
        """Load agent definitions."""
        agents_file = AUTOMATION_PATH / "agents" / "config" / "agents.yaml"
        if agents_file.exists():
            config = load_yaml_file(agents_file)
            return config.get("agents", {})
        return {}

    def _default_agent_handler(self, agent_name: str, action: str, inputs: Dict[str, Any]) -> Any:
        """Default agent handler - returns mock response."""
        return {
            "agent": agent_name,
            "action": action,
            "status": "simulated",
            "message": f"Agent '{agent_name}' would execute: {action[:100]}..."
        }

    def _record_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Record telemetry event."""
        self.telemetry.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        })

    def execute_workflow(
        self,
        workflow_name: str,
        inputs: Dict[str, Any],
        dry_run: bool = False
    ) -> WorkflowContext:
        """
        Execute a workflow.

        Args:
            workflow_name: Name of workflow to execute
            inputs: Input parameters for the workflow
            dry_run: If True, simulate execution without running agents

        Returns:
            WorkflowContext with results
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.workflows[workflow_name]
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        context = WorkflowContext(
            workflow_id=workflow_id,
            inputs=inputs
        )

        self._record_telemetry("workflow_start", {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "dry_run": dry_run
        })

        stages = workflow.get("stages", [])
        completed_stages = set()

        # Execute stages in order, respecting dependencies
        max_iterations = len(stages) * 2  # Prevent infinite loops
        iteration = 0

        while len(completed_stages) < len(stages) and iteration < max_iterations:
            iteration += 1
            progress_made = False

            for stage in stages:
                stage_name = stage.get("name", f"stage_{iteration}")

                if stage_name in completed_stages:
                    continue

                # Check dependencies
                depends_on = stage.get("depends_on", [])
                if not all(dep in completed_stages for dep in depends_on):
                    continue

                # Check condition
                condition = stage.get("condition")
                if condition and not self._evaluate_condition(condition, context):
                    completed_stages.add(stage_name)
                    continue

                # Execute stage
                result = self._execute_stage(stage, context, dry_run)
                context.stage_results[stage_name] = result

                if result.status == TaskStatus.COMPLETED:
                    completed_stages.add(stage_name)
                    context.checkpoint(stage_name)
                    progress_made = True

                    # Store outputs
                    for output_name in stage.get("outputs", []):
                        context.set_output(output_name, result.output)

                elif result.status == TaskStatus.FAILED:
                    # Check error handling
                    if not self._handle_stage_error(stage, result, context):
                        break

            if not progress_made:
                break

        # Record completion
        self._record_telemetry("workflow_complete", {
            "workflow_id": workflow_id,
            "stages_completed": len(completed_stages),
            "total_stages": len(stages),
            "duration_ms": int((datetime.now() - context.start_time).total_seconds() * 1000)
        })

        return context

    def _execute_stage(
        self,
        stage: Dict[str, Any],
        context: WorkflowContext,
        dry_run: bool
    ) -> TaskResult:
        """Execute a single workflow stage."""
        stage_name = stage.get("name", "unnamed")
        agent_name = stage.get("agent", "default_agent")
        action = stage.get("action", "")

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self._record_telemetry("stage_start", {
            "task_id": task_id,
            "stage_name": stage_name,
            "agent": agent_name
        })

        # Gather inputs
        stage_inputs = {}
        for input_name in stage.get("inputs", []):
            if input_name in context.inputs:
                stage_inputs[input_name] = context.inputs[input_name]
            elif input_name in context.outputs:
                stage_inputs[input_name] = context.outputs[input_name]

        try:
            if dry_run:
                output = {
                    "dry_run": True,
                    "would_execute": action[:200],
                    "agent": agent_name,
                    "inputs": list(stage_inputs.keys())
                }
            else:
                output = self.agent_handler(agent_name, action, stage_inputs)

            duration_ms = int((time.time() - start_time) * 1000)

            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )

        self._record_telemetry("stage_complete", {
            "task_id": task_id,
            "stage_name": stage_name,
            "status": result.status.value,
            "duration_ms": result.duration_ms
        })

        return result

    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """Evaluate a stage condition."""
        # Simple condition evaluation
        # In production, use a proper expression evaluator
        if condition == "has_critical_issues":
            return context.outputs.get("has_critical_issues", False)
        if condition.startswith("improvement_score"):
            score = context.outputs.get("improvement_score", 1.0)
            threshold = float(condition.split("<")[1].strip()) if "<" in condition else 0.8
            return score < threshold
        return True

    def _handle_stage_error(
        self,
        stage: Dict[str, Any],
        result: TaskResult,
        context: WorkflowContext
    ) -> bool:
        """Handle stage error. Returns True if workflow should continue."""
        # Default: stop on error
        return False

    def get_workflow_info(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a workflow."""
        return self.workflows.get(workflow_name)

    def list_workflows(self) -> List[str]:
        """List available workflows."""
        return list(self.workflows.keys())


class TaskRouter:
    """Route tasks to appropriate handlers based on intent."""

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load orchestration config."""
        config_file = AUTOMATION_PATH / "orchestration" / "config" / "orchestration.yaml"
        if config_file.exists():
            return load_yaml_file(config_file)
        return {}

    def route(self, task_description: str) -> Dict[str, Any]:
        """
        Route a task to appropriate handler.

        Args:
            task_description: Natural language task description

        Returns:
            Routing decision with recommended tools/agents
        """
        task_lower = task_description.lower()

        keywords = self.config.get("tool_routing", {}).get("intent_extraction", {}).get("keywords", {})
        rules = self.config.get("tool_routing", {}).get("rules", {})

        # Score categories
        scores = {}
        for category, kws in keywords.items():
            score = sum(1 for kw in kws if kw in task_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return {
                "success": False,
                "message": "Could not classify task",
                "suggestion": "Please provide more details about what you want to do"
            }

        # Best match
        best_category = max(scores, key=scores.get)
        confidence = min(scores[best_category] / max(len(task_lower.split()), 1), 1.0)

        tools = rules.get(best_category, {}).get("tools", [])
        threshold = rules.get(best_category, {}).get("confidence_threshold", 0.6)

        return {
            "success": True,
            "task": task_description,
            "category": best_category,
            "confidence": confidence,
            "meets_threshold": confidence >= threshold,
            "recommended_tools": tools,
            "primary_tool": tools[0] if tools else None,
            "all_scores": scores
        }

    def suggest_workflow(self, task_description: str) -> Optional[str]:
        """Suggest a workflow for a task."""
        routing = self.route(task_description)

        if not routing["success"]:
            return None

        category = routing["category"]

        # Map categories to workflows
        workflow_map = {
            "debugging": "bug_fix",
            "refactoring": "refactoring",
            "implementation": "feature_implementation",
            "testing": "code_review",
            "research": "research_synthesis"
        }

        return workflow_map.get(category)


# Convenience functions
def execute_workflow(workflow_name: str, inputs: Dict[str, Any], dry_run: bool = False) -> WorkflowContext:
    """Execute a workflow."""
    executor = WorkflowExecutor()
    return executor.execute_workflow(workflow_name, inputs, dry_run)


def route_task(task_description: str) -> Dict[str, Any]:
    """Route a task."""
    router = TaskRouter()
    return router.route(task_description)


if __name__ == "__main__":
    # Demo execution
    print("=" * 60)
    print("WORKFLOW EXECUTOR DEMO")
    print("=" * 60)

    executor = WorkflowExecutor()

    print("\nAvailable workflows:")
    for wf in executor.list_workflows():
        print(f"  - {wf}")

    print("\n" + "-" * 40)
    print("Executing 'code_review' workflow (dry run)...")
    print("-" * 40)

    context = executor.execute_workflow(
        "code_review",
        inputs={"code_path": "src/main.py"},
        dry_run=True
    )

    print(f"\nWorkflow ID: {context.workflow_id}")
    print(f"Stages completed: {len(context.stage_results)}")

    for stage_name, result in context.stage_results.items():
        print(f"  {stage_name}: {result.status.value}")
