"""
Workflow Executor - Executes workflow DAGs with retry logic and parallel execution

Handles the execution of workflow nodes with support for:
- Parallel execution of independent nodes
- Conditional branching
- Retry logic with exponential backoff
- Context management and data passing
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .workflow_dag import WorkflowDAG, WorkflowNode, NodeStatus, NodeType, NodeCondition


logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for workflow execution"""
    workflow_id: str
    execution_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"
    error: Optional[str] = None
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    skipped_nodes: Set[str] = field(default_factory=set)

    def get_node_output(self, node_id: str) -> Any:
        """Get output from a specific node"""
        return self.node_outputs.get(node_id)

    def set_node_output(self, node_id: str, output: Any):
        """Set output for a specific node"""
        self.node_outputs[node_id] = output
        # Update outputs with latest node output
        if isinstance(output, dict):
            self.outputs.update(output)

    def update_variables(self, variables: Dict[str, Any]):
        """Update context variables"""
        self.variables.update(variables)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "node_outputs": self.node_outputs,
            "variables": self.variables,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "error": self.error,
            "completed_nodes": list(self.completed_nodes),
            "failed_nodes": list(self.failed_nodes),
            "skipped_nodes": list(self.skipped_nodes)
        }


@dataclass
class ExecutionResult:
    """Result of workflow execution"""
    execution_id: str
    workflow_id: str
    status: str  # completed, failed, cancelled
    outputs: Dict[str, Any]
    errors: List[str]
    duration_seconds: float
    node_results: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetryStrategy:
    """Retry strategy for failed nodes"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 exponential_base: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay

    def get_delay(self, retry_count: int) -> float:
        """Calculate retry delay with exponential backoff"""
        delay = self.base_delay * (self.exponential_base ** retry_count)
        return min(delay, self.max_delay)

    def should_retry(self, retry_count: int, error: Exception) -> bool:
        """Determine if should retry based on error and count"""
        if retry_count >= self.max_retries:
            return False

        # Don't retry certain errors
        non_retryable_errors = (ValueError, TypeError, SyntaxError)
        if isinstance(error, non_retryable_errors):
            return False

        return True


class WorkflowExecutor:
    """Executes workflow DAGs"""

    def __init__(self, max_parallel_nodes: int = 10,
                 retry_strategy: Optional[RetryStrategy] = None):
        self.max_parallel_nodes = max_parallel_nodes
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_nodes)
        self.node_handlers: Dict[str, Callable] = {}

    def register_handler(self, node_type: str, handler: Callable):
        """Register handler for node type"""
        self.node_handlers[node_type] = handler

    async def execute(self, dag: WorkflowDAG,
                     context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """Execute workflow DAG"""
        # Validate DAG
        errors = dag.validate()
        if errors:
            return ExecutionResult(
                execution_id="",
                workflow_id=dag.id,
                status="failed",
                outputs={},
                errors=errors,
                duration_seconds=0,
                node_results={}
            )

        # Initialize context
        if not context:
            context = ExecutionContext(
                workflow_id=dag.id,
                execution_id=f"{dag.id}_{datetime.now().isoformat()}"
            )

        # Start execution
        context.start_time = datetime.now()
        logger.info(f"Starting workflow execution: {context.execution_id}")

        try:
            # Execute nodes
            await self._execute_dag(dag, context)

            # Finalize execution
            context.end_time = datetime.now()
            duration = (context.end_time - context.start_time).total_seconds()

            # Determine final status
            if context.failed_nodes:
                context.status = "failed"
            elif context.completed_nodes:
                context.status = "completed"
            else:
                context.status = "empty"

            # Build result
            result = ExecutionResult(
                execution_id=context.execution_id,
                workflow_id=dag.id,
                status=context.status,
                outputs=context.outputs,
                errors=[context.error] if context.error else [],
                duration_seconds=duration,
                node_results=self._build_node_results(dag, context)
            )

            logger.info(f"Workflow execution completed: {context.execution_id}")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            context.end_time = datetime.now()
            duration = (context.end_time - context.start_time).total_seconds()

            return ExecutionResult(
                execution_id=context.execution_id,
                workflow_id=dag.id,
                status="failed",
                outputs=context.outputs,
                errors=[str(e)],
                duration_seconds=duration,
                node_results=self._build_node_results(dag, context)
            )

    async def _execute_dag(self, dag: WorkflowDAG, context: ExecutionContext):
        """Execute DAG nodes"""
        # Mark start node as completed
        if dag.start_node_id:
            context.completed_nodes.add(dag.start_node_id)
            dag.nodes[dag.start_node_id].status = NodeStatus.COMPLETED

        # Execute nodes level by level
        max_iterations = len(dag.nodes) * 2  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get ready nodes
            ready_nodes = dag.get_ready_nodes(context.completed_nodes)
            ready_nodes = [n for n in ready_nodes
                          if n.id not in context.completed_nodes
                          and n.id not in context.failed_nodes
                          and n.id not in context.skipped_nodes]

            if not ready_nodes:
                # Check if all nodes are processed
                all_processed = (
                    len(context.completed_nodes) +
                    len(context.failed_nodes) +
                    len(context.skipped_nodes)
                ) >= len(dag.nodes)

                if all_processed or self._is_workflow_complete(dag, context):
                    break

                # Wait a bit for running nodes
                await asyncio.sleep(0.1)
                continue

            # Execute ready nodes in parallel
            await self._execute_parallel_nodes(ready_nodes, context)

        # Mark end node as completed
        if dag.end_node_id and not context.failed_nodes:
            context.completed_nodes.add(dag.end_node_id)
            dag.nodes[dag.end_node_id].status = NodeStatus.COMPLETED

    async def _execute_parallel_nodes(self, nodes: List[WorkflowNode],
                                     context: ExecutionContext):
        """Execute multiple nodes in parallel"""
        if not nodes:
            return

        # Limit parallelism
        batch_size = min(len(nodes), self.max_parallel_nodes)

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            tasks = [self._execute_node(node, context) for node in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_node(self, node: WorkflowNode,
                           context: ExecutionContext) -> Optional[Any]:
        """Execute single node with retry logic"""
        logger.info(f"Executing node: {node.id} ({node.name})")
        node.start_time = datetime.now()
        node.status = NodeStatus.RUNNING

        try:
            # Check condition if present
            if node.condition:
                if not node.condition.check(context.variables):
                    logger.info(f"Skipping node {node.id} due to condition")
                    node.status = NodeStatus.SKIPPED
                    context.skipped_nodes.add(node.id)
                    return None

            # Execute based on node type
            result = await self._execute_node_handler(node, context)

            # Update context
            node.status = NodeStatus.COMPLETED
            node.end_time = datetime.now()
            context.completed_nodes.add(node.id)

            if result is not None:
                context.set_node_output(node.id, result)
                node.outputs = result if isinstance(result, dict) else {"result": result}

            logger.info(f"Node completed: {node.id}")
            return result

        except Exception as e:
            logger.error(f"Node {node.id} failed: {e}")
            node.error = str(e)

            # Check retry
            if self._should_retry_node(node, e):
                await self._retry_node(node, context, e)
            else:
                node.status = NodeStatus.FAILED
                node.end_time = datetime.now()
                context.failed_nodes.add(node.id)
                context.error = f"Node {node.id} failed: {e}"

            return None

    async def _execute_node_handler(self, node: WorkflowNode,
                                   context: ExecutionContext) -> Any:
        """Execute node handler based on type"""
        if node.type in [NodeType.START, NodeType.END]:
            return None

        if node.type == NodeType.WAIT:
            wait_time = node.inputs.get("seconds", 1)
            await asyncio.sleep(wait_time)
            return {"waited": wait_time}

        if node.type == NodeType.CONDITION:
            # Evaluate condition and return branch
            if node.condition:
                result = node.condition.check(context.variables)
                return {"branch": "true" if result else "false", "result": result}
            return {"branch": "true", "result": True}

        if node.type == NodeType.TASK:
            # Execute handler if available
            if node.handler:
                if asyncio.iscoroutinefunction(node.handler):
                    return await node.handler(node.inputs, context)
                else:
                    # Run sync handler in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.executor, node.handler, node.inputs, context
                    )

            # Check registered handlers
            handler_key = node.inputs.get("handler_type", node.name)
            if handler_key in self.node_handlers:
                handler = self.node_handlers[handler_key]
                if asyncio.iscoroutinefunction(handler):
                    return await handler(node.inputs, context)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.executor, handler, node.inputs, context
                    )

        # Default: return inputs as outputs
        return node.inputs

    def _should_retry_node(self, node: WorkflowNode, error: Exception) -> bool:
        """Check if node should be retried"""
        max_retries = node.metadata.max_retries
        if node.retry_count >= max_retries:
            return False

        return self.retry_strategy.should_retry(node.retry_count, error)

    async def _retry_node(self, node: WorkflowNode,
                         context: ExecutionContext, error: Exception):
        """Retry failed node with exponential backoff"""
        node.retry_count += 1
        delay = self.retry_strategy.get_delay(node.retry_count)

        logger.info(f"Retrying node {node.id} (attempt {node.retry_count})"
                   f" after {delay} seconds")

        node.status = NodeStatus.RETRY
        await asyncio.sleep(delay)

        # Retry execution
        await self._execute_node(node, context)

    def _is_workflow_complete(self, dag: WorkflowDAG,
                             context: ExecutionContext) -> bool:
        """Check if workflow execution is complete"""
        # Check if end node is reachable
        if dag.end_node_id:
            end_node = dag.nodes[dag.end_node_id]
            if end_node.is_ready(context.completed_nodes):
                return True

        # Check if any critical node failed
        for node_id in context.failed_nodes:
            node = dag.nodes.get(node_id)
            if node and node.metadata.priority > 5:  # Critical node
                return True

        # Check if no more nodes can execute
        for node in dag.nodes.values():
            if (node.id not in context.completed_nodes
                and node.id not in context.failed_nodes
                and node.id not in context.skipped_nodes):
                # Check if node can potentially execute
                impossible_deps = context.failed_nodes.intersection(node.dependencies)
                if not impossible_deps:
                    return False  # This node might still execute

        return True

    def _build_node_results(self, dag: WorkflowDAG,
                           context: ExecutionContext) -> Dict[str, Dict[str, Any]]:
        """Build detailed node results"""
        results = {}
        for node_id, node in dag.nodes.items():
            results[node_id] = {
                "name": node.name,
                "type": node.type.value,
                "status": node.status.value,
                "outputs": node.outputs,
                "error": node.error,
                "retry_count": node.retry_count,
                "start_time": node.start_time.isoformat() if node.start_time else None,
                "end_time": node.end_time.isoformat() if node.end_time else None,
                "duration": (
                    (node.end_time - node.start_time).total_seconds()
                    if node.start_time and node.end_time else None
                )
            }
        return results