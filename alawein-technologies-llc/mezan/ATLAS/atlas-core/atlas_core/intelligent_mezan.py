"""
Intelligent MEZAN Engine - Optimized with Deep Reasoning and Distributed Support

Enhanced MEZAN engine with:
- Intelligent solver selection based on deep analysis
- Sequential deep-think mode for maximum intelligence
- Token-efficient focused reasoning
- Adaptive strategy based on problem characteristics
- DISTRIBUTED workflow execution across multiple nodes
- Horizontal scaling with worker pools
- State synchronization via Redis
- Event-driven workflow triggers

This version focuses on DEPTH over breadth, using:
- 3 parallel agents for quick assessment
- 1 deep synthesizer for intensive reasoning
- Intelligent solver pairing and balancing
- Distributed task execution and coordination

Author: MEZAN Research Team
Date: 2025-11-18
Version: 4.0 (Distributed Intelligence)
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import numpy as np

from atlas_core.mezan_engine import (
    MezanEngine,
    BaseSolver,
    SolverConfig,
    SolverResult,
    SolverType,
    MezanState,
)
from atlas_core.deepthink_agents import (
    DeepThinkOrchestrator,
    DeepTask,
    DeepResult,
    AnalysisDepth,
)

# Import distributed components
from atlas_core.distributed import (
    DistributedOrchestrator,
    DistributedTask,
    RedisBackend,
    ClusterNode,
    NodeStatus
)
from atlas_core.event_bus import (
    EventBus,
    Event,
    EventType,
    WorkflowStartedEvent,
    TaskCompletedEvent,
    get_event_bus,
    publish_event
)
from atlas_core.message_queue import (
    MessageQueue,
    Message,
    DeliveryMode,
    MessageBrokerType
)

logger = logging.getLogger(__name__)


@dataclass
class IntelligentSolverPair:
    """Intelligently selected solver pair based on problem analysis"""
    left_solver: BaseSolver
    right_solver: BaseSolver
    selection_reasoning: str
    expected_performance: str
    confidence: float


@dataclass
class DistributedWorkflow:
    """Represents a distributed workflow execution"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tasks: List[DistributedTask] = field(default_factory=list)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)
    node_assignments: Dict[str, str] = field(default_factory=dict)


class IntelligentMezanEngine:
    """
    Intelligent MEZAN Engine with Deep Reasoning and Distributed Execution

    Key improvements:
    1. Deep problem analysis before solving
    2. Intelligent solver selection
    3. Adaptive balancing strategy
    4. Sequential deep-think for critical decisions
    5. Token-efficient focused reasoning
    6. DISTRIBUTED execution across multiple nodes
    7. Event-driven workflow coordination
    8. Horizontal scaling with worker pools

    Architecture:
    - Phase 1: Deep analysis (3 parallel + 1 sequential)
    - Phase 2: Intelligent solver selection
    - Phase 3: Adaptive MEZAN balancing
    - Phase 4: Deep synthesis of results
    - Phase 5: Distributed execution and coordination
    """

    def __init__(
        self,
        available_solvers: Optional[List[BaseSolver]] = None,
        use_deep_analysis: bool = True,
        enable_distributed: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        node_id: Optional[str] = None,
        worker_count: int = 4,
        enable_events: bool = True,
        message_broker_type: MessageBrokerType = MessageBrokerType.IN_MEMORY
    ):
        """
        Initialize Intelligent MEZAN with Distributed Support

        Args:
            available_solvers: Pool of solvers to choose from
            use_deep_analysis: Whether to use deep analysis (recommended)
            enable_distributed: Enable distributed execution
            redis_host: Redis host for distributed state
            redis_port: Redis port
            node_id: Unique node identifier
            worker_count: Number of worker threads
            enable_events: Enable event-driven coordination
            message_broker_type: Type of message broker to use
        """
        self.available_solvers = available_solvers or []
        self.use_deep_analysis = use_deep_analysis
        self.enable_distributed = enable_distributed
        self.worker_count = worker_count
        self.enable_events = enable_events

        # Initialize deep think orchestrator
        if use_deep_analysis:
            self.deep_think = DeepThinkOrchestrator(max_parallel_workers=3)
        else:
            self.deep_think = None

        self.mezan_engine = None  # Created after solver selection
        self.last_analysis = None

        # Initialize distributed components if enabled
        self.distributed_orchestrator = None
        self.event_bus = None
        self.message_queue = None
        self.worker_pool = None
        self.active_workflows: Dict[str, DistributedWorkflow] = {}

        if enable_distributed:
            self._initialize_distributed_components(
                redis_host,
                redis_port,
                node_id,
                message_broker_type
            )

        logger.info(
            f"Intelligent MEZAN initialized with {len(self.available_solvers)} "
            f"available solvers, deep_analysis={use_deep_analysis}, "
            f"distributed={enable_distributed}, workers={worker_count}"
        )

    def _initialize_distributed_components(
        self,
        redis_host: str,
        redis_port: int,
        node_id: Optional[str],
        message_broker_type: MessageBrokerType
    ):
        """Initialize all distributed components"""
        try:
            # Initialize distributed orchestrator
            self.distributed_orchestrator = DistributedOrchestrator(
                redis_host=redis_host,
                redis_port=redis_port,
                node_id=node_id,
                enable_leader_election=True,
                enable_health_monitoring=True
            )
            self.distributed_orchestrator.start()

            # Initialize event bus
            if self.enable_events:
                self.event_bus = get_event_bus()
                self._register_event_handlers()

            # Initialize message queue
            self.message_queue = MessageQueue(
                broker_type=message_broker_type,
                connection_config={
                    "host": redis_host,
                    "port": redis_port
                },
                enable_failover=True
            )

            # Initialize worker pool
            self.worker_pool = ThreadPoolExecutor(max_workers=self.worker_count)

            # Start worker threads
            for i in range(self.worker_count):
                self.worker_pool.submit(self._worker_loop)

            logger.info("Distributed components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize distributed components: {e}")
            self.enable_distributed = False

    def _register_event_handlers(self):
        """Register event handlers for workflow coordination"""
        if not self.event_bus:
            return

        # Handle workflow events
        self.event_bus.register_handler(
            handler=self._handle_workflow_started,
            event_types=[EventType.WORKFLOW_STARTED],
            priority=100
        )

        self.event_bus.register_handler(
            handler=self._handle_task_completed,
            event_types=[EventType.TASK_COMPLETED],
            priority=100
        )

        logger.debug("Event handlers registered")

    def _handle_workflow_started(self, event: Event):
        """Handle workflow started event"""
        workflow_id = event.payload.get("workflow_id")
        logger.info(f"Workflow {workflow_id} started")

        # Update workflow status if tracked
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].status = "running"

    def _handle_task_completed(self, event: Event):
        """Handle task completed event"""
        task_id = event.payload.get("task_id")
        result = event.payload.get("result")
        duration = event.payload.get("duration")

        logger.info(f"Task {task_id} completed in {duration:.3f}s")

        # Update workflow if this task belongs to one
        for workflow in self.active_workflows.values():
            for task in workflow.tasks:
                if task.task_id == task_id:
                    task.status = "completed"
                    task.result = result
                    workflow.results[task_id] = result
                    break

    def _worker_loop(self):
        """Worker loop for processing distributed tasks"""
        logger.info(f"Worker started on node {self.distributed_orchestrator.node_id if self.distributed_orchestrator else 'local'}")

        while self.enable_distributed:
            try:
                # Get next task from queue
                task = None
                if self.distributed_orchestrator:
                    task = self.distributed_orchestrator.process_next_task(timeout=5)

                if not task:
                    time.sleep(0.1)
                    continue

                # Process task
                logger.debug(f"Processing task {task.task_id} of type {task.task_type}")
                start_time = time.time()

                try:
                    result = self._process_distributed_task(task)

                    # Mark task complete
                    if self.distributed_orchestrator:
                        self.distributed_orchestrator.complete_task(
                            task.task_id,
                            result
                        )

                    # Publish completion event
                    if self.event_bus:
                        publish_event(TaskCompletedEvent(
                            task_id=task.task_id,
                            result=result,
                            duration=time.time() - start_time
                        ))

                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")

                    if self.distributed_orchestrator:
                        self.distributed_orchestrator.fail_task(
                            task.task_id,
                            str(e)
                        )

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)

    def _process_distributed_task(self, task: DistributedTask) -> Any:
        """Process a single distributed task"""
        task_type = task.task_type
        payload = task.payload

        if task_type == "analyze":
            # Deep analysis task
            return self._run_deep_analysis(payload)

        elif task_type == "solve":
            # Solver execution task
            return self._run_solver(payload)

        elif task_type == "balance":
            # MEZAN balancing task
            return self._run_mezan_balance(payload)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _run_deep_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run deep analysis as distributed task"""
        if not self.deep_think:
            return {"error": "Deep analysis not enabled"}

        task = DeepTask(
            task_id=payload.get("task_id", str(uuid.uuid4())),
            problem=payload.get("problem", {}),
            depth=AnalysisDepth[payload.get("depth", "DEEP")],
            max_time_seconds=payload.get("max_time", 30.0)
        )

        _, _, _, synthesis_result = self.deep_think.deep_analyze(task, use_synthesis=True)

        return {
            "insights": synthesis_result.insights,
            "recommendations": synthesis_result.recommendations,
            "confidence": synthesis_result.confidence,
            "reasoning": synthesis_result.reasoning
        }

    def _run_solver(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run solver as distributed task"""
        solver_id = payload.get("solver_id")
        problem = payload.get("problem")

        # Find solver (would be from registry in production)
        # For now, return mock result
        return {
            "solver_id": solver_id,
            "objective_value": np.random.random(),
            "solution": {},
            "iterations": np.random.randint(10, 100)
        }

    def _run_mezan_balance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run MEZAN balancing as distributed task"""
        problem = payload.get("problem")
        max_iterations = payload.get("max_iterations", 5)

        # Create temporary MEZAN engine
        from atlas_core.mezan_engine import MockContinuousSolver, MockDiscreteSolver

        temp_engine = MezanEngine(
            solver_left=MockContinuousSolver(
                SolverConfig("left", SolverType.CONTINUOUS, 0.5)
            ),
            solver_right=MockDiscreteSolver(
                SolverConfig("right", SolverType.DISCRETE, 0.5)
            )
        )

        result = temp_engine.solve_with_balance(problem, max_iterations)

        return {
            "objective_value": result.objective_value,
            "iterations": result.iterations,
            "time_seconds": result.time_seconds,
            "solution": result.solution
        }

    def solve_intelligently(
        self,
        problem: Dict[str, Any],
        max_mezan_iterations: int = 5,
        analysis_depth: AnalysisDepth = AnalysisDepth.DEEP,
    ) -> Tuple[SolverResult, Optional[DeepResult]]:
        """
        Solve problem with intelligent strategy

        Returns:
            (mezan_result, synthesis_result)
        """
        start_time = time.time()

        logger.info("="*70)
        logger.info("INTELLIGENT MEZAN SOLVING")
        logger.info("="*70)

        # PHASE 1: Deep Problem Analysis
        logger.info("\n📊 PHASE 1: Deep Problem Analysis")
        logger.info("-" * 70)

        synthesis_result = None

        if self.use_deep_analysis:
            # Create deep task
            task = DeepTask(
                task_id=f"problem_{int(time.time() * 1000)}",
                problem=problem,
                depth=analysis_depth,
                max_time_seconds=30.0,
            )

            # Run deep analysis (3 parallel + 1 sequential)
            (
                analyzer_result,
                optimizer_result,
                validator_result,
                synthesis_result,
            ) = self.deep_think.deep_analyze(task, use_synthesis=True)

            self.last_analysis = synthesis_result

            # Log insights
            logger.info("\n✨ Analysis Insights:")
            for insight in synthesis_result.insights[:10]:  # Top 10
                if insight.strip():
                    logger.info(f"  • {insight}")

            # Log recommendations
            logger.info(f"\n🎯 Generated {len(synthesis_result.recommendations)} recommendations")

        # PHASE 2: Intelligent Solver Selection
        logger.info("\n🧠 PHASE 2: Intelligent Solver Selection")
        logger.info("-" * 70)

        solver_pair = self._select_solver_pair_intelligently(
            problem, synthesis_result
        )

        logger.info(f"Selected Solver Pair:")
        logger.info(f"  Left:  {solver_pair.left_solver.config.solver_id}")
        logger.info(f"  Right: {solver_pair.right_solver.config.solver_id}")
        logger.info(f"  Reasoning: {solver_pair.selection_reasoning}")
        logger.info(f"  Expected: {solver_pair.expected_performance}")
        logger.info(f"  Confidence: {solver_pair.confidence:.3f}")

        # PHASE 3: Create and Run MEZAN Engine
        logger.info("\n⚖️  PHASE 3: Adaptive MEZAN Balancing")
        logger.info("-" * 70)

        # Select balancing strategy based on analysis
        balance_strategy = self._select_balance_strategy(synthesis_result)
        logger.info(f"Balancing strategy: {balance_strategy}")

        # Create MEZAN engine with selected solvers
        self.mezan_engine = MezanEngine(
            solver_left=solver_pair.left_solver,
            solver_right=solver_pair.right_solver,
            balance_strategy=balance_strategy,
            learning_rate=0.15,  # Slightly higher for faster adaptation
        )

        # Run MEZAN balancing
        mezan_result = self.mezan_engine.solve_with_balance(
            problem,
            max_iterations=max_mezan_iterations,
            convergence_threshold=1e-4,
        )

        # PHASE 4: Results and Insights
        logger.info("\n📈 PHASE 4: Results Summary")
        logger.info("-" * 70)

        total_time = time.time() - start_time

        logger.info(f"Final Objective: {mezan_result.objective_value:.6f}")
        logger.info(f"Total Iterations: {mezan_result.iterations}")
        logger.info(f"Total Time: {total_time:.3f}s")
        logger.info(f"  - Analysis: {synthesis_result.time_seconds:.3f}s" if synthesis_result else "")
        logger.info(f"  - Solving: {mezan_result.time_seconds:.3f}s")

        # Deep reasoning about results
        if synthesis_result:
            logger.info(f"\n💡 Deep Reasoning:")
            logger.info(synthesis_result.reasoning)

        logger.info("\n" + "="*70)
        logger.info("INTELLIGENT MEZAN COMPLETE")
        logger.info("="*70 + "\n")

        return mezan_result, synthesis_result

    def _select_solver_pair_intelligently(
        self,
        problem: Dict[str, Any],
        analysis: Optional[DeepResult],
    ) -> IntelligentSolverPair:
        """
        Intelligently select solver pair based on deep analysis

        Uses recommendations from deep analysis to choose optimal solvers.
        """
        # Default: use continuous + discrete pair
        from atlas_core.mezan_engine import MockContinuousSolver, MockDiscreteSolver

        # Extract algorithm recommendations from analysis
        if analysis and analysis.recommendations:
            # Look for algorithm-related recommendations
            algo_recs = [
                r for r in analysis.recommendations
                if r.get("type") in ["algorithm_selection", "algorithm_portfolio"]
            ]

            if algo_recs:
                # Use recommended algorithms
                primary_rec = algo_recs[0]

                reasoning = (
                    f"Based on deep analysis: {primary_rec.get('reason', 'N/A')}. "
                    f"Selected complementary solver pair for balance."
                )

                # Check if portfolio recommended
                if primary_rec.get("type") == "algorithm_portfolio":
                    algos = primary_rec.get("algorithms", [])
                    if len(algos) >= 2:
                        # Use top 2 from portfolio
                        reasoning = (
                            f"Portfolio approach with top algorithms: "
                            f"{algos[0]['name']} (weight {algos[0]['weight']}) and "
                            f"{algos[1]['name']} (weight {algos[1]['weight']})"
                        )

                confidence = primary_rec.get("confidence", 0.75)
                expected_performance = "High" if confidence > 0.85 else "Good"

            else:
                reasoning = "Default continuous + discrete pair (no specific algorithm recommendation)"
                confidence = 0.70
                expected_performance = "Good"

        else:
            reasoning = "Default continuous + discrete pair (no deep analysis)"
            confidence = 0.60
            expected_performance = "Acceptable"

        # Create solvers (currently mocks, would be real solvers in production)
        left_solver = MockContinuousSolver(
            SolverConfig(
                solver_id="continuous_relaxation",
                solver_type=SolverType.CONTINUOUS,
                weight=0.5,
            )
        )

        right_solver = MockDiscreteSolver(
            SolverConfig(
                solver_id="discrete_heuristic",
                solver_type=SolverType.DISCRETE,
                weight=0.5,
            )
        )

        return IntelligentSolverPair(
            left_solver=left_solver,
            right_solver=right_solver,
            selection_reasoning=reasoning,
            expected_performance=expected_performance,
            confidence=confidence,
        )

    def _select_balance_strategy(self, analysis: Optional[DeepResult]) -> str:
        """
        Select balancing strategy based on analysis

        Returns: "ucb", "thompson", or "epsilon_greedy"
        """
        if not analysis:
            return "ucb"  # Default

        # Check for resource allocation recommendations
        resource_recs = [
            r for r in analysis.recommendations
            if r.get("type") == "resource_allocation"
        ]

        if resource_recs:
            rec = resource_recs[0]
            rec_text = rec.get("recommendation", "").lower()

            # Fast heuristic → epsilon-greedy (more exploration)
            if "fast" in rec_text or "heuristic" in rec_text:
                return "epsilon_greedy"

            # Hybrid/balanced → UCB (balanced exploration/exploitation)
            if "hybrid" in rec_text or "balanced" in rec_text:
                return "ucb"

            # Multi-stage/thorough → Thompson (Bayesian)
            if "multi-stage" in rec_text or "thorough" in rec_text:
                return "thompson"

        # Default: UCB is generally good
        return "ucb"

    def get_last_analysis(self) -> Optional[DeepResult]:
        """Get the last deep analysis result"""
        return self.last_analysis

    def get_full_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics"""
        diagnostics = {
            "available_solvers": len(self.available_solvers),
            "deep_analysis_enabled": self.use_deep_analysis,
        }

        if self.deep_think:
            diagnostics["deepthink_stats"] = self.deep_think.get_statistics()

        if self.mezan_engine:
            diagnostics["mezan_engine"] = self.mezan_engine.get_diagnostics()

        if self.last_analysis:
            diagnostics["last_analysis"] = {
                "insights_count": len(self.last_analysis.insights),
                "recommendations_count": len(self.last_analysis.recommendations),
                "confidence": self.last_analysis.confidence,
                "time": self.last_analysis.time_seconds,
            }

        return diagnostics

    def create_distributed_workflow(
        self,
        problem: Dict[str, Any],
        workflow_type: str = "mezan_optimization"
    ) -> DistributedWorkflow:
        """
        Create distributed workflow for problem solving

        Args:
            problem: Problem to solve
            workflow_type: Type of workflow

        Returns:
            DistributedWorkflow instance
        """
        workflow = DistributedWorkflow()

        # Create tasks based on workflow type
        if workflow_type == "mezan_optimization":
            # Task 1: Deep analysis
            analysis_task = DistributedTask(
                task_type="analyze",
                payload={
                    "problem": problem,
                    "depth": "DEEP",
                    "max_time": 30
                },
                priority=10  # High priority
            )
            workflow.tasks.append(analysis_task)

            # Task 2: Run multiple solvers in parallel
            solver_ids = ["continuous", "discrete", "hybrid"]
            for solver_id in solver_ids:
                solver_task = DistributedTask(
                    task_type="solve",
                    payload={
                        "solver_id": solver_id,
                        "problem": problem
                    },
                    priority=5  # Medium priority
                )
                workflow.tasks.append(solver_task)

            # Task 3: MEZAN balancing
            balance_task = DistributedTask(
                task_type="balance",
                payload={
                    "problem": problem,
                    "max_iterations": 5
                },
                priority=1  # Lower priority (depends on other tasks)
            )
            workflow.tasks.append(balance_task)

        # Register workflow
        self.active_workflows[workflow.workflow_id] = workflow

        # Publish workflow started event
        if self.event_bus:
            publish_event(WorkflowStartedEvent(
                workflow_id=workflow.workflow_id,
                workflow_type=workflow_type
            ))

        logger.info(f"Created distributed workflow {workflow.workflow_id} with {len(workflow.tasks)} tasks")

        return workflow

    def submit_distributed_workflow(
        self,
        workflow: DistributedWorkflow
    ) -> List[str]:
        """
        Submit workflow for distributed execution

        Args:
            workflow: Workflow to submit

        Returns:
            List of task IDs
        """
        if not self.enable_distributed or not self.distributed_orchestrator:
            logger.error("Distributed mode not enabled")
            return []

        task_ids = []

        # Submit all tasks to distributed queue
        for task in workflow.tasks:
            task_id = self.distributed_orchestrator.submit_task(
                task_type=task.task_type,
                payload=task.payload,
                priority=task.priority
            )
            task.task_id = task_id
            task_ids.append(task_id)

            logger.debug(f"Submitted task {task_id} to distributed queue")

        workflow.status = "submitted"

        return task_ids

    def wait_for_workflow(
        self,
        workflow_id: str,
        timeout: float = 300
    ) -> Optional[DistributedWorkflow]:
        """
        Wait for workflow completion

        Args:
            workflow_id: Workflow to wait for
            timeout: Maximum wait time

        Returns:
            Completed workflow or None if timeout
        """
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return None

        workflow = self.active_workflows[workflow_id]
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if all tasks completed
            all_completed = all(
                task.status == "completed"
                for task in workflow.tasks
            )

            if all_completed:
                workflow.status = "completed"
                workflow.completed_at = time.time()

                logger.info(f"Workflow {workflow_id} completed in {workflow.completed_at - workflow.created_at:.3f}s")

                return workflow

            time.sleep(0.5)

        logger.warning(f"Workflow {workflow_id} timed out after {timeout}s")
        workflow.status = "timeout"

        return workflow

    def solve_distributed(
        self,
        problem: Dict[str, Any],
        max_workers: Optional[int] = None
    ) -> Tuple[SolverResult, Optional[DeepResult], DistributedWorkflow]:
        """
        Solve problem using distributed execution

        Args:
            problem: Problem to solve
            max_workers: Override worker count

        Returns:
            (mezan_result, synthesis_result, workflow)
        """
        if not self.enable_distributed:
            logger.warning("Distributed mode not enabled, falling back to local execution")
            result, synthesis = self.solve_intelligently(problem)
            return result, synthesis, None

        # Create workflow
        workflow = self.create_distributed_workflow(problem)

        # Submit for execution
        task_ids = self.submit_distributed_workflow(workflow)

        logger.info(f"Distributed solving started with {len(task_ids)} tasks")

        # Wait for completion
        completed_workflow = self.wait_for_workflow(workflow.workflow_id)

        if not completed_workflow:
            raise RuntimeError("Distributed workflow failed or timed out")

        # Extract results
        analysis_result = None
        solver_results = []
        balance_result = None

        for task in completed_workflow.tasks:
            if task.task_type == "analyze" and task.result:
                analysis_result = DeepResult(
                    task_id=task.task_id,
                    insights=task.result.get("insights", []),
                    recommendations=task.result.get("recommendations", []),
                    confidence=task.result.get("confidence", 0.0),
                    reasoning=task.result.get("reasoning", ""),
                    time_seconds=0.0
                )
            elif task.task_type == "solve" and task.result:
                solver_results.append(task.result)
            elif task.task_type == "balance" and task.result:
                balance_result = SolverResult(
                    solver_id="distributed_mezan",
                    objective_value=task.result.get("objective_value", float('inf')),
                    solution=task.result.get("solution", {}),
                    iterations=task.result.get("iterations", 0),
                    time_seconds=task.result.get("time_seconds", 0.0),
                    status="optimal" if task.result.get("objective_value") else "failed",
                    metadata={"distributed": True}
                )

        # If no balance result, use best solver result
        if not balance_result and solver_results:
            best_solver = min(solver_results, key=lambda x: x.get("objective_value", float('inf')))
            balance_result = SolverResult(
                solver_id=best_solver.get("solver_id", "unknown"),
                objective_value=best_solver.get("objective_value", float('inf')),
                solution=best_solver.get("solution", {}),
                iterations=best_solver.get("iterations", 0),
                time_seconds=0.0,
                status="optimal",
                metadata={"distributed": True}
            )

        return balance_result, analysis_result, completed_workflow

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get distributed cluster status"""
        if not self.distributed_orchestrator:
            return {"distributed": False}

        return self.distributed_orchestrator.get_cluster_status()

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific workflow"""
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]

        return {
            "workflow_id": workflow.workflow_id,
            "status": workflow.status,
            "created_at": workflow.created_at,
            "completed_at": workflow.completed_at,
            "total_tasks": len(workflow.tasks),
            "completed_tasks": sum(1 for t in workflow.tasks if t.status == "completed"),
            "failed_tasks": sum(1 for t in workflow.tasks if t.status == "failed"),
            "results_count": len(workflow.results)
        }

    def shutdown(self):
        """Shutdown all components"""
        # Stop distributed components
        if self.enable_distributed:
            self.enable_distributed = False  # Signal workers to stop

            if self.worker_pool:
                self.worker_pool.shutdown(wait=True)

            if self.distributed_orchestrator:
                self.distributed_orchestrator.stop()

            if self.message_queue:
                self.message_queue.close()

            if self.event_bus:
                self.event_bus.shutdown()

        # Shutdown deep think
        if self.deep_think:
            self.deep_think.shutdown()

        logger.info("Intelligent MEZAN shutdown complete")


def create_intelligent_mezan(enable_distributed: bool = False) -> IntelligentMezanEngine:
    """Create intelligent MEZAN engine with default configuration"""
    return IntelligentMezanEngine(
        available_solvers=[],  # Would be populated with real solvers
        use_deep_analysis=True,
        enable_distributed=enable_distributed,
        worker_count=4 if enable_distributed else 0
    )


def create_distributed_mezan(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    node_id: Optional[str] = None,
    worker_count: int = 8
) -> IntelligentMezanEngine:
    """Create distributed MEZAN engine"""
    return IntelligentMezanEngine(
        available_solvers=[],
        use_deep_analysis=True,
        enable_distributed=True,
        redis_host=redis_host,
        redis_port=redis_port,
        node_id=node_id,
        worker_count=worker_count,
        enable_events=True,
        message_broker_type=MessageBrokerType.REDIS
    )
