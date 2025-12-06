"""
MEZAN Orchestrator - Integration Layer

Connects MEZAN Engine with ORCHEX and Ultrathink Parallel Agents.

This module provides the high-level orchestration that combines:
1. MEZAN dual-solver balancing engine
2. 5-team ultrathink parallel agent system
3. ORCHEX research agents
4. Libria optimization solvers

Architecture:

```
┌─────────────────────────────────────────────────────────┐
│            MEZAN Orchestrator (this module)             │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Ultrathink 5-Team System               │   │
│  │  [Optimization] [Performance] [Integration]     │   │
│  │  [Infrastructure] [Learning]                    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │            MEZAN Dual-Solver Engine             │   │
│  │      [Solver L] → [ENGINE] ← [Solver R]        │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │                ORCHEX Agents                      │   │
│  │   (Research, Product, Validation, etc.)         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Author: MEZAN Research Team
Date: 2025-11-18
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import time
from enum import Enum

from atlas_core.mezan_engine import (
    MezanEngine,
    BaseSolver,
    SolverConfig,
    SolverResult,
    SolverType,
    MockContinuousSolver,
    MockDiscreteSolver,
)
from atlas_core.ultrathink_agents import (
    UltrathinkOrchestrator,
    AgentTask,
    AgentResult,
    TeamRole,
    TaskPriority,
    TaskStatus,
)
from atlas_core.engine import ATLASEngine
from atlas_core.blackboard import ATLASBlackboard

logger = logging.getLogger(__name__)


class OrchestratorMode(Enum):
    """Operating modes for MEZAN Orchestrator"""
    FULL = "full"                    # All systems active
    MEZAN_ONLY = "mezan_only"        # Only dual-solver balancing
    ULTRATHINK_ONLY = "ultrathink_only"  # Only parallel agents
    ATLAS_ONLY = "atlas_only"        # Only ORCHEX research agents


@dataclass
class OrchestratorConfig:
    """Configuration for MEZAN Orchestrator"""
    mode: OrchestratorMode = OrchestratorMode.FULL
    redis_url: str = "redis://localhost:6379/0"
    enable_mezan_engine: bool = True
    enable_ultrathink: bool = True
    enable_atlas: bool = True
    max_parallel_workers: int = 5
    balance_strategy: str = "ucb"  # For MEZAN engine
    ultrathink_timeout: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from orchestration"""
    task_id: str
    success: bool
    output: Any
    mezan_result: Optional[SolverResult] = None
    ultrathink_results: List[AgentResult] = field(default_factory=list)
    atlas_result: Optional[Dict] = None
    time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MezanOrchestrator:
    """
    MEZAN Orchestrator - Master Coordination Layer

    Integrates:
    - MEZAN dual-solver balancing engine
    - Ultrathink 5-team parallel agent system
    - ORCHEX research agent orchestration
    - Libria optimization solvers

    Provides unified interface for complex multi-agent optimization tasks.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize MEZAN Orchestrator

        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()

        # Initialize components based on mode
        self.mezan_engine = None
        self.ultrathink = None
        self.atlas_engine = None
        self.blackboard = None

        self._setup_components()

        logger.info(
            f"MEZAN Orchestrator initialized in {self.config.mode.value} mode"
        )

    def _setup_components(self):
        """Setup components based on configuration"""

        # Setup blackboard (shared state)
        if self.config.enable_atlas or self.config.enable_mezan_engine:
            try:
                self.blackboard = ATLASBlackboard(self.config.redis_url)
                logger.info("Blackboard (Redis) connected")
            except Exception as e:
                logger.warning(f"Blackboard unavailable: {e}")
                self.blackboard = None

        # Setup MEZAN dual-solver engine
        if self.config.enable_mezan_engine:
            # Create mock solvers for now
            # In production, these would be real Libria solvers
            solver_left = MockContinuousSolver(
                SolverConfig(
                    solver_id="continuous_flow",
                    solver_type=SolverType.CONTINUOUS,
                    weight=0.5,
                )
            )
            solver_right = MockDiscreteSolver(
                SolverConfig(
                    solver_id="discrete_combinatorial",
                    solver_type=SolverType.DISCRETE,
                    weight=0.5,
                )
            )

            self.mezan_engine = MezanEngine(
                solver_left=solver_left,
                solver_right=solver_right,
                balance_strategy=self.config.balance_strategy,
            )
            logger.info("MEZAN dual-solver engine initialized")

        # Setup Ultrathink parallel agents
        if self.config.enable_ultrathink:
            self.ultrathink = UltrathinkOrchestrator(
                max_workers=self.config.max_parallel_workers
            )
            logger.info("Ultrathink 5-team system initialized")

        # Setup ORCHEX research agents
        if self.config.enable_atlas:
            try:
                self.atlas_engine = ATLASEngine(
                    redis_url=self.config.redis_url,
                    libria_enabled=False,  # Will use MEZAN instead
                )
                logger.info("ORCHEX research agents initialized")
            except Exception as e:
                logger.warning(f"ORCHEX unavailable: {e}")
                self.atlas_engine = None

    def orchestrate(
        self,
        task: Dict[str, Any],
        use_mezan: bool = True,
        use_ultrathink: bool = True,
        use_atlas: bool = False,
    ) -> OrchestrationResult:
        """
        Orchestrate a complex task using available systems

        Args:
            task: Task specification
            use_mezan: Whether to use MEZAN dual-solver engine
            use_ultrathink: Whether to use Ultrathink parallel agents
            use_atlas: Whether to use ORCHEX research agents

        Returns:
            OrchestrationResult with outputs from all systems
        """
        start_time = time.time()
        task_id = task.get("task_id", f"task_{int(time.time() * 1000)}")

        logger.info(
            f"Orchestrating task {task_id} "
            f"(mezan={use_mezan}, ultrathink={use_ultrathink}, ORCHEX={use_atlas})"
        )

        result = OrchestrationResult(
            task_id=task_id,
            success=False,
            output=None,
        )

        # Phase 1: Ultrathink parallel analysis (fast)
        if use_ultrathink and self.ultrathink:
            ultrathink_tasks = self._create_ultrathink_tasks(task)
            result.ultrathink_results = self.ultrathink.execute_parallel(
                ultrathink_tasks,
                timeout=self.config.ultrathink_timeout,
            )
            logger.info(
                f"Ultrathink completed: {len(result.ultrathink_results)} results"
            )

        # Phase 2: MEZAN dual-solver optimization
        if use_mezan and self.mezan_engine:
            # Convert task to problem format
            problem = self._task_to_problem(task)

            # Solve with MEZAN balancing
            result.mezan_result = self.mezan_engine.solve_with_balance(
                problem,
                max_iterations=task.get("mezan_iterations", 5),
            )
            logger.info(
                f"MEZAN completed: objective={result.mezan_result.objective_value:.4f}"
            )

        # Phase 3: ORCHEX research agents (if needed)
        if use_atlas and self.atlas_engine:
            # Use ORCHEX for research/analysis tasks
            result.atlas_result = self._run_atlas_workflow(task)
            logger.info("ORCHEX workflow completed")

        # Synthesize results
        result.output = self._synthesize_results(result)
        result.success = True
        result.time_seconds = time.time() - start_time

        logger.info(
            f"Orchestration complete: {task_id} in {result.time_seconds:.2f}s"
        )

        return result

    def _create_ultrathink_tasks(self, task: Dict[str, Any]) -> List[AgentTask]:
        """Create tasks for Ultrathink parallel teams"""
        tasks = []

        # Optimization team task
        tasks.append(AgentTask(
            task_id=f"{task.get('task_id', 'task')}_opt",
            team=TeamRole.OPTIMIZATION,
            description="Optimize algorithm parameters",
            priority=TaskPriority.HIGH,
            metadata={"algorithm": task.get("algorithm", {})},
        ))

        # Performance team task
        tasks.append(AgentTask(
            task_id=f"{task.get('task_id', 'task')}_perf",
            team=TeamRole.PERFORMANCE,
            description="Evaluate performance metrics",
            priority=TaskPriority.HIGH,
            metadata={"model": task.get("model", {})},
        ))

        # Integration team task
        tasks.append(AgentTask(
            task_id=f"{task.get('task_id', 'task')}_int",
            team=TeamRole.INTEGRATION,
            description="Validate integration",
            priority=TaskPriority.MEDIUM,
            metadata={"solver": task.get("solver", {})},
        ))

        # Infrastructure team task
        tasks.append(AgentTask(
            task_id=f"{task.get('task_id', 'task')}_infra",
            team=TeamRole.INFRASTRUCTURE,
            description="Check infrastructure health",
            priority=TaskPriority.MEDIUM,
            metadata={"monitor": task.get("monitor", {})},
        ))

        # Learning team task
        tasks.append(AgentTask(
            task_id=f"{task.get('task_id', 'task')}_learn",
            team=TeamRole.LEARNING,
            description="Analyze patterns and insights",
            priority=TaskPriority.LOW,
            metadata={"research": task.get("research", {})},
        ))

        return tasks

    def _task_to_problem(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Convert high-level task to MEZAN problem format"""
        return {
            "type": task.get("type", "optimization"),
            "objective": task.get("objective", "minimize"),
            "constraints": task.get("constraints", []),
            "variables": task.get("variables", {}),
            "context": task,
        }

    def _run_atlas_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run ORCHEX research workflow"""
        if not self.atlas_engine:
            return {"status": "atlas_unavailable"}

        # This is a placeholder for ORCHEX integration
        # In full implementation, would run dialectical workflow
        return {
            "status": "completed",
            "workflow": "research",
            "quality": 0.85,
        }

    def _synthesize_results(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Synthesize results from all systems"""
        synthesis = {
            "task_id": result.task_id,
            "systems_used": [],
        }

        # Add MEZAN results
        if result.mezan_result:
            synthesis["systems_used"].append("mezan")
            synthesis["mezan"] = {
                "objective": result.mezan_result.objective_value,
                "iterations": result.mezan_result.iterations,
                "confidence": result.mezan_result.confidence,
                "metadata": result.mezan_result.metadata,
            }

        # Add Ultrathink results
        if result.ultrathink_results:
            synthesis["systems_used"].append("ultrathink")
            synthesis["ultrathink"] = {
                "teams": len(result.ultrathink_results),
                "successful": sum(
                    1 for r in result.ultrathink_results
                    if r.status == TaskStatus.SUCCESS
                ),
                "total_time": sum(r.time_seconds for r in result.ultrathink_results),
                "results": [
                    {
                        "team": r.team.value,
                        "status": r.status.value,
                        "metrics": r.metrics,
                        "message": r.message,
                    }
                    for r in result.ultrathink_results
                ],
            }

        # Add ORCHEX results
        if result.atlas_result:
            synthesis["systems_used"].append("ORCHEX")
            synthesis["ORCHEX"] = result.atlas_result

        return synthesis

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from all systems"""
        diagnostics = {
            "mode": self.config.mode.value,
            "components": {},
        }

        if self.mezan_engine:
            diagnostics["components"]["mezan"] = self.mezan_engine.get_diagnostics()

        if self.ultrathink:
            diagnostics["components"]["ultrathink"] = self.ultrathink.get_team_metrics()

        if self.atlas_engine:
            diagnostics["components"]["ORCHEX"] = {
                "agents_registered": len(self.atlas_engine.agents),
            }

        return diagnostics

    def shutdown(self):
        """Shutdown all systems gracefully"""
        logger.info("Shutting down MEZAN Orchestrator...")

        if self.ultrathink:
            self.ultrathink.shutdown()

        # Close other connections as needed

        logger.info("MEZAN Orchestrator shutdown complete")


def create_default_orchestrator() -> MezanOrchestrator:
    """Create orchestrator with default configuration"""
    return MezanOrchestrator(OrchestratorConfig(
        mode=OrchestratorMode.FULL,
        enable_mezan_engine=True,
        enable_ultrathink=True,
        enable_atlas=False,  # Disable ORCHEX by default (requires Redis)
    ))
