#!/usr/bin/env python3
"""
MEZAN Ultrathink Demo

Demonstrates the MEZAN (Meta-Equilibrium Zero-regret Assignment Network)
orchestrator with ultrathink parallel agent teams.

Usage:
    python mezan_demo.py

Author: MEZAN Research Team
Date: 2025-11-18
"""

import logging
import sys
from pprint import pprint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MEZAN components
try:
    from atlas_core.mezan_orchestrator import (
        MezanOrchestrator,
        OrchestratorConfig,
        OrchestratorMode,
        create_default_orchestrator,
    )
    from atlas_core.mezan_engine import (
        MezanEngine,
        SolverConfig,
        SolverType,
        MockContinuousSolver,
        MockDiscreteSolver,
    )
    from atlas_core.ultrathink_agents import (
        UltrathinkOrchestrator,
        AgentTask,
        TeamRole,
        TaskPriority,
    )
except ImportError as e:
    logger.error(f"Failed to import MEZAN modules: {e}")
    logger.error("Make sure you're running from the ORCHEX-core directory")
    sys.exit(1)


def demo_mezan_engine():
    """Demo 1: MEZAN dual-solver balancing engine"""
    print("\n" + "="*70)
    print("DEMO 1: MEZAN Dual-Solver Balancing Engine")
    print("="*70)
    print("\nArchitecture: [SOLVER_L] → [ENGINE] ← [SOLVER_R]")
    print("Balance Strategy: UCB (Upper Confidence Bound)\n")

    # Create two solvers
    solver_left = MockContinuousSolver(
        SolverConfig(
            solver_id="laplace_flow",
            solver_type=SolverType.CONTINUOUS,
            weight=0.5,
        )
    )

    solver_right = MockDiscreteSolver(
        SolverConfig(
            solver_id="hungarian_local_search",
            solver_type=SolverType.DISCRETE,
            weight=0.5,
        )
    )

    # Create MEZAN engine
    engine = MezanEngine(
        solver_left=solver_left,
        solver_right=solver_right,
        balance_strategy="ucb",
        learning_rate=0.1,
    )

    # Define problem
    problem = {
        "type": "qap",
        "size": 10,
        "objective": "minimize_cost",
    }

    # Solve with balancing
    print("Running MEZAN with 5 balancing iterations...\n")
    result = engine.solve_with_balance(
        problem,
        max_iterations=5,
        convergence_threshold=1e-4,
    )

    print("\nFinal Result:")
    print(f"  Objective: {result.objective_value:.6f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {result.time_seconds:.3f}s")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Final Weights: L={result.metadata['final_weight_left']:.3f}, "
          f"R={result.metadata['final_weight_right']:.3f}")

    # Show diagnostics
    print("\nEngine Diagnostics:")
    pprint(engine.get_diagnostics(), indent=2)


def demo_ultrathink_agents():
    """Demo 2: Ultrathink 5-team parallel agent system"""
    print("\n" + "="*70)
    print("DEMO 2: Ultrathink 5-Team Parallel Agent System")
    print("="*70)
    print("\nTeams: Optimization, Performance, Integration, Infrastructure, Learning")
    print("Execution: Parallel (5 threads)\n")

    # Create orchestrator
    orchestrator = UltrathinkOrchestrator(max_workers=5)

    # Create tasks for all 5 teams
    tasks = [
        AgentTask(
            task_id="opt_001",
            team=TeamRole.OPTIMIZATION,
            description="Optimize QAP algorithm",
            priority=TaskPriority.HIGH,
            metadata={"algorithm": {"type": "qap", "size": 20}},
        ),
        AgentTask(
            task_id="perf_001",
            team=TeamRole.PERFORMANCE,
            description="Benchmark performance",
            priority=TaskPriority.HIGH,
            metadata={"model": {"type": "transformer"}},
        ),
        AgentTask(
            task_id="int_001",
            team=TeamRole.INTEGRATION,
            description="Test Libria integration",
            priority=TaskPriority.MEDIUM,
            metadata={"solver": {"name": "Librex.QAP"}},
        ),
        AgentTask(
            task_id="infra_001",
            team=TeamRole.INFRASTRUCTURE,
            description="Run CI/CD tests",
            priority=TaskPriority.MEDIUM,
            metadata={"test": {"suite": "full"}},
        ),
        AgentTask(
            task_id="learn_001",
            team=TeamRole.LEARNING,
            description="Research pattern analysis",
            priority=TaskPriority.LOW,
            metadata={"research": {"topic": "optimization_patterns"}},
        ),
    ]

    print(f"Executing {len(tasks)} tasks in parallel...\n")

    # Execute in parallel
    results = orchestrator.execute_parallel(tasks, timeout=10.0)

    print(f"\nResults from {len(results)} teams:")
    print("-" * 70)
    for result in results:
        print(f"{result.status.value} {result.team.value:15s} | {result.message}")
        if result.metrics:
            print(f"   Metrics: {result.metrics}")

    # Show team metrics
    print("\nTeam Performance Metrics:")
    print("-" * 70)
    pprint(orchestrator.get_team_metrics(), indent=2)

    orchestrator.shutdown()


def demo_full_orchestration():
    """Demo 3: Full MEZAN orchestration"""
    print("\n" + "="*70)
    print("DEMO 3: Full MEZAN Orchestration")
    print("="*70)
    print("\nIntegrated System:")
    print("  - MEZAN dual-solver engine")
    print("  - Ultrathink 5-team parallel agents")
    print("  - Unified orchestration layer\n")

    # Create orchestrator
    orchestrator = create_default_orchestrator()

    # Define complex task
    task = {
        "task_id": "complex_001",
        "type": "optimization",
        "objective": "minimize",
        "algorithm": {"type": "qap", "size": 15},
        "model": {"type": "neural_network"},
        "solver": {"name": "Librex.Meta"},
        "mezan_iterations": 3,
    }

    print("Orchestrating complex task with all systems...\n")

    # Orchestrate
    result = orchestrator.orchestrate(
        task,
        use_mezan=True,
        use_ultrathink=True,
        use_atlas=False,  # ORCHEX requires Redis
    )

    print("\nOrchestration Result:")
    print(f"  Task ID: {result.task_id}")
    print(f"  Success: {result.success}")
    print(f"  Time: {result.time_seconds:.3f}s")

    print("\nSynthesized Output:")
    pprint(result.output, indent=2)

    # Show diagnostics
    print("\nSystem Diagnostics:")
    pprint(orchestrator.get_diagnostics(), indent=2)

    orchestrator.shutdown()


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" MEZAN ULTRATHINK PARALLEL MULTIAGENT SYSTEM DEMO")
    print("="*70)
    print("\nMEZAN = Meta-Equilibrium Zero-regret Assignment Network")
    print("Ultrathink = 5-Team Parallel Agent Architecture")
    print("\nThis demo showcases:")
    print("  1. Dual-solver balancing engine")
    print("  2. 5-team parallel agent system")
    print("  3. Full integrated orchestration")

    try:
        # Run demos
        demo_mezan_engine()
        demo_ultrathink_agents()
        demo_full_orchestration()

        print("\n" + "="*70)
        print(" DEMO COMPLETE ✅")
        print("="*70)
        print("\nKey Achievements:")
        print("  ✅ MEZAN dual-solver balancing working")
        print("  ✅ 5-team parallel agents executing")
        print("  ✅ Integrated orchestration functional")
        print("\nNext Steps:")
        print("  - Replace mock solvers with real Libria implementations")
        print("  - Integrate with production ORCHEX agents")
        print("  - Deploy to real optimization problems")
        print("  - Benchmark on QAPLIB instances")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
