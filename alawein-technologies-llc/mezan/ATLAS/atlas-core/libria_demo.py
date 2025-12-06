"""
Real Libria Solver Integration Demonstration

Showcases actual Libria solvers from libria-meta package working with MEZAN:
- QAPFlow for assignment problems
- AllocFlow for resource allocation
- WorkFlow for routing
- EvoFlow for multi-objective optimization
- GraphFlow for network design
- DualFlow for robust optimization
- MetaFlow for automatic solver selection

Author: MEZAN Research Team
Date: 2025-11-18
"""

import logging
import sys
import numpy as np
from atlas_core.libria_solvers import (
    create_libria_solver,
    get_available_solvers,
    get_solver_info,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_qapflow():
    """Demonstrate QAPFlow - Assignment problem solver"""
    print("=" * 70)
    print("DEMO 1: QAPFlow - Quadratic Assignment Problem Solver")
    print("=" * 70)
    print()
    print("Solving agent-task assignment with cost matrix...")
    print()

    # Create QAPFlow solver
    solver = create_libria_solver("qapflow")

    # Create assignment problem
    n = 8  # 8 agents, 8 tasks
    np.random.seed(42)
    cost_matrix = np.random.rand(n, n).tolist()

    problem = {
        "fit": cost_matrix,
        "constraints": {"type": "one_to_one"},
    }

    # Solve
    result = solver.solve(problem, parameters={"mode": "heuristic"})

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Objective Value: {result.objective_value:.4f}")
    print(f"✓ Assignment: {result.solution[:5]}... (first 5)")
    print(f"✓ Iterations: {result.iterations}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print(f"✓ Success: {result.success}")
    print()


def demo_allocflow():
    """Demonstrate AllocFlow - Resource allocation with Thompson Sampling"""
    print("=" * 70)
    print("DEMO 2: AllocFlow - Resource Allocation with Thompson Sampling")
    print("=" * 70)
    print()
    print("Allocating budget across multiple options using bandit algorithms...")
    print()

    # Create AllocFlow solver
    solver = create_libria_solver("allocflow")

    # Create allocation problem
    problem = {
        "options": [
            {"id": "option_a", "expected_return": 0.15},
            {"id": "option_b", "expected_return": 0.12},
            {"id": "option_c", "expected_return": 0.18},
            {"id": "option_d", "expected_return": 0.10},
        ],
        "budget": 1000.0,
    }

    # Solve
    result = solver.solve(problem, parameters={"algorithm": "thompson"})

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Total Value: {result.objective_value:.4f}")
    print(f"✓ Allocation: {result.solution}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_workflow():
    """Demonstrate WorkFlow - Workflow routing with confidence tracking"""
    print("=" * 70)
    print("DEMO 3: WorkFlow - Confidence-Aware Workflow Routing")
    print("=" * 70)
    print()
    print("Routing workflow through stages with safety constraints...")
    print()

    # Create WorkFlow solver
    solver = create_libria_solver("workflow")

    # Create workflow problem
    problem = {
        "stages": ["start", "preprocessing", "analysis", "optimization", "validation", "end"],
        "start": "start",
        "end": "end",
        "safety": ["validation"],  # Must include validation
    }

    # Solve
    result = solver.solve(problem)

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Confidence: {result.objective_value:.4f}")
    print(f"✓ Path: {' → '.join(result.solution)}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_evoflow():
    """Demonstrate EvoFlow - Evolutionary multi-objective optimization"""
    print("=" * 70)
    print("DEMO 4: EvoFlow - Evolutionary Multi-Objective Optimization")
    print("=" * 70)
    print()
    print("Solving multi-objective problem with genetic algorithm...")
    print()

    # Create EvoFlow solver
    solver = create_libria_solver("evoflow")

    # Create multi-objective problem
    problem = {
        "type": "multi_objective",
        "objectives": [
            {"name": "cost", "minimize": True},
            {"name": "quality", "minimize": False},
            {"name": "time", "minimize": True},
        ],
        "dimension": 20,
        "bounds": [0, 1],
    }

    # Solve
    result = solver.solve(problem, parameters={"population_size": 50, "generations": 100})

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Best Fitness: {result.objective_value:.4f}")
    print(f"✓ Generations: {result.iterations}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_graphflow():
    """Demonstrate GraphFlow - Network topology optimization"""
    print("=" * 70)
    print("DEMO 5: GraphFlow - Information-Theoretic Network Optimization")
    print("=" * 70)
    print()
    print("Optimizing agent communication network topology...")
    print()

    # Create GraphFlow solver
    solver = create_libria_solver("graphflow")

    # Create network problem
    problem = {
        "nodes": ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"],
        "edges": [
            ("agent_1", "agent_2"),
            ("agent_2", "agent_3"),
            ("agent_3", "agent_4"),
        ],
        "objectives": ["information_flow", "robustness"],
    }

    # Solve
    result = solver.solve(problem)

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Information Flow: {result.objective_value:.4f}")
    print(f"✓ Topology: {result.solution}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_dualflow():
    """Demonstrate DualFlow - Adversarial min-max optimization"""
    print("=" * 70)
    print("DEMO 6: DualFlow - Adversarial Min-Max Optimization")
    print("=" * 70)
    print()
    print("Solving robust optimization under worst-case scenarios...")
    print()

    # Create DualFlow solver
    solver = create_libria_solver("dualflow")

    # Create adversarial problem
    problem = {
        "type": "min_max",
        "player_actions": 5,
        "adversary_actions": 4,
        "payoff_matrix": np.random.rand(5, 4).tolist(),
    }

    # Solve
    result = solver.solve(problem)

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Game Value: {result.objective_value:.4f}")
    print(f"✓ Strategy: {result.solution}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_metaflow():
    """Demonstrate MetaFlow - Automatic solver selection"""
    print("=" * 70)
    print("DEMO 7: MetaFlow - Meta-Solver (Solver-of-Solvers)")
    print("=" * 70)
    print()
    print("Automatically selecting best solver for problem...")
    print()

    # Create MetaFlow solver
    solver = create_libria_solver("metaflow")

    # Create problem with features
    problem = {
        "type": "optimization",
        "features": {
            "size": 30,
            "objectives": 1,
            "constraints": 5,
            "nonlinearity": "high",
            "multimodal": True,
        },
        "budget": {"time": 1.0},
    }

    # Solve
    result = solver.solve(problem)

    # Display results
    print(f"✓ Solver: {result.solver_name}")
    print(f"✓ Selected Algorithm: {result.metadata.get('selected_solver', 'auto')}")
    print(f"✓ Best Value: {result.objective_value:.4f}")
    print(f"✓ Time: {result.time_elapsed:.4f}s")
    print()


def demo_solver_info():
    """Display information about all available solvers"""
    print("=" * 70)
    print("AVAILABLE LIBRIA SOLVERS")
    print("=" * 70)
    print()

    solvers = get_solver_info()
    for solver_id, info in solvers.items():
        print(f"📦 {info['name']} ({solver_id})")
        print(f"   Type: {info['type']}")
        print(f"   Description: {info['description']}")
        print(f"   Best For: {info['best_for']}")
        print()


def main():
    """Run all Libria solver demonstrations"""
    print()
    print("=" * 70)
    print(" REAL LIBRIA SOLVER INTEGRATION - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo showcases actual Libria solvers from libria-meta package")
    print("integrated with MEZAN for production-grade optimization.")
    print()

    # Show available solvers
    demo_solver_info()

    # Run individual solver demos
    try:
        demo_qapflow()
    except Exception as e:
        logger.error(f"QAPFlow demo failed: {e}")

    try:
        demo_allocflow()
    except Exception as e:
        logger.error(f"AllocFlow demo failed: {e}")

    try:
        demo_workflow()
    except Exception as e:
        logger.error(f"WorkFlow demo failed: {e}")

    try:
        demo_evoflow()
    except Exception as e:
        logger.error(f"EvoFlow demo failed: {e}")

    try:
        demo_graphflow()
    except Exception as e:
        logger.error(f"GraphFlow demo failed: {e}")

    try:
        demo_dualflow()
    except Exception as e:
        logger.error(f"DualFlow demo failed: {e}")

    try:
        demo_metaflow()
    except Exception as e:
        logger.error(f"MetaFlow demo failed: {e}")

    print("=" * 70)
    print(" ALL LIBRIA SOLVER DEMOS COMPLETE ✅")
    print("=" * 70)
    print()
    print("🚀 Production Applications:")
    print("  - Agent-task assignment (QAPFlow)")
    print("  - Budget allocation (AllocFlow)")
    print("  - Pipeline orchestration (WorkFlow)")
    print("  - Multi-objective optimization (EvoFlow)")
    print("  - Network topology design (GraphFlow)")
    print("  - Robust optimization (DualFlow)")
    print("  - Automatic algorithm selection (MetaFlow)")
    print()


if __name__ == "__main__":
    main()
