"""
ORCHEX Engine Demo

Demonstrates basic usage of the ORCHEX Engine with multiple
research agents and dialectical workflows.
"""

import logging
import sys

# Add atlas_core to path
sys.path.insert(0, ".")

from atlas_core.engine import ATLASEngine
from atlas_core.agents import create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run ORCHEX Engine demo"""

    logger.info("=" * 70)
    logger.info("ORCHEX Engine Demo - Multi-Agent Research Orchestration")
    logger.info("=" * 70)

    # Initialize ORCHEX Engine (without Libria for demo)
    logger.info("\n1. Initializing ORCHEX Engine...")
    ORCHEX = ATLASEngine(libria_enabled=False)

    # Register research agents
    logger.info("\n2. Registering 10 research agents...")

    agent_types = [
        ("synthesis", "general_research"),
        ("literature_review", "machine_learning"),
        ("hypothesis_generation", "AI_safety"),
        ("critical_analysis", "methodology"),
        ("validation", "statistical_analysis"),
        ("data_analysis", "experimental_data"),
        ("methodology", "research_design"),
        ("ethics_review", "AI_ethics"),
        ("synthesis", "interdisciplinary"),
        ("hypothesis_generation", "theoretical_physics"),
    ]

    for i, (agent_type, specialization) in enumerate(agent_types):
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"agent_{i:02d}",
            specialization=specialization,
            skill_level=0.7 + (i % 3) * 0.1,  # Vary skill levels
            max_tasks=5,
        )
        ORCHEX.register_agent(agent)

    logger.info(f"Registered {len(ORCHEX.agents)} agents")

    # Show engine stats
    logger.info("\n3. ORCHEX Engine Statistics:")
    stats = ORCHEX.get_stats()
    logger.info(f"   Total agents: {stats['total_agents']}")
    logger.info(f"   Libria enabled: {stats['libria_enabled']}")
    logger.info("   Agents by type:")
    for agent_type, count in stats["agents_by_type"].items():
        logger.info(f"     - {agent_type}: {count}")

    # Test task assignment
    logger.info("\n4. Testing task assignment...")
    test_task = {
        "task_id": "demo_task_001",
        "task_type": "synthesis",
        "complexity": 0.7,
        "priority": 0.9,
    }

    assigned_agent_id = ORCHEX.assign_task(test_task)
    logger.info(f"   Task assigned to: {assigned_agent_id}")

    # Execute dialectical workflow
    logger.info("\n5. Executing Thesis-Antithesis-Synthesis workflow...")
    logger.info("   Topic: Neural Architecture Search for AI Safety")

    try:
        result = ORCHEX.execute_workflow(
            workflow_type="thesis_antithesis_synthesis",
            inputs={"topic": "neural_architecture_search_for_AI_safety"},
        )

        logger.info("\n   Workflow Results:")
        logger.info(f"   - Thesis quality: {result['thesis'].get('quality', 0):.2f}")
        logger.info(
            f"   - Antithesis quality: {result['antithesis'].get('quality', 0):.2f}"
        )
        logger.info(
            f"   - Synthesis quality: {result['synthesis'].get('quality', 0):.2f}"
        )

    except Exception as e:
        logger.error(f"   Workflow execution failed: {e}")

    # Show agent execution statistics
    logger.info("\n6. Agent Execution Statistics:")
    for agent_id, agent in ORCHEX.agents.items():
        agent_stats = agent.get_stats()
        if agent_stats["total_tasks"] > 0:
            logger.info(
                f"   {agent_id} ({agent.config.agent_type}): "
                f"{agent_stats['total_tasks']} tasks, "
                f"{agent_stats['success_rate']:.1%} success rate, "
                f"{agent_stats['avg_quality']:.2f} avg quality"
            )

    logger.info("\n" + "=" * 70)
    logger.info("Demo completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
