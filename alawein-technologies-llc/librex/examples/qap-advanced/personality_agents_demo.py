"""
ORCHEX with Personality Agents Demo

Shows how meta-learning and personality agents work in action!
"""

import asyncio
from ORCHEX import ATLASProtocol


async def main():
    """
    Run ORCHEX with personality agents enabled
    """
    print("\n" + "="*80)
    print("🎭 ORCHEX PERSONALITY AGENTS DEMO")
    print("="*80)
    print()
    print("This demo shows how ORCHEX uses personality-based agents")
    print("to make autonomous research more fun and effective!")
    print()
    print("Agents you'll meet:")
    print("  😠 Grumpy Refuter - Never satisfied, finds all flaws")
    print("  🤨 Skeptical Steve - Asks 200 annoying questions")
    print("  😰 Cautious Cathy - Always assesses risks")
    print("  😄 Optimistic Oliver - Generates creative hypotheses")
    print()
    print("Each agent LEARNS from past projects and gets better over time!")
    print("="*80)
    print()

    # Initialize ORCHEX with meta-learning enabled
    protocol = ATLASProtocol(
        output_base_dir="./demo_projects",
        enable_meta_learning=True,  # This is the magic!
    )

    # Run research on a simple topic
    project = await protocol.run_research(
        topic="Improving sort algorithm performance with machine learning",
        domain="computer_science",
        num_hypotheses=3,
        validation_threshold=65.0,
    )

    # Results
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  • Hypotheses generated: {len(project.hypothesis_candidates)}")
    print(f"  • Hypotheses validated: {len(project.validated_hypotheses)}")
    print(f"  • Failures recorded: {len(project.failures)}")
    print(f"  • Output: {project.output_dir}")
    print()

    if project.validated_hypotheses:
        print(f"✓ Selected hypothesis:")
        print(f"  {project.selected_hypothesis.claim[:100]}...")
        print()

    # Show meta-learning summary
    if protocol.enable_meta_learning:
        summary = protocol.meta_learning.get_learning_summary()
        print(f"📊 Meta-Learning Stats:")
        print(f"  • Total projects completed: {summary['total_projects']}")
        print(f"  • Agent performance tracked: {len(summary['agent_performance'])}")
        print()
        print("  Top agents:")
        for agent_id, stats in list(summary['agent_performance'].items())[:3]:
            print(f"    - {agent_id}: {stats.get('pulls', 0)} uses, "
                  f"{stats.get('avg_reward', 0.0):.2f} avg score")

    print()
    print("="*80)
    print("🧠 The agents learned from this project and will be even")
    print("   better on the NEXT research topic you give them!")
    print("="*80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
