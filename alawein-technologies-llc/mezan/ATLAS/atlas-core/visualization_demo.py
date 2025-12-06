"""
Visualization System Demonstration

Shows all visualization capabilities:
- ASCII causal graph visualization
- Graphviz DOT export
- Mermaid diagram export
- Solver performance comparison charts
- Benchmark result summaries

Author: MEZAN Research Team
Date: 2025-11-18
"""

from atlas_core.causal_engine import create_causal_engine
from atlas_core.visualization import (
    CausalGraphVisualizer,
    PerformanceVisualizer,
    visualize_causal_graph,
    visualize_solver_comparison,
)


def demo_causal_graph_ascii():
    """Demonstrate ASCII causal graph visualization"""
    print()
    print("=" * 70)
    print(" DEMO 1: Causal Graph Visualization (ASCII)")
    print("=" * 70)
    print()

    # Create causal engine with built-in knowledge
    engine = create_causal_engine()

    # Visualize as ASCII
    ascii_viz = visualize_causal_graph(engine, format="ascii")
    print(ascii_viz)
    print()


def demo_causal_graph_dot():
    """Demonstrate DOT format export"""
    print("=" * 70)
    print(" DEMO 2: Causal Graph Export (Graphviz DOT)")
    print("=" * 70)
    print()
    print("Exporting causal graph to Graphviz DOT format...")
    print("(Can be rendered with: dot -Tpng output.dot -o graph.png)")
    print()

    engine = create_causal_engine()
    dot_viz = visualize_causal_graph(engine, format="dot", graph_name="MEZAN_Causal")

    # Save to file
    with open("causal_graph.dot", "w") as f:
        f.write(dot_viz)

    print("✓ Exported to: causal_graph.dot")
    print()
    print("Preview (first 30 lines):")
    print("-" * 70)
    lines = dot_viz.split('\n')
    for line in lines[:30]:
        print(line)
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    print()


def demo_causal_graph_mermaid():
    """Demonstrate Mermaid diagram export"""
    print("=" * 70)
    print(" DEMO 3: Causal Graph Export (Mermaid)")
    print("=" * 70)
    print()
    print("Exporting causal graph to Mermaid diagram format...")
    print("(Compatible with GitHub/GitLab markdown)")
    print()

    engine = create_causal_engine()
    mermaid_viz = visualize_causal_graph(engine, format="mermaid")

    # Save to file
    with open("causal_graph.mmd", "w") as f:
        f.write(mermaid_viz)

    print("✓ Exported to: causal_graph.mmd")
    print()
    print("Preview (first 25 lines):")
    print("-" * 70)
    lines = mermaid_viz.split('\n')
    for line in lines[:25]:
        print(line)
    if len(lines) > 25:
        print(f"... ({len(lines) - 25} more lines)")
    print()


def demo_solver_comparison():
    """Demonstrate solver performance comparison visualization"""
    print("=" * 70)
    print(" DEMO 4: Solver Performance Comparison")
    print("=" * 70)
    print()

    # Sample comparison data
    time_comparison = {
        "QAPFlow": 0.0123,
        "AllocFlow": 0.0089,
        "WorkFlow": 0.0034,
        "EvoFlow": 0.1456,
        "GraphFlow": 0.0567,
        "DualFlow": 0.0234,
        "MetaFlow": 0.0198,
    }

    viz = visualize_solver_comparison(
        time_comparison,
        metric="time",
        title="Solver Speed Comparison (Average Time)"
    )
    print(viz)
    print()

    # Objective value comparison
    obj_comparison = {
        "QAPFlow": 0.45,
        "AllocFlow": 0.67,
        "WorkFlow": 0.89,
        "EvoFlow": 0.82,
        "GraphFlow": 0.71,
        "DualFlow": 0.54,
        "MetaFlow": 0.76,
    }

    viz = visualize_solver_comparison(
        obj_comparison,
        metric="objective",
        title="Solver Quality Comparison (Average Objective)"
    )
    print(viz)
    print()


def demo_benchmark_summary():
    """Demonstrate benchmark statistics visualization"""
    print("=" * 70)
    print(" DEMO 5: Benchmark Statistics Summary")
    print("=" * 70)
    print()

    # Sample benchmark statistics
    statistics = {
        "total": 45,
        "successful": 42,
        "failed": 3,
        "time": {
            "mean": 0.0234,
            "median": 0.0198,
            "stdev": 0.0087,
            "min": 0.0045,
            "max": 0.1234,
            "total": 0.9823,
        },
        "objective": {
            "mean": 0.6543,
            "median": 0.6712,
            "stdev": 0.1234,
            "min": 0.2341,
            "max": 0.9123,
        },
        "iterations": {
            "mean": 127.5,
            "median": 112,
            "min": 23,
            "max": 456,
        },
    }

    visualizer = PerformanceVisualizer()
    viz = visualizer.render_benchmark_summary_ascii(statistics)
    print(viz)
    print()


def demo_custom_graph():
    """Demonstrate custom graph visualization"""
    print("=" * 70)
    print(" DEMO 6: Custom Graph Visualization")
    print("=" * 70)
    print()
    print("Creating custom optimization decision graph...")
    print()

    # Create custom nodes and edges
    from atlas_core.causal_engine import CausalNode, CausalEdge, CausalRelationType

    nodes = {
        "problem": CausalNode(
            id="problem",
            name="Optimization Problem",
            type="input",
        ),
        "features": CausalNode(
            id="features",
            name="Problem Features",
            type="analysis",
        ),
        "selector": CausalNode(
            id="selector",
            name="Algorithm Selector",
            type="decision",
        ),
        "solver": CausalNode(
            id="solver",
            name="Selected Solver",
            type="execution",
        ),
        "result": CausalNode(
            id="result",
            name="Optimization Result",
            type="output",
        ),
    }

    edges = [
        CausalEdge(
            source="problem",
            target="features",
            relation_type=CausalRelationType.CAUSES,
            strength=1.0,
        ),
        CausalEdge(
            source="features",
            target="selector",
            relation_type=CausalRelationType.CAUSES,
            strength=0.95,
        ),
        CausalEdge(
            source="selector",
            target="solver",
            relation_type=CausalRelationType.CAUSES,
            strength=0.98,
        ),
        CausalEdge(
            source="solver",
            target="result",
            relation_type=CausalRelationType.CAUSES,
            strength=0.92,
        ),
    ]

    visualizer = CausalGraphVisualizer()

    # ASCII visualization
    ascii_viz = visualizer.visualize(nodes, edges, format="ascii")
    print(ascii_viz)
    print()

    # Mermaid export
    mermaid_viz = visualizer.visualize(nodes, edges, format="mermaid")
    with open("custom_graph.mmd", "w") as f:
        f.write(mermaid_viz)
    print("✓ Custom graph exported to: custom_graph.mmd")
    print()


def main():
    """Run all visualization demonstrations"""
    print()
    print("=" * 70)
    print(" MEZAN VISUALIZATION SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print()
    print("Showcasing visualization capabilities for:")
    print("  ✓ Causal graph visualization (ASCII, DOT, Mermaid)")
    print("  ✓ Solver performance comparison")
    print("  ✓ Benchmark result summaries")
    print("  ✓ Custom graph creation")
    print()

    demo_causal_graph_ascii()
    demo_causal_graph_dot()
    demo_causal_graph_mermaid()
    demo_solver_comparison()
    demo_benchmark_summary()
    demo_custom_graph()

    print("=" * 70)
    print(" ALL VISUALIZATION DEMOS COMPLETE ✅")
    print("=" * 70)
    print()
    print("📊 Outputs Created:")
    print("  • causal_graph.dot (Graphviz DOT format)")
    print("  • causal_graph.mmd (Mermaid diagram)")
    print("  • custom_graph.mmd (Custom decision graph)")
    print()
    print("🎨 Visualization Features:")
    print("  ✓ ASCII art for terminal display")
    print("  ✓ Graphviz DOT for publication-quality diagrams")
    print("  ✓ Mermaid for GitHub/GitLab markdown")
    print("  ✓ Performance comparison charts")
    print("  ✓ Statistical summaries")
    print()


if __name__ == "__main__":
    main()
